"""Proposal v2: alternating-cycle GA, random flow mutations, optional cut descent.

No reference optimum or reference flow is used by any search operator.
"""
from dataclasses import dataclass
import heapq
import random
from time import perf_counter

import numpy as np

from .core import checked_flow,evaluate,greedy
from .redistribute import alternating_redistribute,redistribute_cut
from .search import reroute_neighbors, reverse_neighbors
from .flow_operators import exchange_cycles, min_cost_transport


def transport_seed(supply,demand):
    """Feasible integer transport on a complete bipartite boundary."""
    a,b=list(map(int,supply)),list(map(int,demand))
    if min(a+b)<0 or sum(a)!=sum(b):
        raise ValueError("Incompatible boundary marginals")
    f=np.zeros((len(a),len(b)),dtype=int)
    j=0
    for i in range(len(a)):
        while a[i]:
            while j<len(b) and b[j]==0:
                j+=1
            value=min(a[i],b[j])
            f[i,j]+=value;a[i]-=value;b[j]-=value
    return f


def donor_transport(cost,supply,demand,donor):
    """Lexicographic min bottleneck, then max sum min(flow,donor).

    Primary: the validated alternating-path solver. Secondary: integral
    successive-shortest-path min-cost flow with reduced-cost Dijkstra.
    Each donor-preferred capacity segment costs -1; overflow costs 0.
    Zero marginals are compressed before solving.
    """
    cost=np.asarray(cost,dtype=float)
    donor=np.asarray(donor,dtype=int)
    supply=np.asarray(supply,dtype=int);demand=np.asarray(demand,dtype=int)
    if cost.shape!=(len(supply),len(demand)) or donor.shape!=cost.shape or np.any(donor<0):
        raise ValueError("Invalid donor transport dimensions")
    active_i=np.flatnonzero(supply);active_j=np.flatnonzero(demand)
    if not len(active_i):
        raise ValueError("A source-sink cut must carry positive fleet flow")
    c=cost[np.ix_(active_i,active_j)]
    preference=donor[np.ix_(active_i,active_j)]
    a=supply[active_i];b=demand[active_j]
    initial=transport_seed(a,b)
    primary=alternating_redistribute(c,initial)
    theta=primary['objective']
    nr,nc=c.shape;size=nr+nc+2;source=size-2;sink=size-1
    graph=[[] for _ in range(size)]

    def edge(u,v,capacity,weight):
        forward=[v,len(graph[v]),int(capacity),weight]
        backward=[u,len(graph[u]),0,-weight]
        graph[u].append(forward);graph[v].append(backward)
        return forward

    total=int(a.sum())
    refs=[]
    potential=[0]*size
    for i in range(nr):
        edge(source,i,a[i],0)
        for j in range(nc):
            if c[i,j]>theta:
                continue
            cap=min(int(a[i]),int(b[j]))
            preferred=min(cap,int(preference[i,j]))
            if preferred:
                e=edge(i,nr+j,preferred,-1)
                refs.append((i,j,preferred,e))
                potential[nr+j]=-1
            if cap>preferred:
                e=edge(i,nr+j,cap-preferred,0)
                refs.append((i,j,cap-preferred,e))
    for j in range(nc):
        edge(nr+j,sink,b[j],0)
    potential[sink]=min(potential[nr:nr+nc])
    value=0
    while value<total:
        distance=[float('inf')]*size;distance[source]=0
        parent=[None]*size;queue=[(0,source)]
        while queue:
            dist,u=heapq.heappop(queue)
            if dist!=distance[u]:
                continue
            for index,e in enumerate(graph[u]):
                v,_,cap,weight=e
                if cap:
                    reduced=weight+potential[u]-potential[v]
                    if reduced<0:
                        raise AssertionError("Invalid min-cost potential")
                    candidate=dist+reduced
                    if candidate<distance[v]:
                        distance[v]=candidate;parent[v]=(u,index)
                        heapq.heappush(queue,(candidate,v))
        if parent[sink] is None:
            raise AssertionError("Primary-optimal threshold became infeasible")
        for v in range(size):
            if distance[v]!=float('inf'):
                potential[v]+=distance[v]
        delta=total-value;v=sink
        while v!=source:
            u,index=parent[v];delta=min(delta,graph[u][index][2]);v=u
        v=sink
        while v!=source:
            u,index=parent[v];e=graph[u][index]
            e[2]-=delta;graph[v][e[1]][2]+=delta;v=u
        value+=delta
    small=np.zeros((nr,nc),dtype=int)
    for i,j,cap,e in refs:
        small[i,j]+=cap-e[2]
    out=np.zeros(cost.shape,dtype=int)
    out[np.ix_(active_i,active_j)]=small
    if not np.array_equal(out.sum(axis=1),supply) or not np.array_equal(out.sum(axis=0),demand):
        raise AssertionError("Min-cost transport lost boundary flow")
    return out,theta,int(np.minimum(out,donor).sum())


@dataclass
class Individual:
    flow: np.ndarray
    objective: float
    order: tuple
    starts: np.ndarray
    tails: np.ndarray
    signature: bytes


def individual(instance,flow):
    flow=checked_flow(instance,flow)
    s=evaluate(instance,flow)
    return Individual(flow,s.makespan,tuple(s.order),s.starts,s.tails,flow.tobytes())


def cut_crossover(instance, backbone, donor, rng, cut=None):
    """Proposal v2 section 3.3: retain BOTH internal subgraphs of backbone.

    Solve min bottleneck first, then maximize sum min(g_ij, donor_ij).
    Exchanging the arguments produces the second child (proposal section 4).
    """
    if cut is None:
        cut = rng.randrange(1, instance.size)
    if not 1 <= cut < instance.size:
        raise ValueError("Cut must keep source left and sink right")
    left, right = backbone.order[:cut], backbone.order[cut:]
    ix = np.ix_(left, right)
    original = backbone.flow[ix]
    supply, demand = original.sum(axis=1), original.sum(axis=0)
    cost = np.array([
        [backbone.starts[i] + instance.processing[i] + instance.travel[i][j]
         + backbone.tails[j] for j in right] for i in left
    ])
    crossing, theta, overlap = donor_transport(cost, supply, demand, donor.flow[ix])
    flow = backbone.flow.copy()
    flow[ix] = crossing
    child = individual(instance, flow)
    if abs(theta - child.objective) > 1e-7:
        raise AssertionError("Cut bottleneck differs from whole-DAG makespan")
    if child.objective > backbone.objective + 1e-7:
        raise AssertionError("Backbone cut crossover must not worsen its objective")
    return child, {"cut": cut, "donor_overlap": overlap}


# Compatibility for earlier operator tests; the GA default is cycle_crossover.
crossover = cut_crossover


def cycle_crossover(instance, first, second, rng, deadline=None):
    a, b, stats = exchange_cycles(first.flow, second.flow, rng, deadline)
    return individual(instance, a), individual(instance, b), stats


def random_cut_mutation(instance, parent, rng, beta=0.05, cut=None, deadline=None):
    """Random-cost feasible transport, bounded by (1+beta)*parent makespan."""
    if not np.isfinite(beta) or beta < 0:
        raise ValueError('Mutation beta must be finite and nonnegative')
    if deadline is not None and perf_counter() >= deadline:
        return parent
    if cut is None:
        cut = rng.randrange(1, instance.size)
    if not 1 <= cut < instance.size:
        raise ValueError('Invalid cut')
    left, right = parent.order[:cut], parent.order[cut:]
    ix = np.ix_(left, right)
    old = parent.flow[ix]
    delta = np.array([[parent.starts[i]+instance.processing[i]+instance.travel[i][j]
                       +parent.tails[j] for j in right] for i in left])
    threshold = (1+beta)*parent.objective
    costs = np.array([[rng.randrange(1_000_001) for _ in right] for _ in left])
    crossing = min_cost_transport(old.sum(1), old.sum(0), delta <= threshold+1e-9,
                                  costs, deadline)
    if crossing is None:
        return parent
    flow = parent.flow.copy()
    flow[ix] = crossing
    child = individual(instance, flow)
    if child.objective > threshold+1e-7:
        raise AssertionError('Random cut mutation exceeded degradation limit')
    return child


def cut_intensify(instance, parent, rng, max_nonimproving=5, max_solves=20, deadline=None):
    """Algorithm 3: reset consecutive failure count after each strict improvement."""
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
           for v in (max_nonimproving, max_solves)):
        raise ValueError('Cut descent limits must be positive integers')
    start, current, failed = perf_counter(), parent, 0
    stats = dict(solves=0, steps=0)
    while failed < max_nonimproving and stats['solves'] < max_solves:
        if deadline is not None and perf_counter() >= deadline:
            break
        candidate, _ = cut_crossover(instance, current, current, rng)
        stats['solves'] += 1
        if candidate.objective < current.objective-1e-9:
            current, failed = candidate, 0
            stats['steps'] += 1
        else:
            failed += 1
    stats.update(seconds=perf_counter()-start, gain=parent.objective-current.objective,
                 consecutive_failures=failed)
    return current, stats


def reroute_mutation(instance, parent, rng, max_attempts=64):
    """Random 2x2 flow exchange, NOT TSR's critical-arc neighborhood.

    New arcs must follow the parent's topological order. Delta is sampled
    uniformly from the positive feasible integers; u_ij = min(r_i, r_j).
    Return the unchanged parent if no valid pair is sampled in max_attempts.
    """
    arcs = [(int(i), int(j)) for i, j in zip(*np.nonzero(parent.flow))]
    position = {node: k for k, node in enumerate(parent.order)}
    if len(arcs) < 2:
        return parent
    for _ in range(max_attempts):
        (i, j), (u, v) = rng.sample(arcs, 2)
        if i == u or j == v or position[i] >= position[v] or position[u] >= position[j]:
            continue
        bound = min(
            int(parent.flow[i, j]), int(parent.flow[u, v]),
            min(instance.requirements[i], instance.requirements[v]) - int(parent.flow[i, v]),
            min(instance.requirements[u], instance.requirements[j]) - int(parent.flow[u, j]),
        )
        if bound <= 0:
            continue
        delta = rng.randint(1, bound)
        flow = parent.flow.copy()
        flow[i, j] -= delta
        flow[u, v] -= delta
        flow[i, v] += delta
        flow[u, j] += delta
        return individual(instance, flow)
    return parent


def mutate(instance, parent, rng, beta=0.05, strategy='mixed', deadline=None):
    """The two diversification families of proposal v2 section 3.4."""
    if strategy not in {'mixed', 'random_cut', 'reroute'}:
        raise ValueError('Unknown mutation strategy')
    if strategy == 'random_cut' or (strategy == 'mixed' and rng.random() < 0.5):
        return random_cut_mutation(instance, parent, rng, beta=beta, deadline=deadline), 'redistribute'
    return reroute_mutation(instance, parent, rng), "reroute"


def local_descent(instance, parent, rng, max_steps=3, max_candidates=8, deadline=None):
    """Strict descent using the three existing feasible TSR neighborhoods.

    Each round samples up to K valid reroute moves, K valid reverse moves and
    K cuts, then accepts the best strict improvement found. This is bounded
    local search, not a full neighborhood-optimality claim. No tabu or restart.
    The deadline is the GA's shared wall-clock deadline, not an extra budget.
    """
    if (not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 0
            or not isinstance(max_candidates, int) or isinstance(max_candidates, bool)
            or max_candidates < 1):
        raise ValueError("Local-search steps must be nonnegative and candidates positive integers")
    started = perf_counter()
    current = parent
    stats = {"steps": 0, "reroute": 0, "reverse": 0, "redistribute": 0}

    def expired():
        return deadline is not None and perf_counter() >= deadline

    for _ in range(max_steps):
        if expired():
            break
        best_objective, best_flow = current.objective, None
        for neighbors, kind in ((reroute_neighbors, "reroute"), (reverse_neighbors, "reverse")):
            if expired():
                break
            pool = neighbors(instance, current.flow, deadline=deadline,
                             max_candidates=max_candidates, rng=rng)
            stats[kind] += len(pool)
            for candidate in pool:
                if candidate.objective < best_objective - 1e-9:
                    best_objective, best_flow = candidate.objective, candidate.flow
        cuts = rng.sample(range(1, instance.size), min(max_candidates, instance.size-1))
        for cut in cuts:
            if expired():
                break
            flow, result = redistribute_cut(instance, current.flow, cut)
            stats["redistribute"] += 1
            if result["objective"] < best_objective - 1e-9:
                best_objective, best_flow = result["objective"], flow
        if best_flow is None:
            break
        previous = current.objective
        current = individual(instance, best_flow)
        if current.objective >= previous - 1e-9:
            raise AssertionError("Local descent must strictly improve its input")
        stats["steps"] += 1
    stats.update(seconds=perf_counter()-started,
                 gain=parent.objective-current.objective)
    return current, stats


def select_survivors(candidates, size, duplicate_limit=1):
    """Best P parents+offspring, with an explicitly specified duplicate rule.

    Try the requested per-genotype limit first. If too few distinct candidates
    exist, raise that limit minimally to fill P. Never add random immigrants or
    introduce new candidates; the best candidate always survives.
    Returns (survivors, effective_limit).
    """
    if size < 1 or duplicate_limit < 1 or len(candidates) < size:
        raise ValueError("Insufficient candidates or invalid survivor settings")
    ordered = sorted(candidates, key=lambda x: x.objective)
    limit = duplicate_limit
    while True:
        selected, counts = [], {}
        for candidate in ordered:
            count = counts.get(candidate.signature, 0)
            if count < limit:
                selected.append(candidate)
                counts[candidate.signature] = count + 1
                if len(selected) == size:
                    return selected, limit
        limit += 1


def genetic_search(instance, seed=0, population_size=16, generations=100,
                   crossover_rate=0.9, mutation_rate=0.4, tournament_size=3,
                   duplicate_limit=1, time_limit=None,
                   local_search_steps=0, local_search_candidates=8,
                   crossover_method='cycle', mutation_beta=0.05, mutation_strategy='mixed',
                   intensification_probability=0.0, intensification_failures=5,
                   intensification_max_solves=20, *, initial=None):
    """Proposal v2 Algorithm 2; default is pure alternating-cycle GA.

    P constructive/perturbed initial flows; frozen parents per generation;
    tournament selection; probability pc of one coupled pair of cycle children;
    probability pm of one of the two flow mutations per child; elitist (P+P)
    replacement. Optional CutIntensify uses the same wall-clock deadline.
    Legacy three-neighborhood descent is explicit and cannot be combined with
    CutIntensify. Uniform integer q in [0,alpha], uniform cuts, Lamarckian update.

    If provided, initial is the first flow in the population.
    Wall time includes initialization. Check the deadline before each parent
    pair and each child; a started child is completed. A final partial generation
    is retained and reported. The generation limit is an additional stop rule.
    """
    started = perf_counter()
    integers = (population_size, generations, tournament_size, duplicate_limit,
                local_search_steps, local_search_candidates, intensification_failures,
                intensification_max_solves)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in integers):
        raise ValueError("Population, generations, tournament and duplicate limit must be integers")
    if population_size < 2 or generations < 0 or tournament_size < 1 or duplicate_limit < 1:
        raise ValueError("Invalid GA configuration")
    if local_search_steps < 0 or local_search_candidates < 1:
        raise ValueError("Invalid local-search configuration")
    if intensification_failures < 1 or intensification_max_solves < 1:
        raise ValueError('Invalid cut intensification limits')
    if crossover_method not in {'cycle', 'cut'} or mutation_strategy not in {'mixed', 'random_cut', 'reroute'}:
        raise ValueError('Unknown crossover or mutation method')
    if not np.isfinite(mutation_beta) or mutation_beta < 0:
        raise ValueError('Mutation beta must be finite and nonnegative')
    if not 0 <= intensification_probability <= 1:
        raise ValueError('Intensification probability must be in [0,1]')
    if local_search_steps and intensification_probability:
        raise ValueError('Choose cut intensification OR legacy local descent')
    if not 0 <= crossover_rate <= 1 or not 0 <= mutation_rate <= 1:
        raise ValueError("Crossover/mutation probabilities must be in [0, 1]")
    if time_limit is not None and (not np.isfinite(time_limit) or time_limit < 0):
        raise ValueError("Time limit must be finite and nonnegative")
    deadline = None if time_limit is None else started + time_limit
    rng = random.Random(seed)
    start_flow = greedy(instance) if initial is None else initial
    best = individual(instance, start_flow)
    history = [{"seconds": perf_counter() - started, "generation": 0,
                "offspring": 0, "best": best.objective}]
    statistics = {"crossovers": 0, "redistribute": 0, "reroute": 0,
                  "mutations": 0, "clones_of_backbone": 0,
                  "initial_unique": 1, "max_duplicate_limit_used": duplicate_limit,
                  "local_search_calls": 0, "local_search_improved": 0,
                  "local_search_steps": 0, "local_search_gain": 0.0,
                  "local_search_seconds": 0.0, "local_search_reroute": 0,
                  "local_search_reverse": 0, "local_search_redistribute": 0}
    statistics.update(crossover_seconds=0.0, mutation_seconds=0.0,
                      parent_pairs=0, distinct_parent_pairs=0, novel_crossover_children=0,
                      swapped_parent_pairs=0, mutation_unchanged=0, mutation_improved=0,
                      mutation_equal=0, mutation_worsened=0,
                      offspring_novel_before_intensify=0, offspring_novel_after_intensify=0,
                      intensification_calls=0, intensification_improved=0,
                      intensification_solves=0, intensification_steps=0,
                      intensification_seconds=0.0, intensification_gain=0.0)
    population_history = []
    diagnostic_rng = random.Random(seed ^ 0xBA51)

    def log_population(population, generation):
        # At most 32 sampled unordered pairs; a separate RNG cannot steer search.
        pairs = [(i, j) for i in range(len(population)) for j in range(i)]
        sample = diagnostic_rng.sample(pairs, min(32, len(pairs)))
        distance = sum(float(np.abs(population[i].flow-population[j].flow).sum())/2
                       for i, j in sample)/len(sample) if sample else 0.0
        unique = len({x.signature for x in population})
        population_history.append(dict(generation=generation, seconds=perf_counter()-started,
                                       unique=unique, duplicate_fraction=1-unique/len(population),
                                       mean_pairwise_flow_distance=distance, distance_pairs=len(sample),
                                       best=min(x.objective for x in population)))

    def expired():
        return deadline is not None and perf_counter() >= deadline

    def update(candidate, generation, offspring):
        nonlocal best
        if candidate.objective < best.objective - 1e-9:
            best = candidate
            history.append({"seconds": perf_counter() - started,
                            "generation": generation, "offspring": offspring,
                            "best": best.objective})

    # Proposal line 3: constructive heuristics plus flow-preserving perturbations.
    # Bounded attempts avoid hanging when fewer than P distinct DAGs exist.
    initial = [best]
    signatures = {best.signature}
    for attempt in range(8 * population_size):
        if expired() or len(initial) >= population_size:
            break
        flow = greedy(instance, rng.getrandbits(32), randomize=True)
        candidate = individual(instance, flow)
        if attempt % 2:
            candidate, _ = mutate(instance, candidate, rng, beta=mutation_beta,
                                  strategy=mutation_strategy, deadline=deadline)
        update(candidate, 0, 0)
        if candidate.signature not in signatures:
            initial.append(candidate)
            signatures.add(candidate.signature)
    statistics["initial_unique"] = len(signatures)
    # Fill only with existing valid candidates; no order decoder/immigrant.
    population = [initial[k % len(initial)] for k in range(population_size)]
    population, effective_limit = select_survivors(population, population_size, duplicate_limit)
    statistics["max_duplicate_limit_used"] = effective_limit
    statistics["initial_best"] = best.objective
    statistics['initialization_seconds'] = perf_counter()-started
    log_population(population, 0)
    offspring_total = completed = partial = 0
    for generation in range(1, generations + 1):
        if expired():
            break
        offspring = []
        while len(offspring) < population_size and not expired():
            # Both parents come from the unchanged current generation.
            def tournament():
                return min(rng.sample(population, min(tournament_size, len(population))),
                           key=lambda x: x.objective)
            p, d = tournament(), tournament()
            statistics['parent_pairs'] += 1
            statistics['distinct_parent_pairs'] += int(p.signature != d.signature)
            parent_signatures = {p.signature, d.signature}
            if rng.random() < crossover_rate:
                crossing_started = perf_counter()
                if crossover_method == 'cycle':
                    first, second, cycle_stats = cycle_crossover(instance, p, d, rng, deadline)
                    for key, value in cycle_stats.items():
                        statistics['cycle_'+key] = statistics.get('cycle_'+key, 0)+value
                else:
                    first, _ = cut_crossover(instance, p, d, rng)
                    second, _ = cut_crossover(instance, d, p, rng)
                statistics['crossover_seconds'] += perf_counter()-crossing_started
                children = (first, second)
                statistics["crossovers"] += 2
                statistics["clones_of_backbone"] += int(first.signature == p.signature)
                statistics["clones_of_backbone"] += int(second.signature == d.signature)
                statistics['novel_crossover_children'] += sum(c.signature not in parent_signatures for c in children)
                statistics['swapped_parent_pairs'] += int(p.signature != d.signature
                    and first.signature == d.signature and second.signature == p.signature)
            else:
                children = (p, d)
            for child in children:
                if len(offspring) == population_size:
                    break
                # Finish the first child even if its crossover crossed the deadline.
                if offspring and expired():
                    break
                if rng.random() < mutation_rate:
                    before, mutation_started = child, perf_counter()
                    child, kind = mutate(instance, child, rng, beta=mutation_beta,
                                         strategy=mutation_strategy, deadline=deadline)
                    statistics['mutation_seconds'] += perf_counter()-mutation_started
                    statistics["mutations"] += 1
                    statistics[kind] += 1
                    statistics['mutation_unchanged'] += int(child.signature == before.signature)
                    outcome = ('improved' if child.objective < before.objective-1e-9 else
                               'worsened' if child.objective > before.objective+1e-9 else 'equal')
                    statistics['mutation_'+outcome] += 1
                statistics['offspring_novel_before_intensify'] += int(child.signature not in parent_signatures)
                if (intensification_probability and not expired()
                        and rng.random() < intensification_probability):
                    child, local = cut_intensify(instance, child, rng,
                        max_nonimproving=intensification_failures,
                        max_solves=intensification_max_solves, deadline=deadline)
                    statistics['intensification_calls'] += 1
                    statistics['intensification_improved'] += int(local['steps'] > 0)
                    for key in ('solves', 'steps', 'seconds', 'gain'):
                        statistics['intensification_'+key] += local[key]
                if local_search_steps and not expired():
                    child, local = local_descent(instance, child, rng,
                                                  max_steps=local_search_steps,
                                                  max_candidates=local_search_candidates,
                                                  deadline=deadline)
                    statistics["local_search_calls"] += 1
                    statistics["local_search_improved"] += int(local["steps"] > 0)
                    for key in ("steps", "gain", "seconds", "reroute", "reverse", "redistribute"):
                        statistics[f"local_search_{key}"] += local[key]
                statistics['offspring_novel_after_intensify'] += int(child.signature not in parent_signatures)
                offspring.append(child)
                offspring_total += 1
                update(child, generation, offspring_total)
        if offspring:
            population, effective_limit = select_survivors(
                population + offspring, population_size, duplicate_limit)
            statistics["max_duplicate_limit_used"] = max(
                statistics["max_duplicate_limit_used"], effective_limit)
            log_population(population, generation)
        if len(offspring) == population_size:
            completed += 1
        else:
            partial = len(offspring)
            break
    elapsed = perf_counter() - started
    variant = ('MA' if intensification_probability else 'GA')+'-'+crossover_method
    if local_search_steps:
        variant += '-legacyLS'
    return {"status": "feasible", "algorithm": "GA", "objective": best.objective,
            "proposal_version": 2, "variant": variant,
            "flow": best.flow.copy(), "history": history, "seconds": elapsed,
            "population_history": population_history,
            "generations_completed": completed, "partial_generation_offspring": partial,
            "offspring_completed": offspring_total, "statistics": statistics,
            "population_objectives": [candidate.objective for candidate in population],
            "config": {"seed": seed, "population_size": population_size,
                       "generations_limit": generations, "crossover_rate": crossover_rate,
                       "mutation_rate": mutation_rate, "tournament_size": tournament_size,
                       "duplicate_limit": duplicate_limit, "time_limit": time_limit,
                       "local_search_steps": local_search_steps,
                       "local_search_candidates": local_search_candidates,
                       "crossover_method": crossover_method, "mutation_beta": mutation_beta,
                       "mutation_strategy": mutation_strategy,
                       "intensification_probability": intensification_probability,
                       "intensification_failures": intensification_failures,
                       "intensification_max_solves": intensification_max_solves},
            "stop_reason": "time_limit" if expired() else "generation_limit"}


