"""Longest paths on sparse flow supports, with reproducible tie policies."""
from collections import deque
import heapq


def longest_paths(out, indegree, processing, travel, *, fifo=False):
    """Return topological order, earliest starts and remaining times, or None for a cycle."""
    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    if fifo:
        queue = deque(ready)
        pop, push = queue.popleft, queue.append
    else:
        queue = ready
        pop = lambda: heapq.heappop(queue)
        push = lambda node: heapq.heappush(queue, node)

    order, starts = [], [0.0] * len(out)
    while queue:
        node = pop()
        order.append(node)
        finish = starts[node] + processing[node]
        for nxt in out[node]:
            starts[nxt] = max(starts[nxt], finish + travel[node][nxt])
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                push(nxt)
    if len(order) != len(out):
        return None

    tails = [0.0] * len(out)
    for node in reversed(order):
        for nxt in out[node]:
            tails[node] = max(tails[node], processing[node] + travel[node][nxt] + tails[nxt])
    return order, starts, tails


def reroute_schedule(instance, flow, successors, indegree, i, j, u, v, amount):
    """Evaluate the changed sparse support before copying the dense flow."""
    if i == v or u == j or v == 0 or j == 0 or i == instance.sink or u == instance.sink:
        return None
    out, degrees = successors.copy(), indegree.copy()
    for tail, removed, added in ((i, j, v), (u, v, j)):
        row = successors[tail].copy()
        if flow[tail, removed] == amount:
            row.remove(removed)
            degrees[removed] -= 1
        if not flow[tail, added]:
            row.append(added)
            row.sort()
            degrees[added] += 1
        out[tail] = row
    return longest_paths(out, degrees, instance.processing, instance.travel)
