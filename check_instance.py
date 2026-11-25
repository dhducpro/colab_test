"""Independent CSV reader and primal solution check for the PDP-CR comparison."""

import csv
import math
from collections import deque
from numbers import Integral, Real
from pathlib import Path


def _finite(value, field):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid value for {field}: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"Expected a finite value for {field}: {value!r}")
    return number


def _csv_integer(value, field):
    number = _finite(value, field)
    if not number.is_integer():
        raise ValueError(f"Expected an integer for {field}: {value!r}")
    return int(number)


def read_instance(path, robots):
    """Read Thao's Euclidean CSV. IDs are 0 (depot), then tasks 1 through n.

    We copy the depot as sink n+1. Its inbound/outbound demand is represented
    by fleet size. Task processing times are used exactly as supplied in the
    CSV, while all transfer times use unrounded delivery-to-pickup distances.
    """
    if isinstance(robots, bool) or not isinstance(robots, Integral) or robots < 1:
        raise ValueError("robots must be a positive integer")
    robots = int(robots)
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"index", "x_pickup", "y_pickup", "x_dest", "y_dest",
                    "process_time", "robots_required"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV fields: {sorted(missing)}")
        rows = list(reader)
    if len(rows) < 2 or _csv_integer(rows[0]["index"], "depot index") != 0:
        raise ValueError("Expected a depot with index 0 followed by at least one task")
    rows.sort(key=lambda row: _csv_integer(row["index"], "index"))
    if [_csv_integer(row["index"], "index") for row in rows] != list(range(len(rows))):
        raise ValueError("Task indices must be consecutive and unique")

    requirements = [robots]
    processing = [0.0]
    for task_id, row in enumerate(rows[1:], 1):
        demand = _csv_integer(row["robots_required"], f"robots_required for task {task_id}")
        duration = _finite(row["process_time"], f"process_time for task {task_id}")
        if not 1 <= demand <= robots or duration < 0:
            raise ValueError(f"Invalid demand or duration for task {task_id}")
        requirements.append(demand)
        processing.append(duration)
    requirements.append(robots)
    processing.append(0.0)

    nodes = [*rows, rows[0]]
    pickups = [(_finite(row["x_pickup"], "x_pickup"), _finite(row["y_pickup"], "y_pickup"))
               for row in nodes]
    deliveries = [(_finite(row["x_dest"], "x_dest"), _finite(row["y_dest"], "y_dest"))
                  for row in nodes]
    travel = [[math.dist(origin, destination) for destination in pickups] for origin in deliveries]
    if any(not math.isfinite(value) for row in travel for value in row):
        raise ValueError("Transfer time overflow")
    return dict(name=path.stem, robots=robots, requirements=requirements,
                processing=processing, travel=travel)


def evaluate(instance, flow):
    """Check a dense source-to-sink integer flow and return its earliest makespan.

    Every task must receive and release its required number of robots. A
    positive arc enforces an arrival constraint regardless of its magnitude.
    Since a task can start only after all its robots arrive, the longest
    precedence path to the sink gives the last robot's return time.
    """
    requirements = instance["requirements"]
    size = len(requirements)
    sink = size - 1
    if len(flow) != size or any(len(row) != size for row in flow):
        raise ValueError(f"Expected a {size} by {size} flow matrix")
    children = [[] for _ in range(size)]
    indegree = [0] * size
    incoming = [0] * size
    outgoing = [0] * size
    for i, row in enumerate(flow):
        for j, value in enumerate(row):
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0 or value != int(value)):
                raise ValueError(f"Flow on {i}->{j} must be a nonnegative integer")
            if value == 0:
                continue
            if i == j or i == sink or j == 0:
                raise ValueError(f"Forbidden arc {i}->{j}")
            weight = int(value)
            outgoing[i] += weight
            incoming[j] += weight
            indegree[j] += 1
            children[i].append(j)
    if incoming != [0, *requirements[1:]] or outgoing != [*requirements[:-1], 0]:
        raise ValueError("Robot counts violate depot balance or a task requirement")

    ready = deque([0])
    starts = [0.0] * size
    visited = 0
    while ready:
        i = ready.popleft()
        visited += 1
        finish = starts[i] + instance["processing"][i]
        for j in children[i]:
            starts[j] = max(starts[j], finish + instance["travel"][i][j])
            indegree[j] -= 1
            if indegree[j] == 0:
                ready.append(j)
    if visited != size:
        raise ValueError("Flow contains a directed cycle")
    if not math.isfinite(starts[sink]):
        raise ValueError("Nonfinite makespan")
    return starts[sink]
