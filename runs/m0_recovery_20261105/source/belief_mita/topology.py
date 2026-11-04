"""Static maze graph, escape pockets, and deterministic landmark selection."""

from collections import deque

import numpy as np


class MapTopology:
    UNREACHABLE = 1_000_000

    def __init__(self, walls):
        self.width, self.height = walls.width, walls.height
        self.cells = [(x, y) for x in range(self.width)
                      for y in range(self.height) if not walls[x][y]]
        self.index = {cell: i for i, cell in enumerate(self.cells)}
        self.neighbors = [sorted(self.index[p] for p in
                                ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                                if p in self.index) for x, y in self.cells]
        n = len(self.cells)
        self.distances = np.full((n, n), self.UNREACHABLE, dtype=np.int32)
        for start in range(n):
            row = self.distances[start]
            row[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in self.neighbors[u]:
                    if row[v] == self.UNREACHABLE:
                        row[v] = row[u] + 1
                        queue.append(v)
        self.articulation, self.bridges, self.biconnected_components = self._tarjan()
        mid = self.width // 2
        self.boundary = {i for i, (x, y) in enumerate(self.cells)
                         if x in (mid - 1, mid)
                         and (mid if x == mid - 1 else mid - 1, y) in self.index}
        self.gate = np.full(n, -1, dtype=np.int64)
        self.escape_depth = np.zeros(n, dtype=np.int32)
        self._escape_pockets()

    def _tarjan(self):
        """Iterative Tarjan traversal avoids recursion limits on long corridors."""
        n = len(self.cells)
        discovery = [-1] * n
        low = [-1] * n
        parent = [-1] * n
        children = [0] * n
        articulation, bridges, blocks = set(), [], []
        edges = []
        clock = 0
        for root in range(n):
            if discovery[root] >= 0:
                continue
            discovery[root] = low[root] = clock
            clock += 1
            if not self.neighbors[root]:
                blocks.append({root})
                continue
            stack = [(root, iter(self.neighbors[root]))]
            while stack:
                u, adjacent = stack[-1]
                v = next(adjacent, None)
                if v is not None:
                    if discovery[v] < 0:
                        parent[v] = u
                        children[u] += 1
                        edges.append((u, v))
                        discovery[v] = low[v] = clock
                        clock += 1
                        stack.append((v, iter(self.neighbors[v])))
                    elif v != parent[u] and discovery[v] < discovery[u]:
                        edges.append((u, v))
                        low[u] = min(low[u], discovery[v])
                    continue
                stack.pop()
                p = parent[u]
                if p < 0:
                    if children[u] > 1:
                        articulation.add(u)
                    continue
                low[p] = min(low[p], low[u])
                if low[u] > discovery[p]:
                    bridges.append(tuple(sorted((p, u))))
                if low[u] >= discovery[p]:
                    if parent[p] >= 0:
                        articulation.add(p)
                    block = set()
                    while edges:
                        edge = edges.pop()
                        block.update(edge)
                        if edge == (p, u):
                            break
                    blocks.append(block)
        return articulation, sorted(bridges), sorted(blocks, key=lambda b: tuple(sorted(b)))

    def _escape_pockets(self):
        unseen = set(range(len(self.cells)))
        while unseen:
            start = min(unseen)
            component = set(np.flatnonzero(self.distances[start] < self.UNREACHABLE))
            unseen.difference_update(component)
            anchors = component & self.boundary
            if not anchors:
                # A disconnected region without a crossing uses its graph center as core.
                ids = sorted(component)
                eccentricity = self.distances[np.ix_(ids, ids)].max(axis=1)
                anchors = {ids[k] for k in np.flatnonzero(eccentricity == eccentricity.min())}
            root_distance = self.distances[:, sorted(anchors)].min(axis=1)
            gates = sorted(component & self.articulation,
                           key=lambda g: (root_distance[g], g))
            for gate in gates:
                visited = {gate}
                for neighbor in self.neighbors[gate]:
                    if neighbor in visited:
                        continue
                    side = {neighbor}
                    visited.add(neighbor)
                    queue = deque([neighbor])
                    while queue:
                        u = queue.popleft()
                        for v in self.neighbors[u]:
                            if v not in visited:
                                visited.add(v)
                                side.add(v)
                                queue.append(v)
                    if side & anchors:
                        continue
                    for cell in side:
                        # The first gate is the outermost escape to the central core.
                        if self.gate[cell] < 0:
                            self.gate[cell] = gate
                            self.escape_depth[cell] = self.distances[cell, gate]

    def _spread(self, candidates, selected, count):
        candidates = set(candidates) - set(selected)
        result = []
        while candidates and len(result) < count:
            references = selected + result
            if references:
                next_cell = min(candidates, key=lambda i:
                                (-int(self.distances[i, references].min()), i))
            else:
                next_cell = min(candidates)
            result.append(next_cell)
            candidates.remove(next_cell)
        return result

    def _food_representatives(self, food, selected, count):
        """Farthest-first seeds followed by one deterministic maze-medoid update."""
        seeds = self._spread(food, selected, count)
        if not seeds:
            return []
        ids = sorted(set(food) - set(selected))
        assignments = self.distances[np.ix_(ids, seeds)].argmin(axis=1)
        representatives = []
        for k, seed in enumerate(seeds):
            cluster = [cell for cell, group in zip(ids, assignments) if group == k]
            if cluster:
                costs = self.distances[np.ix_(cluster, cluster)].sum(axis=1)
                representatives.append(cluster[int(costs.argmin())])
            else:
                representatives.append(seed)
        return representatives

    def landmark_indices(self, food=(), capsules=(), count=16):
        """Return distinct topology/capsule/food landmarks, then spread and pad.

        Topology gets up to a quarter of slots, leaving room for capsule and food
        information even in corridor-heavy maps. An empty map returns no indices.
        """
        if count < 0:
            raise ValueError("count must be nonnegative")
        n = len(self.cells)
        if not n or not count:
            return np.empty(0, dtype=np.int64)
        count_distinct = min(count, n)
        structural = self.boundary | self.articulation
        ranked = sorted(structural, key=lambda i:
                        (i not in self.articulation, i not in self.boundary,
                         len(self.neighbors[i]), i))
        selected = ranked[:min(count_distinct, max(1, count // 4))]
        capsule_ids = sorted({self.index[p] for p in capsules if p in self.index})
        for cell in capsule_ids:
            if cell not in selected and len(selected) < count_distinct:
                selected.append(cell)
        food_ids = sorted({self.index[p] for p in food if p in self.index})
        remaining = count_distinct - len(selected)
        selected.extend(self._food_representatives(food_ids, selected,
                                                  min(remaining, max(1, count // 2))))
        selected.extend(self._spread(range(n), selected, count_distinct - len(selected)))
        return np.resize(np.asarray(selected, dtype=np.int64), count)
