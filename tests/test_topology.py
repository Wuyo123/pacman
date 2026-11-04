import importlib.util
from pathlib import Path
import unittest

import numpy as np


class Walls:
    def __init__(self, width, height, cells):
        self.width, self.height = width, height
        self.columns = [[(x, y) not in cells for y in range(height)]
                        for x in range(width)]

    def __getitem__(self, x):
        return self.columns[x]


class TopologyTests(unittest.TestCase):
    def topology(self, width, height, cells):
        path = Path(__file__).parents[1] / "belief_mita" / "topology.py"
        self.assertTrue(path.is_file(), "MapTopology has not been implemented")
        spec = importlib.util.spec_from_file_location("belief_mita_topology", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.MapTopology(Walls(width, height, cells))

    def test_distances_follow_maze_and_disconnected_cells_stay_unreachable(self):
        cells = {(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1),
                 (6, 1)}
        t = self.topology(8, 5, cells)
        self.assertEqual(t.cells, sorted(cells))
        self.assertEqual(t.distances[t.index[1, 1], t.index[3, 1]], 6)
        self.assertGreaterEqual(t.distances[t.index[1, 1], t.index[6, 1]], 1000000)
        np.testing.assert_array_equal(t.distances, t.distances.T)
        np.testing.assert_array_equal(np.diag(t.distances), 0)

    def test_corridor_uses_outer_escape_gate_and_preserves_central_cycle(self):
        core = {(3, 2), (4, 2), (3, 3), (4, 3)}
        t = self.topology(8, 6, core | {(1, 2), (2, 2)})
        for cell, depth in [((1, 2), 2), ((2, 2), 1)]:
            self.assertEqual(t.gate[t.index[cell]], t.index[3, 2])
            self.assertEqual(t.escape_depth[t.index[cell]], depth)
        for cell in core:
            self.assertEqual(t.gate[t.index[cell]], -1)
            self.assertEqual(t.escape_depth[t.index[cell]], 0)
        self.assertEqual(t.articulation, {t.index[2, 2], t.index[3, 2]})
        self.assertEqual(set(t.bridges), {
            tuple(sorted((t.index[1, 2], t.index[2, 2]))),
            tuple(sorted((t.index[2, 2], t.index[3, 2])))})
        self.assertEqual(sorted(map(len, t.biconnected_components)), [2, 2, 4])

    def test_cycle_has_no_invented_dead_ends(self):
        cells = {(x, y) for x in range(1, 6) for y in range(1, 5)
                 if x in (1, 5) or y in (1, 4)}
        t = self.topology(8, 6, cells)
        self.assertEqual(t.articulation, set())
        self.assertEqual(t.bridges, [])
        self.assertEqual(t.biconnected_components, [set(range(len(cells)))])
        self.assertTrue(np.all(t.gate == -1))
        self.assertTrue(np.all(t.escape_depth == 0))

    def test_landmarks_are_deterministic_distinct_valid_and_include_capsule(self):
        cells = {(x, y) for x in range(1, 10) for y in range(1, 7)}
        t = self.topology(12, 8, cells)
        food = [(x, 1) for x in range(1, 10)] + [(x, 6) for x in range(1, 10)]
        first = t.landmark_indices(food=food, capsules=[(1, 4)], count=16)
        second = t.landmark_indices(food=reversed(food), capsules=[(1, 4)], count=16)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (16,))
        self.assertEqual(first.dtype, np.dtype("int64"))
        self.assertEqual(len(set(first)), 16)
        self.assertIn(t.index[1, 4], first)
        self.assertTrue(np.all((first >= 0) & (first < len(cells))))

    def test_small_disconnected_and_empty_maps(self):
        t = self.topology(8, 4, {(1, 1), (2, 1), (3, 1), (6, 2)})
        marks = t.landmark_indices(count=16)
        self.assertEqual(len(marks), 16)
        self.assertEqual(set(marks), set(range(4)))
        self.assertEqual(t.gate[t.index[2, 1]], -1)
        self.assertEqual(t.gate[t.index[6, 2]], -1)
        self.assertEqual(t.landmark_indices(count=0).size, 0)
        empty = self.topology(4, 4, set())
        self.assertEqual(empty.distances.shape, (0, 0))
        self.assertEqual(empty.landmark_indices().size, 0)

    def test_long_corridor_does_not_depend_on_python_recursion_limit(self):
        t = self.topology(1102, 3, {(x, 1) for x in range(1, 1101)})
        self.assertEqual(t.distances[0, -1], 1099)
        self.assertEqual(len(t.articulation), 1098)


if __name__ == "__main__":
    unittest.main()
