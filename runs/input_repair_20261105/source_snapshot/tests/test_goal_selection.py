import unittest
from goal_m0 import selection


class GoalSelectionTests(unittest.TestCase):
    def test_small_neural_margin_selects_simpler_control(self):
        def row(rate):
            return {'completed':20,'failures':0,'over_50ms':0,'score_rate':rate}
        self.assertEqual(selection({'original':row(.6),'guard':row(.9),'heuristic':row(.875)}),'heuristic')
        self.assertEqual(selection({'original':row(.6),'guard':row(.9),'heuristic':row(.8)}),'guard')
        self.assertIsNone(selection({'original':row(.6),'guard':row(.7),'heuristic':row(.7)}))


if __name__ == '__main__':
    unittest.main()
