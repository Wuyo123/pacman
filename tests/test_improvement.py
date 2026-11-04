import unittest
try:
    from improve_m0 import select_candidate
except ImportError:
    select_candidate = None


class ImprovementTests(unittest.TestCase):
    def test_selection_requires_original_gate_and_paired_control_gain(self):
        self.assertIsNotNone(select_candidate, 'controlled experiment runner missing')
        def summary(rate, late=0):
            return {'score_rate':rate,'failures':0,'over_50ms':late,'completed':20}
        self.assertIsNone(select_candidate({'original':summary(.7),'diverse':summary(.7),'policy':summary(.725)}))
        self.assertEqual(select_candidate({'original':summary(.7),'diverse':summary(.8),'policy':summary(.8)}),'policy')
        self.assertEqual(select_candidate({'original':summary(.7),'diverse':summary(.8),'policy':summary(.85,1)}),'diverse')
        self.assertIsNone(select_candidate({'original':summary(.8),'diverse':summary(.8),'policy':summary(.8)}))
        self.assertIsNone(select_candidate({'original':summary(.5,1),'diverse':summary(.8),'policy':summary(.8)}))


if __name__ == '__main__':
    unittest.main()
