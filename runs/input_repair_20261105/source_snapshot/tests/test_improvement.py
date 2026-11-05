import unittest
import json
try:
    from improve_m0 import select_candidate
except ImportError:
    select_candidate = None


class ImprovementTests(unittest.TestCase):
    def test_recovery_selection_distinguishes_data_from_extra_training(self):
        from recovery_m0 import selection
        def summary(rate):
            return {'completed':20,'failures':0,'over_50ms':0,'score_rate':rate}
        self.assertEqual(selection({'original':summary(.7),'old_extra':summary(.75),'recovery':summary(.85)}),'recovery')
        self.assertEqual(selection({'original':summary(.7),'old_extra':summary(.8),'recovery':summary(.75)}),'old_extra')
        self.assertEqual(selection({'original':summary(.7),'old_extra':summary(.75),'recovery':summary(.75)}),'old_extra')
    def test_paired_report_is_json_serializable(self):
        from improve_m0 import paired_comparison
        records = [{'mode':mode,'seed':seed,'red':red,'outcome':int(mode=='diverse')}
                   for seed in range(10) for red in (True,False) for mode in ('original','diverse')]
        self.assertIsInstance(json.dumps(paired_comparison(records,'diverse')),str)

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
