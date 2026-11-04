import unittest
import time
import gc
try:
    from belief_mita.evaluate import gate_result, evaluation_spec
except ImportError:
    gate_result = evaluation_spec = None


class EvaluationTests(unittest.TestCase):
    def test_original_gate_counts_all_twenty_games_and_half_draws(self):
        self.assertIsNotNone(gate_result, 'original M0 evaluation not implemented')
        outcomes = [1] * 10 + [0] * 10
        self.assertEqual(gate_result(outcomes, 7.9, failures=0, over_budget=0)['decision'], 'GO')
        self.assertEqual(gate_result(outcomes[:-1], 7.9, failures=0, over_budget=0)['decision'], 'INCOMPLETE')
        self.assertEqual(gate_result(outcomes, 16, failures=0, over_budget=0)['decision'], 'RETRY_D48')
        self.assertEqual(gate_result(outcomes, 16, failures=0, over_budget=1)['decision'], 'RETRY_D48')
        self.assertEqual(gate_result([0]*20, 2, failures=0, over_budget=0)['decision'], 'M0_NOT_PASSED')
        self.assertEqual(gate_result([1]*20, 2, failures=1, over_budget=0)['decision'], 'INVALID_EXECUTION')

    def test_holdout_seeds_are_disjoint_and_color_paired(self):
        self.assertIsNotNone(evaluation_spec, 'evaluation holdout not implemented')
        specs = [evaluation_spec(i) for i in range(20)]
        self.assertEqual(len(set(s['seed'] for s in specs)), 10)
        self.assertEqual(sum(s['red'] for s in specs), 10)
        self.assertGreater(min(s['seed'] for s in specs), 1000099)

    def test_benchmark_rejects_expired_budget_before_loading_any_data(self):
        from belief_mita.evaluate import benchmark
        with self.assertRaises(TimeoutError):
            benchmark('does-not-exist.pt', 'missing-data', 'unused.json', deadline=time.monotonic()-1)

    def test_gc_is_deferred_and_restored_even_when_decision_fails(self):
        from belief_mita import evaluate
        self.assertTrue(hasattr(evaluate, 'defer_cyclic_gc'), 'deferred GC context missing')
        before = gc.isenabled()
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with evaluate.defer_cyclic_gc():
                self.assertFalse(gc.isenabled())
                raise RuntimeError('test failure')
        self.assertEqual(gc.isenabled(), before)


if __name__ == '__main__':
    unittest.main()
