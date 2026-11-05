import unittest


class InputRepairTests(unittest.TestCase):
    def gates(self, v0_rate, v1_rate, outcomes0, outcomes1, **overrides):
        def gate(score_rate, outcomes):
            return {'results': [{'seed': 9700000 + i // 2, 'red': i % 2 == 0,
                                 'outcome': value, 'score': value}
                                for i, value in enumerate(outcomes)],
                    'score_rate': score_rate, 'completed_games': 20, 'failures': 0,
                    'moves_over_50ms': 0, 'forward_p95_ms': 4.0}
        result = {'v0': gate(v0_rate, outcomes0), 'v1': gate(v1_rate, outcomes1)}
        for label, values in overrides.items():
            result[label].update(values)
        return result

    def test_one_draw_to_win_is_2_5_percentage_points(self):
        from belief_mita.input_repair import compare
        outcomes0 = [0] * 20
        outcomes1 = [1] + [0] * 19
        result = compare(self.gates(0.5, 0.525, outcomes0, outcomes1))
        self.assertAlmostEqual(result['v1_minus_v0_score_rate'], 0.025)
        self.assertAlmostEqual(result['paired_delta_check'], 0.025)
        self.assertEqual(result['decision'], 'NO_IMPROVEMENT_ESTABLISHED')

    def test_failed_v0_control_cannot_be_used_to_claim_improvement(self):
        from belief_mita.input_repair import compare
        outcomes0 = [-1] * 6 + [1] * 14
        outcomes1 = [-1] * 5 + [1] * 15
        gates = self.gates(0.7, 0.75, outcomes0, outcomes1, v0={'failures': 1})
        # Inject the same paired records represented by the score rates.
        gates['v0']['results'] = [{'seed': 9700000 + i // 2, 'red': i % 2 == 0,
                                   'outcome': value, 'score': value}
                                  for i, value in enumerate(outcomes0)]
        gates['v1']['results'] = [{'seed': 9700000 + i // 2, 'red': i % 2 == 0,
                                   'outcome': value, 'score': value}
                                  for i, value in enumerate(outcomes1)]
        result = compare(gates)
        self.assertEqual(result['decision'], 'INVALID_EXECUTION')

    def test_missing_games_are_inconclusive_and_not_a_smaller_denominator(self):
        from belief_mita.input_repair import compare
        gates = self.gates(0.5, 0.75, [0] * 20, [1] * 20)
        gates['v1']['completed_games'] = 19
        gates['v1']['results'] = gates['v1']['results'][:-1]
        result = compare(gates)
        self.assertEqual(result['decision'], 'INCONCLUSIVE_RESOURCE')
        self.assertIsNone(result['paired_delta_check'])


if __name__ == '__main__':
    unittest.main()
