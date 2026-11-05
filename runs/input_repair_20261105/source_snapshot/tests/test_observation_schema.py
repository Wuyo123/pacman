import unittest


class ObservationSchemaTests(unittest.TestCase):
    def test_checkpoint_schema_keeps_v0_compatible_and_rejects_dimension_mismatch(self):
        from belief_mita.schema import validate_checkpoint_schema

        self.assertEqual(validate_checkpoint_schema({'feature_dim': 18}), 0)
        self.assertEqual(validate_checkpoint_schema({'feature_dim': 18, 'observation_version': 0}), 0)
        self.assertEqual(validate_checkpoint_schema({'feature_dim': 37, 'observation_version': 1}), 1)
        for checkpoint in (
            {'feature_dim': 37},
            {'feature_dim': 18, 'observation_version': 1},
            {'feature_dim': 37, 'observation_version': 0},
            {'feature_dim': 99, 'observation_version': 2},
        ):
            with self.subTest(checkpoint=checkpoint), self.assertRaises(ValueError):
                validate_checkpoint_schema(checkpoint)


if __name__ == '__main__':
    unittest.main()
