import unittest
try:
    from belief_mita.collect import game_spec
except ImportError:
    game_spec = None


class CollectionTests(unittest.TestCase):
    def test_two_hundred_games_form_fifty_fifty_color_pairs(self):
        self.assertIsNotNone(game_spec, 'original M0 collector not implemented')
        specs = [game_spec(i) for i in range(200)]
        self.assertEqual(sum(s['red'] for s in specs), 100)
        for first, second in zip(specs[::2], specs[1::2]):
            self.assertEqual(first['seed'], second['seed'])
            self.assertNotEqual(first['red'], second['red'])
            self.assertEqual(first['layout'], 'defaultCapture')


if __name__ == '__main__':
    unittest.main()
