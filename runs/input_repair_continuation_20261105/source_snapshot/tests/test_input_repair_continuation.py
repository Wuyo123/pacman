import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from belief_mita import input_repair_continuation as continuation


class InputRepairContinuationTests(unittest.TestCase):
    def test_replay_reads_the_newly_generated_v0_control_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(continuation, 'OUT', root / 'continuation'), \
                 patch.object(continuation, 'ORIGINAL', root / 'original'):
                (continuation.ORIGINAL / 'v0' / 'evaluation').mkdir(parents=True)
                paths = dict(continuation.replay_paths())
                self.assertEqual(paths['v0'], root / 'continuation' / 'v0' / 'evaluation')
                self.assertNotEqual(paths['v0'], continuation.ORIGINAL / 'v0' / 'evaluation')
                self.assertEqual(paths['v1'], root / 'continuation' / 'v1' / 'evaluation')


if __name__ == '__main__':
    unittest.main()
