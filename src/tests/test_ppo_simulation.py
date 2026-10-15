"""End-to-end check of the offline PPO simulation entry point."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.datasets.workload import read
from src.rl.simulate import run


class SimulationTests(unittest.TestCase):
    def test_pipeline_creates_disjoint_traces_and_reports(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'simulation'
            summary = run(output, total_steps=16, rollout_length=8,
                          requests_per_phase=12)
            traces = [read(output / f'{split}.jsonl')
                      for split in ('train', 'validation', 'held_out')]
            identifiers = [{row['request_id'] for row in trace} for trace in traces]
            self.assertEqual([len(trace) for trace in traces], [36, 36, 36])
            self.assertTrue(identifiers[0].isdisjoint(identifiers[1]))
            self.assertTrue(identifiers[0].isdisjoint(identifiers[2]))
            self.assertTrue(identifiers[1].isdisjoint(identifiers[2]))
            self.assertEqual(set(summary['held_out']),
                             {'ppo_lagrangian', 'always_reuse', 'always_recompute', 'drift_rule'})
            self.assertTrue(Path(summary['checkpoint']).is_file())
            self.assertEqual(json.loads((output / 'summary.json').read_text()), summary)
            self.assertTrue((output / 'training' / 'metrics.jsonl').is_file())


if __name__ == '__main__':
    unittest.main()
