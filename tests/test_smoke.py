"""Smoke test for RED_AGENT. Run from the RED_AGENT folder:

    python -m unittest tests.test_smoke -v

Checks: every module of the red package imports; one validation episode with the RL-only decision order
reproduces the recorded return; the LLM-only path runs with the offline stub LLM (no Ollama needed).
Needs torch, numpy and the CybORG simulator in third_party/cyborg.
"""
import importlib
import json
import pkgutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'checkpoints' / 'general_seed42_final.pt'


class Imports(unittest.TestCase):
    def test_every_module_imports(self):
        import red
        for module in pkgutil.walk_packages(red.__path__, 'red.'):
            importlib.import_module(module.name)


class RlOnlyEpisode(unittest.TestCase):
    def test_seed_30000_reproduces_recorded_return(self):
        from red.evaluation import pipeline_eval as pe
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            pe.main(['--checkpoint', f'general={CHECKPOINT}', '--seeds', '30000', '--output-dir', str(out)])
            summary = json.loads((out / 'evaluation_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['episodes'], 1)
        self.assertAlmostEqual(summary['mean_return'], 128.55, places=2)
        self.assertEqual(summary['decisions_by_source'].get('llm', 0), 0)


class LlmOnlyStub(unittest.TestCase):
    def test_llm_only_takes_every_decision_from_the_llm(self):
        from red.evaluation import pipeline_eval as pe
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            pe.main(['--checkpoint', f'general={CHECKPOINT}', '--seeds', '30000', '--llm', 'stub', '--llm-only',
                     '--output-dir', str(out)])
            summary = json.loads((out / 'evaluation_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(set(summary['decisions_by_source']), {'llm'})


if __name__ == '__main__':
    unittest.main()
