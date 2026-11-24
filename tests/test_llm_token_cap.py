"""Reply caps. The Ollama default stays at 16 tokens (the index-only pipeline depends on it). The Caldera agent loop asks
for a larger cap on each of its three calls, because a plan, a stop reason and an evidence quote do not fit in 16."""
import unittest

from red.caldera import planner
from red.caldera.agent_loop import _ask_choice, _ask_plan, _objective_check
from red.llm.fake import FakeLLM
from red.llm.ollama import OllamaClient


class OllamaCapTests(unittest.TestCase):
    def _sent_cap(self, **kwargs):
        seen = {}

        def transport(url, body, timeout):
            seen['body'] = body
            return {'response': 'ACTION: 0', 'model': 'm'}

        OllamaClient(model='m', transport=transport).complete('prompt', **kwargs)
        return seen['body']['options']['num_predict']

    def test_default_stays_at_16_for_the_index_pipeline(self):
        self.assertEqual(self._sent_cap(), 16)

    def test_a_per_call_cap_is_sent(self):
        self.assertEqual(self._sent_cap(num_predict=256), 256)


class AgentCapTests(unittest.TestCase):
    def test_each_agent_call_asks_for_its_own_cap(self):
        llm = FakeLLM(responder=lambda prompt: 'RENCANA:\n1. Buka kalkulator\n2. Tutup kalkulator\nACTION: 0')
        _ask_plan(llm, 'prompt')
        _ask_choice(llm, 'prompt', 1)
        _objective_check(llm, 'tujuan', 'output')
        self.assertEqual(llm.num_predicts,
                         [planner.PLAN_MAX_TOKENS, planner.CHOICE_MAX_TOKENS, planner.CHECK_MAX_TOKENS])

    def test_the_caps_are_larger_than_the_old_default(self):
        self.assertGreater(planner.PLAN_MAX_TOKENS, 16)
        self.assertGreater(planner.CHOICE_MAX_TOKENS, 16)
        self.assertGreater(planner.CHECK_MAX_TOKENS, 16)


if __name__ == '__main__':
    unittest.main()
