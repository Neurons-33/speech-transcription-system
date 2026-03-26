from services.llm_refine.orchestrator import refine_transcript
from services.llm_refine.client import LLMRefineClient

from tests.fixtures.llm_refine_segments import BASIC_SEGMENTS


class DummyClient(LLMRefineClient):
    def _call_llm(self, prompt):
        # 全部加 FIX_
        return '{"0": "FIX_大家好", "1": "FIX_今天介紹系統"}'


def test_refine_transcript_basic():
    client = DummyClient(api_key="fake")

    out = refine_transcript(
        BASIC_SEGMENTS,
        client=client,
        max_items=2,
        overlap_items=1,
    )

    assert len(out) == len(BASIC_SEGMENTS)
    assert "text" in out[0]


def test_refine_transcript_preserve_order():
    client = DummyClient(api_key="fake")

    out = refine_transcript(
        BASIC_SEGMENTS,
        client=client,
        max_items=2,
        overlap_items=1,
    )

    ids = [s["segment_id"] for s in out]

    assert ids == sorted(ids)