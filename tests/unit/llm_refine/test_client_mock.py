from services.llm_refine.client import LLMRefineClient
from services.llm_refine.schema import ensure_segment_id, build_text_map_from_segments
from services.llm_refine.windowing import chunk_text_map_with_overlap

from tests.fixtures.llm_refine_segments import BASIC_SEGMENTS


class DummyClient(LLMRefineClient):
    def _call_llm(self, prompt):
        return '{"0": "大家好", "1": "今天介紹系統"}'


def test_refine_window_success():
    segs = ensure_segment_id(BASIC_SEGMENTS)
    text_map = build_text_map_from_segments(segs)

    windows = chunk_text_map_with_overlap(
        text_map=text_map,
        max_items=2,
        max_chars=200,
        overlap_items=1,
    )

    client = DummyClient(api_key="fake")

    out = client.refine_window(windows[0])

    assert isinstance(out, dict)
    assert 0 in out