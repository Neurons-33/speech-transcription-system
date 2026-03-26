from services.llm_refine.schema import ensure_segment_id, build_text_map_from_segments
from services.llm_refine.windowing import chunk_text_map_with_overlap
from services.llm_refine.prompt_builder import build_refine_prompt

from tests.fixtures.llm_refine_segments import BASIC_SEGMENTS


def test_build_refine_prompt_contains_system_and_user():
    segments = ensure_segment_id(BASIC_SEGMENTS)
    text_map = build_text_map_from_segments(segments)

    windows = chunk_text_map_with_overlap(
        text_map=text_map,
        max_items=2,
        max_chars=200,
        overlap_items=1,
    )

    prompt = build_refine_prompt(windows[0], mode="minimal")

    assert "system" in prompt
    assert "user" in prompt
    assert "JSON object" in prompt["user"]


def test_build_refine_prompt_contains_local_ids():
    segments = ensure_segment_id(BASIC_SEGMENTS)
    text_map = build_text_map_from_segments(segments)

    windows = chunk_text_map_with_overlap(
        text_map=text_map,
        max_items=2,
        max_chars=200,
        overlap_items=1,
    )

    prompt = build_refine_prompt(windows[0], mode="minimal")

    # 第一窗 local id 通常至少有 0
    assert '"0"' in prompt["user"]