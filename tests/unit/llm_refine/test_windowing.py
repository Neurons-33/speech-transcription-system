# tests/unit/llm_refine/test_windowing.py

from services.llm_refine.schema import (
    ensure_segment_id,
    build_text_map_from_segments,
)

from services.llm_refine.windowing import (
    chunk_text_map_with_overlap,
    map_local_refined_to_global_main_only,
)

from tests.fixtures.llm_refine_segments import BASIC_SEGMENTS


def _build_text_map():
    segs = ensure_segment_id(BASIC_SEGMENTS)
    return build_text_map_from_segments(segs)


def test_chunk_basic():
    text_map = _build_text_map()

    windows = chunk_text_map_with_overlap(
        text_map=text_map,
        max_items=2,
        max_chars=200,
        overlap_items=1,
    )

    assert len(windows) >= 2
    assert all(len(w.main_ids) >= 1 for w in windows)


def test_overlap_exists():
    text_map = _build_text_map()

    windows = chunk_text_map_with_overlap(
        text_map=text_map,
        max_items=2,
        max_chars=200,
        overlap_items=1,
    )

    # 第二個 window 應該會有左側 overlap
    if len(windows) > 1:
        assert len(windows[1].left_context_ids) > 0


def test_mapping_main_only():
    text_map = _build_text_map()

    windows = chunk_text_map_with_overlap(
        text_map=text_map,
        max_items=2,
        max_chars=200,
        overlap_items=1,
    )

    w = windows[0]

    # 模擬 LLM 回傳（全部 local id 都有）
    refined_local = {
        local_id: f"FIXED_{text}"
        for local_id, text in w.local_text_map.items()
    }

    mapped = map_local_refined_to_global_main_only(w, refined_local)

    # 只應該包含 main_ids
    assert set(mapped.keys()) == set(w.main_ids)