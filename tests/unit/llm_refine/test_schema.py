# tests/unit/llm_refine/test_schema.py

import pytest

from services.llm_refine.schema import (
    ensure_segment_id,
    normalize_segments,
    build_text_map_from_segments,
)

from tests.fixtures.llm_refine_segments import (
    BASIC_SEGMENTS,
    SEGMENTS_WITH_ID,
    DUPLICATE_ID_SEGMENTS,
)


def test_ensure_segment_id_adds_id():
    segments = BASIC_SEGMENTS

    out = ensure_segment_id(segments)

    assert all("segment_id" in s for s in out)
    assert out[0]["segment_id"] == 0
    assert out[1]["segment_id"] == 1


def test_ensure_segment_id_keeps_existing():
    segments = SEGMENTS_WITH_ID

    out = ensure_segment_id(segments)

    assert out[0]["segment_id"] == 10
    assert out[1]["segment_id"] == 11


def test_ensure_segment_id_duplicate_should_fail():
    with pytest.raises(ValueError):
        ensure_segment_id(DUPLICATE_ID_SEGMENTS)


def test_normalize_segments_structure():
    segments = BASIC_SEGMENTS

    records = normalize_segments(segments)

    assert len(records) == 3
    assert records[0].segment_id == 0
    assert isinstance(records[0].start, float)
    assert isinstance(records[0].text, str)


def test_build_text_map_from_segments():
    segments = BASIC_SEGMENTS

    text_map = build_text_map_from_segments(segments)

    assert isinstance(text_map, dict)
    assert text_map[0] == "大家好"
    assert text_map[1].startswith("今天")