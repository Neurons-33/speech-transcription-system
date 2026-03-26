import pytest

from services.llm_refine.validator import (
    extract_json_object_text,
    parse_refined_json_object,
    validate_refined_keys,
    validate_refined_content_with_fallback,
    parse_and_validate_refined_output,
    ValidationConfig,
)


def test_extract_json_object_text_plain():
    raw = '{"0": "你好", "1": "世界"}'
    out = extract_json_object_text(raw)
    assert out.startswith("{")
    assert out.endswith("}")


def test_extract_json_object_text_fenced():
    raw = """```json
{
  "0": "你好",
  "1": "世界"
}
```"""
    out = extract_json_object_text(raw)
    assert '"0"' in out


def test_parse_refined_json_object_success():
    raw = '{"0": "你好", "1": "世界"}'
    parsed = parse_refined_json_object(raw)

    assert parsed == {0: "你好", 1: "世界"}


def test_parse_refined_json_object_invalid_key_should_fail():
    raw = '{"A": "你好"}'
    with pytest.raises(ValueError):
        parse_refined_json_object(raw)


def test_validate_refined_keys_success():
    refined = {0: "你好", 1: "世界"}
    validate_refined_keys(refined, expected_ids=[0, 1])


def test_validate_refined_keys_missing_should_fail():
    refined = {0: "你好"}
    with pytest.raises(ValueError):
        validate_refined_keys(refined, expected_ids=[0, 1])


def test_validate_refined_content_empty_should_fallback():
    original = {0: "大家好"}
    refined = {0: ""}

    safe = validate_refined_content_with_fallback(refined, original)

    assert safe[0] == "大家好"


def test_validate_refined_content_aggressive_growth_should_fallback():
    original = {0: "你好"}
    refined = {0: "你好你好你好你好你好你好你好你好你好你好你好"}

    safe = validate_refined_content_with_fallback(
        refined,
        original,
        config=ValidationConfig(max_growth_ratio=2.0, max_abs_growth=5),
    )

    assert safe[0] == "你好"


def test_parse_and_validate_refined_output_success():
    raw = '{"0": "大家好", "1": "今天介紹系統"}'
    original = {0: "大家好", 1: "今天我們介紹系統"}

    safe = parse_and_validate_refined_output(raw, original)

    assert safe[0] == "大家好"
    assert safe[1] == "今天介紹系統"


def test_parse_and_validate_refined_output_missing_key_should_fail():
    raw = '{"0": "大家好"}'
    original = {0: "大家好", 1: "今天我們介紹系統"}

    with pytest.raises(ValueError):
        parse_and_validate_refined_output(raw, original)