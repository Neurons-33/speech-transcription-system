from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict


@dataclass
class ValidationConfig:
    allow_markdown_fence: bool = True
    strip_code_block: bool = True
    ignore_extra_keys: bool = True
    fallback_missing_to_original: bool = True
    fallback_invalid_value_to_original: bool = True
    reject_empty_string: bool = False


def _extract_json_text(raw_text: str, config: ValidationConfig) -> str:
    text = raw_text.strip()

    if config.allow_markdown_fence or config.strip_code_block:
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response.")

    return match.group(0).strip()


def _parse_json_object(raw_text: str, config: ValidationConfig) -> dict:
    json_text = _extract_json_text(raw_text, config)
    data = json.loads(json_text)

    if not isinstance(data, dict):
        raise ValueError("LLM output must be a JSON object.")

    return data


def _safe_to_int_key(key) -> int | None:
    try:
        return int(key)
    except (TypeError, ValueError):
        return None


def parse_and_validate_refined_output(
    raw_text: str,
    original_local_map: Dict[int, str],
    config: ValidationConfig,
) -> Dict[int, str]:
    """
    目標：
    1. 解析 LLM JSON
    2. key 統一轉 int
    3. 只保留 original_local_map 裡存在的 id
    4. 缺失 id 用原文補回
    5. 非法 value 用原文補回
    """
    parsed = _parse_json_object(raw_text, config)

    cleaned_map: Dict[int, str] = {}

    # 先清洗 LLM 回傳
    for raw_key, raw_value in parsed.items():
        local_id = _safe_to_int_key(raw_key)
        if local_id is None:
            continue

        # 忽略不存在於原始 window 的 id
        if local_id not in original_local_map:
            if config.ignore_extra_keys:
                continue
            raise ValueError(f"Unexpected id from LLM output: {local_id}")

        # value 必須是字串
        if not isinstance(raw_value, str):
            if config.fallback_invalid_value_to_original:
                continue
            raise ValueError(f"Value for id {local_id} is not a string.")

        refined_text = raw_value.strip()

        if config.reject_empty_string and refined_text == "":
            if config.fallback_invalid_value_to_original:
                continue
            raise ValueError(f"Value for id {local_id} is empty.")

        cleaned_map[local_id] = refined_text

    # 最終合併：缺失 id 用原文補回
    final_map: Dict[int, str] = {}

    for local_id, original_text in original_local_map.items():
        if local_id in cleaned_map:
            final_map[local_id] = cleaned_map[local_id]
        else:
            if config.fallback_missing_to_original:
                final_map[local_id] = original_text
            else:
                raise ValueError(f"Missing id from LLM output: {local_id}")

    return final_map