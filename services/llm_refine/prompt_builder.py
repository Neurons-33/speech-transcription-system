from __future__ import annotations
import json
from typing import Literal
from services.llm_refine.windowing import WindowChunk

RefineMode = Literal["minimal", "readable"]


def _get_mode_desc(mode: RefineMode) -> str:
    modes = {
        "minimal": "僅修正明顯 ASR 辨識錯誤，嚴禁潤飾語氣或大幅改寫。",
        "readable": "修正辨識錯誤並補上必要標點，可輕微整理語句，但須保留口語感。"
    }
    return modes.get(mode, modes["minimal"])


def _build_system_rules(mode: RefineMode) -> str:
    return f"""你是一位 ASR 逐字稿校對專家。
任務：修正語音辨識錯誤（同音異字、漏字、重複詞、明顯不自然詞語）。

【核心規則】
1. 輸出必須為純 JSON object，Key 與輸入完全一致，不可加 Markdown。
2. 禁止摘要、翻譯、合併、刪除或新增任何 ID。
3. 優先修正內容錯誤，而非只補標點。
4. 保留口語感，不改寫成正式文章。
5. 風格定位：{_get_mode_desc(mode)}"""


def _build_context_note(window: WindowChunk) -> str:
    main_locals = [window.global_to_local[g] for g in window.main_ids]
    if not main_locals:
        return """
[執行指南]
- 所有 ID 皆可視為修正目標，請依上下文校對。
""".strip()

    start, end = main_locals[0], main_locals[-1]

    return f"""
[執行指南]
- 主修正區：Local ID {start} 至 {end}。
- 其他 ID 主要作為上下文參考；若明顯存在 ASR 辨識錯誤，也可一併修正。
- 請利用前後文推斷主修正區中的正確詞彙。
""".strip()


def build_refine_prompt(
    window: WindowChunk,
    mode: RefineMode = "minimal",
) -> dict[str, str]:
    input_obj = {str(lid): txt for lid, txt in window.local_text_map.items()}

    user_prompt = f"""請修正下方 JSON 中的逐字稿錯誤。
請輸出完整 JSON，保留所有輸入 key，不可遺漏、不可新增。

【範例】
輸入：{{"0": "吃味道沒有用", "1": "我上好幾次傷完會比較好一點"}}
輸出：{{"0": "吃胃藥沒有用", "1": "我上好幾次，上完會比較好一點"}}

{_build_context_note(window)}

【待處理資料】
{json.dumps(input_obj, ensure_ascii=False, separators=(",", ":"))}"""

    return {
        "system": _build_system_rules(mode),
        "user": user_prompt,
    }