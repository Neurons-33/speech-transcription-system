from __future__ import annotations

from typing import Dict, List

from services.llm_refine.client import LLMRefineClient
from services.llm_refine.schema import (
    ensure_segment_id,
    build_text_map_from_segments,
)
from services.llm_refine.windowing import (
    chunk_text_map_with_overlap,
    map_local_refined_to_global_main_only,
)


def refine_transcript(
    segments: List[dict],
    client: LLMRefineClient,
    max_items: int = 8,
    max_chars: int = 1200,
    overlap_items: int = 2,
    mode: str = "minimal",
) -> List[dict]:
    """
    主 pipeline：

    segments
    → normalize
    → windowing
    → LLM refine
    → merge
    → 回傳新 segments
    """

    # ----------------------------
    # 1. 確保 segment_id
    # ----------------------------
    segments = ensure_segment_id(segments)

    # ----------------------------
    # 2. 建 text_map
    # ----------------------------
    text_map = build_text_map_from_segments(segments)

    # ----------------------------
    # 3. 切 windows
    # ----------------------------
    windows = chunk_text_map_with_overlap(
        text_map=text_map,
        max_items=max_items,
        max_chars=max_chars,
        overlap_items=overlap_items,
    )

    print(f"[LLM] total segments = {len(segments)}")
    print(f"[LLM] total windows = {len(windows)}")

    # ----------------------------
    # 4. global result
    # ----------------------------
    global_refined: Dict[int, str] = {}

    # ----------------------------
    # 5. iterate windows
    # ----------------------------
    for window in windows:
        refined_local = client.refine_window(window, mode=mode)

        refined_global_main = map_local_refined_to_global_main_only(
            window,
            refined_local,
        )

        for gid, text in refined_global_main.items():
            # 保守策略：不覆蓋已存在（first write wins）
            if gid not in global_refined:
                global_refined[gid] = text

    # ----------------------------
    # 6. fallback missing（理論上不該發生，但保護）
    # ----------------------------
    for gid, original_text in text_map.items():
        if gid not in global_refined:
            global_refined[gid] = original_text

    # ----------------------------
    # 7. 回寫 segments
    # ----------------------------
    output_segments = []

    for seg in segments:
        sid = seg["segment_id"]

        output_segments.append({
            **seg,
            "text": global_refined[sid],
        })

    return output_segments