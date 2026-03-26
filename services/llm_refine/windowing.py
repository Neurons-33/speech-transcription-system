from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from services.llm_refine.schema import TextMap


@dataclass(frozen=True)
class WindowChunk:
    """
    一個可送進 LLM 的文字窗口。

    local_to_global:
        local id -> global segment_id 的映射
        例如 {0: 101, 1: 102, 2: 103}

    global_to_local:
        global segment_id -> local id 的反向映射
        例如 {101: 0, 102: 1, 103: 2}

    left_context_ids / main_ids / right_context_ids:
        這三個都使用 global segment_id。
        真正要套用結果時，只應使用 main_ids。
    """
    window_index: int
    local_text_map: Dict[int, str]
    local_to_global: Dict[int, int]
    global_to_local: Dict[int, int]

    left_context_ids: List[int]
    main_ids: List[int]
    right_context_ids: List[int]

    all_global_ids: List[int]


def estimate_text_cost(text: str) -> int:
    """
    粗略估算文字成本。

    目前先不用 tokenizer，因為你現在要的是穩定結構，不是精準計費。
    這裡故意保守一點：
    - 中文字、英文、數字、標點都直接算長度
    - 空字串回 0
    """
    if not text:
        return 0
    return len(text.strip())


def estimate_entry_cost(seg_id: int, text: str) -> int:
    """
    粗略估算單條 segment 進 prompt 的成本。

    除了 text 本身，也預留一些 JSON 結構成本：
    - id 欄位
    - key 名稱
    - 引號 / 逗號 / 大括號等
    """
    text_cost = estimate_text_cost(text)
    id_cost = len(str(seg_id))
    json_overhead = 24
    return text_cost + id_cost + json_overhead


def _slice_with_overlap(
    ids: Sequence[int],
    start: int,
    end: int,
    overlap_items: int,
) -> tuple[List[int], List[int], List[int]]:
    """
    給定主區 [start:end)，切出：
    - 左 context
    - 主區
    - 右 context
    """
    left_start = max(0, start - overlap_items)
    right_end = min(len(ids), end + overlap_items)

    left_context_ids = list(ids[left_start:start])
    main_ids = list(ids[start:end])
    right_context_ids = list(ids[end:right_end])

    return left_context_ids, main_ids, right_context_ids


def _build_window_chunk(
    window_index: int,
    text_map: TextMap,
    ordered_ids: List[int],
    left_context_ids: List[int],
    main_ids: List[int],
    right_context_ids: List[int],
) -> WindowChunk:
    """
    把 global ids 打包成可送 LLM 的 local window。
    """
    all_global_ids = left_context_ids + main_ids + right_context_ids

    local_to_global: Dict[int, int] = {}
    global_to_local: Dict[int, int] = {}
    local_text_map: Dict[int, str] = {}

    for local_id, global_id in enumerate(all_global_ids):
        local_to_global[local_id] = global_id
        global_to_local[global_id] = local_id
        local_text_map[local_id] = text_map[global_id]

    return WindowChunk(
        window_index=window_index,
        local_text_map=local_text_map,
        local_to_global=local_to_global,
        global_to_local=global_to_local,
        left_context_ids=left_context_ids,
        main_ids=main_ids,
        right_context_ids=right_context_ids,
        all_global_ids=all_global_ids,
    )


def chunk_text_map_with_overlap(
    text_map: TextMap,
    max_items: int = 16,
    max_chars: int = 1500,
    overlap_items: int = 2,
) -> List[WindowChunk]:
    """
    把全域 text_map 切成多個可送 LLM 的 windows，支援 overlap。

    規則：
    1. 先依 global id 排序，建立穩定順序
    2. 主區 main_ids 用 max_items + max_chars 控制大小
    3. 左右各加 overlap_items 當 context
    4. 後續只應套用 main_ids 的修正結果，不套用 context

    注意：
    - overlap_items 必須小於 max_items，否則窗口主體會過小
    - max_items 至少要 >= 1
    """
    if not text_map:
        return []

    if max_items < 1:
        raise ValueError("max_items must be >= 1")

    if overlap_items < 0:
        raise ValueError("overlap_items must be >= 0")

    if overlap_items >= max_items:
        raise ValueError("overlap_items must be smaller than max_items")

    ordered_ids = sorted(text_map.keys())

    windows: List[WindowChunk] = []
    n = len(ordered_ids)
    start = 0
    window_index = 0

    while start < n:
        current_ids: List[int] = []
        current_cost = 0
        end = start

        while end < n and len(current_ids) < max_items:
            seg_id = ordered_ids[end]
            seg_text = text_map[seg_id]
            entry_cost = estimate_entry_cost(seg_id, seg_text)

            if current_ids and (current_cost + entry_cost > max_chars):
                break

            current_ids.append(seg_id)
            current_cost += entry_cost
            end += 1

        # 保底：若單一 entry 就超長，也至少要塞進一窗
        if not current_ids:
            seg_id = ordered_ids[start]
            current_ids = [seg_id]
            end = start + 1

        left_context_ids, main_ids, right_context_ids = _slice_with_overlap(
            ids=ordered_ids,
            start=start,
            end=end,
            overlap_items=overlap_items,
        )

        window = _build_window_chunk(
            window_index=window_index,
            text_map=text_map,
            ordered_ids=ordered_ids,
            left_context_ids=left_context_ids,
            main_ids=main_ids,
            right_context_ids=right_context_ids,
        )
        windows.append(window)

        start = end
        window_index += 1

    return windows


def collect_main_local_ids(window: WindowChunk) -> List[int]:
    """
    取得這個 window 中，屬於 main_ids 的 local ids。
    後面 apply / refiner 階段只應採用這些 local ids 的結果。
    """
    output: List[int] = []
    for global_id in window.main_ids:
        output.append(window.global_to_local[global_id])
    return output


def map_local_refined_to_global_main_only(
    window: WindowChunk,
    refined_local_map: Dict[int, str],
) -> Dict[int, str]:
    """
    把 LLM 回傳的 local refined map 映射回 global，
    但只保留 main_ids 對應結果，避免 overlap 區重複覆蓋。
    """
    output: Dict[int, str] = {}

    for global_id in window.main_ids:
        local_id = window.global_to_local[global_id]
        if local_id in refined_local_map:
            output[global_id] = refined_local_map[local_id]

    return output