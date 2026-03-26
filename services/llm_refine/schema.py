from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SegmentRecord:
    """
    標準化後的 segment 結構。

    segment_id:
        穩定主鍵。不要再依賴 list index。
    start / end:
        原始時間戳，保留給 SRT / alignment 用。
    text:
        原始 ASR 文字。
    speaker:
        先保留欄位，之後 diarization 可直接接。
    """
    segment_id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


TextMap = Dict[int, str]


def ensure_segment_id(segments: List[dict], key: str = "segment_id") -> List[dict]:
    """
    確保每個 segment 都有穩定的 segment_id。

    規則：
    - 若原本已有合法 segment_id，保留
    - 否則補一個新的
    - 不修改原 list 內容，回傳新 list

    注意：
    這裡仍然用 enumerate 補值，但只做「初始化一次」。
    之後整條 pipeline 都應該使用這個 segment_id，
    而不是再次用 enumerate 當主鍵。
    """
    output: List[dict] = []

    for idx, seg in enumerate(segments):
        new_seg = dict(seg)

        raw_id = new_seg.get(key, None)
        if raw_id is None:
            new_seg[key] = idx
        else:
            try:
                new_seg[key] = int(raw_id)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid {key}: {raw_id}")

        output.append(new_seg)

    _assert_unique_segment_ids(output, key=key)
    return output


def _assert_unique_segment_ids(segments: List[dict], key: str = "segment_id") -> None:
    seen = set()

    for seg in segments:
        seg_id = seg.get(key)
        if seg_id in seen:
            raise ValueError(f"Duplicate {key} found: {seg_id}")
        seen.add(seg_id)


def normalize_segments(
    segments: List[dict],
    id_key: str = "segment_id",
) -> List[SegmentRecord]:
    """
    把原始 dict segments 正規化成 SegmentRecord。
    """
    safe_segments = ensure_segment_id(segments, key=id_key)

    output: List[SegmentRecord] = []

    for seg in safe_segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = (seg.get("text") or "").strip()
        speaker = seg.get("speaker")

        if end < start:
            raise ValueError(f"Invalid segment time range: start={start}, end={end}")

        output.append(
            SegmentRecord(
                segment_id=int(seg[id_key]),
                start=start,
                end=end,
                text=text,
                speaker=speaker,
            )
        )

    return output


def build_text_map_from_records(records: List[SegmentRecord]) -> TextMap:
    """
    使用穩定 segment_id 建立 text_map。
    """
    text_map: TextMap = {}

    for record in records:
        text_map[record.segment_id] = record.text.strip()

    return text_map


def build_text_map_from_segments(
    segments: List[dict],
    id_key: str = "segment_id",
) -> TextMap:
    """
    給外部沿用的便利函式。

    與舊版差異：
    - 不再用 enumerate 當最終主鍵
    - 先正規化 segment_id
    """
    records = normalize_segments(segments, id_key=id_key)
    return build_text_map_from_records(records)


def records_to_segments(records: List[SegmentRecord]) -> List[dict]:
    """
    如需要，再轉回 dict 格式。
    """
    return [
        {
            "segment_id": r.segment_id,
            "start": r.start,
            "end": r.end,
            "text": r.text,
            "speaker": r.speaker,
        }
        for r in records
    ]