from typing import Any, Dict, List


def format_srt_timestamp(seconds: float) -> str:
    """
    將秒數轉成 SRT 格式：
    HH:MM:SS,mmm
    """
    ms = int((seconds % 1) * 1000)
    total_seconds = int(seconds)
    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = total_seconds // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def get_segment_display_text(seg: Dict[str, Any]) -> str:
    """
    顯示優先順序：
    refined_text > normalized_text > raw_text > text
    """
    refined = (seg.get("refined_text") or "").strip()
    if refined:
        return refined

    normalized = (seg.get("normalized_text") or "").strip()
    if normalized:
        return normalized

    raw = (seg.get("raw_text") or "").strip()
    if raw:
        return raw

    text = (seg.get("text") or "").strip()
    return text


def get_segment_display_speaker(seg: Dict[str, Any]) -> str | None:
    """
    speaker 顯示優先順序：
    speaker_label > speaker_id
    """
    speaker_label = seg.get("speaker_label")
    if speaker_label:
        return speaker_label

    speaker_id = seg.get("speaker_id")
    if speaker_id:
        return speaker_id

    return None


def build_srt_from_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    從 chunk -> sub_segments 建立 SRT 內容
    """
    lines = []
    subtitle_index = 1

    for chunk in chunks:
        for seg in chunk.get("sub_segments", []):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            text = get_segment_display_text(seg)
            speaker = get_segment_display_speaker(seg)

            if not text:
                continue

            if speaker:
                text = f"[{speaker}] {text}"

            lines.append(str(subtitle_index))
            lines.append(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}")
            lines.append(text)
            lines.append("")

            subtitle_index += 1

    return "\n".join(lines).strip()