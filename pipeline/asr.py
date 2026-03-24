import os

from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor

from utils.audio_utils import extract_audio_segment
from utils.text_utils import (
    clean_transcribed_text,
    normalize_transcribed_text,
    should_skip_segment_text,
)
from models.whisper_model import load_whisper_model

# =========================
# Worker global
# =========================
_WORKER_MODEL = None


# =========================
# Worker init
# =========================
def _init_worker(model_size: str):
    """
    每個 process 啟動時各自載入一份 Whisper model。
    避免把 model 物件直接跨 process 傳遞。
    """
    global _WORKER_MODEL
    _WORKER_MODEL = load_whisper_model(model_size=model_size)

# =========================
# Segment optimization
# =========================
def split_long_segment(
    start: float,
    end: float,
    max_duration: float,
    min_duration: float,
) -> List[Dict[str, float]]:
    """
    只做非重疊硬切。
    """
    results = []
    current = start

    while (end - current) > max_duration:
        results.append(
            {
                "start": current,
                "end": current + max_duration,
            }
        )
        current += max_duration

    if (end - current) >= min_duration:
        results.append(
            {
                "start": current,
                "end": end,
            }
        )

    return results


def optimize_segments_for_multi_speaker(
    segments: List[Dict[str, float]],
    merge_gap: float = 0.4,
    min_duration: float = 0.8,
    max_duration: float = 5.0,
) -> List[Dict[str, float]]:
    """
    多人交談版：
    1. 清洗非法段
    2. 保守 merge
    3. 過長段最後再硬切
    """
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x["start"])

    normalized = []
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])

        if end <= start:
            continue

        duration = end - start
        if duration < min_duration:
            continue

        normalized.append({"start": start, "end": end})

    if not normalized:
        return []

    merged = []
    current = normalized[0].copy()

    for next_seg in normalized[1:]:
        gap = next_seg["start"] - current["end"]
        candidate_duration = next_seg["end"] - current["start"]

        if gap <= merge_gap and candidate_duration <= max_duration:
            current["end"] = next_seg["end"]
        else:
            merged.append(current)
            current = next_seg.copy()

    merged.append(current)

    optimized = []
    for seg in merged:
        start = seg["start"]
        end = seg["end"]
        duration = end - start

        if duration <= max_duration:
            optimized.append(seg)
        else:
            optimized.extend(
                split_long_segment(
                    start=start,
                    end=end,
                    max_duration=max_duration,
                    min_duration=min_duration,
                )
            )

    return optimized


# =========================
# Chunk task preparation
# =========================
def _build_chunk_task(
    audio_path: str,
    chunk_id: int,
    start: float,
    end: float,
    temp_dir: str,
    pad: float,
    keep_chunk_files: bool,
) -> Dict[str, Any]:
    chunk_duration = end - start
    pad_start = max(0.0, start - pad)
    pad_end = end + pad

    chunk_filename = f"chunk_{chunk_id:05d}.wav"
    chunk_path = os.path.join(temp_dir, chunk_filename)

    extract_audio_segment(
        input_path=audio_path,
        start=pad_start,
        end=pad_end,
        output_path=chunk_path,
    )

    return {
        "chunk_id": chunk_id,
        "chunk_start": round(start, 3),
        "chunk_end": round(end, 3),
        "chunk_duration": round(chunk_duration, 3),
        "pad_start": round(pad_start, 3),
        "pad_end": round(pad_end, 3),
        "chunk_path": chunk_path,
        "keep_chunk_files": keep_chunk_files,
    }


def _prepare_chunk_tasks(
    audio_path: str,
    optimized_segments: List[Dict[str, float]],
    temp_dir: str,
    min_duration: float,
    pad: float,
    keep_chunk_files: bool,
) -> List[Dict[str, Any]]:
    """
    在主程序先把所有 chunk wav 切好，之後再做 ASR。
    """
    os.makedirs(temp_dir, exist_ok=True)

    tasks = []

    for i, seg in enumerate(optimized_segments):
        chunk_start = float(seg["start"])
        chunk_end = float(seg["end"])
        chunk_duration = chunk_end - chunk_start

        if chunk_duration < min_duration:
            continue

        task = _build_chunk_task(
            audio_path=audio_path,
            chunk_id=i,
            start=chunk_start,
            end=chunk_end,
            temp_dir=temp_dir,
            pad=pad,
            keep_chunk_files=keep_chunk_files,
        )
        tasks.append(task)

    return tasks


# =========================
# Whisper result parsing
# =========================
def _build_sub_segment(
    *,
    segment_index: int,
    chunk_id: int,
    pad_start: float,
    whisper_start: float,
    whisper_end: float,
    raw_text: str,
) -> Dict[str, Any]:
    """
    建立單一 sub-segment。
    先把 schema 稍微升級，替未來 diarization / refinement 留位。
    """
    abs_start = max(0.0, pad_start + float(whisper_start))
    abs_end = max(abs_start, pad_start + float(whisper_end))

    normalized_text = normalize_transcribed_text(raw_text)

    return {
        "segment_id": f"chunk_{chunk_id:05d}_seg_{segment_index:03d}",
        "start": round(abs_start, 3),
        "end": round(abs_end, 3),
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "refined_text": None,
        "speaker_id": None,
        "speaker_label": None,
        "asr_confidence": None,
        "llm_edited": False,
    }


def _collect_sub_segments(
    *,
    segments_gen,
    chunk_id: int,
    chunk_duration: float,
    pad_start: float,
) -> List[Dict[str, Any]]:
    sub_segments = []

    for idx, s in enumerate(segments_gen):
        raw_text = clean_transcribed_text(s.text.strip())
        seg_duration = float(s.end) - float(s.start)

        if should_skip_segment_text(
            raw_text=raw_text,
            chunk_duration=chunk_duration,
            seg_duration=seg_duration,
        ):
            continue

        sub_segment = _build_sub_segment(
            segment_index=idx,
            chunk_id=chunk_id,
            pad_start=pad_start,
            whisper_start=float(s.start),
            whisper_end=float(s.end),
            raw_text=raw_text,
        )
        sub_segments.append(sub_segment)

    return sub_segments


def _build_chunk_result(
    task: Dict[str, Any],
    sub_segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "chunk_id": task["chunk_id"],
        "chunk_start": task["chunk_start"],
        "chunk_end": task["chunk_end"],
        "chunk_duration": task["chunk_duration"],
        "sub_segments": sub_segments,
        "audio_path": task["chunk_path"] if task["keep_chunk_files"] else None,
    }


def _cleanup_chunk_file(task: Dict[str, Any]) -> None:
    chunk_path = task["chunk_path"]
    keep_chunk_files = task["keep_chunk_files"]

    if not keep_chunk_files and os.path.exists(chunk_path):
        os.remove(chunk_path)


# =========================
# Core chunk transcription
# =========================
def _transcribe_chunk_with_model(
    *,
    model,
    task: Dict[str, Any],
    beam_size: int,
    language: str,
    initial_prompt: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    用指定 model 處理單一 chunk。
    sequential / parallel 共用。
    """
    segments_gen, _ = model.transcribe(
        task["chunk_path"],
        beam_size=beam_size,
        language=language,
        task="transcribe",
        word_timestamps=False,
        initial_prompt=initial_prompt,
        vad_filter=False,
        condition_on_previous_text=False,
        temperature=0.0,
        compression_ratio_threshold=2.0,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )

    sub_segments = _collect_sub_segments(
        segments_gen=segments_gen,
        chunk_id=task["chunk_id"],
        chunk_duration=task["chunk_duration"],
        pad_start=task["pad_start"],
    )

    if not sub_segments:
        _cleanup_chunk_file(task)
        return None

    result = _build_chunk_result(task, sub_segments)
    _cleanup_chunk_file(task)
    return result


def _transcribe_one_chunk(
    task: Dict[str, Any],
    beam_size: int,
    language: str,
    initial_prompt: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    給 ProcessPool worker 用。
    """
    global _WORKER_MODEL

    if _WORKER_MODEL is None:
        raise RuntimeError("Worker model is not initialized.")

    return _transcribe_chunk_with_model(
        model=_WORKER_MODEL,
        task=task,
        beam_size=beam_size,
        language=language,
        initial_prompt=initial_prompt,
    )


# =========================
# Shared preparation
# =========================
def _prepare_transcription_tasks(
    audio_path: str,
    segments: List[Dict[str, float]],
    temp_dir: str,
    merge_gap: float,
    min_duration: float,
    max_duration: float,
    pad: float,
    keep_chunk_files: bool,
    debug: bool,
) -> List[Dict[str, Any]]:
    os.makedirs(temp_dir, exist_ok=True)

    optimized_segments = optimize_segments_for_multi_speaker(
        segments=segments,
        merge_gap=merge_gap,
        min_duration=min_duration,
        max_duration=max_duration,
    )

    if debug:
        print("=" * 60)
        print("[DEBUG] Optimized Segments")
        for i, seg in enumerate(optimized_segments):
            duration = seg["end"] - seg["start"]
            print(f"[{i:03d}] {seg['start']:.2f} -> {seg['end']:.2f} ({duration:.2f}s)")
        print("=" * 60)

    return _prepare_chunk_tasks(
        audio_path=audio_path,
        optimized_segments=optimized_segments,
        temp_dir=temp_dir,
        min_duration=min_duration,
        pad=pad,
        keep_chunk_files=keep_chunk_files,
    )


# =========================
# Sequential version
# =========================
def transcribe_segments_multi_speaker(
    audio_path: str,
    segments: List[Dict[str, float]],
    model,
    temp_dir: str = "outputs/chunks",
    merge_gap: float = 0.2,
    min_duration: float = 0.4,
    max_duration: float = 5.0,
    pad: float = 0.2,
    beam_size: int = 5,
    language: str = "zh",
    initial_prompt: str = "以下是繁體中文的多人談話內容。",
    keep_chunk_files: bool = True,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    單線版。
    """
    tasks = _prepare_transcription_tasks(
        audio_path=audio_path,
        segments=segments,
        temp_dir=temp_dir,
        merge_gap=merge_gap,
        min_duration=min_duration,
        max_duration=max_duration,
        pad=pad,
        keep_chunk_files=keep_chunk_files,
        debug=debug,
    )

    if not tasks:
        return []

    results = []
    for task in tasks:
        result = _transcribe_chunk_with_model(
            model=model,
            task=task,
            beam_size=beam_size,
            language=language,
            initial_prompt=initial_prompt,
        )
        if result is not None:
            results.append(result)

            if debug:
                print(f"[TRANSCRIBED] chunk_{task['chunk_id']:05d}")

    results.sort(key=lambda x: x["chunk_id"])
    return results


# =========================
# CPU parallel version
# =========================
def transcribe_segments_parallel_cpu(
    audio_path: str,
    segments: List[Dict[str, float]],
    model_size: str,
    max_workers: int = 2,
    temp_dir: str = "outputs/chunks",
    merge_gap: float = 0.2,
    min_duration: float = 0.4,
    max_duration: float = 5.0,
    pad: float = 0.2,
    beam_size: int = 1,
    language: str = "zh",
    initial_prompt: str = "以下是繁體中文的多人談話內容。",
    keep_chunk_files: bool = True,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    CPU 平行版。
    """
    tasks = _prepare_transcription_tasks(
        audio_path=audio_path,
        segments=segments,
        temp_dir=temp_dir,
        merge_gap=merge_gap,
        min_duration=min_duration,
        max_duration=max_duration,
        pad=pad,
        keep_chunk_files=keep_chunk_files,
        debug=debug,
    )

    if not tasks:
        return []

    results = []

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(model_size,),
    ) as executor:
        futures = [
            executor.submit(
                _transcribe_one_chunk,
                task,
                beam_size,
                language,
                initial_prompt,
            )
            for task in tasks
        ]

        for future in futures:
            result = future.result()
            if result is not None:
                results.append(result)

    results.sort(key=lambda x: x["chunk_id"])
    return results