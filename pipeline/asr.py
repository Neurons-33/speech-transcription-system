import os
import re
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor

from utils.audio_utils import extract_audio_segment
from utils.text_utils import to_traditional_chinese
from models.whisper_model import load_whisper_model


# =========================
# Worker global
# =========================
_WORKER_MODEL = None


def _init_worker(model_size: str):
    """
    每個 process 啟動時各自載入一份 Whisper model。
    避免把 model 物件直接跨 process 傳遞。
    """
    global _WORKER_MODEL
    _WORKER_MODEL = load_whisper_model(model_size=model_size)


# =========================
# Text utils
# =========================
def clean_transcribed_text(text: str) -> str:
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"[。\.]{2,}", "。", text)
    text = re.sub(r"[!！]{2,}", "！", text)
    text = re.sub(r"[?？]{2,}", "？", text)
    return text.strip()


def is_meaningless_text(text: str) -> bool:
    if not text:
        return True

    stripped = text.strip()

    if stripped == "":
        return True

    if re.fullmatch(r"[，。！？、,.!?…~\\-]+", stripped):
        return True

    return False

def is_low_information_hallucination(
    text: str,
    chunk_duration: float,
    seg_duration: float,
) -> bool:
    """
    判斷是否為低資訊 hallucination
    """

    if not text:
        return True

    text_len = len(text.strip())

    # chunk 太短卻很多字
    if chunk_duration < 0.8 and text_len >= 10:
        return True

    # segment 太短卻很多字
    if seg_duration < 0.6 and text_len >= 10:
        return True

    # 稍短段落但字數異常多
    if seg_duration < 1.0 and text_len >= 14:
        return True

    return False


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
    先求穩定，不做 overlap。
    """
    results = []
    current = start

    while (end - current) > max_duration:
        results.append({
            "start": current,
            "end": current + max_duration,
        })
        current += max_duration

    if (end - current) >= min_duration:
        results.append({
            "start": current,
            "end": end,
        })

    return results


def optimize_segments_for_multi_speaker(
    segments: List[Dict[str, float]],
    merge_gap: float = 0.4,
    min_duration: float = 0.8,
    max_duration: float = 5.0,
) -> List[Dict[str, float]]:
    """
    多人交談版：
    1. 先清洗非法段
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

    # Step 1: 保守 merge
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

    # Step 2: 對過長段做非重疊硬切
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
# Prepare chunk tasks
# =========================
def _prepare_chunk_tasks(
    audio_path: str,
    optimized_segments: List[Dict[str, float]],
    temp_dir: str,
    min_duration: float,
    pad: float,
    keep_chunk_files: bool,
) -> List[Dict[str, Any]]:
    """
    在主程序先把所有 chunk wav 切好，之後再平行 ASR。
    """
    os.makedirs(temp_dir, exist_ok=True)

    tasks = []

    for i, seg in enumerate(optimized_segments):
        chunk_start = seg["start"]
        chunk_end = seg["end"]
        chunk_duration = chunk_end - chunk_start

        if chunk_duration < min_duration:
            continue

        pad_start = max(0.0, chunk_start - pad)
        pad_end = chunk_end + pad

        chunk_filename = f"chunk_{i:05d}.wav"
        chunk_path = os.path.join(temp_dir, chunk_filename)

        extract_audio_segment(
            input_path=audio_path,
            start=pad_start,
            end=pad_end,
            output_path=chunk_path,
        )

        tasks.append(
            {
                "chunk_id": i,
                "chunk_start": round(chunk_start, 3),
                "chunk_end": round(chunk_end, 3),
                "chunk_duration": round(chunk_duration, 3),
                "pad_start": round(pad_start, 3),
                "pad_end": round(pad_end, 3),
                "chunk_path": chunk_path,
                "keep_chunk_files": keep_chunk_files,
            }
        )

    return tasks


# =========================
# Single worker transcription
# =========================
def _transcribe_one_chunk(task: Dict[str, Any], beam_size: int, language: str, initial_prompt: str | None) -> Dict[str, Any] | None:
    """
    單一 worker 處理一個 chunk。
    """
    global _WORKER_MODEL

    if _WORKER_MODEL is None:
        raise RuntimeError("Worker model is not initialized.")

    chunk_id = task["chunk_id"]
    chunk_start = task["chunk_start"]
    chunk_end = task["chunk_end"]
    chunk_duration = task["chunk_duration"]
    pad_start = task["pad_start"]
    chunk_path = task["chunk_path"]
    keep_chunk_files = task["keep_chunk_files"]

    segments_gen, info = _WORKER_MODEL.transcribe(
        chunk_path,
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

    sub_segments = []

    for s in segments_gen:
        raw_text = clean_transcribed_text(s.text.strip())

        if is_meaningless_text(raw_text):
            continue
        seg_duration = float(s.end) - float(s.start)
        text_len = len(raw_text.strip())

        # 👉 新增：語速密度檢查（超關鍵）
        chars_per_sec = text_len / max(seg_duration, 1e-6)

        # 如果語速異常（太快或太慢）
        if chars_per_sec > 12:
            continue

        if is_low_information_hallucination(
            text=raw_text,
            chunk_duration=chunk_duration,
            seg_duration=seg_duration,
        ):
            continue

        text = to_traditional_chinese(raw_text)

        abs_start = pad_start + float(s.start)
        abs_end = pad_start + float(s.end)

        abs_start = max(0.0, abs_start)
        abs_end = max(abs_start, abs_end)

        sub_segments.append(
            {
                "start": round(abs_start, 3),
                "end": round(abs_end, 3),
                "text": text,
                "speaker": None,
            }
        )

    if not sub_segments:
        if not keep_chunk_files and os.path.exists(chunk_path):
            os.remove(chunk_path)
        return None

    result = {
        "chunk_id": chunk_id,
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
        "chunk_duration": chunk_duration,
        "sub_segments": sub_segments,
        "audio_path": chunk_path if keep_chunk_files else None,
    }

    if not keep_chunk_files and os.path.exists(chunk_path):
        os.remove(chunk_path)

    return result


# =========================
# Original sequential version (keep for fallback)
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
    保留原本單線版，作為 fallback。
    """
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

    all_results = []

    for i, seg in enumerate(optimized_segments):
        chunk_start = seg["start"]
        chunk_end = seg["end"]
        chunk_duration = chunk_end - chunk_start

        if chunk_duration < min_duration:
            continue

        pad_start = max(0.0, chunk_start - pad)
        pad_end = chunk_end + pad

        chunk_filename = f"chunk_{i:05d}.wav"
        chunk_path = os.path.join(temp_dir, chunk_filename)

        extract_audio_segment(
            input_path=audio_path,
            start=pad_start,
            end=pad_end,
            output_path=chunk_path,
        )

        segments_gen, info = model.transcribe(
            chunk_path,
            beam_size=beam_size,
            language=language,
            task="transcribe",
            word_timestamps=False,
            initial_prompt=initial_prompt,
            vad_filter=False,
        )

        sub_segments = []

        for s in segments_gen:
            raw_text = clean_transcribed_text(s.text.strip())

            if is_meaningless_text(raw_text):
                continue

            text = to_traditional_chinese(raw_text)

            abs_start = pad_start + float(s.start)
            abs_end = pad_start + float(s.end)

            abs_start = max(0.0, abs_start)
            abs_end = max(abs_start, abs_end)

            sub_segments.append(
                {
                    "start": round(abs_start, 3),
                    "end": round(abs_end, 3),
                    "text": text,
                    "speaker": None,
                }
            )

        if not sub_segments:
            if not keep_chunk_files and os.path.exists(chunk_path):
                os.remove(chunk_path)
            continue

        result = {
            "chunk_id": i,
            "chunk_start": round(chunk_start, 3),
            "chunk_end": round(chunk_end, 3),
            "chunk_duration": round(chunk_duration, 3),
            "sub_segments": sub_segments,
            "audio_path": chunk_path if keep_chunk_files else None,
        }

        all_results.append(result)

        if debug:
            print(f"[TRANSCRIBED] chunk_{i:05d}")

        if not keep_chunk_files and os.path.exists(chunk_path):
            os.remove(chunk_path)

    return all_results


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
    CPU 平行版：
    1. 先做 segment optimize
    2. 主程序先切好 chunk wav
    3. 用 ProcessPoolExecutor 平行 ASR
    4. 回傳 schema 與原本單線版一致
    """
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

    tasks = _prepare_chunk_tasks(
        audio_path=audio_path,
        optimized_segments=optimized_segments,
        temp_dir=temp_dir,
        min_duration=min_duration,
        pad=pad,
        keep_chunk_files=keep_chunk_files,
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