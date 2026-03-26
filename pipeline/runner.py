from __future__ import annotations

import copy
import os
import time
from typing import Any, Dict, List

from pipeline.audio_preprocess import preprocess_audio
from pipeline.vad import detect_speech_segments
from pipeline.asr import transcribe_segments_parallel_cpu

from models.vad_model import load_vad_model

from utils.file_utils import save_text, save_json, get_next_run_id, ensure_dir
from utils.srt_utils import build_srt_from_chunks

from services.llm_refine.client import LLMRefineClient
from services.llm_refine.orchestrator import refine_transcript


# =========================
# Formatting helpers
# =========================
def format_timestamp(seconds: float) -> str:
    """
    將秒數轉成 HH:MM:SS.mmm
    """
    ms = int((seconds % 1) * 1000)
    total_seconds = int(seconds)
    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = total_seconds // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def get_display_text(seg: Dict[str, Any]) -> str:
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


def get_display_speaker(seg: Dict[str, Any]) -> str | None:
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


# =========================
# Transcript builders
# =========================
def build_plain_transcript(chunks: List[Dict[str, Any]]) -> str:
    texts = []

    for chunk in chunks:
        for seg in chunk.get("sub_segments", []):
            text = get_display_text(seg)
            if text:
                texts.append(text)

    return " ".join(texts).strip()


def build_timestamped_transcript(chunks: List[Dict[str, Any]]) -> str:
    lines = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        chunk_start = chunk.get("chunk_start", 0.0)
        chunk_end = chunk.get("chunk_end", 0.0)
        chunk_audio_path = chunk.get("audio_path", "")

        lines.append(
            f"[Chunk {chunk_id:03d}] "
            f"{format_timestamp(chunk_start)} --> {format_timestamp(chunk_end)}"
        )

        if chunk_audio_path:
            lines.append(f"audio: {chunk_audio_path}")

        for seg in chunk.get("sub_segments", []):
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = get_display_text(seg)
            speaker = get_display_speaker(seg)

            if not text:
                continue

            if speaker:
                lines.append(
                    f"  [{format_timestamp(start)} --> {format_timestamp(end)}] "
                    f"{speaker}: {text}"
                )
            else:
                lines.append(
                    f"  [{format_timestamp(start)} --> {format_timestamp(end)}] "
                    f"{text}"
                )

        lines.append("")

    return "\n".join(lines).strip()


def build_transcript_document(
    *,
    file_id: str,
    source_audio: str,
    processed_audio: str,
    chunks: List[Dict[str, Any]],
    language: str,
    pipeline_version: str,
    srt_exported: bool,
    llm_refined: bool,
) -> Dict[str, Any]:
    total_duration = 0.0
    if chunks:
        total_duration = max(chunk.get("chunk_end", 0.0) for chunk in chunks)

    return {
        "file_id": file_id,
        "source_audio": source_audio,
        "processed_audio": processed_audio,
        "language": language,
        "duration": round(total_duration, 3),
        "pipeline_version": pipeline_version,
        "processing": {
            "asr_done": True,
            "diarization_done": False,
            "llm_refined": llm_refined,
            "srt_exported": srt_exported,
        },
        "chunks": chunks,
    }


def flatten_sub_segments_from_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat_segments: List[Dict[str, Any]] = []
    segment_counter = 0

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")

        for sub_index, seg in enumerate(chunk.get("sub_segments", [])):
            raw_text = (
                (seg.get("text") or "").strip()
                or (seg.get("raw_text") or "").strip()
                or (seg.get("normalized_text") or "").strip()
                or (seg.get("refined_text") or "").strip()
            )
            if not raw_text:
                continue

            flat_segments.append({
                "segment_id": segment_counter,
                "chunk_id": chunk_id,
                "sub_index": sub_index,
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": raw_text,
                "raw_text": raw_text,
                "speaker_id": seg.get("speaker_id"),
                "speaker_label": seg.get("speaker_label"),
            })
            segment_counter += 1

    return flat_segments


def apply_refined_segments_to_chunks(
    chunks: List[Dict[str, Any]],
    refined_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # 用 (chunk_id, sub_index) 建立回填索引
    refined_map: Dict[tuple, Dict[str, Any]] = {}

    for seg in refined_segments:
        chunk_id = seg.get("chunk_id")
        sub_index = seg.get("sub_index")
        if chunk_id is None or sub_index is None:
            continue
        refined_map[(chunk_id, sub_index)] = seg

    print("[DEBUG] refined_map keys sample:", list(refined_map.keys())[:5])
    output_chunks: List[Dict[str, Any]] = []
    print("[DEBUG] refined_map size:", len(refined_map))
    for chunk in chunks:
        new_chunk = dict(chunk)
        new_sub_segments = []
        chunk_id = chunk.get("chunk_id")

        for sub_index, seg in enumerate(chunk.get("sub_segments", [])):
            new_seg = dict(seg)

            raw_text = (
                (seg.get("text") or "").strip()
                or (seg.get("raw_text") or "").strip()
                or (seg.get("normalized_text") or "").strip()
                or (seg.get("refined_text") or "").strip()
            )

            # 沒文字就原樣保留
            if not raw_text:
                new_sub_segments.append(new_seg)
                continue

            refined_seg = refined_map.get((chunk_id, sub_index), {})
            print(
                "[DEBUG] apply:",
                (chunk_id, sub_index),
                "->",
                "HIT" if refined_seg else "MISS"
            )
            refined_text = (
                (refined_seg.get("text") or "").strip()
                or raw_text
            )

            # 保留原本 flatten 時建立的 segment_id；若沒有就補上 refined 的
            new_seg["segment_id"] = (
                seg.get("segment_id")
                if seg.get("segment_id") is not None
                else refined_seg.get("segment_id")
            )
            new_seg["raw_text"] = raw_text
            new_seg["refined_text"] = refined_text

            new_sub_segments.append(new_seg)

        new_chunk["sub_segments"] = new_sub_segments
        output_chunks.append(new_chunk)

    return output_chunks


def maybe_refine_chunks_with_llm(
    chunks: List[Dict[str, Any]],
    enable_llm_refine: bool,
    refine_mode: str = "minimal",
) -> tuple[List[Dict[str, Any]], bool]:
    """
    回傳：
    - updated_chunks
    - llm_refined_done

    注意：
    - llm_refined_done 表示 refine 流程有執行完成
    - 不保證所有 segment 都真的被修改
    """
    print("[DEBUG] maybe_refine_chunks_with_llm called")

    if not enable_llm_refine:
        return chunks, False

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[LLM] GEMINI_API_KEY not found. Skip LLM refine.")
        return chunks, False

    flat_segments = flatten_sub_segments_from_chunks(chunks)
    print(f"[LLM] flatten segments: {len(flat_segments)}")

    if not flat_segments:
        print("[LLM] No valid segments found. Skip LLM refine.")
        return chunks, False

    client = LLMRefineClient(
        api_key=api_key,
        model="gemini-2.5-flash-lite",
        max_retries=2,
        retry_delay=1.0,
    )

    print("[LLM] Start transcript refinement...", end=" ")
    t0 = time.perf_counter()

    refined_segments = refine_transcript(
        segments=flat_segments,
        client=client,
        max_items=8,
        max_chars=1200,
        overlap_items=2,
        mode=refine_mode,
    )

    print("\n[DEBUG] refined_segments sample:")
    for r in refined_segments[:3]:
        print({
            "chunk_id": r.get("chunk_id"),
            "sub_index": r.get("sub_index"),
            "text": r.get("text")
        })

    updated_chunks = apply_refined_segments_to_chunks(chunks, refined_segments)

    print(f"done ({time.perf_counter() - t0:.3f}s)")
    return updated_chunks, True


def run_pipeline(
    *,
    input_audio: str,
    whisper_size: str = "medium",
    merge_gap: float = 0.25,
    min_duration: float = 0.4,
    max_duration: float = 5.0,
    pad: float = 0.2,
    workers: int = 2,
    beam_size: int = 3,
    enable_llm_refine: bool = False,
    refine_mode: str = "minimal",
    normalize: bool = True,
    denoise: bool = False,
) -> Dict[str, Any]:
    """
    真正的 pipeline 主入口。
    CLI 與 pytest integration test 都應呼叫這個函式。
    """
    audio_root = "outputs/audio"
    chunks_root = "outputs/chunks"
    transcripts_root = "outputs/transcripts"

    ensure_dir(audio_root)
    ensure_dir(chunks_root)
    ensure_dir(transcripts_root)

    run_id = get_next_run_id(audio_root)

    processed_audio = f"{audio_root}/{run_id}.wav"
    chunks_dir = f"{chunks_root}/{run_id}"
    transcript_dir = f"{transcripts_root}/{run_id}"

    ensure_dir(chunks_dir)
    ensure_dir(transcript_dir)

    txt_path = f"{transcript_dir}/{run_id}.txt"
    json_path = f"{transcript_dir}/{run_id}.json"
    plain_txt_path = f"{transcript_dir}/{run_id}_plain.txt"
    raw_plain_txt_path = f"{transcript_dir}/{run_id}_raw_plain.txt"
    refined_plain_txt_path = f"{transcript_dir}/{run_id}_refined_plain.txt"
    srt_path = f"{transcript_dir}/{run_id}.srt"

    total_start = time.perf_counter()

    print(f"Run ID: {run_id}")

    print("Preprocessing audio...", end=" ")
    t0 = time.perf_counter()
    preprocess_audio(input_audio, processed_audio, normalize=normalize, denoise=denoise)
    print(f"done ({time.perf_counter() - t0:.3f}s)")

    print("Loading VAD model...", end=" ")
    t0 = time.perf_counter()
    vad_model, vad_utils = load_vad_model()
    print(f"done ({time.perf_counter() - t0:.3f}s)")

    print("Detecting speech segments...", end=" ")
    t0 = time.perf_counter()
    speech_segments = detect_speech_segments(
        processed_audio,
        vad_model,
        vad_utils,
    )
    elapsed = time.perf_counter() - t0
    print(f"done ({elapsed:.3f}s) | detected {len(speech_segments)} speech segments")

    print(
        f"Running CPU Parallel ASR "
        f"(workers={workers}, model={whisper_size}, beam={beam_size})...",
        end=" "
    )
    t0 = time.perf_counter()
    chunk_results = transcribe_segments_parallel_cpu(
        audio_path=processed_audio,
        segments=speech_segments,
        model_size=whisper_size,
        max_workers=workers,
        temp_dir=chunks_dir,
        merge_gap=merge_gap,
        min_duration=min_duration,
        max_duration=max_duration,
        pad=pad,
        beam_size=beam_size,
        language="zh",
        initial_prompt=None,
        keep_chunk_files=True,
        debug=False,
    )
    print(f"done ({time.perf_counter() - t0:.3f}s)")

    original_chunk_results = copy.deepcopy(chunk_results)

    chunk_results, llm_refined_done = maybe_refine_chunks_with_llm(
        chunks=chunk_results,
        enable_llm_refine=enable_llm_refine,
        refine_mode=refine_mode,
    )
    original_plain_transcript = build_plain_transcript(original_chunk_results)
    refined_plain_transcript = build_plain_transcript(chunk_results)

    print("\n[DEBUG] FINAL CHECK:")
    for c in chunk_results[:1]:
        for s in c["sub_segments"][:3]:
            print("RAW:", s.get("raw_text"))
            print("REF:", s.get("refined_text"))
            print("----")

    plain_transcript = refined_plain_transcript
    timestamped_transcript = build_timestamped_transcript(chunk_results)
    srt_text = build_srt_from_chunks(chunk_results)

    transcript_doc = build_transcript_document(
        file_id=run_id,
        source_audio=input_audio,
        processed_audio=processed_audio,
        chunks=chunk_results,
        language="zh",
        pipeline_version="v2",
        srt_exported=True,
        llm_refined=llm_refined_done,
    )

    print("Saving outputs...", end=" ")
    t0 = time.perf_counter()
    save_text(timestamped_transcript, txt_path)
    save_text(plain_transcript, plain_txt_path)
    save_text(original_plain_transcript, raw_plain_txt_path)
    save_text(refined_plain_transcript, refined_plain_txt_path)
    save_text(srt_text, srt_path)
    save_json(transcript_doc, json_path)
    print(f"done ({time.perf_counter() - t0:.3f}s)")

    total_elapsed = time.perf_counter() - total_start
    print(f"\nTotal pipeline time: {total_elapsed:.3f}s")

    return {
        "status": "ok",
        "run_id": run_id,
        "source_audio": input_audio,
        "processed_audio": processed_audio,
        "speech_segments": speech_segments,
        "chunk_results": chunk_results,
        "plain_transcript": plain_transcript,
        "original_plain_transcript": original_plain_transcript,
        "refined_plain_transcript": refined_plain_transcript,
        "timestamped_transcript": timestamped_transcript,
        "srt_text": srt_text,
        "transcript_doc": transcript_doc,
        "txt_path": txt_path,
        "plain_txt_path": plain_txt_path,
        "raw_plain_txt_path": raw_plain_txt_path,
        "refined_plain_txt_path": refined_plain_txt_path,
        "srt_path": srt_path,
        "json_path": json_path,
        "chunks_dir": chunks_dir,
        "llm_refined_done": llm_refined_done,
        "total_elapsed": total_elapsed,
    }