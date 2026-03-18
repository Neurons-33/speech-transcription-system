import argparse
import time

from pipeline.audio_preprocess import preprocess_audio
from pipeline.vad import detect_speech_segments
from pipeline.asr import transcribe_segments_parallel_cpu

from models.vad_model import load_vad_model

from utils.file_utils import save_text, save_json, get_next_run_id, ensure_dir


"""
V1 主入口（多人交談版 / CPU 平行 ASR）
重點：
1. 保留時間戳
2. 保留 chunk -> sub_segments 結構
3. 讓 transcript 可作為後續上下文連結依據
4. 只把 ASR 從單線改成 CPU 平行
"""


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


def build_plain_transcript(results: list) -> str:
    """
    純文字版本：
    只把所有 sub_segments text 串起來，適合快速看全文。
    """
    texts = []

    for chunk in results:
        for seg in chunk.get("sub_segments", []):
            text = seg.get("text", "").strip()
            if text:
                texts.append(text)

    return " ".join(texts).strip()


def build_timestamped_transcript(results: list) -> str:
    """
    帶時間戳版本：
    保留每個 sub_segment 的時間範圍，方便回查原始音訊。
    """
    lines = []

    for chunk in results:
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
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", None)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input audio path")
    parser.add_argument("--whisper_size", type=str, default="medium", help="whisper model size")
    parser.add_argument("--merge_gap", type=float, default=0.25, help="merge gap for multi-speaker chunks")
    parser.add_argument("--min_duration", type=float, default=0.4, help="minimum chunk duration")
    parser.add_argument("--max_duration", type=float, default=5.0, help="maximum chunk duration")
    parser.add_argument("--pad", type=float, default=0.2, help="padding seconds for ASR chunk")
    parser.add_argument("--workers", type=int, default=2, help="number of CPU workers")
    parser.add_argument("--beam_size", type=int, default=3, help="beam size for ASR decoding")
    args = parser.parse_args()

    input_audio = args.input

    # outputs structure
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

    print(f"Run ID: {run_id}")

    total_start = time.perf_counter()

    print("Preprocessing audio...", end=" ")
    t0 = time.perf_counter()
    preprocess_audio(input_audio, processed_audio)
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
        vad_utils
    )
    elapsed = time.perf_counter() - t0
    print(f"done ({elapsed:.3f}s) | detected {len(speech_segments)} speech segments")

    print(f"Running CPU Parallel ASR (workers={args.workers}, model={args.whisper_size}, beam={args.beam_size})...", end=" ")
    t0 = time.perf_counter()
    results = transcribe_segments_parallel_cpu(
        audio_path=processed_audio,
        segments=speech_segments,
        model_size=args.whisper_size,
        max_workers=args.workers,
        temp_dir=chunks_dir,
        merge_gap=args.merge_gap,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        pad=args.pad,
        beam_size=args.beam_size,
        language="zh",
        initial_prompt=None,
        keep_chunk_files=True,
        debug=False,
    )
    print(f"done ({time.perf_counter() - t0:.3f}s)")

    # 文字輸出
    plain_transcript = build_plain_transcript(results)
    timestamped_transcript = build_timestamped_transcript(results)

    print("Saving outputs...", end=" ")
    t0 = time.perf_counter()
    save_text(timestamped_transcript, txt_path)
    save_text(plain_transcript, plain_txt_path)
    save_json(results, json_path)
    print(f"done ({time.perf_counter() - t0:.3f}s)")

    total_elapsed = time.perf_counter() - total_start
    print(f"\nTotal pipeline time: {total_elapsed:.3f}s")

    print("\nPlain Transcript:\n")
    print(plain_transcript)

    print(f"\nSaved processed wav : {processed_audio}")
    print(f"Saved transcript txt: {txt_path}")
    print(f"Saved plain txt     : {plain_txt_path}")
    print(f"Saved json          : {json_path}")
    print(f"Saved chunks dir    : {chunks_dir}")


if __name__ == "__main__":
    main()