import torch

'''
回傳格式設計:
[
    {"start": 0.5, "end": 3.2},
    {"start": 4.1, "end": 7.8}
]

V1 不要先做太聰明的切段規則。
先能切出「大概有聲音的範圍」就夠。

'''


def merge_speech_segments(
    segments: list,
    min_gap: float = 0.4,
    min_duration: float = 0.6,
    max_duration: float = 20.0,
):
    """
    Post-process speech segments:
    1. drop very short segments
    2. merge nearby segments
    3. prevent merged segment from becoming too long
    """

    if not segments:
        return []

    # 1) filter out very short segments
    filtered = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        if duration >= min_duration:
            filtered.append(seg)

    if not filtered:
        return []

    # 2) merge nearby segments
    merged = []
    current = filtered[0].copy()

    for nxt in filtered[1:]:
        gap = nxt["start"] - current["end"]
        merged_duration_if_joined = nxt["end"] - current["start"]

        if gap <= min_gap and merged_duration_if_joined <= max_duration:
            current["end"] = nxt["end"]
        else:
            merged.append(current)
            current = nxt.copy()

    merged.append(current)
    return merged


def detect_speech_segments(audio_path: str, model, utils):
    """
    Detect speech segments using Silero VAD, then post-process them.
    """

    (
        get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks,
    ) = utils

    wav = read_audio(audio_path)

    raw_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
        min_silence_duration_ms=900,
        speech_pad_ms=300,
    )

    raw_segments = []
    for seg in raw_timestamps:
        raw_segments.append({
            "start": seg["start"] / 16000,
            "end": seg["end"] / 16000,
        })

    merged_segments = merge_speech_segments(
        raw_segments,
        min_gap=1.0,
        min_duration=1.2,
        max_duration=18.0,
    )

    print(f"[VAD] raw segments: {len(raw_segments)}")
    print(f"[VAD] merged segments: {len(merged_segments)}")

    return merged_segments