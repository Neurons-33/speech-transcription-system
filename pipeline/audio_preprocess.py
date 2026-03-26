import os
import shutil
import subprocess


SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac",
    ".mp4",
}


def is_supported_audio_file(file_path: str) -> bool:
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_AUDIO_EXTENSIONS


def check_ffmpeg_installed():
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg is not installed or not found in PATH."
        )


def build_audio_filters(
    normalize: bool = True,
    denoise: bool = False,
) -> str:
    """
    Build ffmpeg audio filter chain.

    Order matters:
    normalize → denoise
    """

    filters = []

    if normalize:
        # loudness normalization (關鍵)
        filters.append("loudnorm")

    if denoise:
        # basic denoise (可換更高級)
        filters.append("afftdn")

    return ",".join(filters) if filters else None


def preprocess_audio(
    input_path: str,
    output_path: str,
    normalize: bool = True,
    denoise: bool = False,
) -> str:
    """
    Convert input audio/video file into ASR-ready wav.

    Output:
    - wav
    - mono
    - 16kHz
    - PCM 16-bit
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not is_supported_audio_file(input_path):
        raise ValueError(
            f"Unsupported format: {input_path}\n"
            f"Supported: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}"
        )

    check_ffmpeg_installed()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 🔥 build filter chain
    audio_filters = build_audio_filters(
        normalize=normalize,
        denoise=denoise,
    )

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",
    ]

    if audio_filters:
        command += ["-af", audio_filters]

    command += [
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        output_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Audio preprocessing failed.\n"
            f"{result.stderr}"
        )

    if not os.path.exists(output_path):
        raise RuntimeError(f"Output not created: {output_path}")

    return output_path