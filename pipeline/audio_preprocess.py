import os
import shutil
import subprocess


'''
功能:

轉 mono
resample 到 16000 Hz
輸出標準 wav

'''
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
    """
    Check whether the input file extension is supported.
    """
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_AUDIO_EXTENSIONS


def check_ffmpeg_installed():
    """
    Ensure ffmpeg is available in the system PATH.
    """
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg is not installed or not found in PATH. "
            "Please install ffmpeg first."
        )


def preprocess_audio(input_path: str, output_path: str) -> str:
    """
    Convert input audio/video file into standard ASR-ready wav format.

    Output format:
    - wav
    - mono
    - 16kHz
    - PCM 16-bit

    Supported input:
    - wav, mp3, m4a, aac, ogg, flac, mp4
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not is_supported_audio_file(input_path):
        raise ValueError(
            f"Unsupported file format: {input_path}\n"
            f"Supported formats: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}"
        )

    check_ffmpeg_installed()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",                  # overwrite output
        "-i", input_path,      # input file
        "-vn",                 # ignore video stream if exists
        "-acodec", "pcm_s16le",
        "-ac", "1",            # mono
        "-ar", "16000",        # 16kHz
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
            f"Input: {input_path}\n"
            f"Output: {output_path}\n"
            f"ffmpeg error:\n{result.stderr}"
        )

    if not os.path.exists(output_path):
        raise RuntimeError(f"Preprocessed file was not created: {output_path}")

    return output_path