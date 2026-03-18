import os
import subprocess


'''
作用:

放小工具函式，不要把雜項塞進 pipeline。

'''
def extract_audio_segment(
    input_path: str,
    start: float,
    end: float,
    output_path: str
) -> str:
    """
    Extract a segment from audio using ffmpeg.

    Parameters
    ----------
    input_path : str
        Path to source wav file
    start : float
        Segment start time in seconds
    end : float
        Segment end time in seconds
    output_path : str
        Path to save the chunk wav file
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    if end <= start:
        raise ValueError(f"Invalid segment range: start={start}, end={end}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    duration = end - start

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ss", str(start),
        "-t", str(duration),
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
            "Failed to extract audio segment.\n"
            f"start={start}, end={end}\n"
            f"ffmpeg error:\n{result.stderr}"
        )

    return output_path