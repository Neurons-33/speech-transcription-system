from __future__ import annotations

import argparse

from pipeline.runner import run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input audio path")
    parser.add_argument("--whisper_size", type=str, default="medium", help="whisper model size")
    parser.add_argument("--merge_gap", type=float, default=0.25, help="merge gap for multi-speaker chunks")
    parser.add_argument("--min_duration", type=float, default=0.4, help="minimum chunk duration")
    parser.add_argument("--max_duration", type=float, default=5.0, help="maximum chunk duration")
    parser.add_argument("--pad", type=float, default=0.2, help="padding seconds for ASR chunk")
    parser.add_argument("--workers", type=int, default=2, help="number of CPU workers")
    parser.add_argument("--beam_size", type=int, default=3, help="beam size for ASR decoding")
    parser.add_argument("--enable_llm_refine", action="store_true", help="Enable LLM-based transcript refinement")
    parser.add_argument("--refine_mode", type=str, default="minimal", choices=["minimal", "readable"], help="LLM refine mode")
    parser.add_argument("--denoise", action="store_true", help="Enable audio denoise")
    parser.add_argument("--no-normalize", action="store_true", help="Disable audio normalization")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    result = run_pipeline(
        input_audio=args.input,
        whisper_size=args.whisper_size,
        merge_gap=args.merge_gap,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        pad=args.pad,
        workers=args.workers,
        beam_size=args.beam_size,
        enable_llm_refine=args.enable_llm_refine,
        refine_mode=args.refine_mode,
        normalize=not args.no_normalize,
        denoise=args.denoise,
    )

    print("\nPlain Transcript:\n")
    print(result["plain_transcript"])

    print(f"\nSaved processed wav : {result['processed_audio']}")
    print(f"Saved transcript txt: {result['txt_path']}")
    print(f"Saved plain txt     : {result['plain_txt_path']}")
    print(f"Saved srt           : {result['srt_path']}")
    print(f"Saved json          : {result['json_path']}")
    print(f"Saved chunks dir    : {result['chunks_dir']}")


if __name__ == "__main__":
    main()