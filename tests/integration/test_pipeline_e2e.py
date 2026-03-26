from pipeline.runner import run_pipeline


def test_pipeline_e2e_smoke():
    result = run_pipeline(
        input_audio="data/sample.m4a",
        whisper_size="medium",
        workers=2,
        beam_size=1,
        enable_llm_refine=False,
        refine_mode="minimal",
    )

    assert isinstance(result, dict)
    assert result["status"] == "ok"
    assert result["source_audio"].endswith("sample.m4a")
    assert "plain_transcript" in result
    assert "json_path" in result
    assert "srt_path" in result