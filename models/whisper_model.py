from faster_whisper import WhisperModel


'''
功能:
初始化 whisper / faster-whisper
return model
'''


def load_whisper_model(model_size: str = "medium"):
    """
    Load Faster Whisper model for CPU parallel ASR.
    """

    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=2,
        num_workers=1,
    )

    return model