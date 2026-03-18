import torch


'''

初始化 VAD 模型
return model

'''

def load_vad_model():
    """
    Load Silero VAD model
    """

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
    )

    return model, utils