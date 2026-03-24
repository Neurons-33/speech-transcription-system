import re
from opencc import OpenCC


# =========================
# Chinese conversion
# =========================
cc = OpenCC("s2t")


def to_traditional_chinese(text: str) -> str:
    """
    Convert Simplified Chinese to Traditional Chinese.
    """
    if not text:
        return ""
    return cc.convert(text)


# =========================
# Text cleaning
# =========================
def clean_transcribed_text(text: str) -> str:
    """
    基本清理：
    - 去除多餘空白
    - 壓縮重複標點
    """
    if not text:
        return ""

    text = text.strip()

    # normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # collapse repeated punctuation
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"[。\.]{2,}", "。", text)
    text = re.sub(r"[!！]{2,}", "！", text)
    text = re.sub(r"[?？]{2,}", "？", text)

    return text.strip()


def normalize_transcribed_text(text: str) -> str:
    """
    統一出口：
    - clean
    - 轉繁中
    """
    text = clean_transcribed_text(text)
    text = to_traditional_chinese(text)
    return text


# =========================
# Filtering rules
# =========================
def is_meaningless_text(text: str) -> bool:
    """
    過濾：
    - 空字串
    - 只有標點
    """
    if not text:
        return True

    stripped = text.strip()
    if stripped == "":
        return True

    if re.fullmatch(r"[，。！？、,.!?…~\\-]+", stripped):
        return True

    return False


def is_low_information_hallucination(
    text: str,
    chunk_duration: float,
    seg_duration: float,
) -> bool:
    """
    判斷 ASR hallucination：
    - 太短語音 + 太長文本
    """
    if not text:
        return True

    text_len = len(text.strip())

    if chunk_duration < 0.8 and text_len >= 10:
        return True

    if seg_duration < 0.6 and text_len >= 10:
        return True

    if seg_duration < 1.0 and text_len >= 14:
        return True

    return False


def should_skip_segment_text(
    raw_text: str,
    chunk_duration: float,
    seg_duration: float,
) -> bool:
    """
    統一判斷該 segment 是否應該丟棄
    """
    if is_meaningless_text(raw_text):
        return True

    text_len = len(raw_text.strip())
    chars_per_sec = text_len / max(seg_duration, 1e-6)

    # 過密（通常 hallucination）
    if chars_per_sec > 12:
        return True

    if is_low_information_hallucination(
        text=raw_text,
        chunk_duration=chunk_duration,
        seg_duration=seg_duration,
    ):
        return True

    return False