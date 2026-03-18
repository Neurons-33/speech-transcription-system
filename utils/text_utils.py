from opencc import OpenCC

# s2t = simplified to traditional
cc = OpenCC("s2t")


def to_traditional_chinese(text: str) -> str:
    """
    Convert Simplified Chinese to Traditional Chinese.
    """
    return cc.convert(text)