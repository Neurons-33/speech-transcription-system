# tests/fixtures/llm_refine_segments.py

"""
集中管理 llm_refine 測試用的假資料
"""

BASIC_SEGMENTS = [
    {"start": 0.0, "end": 1.0, "text": "大家好"},
    {"start": 1.0, "end": 2.0, "text": "今天我們來介紹這個系統"},
    {"start": 2.0, "end": 3.0, "text": "它目前支援基本的語音轉文字功能"},
]

SEGMENTS_WITH_ID = [
    {"segment_id": 10, "start": 0.0, "end": 1.0, "text": "你好"},
    {"segment_id": 11, "start": 1.0, "end": 2.0, "text": "世界"},
]

DUPLICATE_ID_SEGMENTS = [
    {"segment_id": 1, "start": 0.0, "end": 1.0, "text": "A"},
    {"segment_id": 1, "start": 1.0, "end": 2.0, "text": "B"},
]

EMPTY_TEXT_SEGMENTS = [
    {"start": 0.0, "end": 1.0, "text": ""},
    {"start": 1.0, "end": 2.0, "text": "   "},
]