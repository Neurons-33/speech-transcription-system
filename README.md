# Speech Transcription Pipeline（V1.1 - Structured ASR + SRT）

一個以 Python 建立的語音轉錄處理流程（pipeline），  
目前版本為 **結構化 ASR pipeline（V1.1）**，
支援 JSON schema 與多格式輸出（TXT / SRT）。

---

## 專案狀態

-  V1：CPU baseline（已完成）
-  V1.1：結構化輸出 + SRT（已完成）
-  V2：LLM 轉錄修正（規劃中）
-  V3：GPU + FastAPI + Docker
-  V4：說話者分離（diarization）

---

## 專案目標

建立一套模組化語音處理系統，包含：

- 音訊前處理與格式標準化
- 語音活動偵測（VAD）
- 音訊切段（chunk）
- 語音轉文字（ASR）
- 多格式輸出（JSON / TXT）
- 字幕格式（SRT）
- 可延伸支援 LLM / diarization / API

並作為後續 GPU 與部署版本的基礎。

---

## 目前功能（V1.1）

### 已完成

- 支援 `.m4a / .wav` 音檔輸入
- 音訊前處理（mono / 16k / wav）
- VAD 語音偵測 (Silero)
- chunk 切段與時間對齊
- CPU 平行 ASR (faster-whisper)
- 繁體中文轉換 （OpenCC）
- 結構化 JSON 輸出（file-level schema）
- TXT / Plain TXT / JSON / SRT 多格式輸出

---

## 輸出格式說明

本專案輸出四種逐字稿格式：

### 1 `.json`（核心資料）

- 結構化語音資料（file → chunk → segment）
- 保留時間戳與 metadata
- 支援未來：
  - speaker diarization
  - LLM refinement
  - embedding / search

 **建議作為主資料來源**

---

### 2 `.txt`（帶時間戳）

- chunk + segment 可讀格式
- 適合 debug / trace

---

### 3 `_plain.txt`

- 純文字拼接
- 適合快速閱讀

---

### 4 `.srt`（字幕）

- 可直接用於影片 / 播放器
- 支援時間對齊

---

## JSON Schema（核心設計）

```json
{
  "file_id": "...",
  "processing": {
    "asr_done": true,
    "diarization_done": false,
    "llm_refined": false,
    "srt_exported": true
  },
  "chunks": [
    {
      "chunk_id": 0,
      "sub_segments": [
        {
          "raw_text": "...",
          "normalized_text": "...",
          "refined_text": null,
          "speaker_id": null
        }
      ]
    }
  ]
}
```

此 JSON schema 為整個系統的 **source of truth**，
所有輸出格式（TXT / SRT）皆由此衍生，
並作為未來 LLM、diarization 與 API 的核心資料結構。

---

## 專案結構

```text
.
├─ data/
│   └─ sample.m4a               # 測試音檔
│
├─ models/
│   ├─ vad_model.py             # VAD模型載入 / 管理
│   └─ whisper_model.py         # Whisper模型載入 / 管理
│
├─ outputs/
│   ├─ audio/                   # 預處理後音訊
│   ├─ chunks/                  # 切段結果
│   └─ transcripts/             # JSON source + TXT / SRT views
│
├─ pipeline/
│   ├─ asr.py                   # chunk切分、ASR轉錄與文字過濾
│   ├─ audio_preprocess.py      # 音訊前處理與格式標準化
│   └─ vad.py                   # VAD語音偵測
│
├─ services/                    # (future) LLM / refinement
│
├─ api/                         # (future) FastAPI entrypoint
│
├─ scripts/
│   └─ test_pipeline.py         # CLI測試入口
│
├─ utils/
│   ├─ audio_utils.py           # ffmpeg / audio處理
│   ├─ file_utils.py            # File I/O / 路徑 / 儲存
│   ├─ text_utils.py            # 基礎文字清理與正規化（rule-based）
│   └─ srt_utils.py             # 字幕格式生成
│
├─ requirements.txt
└─ requirements.lock

```
---

## 輸出範例

### pipeline

<p align="center">
  <img src="assets/pipeline.png" width="350">
</p>
本 pipeline 採用 **data-centric design**：

- 每個階段皆為資料轉換（audio → segments → structured transcript）
- JSON 為核心資料來源（source of truth）
- TXT / SRT 為輸出視圖（views）

### TXT 逐字稿

<p align="center">
  <img src="assets/sample_txt.png">
</p>

### JSON 結構

<p align="center">
  <img src="assets/sample_json.png">
</p>

### SRT 結構

<p align="center">
  <img src="assets/sample_srt.png">
</p>

---

## 執行環境

Python 3.11（建議）

CPU 執行（目前未使用 GPU）

---

## 基本執行

```bash
python -m scripts.test_pipeline --input data/sample.m4a
```
---

## 進階參數
```bash
python -m scripts.test_pipeline \
  --input data/sample.m4a \
  --workers 4 \
  --beam_size 3 \
  --whisper_size medium \
  --min_duration 0.6 \
  --merge_gap 0.35 \
  --pad 0.1
```

---

## 參數說明

| 參數 | 說明 |
|------|------|
| `--input` | 輸入音檔路徑 |
| `--workers` | CPU 平行處理數量 |
| `--beam_size` | beam search 大小（越大越準但越慢） |
| `--whisper_size` | 模型大小（base / small / medium） |
| `--min_duration` | 最小語音片段長度（秒） |
| `--merge_gap` | 片段合併間隔（秒） |
| `--pad` | 每段音訊前後 padding（秒） |

---

## 目前限制 (V1.1)

- 僅支援 CPU（速度有限）
- 長音檔處理時間較長
- chunk 策略仍為簡化版本
- 尚未支援說話者分離
- 尚未整合 LLM 修正
- 尚未部署 Web / API

---

## 後續規劃

- GPU 自動偵測與加速
- chunk 策略優化（動態切分）
- speaker diarization（說話者辨識）
- LLM 錯字修正機制
- Colab 測試版本
- Web / API 部署

