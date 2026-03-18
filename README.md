# Speech Transcription Pipeline（V1 - CPU Baseline）

一個以 Python 建立的語音轉錄處理流程（pipeline），  
目前版本為 **CPU 可執行 baseline（V1）**，用於驗證完整語音處理流程。

---

## 專案狀態

-  V1：CPU baseline（已完成）
-  V1.1：字幕輸出（SRT）（開發中）
-  V2：GPU 加速（規劃中）
-  V3：說話者分離（diarization）
-  V4：LLM 轉錄修正

---

## 專案目標

建立一套模組化語音處理系統，包含：

- 音訊前處理與格式標準化
- 語音活動偵測（VAD）
- 音訊切段（chunk）
- 語音轉文字（ASR）
- 多格式輸出（JSON / TXT）
- 可延伸支援字幕格式（SRT）

並作為後續 GPU 與部署版本的基礎。

---

## 目前功能（V1）

### 已完成

- 支援 `.m4a / .wav` 音檔輸入
- 音訊前處理（mono / 16k / wav）
- VAD 語音偵測 (Silero)
- chunk 切段與時間對齊
- CPU 平行 ASR (faster-whisper)
- 逐字稿輸出（繁體中文）
- JSON / TXT 雙格式輸出

---

## 輸出格式說明

本專案輸出三種逐字稿格式：

- `.json`（核心資料）
  - 結構化語音資料（時間戳、chunk、sub-segment）
  - 用於後續處理（LLM / diarization / embedding）

- `.txt`（閱讀用）
  - 人類可讀逐字稿

- `.srt`（規劃中）
  - 字幕格式（可用於影片 / 播放器）

👉 建議以 `.json` 作為主要資料來源。

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
│   └─ transcripts/             # 逐字稿輸出
│
├─ pipeline/
│   ├─ asr.py                   # chunk切分、ASR轉錄與文字過濾
│   ├─ audio_preprocess.py      # 音訊前處理與格式標準化
│   └─ vad.py                   # VAD偵測與語段合併
│
├─ scripts/
│   └─ test_pipeline.py         # CLI測試入口
│
├─ utils/
│   ├─ audio_utils.py
│   ├─ file_utils.py
│   └─ text_utils.py
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

### TXT 逐字稿

<p align="center">
  <img src="assets/sample_txt.png">
</p>

### JSON 結構

<p align="center">
  <img src="assets/sample_json.png">
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

## 目前限制 (V1)

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

