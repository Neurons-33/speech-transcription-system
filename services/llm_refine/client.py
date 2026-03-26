from __future__ import annotations

import time
from typing import Dict, Optional

from services.llm_refine.prompt_builder import build_refine_prompt
from services.llm_refine.validator import (
    parse_and_validate_refined_output,
    ValidationConfig,
)
from services.llm_refine.windowing import WindowChunk


class LLMRefineClient:
    """
    負責：
    - 建 prompt
    - 呼叫 LLM
    - 解析 + 驗證
    - retry 控制
    - fallback 決策
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        max_retries: int = 2,
        retry_delay: float = 1.0,
        validation_config: Optional[ValidationConfig] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.validation_config = validation_config or ValidationConfig()
        self._client = None

    def _get_client(self):
        """
        延遲建立真實 LLM client。
        這樣 unit test 就不會因為缺少 google-genai 而在 import 階段爆掉。
        """
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise ImportError(
                    "google-genai is not installed or unavailable. "
                    "Please install the correct package before using Gemini client."
                ) from e

            self._client = genai.Client(api_key=self.api_key)

        return self._client

    def refine_window(
        self,
        window: WindowChunk,
        mode: str = "minimal",
    ) -> Dict[int, str]:
        """
        對單一 window 做 LLM 修正。

        成功：
            回傳 safe refined map

        失敗：
            fallback → 回傳原文
        """
        prompt = build_refine_prompt(window, mode=mode)

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                raw_text = self._call_llm(prompt)

                refined_map = parse_and_validate_refined_output(
                    raw_text=raw_text,
                    original_local_map=window.local_text_map,
                    config=self.validation_config,
                )

                changed_count = sum(
                    1 for k, v in refined_map.items()
                    if v != window.local_text_map[k]
                )
                print(f"[LLM_REFINE] changed {changed_count}/{len(refined_map)} items in this window")

                return refined_map

            except Exception as e:
                last_error = e

                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    break

        return dict(window.local_text_map)

    def _call_llm(self, prompt: Dict[str, str]) -> str:
        """
        呼叫 Gemini。
        """
        client = self._get_client()

        full_prompt = (
            f"{prompt['system']}\n\n"
            f"{prompt['user']}"
        )

        response = client.models.generate_content(
            model=self.model,
            contents=full_prompt,
        )

        text = getattr(response, "text", None)

        if not text:
            raise ValueError("Empty response from LLM.")

        return text.strip()