"""Wrappers for calling external language model APIs.

The code in this module isolates all interactions with remote providers.
Supported providers include OpenAI’s ChatGPT and any future providers that
expose a simple HTTP endpoint.  API keys must be supplied via environment
variables and are **never** committed to the repository.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
import re


class APILanguageModel:
    """Abstract interface for API‑accessed models."""
    def __init__(self) -> None:
        # Will be populated by concrete implementations when available.
        # Expected keys (when available): prompt_tokens, completion_tokens, total_tokens
        self.last_usage: Dict[str, int] = {}
        # Accumulated numeric usage across all calls (e.g. total_tokens)
        self.accumulated_usage: Dict[str, float] = {}

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> str:
        raise NotImplementedError

    def get_last_usage(self) -> Dict[str, int]:
        """Return the last recorded token usage for this model, or empty dict."""
        return getattr(self, "last_usage", {})

    def accumulate_usage(self, usage: Optional[Dict[str, Any]]):
        """Accumulate numeric fields from a usage dict into `accumulated_usage`.

        Only numeric fields (int/float/strings representing digits) are summed.
        """
        if not usage:
            return
        for k, v in usage.items():
            # ignore None or non-numeric
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if k == "input_tokens" or k == "prompt_tokens":
                self.accumulated_usage["input_tokens"] = self.accumulated_usage.get("input_tokens", 0.0) + fv
            if k == "output_tokens" or k == "completion_tokens":
                self.accumulated_usage["output_tokens"] = self.accumulated_usage.get("output_tokens", 0.0) + fv
                
            # self.accumulated_usage[k] = self.accumulated_usage.get(k, 0.0) + fv
            

    def get_accumulated_usage(self) -> Dict[str, float]:
        """Return a copy of accumulated usage."""
        return dict(self.accumulated_usage)


class OpenAIModel(APILanguageModel):
    """Wrapper around OpenAI’s ChatCompletion API."""

    def __init__(self, model_name: str = "gpt-5", api_key: Optional[str] = None) -> None:
        super().__init__()
        import openai  # type: ignore
        openai.base_url = "https://n1n.ai/v1/"
        self.model_name = model_name
        # Load API key from argument or environment
        key = api_key or os.getenv("OPENAI_API_KEY", "sk-qYNTLF2ymRyxKerLPNYSSQum3fXjHwDcM9rGtTwLXn6HFVoX")
        if not key:
            raise ValueError(
                "OpenAI API key not provided. Set the OPENAI_API_KEY environment variable or pass api_key."
            )
        openai.api_key = key
        self.openai = openai

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, **kwargs: Any) -> str:
        # Use chat completion endpoint with a single system/user message
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        response = self.openai.responses.create(
            model=self.model_name,
            instructions="You are a helpful assistant.",
            input = prompt,
            # messages=messages,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        # Record usage if provided by the API
        usage = response.usage
        if usage:
            # standard fields: prompt_tokens, completion_tokens, total_tokens
            try:
                self.last_usage = {
                    k: int(v) for k, v in usage.items() if isinstance(v, (int, str)) and str(v).isdigit()
                }
            except Exception:
                # best-effort: store raw usage dict
                self.last_usage = dict(usage)
            try:
                self.accumulate_usage(self.last_usage)
            except Exception:
                pass
        return response.output_text


class AnthropicModel(APILanguageModel):
    """Placeholder for Anthropic’s Claude API.

    Anthropic’s API uses a different endpoint and request format.  You must set
    the `ANTHROPIC_API_KEY` environment variable.  See
    https://docs.anthropic.com/claude/reference/ for details.
    """

    def __init__(self, model_name: str = "claude-3-opus-20240229", api_key: Optional[str] = None) -> None:
        super().__init__()
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key not provided. Set the ANTHROPIC_API_KEY environment variable or pass api_key."
            )
        self.model_name = model_name
        self.api_key = key
        self.base_url = "https://api.anthropic.com/v1/messages"

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, **kwargs: Any) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        data = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        resp = requests.post(self.base_url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        # Try to record usage information if present
        usage = result.get("usage") or result.get("token_usage") or {}
        if usage:
            try:
                self.last_usage = {k: int(v) for k, v in usage.items() if isinstance(v, (int, str)) and str(v).isdigit()}
            except Exception:
                self.last_usage = dict(usage)
            try:
                self.accumulate_usage(self.last_usage)
            except Exception:
                pass
        return result["content"][0]["text"].strip()

#gemini-3-pro-preview-11-2025
class GeminiModel(APILanguageModel):
    """Placeholder for Google’s Gemini API.

    Gemini’s public API requires an endpoint and authentication token.  Users
    should implement the specifics according to Google’s documentation.  Until
    implemented, this class raises NotImplementedError.
    """

    """Wrapper around OpenAI’s ChatCompletion API."""
    def __init__(self, model_name: str = "gemini-3-pro-preview-11-2025", api_key: Optional[str] = None) -> None:
        super().__init__()
        import openai  # type: ignore
        openai.api_base = "https://n1n.ai/v1"
        self.model_name = model_name
        # Load API key from argument or environment
        key = api_key or os.getenv("OPENAI_API_KEY", "sk-qYNTLF2ymRyxKerLPNYSSQum3fXjHwDcM9rGtTwLXn6HFVoX")
        if not key:
            raise ValueError(
                "OpenAI API key not provided. Set the OPENAI_API_KEY environment variable or pass api_key."
            )
        openai.api_key = key
        self.openai = openai

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, **kwargs: Any) -> str:
        # Use chat completion endpoint with a single system/user message
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        response = self.openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Record usage if provided by the API
        usage = response.get("usage", {})
        if usage:
            # standard fields: prompt_tokens, completion_tokens, total_tokens
            try:
                self.last_usage = {
                    k: int(v) for k, v in usage.items() if isinstance(v, (int, str)) and str(v).isdigit()
                }
            except Exception:
                # best-effort: store raw usage dict
                self.last_usage = dict(usage)
            try:
                self.accumulate_usage(self.last_usage)
            except Exception:
                pass
        return response["choices"][0]["message"]["content"].strip()

# Add DeepseekSiliconflowModel after GeminiModel

class DeepseekSiliconflowModel(APILanguageModel):
    """Generic wrapper for Deepseek/SiliconFlow or similar custom APIs."""

    def __init__(
        self,
        provider: str = "deepseek-ai",
        model_name: str = "DeepSeek-V3",
        base_url: str = "https://api.siliconflow.cn/v1/chat/completions",
        api_key: Optional[str] = None,
        api_key_header: str = "Authorization",
        api_key_prefix: str = "Bearer ",
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_key_header = api_key_header
        self.api_key_prefix = api_key_prefix
        self.extra_headers = extra_headers or {}
        if not self.api_key:
            raise ValueError("API key not provided. Set DEEPSEEK_API_KEY or pass api_key.")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, **kwargs: Any) -> str:
        headers = {
            self.api_key_header: f"{self.api_key_prefix}{self.api_key}",
            **self.extra_headers,
        }
        # 组合 provider 和 model_name
        model_full_name = f"{self.provider}/{self.model_name}"
        print(f"Sending Request to {self.base_url} with model {model_full_name}")
        data = {
            "model": model_full_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            resp.raise_for_status()
        except requests.HTTPError as e:
            print(f"[Deepseek API Error] {e}\nStatus code: {getattr(e.response, 'status_code', None)}\nResponse text: {getattr(e.response, 'text', None)}")
            raise
        result = resp.json()
        # Record usage information if available
        usage = None
        if isinstance(result, dict):
            usage = result.get("usage") or result.get("token_usage")
            # some services nest usage under choices[0].usage
            if not usage and "choices" in result and result.get("choices"):
                usage = result["choices"][0].get("usage")
        if usage:
            try:
                self.last_usage = {k: int(v) for k, v in usage.items() if isinstance(v, (int, str)) and str(v).isdigit()}
            except Exception:
                self.last_usage = dict(usage)
            try:
                self.accumulate_usage(self.last_usage)
            except Exception:
                pass
        # Try to extract the text from common response formats
        if "choices" in result and result["choices"]:
            return result["choices"][0].get("text") or result["choices"][0].get("message", {}).get("content", "").strip()
        if "result" in result:
            return result["result"].strip()
        if "text" in result:
            return result["text"].strip()
        raise RuntimeError(f"Unknown response format: {result}")


def get_api_model(identifier: str, **kwargs: Any) -> APILanguageModel:
    """Factory to instantiate an API model by identifier.

    Parameters
    ----------
    identifier: str
        A string such as ``openai`` or ``anthropic``.  The identifier may also
        include a colon and model name (e.g. ``openai:gpt-4``).  If no model
        name is provided, a sensible default is used.
    kwargs: Any
        Additional keyword arguments passed to the constructor.

    Returns
    -------
    APILanguageModel
    """
    # Split identifier by first ":" or "/" (e.g. "openai:gpt-4" or "openai/gpt-4")
    match = re.split(r"[:/]", identifier, maxsplit=1)
    provider = match[0]
    model = match[1:]  # model is a list with 0 or 1 element
    model_name = model[0] if model else None
    print(f"Loading API model '{identifier}' with name '{model_name}' and provider '{provider}'")
    # provider = provider.lower()
    if provider in {"openai", "chatgpt"}:
        return OpenAIModel(model_name=model_name or "gpt-3.5-turbo", **kwargs)
    if provider in {"anthropic", "claude"}:
        return AnthropicModel(model_name=model_name or "claude-3-opus-20240229", **kwargs)
    if provider in {"gemini", "google"}:
        return GeminiModel(model_name=model_name or "gemini-pro", **kwargs)
    if provider in {"deepseek", "siliconflow", "custom", "deepseek-ai", "Qwen"}:
        return DeepseekSiliconflowModel(provider=provider, model_name=model_name, **kwargs)  # Use identifier for siliconflow
    raise ValueError(f"Unknown API model provider '{provider}'.")