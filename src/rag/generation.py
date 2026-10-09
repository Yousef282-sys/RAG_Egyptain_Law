"""Generation layer for the Arabic Legal RAG pipeline.

Two interchangeable backends share one interface (`generate` / `stream`):

* `VLLMGenerator` -- talks to a vLLM server through its OpenAI-compatible
  HTTP API (`POST {VLLM_BASE_URL}/chat/completions`, optionally with
  `stream: true` for server-sent events). Uses only the standard library
  (urllib), so it adds no dependency to the API image. Start a server with
  e.g. `vllm serve <model> --port 8001` (see docker-compose.yml).
* `RetrievalOnlyGenerator` -- used when no model is configured. It does NOT
  call a model; its output is explicitly labelled as retrieval-only so it
  is never mistaken for an LLM answer.

Configuration (environment variables):
  GENERATION_MODEL        model name served by vLLM. If unset -> retrieval-only.
  VLLM_BASE_URL           default http://localhost:8001/v1
  VLLM_API_KEY            optional bearer token (vLLM `--api-key`)
  GENERATION_MAX_TOKENS   default 512
  GENERATION_TEMPERATURE  default 0.0
  GENERATION_TIMEOUT_S    default 60

NOTHING here downloads or loads a model. Runtime behaviour against a real
vLLM server is UNVERIFIED in the development sandbox (no vLLM/GPU/network);
the request/stream parsing is unit-tested against a fake local HTTP server
that speaks the same wire format (tests/test_generation.py).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Protocol

SYSTEM_PROMPT = (
    "You are a legal assistant for the Egyptian Civil Code. Answer ONLY from "
    "the provided articles. Cite every claim with its article number in the "
    "form (Article N) / (المادة N). If the articles do not contain the "
    "answer, say that the provided articles do not answer the question. "
    "Answer in the same language as the question."
)

_NO_CONTEXT_ANSWER = "No relevant articles were found for this question."


class GenerationError(RuntimeError):
    """Raised when the generation backend is unreachable or returns an error."""


def build_messages(question: str, sources: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Chat messages with retrieved articles labelled by ARTICLE NUMBER so
    the model can cite them. `sources` items need `article_number`, `text`.
    """
    blocks = []
    for i, s in enumerate(sources, 1):
        num = s.get("article_number")
        label = f"Article {num}" if num is not None else f"Context {i}"
        blocks.append(f"[{label}]\n{s['text']}")
    user = "Articles:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class Generator(Protocol):
    name: str
    is_llm: bool

    def generate(self, question: str, sources: List[Dict[str, Any]]) -> str: ...

    def stream(self, question: str, sources: List[Dict[str, Any]]) -> Iterator[str]: ...


class RetrievalOnlyGenerator:
    """No model configured. Output is labelled, never passed off as LLM text."""

    name = "retrieval-only"
    is_llm = False

    def generate(self, question: str, sources: List[Dict[str, Any]]) -> str:
        if not sources:
            return _NO_CONTEXT_ANSWER
        nums = ", ".join(str(s["article_number"]) for s in sources)
        return (
            "[retrieval-only, no generation model configured] "
            f"Most relevant provision: Article {sources[0]['article_number']}. "
            f"Retrieved articles: {nums}. See sources for the article text."
        )

    def stream(self, question: str, sources: List[Dict[str, Any]]) -> Iterator[str]:
        # Not model streaming: a single complete (labelled) message.
        yield self.generate(question, sources)


class VLLMGenerator:
    """Client for a vLLM OpenAI-compatible server."""

    is_llm = True

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8001/v1",
        api_key: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
    ):
        if not model:
            raise ValueError("model must be a non-empty model name")
        self.model = model
        self.name = f"vllm:{model}"
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s

    def _request(self, question: str, sources: List[Dict[str, Any]], stream: bool):
        body = {
            "model": self.model,
            "messages": build_messages(question, sources),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout_s)
        except urllib.error.HTTPError as exc:
            raise GenerationError(f"vLLM server returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise GenerationError(f"vLLM server unreachable at {self.base_url}: {exc}") from exc

    def generate(self, question: str, sources: List[Dict[str, Any]]) -> str:
        if not sources:
            return _NO_CONTEXT_ANSWER
        with self._request(question, sources, stream=False) as resp:
            try:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise GenerationError(f"malformed vLLM response: {exc}") from exc

    def stream(self, question: str, sources: List[Dict[str, Any]]) -> Iterator[str]:
        """Yield real incremental deltas as vLLM emits them (SSE)."""
        if not sources:
            yield _NO_CONTEXT_ANSWER
            return
        with self._request(question, sources, stream=True) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    return
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise GenerationError(f"malformed vLLM stream chunk: {exc}") from exc
                piece = delta.get("content")
                if piece:
                    yield piece



class GroqGenerator:
    """Generator using the Groq chat-completions API."""

    is_llm = True

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
    ):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for the Groq backend")
        if not model:
            raise ValueError("GROQ_MODEL must be a non-empty model name")

        from groq import Groq

        self.model = model
        self.name = f"groq:{model}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = Groq(api_key=api_key, timeout=timeout_s)

    def generate(self, question: str, sources: List[Dict[str, Any]]) -> str:
        if not sources:
            return _NO_CONTEXT_ANSWER

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=build_messages(question, sources),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            content = response.choices[0].message.content
            if not content:
                raise GenerationError("Groq returned an empty response")
            return content
        except GenerationError:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            detail = f" (HTTP {status})" if status else ""
            raise GenerationError(
                f"Groq generation request failed{detail}: {type(exc).__name__}"
            ) from exc

    def stream(self, question: str, sources: List[Dict[str, Any]]) -> Iterator[str]:
        if not sources:
            yield _NO_CONTEXT_ANSWER
            return

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=build_messages(question, sources),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )
            for chunk in response:
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content
                if piece:
                    yield piece
        except GenerationError:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            detail = f" (HTTP {status})" if status else ""
            raise GenerationError(
                f"Groq streaming request failed{detail}: {type(exc).__name__}"
            ) from exc


def get_generator() -> Generator:
    """Choose vLLM, Groq, or the labelled retrieval-only fallback."""
    # Preserve existing vLLM configuration and behaviour.
    model = os.environ.get("GENERATION_MODEL", "").strip()
    if model:
        return VLLMGenerator(
            model=model,
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1"),
            api_key=os.environ.get("VLLM_API_KEY") or None,
            max_tokens=int(os.environ.get("GENERATION_MAX_TOKENS", "512")),
            temperature=float(os.environ.get("GENERATION_TEMPERATURE", "0.0")),
            timeout_s=float(os.environ.get("GENERATION_TIMEOUT_S", "60")),
        )

    backend = os.environ.get("GENERATION_BACKEND", "retrieval-only").strip().lower()
    if backend == "groq":
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GENERATION_BACKEND=groq requires the GROQ_API_KEY environment variable"
            )
        return GroqGenerator(
            api_key=api_key,
            model=os.environ.get(
                "GROQ_MODEL", "llama-3.3-70b-versatile"
            ).strip(),
            max_tokens=int(os.environ.get("GENERATION_MAX_TOKENS", "512")),
            temperature=float(os.environ.get("GENERATION_TEMPERATURE", "0.0")),
            timeout_s=float(os.environ.get("GENERATION_TIMEOUT_S", "60")),
        )

    if backend not in ("", "retrieval-only"):
        raise ValueError(
            f"Unsupported GENERATION_BACKEND: {backend}. Use 'groq' or 'retrieval-only'."
        )
    return RetrievalOnlyGenerator()


def as_ragas_generate_fn(generator: Generator):
    """Adapt a Generator to src/experiments/ragas_eval.GenerateFn
    `(question, contexts) -> answer`. Contexts carry no article numbers
    there, so they are labelled "Context i" (never a made-up article
    number); use the API path (RagService) for real article citations.
    """

    def _fn(question: str, contexts: List[str]) -> str:
        sources = [{"article_number": None, "text": c} for c in contexts]
        return generator.generate(question, sources)

    return _fn
