"""Tests for src/rag/generation.py. VLLMGenerator is exercised against a
local fake HTTP server speaking the vLLM/OpenAI chat-completions wire
format (non-streaming JSON and SSE). This verifies OUR request building
and response/stream parsing -- it does NOT verify a real vLLM server.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.rag.generation import (
    GenerationError,
    RetrievalOnlyGenerator,
    VLLMGenerator,
    as_ragas_generate_fn,
    build_messages,
    get_generator,
)

SOURCES = [
    {"article_number": 147, "citation": "المادة 147", "text": "العقد شريعة المتعاقدين"},
    {"article_number": 148, "citation": "المادة 148", "text": "يجب تنفيذ العقد"},
]
SEEN = {}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN["body"] = body
        SEEN["auth"] = self.headers.get("Authorization")
        if body["stream"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in ["العقد ", "شريعة ", "المتعاقدين (المادة 147)"]:
                chunk = {"choices": [{"delta": {"content": piece}}]}
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.write(b"data: {\"choices\": [{\"delta\": {}}]}\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            out = {"choices": [{"message": {"content": "answer (المادة 147)"}}]}
            data = json.dumps(out, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


def _server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


def test_build_messages_labels_articles_by_number():
    msgs = build_messages("سؤال", SOURCES)
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert "[Article 147]" in msgs[1]["content"] and "[Article 148]" in msgs[1]["content"]
    assert "سؤال" in msgs[1]["content"]


def test_build_messages_never_invents_article_numbers_for_unnumbered_context():
    msgs = build_messages("q", [{"article_number": None, "text": "t"}])
    assert "[Article" not in msgs[1]["content"]
    assert "[Context 1]" in msgs[1]["content"]


def test_retrieval_only_is_clearly_labelled_and_not_llm():
    g = RetrievalOnlyGenerator()
    assert g.is_llm is False
    assert "retrieval-only" in g.generate("q", SOURCES)
    assert "Article 147" in g.generate("q", SOURCES)
    assert "No relevant articles" in g.generate("q", [])
    assert len(list(g.stream("q", SOURCES))) == 1


def test_vllm_generate_parses_response_and_sends_config():
    srv, url = _server()
    try:
        g = VLLMGenerator("m1", base_url=url, api_key="k", max_tokens=77, temperature=0.0)
        assert g.generate("q", SOURCES) == "answer (المادة 147)"
        assert SEEN["body"]["model"] == "m1" and SEEN["body"]["max_tokens"] == 77
        assert SEEN["body"]["stream"] is False
        assert SEEN["auth"] == "Bearer k"
    finally:
        srv.shutdown()


def test_vllm_stream_yields_real_incremental_deltas():
    srv, url = _server()
    try:
        pieces = list(VLLMGenerator("m1", base_url=url).stream("q", SOURCES))
        assert len(pieces) == 3
        assert "".join(pieces) == "العقد شريعة المتعاقدين (المادة 147)"
        assert SEEN["body"]["stream"] is True
    finally:
        srv.shutdown()


def test_vllm_without_sources_does_not_call_server():
    g = VLLMGenerator("m1", base_url="http://127.0.0.1:1/v1")  # unreachable on purpose
    assert "No relevant articles" in g.generate("q", [])
    assert "No relevant articles" in "".join(g.stream("q", []))


def test_vllm_unreachable_raises_generation_error():
    g = VLLMGenerator("m1", base_url="http://127.0.0.1:1/v1", timeout_s=2)
    try:
        g.generate("q", SOURCES)
    except GenerationError:
        return
    raise AssertionError("expected GenerationError")


def test_get_generator_env_selection():
    import os

    saved = {k: os.environ.get(k) for k in ("GENERATION_MODEL", "VLLM_BASE_URL")}
    try:
        os.environ.pop("GENERATION_MODEL", None)
        assert isinstance(get_generator(), RetrievalOnlyGenerator)
        os.environ["GENERATION_MODEL"] = "some/model"
        os.environ["VLLM_BASE_URL"] = "http://x:9/v1/"
        g = get_generator()
        assert isinstance(g, VLLMGenerator) and g.model == "some/model"
        assert g.base_url == "http://x:9/v1"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_ragas_adapter_matches_generate_fn_shape():
    fn = as_ragas_generate_fn(RetrievalOnlyGenerator())
    assert isinstance(fn("q", ["ctx a", "ctx b"]), str)
