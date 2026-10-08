"""In-memory stand-ins for packages that are NOT installed in the
development sandbox: sentence_transformers, chromadb, mlflow.

They implement only the API surface this project calls, and (for chromadb
and mlflow) enforce a few of the real library's strictness rules so our
wiring is exercised. Passing tests against these stand-ins verifies OUR
code's use of the API -- it does NOT verify the real packages, which stay
UNVERIFIED until run in an environment that has them.
"""
from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np


@contextlib.contextmanager
def installed(*modules):
    """Temporarily put modules in sys.modules (restoring afterwards, and
    dropping project modules that were imported against the fakes).
    """
    saved = {m.__name__: sys.modules.get(m.__name__) for m in modules}
    for m in modules:
        sys.modules[m.__name__] = m
    try:
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        sys.modules.pop("src.experiments.mlflow_model", None)


# --------------------------- sentence_transformers ------------------------


def fake_sentence_transformers(fail_with=None):
    mod = types.ModuleType("sentence_transformers")
    instances = []

    class SentenceTransformer:
        def __init__(self, name, **kwargs):
            if fail_with is not None:
                raise fail_with
            self.name, self.kwargs, self.calls = name, kwargs, []
            instances.append(self)

        def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=False):
            self.calls.append({"texts": list(texts), "normalize": normalize_embeddings})
            # deterministic, text-dependent vectors (not meaningful semantics)
            return np.array(
                [[sum(map(ord, t)) % 97 + 1.0, len(t) % 13 + 1.0, 1.0] for t in texts],
                dtype="float32",
            )

    mod.SentenceTransformer = SentenceTransformer
    mod.instances = instances
    return mod


# ------------------------------- chromadb ---------------------------------


def fake_chromadb():
    mod = types.ModuleType("chromadb")
    persisted = {}  # path -> {collection_name: _Collection} (simulates on-disk persistence)

    class _Collection:
        def __init__(self, name, metadata):
            self.name, self.metadata, self.rows = name, metadata, {}

        def upsert(self, ids, embeddings, documents, metadatas):
            for i, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
                for key, value in meta.items():  # real Chroma rejects None / non-scalars
                    if not isinstance(value, (str, int, float, bool)):
                        raise ValueError(f"Expected metadata value to be a scalar, got {value!r} for {key}")
                self.rows[i] = (np.asarray(emb, dtype="float32"), doc, dict(meta))

        def count(self):
            return len(self.rows)

        def query(self, query_embeddings, n_results):
            if n_results > len(self.rows):
                raise RuntimeError("n_results greater than collection size")
            q = np.asarray(query_embeddings[0], dtype="float32")
            scored = []
            for emb, _doc, meta in self.rows.values():
                cos = float(emb @ q / ((np.linalg.norm(emb) * np.linalg.norm(q)) or 1.0))
                scored.append((1.0 - cos, meta))  # cosine DISTANCE, ascending
            scored.sort(key=lambda x: x[0])
            top = scored[:n_results]
            return {"metadatas": [[m for _, m in top]], "distances": [[d for d, _ in top]]}

    class PersistentClient:
        def __init__(self, path):
            self._cols = persisted.setdefault(str(path), {})

        def delete_collection(self, name):
            if name not in self._cols:
                raise ValueError(f"Collection {name} does not exist")
            del self._cols[name]

        def create_collection(self, name, metadata=None):
            if name in self._cols:
                raise ValueError(f"Collection {name} already exists")
            self._cols[name] = _Collection(name, metadata or {})
            return self._cols[name]

        def get_collection(self, name):
            return self._cols[name]

    mod.PersistentClient = PersistentClient
    mod.persisted = persisted
    return mod


# -------------------------------- mlflow ----------------------------------


def fake_mlflow():
    """Returns (mlflow_module, pyfunc_module, record). `record` collects
    everything logged. Artifact/model paths must exist when logged (as with
    real MLflow).
    """
    record = SimpleNamespace(
        experiment=None, runs=[], params={}, metrics={}, tags={}, artifacts=[],
        pyfunc_logged=[], registered=[], aliases=[],
    )
    mod = types.ModuleType("mlflow")
    pyfunc = types.ModuleType("mlflow.pyfunc")
    tracking = types.ModuleType("mlflow.tracking")
    mod.pyfunc, mod.tracking = pyfunc, tracking

    class _Run:
        def __init__(self, name):
            self.info = SimpleNamespace(run_id=f"fake-run-{len(record.runs) + 1}")
            self.name = name

        def __enter__(self):
            record.runs.append(self.name)
            return self

        def __exit__(self, *exc):
            return False

    mod.set_experiment = lambda name: setattr(record, "experiment", name)
    mod.start_run = lambda run_name=None: _Run(run_name)
    mod.log_param = lambda k, v: record.params.__setitem__(k, v)
    mod.log_params = lambda d: record.params.update(d)
    mod.log_metric = lambda k, v: record.metrics.__setitem__(k, v)
    mod.set_tags = lambda d: record.tags.update(d)

    def log_artifact(path):
        assert Path(path).exists(), f"artifact missing at log time: {path}"
        record.artifacts.append(Path(path).name)

    mod.log_artifact = log_artifact

    class PythonModel:  # base class only
        pass

    def log_model(**kwargs):
        for p in kwargs.get("artifacts", {}).values():
            assert Path(p).exists(), f"model artifact missing at log time: {p}"
        record.pyfunc_logged.append(kwargs)

    pyfunc.PythonModel, pyfunc.log_model = PythonModel, log_model

    def register_model(uri, name):
        record.registered.append((uri, name))
        return SimpleNamespace(version=str(len(record.registered)))

    mod.register_model = register_model

    class MlflowClient:
        def set_registered_model_alias(self, name, alias, version):
            record.aliases.append((name, alias, version))

    tracking.MlflowClient = MlflowClient
    return mod, pyfunc, record
