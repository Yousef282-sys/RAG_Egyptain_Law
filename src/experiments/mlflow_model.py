"""MLflow pyfunc model wrapping a built RAG retrieval configuration, so the
best configuration can be registered in the MLflow Model Registry as a
loadable, reproducible artifact (index + config), not just a name.

Imports `mlflow` at module level -- only import this module when mlflow is
installed (src/experiments/tracking.register_best_configuration does).
UNVERIFIED: never executed here (mlflow is not installed in the sandbox).
"""
from __future__ import annotations

import json

import mlflow.pyfunc


class RagRetrieverModel(mlflow.pyfunc.PythonModel):
    """predict(DataFrame with a `question` column | list of questions) ->
    per question, the retrieved article numbers/citations/scores.
    """

    def load_context(self, context):
        import tempfile
        import zipfile
        from pathlib import Path

        with open(context.artifacts["config"], encoding="utf-8") as f:
            config = json.load(f)

        self._index_tmp = tempfile.TemporaryDirectory(prefix="legal_rag_model_index_")
        with zipfile.ZipFile(context.artifacts["index"], "r") as archive:
            archive.extractall(self._index_tmp.name)

        self._index_path = Path(self._index_tmp.name) / "index"
        self._top_k = config["top_k"]

    def predict(self, context, model_input, params=None):
        from src.rag.retriever import Retriever

        questions = (
            list(model_input["question"]) if hasattr(model_input, "columns") else list(model_input)
        )
        retriever = Retriever(store_dir=str(self._index_path), top_k=self._top_k)
        try:
            return [
                [
                    {
                        "article_number": r["article_number"],
                        "citation": r["citation"],
                        "score": r["score"],
                    }
                    for r in retriever.retrieve(q)
                ]
                for q in questions
            ]
        finally:
            retriever.close()
            del retriever
