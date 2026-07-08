"""
kNN embedding-based similarity for evaluating bug similarity.

Updated to better capture "bug style" rather than "problem semantics" by:
  1) Embedding the patch/diff (correct -> buggy) by default.
  2) Excluding the problem statement by default (can be enabled).
  3) Supporting a NEGATIVE pool and using a RELATIVE margin score:
        margin = sim_to_target_pool - sim_to_negative_pool
  4) Using input_type="document" for direct bug-bug similarity.

The reward can be:
  - absolute: avg_topK cos(query, target_pool)
  - relative: avg_topK cos(query, target_pool) - avg_topK cos(query, negative_pool)
"""

from __future__ import annotations

import difflib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class CodeEmbeddingConfig:
    """Configuration for code embedding similarity scoring."""
    enabled: bool = False
    reward_weight: float = 0.5

    # Embedding model
    model_name: str = "voyage-code-3"  # Voyage AI model (or HuggingFace model path)
    device: str = "cuda"              # only for local models
    batch_size: int = 128             # Voyage supports up to 128

    # Text construction
    include_problem: bool = False     # IMPORTANT: default False now
    embed_mode: str = "diff"          # "diff" | "buggy"
    diff_context_lines: int = 3       # unified diff context lines
    max_chars: int = 32000            # truncate embedding input

    # Similarity scoring
    top_k: int = 5
    normalize_embeddings: bool = True

    # Relative scoring against a negative pool
    use_relative_score: bool = True
    # normalize margin -> [0,1] via sigmoid(margin_temp * margin)
    margin_temperature: float = 5.0

    # Voyage AI specific
    voyage_api_key: Optional[str] = None
    # Default used only if caller doesn't specify; for retrieval, we pass explicitly.
    input_type: str = "document"


def is_voyage_model(model_name: str) -> bool:
    return model_name.startswith("voyage-")


# -----------------------------------------------------------------------------
# Text building helpers
# -----------------------------------------------------------------------------

def _unified_diff_text(
    correct_code: str,
    buggy_code: str,
    context_lines: int = 3,
) -> str:
    """Compute a unified diff (patch) from correct -> buggy code."""
    a = (correct_code or "").splitlines(keepends=True)
    b = (buggy_code or "").splitlines(keepends=True)
    diff = difflib.unified_diff(
        a,
        b,
        fromfile="correct.py",
        tofile="buggy.py",
        n=max(0, int(context_lines)),
    )
    return "".join(diff).strip()


def _build_embedding_text(
    *,
    problem: str,
    buggy_code: str,
    correct_code: Optional[str],
    cfg: CodeEmbeddingConfig,
) -> str:
    """Build the text that we feed into the embedder."""
    buggy_code = buggy_code or ""
    correct_code = correct_code or ""

    mode = (cfg.embed_mode or "diff").lower().strip()
    if mode == "diff" and correct_code.strip():
        patch = _unified_diff_text(correct_code, buggy_code, context_lines=cfg.diff_context_lines)
        # If diff is empty (identical), fall back to buggy code
        core = patch if patch.strip() else buggy_code
        core_header = "Patch (unified diff):"
    else:
        core = buggy_code
        core_header = "Buggy code:"

    if cfg.include_problem:
        text = f"Problem:\n{problem}\n\n{core_header}\n{core}"
    else:
        text = f"{core_header}\n{core}"

    # Truncate to avoid API/model limits
    if cfg.max_chars and len(text) > cfg.max_chars:
        text = text[: cfg.max_chars]
    return text


def _sigmoid(x: float) -> float:
    # numerically safe-ish sigmoid for moderate x
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    else:
        z = np.exp(x)
        return float(z / (1.0 + z))


def build_embedding_text_for_bug(
    *,
    problem: str,
    buggy_code: str,
    correct_code: Optional[str],
    cfg: CodeEmbeddingConfig,
) -> str:
    return _build_embedding_text(
        problem=problem,
        buggy_code=buggy_code,
        correct_code=correct_code,
        cfg=cfg,
    )



# -----------------------------------------------------------------------------
# Embedders
# -----------------------------------------------------------------------------

class VoyageCodeEmbedder:
    """Embed snippets using Voyage AI's code embedding models."""

    def __init__(self, config: Optional[CodeEmbeddingConfig] = None):
        self.config = config or CodeEmbeddingConfig()
        self._client = None
        self._initialized = False

    def _lazy_init(self) -> None:
        if self._initialized:
            return

        import voyageai

        api_key = self.config.voyage_api_key or os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError(
                "Voyage API key not found. Set VOYAGE_API_KEY environment variable "
                "or pass voyage_api_key in config."
            )

        self._client = voyageai.Client(api_key=api_key)
        self._initialized = True
        print(f"Initialized Voyage AI client with model: {self.config.model_name}")

    def embed(self, texts: List[str], input_type: str) -> np.ndarray:
        """Embed a list of strings."""
        self._lazy_init()

        embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            # truncate each string
            if self.config.max_chars:
                batch = [t[: self.config.max_chars] for t in batch]

            try:
                result = self._client.embed(
                    texts=batch,
                    model=self.config.model_name,
                    input_type=input_type,
                )
                batch_embeddings = np.array(result.embeddings)
                embeddings.append(batch_embeddings)
            except Exception as e:
                print(f"Voyage API error: {e}")
                dim = 1024 if "code-3" in self.config.model_name else 1536
                embeddings.append(np.zeros((len(batch), dim), dtype=np.float32))

            if i + self.config.batch_size < len(texts):
                time.sleep(0.1)

        out = np.vstack(embeddings)
        if self.config.normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.maximum(norms, 1e-9)
        return out

    def embed_bug(
        self,
        *,
        problem: str,
        buggy_code: str,
        correct_code: Optional[str],
        input_type: str,
    ) -> np.ndarray:
        text = _build_embedding_text(
            problem=problem,
            buggy_code=buggy_code,
            correct_code=correct_code,
            cfg=self.config,
        )
        return self.embed([text], input_type=input_type)[0]


class LocalCodeEmbedder:
    """Embed snippets using a local HuggingFace model."""

    def __init__(self, config: Optional[CodeEmbeddingConfig] = None):
        self.config = config or CodeEmbeddingConfig()
        self._model = None
        self._tokenizer = None
        self._initialized = False

    def _lazy_init(self) -> None:
        if self._initialized:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading local embedding model: {self.config.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(self.config.model_name, trust_remote_code=True)

        if self.config.device == "cuda" and torch.cuda.is_available():
            self._model = self._model.cuda()

        self._model.eval()
        self._initialized = True
        print(f"  Loaded on device: {next(self._model.parameters()).device}")

    def embed(self, texts: List[str], input_type: str = "document") -> np.ndarray:
        """Embed a list of strings. input_type ignored, kept for API compatibility."""
        self._lazy_init()

        import torch

        embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            if self.config.max_chars:
                batch = [t[: self.config.max_chars] for t in batch]

            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            if self.config.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    batch_embeddings = outputs.pooler_output
                else:
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                        input_mask_expanded.sum(1), min=1e-9
                    )

                if self.config.normalize_embeddings:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def embed_bug(
        self,
        *,
        problem: str,
        buggy_code: str,
        correct_code: Optional[str],
        input_type: str,
    ) -> np.ndarray:
        text = _build_embedding_text(
            problem=problem,
            buggy_code=buggy_code,
            correct_code=correct_code,
            cfg=self.config,
        )
        return self.embed([text], input_type=input_type)[0]


class CodeEmbedder:
    """Unified embedder that selects Voyage vs local model automatically."""

    def __init__(self, config: Optional[CodeEmbeddingConfig] = None):
        self.config = config or CodeEmbeddingConfig()
        if is_voyage_model(self.config.model_name):
            self._embedder = VoyageCodeEmbedder(self.config)
        else:
            self._embedder = LocalCodeEmbedder(self.config)

    def embed(self, texts: List[str], input_type: str) -> np.ndarray:
        return self._embedder.embed(texts, input_type=input_type)

    def embed_bug(
        self,
        *,
        problem: str,
        buggy_code: str,
        correct_code: Optional[str],
        input_type: str,
    ) -> np.ndarray:
        return self._embedder.embed_bug(
            problem=problem,
            buggy_code=buggy_code,
            correct_code=correct_code,
            input_type=input_type,
        )


# -----------------------------------------------------------------------------
# Pools
# -----------------------------------------------------------------------------

@dataclass
class ReferencePool:
    embeddings: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: List[Dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.metadata)

    def add(self, embedding: np.ndarray, meta: Dict[str, Any]) -> None:
        if len(self.embeddings) == 0:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
        self.metadata.append(meta)

    def save(self, path: str) -> None:
        np.save(f"{path}_embeddings.npy", self.embeddings)
        with open(f"{path}_metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Saved pool ({len(self)}) to {path}")

    @classmethod
    def load(cls, path: str) -> "ReferencePool":
        embeddings = np.load(f"{path}_embeddings.npy")
        with open(f"{path}_metadata.json") as f:
            metadata = json.load(f)
        pool = cls(embeddings=embeddings, metadata=metadata)
        print(f"Loaded pool ({len(pool)}) from {path}")
        return pool


# -----------------------------------------------------------------------------
# kNN scorer
# -----------------------------------------------------------------------------

class KNNBugSimilarity:
    """
    kNN-based similarity scoring.

    Supports:
      - absolute score: sim_to_target_pool
      - relative score: sim_to_target_pool - sim_to_negative_pool
    """

    def __init__(
        self,
        embedder: CodeEmbedder,
        target_pool: Optional[ReferencePool] = None,
        negative_pool: Optional[ReferencePool] = None,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.target_pool = target_pool or ReferencePool()
        self.negative_pool = negative_pool  # may be None / empty
        self.top_k = top_k

    def build_pool_from_tasks(
        self,
        tasks: List[Dict[str, Any]],
        pool_name: str = "target",
    ) -> ReferencePool:
        """Build a pool from tasks containing buggy solutions (and ideally reference solutions)."""
        print(f"Building {pool_name} pool from {len(tasks)} tasks...")

        texts: List[str] = []
        metas: List[Dict[str, Any]] = []

        for i, task in enumerate(tasks):
            problem = _get_problem(task)
            bug = _get_buggy_solution(task)
            if not bug:
                continue
            correct = _get_reference_solution(task) or None

            text = _build_embedding_text(
                problem=problem,
                buggy_code=bug,
                correct_code=correct,
                cfg=self.embedder.config,
            )
            texts.append(text)

            metas.append(
                {
                    "task_id": task.get("task_id") or task.get("uid") or task.get("index") or str(i),
                    "problem_preview": (problem or "")[:200],
                    "bug_preview": (bug or "")[:200],
                    "has_reference": bool(correct and str(correct).strip()),
                }
            )

        pool = ReferencePool()
        if not texts:
            print(f"  No valid bugs found for {pool_name} pool.")
            return pool

        print(f"  Embedding {len(texts)} items for {pool_name} pool...")
        # For indexing pools, use input_type="document"
        embs = self.embedder.embed(texts, input_type="document")

        for emb, meta in zip(embs, metas):
            pool.add(emb, meta)

        print(f"  Built {pool_name} pool with {len(pool)} references")
        return pool

    def build_target_pool(self, tasks: List[Dict[str, Any]]) -> ReferencePool:
        self.target_pool = self.build_pool_from_tasks(tasks, pool_name="target")
        return self.target_pool

    def build_negative_pool(self, tasks: List[Dict[str, Any]]) -> ReferencePool:
        self.negative_pool = self.build_pool_from_tasks(tasks, pool_name="negative")
        return self.negative_pool

    def _avg_topk_cosine(self, pool: ReferencePool, query_embedding: np.ndarray, k: int) -> Tuple[float, List[int], np.ndarray]:
        sims = np.dot(pool.embeddings, query_embedding)  # cosine if normalized
        top_k_indices = np.argsort(sims)[-k:][::-1]
        top_k_sims = sims[top_k_indices]
        return float(np.mean(top_k_sims)), top_k_indices.tolist(), top_k_sims

    def score_similarity(
        self,
        problem: str,
        buggy_code: str,
        correct_code: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Returns:
          score_norm: in [0,1] (sigmoid-normalized if relative, else cosine-normalized)
          meta: includes raw cosine and margin
        """
        if len(self.target_pool) == 0:
            return 0.0, {"error": "Empty target pool"}

        k = min(int(top_k or self.top_k), len(self.target_pool))
        cfg = self.embedder.config

        # Embed the query (retrieval-style uses input_type="query")
        q = self.embedder.embed_bug(
            problem=problem,
            buggy_code=buggy_code,
            correct_code=correct_code,
            input_type="query",
        )

        sim_pos, pos_ids, pos_sims = self._avg_topk_cosine(self.target_pool, q, k)

        sim_neg = None
        neg_ids: List[int] = []
        neg_sims = None
        margin = None
        score_norm = None

        if cfg.use_relative_score and self.negative_pool is not None and len(self.negative_pool) > 0:
            k_neg = min(int(top_k or self.top_k), len(self.negative_pool))
            sim_neg, neg_ids, neg_sims = self._avg_topk_cosine(self.negative_pool, q, k_neg)
            margin = float(sim_pos - sim_neg)

            # Normalize margin to [0,1] using sigmoid(temp * margin)
            score_norm = _sigmoid(float(cfg.margin_temperature) * margin)
        else:
            # Absolute similarity normalized from cosine [-1,1] -> [0,1]
            score_norm = float((sim_pos + 1.0) / 2.0)

        meta: Dict[str, Any] = {
            "embed_mode": cfg.embed_mode,
            "include_problem": bool(cfg.include_problem),
            "top_k": int(k),
            "avg_cosine_target": float(sim_pos),
            "target_top_k_similarities": pos_sims.tolist(),
            "target_top_k_task_ids": [self.target_pool.metadata[i].get("task_id", "") for i in pos_ids],
            "score_norm": float(score_norm),
        }

        if sim_neg is not None:
            meta.update(
                {
                    "avg_cosine_negative": float(sim_neg),
                    "negative_top_k_task_ids": [self.negative_pool.metadata[i].get("task_id", "") for i in neg_ids],
                    "negative_top_k_similarities": (neg_sims.tolist() if neg_sims is not None else []),
                    "margin": float(margin),
                    "margin_temperature": float(cfg.margin_temperature),
                }
            )

        return float(score_norm), meta


# -----------------------------------------------------------------------------
# Helpers (dataset schema)
# -----------------------------------------------------------------------------

def _get_problem(task: Dict[str, Any]) -> str:
    extra_info = task.get("extra_info", {})
    if isinstance(extra_info, dict):
        for key in ("question", "instruct_prompt", "complete_prompt", "prompt", "text", "problem", "description"):
            val = extra_info.get(key)
            if isinstance(val, str) and val.strip():
                return val

    for key in ("question", "instruct_prompt", "complete_prompt", "prompt", "text", "problem", "description", "code_prompt"):
        val = task.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _get_buggy_solution(task: Dict[str, Any]) -> Optional[str]:
    extra_info = task.get("extra_info", {})
    if isinstance(extra_info, dict):
        for key in ("buggy_solution", "buggy_sampled_solution", "buggy", "buggy_code", "bug"):
            val = extra_info.get(key)
            if isinstance(val, str) and val.strip():
                return val

    for key in ("buggy_solution", "buggy_sampled_solution", "buggy", "buggy_code", "bug"):
        val = task.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _get_reference_solution(task: Dict[str, Any]) -> str:
    extra_info = task.get("extra_info", {})
    if isinstance(extra_info, dict):
        for key in ("reference_solution", "canonical_solution", "solution", "code", "correct_code"):
            val = extra_info.get(key)
            if isinstance(val, str) and val.strip():
                return val

    for key in ("reference_solution", "canonical_solution", "solution", "code", "correct_code"):
        val = task.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


# -----------------------------------------------------------------------------
# Test runner (kept similar, but updated for diff + relative scoring)
# -----------------------------------------------------------------------------

def run_similarity_comparison(
    knn: KNNBugSimilarity,
    bug1_task: Dict[str, Any],
    bug2_task: Dict[str, Any],
    bug1_label: str,
    bug2_label: str,
) -> Dict[str, Any]:
    problem1 = _get_problem(bug1_task)
    bug1 = _get_buggy_solution(bug1_task)
    corr1 = _get_reference_solution(bug1_task) or None

    problem2 = _get_problem(bug2_task)
    bug2 = _get_buggy_solution(bug2_task)
    corr2 = _get_reference_solution(bug2_task) or None

    if not bug1 or not bug2:
        return {
            "error": "Missing buggy solution",
            "bug1_present": bool(bug1),
            "bug2_present": bool(bug2),
        }

    # Score each bug against the target/negative pools (relative if enabled)
    score1, meta1 = knn.score_similarity(problem1, bug1, correct_code=corr1)
    score2, meta2 = knn.score_similarity(problem2, bug2, correct_code=corr2)

    # Direct similarity: embed both as DOCUMENT for symmetry
    emb1 = knn.embedder.embed_bug(problem=problem1, buggy_code=bug1, correct_code=corr1, input_type="document")
    emb2 = knn.embedder.embed_bug(problem=problem2, buggy_code=bug2, correct_code=corr2, input_type="document")
    direct_sim = float(np.dot(emb1, emb2))
    direct_norm = float((direct_sim + 1.0) / 2.0)

    return {
        "bug1_label": bug1_label,
        "bug2_label": bug2_label,
        "bug1_score_norm": score1,
        "bug2_score_norm": score2,
        "bug1_meta": meta1,
        "bug2_meta": meta2,
        "direct_cosine_similarity": direct_sim,
        "direct_normalized": direct_norm,
    }


# -----------------------------------------------------------------------------
# CLI main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test kNN embedding-based bug similarity (diff + relative scoring).")

    parser.add_argument("--model", type=str, default="voyage-code-3",
                        help="Embedding model: voyage-code-3, voyage-code-2, or HuggingFace model path")

    parser.add_argument("--voyage_api_key", type=str, default=None,
                        help="Voyage AI API key (defaults to VOYAGE_API_KEY env var)")

    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["bugbench", "bugbench_qwen7b_sampled", "bugbench_gpt-oss-20b_sampled"],
                        help="Datasets to compare")

    parser.add_argument("--reference_dataset", type=str, default="bugbench",
                        help="Dataset to use for building TARGET pool")

    parser.add_argument("--negative_datasets", type=str, nargs="*", default=None,
                        help="Datasets to use for building NEGATIVE pool (optional). "
                             "If omitted and >1 datasets provided, uses (datasets - {reference_dataset}).")

    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples per scenario")
    parser.add_argument("--top_k", type=int, default=5, help="Number of nearest neighbors for averaging")

    # New knobs
    parser.add_argument("--embed_mode", type=str, default="diff", choices=["diff", "buggy"],
                        help="Embed mode: diff (recommended) or buggy")
    parser.add_argument("--diff_context_lines", type=int, default=3, help="Unified diff context lines")
    parser.add_argument("--include_problem", action="store_true", default=False,
                        help="Include problem statement in embedding input (default: False)")
    parser.add_argument("--use_relative_score", action="store_true", default=True,
                        help="Use relative score vs negative pool if available (default: True)")
    parser.add_argument("--no_use_relative_score", action="store_false", dest="use_relative_score",
                        help="Disable relative score; use absolute target similarity only")

    parser.add_argument("--margin_temperature", type=float, default=5.0,
                        help="Sigmoid temperature for margin normalization to [0,1]")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for local embeddings (cuda/cpu, ignored for Voyage)")

    parser.add_argument("--save_target_pool", type=str, default=None, help="Path prefix to save target pool")
    parser.add_argument("--save_negative_pool", type=str, default=None, help="Path prefix to save negative pool")
    parser.add_argument("--load_target_pool", type=str, default=None, help="Path prefix to load target pool")
    parser.add_argument("--load_negative_pool", type=str, default=None, help="Path prefix to load negative pool")

    parser.add_argument("--save_results", type=str, default=None, help="Path to save results JSON")

    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    cfg = CodeEmbeddingConfig(
        model_name=args.model,
        include_problem=bool(args.include_problem),
        top_k=int(args.top_k),
        device=str(args.device),
        voyage_api_key=args.voyage_api_key,
        embed_mode=str(args.embed_mode),
        diff_context_lines=int(args.diff_context_lines),
        use_relative_score=bool(args.use_relative_score),
        margin_temperature=float(args.margin_temperature),
    )

    print("=" * 80)
    print("🔬 kNN EMBEDDING BUG SIMILARITY TEST (diff + relative)")
    print("=" * 80)
    print(f"Embedding Model: {args.model}")
    print(f"Reference (TARGET) Dataset: {args.reference_dataset}")
    print(f"Test Datasets: {args.datasets}")
    print(f"Split: {args.split}")
    print(f"Top-K: {args.top_k}")
    print(f"Embed Mode: {cfg.embed_mode}")
    print(f"Include Problem: {cfg.include_problem}")
    print(f"Use Relative Score: {cfg.use_relative_score}")
    print(f"Margin Temperature: {cfg.margin_temperature}")
    print(f"Samples per scenario: {args.n_samples}")

    embedder = CodeEmbedder(cfg)
    knn = KNNBugSimilarity(embedder, top_k=args.top_k)

    from rllm.data.dataset import DatasetRegistry

    # Load/build target pool
    if args.load_target_pool:
        knn.target_pool = ReferencePool.load(args.load_target_pool)
    else:
        print(f"\nLoading TARGET dataset: {args.reference_dataset}:{args.split} ...")
        ref_ds = DatasetRegistry.load_dataset(args.reference_dataset, args.split)
        if ref_ds is None:
            raise SystemExit(f"ERROR: Could not load {args.reference_dataset}:{args.split}")
        ref_data = list(ref_ds.get_data())
        ref_data_with_bugs = [t for t in ref_data if _get_buggy_solution(t)]
        print(f"  Loaded {len(ref_data)} tasks, {len(ref_data_with_bugs)} with buggy solutions")
        knn.build_target_pool(ref_data_with_bugs)
        if args.save_target_pool:
            knn.target_pool.save(args.save_target_pool)

    # Load/build negative pool (optional)
    neg_pool = None
    if args.load_negative_pool:
        neg_pool = ReferencePool.load(args.load_negative_pool)
        knn.negative_pool = neg_pool
    else:
        neg_datasets = args.negative_datasets
        if neg_datasets is None:
            # Default: everything except reference_dataset
            neg_datasets = [d for d in args.datasets if d != args.reference_dataset]

        if neg_datasets:
            neg_tasks: List[Dict[str, Any]] = []
            for ds_name in neg_datasets:
                print(f"\nLoading NEGATIVE dataset: {ds_name}:{args.split} ...")
                ds = DatasetRegistry.load_dataset(ds_name, args.split)
                if ds is None:
                    print(f"  WARNING: Could not load {ds_name}:{args.split}")
                    continue
                data = list(ds.get_data())
                data_with_bugs = [t for t in data if _get_buggy_solution(t)]
                print(f"  Loaded {len(data)} tasks, {len(data_with_bugs)} with buggy solutions")
                neg_tasks.extend(data_with_bugs)

            if neg_tasks:
                knn.build_negative_pool(neg_tasks)
                if args.save_negative_pool:
                    knn.negative_pool.save(args.save_negative_pool)
        else:
            knn.negative_pool = None

    # Load test datasets
    datasets: Dict[str, List[Dict[str, Any]]] = {}
    for ds_name in args.datasets:
        print(f"\nLoading {ds_name}:{args.split} ...")
        ds = DatasetRegistry.load_dataset(ds_name, args.split)
        if ds is None:
            print(f"  WARNING: Could not load {ds_name}:{args.split}")
            continue
        data = list(ds.get_data())
        data_with_bugs = [t for t in data if _get_buggy_solution(t)]
        print(f"  Loaded {len(data)} tasks, {len(data_with_bugs)} with buggy solutions")
        if data_with_bugs:
            datasets[ds_name] = data_with_bugs

    if not datasets:
        raise SystemExit("ERROR: No datasets loaded with buggy solutions")

    # Simple scenario runner (kept minimal)
    import random

    results: List[Dict[str, Any]] = []
    names = list(datasets.keys())

    print("\nRunning random cross-dataset comparisons...")
    for _ in range(int(args.n_samples)):
        ds1, ds2 = random.sample(names, 2) if len(names) >= 2 else (names[0], names[0])
        t1 = random.choice(datasets[ds1])
        t2 = random.choice(datasets[ds2])
        r = run_similarity_comparison(knn, t1, t2, ds1, ds2)
        results.append(r)
        if "error" not in r:
            print(
                f"  {ds1} vs {ds2}: direct={r['direct_cosine_similarity']:.3f}, "
                f"score1={r['bug1_score_norm']:.3f}, score2={r['bug2_score_norm']:.3f}"
            )

    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.save_results}")

    print("\n✅ Done!")