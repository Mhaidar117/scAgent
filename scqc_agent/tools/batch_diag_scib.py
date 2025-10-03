# batch_diag.py
from __future__ import annotations

import os
import json
import math
import time
import dataclasses as dc
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

# Optional imports (scib-metrics is preferred)
try:
    from scib_metrics.nearest_neighbors import pynndescent
    from scib_metrics import (
        kbet,
        ilisi_knn,
        clisi_knn,
        graph_connectivity,
    )
    HAS_SCIB = True
except Exception:
    HAS_SCIB = False

from sklearn.metrics import silhouette_score

# ---- Config -----------------------------------------------------------------

@dc.dataclass
class DiagConfig:
    rep_key: str = "X_scVI"          # which embedding to evaluate on (e.g., X_scVI, X_pca)
    label_key: str = "cell_type"     # biological labels for structure preservation
    batch_key: str = "batch"         # batch labels for mixing
    n_neighbors: int = 15            # must match your Scanpy neighbors for comparability
    subsample: Optional[int] = None  # e.g., 50000; None = use all cells
    seed: int = 0
    outdir: Optional[str] = None     # where to write CSV/PNGs; if None, guess from adata.uns['run_dir']
    use_scanpy_neighbors: bool = False  # if True, read neighbors from adata.obsp/connectivities
    jobs: int = 1                    # threads for pynndescent
    # Advanced knobs
    min_cells_per_label: int = 5     # skip labels with too few cells in some metrics
    # Sanity guards
    require_rep: bool = True

# ---- Utilities ---------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def _maybe_subsample(X: np.ndarray, obs_df: pd.DataFrame, cfg: DiagConfig
                     ) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    n = X.shape[0]
    if cfg.subsample is None or cfg.subsample >= n:
        idx = np.arange(n, dtype=int)
        return X, obs_df, idx
    rng = _rng(cfg.seed)
    idx = rng.choice(n, size=cfg.subsample, replace=False)
    idx.sort()
    return X[idx], obs_df.iloc[idx].copy(), idx

def _ensure_outdir(cfg: DiagConfig, adata) -> str:
    out = cfg.outdir
    if out is None:
        out = (adata.uns.get("run_dir") or ".") + "/batch_diag"
    os.makedirs(out, exist_ok=True)
    return out

def _nn_from_scanpy(adata, cfg: DiagConfig, idx: np.ndarray):
    """Build a scib-metrics-compatible neighbor object from adata.obsp['distances'] if desired."""
    if not HAS_SCIB:
        raise RuntimeError("scib-metrics not available; cannot adapt Scanpy neighbors.")
    # Pull the precomputed Scanpy neighbors and slice to idx
    import scipy.sparse as sp
    D = adata.obsp.get("distances", None)
    if D is None or not sp.isspmatrix(D):
        raise ValueError("No adata.obsp['distances'] found. Disable use_scanpy_neighbors or compute neighbors first.")
    # Convert to knn indices for each row (take the top n_neighbors smallest distances)
    # NOTE: For large matrices this can be memory heavy; pynndescent is recommended.
    n = len(idx)
    # Slice to the subsample
    D_sub = D[idx[:, None], idx]
    # For each row, get k nearest (excluding self)
    indptr = D_sub.tocsr().indptr
    indices = D_sub.tocsr().indices
    data = D_sub.tocsr().data
    knn_idx = np.empty((n, cfg.n_neighbors), dtype=int)
    for i in range(n):
        start, end = indptr[i], indptr[i+1]
        row_idx = indices[start:end]
        row_dist = data[start:end]
        # exclude self (distance 0) and pick k smallest
        mask = row_idx != i
        row_idx = row_idx[mask]
        row_dist = row_dist[mask]
        order = np.argsort(row_dist)[:cfg.n_neighbors]
        knn_idx[i] = row_idx[order]
    # Build a dummy object with the attributes scib-metrics expects
    class _NeighborsResults:
        def __init__(self, indices): self.indices = indices
    return _NeighborsResults(knn_idx)

def _nn_with_pynndescent(X: np.ndarray, cfg: DiagConfig):
    if not HAS_SCIB:
        raise RuntimeError("scib-metrics not installed; pip install scib-metrics")
    return pynndescent(
        X,
        n_neighbors=cfg.n_neighbors,
        random_state=cfg.seed,
        n_jobs=cfg.jobs,
    )

def _nn_batch_purity(nn_indices: np.ndarray, batches: np.ndarray) -> float:
    """Simple % of neighbors that differ in batch (↑ means more mixing)."""
    n, k = nn_indices.shape
    same = (batches[nn_indices] == batches[:, None]).sum(axis=1)
    purity = 1.0 - (same / k)
    return float(np.mean(purity))

def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())

def _nn_batch_entropy(nn_indices: np.ndarray, batches: np.ndarray) -> float:
    """Average entropy of batch distribution in each cell's neighborhood (↑ better)."""
    n, k = nn_indices.shape
    H = np.zeros(n, dtype=float)
    # map batches to ints
    _, inv = np.unique(batches, return_inverse=True)
    nbatch = inv.max() + 1
    for i in range(n):
        neigh = inv[nn_indices[i]]
        counts = np.bincount(neigh, minlength=nbatch)
        p = counts / counts.sum()
        H[i] = _entropy(p)
    # normalize by log(nbatch) to [0,1]
    H /= math.log(nbatch) if nbatch > 1 else 1.0
    return float(H.mean())

# ---- Main API ----------------------------------------------------------------

def run_batch_diagnostics(adata, cfg: Optional[DiagConfig] = None) -> Dict[str, Any]:
    """
    Compute batch-mixing and structure-preservation metrics in pure Python.
    Writes results to `adata.uns['batch_diag']` and returns a dict.
    """
    t0 = time.time()
    cfg = cfg or DiagConfig()

    # --- Select representation
    if cfg.rep_key not in adata.obsm:
        if cfg.require_rep:
            raise KeyError(f"Representation `{cfg.rep_key}` not found in adata.obsm. Available: {list(adata.obsm.keys())}")
        X = adata.X if hasattr(adata, "X") else None
    else:
        X = adata.obsm[cfg.rep_key]
    if X is None:
        raise ValueError("No data matrix to evaluate.")

    obs = adata.obs[[cfg.batch_key, cfg.label_key]].copy()
    # Guard: drop NAs
    obs = obs.dropna(subset=[cfg.batch_key, cfg.label_key])
    keep = obs.index.values
    if X.shape[0] != len(adata):
        raise ValueError("Shape mismatch: X rows must match adata.n_obs.")
    # Align X to kept obs
    idx_full = np.arange(len(adata))
    idx_mask = np.isin(adata.obs_names.values, keep)
    X = X[idx_mask]
    obs = obs.loc[adata.obs_names[idx_mask]]

    # Subsample (optional, deterministic)
    Xs, obs_s, sub_idx_rel = _maybe_subsample(X, obs, cfg)
    batches = obs_s[cfg.batch_key].to_numpy()
    labels = obs_s[cfg.label_key].to_numpy()

    # --- Neighbors
    if cfg.use_scanpy_neighbors:
        nn = _nn_from_scanpy(adata[idx_mask], cfg, sub_idx_rel)
        nn_idx = nn.indices
    else:
        nn = _nn_with_pynndescent(Xs, cfg)
        nn_idx = nn.indices  # shape (n, k)

    results: Dict[str, Any] = {
        "config": dataclasses_asdict(cfg),
        "n_cells_eval": int(Xs.shape[0]),
        "n_neighbors": int(cfg.n_neighbors),
        "rep_key": cfg.rep_key,
        "batch_key": cfg.batch_key,
        "label_key": cfg.label_key,
        "has_scib": bool(HAS_SCIB),
    }

    # --- Core metrics (scib-metrics preferred)
    if HAS_SCIB:
        # kBET (returns acceptance rate, statistic, p-values)
        kb_acc, kb_stat, kb_p = kbet(nn, batches)
        results.update({
            "kbet_acceptance": float(kb_acc),
            "kbet_statistic": float(kb_stat),
            "kbet_pvalue_mean": float(np.mean(kb_p)),
        })
        # LISI
        results["ilisi"] = float(ilisi_knn(nn, batches))
        results["clisi"] = float(clisi_knn(nn, labels))
        # Graph connectivity (structure preservation)
        results["graph_connectivity"] = float(graph_connectivity(nn, labels))
    else:
        # Fallbacks (pure sklearn/NumPy; not identical to LISI/kBET but useful)
        results["asw_batch"] = float(silhouette_score(Xs, batches))
        results["nn_batch_purity"] = float(_nn_batch_purity(nn_idx, batches))
        results["nn_batch_entropy"] = float(_nn_batch_entropy(nn_idx, batches))

    # Always include quick global summaries for dashboards
    results["asw_batch"] = results.get("asw_batch", float(silhouette_score(Xs, batches)))
    results["nn_batch_purity"] = results.get("nn_batch_purity", float(_nn_batch_purity(nn_idx, batches)))
    results["nn_batch_entropy"] = results.get("nn_batch_entropy", float(_nn_batch_entropy(nn_idx, batches)))

    results["elapsed_sec"] = round(time.time() - t0, 3)

    # --- Persist
    outdir = _ensure_outdir(cfg, adata)
    # 1) JSON blob
    with open(os.path.join(outdir, f"batch_diag_{cfg.rep_key}.json"), "w") as f:
        json.dump(results, f, indent=2)
    # 2) CSV one-liner (wide format)
    pd.DataFrame([results]).to_csv(os.path.join(outdir, f"batch_diag_{cfg.rep_key}.csv"), index=False)

    # 3) Stash in AnnData
    diag_uns = adata.uns.get("batch_diag", {})
    diag_uns[cfg.rep_key] = results
    adata.uns["batch_diag"] = diag_uns

    return results

def dataclasses_asdict(cfg: DiagConfig) -> Dict[str, Any]:
    return {f.name: getattr(cfg, f.name) for f in dc.fields(cfg)}
