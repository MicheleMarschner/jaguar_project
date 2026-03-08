'''
how to use:
import wandb
from jaguar.utils.wandb_metrics import dedup_metrics_payload, validate_metric_keys

payload = dedup_metrics_payload(
    n_images_total=12345,
    n_duplicate_images=2345,
    n_kept_images=10000,
    n_burst_groups=890,
    duplicate_rate=2345 / 12345,
    threshold_phash=10,
    cross_identity_collision_rate=0.0032,
    runtime_sec=41.8,
)

warnings = validate_metric_keys(payload)
if warnings:
    print("[W&B metric warnings]")
    for w in warnings:
        print(" -", w)

wandb.log(payload)

'''


from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any


# ----------------------------
# Namespace constants
# ----------------------------
@dataclass(frozen=True)
class WBNS:
    DEDUP: str = "dedup"
    SPLIT: str = "split"
    TRAIN: str = "train"
    VAL: str = "val"
    RETRIEVAL: str = "retrieval"
    EVAL: str = "eval"
    EMBEDDINGS: str = "embeddings"
    XAI: str = "xai"
    QUANTUS: str = "quantus"
    EDA: str = "eda"


NS = WBNS()


# ----------------------------
# Key builder
# ----------------------------
def wb_key(namespace: str, name: str) -> str:
    """
    Build a W&B metric key like 'dedup/duplicate_rate'.
    """
    namespace = namespace.strip().lower().strip("/")
    name = name.strip().strip("/")
    return f"{namespace}/{name}"


def prefix_metrics(namespace: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """
    Convert {'a': 1, 'b': 2} -> {'namespace/a': 1, 'namespace/b': 2}
    """
    return {wb_key(namespace, k): v for k, v in metrics.items()}


# ----------------------------
# Canonical metric names by namespace (optional but useful)
# ----------------------------
CANONICAL = {
    NS.DEDUP: {
        "n_images_total",
        "n_duplicate_images",
        "n_kept_images",
        "n_burst_groups",
        "duplicate_rate",
        "threshold_phash",
        "cross_identity_collision_rate",
        "avg_burst_size",
        "max_burst_size",
    },
    NS.SPLIT: {
        "n_images_total",
        "n_identities_total",
        "train_images",
        "val_images",
        "train_identities",
        "val_identities",
    },
    NS.TRAIN: {
        "loss",
        "lr",
        "epoch",
    },
    NS.VAL: {
        "loss",
        "cmc1",
        "cmc5",
        "cmc10",
        "map",
        "map_at_r",
    },
    NS.RETRIEVAL: {
        "cmc1",
        "cmc5",
        "cmc10",
        "map",
        "map_at_r",
        "median_rank",
        "mean_rank",
        "n_queries",
        "n_gallery",
    },
    NS.EVAL: {
        "runtime_sec",
    },
    NS.EMBEDDINGS: {
        "n_samples",
        "dim",
        "extract_time_sec",
        "throughput_img_s",
    },
    NS.XAI: {
        "map_count",
        "drift_cosine_mean",
        "drift_cosine_std",
        "iou_top5_mean",
        "iou_top5_std",
    },
    NS.QUANTUS: {
        # flexible: quantus/<metric>_mean, quantus/<metric>_std
        "runtime_sec",
    },
    NS.EDA: {
        "n_images",
        "n_identities",
        "median_images_per_identity",
        "duplicate_rate",
        "runtime_sec",
    },
    NS.SYS: {
        "gpu_mem_alloc_mb",
        "gpu_mem_reserved_mb",
        "cpu_percent",
    },
}


def validate_metric_keys(metrics: Mapping[str, Any], *, strict: bool = False) -> list[str]:
    """
    Validate keys like 'dedup/duplicate_rate' against canonical namespaces.
    Returns a list of warnings. If strict=True, raises ValueError on warnings.
    """
    warnings: list[str] = []

    for full_key in metrics.keys():
        if "/" not in full_key:
            warnings.append(f"Metric key '{full_key}' has no namespace (expected 'ns/name').")
            continue

        ns, name = full_key.split("/", 1)
        ns = ns.strip()
        name = name.strip()

        if ns not in CANONICAL:
            warnings.append(f"Unknown namespace '{ns}' in key '{full_key}'.")
            continue

        # Quantus metrics are intentionally flexible
        if ns == NS.QUANTUS:
            continue

        if name not in CANONICAL[ns]:
            warnings.append(
                f"Non-canonical metric name '{name}' in namespace '{ns}' (key '{full_key}')."
            )

    if strict and warnings:
        raise ValueError("Metric key validation failed:\n- " + "\n- ".join(warnings))

    return warnings


# ----------------------------
# Convenience logging payload builders
# ----------------------------
def dedup_metrics_payload(
    *,
    n_images_total: int,
    n_duplicate_images: int,
    n_kept_images: int,
    n_burst_groups: int,
    duplicate_rate: float,
    threshold_phash: int | None = None,
    threshold_dhash: int | None = None,
    cross_identity_collision_rate: float | None = None,
    runtime_sec: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        wb_key(NS.DEDUP, "n_images_total"): n_images_total,
        wb_key(NS.DEDUP, "n_duplicate_images"): n_duplicate_images,
        wb_key(NS.DEDUP, "n_kept_images"): n_kept_images,
        wb_key(NS.DEDUP, "n_burst_groups"): n_burst_groups,
        wb_key(NS.DEDUP, "duplicate_rate"): duplicate_rate,
    }
    if threshold_phash is not None:
        payload[wb_key(NS.DEDUP, "threshold_phash")] = threshold_phash
    if threshold_dhash is not None:
        payload[wb_key(NS.DEDUP, "threshold_dhash")] = threshold_dhash
    if cross_identity_collision_rate is not None:
        payload[wb_key(NS.DEDUP, "cross_identity_collision_rate")] = cross_identity_collision_rate
    if runtime_sec is not None:
        payload[wb_key(NS.DEDUP, "runtime_sec")] = runtime_sec

    for k, v in extra.items():
        payload[wb_key(NS.DEDUP, k)] = v
    return payload


def retrieval_metrics_payload(
    *,
    cmc1: float,
    cmc5: float | None = None,
    cmc10: float | None = None,
    map_: float | None = None,
    map_at_r: float | None = None,
    n_queries: int | None = None,
    n_gallery: int | None = None,
    runtime_sec: float | None = None,
    prefix: str = NS.RETRIEVAL,
) -> dict[str, Any]:
    payload = {
        wb_key(prefix, "cmc1"): cmc1,
    }
    if cmc5 is not None:
        payload[wb_key(prefix, "cmc5")] = cmc5
    if cmc10 is not None:
        payload[wb_key(prefix, "cmc10")] = cmc10
    if map_ is not None:
        payload[wb_key(prefix, "map")] = map_
    if map_at_r is not None:
        payload[wb_key(prefix, "map_at_r")] = map_at_r
    if n_queries is not None:
        payload[wb_key(prefix, "n_queries")] = n_queries
    if n_gallery is not None:
        payload[wb_key(prefix, "n_gallery")] = n_gallery
    if runtime_sec is not None:
        payload[wb_key(prefix, "runtime_sec")] = runtime_sec
    return payload


def quantus_metrics_payload(results: Mapping[str, float], *, suffix: str = "mean") -> dict[str, float]:
    """
    Example:
      {'faithfulness': 0.42, 'sparseness': 0.11}
      -> {'quantus/faithfulness_mean': 0.42, 'quantus/sparseness_mean': 0.11}
    """
    return {wb_key(NS.QUANTUS, f"{k}_{suffix}"): v for k, v in results.items()}