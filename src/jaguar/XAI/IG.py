from collections import defaultdict
from tqdm import tqdm

from jaguar.evaluation.mining import build_easy_positive_pairs, easy_positive
from jaguar.utils.utils import normalize_query_indices
from jaguar.utils.utils_explainer import SimilarityForward, ig_saliency_batched_similarity
import torch
import numpy as np
import torch.nn.functional as F
from captum.attr import IntegratedGradients


from jaguar.config import PATHS, DEVICE
from jaguar.utils.utils_datasets import load_jaguar_from_FO_export
from jaguar.models.foundation_models import FoundationModelWrapper


def load_or_extract_embeddings(model_wrapper, torch_ds, split="training"):
    """
    Returns embeddings as np.ndarray [N,D].
    Loads from disk if available; otherwise extracts and saves.
    """
    folder = PATHS.data / "embeddings"
    folder.mkdir(parents=True, exist_ok=True)

    filename = f"embeddings_{model_wrapper.name}_{split}.npy"
    path = folder / filename

    if path.exists():
        emb = np.load(path)
        print(f"[Info] Loaded embeddings from {path}, shape={emb.shape}")
        return emb

    print(f"[Info] Embeddings not found at {path}. Extracting...")
    imgs_all = [torch_ds[k]["img"] for k in range(len(torch_ds))]  # PIL images
    emb = model_wrapper.extract_embeddings(imgs_all)               # np.ndarray [N,D]
    np.save(path, emb)
    print(f"[Info] Saved embeddings to {path}")
    return emb


#! TODO: Habe ich das schon??  Normalize and build similarity matrix / ranked neighbors
def build_similarity_matrix_and_ranking(emb: np.ndarray):
    emb = emb.astype(np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)  # [N,D]
    sim = emb @ emb.T                                                  # [N,N]
    ranked = np.argsort(-sim, axis=1)                                  # descending
    return sim, ranked


if __name__ == "__main__":
    BATCH_SIZE = 32
    IG_STEPS = 10
    BATCH_SIZE_EXPLAINER = 32
    N_QUERIES = 8

    model_name = "ConvNeXt-V2"   # DINOv2-Base, MiewID, ConvNeXt-V2, MegaDescriptor-L
    dataset_name = "jaguar_init"

    fo_ds, torch_ds = load_jaguar_from_FO_export(
        PATHS.data_export / "init",
        dataset_name=dataset_name,
        processing_fn=None,
        overwrite_db=False,
    )

    print("[Info] Dataset size:", len(torch_ds))

    # Dataset metadata
    filepaths = [str(torch_ds._resolve_path(s[torch_ds.filepath_key])) for s in torch_ds.samples]
    labels = np.asarray(torch_ds.labels)  # if string labels, okay for mining if consistent
    # If needed / available:
    # labels_idx = np.asarray(torch_ds.labels_idx)

    # Load model wrapper
    model_wrapper = FoundationModelWrapper(model_name, device=DEVICE)
    print("[Info] Loaded model:", model_wrapper.name)

    # ---------------------------------------------------------
    # 1) Build embeddings for mining (or load if already saved)
    # ---------------------------------------------------------
    embeddings = load_or_extract_embeddings(model_wrapper, torch_ds, split="training")         # np.ndarray [N,D]
    sim, ranked = build_similarity_matrix_and_ranking(embeddings)

    query_indices = [2]                       # single sample sanity check
    # query_indices = [2, 7, 11, 24]         # subset
    # query_indices = list(range(len(torch_ds)))  # whole dataset

    ref_indices, pair_sims = build_easy_positive_pairs(
        query_indices=query_indices,
        sim=sim,
        ranked=ranked,
        labels=labels,
        easy_positive_fn=easy_positive,
    )

    print(f"[Info] N queries: {len(query_indices)}")
    print(f"[Info] Example pair: q={query_indices[0]} -> ref={int(ref_indices[0])}, sim={float(pair_sims[0]):.4f}")

    q = normalize_query_indices(query_indices, len(torch_ds))
    r = np.asarray(ref_indices, dtype=np.int64).reshape(-1)

    if len(q) != len(r):
        raise ValueError("query_indices and ref_indices must have same length.")

    if pair_sims is not None:
        pair_sims = np.asarray(pair_sims, dtype=np.float32).reshape(-1)
        if len(pair_sims) != len(q):
            raise ValueError("pair_sims must have same length as query_indices.")

    # group positions by reference index (so we can reuse one IG explainer per reference)
    groups = defaultdict(list)
    for pos, ref_idx in enumerate(r):
        groups[int(ref_idx)].append(pos)

    saliency_out = [None] * len(q)
    x_query_out = [None] * len(q)

    model_wrapper.eval()

    for ref_idx, positions in tqdm(groups.items(), desc="IG refs", total=len(groups)):
        # 1) reference embedding once
        ref_img = torch_ds[int(ref_idx)]["img"]  # PIL
        ref_tensor = model_wrapper.preprocess(ref_img).unsqueeze(0).to(model_wrapper.device)

        with torch.no_grad():
            ref_emb = model_wrapper.get_embeddings_tensor(ref_tensor).squeeze(0)  # [D]

        # 2) similarity forward + IG for THIS reference
        sim_model = SimilarityForward(model_wrapper, ref_emb, maximize=True).to(model_wrapper.device).eval()
        ig = IntegratedGradients(sim_model)

        # 3) gather query tensors for this ref-group (in original order)
        q_group = q[np.asarray(positions, dtype=np.int64)]

        xs = []
        for qi in q_group:
            img = torch_ds[int(qi)]["img"]          # PIL
            x = model_wrapper.preprocess(img)       # [3,H,W]
            xs.append(x)
        X_group = torch.stack(xs, dim=0)            # [B,3,H,W] CPU

        # 4) YOUR batched IG stays here
        sal_group = ig_saliency_batched_similarity(
            X_group,
            explainer=ig,
            device=model_wrapper.device,
            steps=IG_STEPS,
            internal_bs=BATCH_SIZE_EXPLAINER,
            batch_size=BATCH_SIZE_EXPLAINER,
        )  # [B,H,W] CPU

        # 5) write back to aligned output order
        for local_k, global_pos in enumerate(positions):
            saliency_out[global_pos] = sal_group[local_k]     # [H,W]
            x_query_out[global_pos] = X_group[local_k].cpu()  # [3,H,W]

    res = {
        "saliency": torch.stack(saliency_out, dim=0),                 # [N,H,W]
        "X_query": torch.stack(x_query_out, dim=0),                   # [N,3,H,W]
        "query_indices": torch.as_tensor(q, dtype=torch.long),        # [N]
        "ref_indices": torch.as_tensor(r, dtype=torch.long),          # [N]
    }
    if pair_sims is not None:
        res["pair_sims"] = torch.as_tensor(pair_sims, dtype=torch.float32)


    save_dir = PATHS.data / "saliency_maps"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"ig_similarity_pair_specific__{model_name}__N{len(query_indices)}.pt"
    torch.save(res, save_path)

    print(f"[Info] Saved to: {save_path}")
    print("[Info] saliency:", tuple(res["saliency"].shape))
    print("[Info] X_query :", tuple(res["X_query"].shape))