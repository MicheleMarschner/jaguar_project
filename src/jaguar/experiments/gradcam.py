'''
Add: Margin(q, pos, neg)
If your question is “what supports correct ranking against an imposter?” → margin

CAM highlights pixels that increase separation: closer to pos AND farther from neg.
'''
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from jaguar.config import PATHS, DEVICE
from jaguar.utils.utils_datasets import load_jaguar_from_FO_export
from jaguar.models.foundation_models import FoundationModelWrapper
from jaguar.evaluation.metrics import ReIDEvalBundle
from jaguar.evaluation.mining import mining_pack_for_query
from jaguar.utils.utils_explainer import generate_similarity_cam

def save_cam_row_panel(
    query_img: Image.Image,
    ref_img: Image.Image,
    cam_overlay_query: np.ndarray,  # uint8 RGB from show_cam_on_image
    out_path: Path,
    title: str,
):
    """
    Saves one row: [Query raw] | [Ref raw] | [Query + CAM overlay]
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    q = np.array(query_img.convert("RGB"))
    r = np.array(ref_img.convert("RGB"))
    o = cam_overlay_query

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(q); axes[0].set_title("Query"); axes[0].axis("off")
    axes[1].imshow(r); axes[1].set_title("Reference"); axes[1].axis("off")
    axes[2].imshow(o); axes[2].set_title(title); axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":

    model_name = "MiewID"         # MegaDescriptor-L, DINOv2-Base, MiewID, ConvNeXt-V2
    dataset_name = "jaguar_stage0"
    base_root = PATHS.data_export

    fo_ds, torch_ds = load_jaguar_from_FO_export(
        PATHS.data_export,
        dataset_name=dataset_name,
        processing_fn=None,
        overwrite_db=False,
    )

    print("[Info] Dataset size:", len(torch_ds))

    filepaths = [str(torch_ds._resolve_path(s[torch_ds.filepath_key])) for s in torch_ds.samples]
    labels    = np.asarray(torch_ds.labels)

    # Load model wrapper
    model_wrapper = FoundationModelWrapper(model_name, device=DEVICE)
    split = "training"
    print("[Info] Loaded model:", model_wrapper.name)

    emb_dir = PATHS.data / "embeddings"
    emb_path = emb_dir / f"embeddings_{model_wrapper.name}_{split}.npy"

    if emb_path.exists():
        embs = np.load(str(emb_path))
        print(f"[Info] Loaded embeddings from {emb_path} shape={embs.shape}")
    else:
        print(f"[Info] Computing embeddings -> {emb_path}")
        batch_size = 32
        all_embs = []
        for start in range(0, len(filepaths), batch_size):
            batch_paths = filepaths[start:start + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            batch_emb = model_wrapper.extract_embeddings(imgs)  # (B,D) numpy
            all_embs.append(batch_emb)

        embs = np.vstack(all_embs)
        np.save(str(emb_path), embs)
        print(f"[Info] Saved embeddings to {emb_path} shape={embs.shape}")

        assert embs.shape[0] == len(filepaths), "Embeddings must align with dataset order!"
        print("[Info] Embedding dim:", embs.shape[1])
    
    bundle = ReIDEvalBundle(
        embeddings=embs,
        labels=labels,
        model=None,          # embeddings already computed
        device=str(DEVICE),
        normalize=True,      # (cosine stable)
    )

    sim = bundle.similarity_matrix()   # (N, N), diagonal is -1
    ranked = bundle.ranked_indices()   # (N, N), sorted by similarity desc

    print("[Info] sim shape:", sim.shape, "ranked shape:", ranked.shape)
    print("[Info] Top-5 neighbors for idx=0:", ranked[0][:5], "sims:", sim[0, ranked[0][:5]])

    idx = 4  # query index
    pack = mining_pack_for_query(
        i=idx,
        sim=sim,
        ranked=ranked,
        labels=labels,
        cluster_id=None,          # omit burst filtering for now
        neg_ranks=(1, 5, 10),
    )

    print(pack)

    q_idx = pack["q"]
    pos_easy_idx = pack["pos_easy"]["idx"]
    pos_hard_idx = pack["pos_hard"]["idx"]

    # pick rank-1 imposter
    neg1_idx = [n["idx"] for n in pack["negs"] if n["rank"] == 1][0]

    query_img = Image.open(filepaths[q_idx]).convert("RGB")
    pos_easy_img = Image.open(filepaths[pos_easy_idx]).convert("RGB")
    pos_hard_img = Image.open(filepaths[pos_hard_idx]).convert("RGB")
    neg1_img = Image.open(filepaths[neg1_idx]).convert("RGB")

    # get embeddings for query / pos / neg and compare
    E = model_wrapper.extract_embeddings([query_img, pos_easy_img, neg1_img])  # (3,D)
    def cos(a,b):
        a = a / (np.linalg.norm(a)+1e-12)
        b = b / (np.linalg.norm(b)+1e-12)
        return float((a*b).sum())

    print("cos(q,pos) =", cos(E[0], E[1]))
    print("cos(q,neg) =", cos(E[0], E[2]))
    print("cos(pos,neg) =", cos(E[1], E[2]))

    x = model_wrapper.preprocess(query_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model_wrapper.model(x)
    print("model output shape:", getattr(out, "shape", type(out)))

    '''

    out_dir = PATHS.results / f"XAI/{model_wrapper.name}_q{q_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # POS easy/hard: attract + repel
    _, vis_pos_easy_att = generate_similarity_cam(model_wrapper, query_img, pos_easy_img, maximize=True)
    _, vis_pos_easy_rep = generate_similarity_cam(model_wrapper, query_img, pos_easy_img, maximize=False)

    _, vis_pos_hard_att = generate_similarity_cam(model_wrapper, query_img, pos_hard_img, maximize=True)
    _, vis_pos_hard_rep = generate_similarity_cam(model_wrapper, query_img, pos_hard_img, maximize=False)

    # NEG rank-1: attract-to-imposter + repel-from-imposter
    _, vis_neg1_att = generate_similarity_cam(model_wrapper, query_img, neg1_img, maximize=True)
    _, vis_neg1_rep = generate_similarity_cam(model_wrapper, query_img, neg1_img, maximize=False)

    # --- save panels ---
    # Pos easy
    save_cam_row_panel(
        query_img=query_img,
        ref_img=pos_easy_img,
        cam_overlay_query=vis_pos_easy_att,
        out_path=out_dir / f"pos_easy_attract_idx{pos_easy_idx}.png",
        title=f"POS EASY • attract • sim={pack['pos_easy']['sim']:.3f}",
    )
    save_cam_row_panel(
        query_img=query_img,
        ref_img=pos_easy_img,
        cam_overlay_query=vis_pos_easy_rep,
        out_path=out_dir / f"pos_easy_repel_idx{pos_easy_idx}.png",
        title=f"POS EASY • repel • sim={pack['pos_easy']['sim']:.3f}",
    )

    # Pos hard
    save_cam_row_panel(
        query_img=query_img,
        ref_img=pos_hard_img,
        cam_overlay_query=vis_pos_hard_att,
        out_path=out_dir / f"pos_hard_attract_idx{pos_hard_idx}.png",
        title=f"POS HARD • attract • sim={pack['pos_hard']['sim']:.3f}",
    )
    save_cam_row_panel(
        query_img=query_img,
        ref_img=pos_hard_img,
        cam_overlay_query=vis_pos_hard_rep,
        out_path=out_dir / f"pos_hard_repel_idx{pos_hard_idx}.png",
        title=f"POS HARD • repel • sim={pack['pos_hard']['sim']:.3f}",
    )

    # Neg rank-1
    neg1_sim = [n["sim"] for n in pack["negs"] if n["rank"] == 1][0]
    save_cam_row_panel(
        query_img=query_img,
        ref_img=neg1_img,
        cam_overlay_query=vis_neg1_att,
        out_path=out_dir / f"neg_r1_attract_idx{neg1_idx}.png",
        title=f"NEG R1 • attract-to-imposter • sim={neg1_sim:.3f}",
    )
    save_cam_row_panel(
        query_img=query_img,
        ref_img=neg1_img,
        cam_overlay_query=vis_neg1_rep,
        out_path=out_dir / f"neg_r1_repel_idx{neg1_idx}.png",
        title=f"NEG R1 • repel-from-imposter • sim={neg1_sim:.3f}",
    )

    print("[Info] Saved:", out_dir)

'''