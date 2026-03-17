import pandas as pd 
import wandb 
from pathlib import Path

from jaguar.retrieval.retrieval_main import ( 
    load_model, 
    load_checkpoint_config 
)
from jaguar.retrieval.retrieval_utils import ( 
    evaluate_retrieval, 
    build_error_table, 
    get_cached_embeddings
) 
from jaguar.retrieval.soup.soup_grouping import ( 
    discover_seed_models, 
    group_models 
)
from jaguar.retrieval.soup.soup_utils import average_checkpoints
from jaguar.config import DATA_ROOT

def run_soup_sensitivity(
    root_dir,
    val_loader,
    labels,
    run_cfg,
    output_dir,
    wandb_run=None,
    extra_checkpoint_dir=None
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = discover_seed_models(root_dir)
    groups = group_models(models, run_cfg["group_by"])

    all_results = []
    
    for group, members in groups.items():
        print(f"\nProcessing group: {group}")
        group_dir = output_dir / str(group)
        group_dir.mkdir(exist_ok=True)
        group_metrics = []
        
        # Keep track of the first model for error comparison
        representative_model_metrics = None
        representative_model_sim = None
        representative_model_name = None

        # ---------------Evaluate individual model---------------
        for idx, m in enumerate(members):
            checkpoint_path = m["path"]
            # Define emebddings directory for fast caching
            embeddings_dir = checkpoint_path / "embeddings"
            embeddings_dir.mkdir(exist_ok=True)

            config = load_checkpoint_config(checkpoint_path)

            model = load_model(checkpoint_path, config)
            
            emb = get_cached_embeddings(
                model,
                val_loader,
                run_cfg.get("tta_modality", None),
                embeddings_dir
            )
    
            metrics, sim_matrix = evaluate_retrieval(
                emb,
                labels,
                qe=run_cfg.get("apply_qe", False), 
                qe_k=run_cfg.get("top_k_expansion", 3),
                rerank=run_cfg.get("apply_rerank", False), 
                k1=run_cfg.get("k1", 20),
                k2=run_cfg.get("k2", 6),
                lambda_value=run_cfg.get("lambda_value", 0.3),
            )
            print(metrics)

            res = {"group": group, "model": m["name"], "type": "individual", **metrics}
            group_metrics.append(metrics)
            all_results.append(res)
            
            # Save representative model info (first model of group)
            if idx == 0:
                representative_model_metrics = metrics
                representative_model_sim = sim_matrix
                representative_model_name = m["name"]

        # ---------------Compute statistics---------------
        df = pd.DataFrame(group_metrics)
        stats = df.agg(["mean", "std"])
        print(stats)
        stats.to_csv(group_dir / "statistics.csv")
        
        if wandb_run:
            stats_table = wandb.Table(columns=["stat"] + df.columns.tolist())
            for stat_name, row in stats.iterrows():
                stats_table.add_data(stat_name, *[row[col] for col in df.columns])
            
            wandb_run.log({f"soup-sensitivity/{group}_stats": stats_table})

        # ---------------Model soup---------------
        if run_cfg["build_model_soup"]:
            print(f"Building soup for group {group}...")
            checkpoints = [m["path"] for m in members]
            soup_state = average_checkpoints([checkpoints[1], checkpoints[3]])
            
            # Use architecture from first member
            config = load_checkpoint_config(checkpoints[0])
            soup_model = load_model(checkpoints[0], config)
            soup_model.load_state_dict(soup_state)
            
            emb_soup = get_cached_embeddings(
                soup_model,
                val_loader,
                run_cfg.get("tta_modality", None),
                group_dir / "embeddings_soup"
            )
            
            soup_metrics, soup_sim = evaluate_retrieval(
                emb_soup,
                labels,
                qe=run_cfg.get("apply_qe", False), 
                qe_k=run_cfg.get("top_k_expansion", 3),
                rerank=run_cfg.get("apply_rerank", False), 
                k1=run_cfg.get("k1", 20),
                k2=run_cfg.get("k2", 6),
                lambda_value=run_cfg.get("lambda_value", 0.3),
            )

            all_results.append({
                "group": group,
                "model": f"soup_{group}",
                "type": "soup",
                **soup_metrics
            })
        
        # Log Error Table for the Soup Model
        if wandb_run and representative_model_sim is not None:
            error_df = build_error_table(soup_sim, labels, val_loader.dataset)
            single_error_df = build_error_table(representative_model_sim, labels, val_loader.dataset)
            
            table_soup = wandb.Table(columns=["query", "pred", "true_label", "pred_label", "similarity"])
            
            # Limit to top 20 for W&B performance 
            for _, row in error_df.head(20).iterrows(): 
                table_soup.add_data( 
                        wandb.Image(str(DATA_ROOT / row['query_path'])), 
                        wandb.Image(str(DATA_ROOT / row['pred_path'])), 
                        row["query_label"], 
                        row["pred_label"], 
                        row["similarity"], 
                )
            wandb_run.log({f"soup-sensitivity/errors_soup": table_soup})
            
            table_single = wandb.Table(columns=["query", "pred", "true_label", "pred_label", "similarity"])
            
            # Limit to top 20 for W&B performance 
            for _, row in single_error_df.head(20).iterrows(): 
                table_single.add_data( 
                        wandb.Image(str(DATA_ROOT / row['query_path'])), 
                        wandb.Image(str(DATA_ROOT / row['pred_path'])), 
                        row["query_label"], 
                        row["pred_label"], 
                        row["similarity"], 
                )
            wandb_run.log({f"soup-sensitivity/errors_{representative_model_name}": table_single})

    # ---------------Leaderboard---------------
    
    df = pd.DataFrame(all_results)
    leaderboard = df.sort_values("mAP", ascending=False)
    print(leaderboard)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    
    if wandb_run:
        leader_table = wandb.Table(columns=["model_number"] + df.columns.tolist())
        for column_name, row in df.iterrows():
            leader_table.add_data(column_name, *[row[col] for col in df.columns])
        wandb_run.log({f"soup-sensitivity/leaderboard_{group}": leader_table})

    return leaderboard