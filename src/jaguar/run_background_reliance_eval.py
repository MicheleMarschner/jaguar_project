"""
2. Baue für die Background-Reliance-Frage keinen komplett neuen Analyse-Stapel

Sondern nur eine zweite Preparation-Funktion für andere Input-Bedingungen:

prepare_query_gallery_retrieval(...)
für normales Retrieval / XAI auf Originalbildern

prepare_query_gallery_retrieval_for_condition(...)
für:

original

jaguar_only

bg_only

Warum:
Die eigentliche Struktur ist dieselbe. Nur die Quelle der Embeddings ändert sich.

3. Trenne Haupt-RQ-Test und Diagnostik sauber
Haupttest

Ein neuer schlanker Runner, z. B.:

run_background_reliance_eval.py

Der macht nur:

fixes Modell laden

Split-Datei als Source of Truth laden

Query/Gallery sauber definieren

für verschiedene Bedingungen evaluieren

mAP, rank1, Delta speichern

Diagnostik

Dein bisheriges MaskAware-/Logit-/Stability-/CAM-Skript bleibt bestehen.

Warum:
Sonst vermischst du:

die eigentliche Forschungsfrage

und viele Zusatzdiagnosen

Die Hauptfrage braucht einen klaren, einfachen, kontrollierten Befund.

4. Die eigentliche RQ so beantworten

Mit fixem Modell und gleichem Eval-Protokoll:

Original → Original

Jaguar-only → Jaguar-only

optional später:

Original → Jaguar-only

Jaguar-only → Original

BG-only → BG-only

Dann berichtest du:

mAP_original

mAP_jaguar_only

delta

Warum:
Das ist der direkteste Test auf Kontextabhängigkeit:

Modell bleibt gleich

nur der Hintergrund ändert sich

Leistungsabfall ist dann interpretierbar


"""

def save_background_reliance_outputs():
    pass


def extract_condition_embeddings():
    pass

def evaluate_background_condition():  # oder direkt ein kleiner Wrapper um evaluate_query_gallery_retrieval(...)
    pass





def run_background_reliance_eval(config, save_dir):
    
    parquet_root = resolve_path(config["data"]["split_data_path"], EXPERIMENTS_STORE)
    data_path = PATHS.data_export / "splits_curated"

    train_processing_fn = build_processing_fn(config, split="train")
    val_processing_fn = build_processing_fn(config, split="val")

    _, train_ds, val_ds = load_split_jaguar_from_FO_export(
        data_path,
        overwrite_db=False,
        parquet_path=parquet_root,
        train_processing_fn=train_processing_fn,
        val_processing_fn=val_processing_fn,
        include_duplicates=config["split"]["include_duplicates"],
        use_fiftyone=config["data"]["use_fiftyone"]
    )

    # Calculate Identities
    num_classes = len(train_ds.label_to_idx)


    # Initialize Model
    model = JaguarIDModel(
        backbone_name=config['model']['backbone_name'],
        num_classes=num_classes,
        head_type=config['model']['head_type'],
        device=DEVICE,
        emb_dim=config['model']['emb_dim'],
        freeze_backbone=config['model']['freeze_backbone'],
        loss_s=config["model"].get("s", 30.0),
        loss_m=config["model"].get("m", 0.5),
    )

    checkpoint = torch.load(checkpoint_dir / "best_model.pth", map_location=DEVICE, weights_only=False)

    ## besonderheiten per model??
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    
    val_ds.transform = model.backbone_wrapper.transform
    train_ds.transform = model.backbone_wrapper.transform

    val_embeddings = load_or_extract_jaguarid_embeddings(
        model=model,
        torch_ds=val_ds,
        split="val",
        batch_size=config["inference"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_tta=config["inference"]["use_tta"],
    )

    train_embeddings = load_or_extract_jaguarid_embeddings(
        model=model,
        torch_ds=train_ds,
        split="train",
        batch_size=config["inference"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_tta=config["inference"]["use_tta"],
    )

    gallery_embeddings = np.concatenate([train_embeddings, query_embeddings], axis=0)
    query_labels = np.asarray(val_ds.labels)
    gallery_labels = np.concatenate([np.asarray(train_ds.labels), np.asarray(val_ds.labels)], axis=0)




if __name__ == "__main__":
    run_background_reliance_eval()

