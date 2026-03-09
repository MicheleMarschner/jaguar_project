
# EXPERIMENTS


frozen backbone: 1e-4
unfrozen backbone: 2e-5

## Kaggle

DEFAULT:
- Base: Kaggle
- Background: white (for train and val)

1. DEDUP   -> [data]: set base-condition for split
2. BACK    -> [model]: set base-condition for model backbone
3. LOSS    -> 
4. MINING
5. AUG
6. OPTIM
7. RESIZ
8. STAB
9. SOUP
10. TTA
11. ENSEMB


### DEDUP Near-Duplicate Training Impact / Deduplication

- Must: burst analysis and separating bursts in val/train
- Must: closed-set
- Experiment: phash 4 (distance 0–4 → very likely duplicate / same burst frame)
    - keep all duplicates
    - remove duplicates: train_k: 1 , val_k: 50 , phash: 4
    - remove duplicates: train_k: 3, val_k: 1 , phash: 4
    - remove duplicates: train_k: 3, val_k: 3, phash: 4
    - remove duplicates: train_k: 1, val_k: 1, phash: 4

- Output: 
    - [data]: set base-condition for split (two train variants: one full and one best result or even 2 - check außerdem die stats dazu!)
    - baseline for future runs (EVA-02)


### BACK    Backbone 

- Must: 2 Splits from DEDUP
- Experiment:
    - MegaDescriptor-L
    - MiewID
    - DINOv2_for_wildlife
    - DINOv2-Base or DINOv3-Base
    - ConvNeXt-V2
    - EVA-02

- Ouput: 
    - [model]: set base-condition for model backbone (2 best over the 2 runs - and check if CNN/ViT)
    - identify strongest ViT and strongest CNN


### LOSS    Loss Comparison




### ENSEMB  Ensemble
- best specialized model    (from MegaDescriptor-L, MiewID, DINOv2_for_wildlife)
- best generic ViT          (from DINOv2-Base, DINOv3-Base, EVA-02) 
- best CNN                  (from ConvNeXt-V2, Efficien-NetB4)




# SCIENTIFIC

DEFAULT:
- Base: Scientific
- Background: white (for train and val)

## BACKGROUND 








# SCIENTIFIC

base-conditions:
[model]: backbone from kaggle Experiment 1

## Experiment 1 Near-Duplicate Training Impact / Deduplication


-> [data]: set base-condition for split

## Experiment 2 Background Reliance / Background Ablation


-> [preprocessing]: set base-condition for background

## Experiment 3 XAI on background (+ Classification Sensitivity)
## Experiment 4 Interpretability sanity
## Experiment 5: Viewpoint Sensitivity 
## Experiment 6: Statistical Stability
## Experiment 7: Hyperbolic embeddings


# KAGGLE

base-conditions:
- pipeline after kaggle notebook (94% score)
- backbone: (set after backbone ablation)

[data] : 
- origin: full train ds
- split: stratified, 5% val

[preprocessing] : 
- background: train & val = "original"


## Experiment 1 Backbone Comparison
## Experiment 2 Data Augmentation Ablation
## Experiment 3 Optimizer & Scheduler Grid
## Experiment 4 Loss Function Comparison
## Experiment 5 Progressive Resizing
## Experiment 6 Hard Negative Mining (switch to scientific if hurtful)
## Experiment 7 TTA
## Experiment 8 Model Soup / Ensemble






### SEED:

- bursts sollte in config rein, muss aber ncith mit training verdrahtet werden (sonst eben nur im setup schritt)
- preprocessing background auch dasselbe

- run_xai_similarity / run_xai_metric_similarity / run_xai_classifiction / xai_background_sensitivity / xai_similarity / spli_and_curate


### Augmentation
1. zeigt euch den Nettoeffekt des gesamten Aug-Pakets
2. aktuelles Setup als Referenz
3. Random Erasing ist oft stark wirksam, aber kann bei feinen ID-Mustern auch schaden
4. Nur Flip + Affine, kein ColorJitter, kein Erasing: testet, ob geometrische Robustheit der Haupttreiber ist
5. Geometrie + Erasing, aber kein ColorJitter: guter Test, ob ColorJitter wirklich hilft oder eher stört


### Losses
- Triplet muss noch richtig gewired werden


### Scheduler / Optimizer
- Aber OneCycleLR sollte pro Batch step() bekommen, nicht pro Epoch. (Batch-wise scheduler)
- OneCycle braucht saubere Parameter
(implementiert - überprüfen!)

Vorschlag: 
AdamW + CosineAnnealingLR   AdamW + OneCycleLR  Muon + CosineAnnealingLR    Muon + OneCycleLR
oder: Adam + JaguardIdScheduler     Adam + CosineAnnealingLR    AdamW + JaguardIdScheduler  AdamW + CosineAnnealingLR



### PORTABLE SETUP
kaggle/default split
folder struktur von data mit round1 und train/train muss gegeben sein
init vs curated anpassen