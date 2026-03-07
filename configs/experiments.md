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