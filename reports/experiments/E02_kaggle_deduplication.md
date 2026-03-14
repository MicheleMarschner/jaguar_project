# E02 Kaggle Deduplication

**Experiment Group:** Interpretability analyses

## Main Research Question
How should near-duplicate and burst redundancy be identified, controlled, and curated in the Jaguar dataset to obtain a leakage-safe evaluation setup and improve Jaguar Re-ID performance?

---

## Stage 1 — Burst Discovery / Near-Duplicate Annotation

### Research Question
To what extent does the Jaguar dataset contain burst-like and near-duplicate redundancy, and how can this redundancy be characterized quantitatively and structurally?

### Intervention
Burst discovery via pHash-based candidate generation, conservative thresholding, and connected-component grouping.

### Method / Procedure
To identify burst-like redundancy, perceptual hashes (pHash) were first computed for all images and compared within identity only. This provides a fast and conservative signal for very close visual similarity while avoiding unsafe links between different individuals at the candidate-generation stage.  
The pHash threshold was then selected using within-identity versus cross-identity diagnostics: the final cutoff was chosen as the largest threshold that still avoided cross-identity collisions.

After thresholding, retained pairs were converted into a graph and grouped via connected components. This is important because burst redundancy is often transitive: if image A matches B and B matches C, all three may belong to the same burst even when A and C are not the strongest direct pair.

pHash was used first because it is efficient, deterministic, and specific to near-duplicate visual structure. By contrast, embedding similarity was not used in isolation for the initial threshold because it is more semantic and therefore less specific to true duplicate bursts. Used too early, it can over-link images that share identity, pose, or background without originating from the same burst event. For this reason, similarity is better suited as a later refinement signal than as the primary criterion for initial burst discovery.

### Evaluation
*The pHash threshold was chosen via within-/cross-identity diagnostics. Cross-identity collisions were zero up to threshold 11 and first appeared at threshold 12 (2/10,000). We therefore used pHash ≤ 11, the largest collision-free threshold, which retained 689 within-identity links without introducing sampled cross-identity links.*

- chosen threshold = 11  
- within links at chosen = 689  
- first unsafe threshold = 12  

### Key Result / Takeaway
[leave for now]

---

## Stage 2 — Split and Duplicate-Aware Curation

### Research Question
Can leakage-safe train/validation splits be constructed by enforcing burst-level separation while optionally controlling redundancy within splits, without violating the intended evaluation protocol?

### Intervention
Burst-aware split construction with strict burst separation across train/validation and optional within-split duplicate-aware curation.

### Method / Procedure
Starting from the burst-annotated manifest, train/validation splits were defined under two possible policies: open-set (identities disjoint across splits) and closed-set (identities may appear in both splits, but burst groups remain disjoint). The project ultimately proceeded with the closed-set protocol, since this matched the Kaggle challenge.

To prevent leakage, splitting was done at burst-group level rather than image level. Each burst group was treated as one indivisible split unit, while non-burst images were treated as singleton units. These units were then assigned to train or validation with approximate identity stratification and mapped back to image rows. This ensures that no near-duplicate burst is split across train and validation.

After splitting, duplicate-aware curation could either be skipped or applied within each split. If duplicates were kept, all images remained in the final dataset. If curation was enabled, each burst was further partitioned into tighter duplicate subclusters using a stricter pHash threshold. This threshold was not fixed a priori; instead, a small sweep over candidate thresholds was used to measure how many images would be removed under each setting in train and validation. This made the redundancy–retention trade-off explicit and allowed selection of a threshold that reduced obvious duplicates without discarding too much data or creating a strong imbalance between splits.

Within each resulting subcluster, up to train_k images were retained in training and up to val_k in validation. These parameters control how much local redundancy is preserved per duplicate set. Representatives were selected using a simple ranking heuristic based on embedding centrality and image sharpness. A possible future refinement would be to prefer a more diverse subset rather than only the most central and sharpest images.

### Evaluation
Split quality was evaluated through burst-leakage checks, split-size summaries, and duplicate-retention statistics under different intra-burst curation thresholds. The final selected setting used a closed-set protocol with strict burst-group separation and duplicate-aware curation (train_k = 3, val_k = 3, intra-burst pHash threshold = 4). Curation reduced the dataset from 1895 to 1634 samples while preserving all 31 identities and a similar train/validation ratio.

Main table to show:
- phash_threshold
- train_dropped
- train_drop_pct
- val_dropped
- val_drop_pct

Suggested summary sentence:
*A small intra-burst pHash sweep was used to quantify curation strength. The final threshold (pHash = 4, with train_k = 3, val_k = 3) was selected as a compromise between removing redundant images and preserving similar train/validation retention.*

### Key Result / Takeaway
[leave for now]

---

## Stage 3 — Experimentation

### Research Question
How does the degree of duplicate-aware curation within a fixed closed-set setup affect Jaguar Re-ID retrieval performance?

### Intervention
Controlled variation of curation strength through different train_k / val_k settings under otherwise fixed experimental conditions.

### Method / Procedure
This stage isolated the effect of duplicate-aware curation on retrieval performance. All other parts of the pipeline were kept fixed, including the burst annotations, the closed-set split protocol, the pHash threshold for intra-burst duplicate subclustering, and the model, training, and evaluation settings. This makes differences in performance attributable to the data variant itself.

The experiment series consisted of a reference full split and several curated split variants. In the full split, all images were retained after burst-aware splitting. In the curated variants, duplicate-aware curation was applied within each split by varying train_k and val_k, i.e. the number of images kept per local duplicate subcluster in training and validation. This enabled comparison across different curation strengths, from more permissive to more aggressive settings.

The goal was to study the trade-off between reducing near-duplicate redundancy and preserving useful within-identity variation. Smaller train_k or val_k values enforce stronger deduplication, while larger values retain more repeated views. The experiment series therefore evaluates not only whether curation is beneficial, but also what level of curation is most appropriate.

### Evaluation
Main comparison across runs:
- condition / run_name
- train_k
- val_k
- id_balanced_mAP
- pairwise_AP
- rank1
- sim_gap
- best_epoch

Optional representative full-vs-curated example:
- Δ id-balanced mAP
- Δ pairwise AP
- Δ rank-1
- Δ sim_gap
- Δ best epoch
- Δ post-peak validation drop

### Key Result / Takeaway
[leave for now]

---

## Overall Conclusion
[leave for now]