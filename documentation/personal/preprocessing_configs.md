**Burst Discovery (1 run - fixed for now)**
BURST_MIN_CLUSTER_SIZE = 2      # fixed
BURST_MAX_WITHIN = 500          # fixed
BURST_MAX_CROSS = 10000         # fixed
PHASH_THRESHOLD = COLLISION_FREE_THRESHOLD    # set by configuration (collision-free)
val_split_size = 0.2                  # fixed
SEED = 51                       # fixed


**Background Ablation (6 runs)**
BACKGROUND = "original", "random_bg", "black_bg", "gray_bg", "white_bg", "blur_bg"      ## ! needs to be applied to dataset as processing_fn


**Deduplication Ablation (50 runs)**
TRAIN_K_PER_DEDUP = 1,3,6      # Parameter     # just for duplicates  
VAL_K_PER_DEDUP = 4, 50        # Parameter      # just for duplicates
SPLIT_STRATEGY = ["open_set","closed_set"]     # Parameter 
INCLUDE_DUPLICATES = True, False    # Parameter, on the fly
PHASH_THRESH_DEDUP = [2, 4, 8, 13]               # Parameter - just for duplicates


FILE PATHS: 


-> if DEDUP_POLICY == "drop_duplicates" filter by "keep_curated" col
-> choose Split type (train, val) in col split_final


