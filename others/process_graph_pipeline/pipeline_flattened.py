import os
import time
from config import (DATA_DIR, OUTPUT_DIR, DATASET_NAME,
                   TRAIN_RATIO, RANDOM_SEED, KEEP_PERCENT, KEEP_SELF_LOOPS,
                   USE_EDGE_STANDARDIZATION, FILE_FORMAT, USE_HEAD_MOTION, HEAD_MOTION_COL,
                   MAX_WORKERS, DATASET, DATA_COL, EID_COL)
                   
from training_ids import create_training_ids_file, validate_training_ids_file
from csv_to_graph import create_graphs_from_flattened
from create_dataset import process_custom_graph_dataset

# Custom paths for the the flattened data requirements
REGION_MAPS = {
    "FC25": "/Users/baileyng/MIND_models/region_names/FC25_regions.txt",
    "FC100": "/Users/baileyng/MIND_models/region_names/FC100_regions.txt",
    "MIND": "/Users/baileyng/MIND_models/region_names/MIND_regions.txt",
}

# TODO: REPLACE WITH ACTUAL FLATTENED MATRIX CSV PATH
FLATTENED_MATRIX_CSV = f"/path/to/your/flattened_matrices_{FILE_FORMAT}.csv"

# Demographics / Target variables CSV
DEMOGRAPHICS_CSV = os.path.join(DATA_DIR, f"combined_data_{DATASET}_no_outliers.csv")

# Dynamic graph path (from config.py) 
EDGE_PERCENT_STR = str(int(KEEP_PERCENT * 100))
GRAPH_DIR = os.path.join(OUTPUT_DIR, f"{FILE_FORMAT}_graphs/raw/{DATASET}{EDGE_PERCENT_STR}")
PROCESSED_DIR = os.path.join(OUTPUT_DIR, f"{FILE_FORMAT}_graphs/processed/{DATASET}{EDGE_PERCENT_STR}")
TRAINING_IDS_FILE = os.path.join(OUTPUT_DIR, f"training_ids/{FILE_FORMAT}_{DATASET}.txt")

def run_flattened_pipeline():
    print("=== Flattened Graph Preprocessing Pipeline ===")
    
    # 1. Create Train splits (Runs on the Demographics CSV)
    print("Step 1: Creating training IDs file...")
    training_ids_file = create_training_ids_file(
        csv_file=DEMOGRAPHICS_CSV,
        output_path=TRAINING_IDS_FILE,
        train_ratio=TRAIN_RATIO,
        random_seed=RANDOM_SEED,
        eid_col=EID_COL,
        file_format=FILE_FORMAT
    )
    print()

    # 2. Process graphs from flattened CSV directly
    print("Step 2: Processing flattened matrices directly into PyG Graphs...")
    regions_txt = REGION_MAPS[FILE_FORMAT]
    
    create_graphs_from_flattened(
        features_csv=FLATTENED_MATRIX_CSV,
        regions_txt=regions_txt,
        train_ids_file=training_ids_file,
        eid_col=EID_COL,
        output_dir=GRAPH_DIR,
        keep_percent=KEEP_PERCENT,
        keep_self_loops=KEEP_SELF_LOOPS,
        standardize=USE_EDGE_STANDARDIZATION
    )
    print()

    # 3. Join with demographic data
    print("Step 3: Creating the dataset with labels...")
    process_custom_graph_dataset(
        data_col=DATA_COL,
        graph_dir=GRAPH_DIR,
        csv_file=DEMOGRAPHICS_CSV,
        output_dir=PROCESSED_DIR,
        dataset_name=DATASET_NAME,
        train_ratio=TRAIN_RATIO,
        random_seed=RANDOM_SEED,
        use_head_motion=USE_HEAD_MOTION,
        head_motion_col=HEAD_MOTION_COL,
        max_workers=MAX_WORKERS
    )
    print()
    print("Pipeline execution completed successfully!")

if __name__ == "__main__":
    start = time.time()
    run_flattened_pipeline()
    print(f"Total runtime: {time.time() - start:.2f}s")