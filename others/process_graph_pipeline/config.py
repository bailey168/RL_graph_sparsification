import os
import argparse

dataset_dict = {"GF": "p20016_i2", "PAL": "p20197_i2", "DSST": "p23324_i2", "TMT": "p6350_i2"}
file_format_dict = {"MIND": "20263_2_0", "FC25": "25752_2_0", "FC100": "25753_2_0"}
matrix_dir_dict = {"MIND": "MIND_full", "FC25": "FC25_matrices", "FC100": "FC100_matrices"}

def parse_args():
    parser = argparse.ArgumentParser(description='Graph preprocessing pipeline')
    parser.add_argument('--dataset', choices=['GF', 'PAL', 'DSST', 'TMT'],
                       help='Dataset to process')
    parser.add_argument('--keep-percent', type=float,
                       help='Percentage of edges to keep in the graph (between 0 and 1)')
    parser.add_argument('--file-format', choices=['MIND', 'FC25', 'FC100'],
                       help='File format to use (MIND, FC25, or FC100)')
    return parser.parse_args()

# Parse command line arguments
args = parse_args()
DATASET = args.dataset
KEEP_PERCENT = args.keep_percent
FILE_FORMAT = args.file_format

# Determine if head motion should be included based on FILE_FORMAT
USE_HEAD_MOTION = FILE_FORMAT in ["FC25", "FC100"]

# Paths
DATA_DIR = "/opt/notebooks/combined_data_no_outliers"
OUTPUT_DIR = "/opt/notebooks"
CSV_FILE = os.path.join(DATA_DIR, f"combined_data_{DATASET}_no_outliers.csv")
TRAINING_IDS_FILE = os.path.join(OUTPUT_DIR, f"training_ids/{args.file_format}_{DATASET}.txt")

# Matrix directory based on file format only (all matrices stored together)
MATRIX_DIR = os.path.join(OUTPUT_DIR, matrix_dir_dict[args.file_format])

# Column names
DATA_COL = dataset_dict[DATASET]
EID_COL = "eid"
HEAD_MOTION_COL = "p25741_i2"

# Parameters
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
KEEP_SELF_LOOPS = True

# Standardization control
USE_EDGE_STANDARDIZATION = True

# Dynamic paths based on parameters
EDGE_PERCENT_STR = str(int(KEEP_PERCENT * 100))
GRAPH_DIR = os.path.join(OUTPUT_DIR, f"{args.file_format}_graphs/raw/{DATASET}{EDGE_PERCENT_STR}")
PROCESSED_DIR = os.path.join(OUTPUT_DIR, f"{args.file_format}_graphs/processed/{DATASET}{EDGE_PERCENT_STR}")

# Dynamic dataset name
DATASET_NAME = f"custom_dataset_selfloops_{KEEP_SELF_LOOPS}_edgeft_None_norm_{USE_EDGE_STANDARDIZATION}"

# Parallel processing configuration
# For I/O-bound tasks (file loading), using more workers than CPUs can improve throughput
# None = auto-detect based on CPU count, or set explicit value (e.g., cpu_count() * 2)
from multiprocessing import cpu_count as _cpu_count
MAX_WORKERS = _cpu_count() * 2  # Optimal for I/O-bound operations with many files