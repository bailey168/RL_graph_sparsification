# File: /graph-preprocessing-pipeline/graph-preprocessing-pipeline/src/training_ids.py

import os
import pandas as pd
import numpy as np
from typing import List
import argparse

def create_training_ids_file(csv_file: str, output_path: str, train_ratio: float = 0.8, 
                           random_seed: int = 42, eid_col: str = 'eid', file_format: str = None) -> str:
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV with {len(df)} rows")
    
    if eid_col not in df.columns:
        raise ValueError(f"Column '{eid_col}' not found in CSV file")
    
    available_eids = df[eid_col].tolist()
    print(f"Found {len(available_eids)} unique EIDs")
    
    np.random.seed(random_seed)
    
    indices = np.random.permutation(len(available_eids))
    split_idx = int(len(available_eids) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_eids = [available_eids[i] for i in train_indices]
    val_eids = [available_eids[i] for i in val_indices]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for eid in train_eids:
            # Just write the raw EID
            f.write(f"{eid}\n")
    
    print(f"Created training IDs file with {len(train_eids)} IDs: {output_path}")
    print(f"Training ratio: {train_ratio:.2f}")
    print(f"Random seed: {random_seed}")
    
    print(f"Sample IDs:")
    for i, eid in enumerate(train_eids[:3]):
        print(f"  {i+1}: {eid}")
    
    return output_path

def validate_training_ids_file(ids_file: str, csv_file: str, eid_col: str = 'eid') -> None:
    # Read EIDs directly as integers
    with open(ids_file, 'r') as f:
        train_eids = set(int(line.strip()) for line in f if line.strip())
    
    df = pd.read_csv(csv_file)
    csv_eids = set(df[eid_col].tolist())
    
    missing_ids = train_eids - csv_eids
    
    if missing_ids:
        print(f"Warning: {len(missing_ids)} training EIDs not found in CSV:")
        print(f"Missing EIDs (first 10): {list(missing_ids)[:10]}")
    else:
        print("✓ All training EIDs found in CSV file")
    
    print(f"Training EIDs: {len(train_eids)}")
    print(f"CSV EIDs: {len(csv_eids)}")
    print(f"EID overlap: {len(train_eids & csv_eids)}")
    
    print("Sample training EIDs:")
    for i, eid in enumerate(list(train_eids)[:3]):
        print(f"  {i+1}: {eid}")