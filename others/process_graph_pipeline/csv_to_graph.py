import os
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import pickle

def parse_region_map(region_file_path: str) -> tuple[list[str], list[tuple[int, int]], list[str]]:
    """
    Parses a region .txt file to determine the global node indices for each edge.
    
    Returns:
     - node_names: List of unique regions acting as nodes.
     - edge_list: A list of (src_index, dst_index) tuples in the same order as the file (and CSV columns).
    """
    with open(region_file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Extract raw region pairs depending on format
    # Example format string: "FC100 (IC1-IC2)" or "lh_bankssts-lh_caudalanteriorcingulate"
    raw_pairs = []
    for line in lines:
        if line.startswith("FC"):
            # Extract the part between parentheses
            inner = line[line.find("(")+1:line.find(")")]
            src, dst = inner.split("-")
        else:
            # MIND format: lh_bankssts-lh_caudalanteriorcingulate
            src, dst = line.split("-")
        raw_pairs.append((src, dst))

    # Determine unique nodes and assign them an index
    unique_nodes = []
    for src, dst in raw_pairs:
        if src not in unique_nodes: unique_nodes.append(src)
        if dst not in unique_nodes: unique_nodes.append(dst)
        
    node_to_idx = {name: idx for idx, name in enumerate(unique_nodes)}

    # Map the lines to edge indices
    edge_list = [(node_to_idx[src], node_to_idx[dst]) for src, dst in raw_pairs]
    
    # Return `lines` as the exact column names expected in the CSV
    return unique_nodes, edge_list, lines


def create_graphs_from_flattened(
    features_csv: str,       # Path to the flattened matrix CSV
    regions_txt: str,        # Path to the regions .txt file 
    train_ids_file: str,     # Path to the train EIDs txt file
    eid_col: str,            # The column name representing patient ID in the features CSV
    output_dir: str,
    keep_percent: float = 0.5,
    keep_self_loops: bool = True,
    standardize: bool = True
):
    print("Loading data...")
    start_time = time.time()
    
    # 1. Parse regions and get node mapping AND required column order
    node_names, edge_list_indices, expected_cols = parse_region_map(regions_txt)
    num_nodes = len(node_names)
    num_edges = len(edge_list_indices)
    
    print(f"Detected {num_nodes} unique nodes and {num_edges} edges from {regions_txt}")
    
    # 2. Load the flattened CSV
    # Assuming the CSV has an 'eid' column and the rest are edge features in the same order as the txt file
    df = pd.read_csv(features_csv)
    
    if eid_col not in df.columns:
        raise ValueError(f"Could not find EID column '{eid_col}' in flattened CSV.")
        
    eids = df[eid_col].astype(str).values
    
    # Isolate just the edge feature columns. 
    # Adjust this logic depending on how your flattened CSV is structured.
    # We assume all columns except the eid column are edge features.
    feature_cols = [c for c in df.columns if c != eid_col]
    
    if len(feature_cols) != num_edges:
        print(f"Warning: Number of feature columns in CSV ({len(feature_cols)}) "
              f"does not match number of edges in txt file ({num_edges}). "
              "Check your column alignment.")

    # After checking for eid_col, add this check:
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing {len(missing_cols)} expected columns in CSV, e.g., {missing_cols[:3]}")

    # Explicitly slice the DataFrame using the ordered list!
    X = df[expected_cols].values
    
    # Handle NaNs globally
    np.nan_to_num(X, nan=0.0, copy=False)

    # 3. Standardization based on Training IDs
    if standardize and train_ids_file and os.path.exists(train_ids_file):
        with open(train_ids_file, 'r') as f:
            train_eids = set(line.strip() for line in f if line.strip())
            
        print(f"Loaded {len(train_eids)} training IDs for standardizer fitting.")
        
        # Find which rows belong to the training set
        train_mask = np.isin(eids, list(train_eids))
        X_train = X[train_mask]
        
        print(f"Fitting StandardScaler on {len(X_train)} training subjects...")
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        # Transform the ENTIRE dataset
        X = scaler.transform(X)
        
        # Save standardizer for later use
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "edge_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
            
    elif standardize:
        print("Warning: Standardize is True but no train IDs provided. Standardizing across entire dataset (Data Leakage Risk!).")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # 4. Global edges mapping (preparing arrays for fast index slicing)
    src_global = np.array([e[0] for e in edge_list_indices], dtype=np.int64)
    dst_global = np.array([e[1] for e in edge_list_indices], dtype=np.int64)
    
    # We'll need to sort edges by magnitude per subject. 
    k = max(1, int(np.floor(keep_percent * num_edges)))
    print(f"Constructing graphs, keeping top {k} edges ({keep_percent*100}% per subject) ...")

    os.makedirs(output_dir, exist_ok=True)
    
    # Self-loops pre-computation
    if keep_self_loops:
        self_nodes = torch.arange(num_nodes, dtype=torch.long)
        self_edges = torch.stack([self_nodes, self_nodes], dim=0)
        self_attr = torch.zeros(num_nodes, dtype=torch.float32)

    # 5. Build Graph per Subject efficiently
    nodes_x = torch.arange(num_nodes, dtype=torch.int64)
    
    successful = 0
    
    # Vectorized sorting across rows (if data fits in memory, this is fast)
    # We get the indices of the top K largest absolute values for EVERY row
    top_k_indices_per_row = np.argsort(np.abs(X), axis=1)[:, -k:][:, ::-1]

    for i, eid in enumerate(eids):
        try:
            row_vals = X[i]
            top_idx = top_k_indices_per_row[i]
            
            # The selected edges
            w_u = row_vals[top_idx]
            src_u = src_global[top_idx]
            dst_u = dst_global[top_idx]
            
            # Make undirected graph
            src = np.concatenate([src_u, dst_u])
            dst = np.concatenate([dst_u, src_u])
            w = np.concatenate([w_u, w_u])
            
            edge_index = torch.from_numpy(np.stack([src, dst])).long()
            edge_attr = torch.from_numpy(w).float()
            
            if keep_self_loops:
                edge_index = torch.cat([edge_index, self_edges], dim=1)
                edge_attr = torch.cat([edge_attr, self_attr], dim=0)
                
            data = Data(x=nodes_x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
            data.node_names = node_names
            
            # Save format matches old pipeline expecting '{eid}_fc_graph.pt'
            output_file = os.path.join(output_dir, f"{eid}_fc_graph.pt")
            torch.save(data, output_file)
            successful += 1
            
            if successful % 1000 == 0:
                print(f"  Processed {successful}/{len(eids)} graphs...")
                
        except Exception as e:
            print(f"Failed converting EID {eid}: {e}")

    print(f"Successfully generated {successful} graph objects in {time.time() - start_time:.2f} seconds.")
