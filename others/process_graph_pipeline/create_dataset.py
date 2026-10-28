import os
import pandas as pd
import numpy as np
import torch
import pickle
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_graph_data_with_labels(data_col: str, graph_dir: str, csv_file: str, 
                                use_head_motion: bool = False, head_motion_col: str = None,
                                max_workers: int = None) -> List[Dict]:
    """Load graph data with labels - optimized sequential version."""
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV with {len(df)} rows")

    age_col = "p21003_i2"
    sex_col = "p31"
    assessment_centre_col = "p54_i2"
    
    required_cols = [data_col, age_col, sex_col, assessment_centre_col]
    if use_head_motion:
        if head_motion_col is None:
            raise ValueError("head_motion_col must be provided when use_head_motion is True")
        required_cols.append(head_motion_col)
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        return []

    # OPTIMIZATION 1: Use vectorized dict creation instead of iterrows
    eid_to_data = df.set_index('eid')[required_cols].to_dict('index')
    
    # Rename keys to match expected format
    col_mapping = {data_col: 'label', age_col: 'age', sex_col: 'sex', 
                   assessment_centre_col: 'assessment_centre'}
    if use_head_motion:
        col_mapping[head_motion_col] = 'head_motion'
    
    eid_to_data = {
        eid: {col_mapping.get(k, k): v for k, v in data.items()}
        for eid, data in eid_to_data.items()
    }

    graph_files = [f for f in os.listdir(graph_dir) if f.endswith('_fc_graph.pt')]
    print(f"Found {len(graph_files)} graph files")
    
    matched_data = []
    unmatched_eids = []
    errors = []
    
    # OPTIMIZATION 2: Pre-compute full paths
    graph_dir_path = graph_dir  # avoid repeated string ops
    
    for i, file in enumerate(graph_files):
        try:
            eid = int(file.split('_', 1)[0])  # split once is faster
            
            if eid in eid_to_data:
                graph_path = os.path.join(graph_dir_path, file)
                graph_data = torch.load(graph_path, weights_only=False)
                
                row_data = eid_to_data[eid]
                graph_data.y = torch.tensor([row_data['label']], dtype=torch.float32)
                
                data_dict = {
                    'eid': eid,
                    'graph': graph_data,
                    'label': row_data['label'],
                    'age': row_data['age'],
                    'sex': row_data['sex'],
                    'assessment_centre': row_data['assessment_centre'],
                    'file': file
                }
                
                if use_head_motion:
                    data_dict['head_motion'] = row_data['head_motion']
                
                matched_data.append(data_dict)
            else:
                unmatched_eids.append(eid)
                
        except Exception as e:
            errors.append({'error': str(e), 'file': file})
        
        if (i + 1) % 10000 == 0:  # Less frequent logging
            print(f"  Loaded {i + 1}/{len(graph_files)} graph files...")
    
    print(f"Successfully matched {len(matched_data)} graphs with labels and demographics")
    if unmatched_eids:
        print(f"Warning: {len(unmatched_eids)} eids not found in CSV")
    if errors:
        print(f"Warning: {len(errors)} files had errors during loading")
        for err in errors[:5]:
            print(f"  {err}")
    
    return matched_data

def process_demographics(data_list, age_scaler, assessment_centre_encoder, 
                        head_motion_scaler=None, use_head_motion=False):
    processed_data = []

    for item in data_list:
        graph_data = item['graph']
        age_scaled = age_scaler.transform([[item['age']]])[0]
        
        # Convert sex to numeric (0 for Female, 1 for Male, or adjust as needed)
        sex_value = item['sex']
        if isinstance(sex_value, str):
            sex = [1.0 if sex_value.lower() == 'male' else 0.0]
        else:
            sex = [float(sex_value)]
        
        assessment_centre_onehot = assessment_centre_encoder.transform([[item['assessment_centre']]])[0]

        if use_head_motion and head_motion_scaler is not None:
            head_motion_scaled = head_motion_scaler.transform([[item['head_motion']]])[0]
            demo_features = np.concatenate([age_scaled, sex, assessment_centre_onehot, head_motion_scaled])
        else:
            demo_features = np.concatenate([age_scaled, sex, assessment_centre_onehot])
        
        graph_data.demographics = torch.tensor(demo_features, dtype=torch.float32).unsqueeze(0)

        processed_data.append(graph_data)

    return processed_data

def create_train_test_split(matched_data: List[Dict], train_ratio: float = 0.8, random_seed: int = 42,
                           use_head_motion: bool = False) -> Tuple[List, List, dict]:
    np.random.seed(random_seed)
    
    indices = np.random.permutation(len(matched_data))
    split_idx = int(len(matched_data) * train_ratio)
    
    train_raw = [matched_data[i] for i in indices[:split_idx]]
    test_raw = [matched_data[i] for i in indices[split_idx:]]

    print(f"Train set: {len(train_raw)} samples")
    print(f"Test set: {len(test_raw)} samples")

    train_ages = np.array([item['age'] for item in train_raw]).reshape(-1, 1)
    train_assessment_centres = np.array([item['assessment_centre'] for item in train_raw]).reshape(-1, 1)

    age_scaler = StandardScaler()
    age_scaler.fit(train_ages)

    assessment_centre_encoder = OneHotEncoder(drop='first', sparse_output=False)
    assessment_centre_encoder.fit(train_assessment_centres)

    print(f"Age scaler fitted - mean: {age_scaler.mean_[0]:.2f}, std: {age_scaler.scale_[0]:.2f}")
    print(f"Assessment centre encoder fitted - categories: {assessment_centre_encoder.categories_[0]}")

    preprocessors = {
        'age_scaler': age_scaler,
        'assessment_centre_encoder': assessment_centre_encoder
    }

    # Handle head motion if requested
    head_motion_scaler = None
    if use_head_motion:
        train_head_motion = np.array([item['head_motion'] for item in train_raw]).reshape(-1, 1)
        head_motion_scaler = StandardScaler()
        head_motion_scaler.fit(train_head_motion)
        preprocessors['head_motion_scaler'] = head_motion_scaler
        print(f"Head motion scaler fitted - mean: {head_motion_scaler.mean_[0]:.2f}, std: {head_motion_scaler.scale_[0]:.2f}")

    train_data = process_demographics(train_raw, age_scaler, assessment_centre_encoder, 
                                     head_motion_scaler, use_head_motion)
    test_data = process_demographics(test_raw, age_scaler, assessment_centre_encoder, 
                                    head_motion_scaler, use_head_motion)

    return train_data, test_data, preprocessors

def save_dataset_in_standard_format(train_data: List, test_data: List, output_dir: str, dataset_name: str,
                                   preprocessors: dict = None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    struct = {
        "geometric": {
            "gcn_train_data": train_data,
            "gcn_test_data": test_data,
            "sgcn_train_data": train_data,
            "sgcn_test_data": test_data
        }
    }

    if preprocessors:
        struct['preprocessing'] = preprocessors
    
    output_path = os.path.join(output_dir, f"{dataset_name}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(struct, f)
    
    print(f"Successfully saved dataset to {output_path}")

def process_custom_graph_dataset(data_col: str, graph_dir: str, csv_file: str, output_dir: str, 
                                dataset_name: str = "custom_graph_dataset", 
                                train_ratio: float = 0.8, random_seed: int = 42,
                                use_head_motion: bool = False, head_motion_col: str = None,
                                max_workers: int = None):
    
    matched_data = load_graph_data_with_labels(data_col, graph_dir, csv_file, 
                                               use_head_motion, head_motion_col, max_workers)
    
    if not matched_data:
        print("No matched data found. Please check your file paths and naming conventions.")
        return
    
    train_data, test_data, preprocessors = create_train_test_split(matched_data, train_ratio, random_seed,
                                                                   use_head_motion)
    
    num_train = len(train_data)
    num_test = len(test_data)
    dir_name = f"train_{num_train}_test_{num_test}"
    final_output_dir = os.path.join(output_dir, dir_name)
    
    save_dataset_in_standard_format(train_data, test_data, final_output_dir, dataset_name, preprocessors)
    
    return final_output_dir

if __name__ == "__main__":
    from config import (DATA_COL, GRAPH_DIR, CSV_FILE, DATASET_NAME, 
                       TRAIN_RATIO, RANDOM_SEED, USE_HEAD_MOTION, HEAD_MOTION_COL,
                       PROCESSED_DIR)

    output_dir = PROCESSED_DIR

    result_dir = process_custom_graph_dataset(
        data_col=DATA_COL,
        graph_dir=GRAPH_DIR,
        csv_file=CSV_FILE,
        output_dir=output_dir,
        dataset_name=DATASET_NAME,
        train_ratio=TRAIN_RATIO,
        random_seed=RANDOM_SEED,
        use_head_motion=USE_HEAD_MOTION,
        head_motion_col=HEAD_MOTION_COL
    )