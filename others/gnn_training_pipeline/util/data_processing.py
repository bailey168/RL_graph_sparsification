"""Data processing utilities for graph datasets."""
import os
import pickle


def read_cached_graph_dataset(num_train, num_test, dataset_name, parent_dir=None, 
                              sparsity=100, dataset_type='FC25'):
    """
    Load cached graph dataset from disk.
    
    Args:
        num_train: Number of training samples
        num_test: Number of test samples
        dataset_name: Name of the dataset
        parent_dir: Parent directory containing datasets
        sparsity: Sparsity level of the graphs
        dataset_type: Type of dataset ('FC25', 'FC100', or 'MIND')
    
    Returns:
        Dictionary containing graph data for training and testing
    """
    assert num_train is not None and isinstance(num_train, int)
    assert num_test is not None and isinstance(num_test, int)
    
    # Build path to dataset
    curr_path = parent_dir or os.path.dirname(os.path.realpath('__file__'))
    base_path = os.path.join(curr_path, f"{dataset_type}_Dataset{sparsity}", "RawGraph")
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"Dataset directory not found: {base_path}\n"
            f"Dataset type: {dataset_type}, Sparsity: {sparsity}"
        )
    
    dir_name = f"train_{num_train}_test_{num_test}"
    full_dirpath = os.path.join(base_path, dir_name)
    
    if not os.path.exists(full_dirpath):
        raise FileNotFoundError(
            f"Dataset split not found: {full_dirpath}\n"
            f"num_train={num_train}, num_test={num_test}"
        )
    
    full_path = os.path.join(full_dirpath, f"{dataset_name}.pkl")
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Dataset file not found: {full_path}")
    
    print(f"Loading dataset from: {full_path}")
    
    with open(full_path, "rb") as f:
        data = pickle.load(f)
    
    return data
