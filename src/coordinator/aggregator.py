import numpy as np
import torch


def federated_average(model_updates: list) -> dict:
    """
    Implements the Federated Averaging algorithm.
    
    Args:
        model_updates: List of dicts, each containing:
            - 'platform_id': int
            - 'model_state': dict (keys are layer names, values are weight lists)
            - 'n_samples': int
            - 'local_rounds': int
            - 'model_version': int
    
    Returns:
        Aggregated model_state dict with weighted averaged weights.
    """
    if not model_updates:
        return {}
    
    # Calculate total samples
    total_samples = sum(update['n_samples'] for update in model_updates)
    
    if total_samples == 0:
        return {}
    
    # Get layer names from first update
    layer_names = model_updates[0]['model_state'].keys()
    
    # Initialize aggregated state
    aggregated_state = {}
    
    # For each layer, calculate weighted average
    for layer_name in layer_names:
        weighted_sum = None
        
        for update in model_updates:
            weight = update['n_samples'] / total_samples
            layer_weights = np.array(update['model_state'][layer_name])
            
            if weighted_sum is None:
                weighted_sum = weight * layer_weights
            else:
                weighted_sum += weight * layer_weights
        
        aggregated_state[layer_name] = weighted_sum.tolist()
    
    return aggregated_state


def deserialize_model_state(model_state: dict) -> dict:
    """
    Convert lists back to torch tensors.
    
    Args:
        model_state: Dict with layer names as keys and weight lists as values.
    
    Returns:
        Dict with layer names as keys and torch.FloatTensor as values.
    """
    tensor_state = {}
    
    for key in model_state:
        numpy_array = np.array(model_state[key])
        tensor_state[key] = torch.FloatTensor(numpy_array)
    
    return tensor_state


def serialize_model_state(model_state: dict) -> dict:
    """
    Convert torch tensors to lists.
    
    Args:
        model_state: Dict with layer names as keys and torch tensors as values.
    
    Returns:
        Dict with layer names as keys and lists as values.
    """
    serialized_state = {}
    
    for key in model_state:
        serialized_state[key] = model_state[key].cpu().numpy().tolist()
    
    return serialized_state
