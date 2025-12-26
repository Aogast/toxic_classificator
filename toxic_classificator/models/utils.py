"""
Utility functions for model training and inference
"""
import json
import torch
from typing import Dict, List, Any
from pathlib import Path


def load_json_data(file_path: Path) -> List[Dict]:
    """
    Load data from JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of data samples
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_data(data: List[Dict], file_path: Path):
    """
    Save data to JSON file
    
    Args:
        data: List of data samples
        file_path: Path to save JSON file
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_conversation(prompt: str, response: str = None) -> List[Dict[str, str]]:
    """
    Format conversation for chat models
    
    Args:
        prompt: User prompt
        response: Assistant response (optional)
        
    Returns:
        List of message dictionaries
    """
    messages = [{"role": "user", "content": prompt}]
    if response:
        messages.append({"role": "assistant", "content": response})
    return messages


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'trainable_percent': 100 * trainable_params / total_params if total_params > 0 else 0
    }


def print_trainable_parameters(model: torch.nn.Module):
    """
    Print trainable parameters information
    
    Args:
        model: PyTorch model
    """
    params = count_parameters(model)
    print(f"Trainable params: {params['trainable']:,} || "
          f"Total params: {params['total']:,} || "
          f"Trainable%: {params['trainable_percent']:.2f}%")

