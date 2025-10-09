#!/usr/bin/env python3
"""
DIG (Discretized Integrated Gradients) implementation for BERT models.
Formatted similar to egrad_integral_bert function.
"""

import sys
import numpy as np
import torch
import random
import time
import inspect
from typing import Dict, List, Optional, Union

# Import DIG components
from dig import DiscretetizedIntegratedGradients
from attributions import run_dig_explanation
from metrics import eval_log_odds, eval_comprehensiveness, eval_sufficiency
import monotonic_paths

# Global cache for models
cache = {}

def dig_bert(
    text: str,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    dataset: str = "sst2",
    steps: int = 30,
    topk: int = 20,
    factor: int = 0,
    strategy: str = "greedy",
    knn_nbrs: int = 500,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    show_special_tokens: bool = False,
    seed: int = 42,
) -> Dict[str, Union[List[str], torch.Tensor, float]]:
    """
    DIG (Discretized Integrated Gradients) attribution analysis for BERT models.
    
    Args:
        text: Input text to analyze
        model_name: HuggingFace model name
        dataset: Dataset type for auxiliary data ('sst2', 'imdb', 'rotten')
        steps: Number of integration steps
        topk: Top-k percentage for evaluation metrics
        factor: Factor for path upscaling
        strategy: Path generation strategy ('greedy', 'maxcount')
        knn_nbrs: Number of KNN neighbors for auxiliary data
        device: Device to run on
        show_special_tokens: Whether to include special tokens in output
        seed: Random seed for reproducibility
        
    Returns:
        dict: {
            'tokens': List of tokens,
            'attributions': Attribution scores tensor,
            'logits': Model logits tensor,
            'log_odd': Log-odds metric,
            'comp': Comprehensiveness metric,
            'suff': Sufficiency metric,
            'predicted_label': Predicted class,
            'time': Computation time
        }
    """
    
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    start_time = time.perf_counter()
    
    # --- Load model/tokenizer and auxiliary data ---
    global cache
    cache_key = f"{model_name}_{dataset}"
    
    if cache.get(cache_key, None) is None:
        print(f"Model {model_name} not found in cache, loading from scratch")
        tmp = {}
        
        # Import model-specific functions based on model name
        if "distilbert" in model_name.lower():
            from distilbert_helper import nn_forward_func, nn_init, get_inputs, get_base_token_emb, get_word_embeddings, get_tokens, load_mappings
        elif "bert" in model_name.lower():
            from bert_helper import nn_forward_func, nn_init, get_inputs, get_base_token_emb, get_word_embeddings, get_tokens, load_mappings
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
        
        # Initialize model and tokenizer
        nn_init(device, dataset)
        
        # Load auxiliary data (KNN mappings)
        auxiliary_data = load_mappings(dataset, knn_nbrs=knn_nbrs)
        
        # Get base token embedding
        base_token_emb = get_base_token_emb(device)
        
        tmp["nn_forward_func"] = nn_forward_func
        tmp["get_inputs"] = get_inputs
        tmp["get_base_token_emb"] = get_base_token_emb
        tmp["get_tokens"] = get_tokens
        tmp["auxiliary_data"] = auxiliary_data
        tmp["base_token_emb"] = base_token_emb
        
        cache[cache_key] = tmp
    else:
        print(f"Using cached model {model_name}")
        nn_forward_func = cache[cache_key]["nn_forward_func"]
        get_inputs = cache[cache_key]["get_inputs"]
        get_base_token_emb = cache[cache_key]["get_base_token_emb"]
        get_tokens = cache[cache_key]["get_tokens"]
        auxiliary_data = cache[cache_key]["auxiliary_data"]
        base_token_emb = cache[cache_key]["base_token_emb"]
    
    # Initialize DIG attribution function
    attr_func = DiscretetizedIntegratedGradients(nn_forward_func)
    
    # Process the input sentence
    print(f"Processing sentence: '{text}'")
    inputs = get_inputs(text, device)
    
    # Extract components
    input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask = inputs
    
    # Get initial logits for the original input
    print("Computing initial logits...")
    initial_logits = nn_forward_func(input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True)
    print(f"Initial logits: {initial_logits}")
    predicted_label = int(torch.argmax(initial_logits).item())
    print(f"Predicted label: {predicted_label}")
    
    # Generate monotonic paths for DIG
    print("Generating monotonic paths...")
    scaled_features = monotonic_paths.scale_inputs(
        input_ids.squeeze().tolist(), 
        ref_input_ids.squeeze().tolist(),
        device, auxiliary_data, 
        steps=steps, 
        factor=factor, 
        strategy=strategy
    )
    
    # Run DIG attribution
    print("Computing DIG attributions...")
    attr = run_dig_explanation(
        attr_func, scaled_features, position_embed, type_embed, attention_mask, 
        (2**factor)*(steps+1)+1
    )
    
    # Compute evaluation metrics
    print("Computing evaluation metrics...")
    log_odd, pred = eval_log_odds(
        nn_forward_func, input_embed, position_embed, type_embed, attention_mask, 
        base_token_emb, attr, topk=topk
    )
    comp = eval_comprehensiveness(
        nn_forward_func, input_embed, position_embed, type_embed, attention_mask, 
        base_token_emb, attr, topk=topk
    )
    suff = eval_sufficiency(
        nn_forward_func, input_embed, position_embed, type_embed, attention_mask, 
        base_token_emb, attr, topk=topk
    )
    
    # Get tokens for interpretation
    tokens = get_tokens(input_ids)
    
    # Filter special tokens if requested
    if not show_special_tokens:
        # Keep only non-special tokens (typically positions 1 to -2)
        if len(tokens) > 2:
            tokens = tokens[1:-1]  # Remove [CLS] and [SEP]
            attr = attr[1:-1]
    
    end_time = time.perf_counter()
    
    return {
        "tokens": tokens,
        "attributions": attr.detach().cpu(),
        "logits": initial_logits.detach().cpu(),
        "log_odd": float(log_odd),
        "comp": float(comp),
        "suff": float(suff),
        "predicted_label": int(predicted_label),
        "time": end_time - start_time
    }


def main():
    """Example usage of dig_bert function."""
    
    # The sentence to analyze
    sentence = "This is a really bad movie, although it has a promising start, it ended on a very low note."
    
    try:
        # Run DIG analysis
        results = dig_bert(
            text=sentence,
            model_name="distilbert-base-uncased-finetuned-sst-2-english",
            dataset="sst2",
            steps=30,
            topk=20,
            factor=0,
            strategy="greedy",
            show_special_tokens=False
        )
        
        # Print results
        print(f"Logits: {results['logits']}")
        print(f"Attributions: {results['attributions']}")
        for tok, val in zip(results["tokens"], results["attributions"]): 
            print(f"{tok:>12s} : {val:+.6f}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
