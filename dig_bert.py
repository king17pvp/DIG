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
    """
    
    start_time = time.perf_counter()
    
    global cache
    cache_key = f"{model_name}_{dataset}"
    
    if cache.get(cache_key, None) is None:
        print(f"Model {model_name} not found in cache, loading from scratch")
        tmp = {}
        print(model_name, dataset)
        # Import model-specific functions based on model name
        if "distilbert" in model_name.lower():
            print(f"Using distilbert model")
            from distilbert_helper import nn_forward_func, nn_init, get_inputs, get_base_token_emb, get_word_embeddings, get_tokens, load_mappings
        elif "bert" in model_name.lower():
            from bert_helper import nn_forward_func, nn_init, get_inputs, get_base_token_emb, get_word_embeddings, get_tokens, load_mappings
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
        
        model, tokenizer = nn_init(device, dataset)
        auxiliary_data = load_mappings(dataset, knn_nbrs=knn_nbrs)
        base_token_emb = get_base_token_emb(model, tokenizer, device)
        
        tmp["model"] = model
        tmp["tokenizer"] = tokenizer
        tmp["nn_forward_func"] = nn_forward_func
        tmp["get_inputs"] = get_inputs
        tmp["get_base_token_emb"] = get_base_token_emb
        tmp["get_tokens"] = get_tokens
        tmp["auxiliary_data"] = auxiliary_data
        tmp["base_token_emb"] = base_token_emb
        
        cache[cache_key] = tmp
    else:
        print(f"Using cached model {model_name}")
        model = cache[cache_key]["model"]
        tokenizer = cache[cache_key]["tokenizer"]
        nn_forward_func = cache[cache_key]["nn_forward_func"]
        get_inputs = cache[cache_key]["get_inputs"]
        get_base_token_emb = cache[cache_key]["get_base_token_emb"]
        get_tokens = cache[cache_key]["get_tokens"]
        auxiliary_data = cache[cache_key]["auxiliary_data"]
        base_token_emb = cache[cache_key]["base_token_emb"]
    
    attr_func = DiscretetizedIntegratedGradients(lambda *args, **kwargs: nn_forward_func(model, *args, **kwargs))
    inputs = get_inputs(model, tokenizer, text, device)
    
    input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask = inputs
    initial_logits = model(inputs_embeds=input_embed, attention_mask=attention_mask).logits[0]
    print("Generating monotonic paths...")
    scaled_features = monotonic_paths.scale_inputs(
        input_ids.squeeze().tolist(), 
        ref_input_ids.squeeze().tolist(),
        device, auxiliary_data, 
        steps=steps, 
        factor=factor, 
        strategy=strategy
    )
    
    attr = run_dig_explanation(
        attr_func, scaled_features, position_embed, type_embed, attention_mask, 
        (2**factor)*(steps+1)+1
    )
    
    log_odd, pred = eval_log_odds(
        lambda *args, **kwargs: nn_forward_func(model, *args, **kwargs), input_embed, position_embed, type_embed, attention_mask, 
        base_token_emb, attr, topk=topk
    )
    comp = eval_comprehensiveness(
        lambda *args, **kwargs: nn_forward_func(model, *args, **kwargs), input_embed, position_embed, type_embed, attention_mask, 
        base_token_emb, attr, topk=topk
    )
    suff = eval_sufficiency(
        lambda *args, **kwargs: nn_forward_func(model, *args, **kwargs), input_embed, position_embed, type_embed, attention_mask, 
        base_token_emb, attr, topk=topk
    )
    
    # Get tokens for interpretation
    tokens = get_tokens(tokenizer, input_ids)
    
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
        "time": end_time - start_time
    }


def main():
    """Example usage of dig_bert function."""
    sentence = "This is a really bad movie, although it has a promising start, it ended on a very low note"
    res_dig = dig_bert(
        text=sentence,
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        dataset="sst2",
        steps=30,
        topk=20,
        factor=1,
        strategy="greedy",
        show_special_tokens=False
    )
    
    print(f"Log odds: {res_dig['log_odd']}")
    print(f"Comprehensiveness: {res_dig['comp']}")
    print(f"Sufficiency: {res_dig['suff']}")
    for tok, val in zip(res_dig["tokens"], res_dig["attributions"]): 
        print(f"{tok:>12s} : {val:+.6f}")

if __name__ == "__main__":
    main()