# import packages
import torch


GEMMA3_CONFIG = {
    "vocab_size": 50257,
    "context_length": 32768,
    "emb_dim": 640,
    "n_heads": 4,
    "n_layers": 18,
    "hid_dim": 2048,
    "head_dim": 256,
    "rope_local_base": 10000.0,
    "rope_base": 1000000.0,
    "sliding_window": 512,
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention"
    ],
    "dtype": torch.bfloat16
}