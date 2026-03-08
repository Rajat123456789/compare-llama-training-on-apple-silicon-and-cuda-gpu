"""
Compare memory usage: LoRA vs Full Fine-tuning.

Provides memory benchmarks to demonstrate LoRA efficiency.
"""

import torch
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from config import (
    MODEL_ID,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    get_device,
    configure_mps,
)


def get_model_memory_mb(model):
    """Calculate model memory in MB."""
    total_params = sum(p.numel() for p in model.parameters())
    # Assume BF16 (2 bytes per param)
    memory_mb = (total_params * 2) / (1024 ** 2)
    return memory_mb, total_params


def get_process_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def profile_memory():
    """Profile memory usage for LoRA vs full fine-tuning."""
    configure_mps()
    device = get_device()
    
    print("\n" + "="*70)
    print("MEMORY PROFILING: LoRA vs Full Fine-Tuning")
    print("="*70)
    
    # Baseline memory
    baseline_mem = get_process_memory_mb()
    print(f"\nBaseline process memory: {baseline_mem:.1f} MB")
    
    # Load base model
    print(f"\nLoading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "mps":
        base_model = base_model.to("mps")
    
    base_mem, base_params = get_model_memory_mb(base_model)
    process_mem_base = get_process_memory_mb()
    
    print(f"Base model parameters: {base_params:,}")
    print(f"Base model memory:     {base_mem:.1f} MB")
    print(f"Process memory:        {process_mem_base:.1f} MB")
    
    # Count trainable params for full fine-tuning
    trainable_full = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    optimizer_mem_full = (trainable_full * 8) / (1024 ** 2)  # Adam: 2 states per param, 4 bytes each
    
    # Apply LoRA
    print(f"\nApplying LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(base_model, peft_config)
    
    lora_mem, lora_params = get_model_memory_mb(lora_model)
    process_mem_lora = get_process_memory_mb()
    
    trainable_lora = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    trainable_pct = (trainable_lora / base_params) * 100
    optimizer_mem_lora = (trainable_lora * 8) / (1024 ** 2)
    
    print(f"LoRA model parameters: {lora_params:,}")
    print(f"Trainable parameters:  {trainable_lora:,} ({trainable_pct:.2f}%)")
    print(f"LoRA model memory:     {lora_mem:.1f} MB")
    print(f"Process memory:        {process_mem_lora:.1f} MB")
    
    # Comparison table
    print("\n" + "="*70)
    print("COMPARISON: Full Fine-Tuning vs LoRA")
    print("="*70)
    print(f"{'Metric':<35} {'Full FT':>15} {'LoRA':>15}")
    print("-"*70)
    print(f"{'Trainable Parameters':<35} {trainable_full:>15,} {trainable_lora:>15,}")
    print(f"{'Trainable % of Total':<35} {100.0:>14.2f}% {trainable_pct:>14.2f}%")
    print(f"{'Model Memory (MB)':<35} {base_mem:>15.1f} {lora_mem:>15.1f}")
    print(f"{'Optimizer Memory Est. (MB)':<35} {optimizer_mem_full:>15.1f} {optimizer_mem_lora:>15.1f}")
    print(f"{'Total Training Memory Est. (MB)':<35} {base_mem + optimizer_mem_full:>15.1f} {lora_mem + optimizer_mem_lora:>15.1f}")
    
    # Savings
    param_reduction = ((trainable_full - trainable_lora) / trainable_full) * 100
    mem_reduction = ((optimizer_mem_full - optimizer_mem_lora) / optimizer_mem_full) * 100
    
    print("\n" + "="*70)
    print("SAVINGS WITH LoRA")
    print("="*70)
    print(f"Parameter reduction:  {param_reduction:.1f}%")
    print(f"Optimizer mem saved:  {mem_reduction:.1f}%")
    print(f"Memory saved:         ~{optimizer_mem_full - optimizer_mem_lora:.1f} MB")
    print("="*70)
    
    return {
        "full_ft_params": trainable_full,
        "lora_params": trainable_lora,
        "param_reduction_pct": param_reduction,
        "full_ft_mem": base_mem + optimizer_mem_full,
        "lora_mem": lora_mem + optimizer_mem_lora,
        "mem_reduction_pct": mem_reduction,
    }


if __name__ == "__main__":
    profile_memory()
