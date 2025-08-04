#!/usr/bin/env python3
"""
Performance tuning script for Octo memory optimizations.
This helps you find the optimal prefetch and dataset settings.
"""

import psutil

# Import from same directory
from optimize_memory import (
    get_memory_usage_gb,
    get_recommended_prefetch_size,
    adjust_prefetch_for_performance
)

def analyze_your_system():
    """Analyze system and provide personalized recommendations."""
    
    print("=== System Analysis for Octo Training ===")
    
    # System specs
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    current_usage_gb = get_memory_usage_gb()
    
    print(f"💾 Total RAM: {total_memory_gb:.0f}GB")
    print(f"📊 Available RAM: {available_memory_gb:.0f}GB") 
    print(f"🔍 Current usage: {current_usage_gb:.0f}GB")
    
    # Calculate safe memory budget for training
    safe_memory_budget = available_memory_gb * 0.6  # Leave 40% buffer
    print(f"🎯 Safe memory budget for training: {safe_memory_budget:.0f}GB")
    
    print("\n=== Recommendations ===")
    
    # Scenario analysis
    if available_memory_gb >= 300:
        print("🚀 HIGH MEMORY SYSTEM")
        print("   • prefetch: 4-6 (you can use higher)")
        print("   • max_datasets: 4 (use all datasets)")
        print("   • shuffle_buffer: 20-50 (more randomness)")
        rec_prefetch = get_recommended_prefetch_size(available_memory_gb, 8, 4)
        rec_datasets = 4
        
    elif available_memory_gb >= 150:
        print("⚖️ MEDIUM MEMORY SYSTEM") 
        print("   • prefetch: 3-4 (moderate)")
        print("   • max_datasets: 3 (most datasets)")
        print("   • shuffle_buffer: 10-20 (balanced)")
        rec_prefetch = get_recommended_prefetch_size(available_memory_gb, 8, 3)
        rec_datasets = 3
        
    else:
        print("🔧 MEMORY-CONSTRAINED SYSTEM")
        print("   • prefetch: 2 (conservative - what I set)")
        print("   • max_datasets: 2 (what I set)")
        print("   • shuffle_buffer: 5-10 (minimal)")
        rec_prefetch = 2
        rec_datasets = 2
    
    print(f"\n📝 SPECIFIC RECOMMENDATIONS FOR YOUR SYSTEM:")
    print(f"   • Recommended prefetch: {rec_prefetch}")
    print(f"   • Recommended max_datasets: {rec_datasets}")
    
    # Generate config snippet
    print(f"\n🔧 TO ADJUST YOUR SETTINGS:")
    print("1. In finetune.py, change:")
    print(f"   .prefetch({rec_prefetch})  # instead of prefetch(2)")
    
    print("2. In optimize_memory.py, change:")
    print(f"   max_datasets={rec_datasets}  # instead of max_datasets=2")
    
    if available_memory_gb >= 200:
        print("3. In config_offline.py, you could increase:")
        print("   shuffle_buffer_size=20  # instead of 10")
        print("   val_shuffle_buffer_size=10  # instead of 5")
    
    return rec_prefetch, rec_datasets

def estimate_memory_usage(batch_size, num_datasets, prefetch_size, shuffle_buffer):
    """Estimate memory usage with given settings."""
    
    # Rough estimates based on typical Octo training
    batch_memory_gb = batch_size * 0.3  # ~300MB per batch size unit
    
    # Memory components
    prefetch_memory = batch_memory_gb * prefetch_size * num_datasets
    shuffle_memory = batch_memory_gb * shuffle_buffer * num_datasets  
    model_memory = 15  # Approximate model + optimizer memory
    system_overhead = 10  # OS + other processes
    
    total_estimated = prefetch_memory + shuffle_memory + model_memory + system_overhead
    
    print(f"\n📊 MEMORY USAGE ESTIMATE:")
    print(f"   • Prefetch memory: {prefetch_memory:.1f}GB")
    print(f"   • Shuffle buffer memory: {shuffle_memory:.1f}GB") 
    print(f"   • Model + optimizer: {model_memory:.1f}GB")
    print(f"   • System overhead: {system_overhead:.1f}GB")
    print(f"   • TOTAL ESTIMATED: {total_estimated:.1f}GB")
    
    return total_estimated

def compare_settings():
    """Compare different setting combinations."""
    
    print("\n=== SETTING COMPARISONS ===")
    
    scenarios = [
        ("Conservative (current)", 8, 2, 2, 10),
        ("Balanced", 8, 3, 4, 15), 
        ("Aggressive", 8, 4, 6, 25),
        ("Your original", 8, 4, "∞", 100),
    ]
    
    print("Setting              | Est.Memory | Performance | Risk")
    print("-" * 55)
    
    for name, batch_size, datasets, prefetch, shuffle in scenarios:
        if prefetch == "∞":
            memory = "450GB+"
            performance = "High"
            risk = "OOM"
        else:
            memory = f"{estimate_memory_usage(batch_size, datasets, prefetch, shuffle):.0f}GB"
            
            if prefetch >= 4:
                performance = "High"
            elif prefetch >= 3:
                performance = "Medium"
            else:
                performance = "Lower"
                
            if prefetch <= 2 and datasets <= 2:
                risk = "Low"
            elif prefetch <= 4 and datasets <= 3:
                risk = "Medium"  
            else:
                risk = "High"
        
        print(f"{name:<20} | {memory:>9} | {performance:>11} | {risk}")

if __name__ == "__main__":
    analyze_your_system()
    compare_settings()
    
    print("\n💡 KEY INSIGHT:")
    print("I chose prefetch(2) and max_datasets=2 as a SAFE STARTING POINT")
    print("given your 450GB memory crisis. You can increase them based on")
    print("your system capacity and performance needs!")