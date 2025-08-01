#!/usr/bin/env python3
"""
Simple memory checker using only built-in modules
"""

import resource
import gc
import os

def check_memory():
    """Check current memory usage"""
    gc.collect()
    
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(f"Peak memory usage: {usage.ru_maxrss / 1024:.1f} MB")
    
    # Check /proc/self/status for current usage
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    rss_kb = int(line.split()[1])
                    print(f"Current RSS: {rss_kb / 1024:.1f} MB ({rss_kb / 1024 / 1024:.2f} GB)")
                    break
    except:
        print("Could not read /proc/self/status")
    
    print(f"Python objects: {len(gc.get_objects())}")

def main():
    print("=== Simple Memory Check ===")
    check_memory()
    
    print("\n=== Creating some data ===")
    # Create some test data to see memory impact
    data = []
    for i in range(1000):
        data.append([0] * 1000)  # 1M integers
    
    check_memory()
    
    print("\n=== After garbage collection ===")
    del data
    gc.collect()
    check_memory()

if __name__ == "__main__":
    main()