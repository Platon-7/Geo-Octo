#!/usr/bin/env python3
"""
Script to help find and validate Octo checkpoint directory structure.
Run this to find the correct path for your checkpoint.
"""

import os
import sys
import glob

def find_checkpoint_directories(base_path):
    """Find potential checkpoint directories."""
    print(f"Searching for checkpoints in: {base_path}")
    
    # Common patterns for Octo checkpoints
    patterns = [
        "**/checkpoint",
        "**/default",
        "**/config.json",
        "**/*experiment*/**/checkpoint*",
        "**/*experiment*/**/default*"
    ]
    
    found_paths = set()
    
    for pattern in patterns:
        full_pattern = os.path.join(base_path, pattern)
        matches = glob.glob(full_pattern, recursive=True)
        for match in matches:
            if os.path.isdir(match):
                found_paths.add(match)
            elif match.endswith('config.json'):
                # If we find config.json, the directory containing it is likely the checkpoint
                found_paths.add(os.path.dirname(match))
    
    return sorted(found_paths)

def validate_checkpoint_directory(checkpoint_dir):
    """Validate that a directory contains the required Octo checkpoint files."""
    required_files = [
        'config.json',
        'example_batch.msgpack', 
        'dataset_statistics.json'
    ]
    
    print(f"\nValidating checkpoint directory: {checkpoint_dir}")
    
    if not os.path.isdir(checkpoint_dir):
        print("❌ Not a directory")
        return False
    
    print("📁 Directory contents:")
    try:
        contents = os.listdir(checkpoint_dir)
        for item in sorted(contents):
            item_path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(item_path):
                print(f"   📁 {item}/")
            else:
                print(f"   📄 {item}")
    except PermissionError:
        print("   ❌ Permission denied")
        return False
    
    print("\n🔍 Checking required files:")
    all_present = True
    for file in required_files:
        file_path = os.path.join(checkpoint_dir, file)
        if os.path.exists(file_path):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (missing)")
            all_present = False
    
    # Check for parameter files (usually .msgpack or .pkl files)
    param_files = [f for f in contents if f.endswith(('.msgpack', '.pkl')) and f != 'example_batch.msgpack']
    if param_files:
        print(f"\n📊 Parameter files found: {len(param_files)}")
        for pf in param_files[:3]:  # Show first 3
            print(f"   📄 {pf}")
        if len(param_files) > 3:
            print(f"   ... and {len(param_files) - 3} more")
    else:
        print("\n❌ No parameter files found")
        all_present = False
    
    return all_present

def main():
    print("="*60)
    print("OCTO CHECKPOINT FINDER AND VALIDATOR")
    print("="*60)
    
    # Try different possible base paths
    possible_base_paths = [
        "/home/pkarageorgis/geo_octo",
        "/home/pkarageorgis/geo_octo/octo",
        "/home/pkarageorgis",
        "."
    ]
    
    all_checkpoints = []
    
    for base_path in possible_base_paths:
        if os.path.exists(base_path):
            print(f"\n🔍 Searching in: {base_path}")
            checkpoints = find_checkpoint_directories(base_path)
            all_checkpoints.extend(checkpoints)
            
            if checkpoints:
                print(f"   Found {len(checkpoints)} potential checkpoint directories")
                for cp in checkpoints[:3]:  # Show first 3
                    print(f"   📁 {cp}")
                if len(checkpoints) > 3:
                    print(f"   ... and {len(checkpoints) - 3} more")
        else:
            print(f"\n❌ Path does not exist: {base_path}")
    
    if not all_checkpoints:
        print("\n❌ No checkpoint directories found!")
        print("\nTips:")
        print("1. Make sure you're running this from the correct directory")
        print("2. Check that the checkpoint path is accessible")
        print("3. Octo checkpoints should be directories, not files")
        return
    
    print(f"\n{'='*60}")
    print("VALIDATING CHECKPOINT DIRECTORIES")
    print("="*60)
    
    valid_checkpoints = []
    
    for checkpoint_dir in all_checkpoints:
        if validate_checkpoint_directory(checkpoint_dir):
            print("✅ Valid Octo checkpoint!")
            valid_checkpoints.append(checkpoint_dir)
        else:
            print("❌ Invalid or incomplete checkpoint")
        print("-" * 40)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    
    if valid_checkpoints:
        print(f"✅ Found {len(valid_checkpoints)} valid checkpoint(s):")
        for i, cp in enumerate(valid_checkpoints, 1):
            print(f"\n{i}. {cp}")
        
        print(f"\n🎯 RECOMMENDED: Use this path in your evaluation script:")
        print(f"MODEL_PATH = \"{valid_checkpoints[0]}\"")
        
        # Show how to update the script
        print(f"\n📝 To fix your evaluation script, change line ~37 to:")
        print(f'MODEL_PATH = "{valid_checkpoints[0]}"')
        
    else:
        print("❌ No valid Octo checkpoints found!")
        print("\nThe checkpoint should contain:")
        print("- config.json")
        print("- example_batch.msgpack") 
        print("- dataset_statistics.json")
        print("- Parameter files (.msgpack or .pkl)")

if __name__ == "__main__":
    main()