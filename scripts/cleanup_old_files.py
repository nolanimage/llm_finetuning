#!/usr/bin/env python3
"""
Cleanup script to remove old experimental files and organize the project.
This script helps clean up the messy file structure.
"""

import os
import shutil
from pathlib import Path
import argparse

# Files and directories to remove
OLD_TRAIN_SCRIPTS = [
    'train_working_mps.py',
    'train_working_mps_fast.py',
    'train_working_mps_2epochs.py',
    'train_working_mps_fixed_lr.py',
    'train_working_mps_ultimate.py',
    'train_working_mps_ultimate_safe.py',
    'train_working_mps_ultimate_proven.py',
    'train_working_mps_ultimate_conservative.py',
    'train_working_mps_ultimate_super_conservative.py',
    'train_working_mps_constant_lr.py',
    'train_working_mps_fastest.py',
    'train_working_mps_high_memory.py',
    'train_working_mps.py',
    'train_working.py',
    'train_simple_cpu.py',
    'train_simple_fast.py',
    'train_mps_fixed.py',
    'train_fast.py',
    'train_5000_articles.py',
    'train_batch2.py',
    'train_test_100_samples.py',
    'train_test_1000_samples.py',
    'train_chatglm3_1.5b_fast.py',
    'continue_from_batch2.py',
]

OLD_SCRIPTS = [
    'cleanup.py',
    'cleanup.sh',
    'quick_clean.sh',
    'extract_wikipedia_simple.py',
    'prepare_wikipedia_data.py',
    'debug_model_structure.py',
    'model_size_comparison.py',
    'test_lora.py',
]

OLD_DIRS = [
    'results_working_mps',
    'results_working_mps_fast',
    'results_working_mps_fastest',
    'results_working_mps_fixed_lr',
    'results_working_mps_high_memory',
    'results_working_mps_ultimate',
    'results_working_mps_ultimate_conservative',
    'results_working_mps_ultimate_proven',
    'results_working_mps_ultimate_super_conservative',
    'results_test_100_samples',
    'results_test_1000_samples',
    'results_chinese_glm_5000',
    'results_chinese_glm_fixed',
    'results_chinese_glm_mps_10epochs',
    'results_chinese_glm_mps_fast',
    'results_chinese_glm_optimized',
    'results_chinese_glm_optimized_fresh',
    'results_chinese_glm_test',
    'chinese_glm_finetuned_batch2',
    'chinese_glm_finetuned_clean',
    'chinese_glm_finetuned_mps_fast',
    'chinese_glm_finetuned_simple',
    'chinese_glm_working_mps',
    'logs_working_mps',
    'logs_working_mps_fast',
    'logs_working_mps_fastest',
    'logs_working_mps_fixed_lr',
    'logs_working_mps_high_memory',
    'logs_working_mps_ultimate',
    'logs_working_mps_ultimate_conservative',
    'logs_working_mps_ultimate_proven',
    'logs_working_mps_ultimate_super_conservative',
    'logs_chinese_glm_fixed',
    'logs_chinese_glm_mps_10epochs',
    'logs_chinese_glm_mps_fast',
    'logs_chinese_glm_optimized',
    'logs_chinese_glm_optimized_fresh',
]


def cleanup_files(dry_run=False):
    """Remove old experimental files."""
    removed = []
    errors = []
    
    # Remove old train scripts
    for file in OLD_TRAIN_SCRIPTS + OLD_SCRIPTS:
        path = Path(file)
        if path.exists():
            if dry_run:
                print(f"[DRY RUN] Would remove: {file}")
            else:
                try:
                    path.unlink()
                    removed.append(file)
                    print(f"✓ Removed: {file}")
                except Exception as e:
                    errors.append((file, str(e)))
                    print(f"✗ Error removing {file}: {e}")
    
    return removed, errors


def cleanup_directories(dry_run=False, keep_outputs=False):
    """Remove old result and log directories."""
    removed = []
    errors = []
    
    for dir_name in OLD_DIRS:
        path = Path(dir_name)
        if path.exists() and path.is_dir():
            if keep_outputs and 'results' in dir_name:
                print(f"⊘ Skipping (keep_outputs=True): {dir_name}")
                continue
            
            if dry_run:
                print(f"[DRY RUN] Would remove: {dir_name}")
            else:
                try:
                    shutil.rmtree(path)
                    removed.append(dir_name)
                    print(f"✓ Removed directory: {dir_name}")
                except Exception as e:
                    errors.append((dir_name, str(e)))
                    print(f"✗ Error removing {dir_name}: {e}")
    
    return removed, errors


def main():
    parser = argparse.ArgumentParser(
        description='Cleanup old experimental files and directories'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without actually removing'
    )
    parser.add_argument(
        '--keep-outputs',
        action='store_true',
        help='Keep result directories (only remove log directories)'
    )
    parser.add_argument(
        '--files-only',
        action='store_true',
        help='Only remove files, not directories'
    )
    parser.add_argument(
        '--dirs-only',
        action='store_true',
        help='Only remove directories, not files'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cleanup Script - Removing Old Experimental Files")
    print("=" * 60)
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be deleted\n")
    
    removed_files = []
    removed_dirs = []
    errors = []
    
    # Cleanup files
    if not args.dirs_only:
        print("\n📄 Cleaning up old files...")
        files, file_errors = cleanup_files(dry_run=args.dry_run)
        removed_files.extend(files)
        errors.extend(file_errors)
    
    # Cleanup directories
    if not args.files_only:
        print("\n📁 Cleaning up old directories...")
        dirs, dir_errors = cleanup_directories(
            dry_run=args.dry_run,
            keep_outputs=args.keep_outputs
        )
        removed_dirs.extend(dirs)
        errors.extend(dir_errors)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Files removed: {len(removed_files)}")
    print(f"Directories removed: {len(removed_dirs)}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\n⚠️  Errors encountered:")
        for item, error in errors:
            print(f"  - {item}: {error}")
    
    if args.dry_run:
        print("\n💡 Run without --dry-run to actually remove files")
    else:
        print("\n✅ Cleanup complete!")


if __name__ == "__main__":
    main()
