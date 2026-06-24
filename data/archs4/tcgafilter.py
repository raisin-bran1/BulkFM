#!/usr/bin/env python3
import argparse
import os
import h5py
import numpy as np

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Filter out TCGA data leakage from an ARCHS4 human expression matrix HDF5 file."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to the ARCHS4 human_matrix.h5 file"
    )
    parser.add_argument(
        "-o", "--output",
        default="clean_indices.npy",
        type=str,
        help="Output path to save the clean sample indices array (default: clean_indices.npy)"
    )
    
    args = parser.parse_args()

    # Validate file existence
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Could not find the specified H5 file at: {args.input}")

    print(f"Opening ARCHS4 file: {args.input}...")
    
    with h5py.File(args.input, "r") as f:
        # Extract metadata text arrays and decode from byte strings
        print("Extracting metadata arrays...")
        series_ids = np.array([s.decode('utf-8') for s in f["meta/samples/series_id"][:]])
        source_names = np.array([s.decode('utf-8').lower() for s in f["meta/samples/source_name_ch1"][:]])
        characteristics = np.array([s.decode('utf-8').lower() for s in f["meta/samples/characteristics_ch1"][:]])

    # Define known blacklisted GEO series
    blacklisted_series = {"GSE62944", "GSE62945"}

    print("Scanning for TCGA signatures...")
    # Mask 1: Look for specific TCGA-associated GEO series numbers
    series_mask = np.array([sid in blacklisted_series for sid in series_ids])

    # Mask 2: Catch any isolated uploads mentioning "tcga" or "the cancer genome atlas"
    text_mask = np.array([
        ("tcga" in src) or ("the cancer genome atlas" in src) or ("tcga" in char)
        for src, char in zip(source_names, characteristics)
    ])

    # Combine masks (True for contaminated samples)
    contaminated_mask = series_mask | text_mask

    # Invert mask to get the keep indices
    clean_indices = np.where(~contaminated_mask)[0]

    # Quick report to terminal
    total_samples = len(contaminated_mask)
    dropped_count = contaminated_mask.sum()
    
    print("\n--- Filtering Summary ---")
    print(f"Total initial samples: {total_samples:,}")
    print(f"TCGA leak samples dropped: {dropped_count:,} ({dropped_count/total_samples:.2%})")
    print(f"Clean samples remaining:   {len(clean_indices):,}")
    print("-------------------------\n")

    # Save the numpy index file
    np.save(args.output, clean_indices)
    print(f"Successfully saved clean index mask to: {args.output}")

if __name__ == "__main__":
    main()