#!/usr/bin/env python3
"""
Batch quantify LFB with heatmap generation.

Searches subdirectories for:
  - mask_intact_tissue.png (or .npy) in intact_analysis/
  - .tiff file in L0_T256_S256/

For each found pair:
  - Quantify LFB in masked regions
  - Generate heatmap (brighter LFB = hotter colors)
  - Save results

Usage:
    python batch_lfb_heatmap.py \
        --input_dir ./phas_clam_outputs \
        --output_dir ./lfb_results
"""

import os
import glob
import json
import argparse
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def find_slide_pairs(input_dir: str) -> list:
    """
    Find pairs of mask and TIFF files in subdirectories.
    
    Returns:
        List of (slide_name, mask_path, tiff_path) tuples
    """
    pairs = []
    
    # Search for slide_* subdirectories
    slide_dirs = glob.glob(os.path.join(input_dir, "slide_*"))
    
    for slide_dir in slide_dirs:
        slide_name = os.path.basename(slide_dir)
        
        # Look for L0_T256_S256 directory
        l0_dir = os.path.join(slide_dir, "L0_T256_S256")
        
        if not os.path.exists(l0_dir):
            continue
        
        # Find mask file in intact_analysis
        mask_path = None
        intact_dirs = glob.glob(os.path.join(l0_dir, "intact_analysis*"))
        
        for intact_dir in intact_dirs:
            # Try PNG first
            mask_png = os.path.join(intact_dir, "mask_intact_tissue.png")
            if os.path.exists(mask_png):
                mask_path = mask_png
                break
            
            # Try NPY
            mask_npy = os.path.join(intact_dir, "mask_intact_tissue.npy")
            if os.path.exists(mask_npy):
                mask_path = mask_npy
                break
        
        if mask_path is None:
            print(f"  ✗ {slide_name}: No mask found")
            continue
        
        # Find TIFF file in L0_T256_S256
        tiff_files = glob.glob(os.path.join(l0_dir, "*.tiff")) + \
                     glob.glob(os.path.join(l0_dir, "*.tif"))
        
        if len(tiff_files) == 0:
            print(f"  ✗ {slide_name}: No TIFF found")
            continue
        
        # Use first TIFF found
        tiff_path = tiff_files[0]
        
        pairs.append((slide_name, mask_path, tiff_path))
        print(f"  ✓ {slide_name}")
        print(f"      Mask: {os.path.basename(mask_path)}")
        print(f"      TIFF: {os.path.basename(tiff_path)}")
    
    return pairs


def load_mask(mask_path: str) -> np.ndarray:
    """Load mask from PNG or NPY file."""
    ext = Path(mask_path).suffix.lower()
    
    if ext == '.npy':
        mask = np.load(mask_path)
        # Convert boolean to uint8
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask from {mask_path}")
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return mask


def load_tiff(tiff_path: str) -> np.ndarray:
    """Load TIFF image."""
    try:
        import tifffile
        img = tifffile.imread(tiff_path)
        
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:
            img = img[:, :, :3]
        
        return img
    except:
        # Fallback to OpenCV
        img = cv2.imread(tiff_path)
        if img is None:
            raise FileNotFoundError(f"Could not load TIFF from {tiff_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def lfb_color_deconvolution(rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate LFB and CV stains using color deconvolution.
    
    Returns:
        lfb_channel: LFB-deconvolved channel (uint8)
        cv_channel: CV-deconvolved channel (uint8)
    """
    # Normalize to [0, 1]
    rgb_norm = rgb_image.astype(np.float32) / 255.0
    
    # Add epsilon to avoid log(0)
    epsilon = 1e-6
    rgb_norm = np.maximum(rgb_norm, epsilon)
    
    # Convert to optical density
    od = -np.log10(rgb_norm)
    
    # LFB-CV stain matrix
    stain_matrix = np.array([
        [0.20, 0.80, 0.60],  # LFB (blue/cyan)
        [0.80, 0.10, 0.50],  # CV (purple/magenta)
        [0.40, 0.40, 0.20]   # Background
    ]).T
    
    # Normalize columns
    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=0)
    
    # Reshape for matrix operations
    od_flat = od.reshape(-1, 3)
    
    # Solve for concentrations
    concentrations = od_flat @ np.linalg.pinv(stain_matrix.T)
    
    # Reshape back
    concentrations = concentrations.reshape(rgb_image.shape[0], rgb_image.shape[1], 3)
    
    # Extract LFB and CV channels
    lfb_channel = concentrations[:, :, 0]
    cv_channel = concentrations[:, :, 1]
    
    # Normalize to [0, 255]
    lfb_channel = np.clip(lfb_channel * 255 / (np.max(lfb_channel) + epsilon), 0, 255).astype(np.uint8)
    cv_channel = np.clip(cv_channel * 255 / (np.max(cv_channel) + epsilon), 0, 255).astype(np.uint8)
    
    return lfb_channel, cv_channel


def resize_mask_to_image(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize mask to match image dimensions."""
    h, w = target_shape
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mask_resized = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
    return mask_resized


def quantify_masked_lfb(lfb_channel: np.ndarray,
                       mask: np.ndarray,
                       threshold: int = 30) -> Dict:
    """Quantify LFB in masked regions."""
    masked_region = mask > 0
    lfb_masked = lfb_channel[masked_region]
    
    if len(lfb_masked) == 0:
        return {
            "n_pixels_total": 0,
            "n_pixels_positive": 0,
            "percent_positive": 0.0,
            "mean_intensity": 0.0,
            "median_intensity": 0.0,
            "std_intensity": 0.0
        }
    
    positive_pixels = lfb_masked > threshold
    
    return {
        "n_pixels_total": int(len(lfb_masked)),
        "n_pixels_positive": int(np.sum(positive_pixels)),
        "percent_positive": float(100 * np.sum(positive_pixels) / len(lfb_masked)),
        "mean_intensity": float(np.mean(lfb_masked)),
        "median_intensity": float(np.median(lfb_masked)),
        "std_intensity": float(np.std(lfb_masked)),
        "sum_intensity": float(np.sum(lfb_masked)),
        "max_intensity": float(np.max(lfb_masked)),
        "min_intensity": float(np.min(lfb_masked)),
        "threshold_used": int(threshold)
    }


def create_lfb_heatmap(lfb_channel: np.ndarray,
                      mask: np.ndarray,
                      output_path: str,
                      title: str = "LFB Heatmap"):
    """
    Create heatmap where brighter LFB = hotter colors.
    Only shows masked regions.
    
    Args:
        lfb_channel: LFB intensity values
        mask: Binary mask (255 = show, 0 = hide)
        output_path: Where to save heatmap
        title: Title for the plot
    """
    # Create masked LFB (set non-masked areas to NaN for transparency)
    lfb_masked = lfb_channel.astype(float).copy()
    lfb_masked[mask == 0] = np.nan
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap using 'hot' colormap (dark = low, bright = high)
    # Or use 'jet' for classic rainbow
    im = ax.imshow(lfb_masked, cmap='hot', interpolation='nearest', vmin=0, vmax=255)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('LFB Intensity (Brighter = More Myelin)', rotation=270, labelpad=20)
    
    # Title and labels
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_composite_visualization(rgb_image: np.ndarray,
                                   lfb_channel: np.ndarray,
                                   mask: np.ndarray,
                                   output_path: str,
                                   title: str = "LFB Analysis"):
    """
    Create composite visualization with:
    - Original RGB
    - Masked LFB
    - LFB heatmap
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(rgb_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Masked LFB (grayscale)
    lfb_display = lfb_channel.copy()
    lfb_display[mask == 0] = 0
    axes[1].imshow(lfb_display, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('LFB Channel (Masked)')
    axes[1].axis('off')
    
    # LFB heatmap
    lfb_heat = lfb_channel.astype(float).copy()
    lfb_heat[mask == 0] = np.nan
    im = axes[2].imshow(lfb_heat, cmap='hot', interpolation='nearest', vmin=0, vmax=255)
    axes[2].set_title('LFB Heatmap (Hot = High)')
    axes[2].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('LFB Intensity', rotation=270, labelpad=15)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_slide(slide_name: str,
                 mask_path: str,
                 tiff_path: str,
                 output_dir: str,
                 threshold: int = 30) -> Dict:
    """Process a single slide."""
    
    print(f"\n{'='*80}")
    print(f"Processing: {slide_name}")
    print(f"{'='*80}")
    
    result = {
        "slide_name": slide_name,
        "status": "unknown",
        "message": ""
    }
    
    try:
        # Create slide output directory
        slide_output_dir = os.path.join(output_dir, slide_name)
        os.makedirs(slide_output_dir, exist_ok=True)
        
        # Load mask
        print("  Loading mask...")
        mask = load_mask(mask_path)
        print(f"    Mask shape: {mask.shape}")
        print(f"    Masked pixels: {np.sum(mask > 0):,} ({100*np.sum(mask>0)/mask.size:.2f}%)")
        
        # Load TIFF
        print("  Loading TIFF...")
        rgb_image = load_tiff(tiff_path)
        print(f"    Image shape: {rgb_image.shape}")
        
        # Resize mask to match image
        print("  Resizing mask to match image...")
        mask_resized = resize_mask_to_image(mask, rgb_image.shape[:2])
        
        # Color deconvolution
        print("  Performing color deconvolution...")
        lfb_channel, cv_channel = lfb_color_deconvolution(rgb_image)
        print(f"    LFB range: {lfb_channel.min()} - {lfb_channel.max()}")
        
        # Quantify LFB in masked regions
        print("  Quantifying LFB in masked regions...")
        metrics = quantify_masked_lfb(lfb_channel, mask_resized, threshold)
        print(f"    LFB-positive: {metrics['percent_positive']:.2f}%")
        print(f"    Mean intensity: {metrics['mean_intensity']:.2f}")
        
        # Save masked LFB channel
        lfb_masked_display = lfb_channel.copy()
        lfb_masked_display[mask_resized == 0] = 0
        lfb_masked_path = os.path.join(slide_output_dir, "lfb_masked.png")
        cv2.imwrite(lfb_masked_path, lfb_masked_display)
        
        # Create heatmap
        print("  Creating heatmap...")
        heatmap_path = os.path.join(slide_output_dir, "lfb_heatmap.png")
        create_lfb_heatmap(lfb_channel, mask_resized, heatmap_path, 
                          title=f"{slide_name} - LFB Heatmap")
        
        # Create composite visualization
        print("  Creating composite visualization...")
        composite_path = os.path.join(slide_output_dir, "composite.png")
        create_composite_visualization(rgb_image, lfb_channel, mask_resized,
                                      composite_path, title=slide_name)
        
        # Save results JSON
        results_json = {
            "slide_name": slide_name,
            "mask_path": mask_path,
            "tiff_path": tiff_path,
            "mask_shape": list(mask.shape),
            "image_shape": list(rgb_image.shape),
            "quantification": metrics
        }
        
        json_path = os.path.join(slide_output_dir, "results.json")
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        result["status"] = "success"
        result["message"] = "Complete"
        result["metrics"] = metrics
        result["output_dir"] = slide_output_dir
        
        print(f"  ✓ Saved results to {slide_output_dir}")
        
    except Exception as e:
        result["status"] = "failed"
        result["message"] = str(e)
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def consolidate_results(results: list, output_dir: str):
    """Create consolidated CSV of all results."""
    rows = []
    
    for result in results:
        if result["status"] == "success" and "metrics" in result:
            row = {
                "slide_name": result["slide_name"],
                "status": result["status"],
                **result["metrics"]
            }
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "all_quantifications.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Consolidated CSV saved to {csv_path}")
        return df
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Batch LFB quantification with heatmap generation"
    )
    
    parser.add_argument("--input_dir", required=True,
                       help="Input directory containing slide_* subdirectories")
    
    parser.add_argument("--output_dir", default="./lfb_results",
                       help="Output directory for results")
    
    parser.add_argument("--threshold", type=int, default=30,
                       help="LFB intensity threshold for positive pixels (default: 30)")
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print("BATCH LFB QUANTIFICATION WITH HEATMAPS")
    print(f"{'='*80}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"LFB threshold: {args.threshold}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all slide pairs
    print(f"\n{'='*80}")
    print("FINDING SLIDES")
    print(f"{'='*80}")
    
    pairs = find_slide_pairs(args.input_dir)
    
    if len(pairs) == 0:
        print("No valid slide pairs found!")
        print("Looking for:")
        print("  - mask_intact_tissue.png or .npy in intact_analysis*/")
        print("  - *.tiff or *.tif in L0_T256_S256/")
        return
    
    print(f"\nFound {len(pairs)} slides to process")
    
    # Process each slide
    results = []
    
    for slide_name, mask_path, tiff_path in pairs:
        result = process_slide(slide_name, mask_path, tiff_path, 
                              args.output_dir, args.threshold)
        results.append(result)
    
    # Generate summary
    print(f"\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    
    status_counts = {}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    
    for status, count in status_counts.items():
        print(f"{status}: {count} slides")
    
    # Consolidate results
    df = consolidate_results(results, args.output_dir)
    
    if df is not None:
        print(f"\nQuantification Summary:")
        print(f"  Mean % LFB-positive: {df['percent_positive'].mean():.2f}%")
        print(f"  Median % LFB-positive: {df['percent_positive'].median():.2f}%")
        print(f"  Mean LFB intensity: {df['mean_intensity'].mean():.2f}")
        print(f"  Median LFB intensity: {df['median_intensity'].median():.2f}")
    
    # Save processing summary
    summary_df = pd.DataFrame(results)
    summary_csv = os.path.join(args.output_dir, "batch_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\n✓ All results saved to {args.output_dir}")
    print(f"\nOutput structure:")
    print(f"  {args.output_dir}/")
    print(f"    ├── all_quantifications.csv       # Combined metrics")
    print(f"    ├── batch_summary.csv             # Processing status")
    print(f"    ├── slide_*/")
    print(f"    │   ├── results.json              # Quantification data")
    print(f"    │   ├── lfb_masked.png            # Masked LFB channel")
    print(f"    │   ├── lfb_heatmap.png           # Heatmap (hot colors)")
    print(f"    │   └── composite.png             # All three panels")


if __name__ == "__main__":
    main()
