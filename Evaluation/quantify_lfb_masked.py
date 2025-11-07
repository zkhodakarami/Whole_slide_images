#!/usr/bin/env python3
"""
Quantify LFB in low-attention regions using mask-based approach.

This script:
1. Loads the low-attention mask
2. Applies color deconvolution to the slide
3. Masks the LFB channel to keep only low-attention regions
4. Quantifies LFB in those regions

Usage:
    python quantify_lfb_masked.py \
        --slide slide.tif \
        --mask low_attention_mask.png \
        --output results.json \
        --patch_size 256
"""

import numpy as np
import cv2
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict
from skimage import color as skcolor


def load_mask(mask_path: str, patch_size: int) -> np.ndarray:
    """
    Load binary mask from file.
    
    Args:
        mask_path: Path to mask PNG
        patch_size: CLAM patch size to calculate scaling
    
    Returns:
        Binary mask (0 or 255)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask from {mask_path}")
    
    # Ensure binary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    print(f"Loaded mask: {binary_mask.shape}")
    print(f"Mask pixels (white): {np.sum(binary_mask > 0)} / {binary_mask.size}")
    
    return binary_mask


def load_slide(slide_path: str, level: int = 0) -> np.ndarray:
    """
    Load slide image.
    
    Args:
        slide_path: Path to slide TIFF
        level: Pyramid level to load
    
    Returns:
        RGB image array
    """
    try:
        import openslide
        slide = openslide.OpenSlide(slide_path)
        
        dims = slide.level_dimensions[level]
        region = slide.read_region((0, 0), level, dims)
        rgb = np.array(region.convert('RGB'))
        
        slide.close()
        print(f"Loaded slide at level {level}: {rgb.shape}")
        return rgb
        
    except ImportError:
        print("OpenSlide not available, using tifffile...")
        import tifffile
        
        img = tifffile.imread(slide_path)
        
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:
            img = img[:, :, :3]
        
        # If level > 0, downsample
        if level > 0:
            factor = 2 ** level
            h, w = img.shape[0] // factor, img.shape[1] // factor
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        
        print(f"Loaded slide: {img.shape}")
        return img


def lfb_color_deconvolution(rgb_image: np.ndarray, 
                            alpha: float = 1.0,
                            beta: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate LFB and CV stains using color deconvolution.
    
    Args:
        rgb_image: RGB image (H, W, 3), uint8
        alpha: Brightness adjustment for LFB channel
        beta: Brightness adjustment for CV channel
    
    Returns:
        lfb_channel: LFB-deconvolved channel (uint8)
        cv_channel: CV-deconvolved channel (uint8)
    """
    print("Performing color deconvolution...")
    
    # Normalize to [0, 1]
    rgb_norm = rgb_image.astype(np.float32) / 255.0
    
    # Add epsilon to avoid log(0)
    epsilon = 1e-6
    rgb_norm = np.maximum(rgb_norm, epsilon)
    
    # Convert to optical density
    od = -np.log10(rgb_norm)
    
    # LFB-CV stain matrix (approximate)
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
    lfb_channel = concentrations[:, :, 0] * alpha
    cv_channel = concentrations[:, :, 1] * beta
    
    # Normalize to [0, 255]
    lfb_channel = np.clip(lfb_channel * 255 / (np.max(lfb_channel) + epsilon), 0, 255).astype(np.uint8)
    cv_channel = np.clip(cv_channel * 255 / (np.max(cv_channel) + epsilon), 0, 255).astype(np.uint8)
    
    print(f"LFB channel range: {lfb_channel.min()} - {lfb_channel.max()}")
    
    return lfb_channel, cv_channel


def resize_mask_to_image(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize mask to match image dimensions.
    
    Args:
        mask: Binary mask
        target_shape: (height, width) of target image
    
    Returns:
        Resized mask
    """
    h, w = target_shape
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Ensure binary after resize
    _, mask_resized = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
    
    print(f"Resized mask from {mask.shape} to {mask_resized.shape}")
    
    return mask_resized


def quantify_masked_lfb(lfb_channel: np.ndarray,
                       mask: np.ndarray,
                       threshold: int = 30) -> Dict:
    """
    Quantify LFB in masked regions only.
    
    Args:
        lfb_channel: LFB-deconvolved channel (uint8)
        mask: Binary mask (255 = low-attention, 0 = ignore)
        threshold: Intensity threshold for LFB-positive pixels
    
    Returns:
        Dictionary with quantification metrics
    """
    print("Quantifying LFB in masked regions...")
    
    # Get pixels in masked region
    masked_region = mask > 0
    lfb_masked = lfb_channel[masked_region]
    
    if len(lfb_masked) == 0:
        print("Warning: No pixels in masked region!")
        return {
            "n_pixels_total": 0,
            "n_pixels_positive": 0,
            "percent_positive": 0.0,
            "mean_intensity": 0.0,
            "median_intensity": 0.0,
            "std_intensity": 0.0
        }
    
    # Threshold for positive pixels
    positive_pixels = lfb_masked > threshold
    
    metrics = {
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
    
    print(f"  Total pixels in mask: {metrics['n_pixels_total']:,}")
    print(f"  LFB-positive pixels: {metrics['n_pixels_positive']:,} ({metrics['percent_positive']:.2f}%)")
    print(f"  Mean LFB intensity: {metrics['mean_intensity']:.2f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Quantify LFB in low-attention masked regions"
    )
    
    parser.add_argument("--slide", required=True,
                       help="Path to whole slide image (TIFF)")
    
    parser.add_argument("--mask", required=True,
                       help="Path to low-attention mask (PNG)")
    
    parser.add_argument("--output", required=True,
                       help="Output JSON file for results")
    
    parser.add_argument("--patch_size", type=int, default=256,
                       help="CLAM patch size (default: 256)")
    
    parser.add_argument("--level", type=int, default=0,
                       help="Pyramid level to analyze (default: 0)")
    
    parser.add_argument("--threshold", type=int, default=30,
                       help="LFB intensity threshold for positive pixels (default: 30)")
    
    parser.add_argument("--save_masked_lfb", type=str, default=None,
                       help="Optional: save masked LFB channel as image")
    
    parser.add_argument("--save_overlay", type=str, default=None,
                       help="Optional: save RGB overlay with mask")
    
    args = parser.parse_args()
    
    # Load mask
    mask = load_mask(args.mask, args.patch_size)
    
    # Load slide
    rgb_image = load_slide(args.slide, args.level)
    
    # Resize mask to match slide dimensions
    mask_resized = resize_mask_to_image(mask, rgb_image.shape[:2])
    
    # Color deconvolution
    lfb_channel, cv_channel = lfb_color_deconvolution(rgb_image)
    
    # Apply mask to LFB channel (set non-masked regions to 0)
    lfb_masked_display = lfb_channel.copy()
    lfb_masked_display[mask_resized == 0] = 0
    
    # Save masked LFB channel if requested
    if args.save_masked_lfb:
        cv2.imwrite(args.save_masked_lfb, lfb_masked_display)
        print(f"Saved masked LFB channel to {args.save_masked_lfb}")
    
    # Create overlay if requested
    if args.save_overlay:
        # Create RGB overlay: original image with mask in red
        overlay = rgb_image.copy()
        overlay[mask_resized > 0] = overlay[mask_resized > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
        cv2.imwrite(args.save_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved overlay to {args.save_overlay}")
    
    # Quantify LFB in masked regions
    metrics = quantify_masked_lfb(lfb_channel, mask_resized, args.threshold)
    
    # Add metadata
    results = {
        "slide_path": str(args.slide),
        "mask_path": str(args.mask),
        "level": args.level,
        "patch_size": args.patch_size,
        "mask_shape": list(mask.shape),
        "slide_shape": list(rgb_image.shape),
        "scale_factor": args.patch_size,
        "quantification": metrics
    }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results to {output_path}")
    print("\nSummary:")
    print(f"  Low-attention area: {metrics['n_pixels_total']:,} pixels")
    print(f"  LFB-positive: {metrics['percent_positive']:.2f}%")
    print(f"  Mean LFB intensity: {metrics['mean_intensity']:.2f}")


if __name__ == "__main__":
    main()
