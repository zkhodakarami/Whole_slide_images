# Extract Regions for LFB Quantification

Simple workflow to extract low-attention regions and quantify LFB staining.

## Workflow

1. **Extract regions** from low-attention masks → JSON files with coordinates
2. **Quantify LFB** in those regions → CSV with measurements

## Step 1: Extract Region Coordinates

### Single Slide

```bash
python extract_regions.py \
    --mask ./phas_clam_outputs/slide_DI_123456_ABC-01-LFB+CV/L0_T256_S256/intact_analysis_p5/low_attention_mask.png \
    --output regions.json \
    --patch_size 256
```

**Output:** `regions.json` with polygon coordinates

### Batch Processing

```bash
# Create slide list
cat > slides.txt << EOF
DI_123456_ABC-01-LFB+CV
DI_123456_ABC-02-LFB+CV
DI_789012_DEF-01-LFB+CV
EOF

# Extract all regions
python batch_extract_regions.py \
    --input slides.txt \
    --clam_output ./phas_clam_outputs \
    --output_dir ./regions \
    --patch_size 256
```

**Output:** 
```
./regions/
├── DI_123456_ABC-01-LFB+CV_regions.json
├── DI_123456_ABC-02-LFB+CV_regions.json
└── DI_789012_DEF-01-LFB+CV_regions.json
```

## Step 2: Quantify LFB in Regions

### Single Slide

```bash
python quantify_lfb_in_regions.py \
    --slide /path/to/DI_123456_ABC-01-LFB+CV.tif \
    --regions regions.json \
    --output quantification.csv \
    --threshold 30 \
    --save_lfb lfb_channel.png
```

**Output:** `quantification.csv` with LFB measurements per region

### Batch Processing

```bash
# Process all slides
for slide in DI_123456_ABC-01-LFB+CV DI_123456_ABC-02-LFB+CV; do
    echo "Quantifying $slide..."
    
    # Find TIFF path
    SAMPLE_NUM=$(echo $slide | grep -oP 'DI_\K\d+')
    TIFF="/mnt/hippogang/histology/irwin/archive/INDD${SAMPLE_NUM}/histo_raw/${slide}.tif"
    
    python quantify_lfb_in_regions.py \
        --slide "$TIFF" \
        --regions "./regions/${slide}_regions.json" \
        --output "./quantification/${slide}_lfb.csv" \
        --threshold 30
done

# Combine all results
python -c "
import pandas as pd
import glob

csvs = glob.glob('./quantification/*_lfb.csv')
dfs = []
for csv in csvs:
    df = pd.read_csv(csv)
    slide_name = csv.split('/')[-1].replace('_lfb.csv', '')
    df['slide_id'] = slide_name
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv('./all_quantifications.csv', index=False)
print(f'Combined {len(combined)} regions from {len(csvs)} slides')
"
```

## Output Format

### regions.json
```json
{
  "regions": [
    {
      "region_id": 0,
      "coordinates": [[x1, y1], [x2, y2], ...],
      "bbox": [x_min, y_min, x_max, y_max],
      "centroid": [cx, cy],
      "area_wsi_units": 524288.0
    }
  ],
  "metadata": {
    "patch_size": 256,
    "scale_factor": 256,
    "n_regions": 15
  }
}
```

### quantification.csv
```csv
region_id,n_pixels_total,n_pixels_positive,percent_positive,mean_intensity,...
0,10000,3500,35.0,45.2,...
1,8500,2100,24.7,38.5,...
```

## Parameters

### Your CLAM Settings
- **Patch size:** 256 (from your ClamResumeSuperpixels.py)
- **Level:** 0 (full resolution)
- **Scale:** 256 (each mask pixel = one 256×256 patch)

Always use `--patch_size 256`

### LFB Quantification
- `--threshold`: LFB intensity threshold for positive pixels (default: 30)
- `--level`: Pyramid level to analyze (default: 0)
- `--save_lfb`: Optional, save LFB channel as image

## Complete Example

```bash
# 1. Already ran CLAM intact analysis
python batch_process_heatmap_tissue_analyser.py \
    --input cases.txt --intact_only --percentile 5

# 2. Extract regions
python batch_extract_regions.py \
    --input cases.txt \
    --clam_output ./phas_clam_outputs \
    --output_dir ./regions

# 3. Quantify LFB (for each slide)
for json_file in ./regions/*.json; do
    slide_name=$(basename "$json_file" _regions.json)
    sample_num=$(echo "$slide_name" | grep -oP 'DI_\K\d+')
    
    tiff="/mnt/hippogang/histology/irwin/archive/INDD${sample_num}/histo_raw/${slide_name}.tif"
    
    if [ -f "$tiff" ]; then
        python quantify_lfb_in_regions.py \
            --slide "$tiff" \
            --regions "$json_file" \
            --output "./quantification/${slide_name}_lfb.csv"
    fi
done

# 4. Analyze results
python analyze_results.py
```

## Files You Need

1. **extract_regions.py** - Extract polygon coordinates
2. **batch_extract_regions.py** - Batch extraction
3. **quantify_lfb_in_regions.py** - Quantify LFB in regions

That's it! No PHAS, no database IDs needed. Just extract coordinates and quantify.
