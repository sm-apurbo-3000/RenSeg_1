"""
Test script for SAM segmentation module
"""

import os
import numpy as np
from sam_segmentation import SAMSegmentationPipeline, download_sam_checkpoint
from kidney_preprocessing import KidneyPreprocessingPipeline

print("SAM Segmentation Test")
print("=" * 60)

# Step 1: Download SAM checkpoint if needed
print("\n[1] Checking SAM checkpoint...")
try:
    checkpoint_path = download_sam_checkpoint(model_type='vit_b', output_dir='.')
    print(f"Checkpoint ready: {checkpoint_path}")
except Exception as e:
    print(f"Error downloading checkpoint: {e}")
    print("\nPlease manually download from:")
    print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
    exit(1)

# Step 2: Load preprocessed results
print("\n[2] Loading preprocessed results...")
output_dir = "output"
features_path = os.path.join(output_dir, "features_matrix.npy")
image_paths_file = os.path.join(output_dir, "image_paths.txt")

if not os.path.exists(image_paths_file):
    print("Error: No preprocessed results found. Run main.py first.")
    exit(1)

with open(image_paths_file, 'r') as f:
    image_paths = [line.strip() for line in f.readlines()]

print(f"Found {len(image_paths)} preprocessed images")

# Step 3: Re-process a few images to get preprocessed results with metadata
print("\n[3] Re-processing sample images...")
pipeline = KidneyPreprocessingPipeline(
    target_size=(256, 256),
    extract_features=True,
    use_quickshift=True,
    use_gabor=False,
    use_localization=True
)

# Process first 5 images
sample_paths = image_paths[:5]
sample_results = []

for path in sample_paths:
    try:
        result = pipeline.process_image(path, verbose=False)
        if 'error' not in result:
            sample_results.append(result)
    except Exception as e:
        print(f"Error processing {path}: {e}")

print(f"Successfully processed {len(sample_results)} sample images")

# Step 4: Initialize SAM pipeline
print("\n[4] Initializing SAM segmentation pipeline...")
try:
    sam_pipeline = SAMSegmentationPipeline(
        model_type='vit_b',
        checkpoint_path=checkpoint_path,
        use_automatic=True,
        min_mask_area=100
    )
    print("SAM pipeline initialized successfully")
except Exception as e:
    print(f"Error initializing SAM: {e}")
    exit(1)

# Step 5: Segment sample images
print("\n[5] Segmenting sample images...")
seg_results = sam_pipeline.segment_batch(sample_results, verbose=True)

# Step 6: Display results
print("\n[6] Segmentation Results:")
print("=" * 60)
for idx, result in enumerate(seg_results):
    if 'error' not in result:
        print(f"Image {idx + 1}:")
        print(f"  - Segments: {result['num_segments']}")
        print(f"  - Shape: {result['image_shape']}")
        print(f"  - Method: {result['method']}")
        print(f"  - Model: {result['model_type']}")
    else:
        print(f"Image {idx + 1}: ERROR - {result['error']}")

# Step 7: Save sample results
print("\n[7] Saving sample segmentation results...")
output_seg_dir = "output_segmentation"
os.makedirs(output_seg_dir, exist_ok=True)

for idx, result in enumerate(seg_results):
    if 'error' not in result:
        seg_path = os.path.join(output_seg_dir, f"sam_seg_{idx:03d}.npy")
        np.save(seg_path, result['segmentation'])

print(f"Saved {len(seg_results)} segmentation results to {output_seg_dir}/")

print("\n" + "=" * 60)
print("SAM TEST COMPLETE!")
print("=" * 60)
