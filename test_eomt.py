"""
Test script for EoMT segmentation module
"""

import os
import numpy as np
from eomt_segmentation import EoMTSegmentationPipeline
from kidney_preprocessing import KidneyPreprocessingPipeline

print("EoMT Segmentation Test")
print("=" * 60)

# Step 1: Load preprocessed results
print("\n[1] Loading preprocessed image paths...")
output_dir = "output"
image_paths_file = os.path.join(output_dir, "image_paths.txt")

if not os.path.exists(image_paths_file):
    print("Error: No preprocessed results found. Run main.py first.")
    exit(1)

with open(image_paths_file, 'r') as f:
    image_paths = [line.strip() for line in f.readlines()]

print(f"Found {len(image_paths)} preprocessed images")

# Step 2: Re-process a few images
print("\n[2] Re-processing sample images...")
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

# Step 3: Initialize EoMT pipeline
print("\n[3] Initializing EoMT segmentation pipeline...")
try:
    eomt_pipeline = EoMTSegmentationPipeline(
        img_size=256,
        patch_size=16,
        min_area=100
    )
    print("EoMT pipeline initialized successfully")
    print("Note: Using randomly initialized model (not trained)")
except Exception as e:
    print(f"Error initializing EoMT: {e}")
    exit(1)

# Step 4: Segment sample images
print("\n[4] Segmenting sample images...")
seg_results = eomt_pipeline.segment_batch(sample_results, verbose=True)

# Step 5: Display results
print("\n[5] Segmentation Results:")
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

# Step 6: Save sample results
print("\n[6] Saving sample segmentation results...")
output_seg_dir = "output_segmentation"
os.makedirs(output_seg_dir, exist_ok=True)

for idx, result in enumerate(seg_results):
    if 'error' not in result:
        seg_path = os.path.join(output_seg_dir, f"eomt_seg_{idx:03d}.npy")
        np.save(seg_path, result['segmentation'])

print(f"Saved {len(seg_results)} segmentation results to {output_seg_dir}/")

print("\n" + "=" * 60)
print("EoMT TEST COMPLETE!")
print("=" * 60)
print("\nNote: This model uses random initialization.")
print("For production use, train the model on labeled kidney CT data.")
