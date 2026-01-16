"""
Test script for DINOv2 segmentation module
"""

import os
import numpy as np
from dinov2_segmentation import DINOv2SegmentationPipeline
from kidney_preprocessing import KidneyPreprocessingPipeline

print("DINOv2 Segmentation Test")
print("=" * 60)

# Step 1: Load preprocessed image paths
print("\n[1] Loading preprocessed image paths...")
output_dir = "output"
image_paths_file = os.path.join(output_dir, "image_paths.txt")

if not os.path.exists(image_paths_file):
    print("Error: No preprocessed results found. Run main.py first.")
    exit(1)

with open(image_paths_file, 'r') as f:
    image_paths = [line.strip() for line in f.readlines()]

print(f"Found {len(image_paths)} preprocessed images")

# Step 2: Re-process sample images
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

# Step 3: Initialize DINOv2 pipeline
print("\n[3] Initializing DINOv2 segmentation pipeline...")
try:
    dinov2_pipeline = DINOv2SegmentationPipeline(
        n_clusters=5
    )
    print("DINOv2 pipeline initialized successfully")
except Exception as e:
    print(f"Error initializing DINOv2: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 4: Segment sample images
print("\n[4] Segmenting sample images...")
seg_results = dinov2_pipeline.segment_batch(sample_results, verbose=True)

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

# Step 6: Extract features for clustering
print("\n[6] Extracting DINOv2 features for clustering...")
feature_matrix = dinov2_pipeline.extract_features_for_clustering(sample_results, verbose=True)
print(f"Feature matrix shape: {feature_matrix.shape}")

# Step 7: Save results
print("\n[7] Saving results...")
output_seg_dir = "output_segmentation"
os.makedirs(output_seg_dir, exist_ok=True)

for idx, result in enumerate(seg_results):
    if 'error' not in result:
        seg_path = os.path.join(output_seg_dir, f"dinov2_seg_{idx:03d}.npy")
        np.save(seg_path, result['segmentation'])

# Save feature matrix
feat_path = os.path.join(output_seg_dir, "dinov2_features.npy")
np.save(feat_path, feature_matrix)

print(f"Saved {len(seg_results)} segmentation results to {output_seg_dir}/")
print(f"Saved feature matrix: {feat_path}")

print("\n" + "=" * 60)
print("DINOv2 TEST COMPLETE!")
print("=" * 60)
