"""
Comprehensive test script for all segmentation models
"""

import os
import numpy as np
from unified_segmentation import UnifiedSegmentationPipeline
from kidney_preprocessing import KidneyPreprocessingPipeline

print("=" * 60)
print("COMPREHENSIVE SEGMENTATION TEST")
print("=" * 60)

# Configuration
MAX_TEST_IMAGES = 10  # Test on 10 images
USE_LOCALIZATION = True
OUTPUT_DIR = "output_segmentation"

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

# Step 2: Re-process test images
print(f"\n[2] Re-processing {MAX_TEST_IMAGES} test images...")
pipeline = KidneyPreprocessingPipeline(
    target_size=(256, 256),
    extract_features=True,
    use_quickshift=True,
    use_gabor=False,
    use_localization=USE_LOCALIZATION
)

# Process test subset
test_paths = image_paths[:MAX_TEST_IMAGES]
test_results = []

for idx, path in enumerate(test_paths):
    print(f"  Processing {idx + 1}/{MAX_TEST_IMAGES}...", end=' ')
    try:
        result = pipeline.process_image(path, verbose=False)
        if 'error' not in result:
            test_results.append(result)
            print("OK")
        else:
            print(f"Error: {result['error']}")
    except Exception as e:
        print(f"Exception: {e}")

print(f"Successfully processed {len(test_results)} images")

# Step 3: Initialize unified segmentation pipeline
print("\n[3] Initializing unified segmentation pipeline...")

# Determine which models to use
models_to_use = ['sam', 'eomt', 'ram', 'dinov2']

# Check if SAM checkpoint exists
sam_checkpoint = "sam_vit_b_01ec64.pth"
if not os.path.exists(sam_checkpoint):
    print(f"Warning: SAM checkpoint not found: {sam_checkpoint}")
    print("SAM will be excluded from testing")
    models_to_use.remove('sam')
    sam_checkpoint = None

try:
    unified_pipeline = UnifiedSegmentationPipeline(
        models_to_use=models_to_use,
        sam_checkpoint=sam_checkpoint
    )
except Exception as e:
    print(f"Error initializing pipeline: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 4: Run segmentation on all test images
print(f"\n[4] Running segmentation on {len(test_results)} images...")
results = unified_pipeline.segment_batch(
    test_results,
    verbose=True,
    save_results=True,
    output_dir=OUTPUT_DIR
)

# Step 5: Print statistics
unified_pipeline.print_statistics(results['statistics'])

# Step 6: Model comparison
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison = unified_pipeline.compare_segmentations(results)

print(f"\nNumber of images: {comparison['num_images']}")
print(f"Models tested: {', '.join(comparison['models'])}")

if 'segments' in comparison['metrics']:
    print("\nAverage number of segments:")
    for model, avg_segs in comparison['metrics']['segments'].items():
        print(f"  - {model.upper()}: {avg_segs:.1f}")

if 'time' in comparison['metrics']:
    print("\nAverage processing time:")
    for model, avg_time in comparison['metrics']['time'].items():
        print(f"  - {model.upper()}: {avg_time:.3f}s")

# Step 7: Save summary
print(f"\n[7] Saving results summary...")
summary_file = os.path.join(OUTPUT_DIR, "segmentation_summary.txt")

with open(summary_file, 'w') as f:
    f.write("SEGMENTATION SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Number of images: {comparison['num_images']}\n")
    f.write(f"Models: {', '.join(comparison['models'])}\n\n")

    f.write("STATISTICS BY MODEL:\n")
    f.write("-" * 60 + "\n")
    for model, stats in results['statistics'].items():
        f.write(f"\n{model.upper()}:\n")
        f.write(f"  Successful: {stats['successful']}\n")
        if stats['avg_time'] is not None:
            f.write(f"  Avg time: {stats['avg_time']:.3f}s\n")
            f.write(f"  Total time: {stats['total_time']:.2f}s\n")
        if stats['avg_segments'] is not None:
            f.write(f"  Avg segments: {stats['avg_segments']:.1f} ± {stats['std_segments']:.1f}\n")

    f.write("\n" + "=" * 60 + "\n")

print(f"Summary saved: {summary_file}")

# Step 8: List output files
print(f"\n[8] Output files:")
output_files = os.listdir(OUTPUT_DIR)
segmentation_files = [f for f in output_files if f.endswith('.npy')]

for model in comparison['models']:
    model_files = [f for f in segmentation_files if f.startswith(model)]
    print(f"  - {model.upper()}: {len(model_files)} files")

print("\n" + "=" * 60)
print("COMPREHENSIVE TEST COMPLETE!")
print("=" * 60)
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print(f"Summary report: {summary_file}")
