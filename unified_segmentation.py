"""
Unified Segmentation Pipeline

This module provides a unified interface for running all segmentation models
(SAM, EoMT, RAM, DINOv2) and comparing their results.
"""

import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import time

# Import all segmentation modules
from sam_segmentation import SAMSegmentationPipeline
from eomt_segmentation import EoMTSegmentationPipeline
from ram_segmentation import RAMSegmentationPipeline
from dinov2_segmentation import DINOv2SegmentationPipeline
from kidney_preprocessing import KidneyPreprocessingPipeline


class UnifiedSegmentationPipeline:
    """Unified pipeline for all segmentation models."""

    def __init__(self,
                 models_to_use: Optional[List[str]] = None,
                 sam_checkpoint: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize unified segmentation pipeline.

        Args:
            models_to_use: List of models to use ('sam', 'eomt', 'ram', 'dinov2')
                          If None, use all models
            sam_checkpoint: Path to SAM checkpoint (if using SAM)
            device: Device to use
        """
        if models_to_use is None:
            models_to_use = ['sam', 'eomt', 'ram', 'dinov2']

        self.models_to_use = models_to_use
        self.pipelines = {}
        self.device = device

        print("=" * 60)
        print("Initializing Unified Segmentation Pipeline")
        print("=" * 60)

        # Initialize requested models
        if 'sam' in models_to_use:
            print("\n[1/4] Initializing SAM...")
            try:
                self.pipelines['sam'] = SAMSegmentationPipeline(
                    checkpoint_path=sam_checkpoint,
                    device=device
                )
                print("SAM initialized successfully")
            except Exception as e:
                print(f"Warning: SAM initialization failed: {e}")

        if 'eomt' in models_to_use:
            print("\n[2/4] Initializing EoMT...")
            try:
                self.pipelines['eomt'] = EoMTSegmentationPipeline(
                    device=device
                )
                print("EoMT initialized successfully")
            except Exception as e:
                print(f"Warning: EoMT initialization failed: {e}")

        if 'ram' in models_to_use:
            print("\n[3/4] Initializing RAM...")
            try:
                self.pipelines['ram'] = RAMSegmentationPipeline(
                    device=device
                )
                print("RAM initialized successfully")
            except Exception as e:
                print(f"Warning: RAM initialization failed: {e}")

        if 'dinov2' in models_to_use:
            print("\n[4/4] Initializing DINOv2...")
            try:
                self.pipelines['dinov2'] = DINOv2SegmentationPipeline(
                    device=device
                )
                print("DINOv2 initialized successfully")
            except Exception as e:
                print(f"Warning: DINOv2 initialization failed: {e}")

        print("\n" + "=" * 60)
        print(f"Initialized {len(self.pipelines)}/{len(models_to_use)} models")
        print("=" * 60)

    def segment_single_image(self,
                            preprocessed_result: Dict,
                            verbose: bool = False) -> Dict:
        """
        Segment single image using all available models.

        Args:
            preprocessed_result: Preprocessed image result
            verbose: Print details

        Returns:
            Dictionary with results from all models
        """
        results = {
            'image_path': preprocessed_result.get('image_path', ''),
            'segmentations': {},
            'timings': {},
            'statistics': {}
        }

        for model_name, pipeline in self.pipelines.items():
            if verbose:
                print(f"  Running {model_name.upper()}...", end=' ')

            try:
                start_time = time.time()
                seg_result = pipeline.segment_preprocessed_image(preprocessed_result)
                elapsed = time.time() - start_time

                results['segmentations'][model_name] = seg_result['segmentation']
                results['timings'][model_name] = elapsed
                results['statistics'][model_name] = {
                    'num_segments': seg_result['num_segments'],
                    'method': seg_result.get('method', model_name),
                    'model_type': seg_result.get('model_type', model_name)
                }

                if verbose:
                    print(f"Done ({elapsed:.2f}s, {seg_result['num_segments']} segments)")

            except Exception as e:
                if verbose:
                    print(f"Failed: {e}")
                results['segmentations'][model_name] = None
                results['timings'][model_name] = None
                results['statistics'][model_name] = {'error': str(e)}

        return results

    def segment_batch(self,
                     preprocessed_results: List[Dict],
                     verbose: bool = True,
                     save_results: bool = True,
                     output_dir: str = "output_segmentation") -> Dict:
        """
        Segment batch of images using all models.

        Args:
            preprocessed_results: List of preprocessed results
            verbose: Print progress
            save_results: Save segmentation results
            output_dir: Output directory

        Returns:
            Dictionary with all results and statistics
        """
        all_results = []
        total = len(preprocessed_results)

        print("\n" + "=" * 60)
        print(f"Segmenting {total} images with {len(self.pipelines)} models")
        print("=" * 60)

        for idx, prep_result in enumerate(preprocessed_results):
            if verbose:
                print(f"\n[{idx + 1}/{total}] Processing image...")

            result = self.segment_single_image(prep_result, verbose=verbose)
            result['image_index'] = idx
            all_results.append(result)

        # Compute statistics
        statistics = self._compute_statistics(all_results)

        # Save results if requested
        if save_results:
            self._save_results(all_results, output_dir, verbose=verbose)

        return {
            'results': all_results,
            'statistics': statistics,
            'num_images': total,
            'models': list(self.pipelines.keys())
        }

    def _compute_statistics(self, all_results: List[Dict]) -> Dict:
        """Compute aggregate statistics across all images and models."""
        stats = {}

        for model_name in self.pipelines.keys():
            timings = [r['timings'][model_name] for r in all_results
                      if r['timings'].get(model_name) is not None]
            num_segments = [r['statistics'][model_name]['num_segments']
                           for r in all_results
                           if model_name in r['statistics']
                           and 'num_segments' in r['statistics'][model_name]]

            stats[model_name] = {
                'avg_time': np.mean(timings) if timings else None,
                'total_time': np.sum(timings) if timings else None,
                'avg_segments': np.mean(num_segments) if num_segments else None,
                'std_segments': np.std(num_segments) if num_segments else None,
                'successful': len([r for r in all_results
                                  if model_name in r['segmentations']
                                  and r['segmentations'][model_name] is not None])
            }

        return stats

    def _save_results(self, all_results: List[Dict], output_dir: str, verbose: bool = True):
        """Save segmentation results to disk."""
        os.makedirs(output_dir, exist_ok=True)

        for idx, result in enumerate(all_results):
            for model_name, segmentation in result['segmentations'].items():
                if segmentation is not None:
                    filepath = os.path.join(output_dir, f"{model_name}_seg_{idx:03d}.npy")
                    np.save(filepath, segmentation)

        if verbose:
            print(f"\nResults saved to: {output_dir}/")

    def print_statistics(self, statistics: Dict):
        """Print formatted statistics."""
        print("\n" + "=" * 60)
        print("SEGMENTATION STATISTICS")
        print("=" * 60)

        for model_name, stats in statistics.items():
            print(f"\n{model_name.upper()}:")
            print(f"  - Successful: {stats['successful']}")
            if stats['avg_time'] is not None:
                print(f"  - Avg time: {stats['avg_time']:.3f}s")
                print(f"  - Total time: {stats['total_time']:.2f}s")
            if stats['avg_segments'] is not None:
                print(f"  - Avg segments: {stats['avg_segments']:.1f} ± {stats['std_segments']:.1f}")

    def compare_segmentations(self,
                             results: Dict,
                             metrics: List[str] = ['segments', 'time']) -> Dict:
        """
        Compare segmentation results across models.

        Args:
            results: Results from segment_batch
            metrics: Metrics to compare

        Returns:
            Comparison dictionary
        """
        comparison = {
            'models': results['models'],
            'num_images': results['num_images'],
            'metrics': {}
        }

        stats = results['statistics']

        if 'segments' in metrics:
            comparison['metrics']['segments'] = {
                model: stats[model]['avg_segments']
                for model in results['models']
                if stats[model]['avg_segments'] is not None
            }

        if 'time' in metrics:
            comparison['metrics']['time'] = {
                model: stats[model]['avg_time']
                for model in results['models']
                if stats[model]['avg_time'] is not None
            }

        return comparison


def run_complete_pipeline(dataset_dir: str = "Dataset",
                          max_per_class: int = 10,
                          models: Optional[List[str]] = None,
                          sam_checkpoint: Optional[str] = None,
                          output_dir: str = "output_segmentation",
                          use_localization: bool = True) -> Dict:
    """
    Run complete preprocessing and segmentation pipeline.

    Args:
        dataset_dir: Dataset directory
        max_per_class: Max images per class
        models: Models to use (None = all)
        sam_checkpoint: SAM checkpoint path
        output_dir: Output directory
        use_localization: Use YOLOv8 localization

    Returns:
        Complete results dictionary
    """
    print("=" * 60)
    print("COMPLETE SEGMENTATION PIPELINE")
    print("=" * 60)

    # Step 1: Load dataset
    from kidney_preprocessing import load_dataset
    print("\n[1] Loading dataset...")
    dataset = load_dataset(dataset_dir, max_per_class=max_per_class)

    all_paths = []
    all_labels = []
    for cls, paths in dataset.items():
        all_paths.extend(paths)
        all_labels.extend([cls] * len(paths))

    print(f"Loaded {len(all_paths)} images")

    # Step 2: Preprocess images
    print("\n[2] Preprocessing images...")
    preprocessing_pipeline = KidneyPreprocessingPipeline(
        target_size=(256, 256),
        extract_features=True,
        use_quickshift=True,
        use_gabor=False,
        use_localization=use_localization
    )

    preprocessed_results = preprocessing_pipeline.process_batch(all_paths, verbose=True)
    print(f"Preprocessing complete: {len(preprocessed_results)} images")

    # Step 3: Segment images
    print("\n[3] Running segmentation models...")
    segmentation_pipeline = UnifiedSegmentationPipeline(
        models_to_use=models,
        sam_checkpoint=sam_checkpoint
    )

    results = segmentation_pipeline.segment_batch(
        preprocessed_results,
        verbose=True,
        save_results=True,
        output_dir=output_dir
    )

    # Step 4: Print statistics
    segmentation_pipeline.print_statistics(results['statistics'])

    # Step 5: Compare models
    comparison = segmentation_pipeline.compare_segmentations(results)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)

    return {
        'preprocessed': preprocessed_results,
        'segmentation': results,
        'comparison': comparison,
        'labels': all_labels
    }


if __name__ == "__main__":
    print("Unified Segmentation Pipeline")
    print("=" * 60)
    print("\nThis module provides a unified interface for all segmentation models.")
    print("\nUsage:")
    print("  from unified_segmentation import UnifiedSegmentationPipeline")
    print("  pipeline = UnifiedSegmentationPipeline()")
    print("  results = pipeline.segment_batch(preprocessed_results)")
    print("\nOr run complete pipeline:")
    print("  from unified_segmentation import run_complete_pipeline")
    print("  results = run_complete_pipeline()")
