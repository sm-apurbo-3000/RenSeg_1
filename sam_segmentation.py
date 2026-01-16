"""
Segment Anything Model (SAM) Integration Module

This module provides SAM-based segmentation for kidney CT images,
generating high-quality segmentation masks for clustering.
"""

import numpy as np
import cv2
import torch
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import os

try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: segment-anything not installed. Run: pip install segment-anything")


class SAMSegmenter:
    """Segment Anything Model (SAM) for kidney CT segmentation."""

    def __init__(self,
                 model_type: str = "vit_b",
                 checkpoint_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize SAM segmenter.

        Args:
            model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            checkpoint_path: Path to SAM checkpoint file
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if not SAM_AVAILABLE:
            raise ImportError("segment-anything package not installed")

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model_type = model_type
        self.checkpoint_path = checkpoint_path

        # Model type to default checkpoint mapping
        self.default_checkpoints = {
            'vit_b': 'sam_vit_b_01ec64.pth',
            'vit_l': 'sam_vit_l_0b3195.pth',
            'vit_h': 'sam_vit_h_4b8939.pth'
        }

        # Initialize model
        self.model = None
        self.predictor = None
        self.mask_generator = None

        self._load_model()

    def _load_model(self):
        """Load SAM model and initialize predictor."""
        # Determine checkpoint path
        if self.checkpoint_path is None:
            checkpoint_name = self.default_checkpoints.get(self.model_type)
            if checkpoint_name and os.path.exists(checkpoint_name):
                self.checkpoint_path = checkpoint_name
            else:
                raise ValueError(
                    f"No checkpoint found. Download from: "
                    f"https://github.com/facebookresearch/segment-anything#model-checkpoints"
                )

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        print(f"Loading SAM model: {self.model_type}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Device: {self.device}")

        # Load model
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.model.to(device=self.device)
        self.model.eval()

        # Initialize predictor for point/box prompts
        self.predictor = SamPredictor(self.model)

        print("SAM model loaded successfully")

    def initialize_auto_mask_generator(self,
                                       points_per_side: int = 32,
                                       pred_iou_thresh: float = 0.88,
                                       stability_score_thresh: float = 0.95,
                                       crop_n_layers: int = 0,
                                       crop_n_points_downscale_factor: int = 1,
                                       min_mask_region_area: int = 100):
        """
        Initialize automatic mask generator for prompt-free segmentation.

        Args:
            points_per_side: Number of points sampled per side
            pred_iou_thresh: IoU threshold for filtering masks
            stability_score_thresh: Stability score threshold
            crop_n_layers: Number of crop layers
            crop_n_points_downscale_factor: Downscale factor for points
            min_mask_region_area: Minimum mask area in pixels
        """
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area
        )
        print(f"Auto mask generator initialized with {points_per_side}x{points_per_side} grid")

    def segment_with_points(self,
                           image: np.ndarray,
                           point_coords: np.ndarray,
                           point_labels: np.ndarray,
                           multimask_output: bool = False) -> Dict:
        """
        Segment using point prompts.

        Args:
            image: Input image (H, W) or (H, W, 3)
            point_coords: Point coordinates (N, 2) as [x, y]
            point_labels: Point labels (N,) where 1=foreground, 0=background
            multimask_output: Whether to output multiple masks

        Returns:
            Dictionary with masks, scores, and logits
        """
        # Ensure RGB format
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image

        # Set image
        self.predictor.set_image(image_rgb)

        # Predict
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output
        )

        return {
            'masks': masks,
            'scores': scores,
            'logits': logits,
            'point_coords': point_coords,
            'point_labels': point_labels
        }

    def segment_with_box(self,
                        image: np.ndarray,
                        box: np.ndarray,
                        multimask_output: bool = False) -> Dict:
        """
        Segment using bounding box prompt.

        Args:
            image: Input image (H, W) or (H, W, 3)
            box: Bounding box [x1, y1, x2, y2]
            multimask_output: Whether to output multiple masks

        Returns:
            Dictionary with masks, scores, and logits
        """
        # Ensure RGB format
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image

        # Set image
        self.predictor.set_image(image_rgb)

        # Predict
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=multimask_output
        )

        return {
            'masks': masks,
            'scores': scores,
            'logits': logits,
            'box': box
        }

    def segment_automatic(self, image: np.ndarray) -> List[Dict]:
        """
        Automatic segmentation without prompts.

        Args:
            image: Input image (H, W) or (H, W, 3)

        Returns:
            List of mask dictionaries with keys: segmentation, area, bbox,
            predicted_iou, point_coords, stability_score, crop_box
        """
        if self.mask_generator is None:
            print("Initializing auto mask generator with default settings...")
            self.initialize_auto_mask_generator()

        # Ensure RGB format
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image

        # Generate masks
        masks = self.mask_generator.generate(image_rgb)

        return masks

    def segment_with_roi(self,
                        image: np.ndarray,
                        roi_info: Dict,
                        use_box_prompt: bool = True) -> Dict:
        """
        Segment using ROI information from localization.

        Args:
            image: Input image (H, W) or (H, W, 3)
            roi_info: ROI information from kidney_localizer
            use_box_prompt: Use ROI bbox as prompt, otherwise automatic

        Returns:
            Segmentation results dictionary
        """
        if roi_info.get('detected', False) and use_box_prompt:
            # Use expanded bbox as prompt
            box = np.array(roi_info['expanded_bbox'])
            return self.segment_with_box(image, box, multimask_output=True)
        else:
            # Automatic segmentation
            masks = self.segment_automatic(image)
            return {'masks': masks, 'method': 'automatic'}

    def filter_masks_by_area(self,
                            masks: List[Dict],
                            min_area: int = 100,
                            max_area: Optional[int] = None) -> List[Dict]:
        """
        Filter masks by area.

        Args:
            masks: List of mask dictionaries from segment_automatic
            min_area: Minimum mask area in pixels
            max_area: Maximum mask area in pixels (None for no limit)

        Returns:
            Filtered list of masks
        """
        filtered = []
        for mask in masks:
            area = mask['area']
            if area >= min_area:
                if max_area is None or area <= max_area:
                    filtered.append(mask)
        return filtered

    def get_combined_mask(self, masks: List[Dict], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Combine multiple masks into single segmentation map.

        Args:
            masks: List of mask dictionaries
            image_shape: (H, W) of output mask

        Returns:
            Segmentation map with region labels (H, W)
        """
        h, w = image_shape
        segmentation = np.zeros((h, w), dtype=np.int32)

        # Sort masks by area (largest first) to prioritize larger regions
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        for idx, mask in enumerate(sorted_masks):
            mask_binary = mask['segmentation']
            # Assign region label (1-indexed)
            segmentation[mask_binary] = idx + 1

        return segmentation

    def visualize_masks(self,
                       image: np.ndarray,
                       masks: Union[List[Dict], np.ndarray],
                       alpha: float = 0.5) -> np.ndarray:
        """
        Visualize segmentation masks overlaid on image.

        Args:
            image: Input image (H, W) or (H, W, 3)
            masks: List of mask dicts or segmentation map
            alpha: Transparency for overlay

        Returns:
            Visualization image (H, W, 3)
        """
        # Ensure RGB
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            vis = image.copy()

        # Normalize if needed
        if vis.max() <= 1.0:
            vis = (vis * 255).astype(np.uint8)

        # Handle different mask formats
        if isinstance(masks, list):
            # List of mask dictionaries
            for mask in masks:
                mask_binary = mask['segmentation']
                color = np.random.randint(0, 255, size=3)
                vis[mask_binary] = vis[mask_binary] * (1 - alpha) + color * alpha
        else:
            # Segmentation map
            num_regions = masks.max()
            for i in range(1, num_regions + 1):
                mask_binary = masks == i
                color = np.random.randint(0, 255, size=3)
                vis[mask_binary] = vis[mask_binary] * (1 - alpha) + color * alpha

        return vis.astype(np.uint8)


class SAMSegmentationPipeline:
    """Complete SAM segmentation pipeline for kidney CT images."""

    def __init__(self,
                 model_type: str = "vit_b",
                 checkpoint_path: Optional[str] = None,
                 use_automatic: bool = True,
                 min_mask_area: int = 100,
                 device: Optional[str] = None):
        """
        Initialize SAM segmentation pipeline.

        Args:
            model_type: SAM model type
            checkpoint_path: Path to checkpoint
            use_automatic: Use automatic segmentation
            min_mask_area: Minimum mask area threshold
            device: Device to use
        """
        self.segmenter = SAMSegmenter(model_type, checkpoint_path, device)
        self.use_automatic = use_automatic
        self.min_mask_area = min_mask_area

        if use_automatic:
            self.segmenter.initialize_auto_mask_generator(
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                min_mask_region_area=min_mask_area
            )

    def segment_preprocessed_image(self,
                                   preprocessed_result: Dict,
                                   use_roi_prompt: bool = True) -> Dict:
        """
        Segment a preprocessed image result.

        Args:
            preprocessed_result: Result from KidneyPreprocessingPipeline
            use_roi_prompt: Use ROI info as prompt if available

        Returns:
            Segmentation result dictionary
        """
        # Get preprocessed image
        image = preprocessed_result['preprocessed']['clahe']

        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Check if ROI info available
        roi_info = preprocessed_result['preprocessed'].get('roi_info', {})

        if use_roi_prompt and roi_info.get('detected', False):
            # Use ROI-guided segmentation
            result = self.segmenter.segment_with_roi(image, roi_info, use_box_prompt=True)

            # If multiple masks returned, select best
            if 'masks' in result and len(result['masks']) > 0:
                if isinstance(result['masks'], np.ndarray) and len(result['masks'].shape) == 3:
                    # Multiple masks from box prompt
                    best_idx = np.argmax(result['scores'])
                    segmentation = result['masks'][best_idx].astype(np.int32)
                else:
                    # List of masks from automatic
                    filtered = self.segmenter.filter_masks_by_area(
                        result['masks'],
                        min_area=self.min_mask_area
                    )
                    segmentation = self.segmenter.get_combined_mask(
                        filtered,
                        image.shape[:2]
                    )
        else:
            # Automatic segmentation
            masks = self.segmenter.segment_automatic(image)
            filtered = self.segmenter.filter_masks_by_area(masks, min_area=self.min_mask_area)
            segmentation = self.segmenter.get_combined_mask(filtered, image.shape[:2])

        return {
            'segmentation': segmentation,
            'num_segments': segmentation.max(),
            'image_shape': image.shape[:2],
            'method': 'sam',
            'model_type': self.segmenter.model_type
        }

    def segment_batch(self,
                     preprocessed_results: List[Dict],
                     verbose: bool = True) -> List[Dict]:
        """
        Segment batch of preprocessed images.

        Args:
            preprocessed_results: List of preprocessing results
            verbose: Print progress

        Returns:
            List of segmentation results
        """
        results = []
        total = len(preprocessed_results)

        for idx, prep_result in enumerate(preprocessed_results):
            if verbose and (idx + 1) % 10 == 0:
                print(f"Segmenting: {idx + 1}/{total}")

            try:
                seg_result = self.segment_preprocessed_image(prep_result)
                seg_result['image_index'] = idx
                seg_result['image_path'] = prep_result.get('image_path', '')
                results.append(seg_result)
            except Exception as e:
                print(f"Error segmenting image {idx}: {e}")
                results.append({
                    'error': str(e),
                    'image_index': idx,
                    'image_path': prep_result.get('image_path', '')
                })

        if verbose:
            successful = sum(1 for r in results if 'error' not in r)
            print(f"Segmentation complete: {successful}/{total} successful")

        return results


def download_sam_checkpoint(model_type: str = "vit_b", output_dir: str = ".") -> str:
    """
    Download SAM checkpoint if not exists.

    Args:
        model_type: Model type ('vit_b', 'vit_l', 'vit_h')
        output_dir: Directory to save checkpoint

    Returns:
        Path to checkpoint file
    """
    checkpoint_urls = {
        'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
    }

    checkpoint_names = {
        'vit_b': 'sam_vit_b_01ec64.pth',
        'vit_l': 'sam_vit_l_0b3195.pth',
        'vit_h': 'sam_vit_h_4b8939.pth'
    }

    if model_type not in checkpoint_urls:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint_path = os.path.join(output_dir, checkpoint_names[model_type])

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint already exists: {checkpoint_path}")
        return checkpoint_path

    print(f"Downloading SAM checkpoint: {model_type}")
    print(f"URL: {checkpoint_urls[model_type]}")
    print(f"This may take a few minutes...")

    try:
        import urllib.request
        urllib.request.urlretrieve(checkpoint_urls[model_type], checkpoint_path)
        print(f"Downloaded: {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"Download failed: {e}")
        print(f"Please manually download from:")
        print(f"  {checkpoint_urls[model_type]}")
        raise


if __name__ == "__main__":
    print("SAM Segmentation Module")
    print("=" * 60)
    print("\nSegment Anything Model (SAM) for kidney CT segmentation")
    print("\nUsage:")
    print("  from sam_segmentation import SAMSegmentationPipeline")
    print("  pipeline = SAMSegmentationPipeline(model_type='vit_b')")
    print("  results = pipeline.segment_batch(preprocessed_results)")
    print("\nModel checkpoints:")
    print("  - vit_b: 375 MB (fast, recommended)")
    print("  - vit_l: 1.2 GB (balanced)")
    print("  - vit_h: 2.4 GB (best quality, slow)")
    print("\nDownload checkpoints:")
    print("  python sam_segmentation.py --download vit_b")
