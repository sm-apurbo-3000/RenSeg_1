"""
Kidney Localization Module using YOLOv8

This module provides kidney detection and localization capabilities using YOLOv8
to identify kidney regions of interest (ROI) before preprocessing.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional
from ultralytics import YOLO
import os


class KidneyLocalizer:
    """YOLOv8-based kidney localization for CT images."""

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45, use_pretrained: bool = True):
        """
        Initialize the kidney localizer.

        Args:
            model_path: Path to custom trained YOLO model. If None, uses pretrained YOLOv8n
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            use_pretrained: Whether to use pretrained YOLOv8 (fallback for general object detection)
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.model_path = model_path

        # Initialize model
        if model_path and os.path.exists(model_path):
            print(f"Loading custom YOLO model from: {model_path}")
            self.model = YOLO(model_path)
        elif use_pretrained:
            print("Loading pretrained YOLOv8n model...")
            self.model = YOLO('yolov8n.pt')  # Nano model for speed
        else:
            raise ValueError("No valid model path provided and use_pretrained=False")

    def detect_kidneys(self, image: np.ndarray, verbose: bool = False) -> List[Dict]:
        """
        Detect kidney regions in the image.

        Args:
            image: Input image (grayscale or RGB)
            verbose: Print detection details

        Returns:
            List of detection dictionaries with keys: bbox, confidence, class_id
        """
        # Ensure RGB format for YOLO
        if len(image.shape) == 2:
            # Grayscale image - convert to RGB by stacking
            # Ensure uint8 type
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Single channel with explicit dimension
            image_2d = image.squeeze()
            if image_2d.dtype != np.uint8:
                if image_2d.max() <= 1.0:
                    image_2d = (image_2d * 255).astype(np.uint8)
                else:
                    image_2d = image_2d.astype(np.uint8)
            image_rgb = cv2.cvtColor(image_2d, cv2.COLOR_GRAY2BGR)
        else:
            # Already RGB/BGR
            image_rgb = image

        # Run inference
        results = self.model(image_rgb, conf=self.confidence_threshold,
                            iou=self.iou_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                detections.append({
                    'bbox': bbox,
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': self.model.names[cls_id] if cls_id < len(self.model.names) else 'unknown'
                })

                if verbose:
                    print(f"  Detection: {self.model.names[cls_id]}, Conf: {conf:.3f}, BBox: {bbox}")

        return detections

    def get_roi_from_detections(self, image: np.ndarray, detections: List[Dict],
                                 expand_ratio: float = 0.1) -> Tuple[np.ndarray, Dict]:
        """
        Extract ROI from detections.

        Args:
            image: Original image
            detections: List of detections from detect_kidneys()
            expand_ratio: Expand bounding box by this ratio (0.1 = 10% expansion)

        Returns:
            roi_image: Cropped region of interest
            roi_info: Dictionary with bbox coordinates and metadata
        """
        h, w = image.shape[:2]

        if not detections:
            # No detection - return full image with fallback ROI
            return image, {
                'bbox': [0, 0, w, h],
                'expanded_bbox': [0, 0, w, h],
                'detected': False,
                'confidence': 0.0,
                'method': 'fallback_full_image'
            }

        # Get highest confidence detection
        best_detection = max(detections, key=lambda x: x['confidence'])
        x1, y1, x2, y2 = best_detection['bbox']

        # Expand bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        expand_w = bbox_w * expand_ratio
        expand_h = bbox_h * expand_ratio

        x1_exp = max(0, int(x1 - expand_w))
        y1_exp = max(0, int(y1 - expand_h))
        x2_exp = min(w, int(x2 + expand_w))
        y2_exp = min(h, int(y2 + expand_h))

        # Extract ROI
        roi = image[y1_exp:y2_exp, x1_exp:x2_exp]

        roi_info = {
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'expanded_bbox': [x1_exp, y1_exp, x2_exp, y2_exp],
            'detected': True,
            'confidence': best_detection['confidence'],
            'class_name': best_detection['class_name'],
            'method': 'yolo_detection',
            'original_size': (w, h),
            'roi_size': (x2_exp - x1_exp, y2_exp - y1_exp)
        }

        return roi, roi_info

    def localize_and_crop(self, image: np.ndarray, expand_ratio: float = 0.1,
                         verbose: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        One-step localization and cropping.

        Args:
            image: Input image
            expand_ratio: Expand bounding box by this ratio
            verbose: Print details

        Returns:
            roi_image: Cropped ROI
            roi_info: ROI metadata
        """
        detections = self.detect_kidneys(image, verbose=verbose)
        roi, roi_info = self.get_roi_from_detections(image, detections, expand_ratio)

        return roi, roi_info

    def create_localization_mask(self, image_shape: Tuple[int, int],
                                roi_info: Dict) -> np.ndarray:
        """
        Create a binary mask for the localized region.

        Args:
            image_shape: (height, width) of original image
            roi_info: ROI information from get_roi_from_detections

        Returns:
            Binary mask (0-1) with 1s in ROI region
        """
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.float32)

        if roi_info['detected']:
            x1, y1, x2, y2 = roi_info['expanded_bbox']
            mask[y1:y2, x1:x2] = 1.0
        else:
            mask[:, :] = 1.0  # Full image if no detection

        return mask


class LocalizationAwareCropper:
    """Handles intelligent cropping based on localization results."""

    def __init__(self, target_size: Tuple[int, int] = (256, 256),
                 min_roi_size: int = 64):
        """
        Initialize cropper.

        Args:
            target_size: Target size for resized ROI
            min_roi_size: Minimum ROI size to consider valid
        """
        self.target_size = target_size
        self.min_roi_size = min_roi_size

    def smart_crop_and_resize(self, image: np.ndarray, roi_info: Dict) -> np.ndarray:
        """
        Intelligently crop and resize based on ROI info.

        Args:
            image: Original image
            roi_info: ROI information from localization

        Returns:
            Cropped and resized image
        """
        if not roi_info['detected']:
            # No detection - resize full image
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

        x1, y1, x2, y2 = roi_info['expanded_bbox']
        roi_w, roi_h = roi_info['roi_size']

        # Check if ROI is too small
        if roi_w < self.min_roi_size or roi_h < self.min_roi_size:
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

        # Extract and resize ROI
        roi = image[y1:y2, x1:x2]
        resized = cv2.resize(roi, self.target_size, interpolation=cv2.INTER_LINEAR)

        return resized

    def maintain_aspect_ratio_crop(self, image: np.ndarray, roi_info: Dict,
                                   padding: bool = True) -> np.ndarray:
        """
        Crop maintaining aspect ratio with optional padding.

        Args:
            image: Original image
            roi_info: ROI information
            padding: Add padding to maintain aspect ratio

        Returns:
            Processed image
        """
        if not roi_info['detected']:
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

        x1, y1, x2, y2 = roi_info['expanded_bbox']
        roi = image[y1:y2, x1:x2]

        roi_h, roi_w = roi.shape[:2]
        target_h, target_w = self.target_size

        # Calculate aspect ratios
        roi_aspect = roi_w / roi_h
        target_aspect = target_w / target_h

        if padding:
            # Resize to fit within target, then pad
            if roi_aspect > target_aspect:
                # ROI is wider - fit to width
                new_w = target_w
                new_h = int(target_w / roi_aspect)
            else:
                # ROI is taller - fit to height
                new_h = target_h
                new_w = int(target_h * roi_aspect)

            resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Create padded image
            if len(image.shape) == 2:
                padded = np.zeros(self.target_size, dtype=resized.dtype)
            else:
                padded = np.zeros((*self.target_size, image.shape[2]), dtype=resized.dtype)

            # Center the resized ROI
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

            return padded
        else:
            # Simple resize without padding
            return cv2.resize(roi, self.target_size, interpolation=cv2.INTER_LINEAR)


if __name__ == "__main__":
    print("Kidney Localizer Module")
    print("=" * 50)
    print("\nThis module provides YOLOv8-based kidney localization.")
    print("\nUsage:")
    print("  from kidney_localizer import KidneyLocalizer")
    print("  localizer = KidneyLocalizer()")
    print("  roi, roi_info = localizer.localize_and_crop(image)")
