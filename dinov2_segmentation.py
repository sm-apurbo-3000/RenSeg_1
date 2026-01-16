"""
DINOv2 Feature Extraction for Segmentation Module

This module uses DINOv2 (Self-Distillation with No Labels v2) for extracting
rich semantic features that can guide segmentation and clustering.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import os

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class DINOv2FeatureExtractor:
    """DINOv2-based feature extraction for segmentation."""

    def __init__(self,
                 model_name: str = "vit_base_patch14_dinov2.lvd142m",
                 device: Optional[str] = None):
        """
        Initialize DINOv2 feature extractor.

        Args:
            model_name: DINOv2 model name from timm
            device: Device to use
        """
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for DINOv2")

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model_name = model_name

        print(f"Loading DINOv2 model: {model_name}")
        print(f"Device: {self.device}")

        # Load DINOv2 model
        try:
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,  # Remove classification head
                features_only=False
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            print("DINOv2 model loaded successfully")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Falling back to standard ViT...")
            self.model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=True,
                num_classes=0
            )
            self.model = self.model.to(self.device)
            self.model.eval()

        # Get model config
        self.img_size = 224 if '224' in model_name else 518
        self.patch_size = 14 if 'patch14' in model_name else 16

    def extract_features(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract dense features from image.

        Args:
            image: Input image (H, W) or (H, W, 3)

        Returns:
            features: Dense feature tensor (N_patches, feature_dim)
            spatial_shape: (H_patches, W_patches)
        """
        # Preprocess
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize to model input size
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Normalize
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # To tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model(image_tensor)

        # Calculate spatial shape
        num_patches = self.img_size // self.patch_size
        spatial_shape = (num_patches, num_patches)

        return features, spatial_shape

    def extract_patch_features(self, image: np.ndarray) -> Dict:
        """
        Extract patch-level features with spatial information.

        Args:
            image: Input image

        Returns:
            Dictionary with features and spatial info
        """
        # Preprocess
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize to model input size
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Normalize
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # To tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.device)

        # Extract features using forward_features to get patch tokens
        with torch.no_grad():
            # Try to get patch features if model supports it
            if hasattr(self.model, 'forward_features'):
                features = self.model.forward_features(image_tensor)
            else:
                features = self.model(image_tensor)

        # Calculate spatial shape
        num_patches = self.img_size // self.patch_size
        spatial_shape = (num_patches, num_patches)

        # Handle different feature formats
        if len(features.shape) == 2:
            # Global features (B, D) - create spatial grid
            B, D = features.shape
            H, W = spatial_shape
            # Replicate features to create spatial map
            features_spatial = features.unsqueeze(1).repeat(1, H * W, 1)
            features_spatial = features_spatial.reshape(B, H, W, D)
        elif len(features.shape) == 3:
            # Patch features (B, N, D)
            B, N, D = features.shape
            H, W = spatial_shape
            # Reshape to spatial (excluding CLS token if present)
            if N == H * W + 1:
                # Has CLS token - remove it
                features_spatial = features[:, 1:, :].reshape(B, H, W, D)
            else:
                features_spatial = features.reshape(B, H, W, D)
        else:
            features_spatial = features

        return {
            'features': features_spatial.cpu().numpy(),
            'spatial_shape': spatial_shape,
            'feature_dim': features_spatial.shape[-1]
        }

    def compute_similarity_map(self,
                              image: np.ndarray,
                              query_point: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Compute feature similarity map.

        Args:
            image: Input image
            query_point: (x, y) query point (if None, use center)

        Returns:
            Similarity map (H, W)
        """
        features, spatial_shape = self.extract_features(image)
        H, W = spatial_shape

        # Get query feature
        if query_point is None:
            # Use center point
            query_idx = (H // 2) * W + (W // 2)
        else:
            x, y = query_point
            # Map to patch coordinates
            px = int(x / image.shape[1] * W)
            py = int(y / image.shape[0] * H)
            query_idx = py * W + px

        # Reshape features to spatial
        features_2d = features.view(1, H, W, -1)
        query_feat = features_2d[0, query_idx // W, query_idx % W]

        # Compute cosine similarity
        features_flat = features_2d.view(-1, features.shape[-1])
        query_feat_norm = F.normalize(query_feat.unsqueeze(0), dim=1)
        features_norm = F.normalize(features_flat, dim=1)

        similarity = torch.mm(query_feat_norm, features_norm.t())
        similarity_map = similarity.view(H, W).cpu().numpy()

        # Upsample to image size
        similarity_map = cv2.resize(similarity_map, (image.shape[1], image.shape[0]))

        return similarity_map


class DINOv2Segmenter:
    """DINOv2-based segmentation using feature clustering."""

    def __init__(self,
                 feature_extractor: Optional[DINOv2FeatureExtractor] = None,
                 device: Optional[str] = None):
        """
        Initialize DINOv2 segmenter.

        Args:
            feature_extractor: DINOv2 feature extractor
            device: Device to use
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if feature_extractor is None:
            self.feature_extractor = DINOv2FeatureExtractor(device=self.device)
        else:
            self.feature_extractor = feature_extractor

        print("DINOv2 segmenter initialized")

    def segment_with_kmeans(self,
                           image: np.ndarray,
                           n_clusters: int = 5) -> np.ndarray:
        """
        Segment image using K-means on DINOv2 features.

        Args:
            image: Input image
            n_clusters: Number of clusters

        Returns:
            Segmentation map
        """
        from sklearn.cluster import KMeans

        # Extract features
        feat_dict = self.feature_extractor.extract_patch_features(image)
        features = feat_dict['features']

        # Flatten features for clustering
        if len(features.shape) > 2:
            features_flat = features.reshape(-1, features.shape[-1])
        else:
            features_flat = features

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features_flat)

        # Reshape to spatial
        H, W = feat_dict['spatial_shape']
        labels_spatial = labels.reshape(H, W)

        # Upsample to image size
        segmentation = cv2.resize(
            labels_spatial.astype(np.float32),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        return segmentation.astype(np.int32)

    def segment_with_threshold(self,
                               image: np.ndarray,
                               query_point: Optional[Tuple[int, int]] = None,
                               threshold: float = 0.5) -> np.ndarray:
        """
        Segment image using similarity threshold.

        Args:
            image: Input image
            query_point: Query point (if None, use center)
            threshold: Similarity threshold

        Returns:
            Binary segmentation mask
        """
        # Compute similarity map
        similarity = self.feature_extractor.compute_similarity_map(image, query_point)

        # Threshold
        mask = (similarity > threshold).astype(np.uint8)

        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask


class DINOv2SegmentationPipeline:
    """Complete DINOv2 segmentation pipeline."""

    def __init__(self,
                 model_name: str = "vit_base_patch14_dinov2.lvd142m",
                 n_clusters: int = 5,
                 device: Optional[str] = None):
        """
        Initialize DINOv2 segmentation pipeline.

        Args:
            model_name: DINOv2 model name
            n_clusters: Number of clusters for segmentation
            device: Device to use
        """
        try:
            self.feature_extractor = DINOv2FeatureExtractor(model_name, device)
        except Exception as e:
            print(f"Error loading DINOv2: {e}")
            print("Using fallback ViT model...")
            self.feature_extractor = DINOv2FeatureExtractor("vit_base_patch16_224", device)

        self.segmenter = DINOv2Segmenter(self.feature_extractor, device)
        self.n_clusters = n_clusters

    def segment_preprocessed_image(self,
                                   preprocessed_result: Dict,
                                   method: str = 'kmeans') -> Dict:
        """
        Segment a preprocessed image result.

        Args:
            preprocessed_result: Result from KidneyPreprocessingPipeline
            method: Segmentation method ('kmeans' or 'threshold')

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

        # Segment based on method
        if method == 'kmeans':
            segmentation = self.segmenter.segment_with_kmeans(image, self.n_clusters)
            num_segments = self.n_clusters
        else:  # threshold
            # Use ROI center if available
            roi_info = preprocessed_result['preprocessed'].get('roi_info', {})
            if roi_info.get('detected', False):
                bbox = roi_info['expanded_bbox']
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                query_point = (center_x, center_y)
            else:
                query_point = None

            segmentation = self.segmenter.segment_with_threshold(image, query_point)
            from scipy.ndimage import label
            labeled, num_segments = label(segmentation)
            segmentation = labeled

        return {
            'segmentation': segmentation,
            'num_segments': num_segments,
            'image_shape': image.shape[:2],
            'method': f'dinov2_{method}',
            'model_type': 'dinov2'
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

    def extract_features_for_clustering(self,
                                       preprocessed_results: List[Dict],
                                       verbose: bool = True) -> np.ndarray:
        """
        Extract DINOv2 features for downstream clustering.

        Args:
            preprocessed_results: List of preprocessing results
            verbose: Print progress

        Returns:
            Feature matrix (N_images, feature_dim)
        """
        features_list = []
        total = len(preprocessed_results)

        for idx, prep_result in enumerate(preprocessed_results):
            if verbose and (idx + 1) % 10 == 0:
                print(f"Extracting features: {idx + 1}/{total}")

            try:
                image = prep_result['preprocessed']['clahe']
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)

                feat_dict = self.feature_extractor.extract_patch_features(image)
                features = feat_dict['features']

                # Use mean pooling to get global feature
                if len(features.shape) > 2:
                    features_global = features.mean(axis=(0, 1))
                else:
                    features_global = features.squeeze()

                features_list.append(features_global)

            except Exception as e:
                print(f"Error extracting features from image {idx}: {e}")

        if verbose:
            print(f"Feature extraction complete: {len(features_list)} images")

        return np.array(features_list)


if __name__ == "__main__":
    print("DINOv2 Segmentation Module")
    print("=" * 60)
    print("\nDINOv2 self-supervised features for segmentation")
    print("\nUsage:")
    print("  from dinov2_segmentation import DINOv2SegmentationPipeline")
    print("  pipeline = DINOv2SegmentationPipeline()")
    print("  results = pipeline.segment_batch(preprocessed_results)")
    print("\nFeatures:")
    print("  - Self-supervised learning (no labels needed)")
    print("  - Rich semantic features")
    print("  - K-means and threshold-based segmentation")
    print("  - Feature extraction for clustering")
    print("\nModel: DINOv2 ViT-B/14")
