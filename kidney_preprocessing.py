"""
Kidney CT Image Preprocessing for Unsupervised Segmentation

This module implements a comprehensive preprocessing pipeline for kidney CT images,
designed to prepare data for SAM segmentation and Torque clustering.

Based on methodologies from:
1. Zeng et al. - Unsupervised Domain Translation for Kidney Segmentation (ISBI 2021)
2. RenSeg (Faruk et al.) - Contour-Guided Quickshift Segmentation (IEEE JBHI 2025)
3. Salehinejad et al. - Radial Transform Sampling (GlobalSIP 2018)

Key Features:
- HU Windowing: Proper CT intensity handling for kidney ROI extraction
- Multi-Feature Extraction: Beyond intensity-only (texture, spatial, edge features)
- Anatomically Constrained Post-Processing: Leveraging kidney anatomy knowledge
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Image processing
from skimage import exposure, filters, morphology, measure
from skimage.segmentation import quickshift
from skimage.filters import sobel, gabor, gaussian
from skimage.feature import local_binary_pattern
from skimage.util import img_as_float
from scipy import ndimage
from scipy.ndimage import label, binary_fill_holes, binary_dilation
from scipy.signal import find_peaks

# Kidney localization
try:
    from kidney_localizer import KidneyLocalizer, LocalizationAwareCropper
    LOCALIZATION_AVAILABLE = True
except ImportError:
    LOCALIZATION_AVAILABLE = False
    print("Warning: kidney_localizer not available. Localization disabled.")


class HUWindowProcessor:
    """Hounsfield Unit Windowing for CT Images (adaptive for pre-processed images)."""

    def __init__(self, adaptive: bool = True):
        self.adaptive = adaptive

    def apply_window(self, image: np.ndarray, center: float, width: float) -> np.ndarray:
        """Apply CT windowing transformation."""
        img_min = center - width / 2
        img_max = center + width / 2
        windowed = np.clip(image, img_min, img_max)
        return ((windowed - img_min) / (img_max - img_min)).astype(np.float32)

    def estimate_params(self, image: np.ndarray) -> Dict[str, float]:
        """Estimate window parameters from histogram."""
        img_float = image.astype(np.float32)
        hist, bins = np.histogram(img_float.flatten(), bins=256)
        hist_smooth = ndimage.gaussian_filter1d(hist.astype(float), sigma=3)
        peaks, _ = find_peaks(hist_smooth, height=hist_smooth.max() * 0.1, distance=20)

        if len(peaks) > 0:
            center = np.median(bins[peaks])
            width = np.percentile(img_float, 95) - np.percentile(img_float, 5)
        else:
            center = np.percentile(img_float, 50)
            width = np.percentile(img_float, 95) - np.percentile(img_float, 5)

        return {'center': center, 'width': max(width, 50)}

    def adaptive_kidney_window(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive windowing for kidney visualization."""
        params = self.estimate_params(image)
        params['center'] *= 1.1
        params['width'] *= 0.8
        return self.apply_window(image, params['center'], params['width'])

    def multi_window_fusion(self, image: np.ndarray) -> np.ndarray:
        """Fuse multiple window settings."""
        params = self.estimate_params(image)
        soft = self.apply_window(image, params['center'], params['width'])
        high = self.apply_window(image, params['center'] * 1.5, params['width'] * 0.5)
        low = self.apply_window(image, params['center'] * 0.8, params['width'] * 1.2)
        return (0.5 * soft + 0.25 * high + 0.25 * low).astype(np.float32)


class ImagePreprocessor:
    """Comprehensive image preprocessing pipeline."""

    def __init__(self, target_size: Tuple[int, int] = (256, 256), use_localization: bool = False):
        self.target_size = target_size
        self.hu_processor = HUWindowProcessor(adaptive=True)
        self.use_localization = use_localization and LOCALIZATION_AVAILABLE

        # Initialize localizer if enabled
        if self.use_localization:
            self.localizer = KidneyLocalizer(use_pretrained=True)
            self.cropper = LocalizationAwareCropper(target_size=target_size)
        else:
            self.localizer = None
            self.cropper = None

    def load_image(self, path: str) -> np.ndarray:
        """Load image from file path."""
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load: {path}")
        # Ensure 2D array (some images may have shape (H, W, 1))
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image.squeeze()
        return image

    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(image)

    def morphological_clean(self, image: np.ndarray, k: int = 5) -> np.ndarray:
        """Apply morphological opening and closing."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    def gaussian_smooth(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing."""
        return gaussian(image, sigma=sigma, preserve_range=True).astype(image.dtype)

    def extract_edges(self, image: np.ndarray) -> np.ndarray:
        """Extract edge features using Sobel."""
        return sobel(img_as_float(image)).astype(np.float32)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        image = image.astype(np.float32)
        if image.max() - image.min() > 0:
            return (image - image.min()) / (image.max() - image.min())
        return image

    def preprocess(self, image: np.ndarray, verbose: bool = False) -> Dict[str, np.ndarray]:
        """Complete preprocessing pipeline."""
        results = {}

        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Localization step (optional)
        if self.use_localization and self.localizer is not None:
            roi, roi_info = self.localizer.localize_and_crop(image, expand_ratio=0.15, verbose=verbose)
            results['roi_info'] = roi_info
            results['localization_mask'] = self.localizer.create_localization_mask(image.shape, roi_info)

            # Use cropper for intelligent resizing
            resized = self.cropper.smart_crop_and_resize(image, roi_info)
            results['localized'] = True

            if verbose:
                print(f"  Localization: {roi_info['method']}, Conf: {roi_info.get('confidence', 0):.3f}")
        else:
            # Standard resize without localization
            resized = self.resize(image)
            results['localized'] = False

        results['original_resized'] = resized

        # HU windowing
        hu = self.hu_processor.adaptive_kidney_window(resized.astype(np.float32))
        results['hu_windowed'] = hu
        results['multi_window'] = self.hu_processor.multi_window_fusion(resized.astype(np.float32))

        # CLAHE
        clahe_in = (hu * 255).astype(np.uint8) if hu.max() <= 1 else hu.astype(np.uint8)
        results['clahe'] = self.apply_clahe(clahe_in)

        # Morphological
        results['morphological'] = self.morphological_clean(results['clahe'])

        # Smoothing
        results['smoothed'] = self.gaussian_smooth(results['morphological'].astype(np.float32))

        # Edges
        results['edges'] = self.extract_edges(results['smoothed'])

        # Final
        results['final'] = self.normalize(results['smoothed'])

        return results


class MultiFeatureExtractor:
    """Extract multiple features for robust clustering."""

    def __init__(self, use_intensity=True, use_texture=True, use_spatial=True,
                 use_edges=True, use_gabor=False):
        self.use_intensity = use_intensity
        self.use_texture = use_texture
        self.use_spatial = use_spatial
        self.use_edges = use_edges
        self.use_gabor = use_gabor
        self.lbp_radius = 3
        self.lbp_points = 24

    def _norm(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        arr = arr.astype(np.float32)
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / rng if rng > 0 else arr

    def intensity_features(self, image: np.ndarray) -> np.ndarray:
        """Extract intensity-based features."""
        h, w = image.shape[:2]
        intensity = self._norm(image).reshape(h, w, 1)
        local_mean = gaussian(image, sigma=5, preserve_range=True)
        local_mean = self._norm(local_mean).reshape(h, w, 1)
        local_std = ndimage.generic_filter(image.astype(np.float64), np.std, size=5)
        local_std = self._norm(local_std).reshape(h, w, 1)
        return np.concatenate([intensity, local_mean, local_std], axis=2)

    def texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features using Local Binary Pattern."""
        img_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        lbp = local_binary_pattern(img_uint8, self.lbp_points, self.lbp_radius, method='uniform')
        return self._norm(lbp).reshape(image.shape[0], image.shape[1], 1)

    def spatial_features(self, image: np.ndarray) -> np.ndarray:
        """Extract spatial position features."""
        h, w = image.shape[:2]
        y, x = np.mgrid[0:h, 0:w]

        x_norm = x.astype(np.float32) / (w - 1)
        y_norm = y.astype(np.float32) / (h - 1)

        cx, cy = w / 2, h / 2
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        dist_norm = self._norm(dist)

        theta = np.arctan2(y - cy, x - cx)
        theta_norm = (theta + np.pi) / (2 * np.pi)

        return np.stack([x_norm, y_norm, dist_norm, theta_norm], axis=2).astype(np.float32)

    def edge_features(self, image: np.ndarray) -> np.ndarray:
        """Extract edge and gradient features."""
        img_float = img_as_float(image)

        sobel_x = ndimage.sobel(img_float, axis=1)
        sobel_y = ndimage.sobel(img_float, axis=0)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        grad_dir = np.arctan2(sobel_y, sobel_x)
        grad_dir_norm = (grad_dir + np.pi) / (2 * np.pi)
        laplacian = ndimage.laplace(img_float)

        return np.stack([
            self._norm(sobel_x), self._norm(sobel_y),
            self._norm(grad_mag), grad_dir_norm,
            self._norm(np.abs(laplacian))
        ], axis=2).astype(np.float32)

    def gabor_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Gabor filter features."""
        img_float = img_as_float(image)
        features = []
        for freq in [0.1, 0.2, 0.4]:
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                real, imag = gabor(img_float, frequency=freq, theta=theta)
                features.append(self._norm(np.sqrt(real**2 + imag**2)))
        return np.stack(features, axis=2).astype(np.float32)

    def extract_all(self, image: np.ndarray, preprocessed: Dict = None,
                    verbose: bool = False) -> np.ndarray:
        """Extract all selected features."""
        features = []

        img = preprocessed.get('final', image) if preprocessed else image
        img_tex = preprocessed.get('clahe', image) if preprocessed else image
        img_edge = preprocessed.get('smoothed', image) if preprocessed else image

        if self.use_intensity:
            f = self.intensity_features(img)
            features.append(f)
            if verbose: print(f"  Intensity: {f.shape}")

        if self.use_texture:
            f = self.texture_features(img_tex)
            features.append(f)
            if verbose: print(f"  Texture: {f.shape}")

        if self.use_spatial:
            f = self.spatial_features(image)
            features.append(f)
            if verbose: print(f"  Spatial: {f.shape}")

        if self.use_edges:
            f = self.edge_features(img_edge)
            features.append(f)
            if verbose: print(f"  Edges: {f.shape}")

        if self.use_gabor:
            f = self.gabor_features(img_tex)
            features.append(f)
            if verbose: print(f"  Gabor: {f.shape}")

        all_feat = np.concatenate(features, axis=2)
        if verbose: print(f"  Total: {all_feat.shape}")
        return all_feat


class AnatomicalConstraints:
    """Apply anatomical constraints for kidney segmentation."""

    def __init__(self, min_size: int = 1000, max_size: int = 50000):
        self.min_size = min_size
        self.max_size = max_size

    def create_roi_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create region of interest mask."""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.float32)
        mask[int(0.2*h):int(0.8*h), int(0.1*w):int(0.9*w)] = 1.0
        return gaussian(mask, sigma=20, preserve_range=True)

    def filter_by_size(self, mask: np.ndarray) -> np.ndarray:
        """Filter regions by size constraints."""
        labeled, _ = label(mask > 0)
        regions = measure.regionprops(labeled)

        filtered = np.zeros_like(mask)
        valid = [(r, r.area) for r in regions if self.min_size <= r.area <= self.max_size]
        valid.sort(key=lambda x: x[1], reverse=True)

        for i, (region, _) in enumerate(valid[:4]):
            filtered[labeled == region.label] = i + 1
        return filtered

    def apply_shape_constraints(self, mask: np.ndarray) -> np.ndarray:
        """Apply shape-based constraints."""
        filled = binary_fill_holes(mask > 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


class ContourGuidedQuickshift:
    """Contour-guided quickshift from RenSeg paper."""

    def __init__(self, kernel_size: int = 3, ratio: float = 0.5,
                 max_dist: float = 8, sigma: float = 1.0, merge_tau: float = 0.05):
        self.kernel_size = kernel_size
        self.ratio = ratio
        self.max_dist = max_dist
        self.sigma = sigma
        self.merge_tau = merge_tau

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Apply contour-guided quickshift segmentation."""
        if image.dtype != np.float64:
            image = image.astype(np.float64)
        if image.max() > 1:
            image = image / image.max()

        # Convert to RGB for quickshift
        image_rgb = np.stack([image] * 3, axis=2) if len(image.shape) == 2 else image

        # Initial segmentation
        segments = quickshift(image_rgb, kernel_size=self.kernel_size,
                              max_dist=self.max_dist, ratio=self.ratio, sigma=self.sigma)

        # Merge similar regions
        segments = self._merge_regions(image, segments)
        return segments

    def _merge_regions(self, image: np.ndarray, segments: np.ndarray) -> np.ndarray:
        """Merge similar adjacent regions."""
        means = {lbl: np.mean(image[segments == lbl]) for lbl in np.unique(segments)}

        changed = True
        iters = 0
        while changed and iters < 50:
            changed = False
            iters += 1

            for lbl in list(means.keys()):
                if lbl not in means:
                    continue

                mask = segments == lbl
                dilated = binary_dilation(mask, iterations=1)
                neighbors = np.unique(segments[dilated & ~mask])

                for nbr in neighbors:
                    if nbr not in means:
                        continue
                    if abs(means[lbl] - means[nbr]) < self.merge_tau:
                        new_lbl = min(lbl, nbr)
                        old_lbl = max(lbl, nbr)
                        segments[segments == old_lbl] = new_lbl
                        means[new_lbl] = (means[new_lbl] + means[old_lbl]) / 2
                        del means[old_lbl]
                        changed = True
                        break

        return segments


class KidneyPreprocessingPipeline:
    """Complete preprocessing pipeline for kidney CT images."""

    def __init__(self, target_size=(256, 256), extract_features=True,
                 use_quickshift=True, use_gabor=False, use_localization=False):
        self.target_size = target_size
        self.extract_features = extract_features
        self.use_quickshift = use_quickshift
        self.use_localization = use_localization

        self.preprocessor = ImagePreprocessor(target_size, use_localization=use_localization)
        self.feature_extractor = MultiFeatureExtractor(
            use_intensity=True, use_texture=True,
            use_spatial=True, use_edges=True, use_gabor=use_gabor
        )
        self.quickshift = ContourGuidedQuickshift()
        self.constraints = AnatomicalConstraints()

    def process_image(self, path: str, verbose: bool = False) -> Dict:
        """Process a single image through the pipeline."""
        result = {'path': path}

        image = self.preprocessor.load_image(path)
        result['original'] = image

        preprocessed = self.preprocessor.preprocess(image)
        result['preprocessed'] = preprocessed

        if self.extract_features:
            result['features'] = self.feature_extractor.extract_all(
                preprocessed['final'], preprocessed, verbose
            )

        if self.use_quickshift:
            result['segments'] = self.quickshift.segment(preprocessed['final'])

        result['roi_mask'] = self.constraints.create_roi_mask(self.target_size)

        return result

    def process_batch(self, paths: List[str], verbose: bool = True) -> List[Dict]:
        """Process multiple images."""
        results = []
        for i, path in enumerate(paths):
            if verbose and i % 10 == 0:
                print(f"Processing {i+1}/{len(paths)}")
            try:
                results.append(self.process_image(path))
            except Exception as e:
                print(f"Error: {path}: {e}")
                results.append({'path': path, 'error': str(e)})
        return results

    def prepare_for_clustering(self, results: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for Torque clustering."""
        all_features = []
        image_indices = []
        paths = []

        for idx, result in enumerate(results):
            if 'features' not in result:
                continue

            features = result['features']
            h, w, n_feat = features.shape

            all_features.append(features.reshape(-1, n_feat))
            image_indices.extend([idx] * (h * w))
            paths.append(result['path'])

        return np.vstack(all_features), np.array(image_indices), paths


def load_dataset(data_dir: str, classes: List[str] = None,
                 max_per_class: int = None) -> Dict[str, List[str]]:
    """Load CT Kidney Dataset.

    Args:
        data_dir: Path to dataset directory
        classes: List of class names to load (default: ['Normal', 'Cyst', 'Tumor', 'Stone'])
        max_per_class: Maximum number of images per class

    Returns:
        Dictionary mapping class names to lists of image paths
    """
    if classes is None:
        classes = ['Normal', 'Cyst', 'Tumor', 'Stone']

    dataset = {}
    base_data_path = os.path.join(data_dir, 'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone')
    base_data_path = os.path.join(base_data_path, 'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone')

    for cls in classes:
        cls_dir = os.path.join(base_data_path, cls)
        if not os.path.exists(cls_dir):
            print(f"Warning: {cls_dir} not found")
            continue

        paths = sorted([str(p) for p in Path(cls_dir).glob('*.jpg')] +
                       [str(p) for p in Path(cls_dir).glob('*.png')])

        if max_per_class:
            paths = paths[:max_per_class]

        dataset[cls] = paths
        print(f"  {cls}: {len(paths)} images")

    return dataset

