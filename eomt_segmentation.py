"""
Encoder-only Mask Transformer (EoMT) Segmentation Module

This module implements a simplified encoder-only mask transformer for
efficient kidney CT image segmentation.
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
    print("Warning: timm not installed. Run: pip install timm")


class PatchEmbedding(nn.Module):
    """Patch embedding layer for image tokenization."""

    def __init__(self, img_size=256, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x, (H, W)


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with multi-head self-attention."""

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class MaskDecoder(nn.Module):
    """Lightweight mask decoder for segmentation."""

    def __init__(self, embed_dim=768, num_classes=1, img_size=256, patch_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_size = patch_size

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Upsampling to original resolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x, spatial_shape):
        # x: (B, num_patches, embed_dim)
        B, N, C = x.shape
        H, W = spatial_shape

        # Reshape to spatial
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # Decode
        x = self.decoder(x)
        x = self.upsample(x)

        return x


class EoMTModel(nn.Module):
    """Encoder-only Mask Transformer (EoMT) for segmentation."""

    def __init__(self,
                 img_size=256,
                 patch_size=16,
                 in_channels=1,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1,
                 dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Positional embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer encoder
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Mask decoder
        self.decoder = MaskDecoder(embed_dim, num_classes, img_size, patch_size)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, C, H, W)
        B = x.shape[0]

        # Patch embedding
        x, spatial_shape = self.patch_embed(x)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer encoder
        for block in self.encoder:
            x = block(x)

        x = self.norm(x)

        # Decode to mask
        mask = self.decoder(x, spatial_shape)

        return mask

    def predict_mask(self, x):
        """Predict segmentation mask."""
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 1:
                # Binary segmentation
                mask = torch.sigmoid(logits) > 0.5
            else:
                # Multi-class segmentation
                mask = torch.argmax(logits, dim=1)
        return mask


class EoMTSegmenter:
    """EoMT-based segmenter for kidney CT images."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 img_size: int = 256,
                 patch_size: int = 16,
                 device: Optional[str] = None,
                 use_pretrained: bool = False):
        """
        Initialize EoMT segmenter.

        Args:
            model_path: Path to trained model checkpoint
            img_size: Input image size
            patch_size: Patch size for tokenization
            device: Device to use
            use_pretrained: Use pretrained weights (if available)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.img_size = img_size
        self.patch_size = patch_size

        # Initialize model
        self.model = EoMTModel(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,
            embed_dim=384,  # Smaller for efficiency
            depth=6,        # Fewer layers
            num_heads=6,
            mlp_ratio=4.0,
            num_classes=1,  # Binary segmentation
            dropout=0.1
        ).to(self.device)

        # Load checkpoint if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading EoMT checkpoint: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded successfully")
        else:
            print("Using randomly initialized EoMT model")
            print("Note: For best results, train the model on kidney CT data")

        self.model.eval()

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment a single image.

        Args:
            image: Input image (H, W) or (H, W, 3)

        Returns:
            Binary segmentation mask (H, W)
        """
        # Preprocess
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize if needed
        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = cv2.resize(image, (self.img_size, self.img_size))

        # Normalize
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            image = image / 255.0

        # To tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.device)

        # Predict
        with torch.no_grad():
            mask = self.model.predict_mask(image_tensor)

        # To numpy
        mask_np = mask.squeeze().cpu().numpy()

        return mask_np.astype(np.uint8)

    def segment_with_postprocessing(self,
                                   image: np.ndarray,
                                   min_area: int = 100,
                                   fill_holes: bool = True) -> np.ndarray:
        """
        Segment with morphological postprocessing.

        Args:
            image: Input image
            min_area: Minimum connected component area
            fill_holes: Fill holes in mask

        Returns:
            Processed segmentation mask
        """
        # Get initial mask
        mask = self.segment(image)

        # Fill holes
        if fill_holes:
            from scipy.ndimage import binary_fill_holes
            mask = binary_fill_holes(mask).astype(np.uint8)

        # Remove small components
        from scipy.ndimage import label
        labeled, num_features = label(mask)
        for i in range(1, num_features + 1):
            component = labeled == i
            if component.sum() < min_area:
                mask[component] = 0

        return mask


class EoMTSegmentationPipeline:
    """Complete EoMT segmentation pipeline for kidney CT images."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 img_size: int = 256,
                 patch_size: int = 16,
                 device: Optional[str] = None,
                 min_area: int = 100):
        """
        Initialize EoMT segmentation pipeline.

        Args:
            model_path: Path to trained model
            img_size: Input image size
            patch_size: Patch size
            device: Device to use
            min_area: Minimum segment area
        """
        self.segmenter = EoMTSegmenter(model_path, img_size, patch_size, device)
        self.min_area = min_area

    def segment_preprocessed_image(self, preprocessed_result: Dict) -> Dict:
        """
        Segment a preprocessed image result.

        Args:
            preprocessed_result: Result from KidneyPreprocessingPipeline

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

        # Segment
        mask = self.segmenter.segment_with_postprocessing(
            image,
            min_area=self.min_area,
            fill_holes=True
        )

        # Label connected components
        from scipy.ndimage import label
        labeled, num_segments = label(mask)

        return {
            'segmentation': labeled,
            'num_segments': num_segments,
            'image_shape': image.shape[:2],
            'method': 'eomt',
            'model_type': 'encoder_only_mask_transformer'
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


# Training utilities (optional)
class EoMTTrainer:
    """Trainer for EoMT model."""

    def __init__(self,
                 model: EoMTModel,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_step(self, images, masks):
        """Single training step."""
        self.model.train()

        images = images.to(self.device)
        masks = masks.to(self.device)

        # Forward
        outputs = self.model(images)
        loss = self.criterion(outputs, masks)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
        print(f"Checkpoint saved: {path}")


if __name__ == "__main__":
    print("EoMT Segmentation Module")
    print("=" * 60)
    print("\nEncoder-only Mask Transformer for kidney CT segmentation")
    print("\nUsage:")
    print("  from eomt_segmentation import EoMTSegmentationPipeline")
    print("  pipeline = EoMTSegmentationPipeline()")
    print("  results = pipeline.segment_batch(preprocessed_results)")
    print("\nModel architecture:")
    print("  - Patch size: 16x16")
    print("  - Embedding dim: 384")
    print("  - Transformer depth: 6 layers")
    print("  - Attention heads: 6")
    print("\nNote: This model uses random initialization.")
    print("For best results, train on kidney-specific data.")
