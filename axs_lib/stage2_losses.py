"""
Stage 2: Prithvi Refinement Loss Functions

This module implements specialized loss functions for Stage 2 of the Axion-Sat pipeline,
ensuring that Prithvi's refinement maintains spectral plausibility, geometric consistency,
and realistic texture detail.

Loss Components:
    1. Spectral Plausibility Loss
       - NDVI/EVI RMSE: Ensures vegetation indices remain physically plausible
       - Spectral Angle Mapper (SAM): Validates spectral similarity to reference S2/HLS
       
    2. Identity/Edge-Guard Loss
       - Spatial Consistency: Penalizes geometric changes from input
       - Edge Preservation: Maintains structural information from Stage 1
       
    3. PatchGAN Discriminator
       - Lightweight adversarial loss for texture detail
       - Low weight (0.05) to avoid overwhelming spectral constraints

Philosophy:
    Stage 2 refinement should enhance features WITHOUT altering fundamental spectral
    relationships or spatial structure. These losses ensure Prithvi acts as a refinement
    filter rather than a generative model.

Author: Axion-Sat Project
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union


# ============================================================================
# Vegetation Index Computation
# ============================================================================

def compute_ndvi(
    red: torch.Tensor,
    nir: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Normalized Difference Vegetation Index (NDVI).
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Range: [-1, 1]
    - Healthy vegetation: 0.3 to 0.8
    - Bare soil: ~0.1
    - Water: < 0
    
    Args:
        red: Red band (B04) tensor, shape (B, 1, H, W) or (B, H, W)
        nir: NIR band (B08) tensor, shape (B, 1, H, W) or (B, H, W)
        eps: Small constant for numerical stability
        
    Returns:
        NDVI tensor, shape (B, 1, H, W) or (B, H, W)
    """
    numerator = nir - red
    denominator = nir + red + eps
    return numerator / denominator


def compute_evi(
    blue: torch.Tensor,
    red: torch.Tensor,
    nir: torch.Tensor,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Enhanced Vegetation Index (EVI).
    
    EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)
    
    Range: [-1, 1] (typically 0 to 1 for vegetation)
    - More sensitive to canopy variations than NDVI
    - Better performance in high biomass areas
    
    Args:
        blue: Blue band (B02) tensor, shape (B, 1, H, W) or (B, H, W)
        red: Red band (B04) tensor, shape (B, 1, H, W) or (B, H, W)
        nir: NIR band (B08) tensor, shape (B, 1, H, W) or (B, H, W)
        G: Gain factor (default: 2.5)
        C1: Red correction coefficient (default: 6.0)
        C2: Blue correction coefficient (default: 7.5)
        L: Canopy background adjustment (default: 1.0)
        eps: Small constant for numerical stability
        
    Returns:
        EVI tensor, shape (B, 1, H, W) or (B, H, W)
    """
    numerator = G * (nir - red)
    denominator = nir + C1 * red - C2 * blue + L + eps
    return numerator / denominator


# ============================================================================
# Spectral Plausibility Loss
# ============================================================================

class SpectralPlausibilityLoss(nn.Module):
    """
    Ensures refined outputs maintain physically plausible spectral relationships.
    
    Components:
        - NDVI RMSE: Penalizes deviation from expected vegetation index
        - EVI RMSE: Secondary vegetation index validation
        - Spectral Angle Mapper (SAM): Validates spectral similarity to reference
        
    This loss ensures that Stage 2 refinement doesn't introduce spectral artifacts
    or unrealistic vegetation signatures.
    """
    
    def __init__(
        self,
        ndvi_weight: float = 1.0,
        evi_weight: float = 0.5,
        sam_weight: float = 0.3,
        use_sam: bool = True,
    ):
        """
        Args:
            ndvi_weight: Weight for NDVI RMSE loss
            evi_weight: Weight for EVI RMSE loss
            sam_weight: Weight for Spectral Angle Mapper loss
            use_sam: Whether to compute SAM loss (requires reference data)
        """
        super().__init__()
        
        self.ndvi_weight = ndvi_weight
        self.evi_weight = evi_weight
        self.sam_weight = sam_weight
        self.use_sam = use_sam
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reference: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute spectral plausibility loss.
        
        Args:
            pred: Predicted refined optical, shape (B, 4, H, W)
                  Channels: [B02, B03, B04, B08]
            target: Target optical (Stage 1 output), shape (B, 4, H, W)
            reference: Optional reference S2/HLS data, shape (B, 4, H, W)
            
        Returns:
            Dict with 'loss' (total) and individual components
        """
        # Extract bands
        # Assuming band order: [B02, B03, B04, B08] = [Blue, Green, Red, NIR]
        pred_blue, pred_green, pred_red, pred_nir = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
        tgt_blue, tgt_green, tgt_red, tgt_nir = target[:, 0:1], target[:, 1:2], target[:, 2:3], target[:, 3:4]
        
        losses = {}
        
        # ====================================================================
        # NDVI RMSE Loss
        # ====================================================================
        
        pred_ndvi = compute_ndvi(pred_red, pred_nir)
        tgt_ndvi = compute_ndvi(tgt_red, tgt_nir)
        
        ndvi_loss = F.mse_loss(pred_ndvi, tgt_ndvi).sqrt()
        losses['ndvi_rmse'] = ndvi_loss
        
        # ====================================================================
        # EVI RMSE Loss
        # ====================================================================
        
        pred_evi = compute_evi(pred_blue, pred_red, pred_nir)
        tgt_evi = compute_evi(tgt_blue, tgt_red, tgt_nir)
        
        evi_loss = F.mse_loss(pred_evi, tgt_evi).sqrt()
        losses['evi_rmse'] = evi_loss
        
        # ====================================================================
        # Spectral Angle Mapper (SAM) Loss
        # ====================================================================
        
        if self.use_sam and reference is not None:
            sam_loss = self._compute_sam(pred, reference)
            losses['sam'] = sam_loss
        else:
            # Use target as reference if no external reference provided
            if self.use_sam:
                sam_loss = self._compute_sam(pred, target)
                losses['sam'] = sam_loss
            else:
                sam_loss = torch.tensor(0.0, device=pred.device)
                losses['sam'] = sam_loss
        
        # ====================================================================
        # Total Loss
        # ====================================================================
        
        total_loss = (
            self.ndvi_weight * ndvi_loss +
            self.evi_weight * evi_loss +
            self.sam_weight * sam_loss
        )
        
        losses['loss'] = total_loss
        
        return losses
    
    def _compute_sam(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute Spectral Angle Mapper (SAM) loss.
        
        SAM measures the spectral similarity between two spectra by computing
        the angle between them in n-dimensional space.
        
        Args:
            pred: Predicted spectra, shape (B, C, H, W)
            target: Target spectra, shape (B, C, H, W)
            eps: Small constant for numerical stability
            
        Returns:
            Mean SAM angle in radians
        """
        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W)
        pred_flat = pred.flatten(2)
        target_flat = target.flatten(2)
        
        # Compute dot product
        dot_product = (pred_flat * target_flat).sum(dim=1)  # (B, H*W)
        
        # Compute norms
        pred_norm = pred_flat.norm(dim=1) + eps  # (B, H*W)
        target_norm = target_flat.norm(dim=1) + eps  # (B, H*W)
        
        # Compute cosine similarity
        cos_angle = dot_product / (pred_norm * target_norm)
        
        # Clamp to valid range for acos
        cos_angle = torch.clamp(cos_angle, -1.0 + eps, 1.0 - eps)
        
        # Compute angle
        angle = torch.acos(cos_angle)
        
        # Return mean angle
        return angle.mean()


# ============================================================================
# Identity/Edge-Guard Loss
# ============================================================================

class IdentityEdgeGuardLoss(nn.Module):
    """
    Penalizes geometric changes between Stage 1 input and Stage 2 output.
    
    Components:
        - Identity Loss: L1 distance between input and output
        - Edge Preservation: Maintains edge structure using Sobel gradients
        
    This ensures Stage 2 acts as a refinement filter rather than a generative
    model that might introduce spurious geometric changes.
    """
    
    def __init__(
        self,
        identity_weight: float = 1.0,
        edge_weight: float = 0.5,
        edge_threshold: float = 0.1
    ):
        """
        Args:
            identity_weight: Weight for identity loss
            edge_weight: Weight for edge preservation loss
            edge_threshold: Threshold for edge detection (normalized)
        """
        super().__init__()
        
        self.identity_weight = identity_weight
        self.edge_weight = edge_weight
        self.edge_threshold = edge_threshold
        
        # Sobel filters for edge detection
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0) / 8.0)
        
        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]]
        ], dtype=torch.float32).unsqueeze(0) / 8.0)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute identity and edge-guard loss.
        
        Args:
            pred: Predicted refined optical, shape (B, 4, H, W)
            target: Target optical (Stage 1 output), shape (B, 4, H, W)
            
        Returns:
            Dict with 'loss' (total) and individual components
        """
        losses = {}
        
        # ====================================================================
        # Identity Loss (L1)
        # ====================================================================
        
        identity_loss = F.l1_loss(pred, target)
        losses['identity'] = identity_loss
        
        # ====================================================================
        # Edge Preservation Loss
        # ====================================================================
        
        edge_loss = self._compute_edge_loss(pred, target)
        losses['edge_preservation'] = edge_loss
        
        # ====================================================================
        # Total Loss
        # ====================================================================
        
        total_loss = (
            self.identity_weight * identity_loss +
            self.edge_weight * edge_loss
        )
        
        losses['loss'] = total_loss
        
        return losses
    
    def _compute_edge_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge preservation loss using Sobel gradients.
        
        Args:
            pred: Predicted tensor, shape (B, C, H, W)
            target: Target tensor, shape (B, C, H, W)
            
        Returns:
            Edge preservation loss
        """
        B, C, H, W = pred.shape
        
        # Compute edges for each channel separately
        pred_edges_x = F.conv2d(
            pred.reshape(B * C, 1, H, W),
            self.sobel_x,
            padding=1
        ).reshape(B, C, H, W)
        
        pred_edges_y = F.conv2d(
            pred.reshape(B * C, 1, H, W),
            self.sobel_y,
            padding=1
        ).reshape(B, C, H, W)
        
        target_edges_x = F.conv2d(
            target.reshape(B * C, 1, H, W),
            self.sobel_x,
            padding=1
        ).reshape(B, C, H, W)
        
        target_edges_y = F.conv2d(
            target.reshape(B * C, 1, H, W),
            self.sobel_y,
            padding=1
        ).reshape(B, C, H, W)
        
        # Compute edge magnitude
        pred_edge_mag = torch.sqrt(pred_edges_x**2 + pred_edges_y**2 + 1e-8)
        target_edge_mag = torch.sqrt(target_edges_x**2 + target_edges_y**2 + 1e-8)
        
        # L1 loss on edge magnitudes
        edge_loss = F.l1_loss(pred_edge_mag, target_edge_mag)
        
        return edge_loss


# ============================================================================
# Lightweight PatchGAN Discriminator
# ============================================================================

class PatchGANDiscriminator(nn.Module):
    """
    Lightweight PatchGAN discriminator for texture detail.
    
    Architecture:
        - 5 convolutional layers with increasing channels
        - Instance normalization for stability
        - LeakyReLU activations
        - Outputs per-patch predictions (not global)
        
    Memory-efficient design:
        - Small number of filters
        - No fully connected layers
        - Patch-based output (reduces memory vs global discriminator)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        base_filters: int = 32,
        num_layers: int = 4
    ):
        """
        Args:
            in_channels: Number of input channels (4 for S2 bands)
            base_filters: Base number of filters (doubled each layer)
            num_layers: Number of convolutional layers
        """
        super().__init__()
        
        layers = []
        
        # First layer: no normalization
        layers.append(nn.Conv2d(
            in_channels,
            base_filters,
            kernel_size=4,
            stride=2,
            padding=1
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Intermediate layers
        nf_mult = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # Cap at 8x base filters
            
            layers.append(nn.Conv2d(
                base_filters * nf_mult_prev,
                base_filters * nf_mult,
                kernel_size=4,
                stride=2,
                padding=1
            ))
            layers.append(nn.InstanceNorm2d(base_filters * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layer
        nf_mult = min(2 ** num_layers, 8)
        layers.append(nn.Conv2d(
            base_filters * nf_mult,
            1,
            kernel_size=4,
            stride=1,
            padding=1
        ))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.
        
        Args:
            x: Input tensor, shape (B, 4, H, W)
            
        Returns:
            Per-patch predictions, shape (B, 1, H', W')
            where H' and W' depend on input size and num_layers
        """
        return self.model(x)


class PatchGANLoss(nn.Module):
    """
    Adversarial loss using lightweight PatchGAN discriminator.
    
    Uses least-squares GAN (LSGAN) loss for stability:
        - Discriminator: minimize (D(real) - 1)^2 + D(fake)^2
        - Generator: minimize (D(fake) - 1)^2
    """
    
    def __init__(
        self,
        discriminator: Optional[PatchGANDiscriminator] = None,
        in_channels: int = 4,
        base_filters: int = 32,
        num_layers: int = 4
    ):
        """
        Args:
            discriminator: Pre-built discriminator (if None, creates new one)
            in_channels: Number of input channels
            base_filters: Base number of filters for discriminator
            num_layers: Number of layers for discriminator
        """
        super().__init__()
        
        if discriminator is None:
            self.discriminator = PatchGANDiscriminator(
                in_channels=in_channels,
                base_filters=base_filters,
                num_layers=num_layers
            )
        else:
            self.discriminator = discriminator
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mode: str = 'generator'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PatchGAN loss.
        
        Args:
            pred: Predicted refined optical, shape (B, 4, H, W)
            target: Target optical (real data), shape (B, 4, H, W)
            mode: 'generator' or 'discriminator'
            
        Returns:
            Dict with 'loss' and auxiliary metrics
        """
        if mode == 'generator':
            # Generator tries to fool discriminator
            pred_fake = self.discriminator(pred)
            loss = F.mse_loss(pred_fake, torch.ones_like(pred_fake))
            
            return {
                'loss': loss,
                'pred_fake_score': pred_fake.mean().item()
            }
            
        elif mode == 'discriminator':
            # Discriminator tries to distinguish real from fake
            pred_real = self.discriminator(target.detach())
            pred_fake = self.discriminator(pred.detach())
            
            loss_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
            loss_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
            
            loss = (loss_real + loss_fake) * 0.5
            
            return {
                'loss': loss,
                'loss_real': loss_real.item(),
                'loss_fake': loss_fake.item(),
                'pred_real_score': pred_real.mean().item(),
                'pred_fake_score': pred_fake.mean().item()
            }
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'generator' or 'discriminator'.")


# ============================================================================
# Combined Stage 2 Loss
# ============================================================================

class Stage2Loss(nn.Module):
    """
    Combined loss function for Stage 2 Prithvi refinement.
    
    Components:
        1. Spectral Plausibility (weight ~1.0)
        2. Identity/Edge-Guard (weight ~0.5)
        3. PatchGAN Adversarial (weight ~0.05)
        
    Total loss encourages:
        - Spectral realism (vegetation indices, spectral angles)
        - Geometric consistency (no spurious changes)
        - Texture detail (realistic high-frequency content)
    """
    
    def __init__(
        self,
        spectral_weight: float = 1.0,
        identity_weight: float = 0.5,
        adversarial_weight: float = 0.05,
        ndvi_weight: float = 1.0,
        evi_weight: float = 0.5,
        sam_weight: float = 0.3,
        use_sam: bool = True,
        edge_weight: float = 0.5,
        discriminator: Optional[PatchGANDiscriminator] = None
    ):
        """
        Args:
            spectral_weight: Weight for spectral plausibility loss
            identity_weight: Weight for identity/edge-guard loss
            adversarial_weight: Weight for adversarial loss (keep low!)
            ndvi_weight: Weight for NDVI within spectral loss
            evi_weight: Weight for EVI within spectral loss
            sam_weight: Weight for SAM within spectral loss
            use_sam: Whether to use Spectral Angle Mapper
            edge_weight: Weight for edge preservation within identity loss
            discriminator: Optional pre-built discriminator
        """
        super().__init__()
        
        self.spectral_weight = spectral_weight
        self.identity_weight = identity_weight
        self.adversarial_weight = adversarial_weight
        
        # Component losses
        self.spectral_loss = SpectralPlausibilityLoss(
            ndvi_weight=ndvi_weight,
            evi_weight=evi_weight,
            sam_weight=sam_weight,
            use_sam=use_sam
        )
        
        self.identity_loss = IdentityEdgeGuardLoss(
            identity_weight=1.0,
            edge_weight=edge_weight
        )
        
        self.adversarial_loss = PatchGANLoss(
            discriminator=discriminator
        )
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
        mode: str = 'generator'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined Stage 2 loss.
        
        Args:
            pred: Predicted refined optical, shape (B, 4, H, W)
            target: Target optical (Stage 1 output), shape (B, 4, H, W)
            reference: Optional reference S2/HLS data, shape (B, 4, H, W)
            mode: 'generator' or 'discriminator'
            
        Returns:
            Dict with 'loss' (total) and all component losses
        """
        losses = {}
        
        if mode == 'generator':
            # Compute all generator losses
            
            # Spectral plausibility
            spectral_losses = self.spectral_loss(pred, target, reference)
            for k, v in spectral_losses.items():
                if k == 'loss':
                    losses['spectral_loss'] = v
                else:
                    losses[f'spectral_{k}'] = v
            
            # Identity/edge-guard
            identity_losses = self.identity_loss(pred, target)
            for k, v in identity_losses.items():
                if k == 'loss':
                    losses['identity_loss'] = v
                else:
                    losses[f'identity_{k}'] = v
            
            # Adversarial
            adv_losses = self.adversarial_loss(pred, target, mode='generator')
            losses['adversarial_loss'] = adv_losses['loss']
            losses['adv_pred_fake_score'] = adv_losses['pred_fake_score']
            
            # Total generator loss
            total_loss = (
                self.spectral_weight * losses['spectral_loss'] +
                self.identity_weight * losses['identity_loss'] +
                self.adversarial_weight * losses['adversarial_loss']
            )
            
            losses['loss'] = total_loss
            
        elif mode == 'discriminator':
            # Compute discriminator loss only
            disc_losses = self.adversarial_loss(pred, target, mode='discriminator')
            losses.update(disc_losses)
        
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        return losses


# ============================================================================
# Testing and Demo
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Stage 2 Loss Functions - Test Suite")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}\n")
    
    # Test parameters
    batch_size = 2
    H, W = 256, 256
    
    # Create dummy data
    pred = torch.randn(batch_size, 4, H, W).to(device) * 0.5 + 0.5
    target = torch.randn(batch_size, 4, H, W).to(device) * 0.5 + 0.5
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    
    print("Test 1: Spectral Plausibility Loss")
    print("-" * 70)
    
    spectral_loss = SpectralPlausibilityLoss().to(device)
    losses = spectral_loss(pred, target)
    
    print(f"Total loss: {losses['loss'].item():.6f}")
    print(f"  NDVI RMSE: {losses['ndvi_rmse'].item():.6f}")
    print(f"  EVI RMSE: {losses['evi_rmse'].item():.6f}")
    print(f"  SAM: {losses['sam'].item():.6f}")
    print("✓ Spectral loss test passed\n")
    
    print("Test 2: Identity/Edge-Guard Loss")
    print("-" * 70)
    
    identity_loss = IdentityEdgeGuardLoss().to(device)
    losses = identity_loss(pred, target)
    
    print(f"Total loss: {losses['loss'].item():.6f}")
    print(f"  Identity: {losses['identity'].item():.6f}")
    print(f"  Edge preservation: {losses['edge_preservation'].item():.6f}")
    print("✓ Identity/edge-guard loss test passed\n")
    
    print("Test 3: PatchGAN Discriminator")
    print("-" * 70)
    
    discriminator = PatchGANDiscriminator(in_channels=4, base_filters=32, num_layers=4).to(device)
    output = discriminator(pred)
    
    print(f"Input shape: {pred.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test adversarial loss
    patchgan_loss = PatchGANLoss(discriminator=discriminator)
    
    gen_losses = patchgan_loss(pred, target, mode='generator')
    disc_losses = patchgan_loss(pred, target, mode='discriminator')
    
    print(f"\nGenerator loss: {gen_losses['loss'].item():.6f}")
    print(f"  Fake score: {gen_losses['pred_fake_score']:.3f}")
    
    print(f"Discriminator loss: {disc_losses['loss'].item():.6f}")
    print(f"  Real score: {disc_losses['pred_real_score']:.3f}")
    print(f"  Fake score: {disc_losses['pred_fake_score']:.3f}")
    print("✓ PatchGAN test passed\n")
    
    print("Test 4: Combined Stage 2 Loss")
    print("-" * 70)
    
    stage2_loss = Stage2Loss(
        spectral_weight=1.0,
        identity_weight=0.5,
        adversarial_weight=0.05
    ).to(device)
    
    gen_losses = stage2_loss(pred, target, mode='generator')
    
    print(f"Total generator loss: {gen_losses['loss'].item():.6f}")
    print(f"  Spectral: {gen_losses['spectral_loss'].item():.6f}")
    print(f"  Identity: {gen_losses['identity_loss'].item():.6f}")
    print(f"  Adversarial: {gen_losses['adversarial_loss'].item():.6f}")
    
    disc_losses = stage2_loss(pred, target, mode='discriminator')
    print(f"\nDiscriminator loss: {disc_losses['loss'].item():.6f}")
    print("✓ Combined loss test passed\n")
    
    print("Test 5: Memory Usage")
    print("-" * 70)
    
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Forward + backward pass
        pred_train = torch.randn(2, 4, 256, 256, device=device, requires_grad=True)
        target_train = torch.randn(2, 4, 256, 256, device=device)
        
        losses = stage2_loss(pred_train, target_train, mode='generator')
        losses['loss'].backward()
        
        mem_allocated = torch.cuda.memory_allocated() / 1024**2
        mem_peak = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"Memory allocated: {mem_allocated:.1f} MB")
        print(f"Memory peak: {mem_peak:.1f} MB")
        
        if mem_peak < 1000:
            print("✓ Memory usage acceptable for low-VRAM systems")
        else:
            print(f"⚠️  Memory usage high: {mem_peak:.1f} MB")
    else:
        print("Skipped (no CUDA)")
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
    
    print("\nUsage example:")
    print("""
from axs_lib.stage2_losses import Stage2Loss

# Create loss function
criterion = Stage2Loss(
    spectral_weight=1.0,
    identity_weight=0.5,
    adversarial_weight=0.05
)

# Training loop
for batch in dataloader:
    # Forward pass
    pred = model(stage1_output)
    
    # Compute generator loss
    losses = criterion(pred, stage1_output, mode='generator')
    gen_loss = losses['loss']
    
    # Backward and optimize generator
    gen_loss.backward()
    optimizer.step()
    
    # Update discriminator every N steps
    if step % disc_update_freq == 0:
        disc_losses = criterion(pred.detach(), stage1_output, mode='discriminator')
        disc_loss = disc_losses['loss']
        
        disc_loss.backward()
        disc_optimizer.step()
""")
