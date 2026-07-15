"""
Temporal Attention Extraction & Integration Utilities

Helper functions untuk:
1. Extract temporal attention scores dari model checkpoint
2. Hooks untuk capture intermediate activations
3. Integration dengan biomechanical validator
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Tuple, Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("attention_utils")


# ============================================================================
# HOOK UTILITIES: Extract intermediate attention scores
# ============================================================================

class TemporalAttentionHook:
    """Hook untuk capture temporal attention scores dari model forward pass."""
    
    def __init__(self):
        self.attention_scores = None
        self.handle = None
    
    def __call__(self, module, input, output):
        """Capture output dari temporal_attention layer."""
        # Output shape: (B, 1, T', L', 1) setelah conv3d
        # Setelah mean over spatial dims: (B, 1, T', 1, 1)
        # Setelah softmax: normalized weights per-frame
        
        if isinstance(output, torch.Tensor):
            # Extract temporal dimension (T')
            scores = output.squeeze()  # Remove singleton dimensions
            
            # Handle berbagai shape cases
            if scores.dim() == 1:
                # Single batch → (T',)
                self.attention_scores = scores.detach().cpu().numpy()
            elif scores.dim() == 2:
                # Multiple batches → (B, T') — ambil first batch saja
                self.attention_scores = scores[0].detach().cpu().numpy()
            else:
                log.warning(f"Unexpected attention output shape: {scores.shape}")
                self.attention_scores = scores.mean(dim=0).detach().cpu().numpy()
    
    def register(self, model: nn.Module, layer_name: str = "temporal_attention"):
        """Register hook pada layer tertentu."""
        for name, module in model.named_modules():
            if name == layer_name or layer_name in name:
                self.handle = module.register_forward_hook(self)
                log.info(f"Hook registered on {name}")
                return
        
        log.warning(f"Layer {layer_name} not found in model")
    
    def unregister(self):
        """Remove hook."""
        if self.handle is not None:
            self.handle.remove()
            log.info("Hook unregistered")
    
    def get_scores(self) -> np.ndarray:
        """Get captured attention scores."""
        if self.attention_scores is None:
            raise RuntimeError("No attention scores captured. Run forward pass first.")
        return self.attention_scores


# ============================================================================
# MODEL INFERENCE
# ============================================================================

def extract_temporal_attention_from_checkpoint(
    model_path: Path,
    video_tensor: np.ndarray,  # (64, 33, 3)
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_class=None,  # Jika None, akan di-import dari src.models
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract temporal attention scores dari checkpoint.
    
    Args:
        model_path: Path ke checkpoint (.pth file)
        video_tensor: Input skeleton (64, 33, 3)
        device: Device untuk inference ("cuda" atau "cpu")
        model_class: Class model (default: AttentiveSkel3D)
    
    Returns:
        (attention_scores, logits) — attention shape (32,), logits shape (2,)
    """
    if model_class is None:
        from ..models.model_3dcnn import AttentiveSkel3D
        model_class = AttentiveSkel3D
    
    # Load checkpoint
    checkpoint = torch.load(str(model_path), map_location=device)
    
    # Initialize model
    model = model_class(num_classes=2, use_temporal_attention=True)
    
    # Load weights (handle berbagai checkpoint format)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    log.info(f"Model loaded from {model_path}")
    
    # Setup hook untuk capture attention
    hook = TemporalAttentionHook()
    hook.register(model, layer_name="temporal_attention")
    
    # Prepare input
    input_tensor = torch.from_numpy(video_tensor).unsqueeze(0).to(device).float()
    # Input shape: (1, 64, 33, 3) → batch_size=1
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_tensor)
    
    # Get attention scores dan logits
    attention_scores = hook.get_scores()
    logits_np = logits.squeeze(0).detach().cpu().numpy()
    
    hook.unregister()
    
    log.info(f"Attention shape: {attention_scores.shape}, Logits: {logits_np}")
    
    return attention_scores, logits_np


# ============================================================================
# VALIDATOR INTEGRATION
# ============================================================================

def create_validator_wrapper(
    exercise_type: str,
    validator_class,  # BiomechanicalValidator dari src.data
    criterion_key: str = "all",  # "valgus", "hip_angle", "knee_angle", etc.
) -> Callable:
    """
    Create validator function wrapper untuk digunakan dengan FrameLevelLocalization.
    
    Args:
        exercise_type: "Squat", "BenchPress", atau "Deadlift"
        validator_class: BiomechanicalValidator class
        criterion_key: Specific criterion atau "all"
    
    Returns:
        Callable(frame) → (is_valid, metric_value, threshold)
    """
    
    if criterion_key == "all":
        # Validate semua kriteria, return aggregate
        def validator_all(frame: np.ndarray) -> Tuple[bool, float, float]:
            """Aggregate validator untuk semua kriteria."""
            frame_batch = frame[np.newaxis, ...]  # Add batch dim
            
            if exercise_type == "Squat":
                result = validator_class.validate_squat(frame_batch)
            elif exercise_type == "BenchPress":
                result = validator_class.validate_benchpress(frame_batch)
            elif exercise_type == "Deadlift":
                result = validator_class.validate_deadlift(frame_batch)
            else:
                raise ValueError(f"Unknown exercise type: {exercise_type}")
            
            # result = {
            #   "is_valid": bool,
            #   "criteria": {...},
            #   "details": {...}
            # }
            is_valid = result.get("is_valid", True)
            
            # Return dummy metric if no specific metric available
            metric_value = 0.0
            threshold = 0.0
            
            return is_valid, metric_value, threshold
        
        return validator_all
    
    else:
        # Specific criterion
        def validator_specific(frame: np.ndarray) -> Tuple[bool, float, float]:
            """Specific criterion validator."""
            frame_batch = frame[np.newaxis, ...]
            
            if exercise_type == "Squat":
                result = validator_class.validate_squat(frame_batch)
                criteria_map = {
                    "valgus": ("valgus_ratio", "valgus_ratio_threshold"),
                    "hip_angle": ("hip_angle", "hip_angle_threshold"),
                    "knee_angle": ("knee_angle", "knee_angle_threshold"),
                }
            elif exercise_type == "BenchPress":
                result = validator_class.validate_benchpress(frame_batch)
                criteria_map = {
                    "elbow_angle": ("elbow_angle", "elbow_angle_threshold"),
                }
            elif exercise_type == "Deadlift":
                result = validator_class.validate_deadlift(frame_batch)
                criteria_map = {
                    "spine_inclination": ("spine_inclination", "spine_inclination_threshold"),
                }
            else:
                raise ValueError(f"Unknown exercise: {exercise_type}")
            
            if criterion_key not in criteria_map:
                raise ValueError(f"Unknown criterion: {criterion_key}")
            
            metric_key, threshold_key = criteria_map[criterion_key]
            
            # Extract from result
            details = result.get("details", {})
            metric_value = details.get(metric_key, 0.0)
            threshold = details.get(threshold_key, 0.0)
            
            # Determine is_valid for this specific criterion
            criteria = result.get("criteria", {})
            is_valid = criteria.get(criterion_key, True)
            
            return is_valid, metric_value, threshold
        
        return validator_specific


def create_biomechanical_frame_validators(
    exercise_type: str,
    validator_class,
) -> Dict[str, Callable]:
    """
    Create dict of validators untuk semua criteria dari suatu exercise.
    
    Args:
        exercise_type: "Squat", "BenchPress", atau "Deadlift"
        validator_class: BiomechanicalValidator class
    
    Returns:
        Dict[criterion_name] → validator_func
    """
    
    if exercise_type == "Squat":
        criteria = ["valgus_ratio", "hip_angle", "knee_angle"]
    elif exercise_type == "BenchPress":
        criteria = ["elbow_angle"]
    elif exercise_type == "Deadlift":
        criteria = ["spine_inclination"]
    else:
        raise ValueError(f"Unknown exercise: {exercise_type}")
    
    validators = {}
    for criterion in criteria:
        validators[criterion] = create_validator_wrapper(
            exercise_type, validator_class, criterion_key=criterion
        )
    
    return validators


# ============================================================================
# BATCH ANALYSIS
# ============================================================================

def analyze_batch_videos(
    video_tensors: Dict[str, np.ndarray],  # name → (64, 33, 3) tensor
    exercise_type: str,
    attention_dict: Dict[str, np.ndarray],  # name → attention scores
    checkpoint_path: Path = None,
    output_dir: Path = None,
) -> Dict[str, dict]:
    """
    Analyze multiple videos in batch.
    
    Args:
        video_tensors: Dict of video tensors
        exercise_type: "Squat", "BenchPress", atau "Deadlift"
        attention_dict: Pre-computed attention scores atau dict kosong
        checkpoint_path: If provided, extract attention from checkpoint
        output_dir: Output directory untuk results
    
    Returns:
        Dict[video_name] → results dict
    """
    from .frame_level_localization import FrameLevelLocalization
    
    results = {}
    
    for video_name, tensor in video_tensors.items():
        log.info(f"Analyzing {video_name}...")
        
        # Initialize localization
        loc = FrameLevelLocalization(
            video_tensor=tensor,
            exercise_type=exercise_type,
            fps=30.0,
            video_name=video_name,
        )
        
        # Set temporal attention
        if video_name in attention_dict:
            loc.set_temporal_attention(attention_dict[video_name])
        elif checkpoint_path:
            # Extract dari checkpoint
            att_scores, logits = extract_temporal_attention_from_checkpoint(
                checkpoint_path, tensor
            )
            loc.set_temporal_attention(att_scores)
        else:
            log.warning(f"No attention scores for {video_name}")
        
        # TODO: Analyze dengan validator
        # loc.analyze_with_validator(validator_func, metric_name, landmarks)
        
        # Compute summary
        summary = loc.compute_temporal_summary()
        
        results[video_name] = {
            "localization": loc,
            "summary": summary,
            "df": loc.to_dataframe(),
        }
        
        # Save results jika output_dir provided
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / f"{video_name}_frame_scores.csv"
            loc.save_frame_scores_csv(csv_path)
    
    return results
