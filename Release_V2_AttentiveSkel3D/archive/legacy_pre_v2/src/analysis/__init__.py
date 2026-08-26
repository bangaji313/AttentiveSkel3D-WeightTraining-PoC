"""
Analysis module for biomechanical validation audit and temporal error localization.

This module provides comprehensive tools for:

1. Mathematical Documentation (biomechanical_validator_audit.py):
   - Documenting biomechanical rules with mathematical rigor
   - Testing scale invariance of angles and ratios
   - Analyzing per-frame validation metrics
   - Exporting structured analysis reports

2. Frame-Level Temporal Localization (frame_level_localization.py):
   - Detecting error frames and critical phases
   - Analyzing temporal patterns of biomechanical violations
   - Computing frame-level annotations
   - Generating timeline visualizations

3. Video Annotation (video_annotation.py):
   - Overlaying skeleton with biomechanical feedback
   - Highlighting relevant landmarks
   - Creating annotated video output with frame-level metrics

4. Attention Integration (attention_utils.py):
   - Extracting temporal attention scores from model checkpoints
   - Hooks for capturing intermediate activations
   - Integration with biomechanical validators
   - Batch analysis utilities

Key Separation:
- Biomechanical Validator: Ground truth frame-level classification
- Temporal Attention: Model's learned focus distribution (NOT frame-level classification)
- Sequence Prediction: Overall movement quality from classifier head

Author: Research Team
Date: 2025
"""

# ====================================================
# Biomechanical Validator Audit
# ====================================================
from .biomechanical_validator_audit import (
    BiomechanicalRulesDocumentation,
    BiomechanicalRule,
    StableAngleCalculator,
    ScaleInvarianceAnalyzer,
    PerFrameAnalyzer,
    export_rules_documentation,
)

# ====================================================
# Frame-Level Temporal Localization
# ====================================================
from .frame_level_localization import (
    FrameAnnotation,
    TemporalErrorSummary,
    FrameLevelLocalization,
    plot_temporal_timeline,
    get_critical_frames_dict,
)

# ====================================================
# Video Annotation
# ====================================================
from .video_annotation import (
    VideoAnnotationConfig,
    VideoAnnotator,
    extract_critical_frames,
)

# ====================================================
# Attention Utils
# ====================================================
from .attention_utils import (
    TemporalAttentionHook,
    extract_temporal_attention_from_checkpoint,
    create_validator_wrapper,
    create_biomechanical_frame_validators,
    analyze_batch_videos,
)

# ====================================================
# Batch Processing & Analysis
# ====================================================
from .batch_utils import (
    BatchAnalysisResult,
    batch_analyze_videos,
    aggregate_batch_results,
    compute_batch_statistics,
    generate_batch_report,
    save_batch_results_json,
)

__all__ = [
    # Validator audit
    "BiomechanicalRulesDocumentation",
    "BiomechanicalRule",
    "StableAngleCalculator",
    "ScaleInvarianceAnalyzer",
    "PerFrameAnalyzer",
    "export_rules_documentation",
    
    # Temporal localization
    "FrameAnnotation",
    "TemporalErrorSummary",
    "FrameLevelLocalization",
    "plot_temporal_timeline",
    "get_critical_frames_dict",
    
    # Video annotation
    "VideoAnnotationConfig",
    "VideoAnnotator",
    "extract_critical_frames",
    
    # Attention utils
    "TemporalAttentionHook",
    "extract_temporal_attention_from_checkpoint",
    "create_validator_wrapper",
    "create_biomechanical_frame_validators",
    "analyze_batch_videos",
    
    # Batch processing
    "BatchAnalysisResult",
    "batch_analyze_videos",
    "aggregate_batch_results",
    "compute_batch_statistics",
    "generate_batch_report",
    "save_batch_results_json",
]

__version__ = "1.2.0"
