"""
Frame-Level Temporal Error Localization Module

Memisahkan secara tegas antara:
1. Biomechanical Validator (ground truth frame-level)
2. Temporal Attention Score (dari model)
3. Sequence-level model prediction (output classifier)

Catatan Penting:
- Validator menunjukkan KAPAN aturan biomekanis dilanggar
- Temporal Attention menunjukkan KAPAN model memberikan fokus terbesar
- Keduanya independent: temporal attention bukan frame-level classification
  (karena model hanya dilatih dengan sequence-level label)

Author: Research Team
Date: 2025
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d
import torch

log = logging.getLogger("frame_localization")


# ============================================================================
# DATACLASSES: Struktural temporal annotation per-frame
# ============================================================================

@dataclass
class FrameAnnotation:
    """Anotasi lengkap satu frame dengan biomechanical validator + attention."""
    frame_index: int
    original_timestamp: float
    biomechanical_metric_name: str
    metric_value: float
    threshold: float
    validator_status: str  # "VALID" or "INVALID"
    validator_label: int  # 0 (valid) or 1 (invalid/error)
    temporal_attention_score: float
    relevant_landmarks: List[int]


@dataclass
class TemporalErrorSummary:
    """Ringkasan temporal error yang dideteksi dalam satu gerakan."""
    first_error_frame: Optional[int]
    peak_error_frame: Optional[int]
    last_error_frame: Optional[int]
    error_frame_indices: List[int]
    error_frame_count: int
    error_frame_ratio: float
    
    critical_phase_start: Optional[int]
    critical_phase_end: Optional[int]
    critical_phase_duration: int  # frame count
    
    peak_temporal_attention_frame: int
    peak_temporal_attention_score: float
    
    mean_attention_in_errors: float  # attention rerata saat error
    mean_attention_in_valid: float   # attention rerata saat valid


# ============================================================================
# CLASS: Frame-level Localization
# ============================================================================

class FrameLevelLocalization:
    """
    Lokalisasi error temporal dengan analisis biomekanis + attention temporal.
    
    Workflow:
    1. Load tensor video (64, 33, 3)
    2. Ekstrak biomechanical validator ground truth per-frame
    3. Load temporal attention scores dari model (T'=32 atau T=64)
    4. Interpolasi attention jika T' ≠ 64
    5. Detect critical phases (consecutive error frames)
    6. Generate per-frame annotation table
    7. Visualisasi timeline (attention + metrics + threshold + errors)
    8. Overlay video dengan skeleton annotation
    """
    
    def __init__(
        self,
        video_tensor: np.ndarray,  # (64, 33, 3) — frame, landmark, xyz
        exercise_type: str,  # "Squat", "BenchPress", "Deadlift"
        fps: float = 30.0,
        video_name: str = "sample_video",
    ):
        """
        Args:
            video_tensor: (64, 33, 3) tensor dengan frame, landmark, xyz
            exercise_type: Jenis gerakan untuk dispatch validator
            fps: Frame per second (default 30 FPS → 2.13 detik untuk 64 frame)
            video_name: Nama video untuk output file
        """
        self.video_tensor = video_tensor.astype(np.float32)
        self.exercise_type = exercise_type
        self.fps = fps
        self.video_name = video_name
        
        self.num_frames = self.video_tensor.shape[0]
        assert self.num_frames == 64, f"Expected 64 frames, got {self.num_frames}"
        
        # Timestamps per frame (dalam detik)
        self.frame_timestamps = np.arange(self.num_frames) / self.fps
        
        # Container untuk hasil
        self.frame_annotations: List[FrameAnnotation] = []
        self.temporal_summary: Optional[TemporalErrorSummary] = None
        self.attention_scores_interpolated: Optional[np.ndarray] = None
        
        log.info(
            "FrameLevelLocalization initialized: %s, %d frames, %.2f FPS",
            exercise_type, self.num_frames, fps
        )
    
    def set_temporal_attention(
        self,
        attention_scores: np.ndarray,
        attention_source: str = "model_forward_pass",
    ) -> None:
        """
        Set temporal attention scores dari model.
        
        Args:
            attention_scores: Shape (64,) atau (32,) — scores per-frame
            attention_source: Deskripsi source (e.g., "model_forward_pass", "activation_hook")
        
        Raises:
            ValueError: Jika shape attention tidak kompatibel
        """
        attention_scores = attention_scores.astype(np.float32).flatten()
        
        if len(attention_scores) == self.num_frames:
            # Sudah 64 frames — gunakan langsung
            self.attention_scores_interpolated = attention_scores
            log.info("Attention scores set directly (already 64 frames)")
        
        elif len(attention_scores) == 32:
            # Hasil dari conv_block_2 MaxPool3d(2,2,1) — upsampling ke 64
            self.attention_scores_interpolated = self._interpolate_attention(
                attention_scores, target_length=self.num_frames
            )
            log.info("Attention scores upsampled: 32 → 64 frames")
        
        else:
            raise ValueError(
                f"Attention scores length {len(attention_scores)} not compatible. "
                f"Expected 32 (post-conv_block_2) or 64 (original)."
            )
        
        log.info(f"Attention source: {attention_source}")
    
    @staticmethod
    def _interpolate_attention(
        attention_scores: np.ndarray,
        target_length: int,
    ) -> np.ndarray:
        """
        Interpolasi linear attention scores dari resolusi asli ke target.
        
        Args:
            attention_scores: Shape (N,) — original scores
            target_length: Target length setelah interpolasi
        
        Returns:
            Interpolated scores shape (target_length,)
        """
        n = len(attention_scores)
        x_orig = np.linspace(0, 1, n)
        x_target = np.linspace(0, 1, target_length)
        
        # Linear interpolation
        f_interp = interp1d(x_orig, attention_scores, kind="linear", fill_value="extrapolate")
        interpolated = f_interp(x_target)
        
        # Normalize tetap dalam range [0, 1] jika input dalam range
        if attention_scores.min() >= 0 and attention_scores.max() <= 1:
            interpolated = np.clip(interpolated, 0, 1)
        
        return interpolated
    
    def analyze_with_validator(
        self,
        validator_func,  # Callable yang returns (is_valid: bool, metric_value: float, threshold: float)
        metric_name: str,
        relevant_landmarks: List[int],
    ) -> None:
        """
        Analisis frame-by-frame dengan biomechanical validator.
        
        Args:
            validator_func: Callable(tensor_frame) → (bool, float, float)
                           Returns (is_valid, metric_value, threshold)
            metric_name: Nama metric (e.g., "knee_flexion_angle")
            relevant_landmarks: List of landmark indices used by this metric
        """
        if self.attention_scores_interpolated is None:
            log.warning("Attention scores not set! Using dummy values")
            self.attention_scores_interpolated = np.ones(self.num_frames) / self.num_frames
        
        self.frame_annotations = []
        
        for frame_idx in range(self.num_frames):
            frame_data = self.video_tensor[frame_idx]  # (33, 3)
            
            # Validator result
            is_valid, metric_value, threshold = validator_func(frame_data)
            
            annotation = FrameAnnotation(
                frame_index=frame_idx,
                original_timestamp=self.frame_timestamps[frame_idx],
                biomechanical_metric_name=metric_name,
                metric_value=metric_value,
                threshold=threshold,
                validator_status="VALID" if is_valid else "INVALID",
                validator_label=0 if is_valid else 1,
                temporal_attention_score=self.attention_scores_interpolated[frame_idx],
                relevant_landmarks=relevant_landmarks,
            )
            self.frame_annotations.append(annotation)
        
        log.info(f"Analyzed {self.num_frames} frames with validator: {metric_name}")
    
    def compute_temporal_summary(self) -> TemporalErrorSummary:
        """
        Hitung ringkasan temporal error dan critical phases.
        
        Returns:
            TemporalErrorSummary object
        """
        if not self.frame_annotations:
            raise RuntimeError("No frame annotations available. Call analyze_with_validator first.")
        
        # Extract validator status dan attention scores
        validator_labels = np.array([ann.validator_label for ann in self.frame_annotations])
        attention_scores = np.array([ann.temporal_attention_score for ann in self.frame_annotations])
        frame_indices = np.arange(self.num_frames)
        
        # Error frames
        error_frame_indices = frame_indices[validator_labels == 1].tolist()
        error_frame_count = len(error_frame_indices)
        error_frame_ratio = error_frame_count / self.num_frames
        
        # First, peak, last error frame
        first_error_frame = error_frame_indices[0] if error_frame_indices else None
        last_error_frame = error_frame_indices[-1] if error_frame_indices else None
        
        if error_frame_indices:
            peak_error_frame = error_frame_indices[
                np.argmin(validator_labels[error_frame_indices])
            ]  # semantik: frame dengan error terbesar
        else:
            peak_error_frame = None
        
        # Critical phase (consecutive error frames)
        critical_phase_start = None
        critical_phase_end = None
        max_consecutive = 0
        current_start = None
        current_count = 0
        
        for i in range(self.num_frames):
            if validator_labels[i] == 1:
                if current_start is None:
                    current_start = i
                    current_count = 1
                else:
                    current_count += 1
            else:
                if current_count > max_consecutive:
                    max_consecutive = current_count
                    critical_phase_start = current_start
                    critical_phase_end = current_start + current_count - 1
                current_start = None
                current_count = 0
        
        # Check last run
        if current_count > max_consecutive:
            max_consecutive = current_count
            critical_phase_start = current_start
            critical_phase_end = current_start + current_count - 1
        
        critical_phase_duration = (
            (critical_phase_end - critical_phase_start + 1)
            if critical_phase_start is not None
            else 0
        )
        
        # Peak temporal attention frame
        peak_temporal_attention_frame = int(np.argmax(attention_scores))
        peak_temporal_attention_score = float(attention_scores[peak_temporal_attention_frame])
        
        # Mean attention in error vs. valid frames
        mean_attention_in_errors = (
            attention_scores[validator_labels == 1].mean()
            if error_frame_count > 0
            else 0.0
        )
        mean_attention_in_valid = (
            attention_scores[validator_labels == 0].mean()
            if (validator_labels == 0).sum() > 0
            else 0.0
        )
        
        summary = TemporalErrorSummary(
            first_error_frame=first_error_frame,
            peak_error_frame=peak_error_frame,
            last_error_frame=last_error_frame,
            error_frame_indices=error_frame_indices,
            error_frame_count=error_frame_count,
            error_frame_ratio=error_frame_ratio,
            critical_phase_start=critical_phase_start,
            critical_phase_end=critical_phase_end,
            critical_phase_duration=critical_phase_duration,
            peak_temporal_attention_frame=peak_temporal_attention_frame,
            peak_temporal_attention_score=peak_temporal_attention_score,
            mean_attention_in_errors=mean_attention_in_errors,
            mean_attention_in_valid=mean_attention_in_valid,
        )
        
        self.temporal_summary = summary
        log.info(f"Temporal summary computed: {error_frame_count} error frames")
        
        return summary
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert frame annotations to pandas DataFrame."""
        rows = [asdict(ann) for ann in self.frame_annotations]
        df = pd.DataFrame(rows)
        return df
    
    def save_frame_scores_csv(self, output_path: Path) -> None:
        """Simpan frame-level scores ke CSV."""
        df = self.to_dataframe()
        df.to_csv(str(output_path), index=False, encoding="utf-8")
        log.info(f"Frame scores saved: {output_path}")


# ============================================================================
# VISUALIZATION: Timeline Plot
# ============================================================================

def plot_temporal_timeline(
    localization: FrameLevelLocalization,
    summary: TemporalErrorSummary,
    output_path: Path,
    figsize: Tuple[int, int] = (16, 8),
) -> None:
    """
    Plot timeline dengan temporal attention + biomechanical metrics + errors.
    
    Args:
        localization: FrameLevelLocalization instance
        summary: TemporalErrorSummary dari compute_temporal_summary()
        output_path: Path untuk menyimpan figure
        figsize: Figure size (width, height)
    """
    df = localization.to_dataframe()
    frame_idx = df["frame_index"].values
    timestamps = df["original_timestamp"].values
    attention = df["temporal_attention_score"].values
    metric_value = df["metric_value"].values
    threshold = df["threshold"].values
    validator_status = df["validator_status"].values
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Panel 1 (primary axis): Temporal Attention + Metrics
    # =========================================================
    ax1.set_xlabel("Frame Index", fontsize=12)
    ax1.set_ylabel("Value", fontsize=12)
    ax1.set_xlim(0, len(frame_idx) - 1)
    
    # Temporal attention as shaded area
    ax1.fill_between(frame_idx, 0, attention, alpha=0.3, label="Temporal Attention", color="blue")
    ax1.plot(frame_idx, attention, color="blue", linewidth=2, marker="o", markersize=3)
    
    # Biomechanical metric
    ax1.plot(frame_idx, metric_value, color="green", linewidth=2, marker="s", 
             markersize=3, label="Biomechanical Metric")
    
    # Threshold as horizontal line
    thresh_val = threshold[0] if len(set(threshold)) == 1 else threshold.mean()
    ax1.axhline(thresh_val, color="red", linestyle="--", linewidth=2, label=f"Threshold ({thresh_val:.1f}°)")
    
    # Mark error frames as red background
    error_mask = validator_status == "INVALID"
    error_indices = frame_idx[error_mask]
    if len(error_indices) > 0:
        for err_idx in error_indices:
            ax1.axvspan(err_idx - 0.5, err_idx + 0.5, alpha=0.2, color="red")
    
    # Mark critical phase
    if summary.critical_phase_start is not None:
        ax1.axvspan(
            summary.critical_phase_start - 0.5,
            summary.critical_phase_end + 0.5,
            alpha=0.15,
            color="orange",
            label=f"Critical Phase (f{summary.critical_phase_start}–{summary.critical_phase_end})"
        )
    
    # Mark peak error frame with vertical line
    if summary.peak_error_frame is not None:
        ax1.axvline(summary.peak_error_frame, color="darkred", linestyle=":", linewidth=2,
                   label=f"Peak Error Frame ({summary.peak_error_frame})")
    
    # Mark peak attention frame
    ax1.axvline(summary.peak_temporal_attention_frame, color="darkblue", linestyle=":", 
               linewidth=2, label=f"Peak Attention Frame ({summary.peak_temporal_attention_frame})")
    
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Title with statistics
    title = (
        f"{localization.video_name} — {localization.exercise_type}\n"
        f"Error frames: {summary.error_frame_count}/{localization.num_frames} "
        f"({summary.error_frame_ratio*100:.1f}%) | "
        f"Critical phase: f{summary.critical_phase_start}–{summary.critical_phase_end} "
        f"({summary.critical_phase_duration} frames)"
    )
    ax1.set_title(title, fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    log.info(f"Timeline plot saved: {output_path}")
    plt.close()


# ============================================================================
# ANNOTATION HELPERS
# ============================================================================

def get_critical_frames_dict(summary: TemporalErrorSummary) -> Dict[str, Optional[int]]:
    """Extract critical frame indices as dictionary."""
    return {
        "first_error_frame": summary.first_error_frame,
        "peak_error_frame": summary.peak_error_frame,
        "last_error_frame": summary.last_error_frame,
        "critical_phase_start": summary.critical_phase_start,
        "critical_phase_end": summary.critical_phase_end,
        "peak_temporal_attention_frame": summary.peak_temporal_attention_frame,
    }
