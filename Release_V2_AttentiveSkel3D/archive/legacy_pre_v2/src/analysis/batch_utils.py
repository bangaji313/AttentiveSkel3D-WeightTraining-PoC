"""
Batch Processing & Analysis Utilities untuk Temporal Localization

Helper functions untuk:
1. Batch analysis dari multiple videos
2. Result aggregation dan comparative analysis
3. Quality metrics computation
4. Report generation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from dataclasses import asdict, dataclass

log = logging.getLogger("batch_utils")


@dataclass
class BatchAnalysisResult:
    """Container untuk hasil batch analysis."""
    
    video_name: str
    exercise_type: str
    duration_frames: int
    duration_seconds: float
    
    # Error statistics
    error_frame_count: int
    error_frame_ratio: float
    first_error_frame: Optional[int]
    last_error_frame: Optional[int]
    
    # Critical phase
    critical_phase_duration: int
    critical_phase_start: Optional[int]
    critical_phase_end: Optional[int]
    
    # Attention statistics
    peak_attention_frame: int
    peak_attention_score: float
    mean_attention_in_errors: float
    mean_attention_in_valid: float
    attention_alignment_score: float  # (mean_in_errors - mean_in_valid) / mean_in_valid
    
    # CSV paths
    csv_path: Optional[Path] = None
    timeline_path: Optional[Path] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["csv_path"] = str(d["csv_path"]) if d["csv_path"] else None
        d["timeline_path"] = str(d["timeline_path"]) if d["timeline_path"] else None
        return d


# ============================================================================
# BATCH ANALYSIS FUNCTIONS
# ============================================================================

def batch_analyze_videos(
    video_dict: Dict[str, np.ndarray],
    exercise_type: str,
    attention_dict: Dict[str, np.ndarray],
    validator_func,
    metric_name: str,
    output_dir: Path,
    fps: float = 30.0,
    include_plots: bool = True,
    include_video: bool = False,
) -> Dict[str, BatchAnalysisResult]:
    """
    Analyze multiple videos dalam batch.
    
    Args:
        video_dict: {video_name: tensor (64, 33, 3)}
        exercise_type: "Squat", "BenchPress", "Deadlift"
        attention_dict: {video_name: attention_scores (32 atau 64)}
        validator_func: Callable[[frame], (is_valid, metric, threshold)]
        metric_name: Nama metric untuk documentation
        output_dir: Base output directory
        fps: Frame rate
        include_plots: Generate timeline plots
        include_video: Generate annotated videos (slower)
    
    Returns:
        Dict[video_name] → BatchAnalysisResult
    """
    from .frame_level_localization import FrameLevelLocalization, plot_temporal_timeline
    from .video_annotation import VideoAnnotator, VideoAnnotationConfig
    
    results = {}
    total = len(video_dict)
    
    for idx, (video_name, tensor) in enumerate(video_dict.items(), 1):
        log.info(f"[{idx}/{total}] Analyzing {video_name}...")
        
        try:
            # Initialize localization
            loc = FrameLevelLocalization(
                video_tensor=tensor,
                exercise_type=exercise_type,
                fps=fps,
                video_name=video_name,
            )
            
            # Set attention
            if video_name in attention_dict:
                loc.set_temporal_attention(attention_dict[video_name])
            else:
                log.warning(f"No attention scores for {video_name}, skipping")
                continue
            
            # Analyze dengan validator
            loc.analyze_with_validator(
                validator_func=validator_func,
                metric_name=metric_name,
                relevant_landmarks=[11, 23, 25],  # Standard untuk Squat
            )
            
            # Compute summary
            summary = loc.compute_temporal_summary()
            
            # Create output subdirectory
            video_output_dir = output_dir / video_name
            video_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save CSV
            csv_path = video_output_dir / "frame_scores.csv"
            loc.save_frame_scores_csv(csv_path)
            log.debug(f"  → Saved CSV: {csv_path}")
            
            # Plot timeline
            timeline_path = None
            if include_plots:
                timeline_path = video_output_dir / "timeline.png"
                try:
                    plot_temporal_timeline(loc, summary, timeline_path)
                    log.debug(f"  → Saved timeline: {timeline_path}")
                except Exception as e:
                    log.warning(f"  ⚠ Timeline plot failed: {e}")
            
            # Create annotated video
            if include_video:
                try:
                    config = VideoAnnotationConfig()
                    annotator = VideoAnnotator(
                        video_tensor=tensor,
                        frame_annotations=loc.frame_annotations,
                        fps=fps,
                        output_video_path=video_output_dir / "annotated.mp4",
                        config=config,
                    )
                    annotator.create_video()
                    log.debug(f"  → Saved video: {video_output_dir}/annotated.mp4")
                except Exception as e:
                    log.warning(f"  ⚠ Video creation failed: {e}")
            
            # Create result
            attention_alignment = (
                (summary.mean_attention_in_errors - summary.mean_attention_in_valid)
                / (summary.mean_attention_in_valid + 1e-6)
                * 100  # Percentage
            )
            
            result = BatchAnalysisResult(
                video_name=video_name,
                exercise_type=exercise_type,
                duration_frames=loc.num_frames,
                duration_seconds=loc.num_frames / fps,
                error_frame_count=summary.error_frame_count,
                error_frame_ratio=summary.error_frame_ratio,
                first_error_frame=summary.first_error_frame,
                last_error_frame=summary.last_error_frame,
                critical_phase_duration=summary.critical_phase_duration,
                critical_phase_start=summary.critical_phase_start,
                critical_phase_end=summary.critical_phase_end,
                peak_attention_frame=summary.peak_temporal_attention_frame,
                peak_attention_score=summary.peak_temporal_attention_score,
                mean_attention_in_errors=summary.mean_attention_in_errors,
                mean_attention_in_valid=summary.mean_attention_in_valid,
                attention_alignment_score=attention_alignment,
                csv_path=csv_path,
                timeline_path=timeline_path,
            )
            
            results[video_name] = result
            log.info(f"  ✓ Completed: {summary.error_frame_count} errors, "
                    f"alignment={attention_alignment:.1f}%")
        
        except Exception as e:
            log.error(f"  ✗ Failed: {e}", exc_info=True)
            continue
    
    log.info(f"Batch analysis completed: {len(results)}/{total} videos")
    return results


# ============================================================================
# AGGREGATION & COMPARATIVE ANALYSIS
# ============================================================================

def aggregate_batch_results(
    results: Dict[str, BatchAnalysisResult],
) -> pd.DataFrame:
    """
    Aggregate batch results into DataFrame untuk analysis.
    
    Args:
        results: Dict dari batch_analyze_videos()
    
    Returns:
        DataFrame dengan normalized metrics
    """
    rows = []
    for video_name, result in results.items():
        row = result.to_dict()
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def compute_batch_statistics(
    results: Dict[str, BatchAnalysisResult],
) -> dict:
    """
    Compute aggregate statistics dari batch results.
    
    Args:
        results: Dict dari batch_analyze_videos()
    
    Returns:
        Dict dengan statistics
    """
    if not results:
        return {}
    
    df = aggregate_batch_results(results)
    
    stats = {
        "total_videos": len(results),
        "error_ratio": {
            "mean": float(df["error_frame_ratio"].mean()),
            "std": float(df["error_frame_ratio"].std()),
            "min": float(df["error_frame_ratio"].min()),
            "max": float(df["error_frame_ratio"].max()),
        },
        "critical_phase_duration": {
            "mean": float(df["critical_phase_duration"].mean()),
            "std": float(df["critical_phase_duration"].std()),
            "max": float(df["critical_phase_duration"].max()),
        },
        "attention_alignment_score": {
            "mean": float(df["attention_alignment_score"].mean()),
            "std": float(df["attention_alignment_score"].std()),
            "min": float(df["attention_alignment_score"].min()),
            "max": float(df["attention_alignment_score"].max()),
        },
    }
    
    return stats


def generate_batch_report(
    results: Dict[str, BatchAnalysisResult],
    output_path: Path,
    title: str = "Temporal Localization Batch Report",
) -> None:
    """
    Generate Markdown report dari batch results.
    """
    df = aggregate_batch_results(results)
    stats = compute_batch_statistics(results)
    
    # Sort by error_frame_ratio (descending)
    df = df.sort_values("error_frame_ratio", ascending=False)
    
    # Generate markdown
    md = f"# {title}\n\n"
    md += f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n"
    
    md += "## Summary Statistics\n\n"
    md += f"- Total Videos: {stats['total_videos']}\n\n"
    
    md += "### Error Ratio Statistics\n"
    md += f"- Mean: {stats['error_ratio']['mean']:.1%}\n"
    md += f"- Std: {stats['error_ratio']['std']:.1%}\n"
    md += f"- Range: [{stats['error_ratio']['min']:.1%}, {stats['error_ratio']['max']:.1%}]\n\n"
    
    md += "## Per-Video Results\n\n"
    
    # Write file
    with open(output_path, "w") as f:
        f.write(md)
    
    log.info(f"Report generated: {output_path}")


def save_batch_results_json(
    results: Dict[str, BatchAnalysisResult],
    output_path: Path,
) -> None:
    """
    Save batch results to JSON untuk downstream processing.
    """
    data = {name: result.to_dict() for name, result in results.items()}
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    
    log.info(f"Results saved to JSON: {output_path}")
