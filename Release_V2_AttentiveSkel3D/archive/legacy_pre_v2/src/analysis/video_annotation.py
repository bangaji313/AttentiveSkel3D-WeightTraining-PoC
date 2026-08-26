"""
Video Overlay Annotation Module

Overlay skeleton MediaPipe pada frame asli dengan:
- Warna hijau untuk joint valid
- Warna merah untuk joint invalid
- Highlight landmark yang dipakai oleh biomechanical rule
- Label rule, metric value, threshold, status
- Frame index dan timestamp

Requires: OpenCV, MediaPipe, numpy, matplotlib
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np
import cv2

log = logging.getLogger("video_annotation")


@dataclass
class VideoAnnotationConfig:
    """Konfigurasi untuk video annotation."""
    
    # Warna (BGR format untuk OpenCV)
    color_valid: Tuple[int, int, int] = (0, 255, 0)      # Hijau
    color_invalid: Tuple[int, int, int] = (0, 0, 255)    # Merah
    color_highlight: Tuple[int, int, int] = (255, 255, 0)  # Cyan
    color_text: Tuple[int, int, int] = (255, 255, 255)   # Putih
    
    # Ukuran visual
    joint_radius: int = 8
    joint_thickness: int = -1  # -1 = filled circle
    connection_thickness: int = 3
    font_size: float = 0.6
    font_thickness: int = 1
    
    # MediaPipe BlazePose connections (33 landmarks)
    # Definisi koneksi antar landmark
    connections: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        """Set default MediaPipe BlazePose connections jika belum didefinisikan."""
        if self.connections is None:
            # Koneksi standar MediaPipe BlazePose
            self.connections = [
                # Wajah (tidak semua, hanya key points)
                (0, 1), (1, 2), (2, 3), (3, 7),
                (0, 4), (4, 5), (5, 6), (6, 8),
                
                # Tubuh (torso)
                (9, 10),
                (11, 12),  # Bahu
                (11, 23), (12, 24),  # Bahu-Pinggul
                (23, 24),  # Pinggul
                
                # Lengan kiri
                (11, 13), (13, 15),
                
                # Lengan kanan
                (12, 14), (14, 16),
                
                # Tangan (simplified)
                (15, 21), (16, 22),
                
                # Kaki kiri
                (23, 25), (25, 27),
                
                # Kaki kanan
                (24, 26), (26, 28),
            ]


class VideoAnnotator:
    """Annotasi skeleton pada frame video."""
    
    def __init__(
        self,
        video_tensor: np.ndarray,  # (64, 33, 3)
        frame_annotations,  # List[FrameAnnotation]
        fps: float = 30.0,
        output_video_path: Path = None,
        config: VideoAnnotationConfig = None,
    ):
        """
        Args:
            video_tensor: (64, 33, 3) — skeleton pose sequence
            frame_annotations: List of FrameAnnotation objects
            fps: Frame per second untuk output video
            output_video_path: Path untuk menyimpan video annotated
            config: VideoAnnotationConfig
        """
        self.video_tensor = video_tensor
        self.frame_annotations = frame_annotations
        self.fps = fps
        self.output_video_path = output_video_path
        self.config = config or VideoAnnotationConfig()
        
        self.num_frames = len(video_tensor)
        self.height = 720  # Default height untuk render
        self.width = 1280  # Default width untuk render
        
        log.info(
            f"VideoAnnotator initialized: {self.num_frames} frames, "
            f"{self.width}x{self.height}, {fps} FPS"
        )
    
    @staticmethod
    def normalize_skeleton(
        skeleton: np.ndarray,  # (33, 3)
        target_width: int = 1280,
        target_height: int = 720,
    ) -> np.ndarray:
        """
        Normalize skeleton coordinates dari range [-1, 1] ke pixel range.
        
        Asumsi: skeleton sudah normalized ke [-1, 1] range (dari preprocessing).
        
        Args:
            skeleton: (33, 3) array dengan x, y, z coordinates
            target_width: Target width dalam pixel
            target_height: Target height dalam pixel
        
        Returns:
            Normalized skeleton untuk pixel space (33, 2) — hanya x, y
        """
        x = skeleton[:, 0]  # Left-right
        y = skeleton[:, 1]  # Top-bottom
        
        # Map [-1, 1] → [0, width] dan [0, height]
        x_pixel = ((x + 1) / 2 * target_width).astype(int)
        y_pixel = ((y + 1) / 2 * target_height).astype(int)
        
        return np.column_stack([x_pixel, y_pixel])
    
    def create_frame_image(
        self,
        frame_idx: int,
        background_color: Tuple[int, int, int] = (50, 50, 50),
    ) -> np.ndarray:
        """
        Create annotated frame image.
        
        Args:
            frame_idx: Frame index (0-63)
            background_color: BGR color untuk background
        
        Returns:
            Annotated image (H, W, 3) uint8
        """
        # Create blank image
        img = np.full((self.height, self.width, 3), background_color, dtype=np.uint8)
        
        # Get skeleton coordinates
        skeleton = self.video_tensor[frame_idx]  # (33, 3)
        skeleton_pixel = self.normalize_skeleton(skeleton, self.width, self.height)
        
        # Get annotation
        annotation = self.frame_annotations[frame_idx]
        
        # Determine color based on validator status
        is_valid = annotation.validator_status == "VALID"
        joint_color = self.config.color_valid if is_valid else self.config.color_invalid
        
        # Draw connections first (so they appear behind joints)
        for start_idx, end_idx in self.config.connections:
            start_pos = tuple(skeleton_pixel[start_idx])
            end_pos = tuple(skeleton_pixel[end_idx])
            
            # Check if both endpoints are valid (not too far out of bounds)
            if (0 <= start_pos[0] < self.width and 0 <= start_pos[1] < self.height and
                0 <= end_pos[0] < self.width and 0 <= end_pos[1] < self.height):
                cv2.line(
                    img,
                    start_pos,
                    end_pos,
                    joint_color,
                    self.config.connection_thickness,
                )
        
        # Draw joints
        for idx, pos in enumerate(skeleton_pixel):
            # Highlight relevant landmarks untuk rule ini
            if idx in annotation.relevant_landmarks:
                color = self.config.color_highlight
                radius = self.config.joint_radius + 3
            else:
                color = joint_color
                radius = self.config.joint_radius
            
            # Check bounds
            if 0 <= pos[0] < self.width and 0 <= pos[1] < self.height:
                cv2.circle(
                    img,
                    tuple(pos),
                    radius,
                    color,
                    self.config.joint_thickness,
                )
        
        # Add text annotations (top-left area)
        text_lines = [
            f"Frame: {annotation.frame_index} / {self.num_frames - 1}",
            f"Time: {annotation.original_timestamp:.2f}s",
            "",
            f"Rule: {annotation.biomechanical_metric_name}",
            f"Value: {annotation.metric_value:.2f}°",
            f"Threshold: {annotation.threshold:.2f}°",
            f"Status: {annotation.validator_status}",
            "",
            f"Attention: {annotation.temporal_attention_score:.4f}",
        ]
        
        y_offset = 30
        for line in text_lines:
            cv2.putText(
                img,
                line,
                (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_size,
                self.config.color_text,
                self.config.font_thickness,
            )
            y_offset += 25
        
        return img
    
    def create_video(self) -> None:
        """Create annotated video dan simpan."""
        if self.output_video_path is None:
            raise ValueError("output_video_path must be set")
        
        self.output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Video codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(self.output_video_path),
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        
        if not out.isOpened():
            raise RuntimeError(f"Cannot open video writer: {self.output_video_path}")
        
        for frame_idx in range(self.num_frames):
            frame_img = self.create_frame_image(frame_idx)
            out.write(frame_img)
            
            if (frame_idx + 1) % 10 == 0:
                log.info(f"Processed {frame_idx + 1}/{self.num_frames} frames")
        
        out.release()
        log.info(f"Annotated video saved: {self.output_video_path}")
    
    def save_critical_frames_images(
        self,
        critical_frames_dict: Dict[str, Optional[int]],
        output_dir: Path,
    ) -> None:
        """
        Save images of critical frames (error, peak attention, etc).
        
        Args:
            critical_frames_dict: Dict dengan key: frame_index
            output_dir: Directory untuk menyimpan images
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for label, frame_idx in critical_frames_dict.items():
            if frame_idx is None:
                continue
            
            if not (0 <= frame_idx < self.num_frames):
                log.warning(f"Frame index {frame_idx} out of bounds, skipping {label}")
                continue
            
            frame_img = self.create_frame_image(frame_idx)
            output_path = output_dir / f"{label}_frame_{frame_idx:02d}.png"
            cv2.imwrite(str(output_path), frame_img)
            log.info(f"Saved critical frame: {output_path}")


# ============================================================================
# HELPERS
# ============================================================================

def extract_critical_frames(
    annotator: VideoAnnotator,
    critical_frames_dict: Dict[str, Optional[int]],
    output_dir: Path,
) -> None:
    """Convenience function untuk extract dan save critical frames."""
    annotator.save_critical_frames_images(critical_frames_dict, output_dir)
