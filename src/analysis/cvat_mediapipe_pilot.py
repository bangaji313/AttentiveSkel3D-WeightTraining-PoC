from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import contextlib
import io
import json
import xml.etree.ElementTree as ET

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from data.preprocess import DataPreprocessor


LANDMARK_NAMES: list[str] = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

LANDMARK_INDEX = {name: idx for idx, name in enumerate(LANDMARK_NAMES)}

OPTIONAL_REFERENCE_JOINTS: list[str] = [
    "nose",
]

CVAT_12_JOINTS: list[str] = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SQUAT_RESEARCH_JOINTS: dict[str, list[str]] = {
    "minimum_validator_subset": [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    "recommended_cvat_12_joint_subset": CVAT_12_JOINTS,
    "optional_reference_joint": OPTIONAL_REFERENCE_JOINTS,
    "full_mediapipe_output": LANDMARK_NAMES,
}

POSE_CONNECTIONS = tuple(sorted(mp.solutions.pose.POSE_CONNECTIONS))


@dataclass
class ExtractionBundle:
    video_path: Path
    frames_bgr: list[np.ndarray]
    pose_array: np.ndarray
    detected_mask: np.ndarray
    timestamps_sec: np.ndarray
    fps: float
    frame_width: int
    frame_height: int


def extract_mediapipe_landmarks(
    video_path: str | Path,
    *,
    model_complexity: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> ExtractionBundle:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames_bgr: list[np.ndarray] = []
    pose_rows: list[np.ndarray] = []
    detected_flags: list[bool] = []
    timestamps: list[float] = []

    with mp.solutions.pose.Pose(
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose_model:
        frame_index = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frames_bgr.append(frame_bgr)
            timestamps.append(frame_index / max(fps, 1e-6))
            result = pose_model.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            if result.pose_landmarks:
                pose_frame = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark],
                    dtype=np.float32,
                )
                detected_flags.append(True)
            else:
                pose_frame = np.full((33, 4), np.nan, dtype=np.float32)
                pose_frame[:, 3] = 0.0
                detected_flags.append(False)

            pose_rows.append(pose_frame)
            frame_index += 1

    cap.release()

    return ExtractionBundle(
        video_path=video_path,
        frames_bgr=frames_bgr,
        pose_array=np.stack(pose_rows, axis=0),
        detected_mask=np.asarray(detected_flags, dtype=bool),
        timestamps_sec=np.asarray(timestamps, dtype=np.float32),
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
    )


def flatten_landmarks(bundle: ExtractionBundle) -> pd.DataFrame:
    pose = bundle.pose_array
    records: list[dict[str, object]] = []
    for frame_index in range(pose.shape[0]):
        for landmark_index, landmark_name in enumerate(LANDMARK_NAMES):
            x, y, z, visibility = pose[frame_index, landmark_index]
            records.append(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": float(bundle.timestamps_sec[frame_index]),
                    "landmark_index": landmark_index,
                    "landmark_name": landmark_name,
                    "x_norm": None if np.isnan(x) else float(x),
                    "y_norm": None if np.isnan(y) else float(y),
                    "z_rel": None if np.isnan(z) else float(z),
                    "visibility": float(visibility),
                    "pose_detected": bool(bundle.detected_mask[frame_index]),
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_visibility(
    pose_array: np.ndarray,
    joint_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    total_frames = pose_array.shape[0]
    for joint_name in joint_names:
        joint_index = LANDMARK_INDEX[joint_name]
        visibility = pose_array[:, joint_index, 3]
        valid = np.isfinite(pose_array[:, joint_index, 0])
        rows.append(
            {
                "joint_name": joint_name,
                "mean_visibility": float(np.nanmean(visibility)),
                "min_visibility": float(np.nanmin(visibility)),
                "coverage_ratio": float(valid.sum() / max(total_frames, 1)),
            }
        )
    return pd.DataFrame(rows).sort_values(["coverage_ratio", "mean_visibility"], ascending=[False, False])


def run_preprocess_pipeline(
    pose_array: np.ndarray,
    *,
    target_frames: int = 64,
    preprocessor: DataPreprocessor | None = None,
) -> dict[str, np.ndarray]:
    if preprocessor is None:
        preprocessor = DataPreprocessor(target_frames=target_frames)

    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        cleaned = preprocessor.filter_and_clean(pose_array)
        smoothed = preprocessor.smooth_data(cleaned)
        normalized = preprocessor.spatial_normalize(smoothed)
        resampled = preprocessor.temporal_resample(normalized, target_frames=target_frames)

    return {
        "cleaned": cleaned,
        "smoothed": smoothed,
        "normalized": normalized,
        "resampled": resampled,
        "captured_stdout": np.array(stdout_buffer.getvalue().splitlines(), dtype=object),
    }


def select_representative_frames(total_frames: int, count: int = 4) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames <= count:
        return list(range(total_frames))
    return sorted({int(value) for value in np.linspace(0, total_frames - 1, count)})


def _point_to_pixel(x_norm: float, y_norm: float, frame_width: int, frame_height: int) -> tuple[int, int]:
    x_px = int(np.clip(round(x_norm * frame_width), 0, max(frame_width - 1, 0)))
    y_px = int(np.clip(round(y_norm * frame_height), 0, max(frame_height - 1, 0)))
    return x_px, y_px


def draw_pose_overlay(
    frame_bgr: np.ndarray,
    pose_frame: np.ndarray,
    *,
    highlight_joint_names: list[str] | None = None,
    label_joint_names: list[str] | None = None,
    line_color: tuple[int, int, int] = (80, 80, 80),
) -> np.ndarray:
    overlay = frame_bgr.copy()
    highlight_indices = {LANDMARK_INDEX[name] for name in (highlight_joint_names or [])}
    label_indices = {LANDMARK_INDEX[name] for name in (label_joint_names or [])}
    frame_height, frame_width = overlay.shape[:2]

    valid_mask = np.isfinite(pose_frame[:, 0]) & np.isfinite(pose_frame[:, 1])

    for start_idx, end_idx in POSE_CONNECTIONS:
        if not (valid_mask[start_idx] and valid_mask[end_idx]):
            continue
        pt1 = _point_to_pixel(float(pose_frame[start_idx, 0]), float(pose_frame[start_idx, 1]), frame_width, frame_height)
        pt2 = _point_to_pixel(float(pose_frame[end_idx, 0]), float(pose_frame[end_idx, 1]), frame_width, frame_height)
        cv2.line(overlay, pt1, pt2, line_color, 2, cv2.LINE_AA)

    for landmark_index, landmark_name in enumerate(LANDMARK_NAMES):
        if not valid_mask[landmark_index]:
            continue
        point = _point_to_pixel(float(pose_frame[landmark_index, 0]), float(pose_frame[landmark_index, 1]), frame_width, frame_height)
        if landmark_index in highlight_indices:
            radius = 8
            color = (40, 180, 255)
        else:
            radius = 4
            color = (0, 210, 120)
        cv2.circle(overlay, point, radius, color, -1, cv2.LINE_AA)
        if landmark_index in label_indices:
            cv2.putText(
                overlay,
                landmark_name.replace("left_", "L-").replace("right_", "R-"),
                (point[0] + 6, point[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return overlay


def find_candidate_cvat_exports(project_root: str | Path) -> list[Path]:
    project_root = Path(project_root)
    patterns = [
        "annotations/**/*.xml",
        "annotations/**/*.json",
        "data/annotations/**/*.xml",
        "data/annotations/**/*.json",
        "results/**/*.xml",
        "results/**/*.json",
        "data/raw/**/*.xml",
        "data/raw/**/*.json",
    ]
    results: list[Path] = []
    for pattern in patterns:
        results.extend(project_root.glob(pattern))
    unique_results = sorted({path.resolve() for path in results if path.is_file()})
    return [path for path in unique_results if path.suffix.lower() in {".xml", ".json"}]


def parse_cvat_skeleton_export(
    export_path: str | Path,
    *,
    joint_names: list[str] | None = None,
) -> pd.DataFrame:
    export_path = Path(export_path)
    joint_names = joint_names or CVAT_12_JOINTS
    suffix = export_path.suffix.lower()
    if suffix == ".xml":
        return _parse_cvat_xml(export_path, joint_names)
    if suffix == ".json":
        return _parse_cvat_json(export_path, joint_names)
    raise ValueError(f"Unsupported CVAT export suffix: {suffix}")


def _parse_cvat_xml(export_path: Path, joint_names: list[str]) -> pd.DataFrame:
    root = ET.parse(export_path).getroot()
    records: list[dict[str, object]] = []

    for skeleton_node in root.findall(".//skeleton"):
        frame_index = int(skeleton_node.attrib.get("frame", skeleton_node.attrib.get("id", 0)))
        points_text = skeleton_node.attrib.get("points", "")
        records.extend(_points_text_to_records(points_text, frame_index, joint_names))

    for points_node in root.findall(".//points"):
        frame_index = int(points_node.attrib.get("frame", points_node.attrib.get("id", 0)))
        points_text = points_node.attrib.get("points", "")
        records.extend(_points_text_to_records(points_text, frame_index, joint_names))

    if not records:
        raise ValueError(f"No skeleton points found in XML export: {export_path}")
    return pd.DataFrame.from_records(records)


def _parse_cvat_json(export_path: Path, joint_names: list[str]) -> pd.DataFrame:
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []

    shapes = payload.get("shapes") or payload.get("annotations") or []
    for shape in shapes:
        frame_index = int(shape.get("frame", shape.get("image_id", 0)))
        points = shape.get("points")
        if isinstance(points, str):
            records.extend(_points_text_to_records(points, frame_index, joint_names))
        elif isinstance(points, list):
            coords = list(points)
            if len(coords) % 2 != 0:
                continue
            records.extend(_point_list_to_records(coords, frame_index, joint_names))

    if not records:
        raise ValueError(f"No skeleton points found in JSON export: {export_path}")
    return pd.DataFrame.from_records(records)


def _points_text_to_records(points_text: str, frame_index: int, joint_names: list[str]) -> list[dict[str, object]]:
    coords: list[float] = []
    for token in points_text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        coords.append(float(token))
    return _point_list_to_records(coords, frame_index, joint_names)


def _point_list_to_records(coords: list[float], frame_index: int, joint_names: list[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    pair_count = min(len(coords) // 2, len(joint_names))
    for joint_order in range(pair_count):
        x_value = float(coords[joint_order * 2])
        y_value = float(coords[joint_order * 2 + 1])
        records.append(
            {
                "frame_index": frame_index,
                "joint_order": joint_order,
                "joint_name": joint_names[joint_order],
                "x_px": x_value,
                "y_px": y_value,
            }
        )
    return records


def mediapipe_2d_dataframe(
    bundle: ExtractionBundle,
    joint_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frame_index in range(bundle.pose_array.shape[0]):
        pose_frame = bundle.pose_array[frame_index]
        for joint_name in joint_names:
            joint_index = LANDMARK_INDEX[joint_name]
            x_norm, y_norm, _z, _visibility = pose_frame[joint_index]
            if not np.isfinite(x_norm) or not np.isfinite(y_norm):
                continue
            x_px, y_px = _point_to_pixel(float(x_norm), float(y_norm), bundle.frame_width, bundle.frame_height)
            rows.append(
                {
                    "frame_index": frame_index,
                    "joint_name": joint_name,
                    "x_px_mp": x_px,
                    "y_px_mp": y_px,
                }
            )
    return pd.DataFrame.from_records(rows)


def compare_cvat_vs_mediapipe(
    cvat_df: pd.DataFrame,
    bundle: ExtractionBundle,
    *,
    joint_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joint_names = joint_names or CVAT_12_JOINTS
    mp_df = mediapipe_2d_dataframe(bundle, joint_names)
    merged = cvat_df.merge(mp_df, on=["frame_index", "joint_name"], how="inner")
    if merged.empty:
        raise ValueError("CVAT and MediaPipe have no overlapping frame/joint pairs.")

    torso_by_frame = _estimate_torso_pixels(bundle)
    merged["torso_px"] = merged["frame_index"].map(torso_by_frame).fillna(np.hypot(bundle.frame_width, bundle.frame_height))
    merged["error_px"] = np.sqrt((merged["x_px"] - merged["x_px_mp"]) ** 2 + (merged["y_px"] - merged["y_px_mp"]) ** 2)
    merged["error_norm_torso"] = merged["error_px"] / merged["torso_px"].replace(0, np.nan)

    summary = (
        merged.groupby("joint_name", as_index=False)
        .agg(
            compared_points=("error_px", "size"),
            mean_error_px=("error_px", "mean"),
            median_error_px=("error_px", "median"),
            mean_error_norm_torso=("error_norm_torso", "mean"),
        )
        .sort_values("mean_error_norm_torso")
    )
    return merged, summary


def _estimate_torso_pixels(bundle: ExtractionBundle) -> dict[int, float]:
    torso_lengths: dict[int, float] = {}
    l_shoulder = LANDMARK_INDEX["left_shoulder"]
    r_shoulder = LANDMARK_INDEX["right_shoulder"]
    l_hip = LANDMARK_INDEX["left_hip"]
    r_hip = LANDMARK_INDEX["right_hip"]

    for frame_index, pose_frame in enumerate(bundle.pose_array):
        relevant = pose_frame[[l_shoulder, r_shoulder, l_hip, r_hip], :2]
        if not np.isfinite(relevant).all():
            continue
        mid_shoulder = relevant[:2].mean(axis=0)
        mid_hip = relevant[2:].mean(axis=0)
        shoulder_px = np.array(_point_to_pixel(float(mid_shoulder[0]), float(mid_shoulder[1]), bundle.frame_width, bundle.frame_height))
        hip_px = np.array(_point_to_pixel(float(mid_hip[0]), float(mid_hip[1]), bundle.frame_width, bundle.frame_height))
        torso_lengths[frame_index] = float(np.linalg.norm(shoulder_px - hip_px))
    return torso_lengths