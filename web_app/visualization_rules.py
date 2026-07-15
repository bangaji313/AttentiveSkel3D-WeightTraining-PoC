from __future__ import annotations

from typing import Iterable

import numpy as np


def rule_landmarks(rule: dict) -> list[int]:
    points: list[int] = []
    for key in ("lm_a", "lm_b", "lm_c"):
        value = rule.get(key)
        if isinstance(value, int) and value not in points:
            points.append(value)
    return points


def exercise_relevant_landmarks(rules: Iterable[dict]) -> list[int]:
    points: list[int] = []
    for rule in rules:
        for index in rule_landmarks(rule):
            if index not in points:
                points.append(index)
    return points


def rule_marker_labels(rule: dict) -> dict[int, str]:
    labels: dict[int, str] = {}
    for label, key in (("A", "lm_a"), ("B", "lm_b"), ("C", "lm_c")):
        value = rule.get(key)
        if isinstance(value, int):
            labels[value] = label
    return labels


def _angle_metric(frame_xyz: np.ndarray, a: int, b: int, c: int) -> float:
    ba = frame_xyz[a] - frame_xyz[b]
    bc = frame_xyz[c] - frame_xyz[b]
    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 180.0
    cos_angle = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def rule_metric_for_frame(exercise: str, rule: dict, frame_xyz: np.ndarray) -> dict:
    frame = np.asarray(frame_xyz, dtype=np.float32)
    if frame.shape != (33, 3):
        raise ValueError(f"frame_xyz must have shape (33, 3), got {frame.shape!r}")

    metric_value: float | None = None
    threshold = float(rule.get("threshold_val", np.nan))
    status: str = "N/A"

    if exercise == "Squat":
        if rule.get("name") == "Knee Valgus":
            left_knee = frame[25, 0]
            right_knee = frame[26, 0]
            left_ankle = frame[27, 0]
            right_ankle = frame[28, 0]
            knee_width = abs(left_knee - right_knee)
            ankle_width = abs(left_ankle - right_ankle)
            metric_value = float(knee_width / max(ankle_width, 1e-8))
            status = "VALID" if metric_value >= threshold else "INVALID"
        elif rule.get("name") == "Hip Flexion Angle":
            metric_value = _angle_metric(frame, 11, 23, 25)
            status = "VALID" if metric_value <= threshold else "INVALID"
        elif rule.get("name") == "Squat Depth (Knee Angle)":
            metric_value = _angle_metric(frame, 23, 25, 27)
            status = "VALID" if metric_value <= threshold else "INVALID"

    elif exercise == "BenchPress":
        if rule.get("name") == "Elbow ROM":
            left_angle = _angle_metric(frame, 11, 13, 15)
            right_angle = _angle_metric(frame, 12, 14, 16)
            metric_value = float((left_angle + right_angle) / 2.0)
            status = "VALID" if metric_value <= threshold else "INVALID"

    elif exercise == "Deadlift":
        if rule.get("name") == "Spine Inclination":
            mid_shoulder = (frame[11] + frame[12]) / 2.0
            mid_hip = (frame[23] + frame[24]) / 2.0
            spine_vec = mid_shoulder - mid_hip
            vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            norm_spine = float(np.linalg.norm(spine_vec))
            if norm_spine >= 1e-8:
                cos_angle = float(np.dot(spine_vec, vertical) / (norm_spine * 1.0))
                cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
                metric_value = float(np.degrees(np.arccos(cos_angle)))
                status = "VALID" if 20.0 <= metric_value <= threshold else "INVALID"

    return {
        "exercise": exercise,
        "rule_name": str(rule.get("name", "Unknown Rule")),
        "metric_value": metric_value,
        "threshold": threshold,
        "status": status,
        "label_map": rule_marker_labels(rule),
        "relevant_landmarks": rule_landmarks(rule),
        "segment": (rule.get("lm_a"), rule.get("lm_b"), rule.get("lm_c")),
    }
