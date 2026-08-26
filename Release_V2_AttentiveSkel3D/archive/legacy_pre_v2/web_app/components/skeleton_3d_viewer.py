from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import plotly.graph_objects as go

LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]

POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (0, 8),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (27, 31),
    (28, 32),
)

LANDMARK_COUNT = len(LANDMARK_NAMES)

VIEW_PRESETS: dict[str, dict[str, dict[str, float]]] = {
    "front": {"eye": {"x": 0.0, "y": 0.0, "z": 2.6}, "up": {"x": 0.0, "y": 1.0, "z": 0.0}},
    "left": {"eye": {"x": -2.6, "y": 0.1, "z": 0.35}, "up": {"x": 0.0, "y": 1.0, "z": 0.0}},
    "right": {"eye": {"x": 2.6, "y": 0.1, "z": 0.35}, "up": {"x": 0.0, "y": 1.0, "z": 0.0}},
    "isometric": {"eye": {"x": 1.85, "y": 1.75, "z": 1.45}, "up": {"x": 0.0, "y": 1.0, "z": 0.0}},
}

LEFT_COLOR = "#2b6cb0"
RIGHT_COLOR = "#c53030"
CENTER_COLOR = "#6b7280"
LOW_VISIBILITY_COLOR = "#a0aec0"
ATTENTION_COLOR = "#f59e0b"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def _mix_colors(base: str, overlay: str, weight: float) -> str:
    clamped = float(np.clip(weight, 0.0, 1.0))
    base_rgb = np.array(_hex_to_rgb(base), dtype=np.float32)
    overlay_rgb = np.array(_hex_to_rgb(overlay), dtype=np.float32)
    mixed = np.round(base_rgb * (1.0 - clamped) + overlay_rgb * clamped).astype(np.int32)
    mixed = np.clip(mixed, 0, 255)
    return _rgb_to_hex(tuple(int(v) for v in mixed))


def _classify_side(landmark_name: str) -> str:
    if landmark_name.startswith("LEFT_"):
        return "left"
    if landmark_name.startswith("RIGHT_"):
        return "right"
    return "center"


def _normalize_display_coordinates(coords: np.ndarray) -> np.ndarray:
    centered = coords.astype(np.float32, copy=True)
    origin = np.mean(centered, axis=0)
    centered -= origin
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale > 0:
        centered /= scale
    return centered


def prepare_frame_payload(
    frame_points: np.ndarray | Sequence[Sequence[float]],
    visibility: np.ndarray | Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(frame_points, dtype=np.float32)

    if coords.ndim != 2:
        raise ValueError(f"frame_points must be a 2D array, got shape {coords.shape!r}")

    if coords.shape[0] != LANDMARK_COUNT:
        raise ValueError(f"frame_points must contain {LANDMARK_COUNT} landmarks, got {coords.shape[0]}")

    extracted_visibility: np.ndarray | None = None
    if coords.shape[1] == 4:
        extracted_visibility = coords[:, 3].copy()
        coords = coords[:, :3].copy()
    elif coords.shape[1] == 3:
        coords = coords.copy()
    else:
        raise ValueError("frame_points must have 3 or 4 columns per landmark")

    if visibility is None:
        vis = extracted_visibility if extracted_visibility is not None else np.ones(LANDMARK_COUNT, dtype=np.float32)
    else:
        vis = np.asarray(visibility, dtype=np.float32)

    if vis.shape != (LANDMARK_COUNT,):
        raise ValueError(f"visibility must have shape ({LANDMARK_COUNT},), got {vis.shape!r}")

    if not np.isfinite(coords).all():
        raise ValueError("frame_points contains NaN or Inf")
    if not np.isfinite(vis).all():
        raise ValueError("visibility contains NaN or Inf")

    return coords, vis


def _attention_to_scale(attention: np.ndarray | Sequence[float] | None) -> np.ndarray | None:
    if attention is None:
        return None

    attn = np.asarray(attention, dtype=np.float32)
    if attn.shape != (LANDMARK_COUNT,):
        raise ValueError(f"attention must have shape ({LANDMARK_COUNT},), got {attn.shape!r}")
    if not np.isfinite(attn).all():
        raise ValueError("attention contains NaN or Inf")

    span = float(attn.max() - attn.min())
    if span <= 1e-8:
        return np.full(LANDMARK_COUNT, 0.5, dtype=np.float32)
    return (attn - attn.min()) / span


def _point_color(side: str, visibility: float) -> str:
    if visibility < 0.35:
        return LOW_VISIBILITY_COLOR
    if side == "left":
        return LEFT_COLOR
    if side == "right":
        return RIGHT_COLOR
    return CENTER_COLOR


def _marker_sizes(attention: np.ndarray | None, visibility: np.ndarray) -> list[float]:
    base = 6.0 + (visibility.clip(0.0, 1.0) * 2.5)
    if attention is None:
        return base.tolist()
    return (base + attention * 4.0).tolist()


def _camera_for_view(view: str) -> dict[str, dict[str, float]]:
    normalized = view.lower().strip()
    if normalized not in VIEW_PRESETS:
        raise ValueError(f"Unsupported view '{view}'. Expected one of: {sorted(VIEW_PRESETS)}")
    return VIEW_PRESETS[normalized]


def _group_connection(a_index: int, b_index: int) -> str:
    a_side = _classify_side(LANDMARK_NAMES[a_index])
    b_side = _classify_side(LANDMARK_NAMES[b_index])
    if a_side == b_side:
        return a_side
    return "center"


def _line_trace(name: str, color: str, segments: list[tuple[np.ndarray, np.ndarray]], width: float = 4.0) -> go.Scatter3d:
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []

    for start, end in segments:
        xs.extend([float(start[0]), float(end[0]), None])
        ys.extend([float(start[1]), float(end[1]), None])
        zs.extend([float(start[2]), float(end[2]), None])

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name=name,
        line=dict(color=color, width=width),
        hoverinfo="skip",
        showlegend=True,
    )


def create_3d_skeleton_figure(
    frame_points: np.ndarray | Sequence[Sequence[float]],
    *,
    visibility: np.ndarray | Sequence[float] | None = None,
    attention: np.ndarray | Sequence[float] | None = None,
    extra_hover_data: dict[str, np.ndarray | Sequence[float] | None] | None = None,
    interpolated_landmarks: Iterable[int] | None = None,
    highlight_landmarks: Iterable[int] | None = None,
    label_overrides: dict[int, str] | None = None,
    display_mode: str = "raw_pose",
    title: str | None = None,
    view: str = "isometric",
    normalize_display: bool = True,
    show_axes: bool = True,
    show_labels: bool = True,
    show_bbox: bool = False,
) -> go.Figure:
    coords, vis = prepare_frame_payload(frame_points, visibility=visibility)
    display_coords = _normalize_display_coordinates(coords) if normalize_display else coords.copy()
    attn = _attention_to_scale(attention)
    interpolated = set(int(index) for index in interpolated_landmarks or ())
    highlighted = {int(index) for index in highlight_landmarks or ()}
    overrides = {int(index): str(text) for index, text in (label_overrides or {}).items()}

    extra_hover_columns: list[tuple[str, np.ndarray]] = []
    if extra_hover_data:
        for key, values in extra_hover_data.items():
            if values is None:
                continue
            column = np.asarray(values, dtype=np.float32).reshape(-1)
            if column.shape != (LANDMARK_COUNT,):
                raise ValueError(f"extra_hover_data['{key}'] must have shape ({LANDMARK_COUNT},), got {column.shape!r}")
            if not np.isfinite(column).all():
                raise ValueError(f"extra_hover_data['{key}'] contains NaN or Inf")
            extra_hover_columns.append((key, column))

    normalized_display_mode = display_mode.strip().lower().replace(" ", "_")
    if normalized_display_mode not in {"raw_pose", "occlusion_visibility", "interpolated_landmarks"}:
        raise ValueError(
            "display_mode must be one of: raw_pose, occlusion_visibility, interpolated_landmarks"
        )

    grouped_segments: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"left": [], "right": [], "center": []}

    for start_index, end_index in POSE_CONNECTIONS:
        if start_index >= LANDMARK_COUNT or end_index >= LANDMARK_COUNT:
            continue
        group = _group_connection(start_index, end_index)
        grouped_segments[group].append((display_coords[start_index], display_coords[end_index]))

    marker_colors = [
        _point_color(_classify_side(LANDMARK_NAMES[index]), float(vis[index])) for index in range(LANDMARK_COUNT)
    ]
    if attn is not None:
        marker_colors = [
            _mix_colors(marker_colors[index], ATTENTION_COLOR, float(attn[index])) for index in range(LANDMARK_COUNT)
        ]
    marker_sizes = _marker_sizes(attn, vis)
    marker_line_width = 1.4 if attn is not None or interpolated else 0.8

    if normalized_display_mode == "occlusion_visibility":
        marker_colors = [LOW_VISIBILITY_COLOR if float(vis[index]) < 0.4 else marker_colors[index] for index in range(LANDMARK_COUNT)]
    if normalized_display_mode == "interpolated_landmarks" and interpolated:
        marker_colors = [ATTENTION_COLOR if index in interpolated else marker_colors[index] for index in range(LANDMARK_COUNT)]

    custom_columns = [
        np.arange(LANDMARK_COUNT, dtype=np.int32),
        np.asarray(LANDMARK_NAMES, dtype=object),
        vis.astype(np.float32),
        np.asarray([1 if index in interpolated else 0 for index in range(LANDMARK_COUNT)], dtype=np.int32),
    ]
    custom_names = ["index", "name", "visibility", "interpolated"]
    for key, column in extra_hover_columns:
        custom_columns.append(column.astype(np.float32))
        custom_names.append(key)
    customdata = np.column_stack(custom_columns)

    marker_text = [overrides.get(index, name.replace("_", " ").title()) for index, name in enumerate(LANDMARK_NAMES)] if show_labels else None

    hovertemplate = (
        "Landmark: %{customdata[1]}<br>"
        "Index: %{customdata[0]}<br>"
        "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<br>"
        "Visibility: %{customdata[2]:.3f}<br>"
        "Interpolated: %{customdata[3]}"
    )
    hover_index = 4
    for key, _ in extra_hover_columns:
        if key == "prior":
            hovertemplate += f"<br>Prior P(v): %{{customdata[{hover_index}]:.3f}}"
        elif key == "learned":
            hovertemplate += f"<br>Learned A_s(v): %{{customdata[{hover_index}]:.3f}}"
        elif key == "fused":
            hovertemplate += f"<br>Fused A_f(v): %{{customdata[{hover_index}]:.3f}}"
        elif key == "occlusion":
            hovertemplate += f"<br>Occlusion: %{{customdata[{hover_index}]:.3f}}"
        else:
            hovertemplate += f"<br>{key}: %{{customdata[{hover_index}]:.3f}}"
        hover_index += 1
    if highlighted:
        hovertemplate += "<br>Highlighted: %{customdata[0]}"

    marker_trace = go.Scatter3d(
        x=display_coords[:, 0],
        y=display_coords[:, 1],
        z=display_coords[:, 2],
        mode="markers+text" if show_labels else "markers",
        name="Landmarks",
        text=marker_text,
        textposition="top center",
        marker=dict(
            size=marker_sizes,
            color=marker_colors,
            line=dict(color=ATTENTION_COLOR if attn is not None or highlighted else "#111827", width=marker_line_width),
            opacity=0.98,
        ),
        customdata=customdata,
        hovertemplate=hovertemplate + "<extra></extra>",
        showlegend=True,
    )

    fig = go.Figure()
    fig.add_trace(_line_trace("Left chain", LEFT_COLOR, grouped_segments["left"]))
    fig.add_trace(_line_trace("Right chain", RIGHT_COLOR, grouped_segments["right"]))
    fig.add_trace(_line_trace("Center chain", CENTER_COLOR, grouped_segments["center"], width=3.5))
    fig.add_trace(marker_trace)

    if show_bbox:
        mins = display_coords.min(axis=0)
        maxs = display_coords.max(axis=0)
        bbox_points = np.array([
            [mins[0], mins[1], mins[2]], [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]], [mins[0], maxs[1], mins[2]], [mins[0], mins[1], mins[2]],
            [mins[0], mins[1], maxs[2]], [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]], [mins[0], maxs[1], maxs[2]], [mins[0], mins[1], maxs[2]],
            [mins[0], mins[1], mins[2]], [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], mins[2]], [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], mins[2]], [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], mins[2]], [mins[0], maxs[1], maxs[2]],
        ], dtype=np.float32)
        fig.add_trace(
            go.Scatter3d(
                x=bbox_points[:, 0],
                y=bbox_points[:, 1],
                z=bbox_points[:, 2],
                mode="lines",
                name="Bounding Box",
                line=dict(color="#fbbf24", width=2),
                hoverinfo="skip",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title or "Mini 3D Skeleton Viewer",
        margin=dict(l=0, r=0, t=48, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font=dict(color="#e5e7eb", family="Inter, Segoe UI, Arial, sans-serif"),
        scene=dict(
            aspectmode="cube",
            camera=_camera_for_view(view),
            xaxis=dict(title="X", visible=show_axes, color="#e5e7eb", gridcolor="#1f2937", zerolinecolor="#1f2937"),
            yaxis=dict(title="Y", visible=show_axes, color="#e5e7eb", gridcolor="#1f2937", zerolinecolor="#1f2937"),
            zaxis=dict(title="Z", visible=show_axes, color="#e5e7eb", gridcolor="#1f2937", zerolinecolor="#1f2937"),
            bgcolor="#0b1220",
        ),
        showlegend=True,
    )

    if normalize_display:
        fig.add_annotation(
            text="Display-normalized copy only; source tensor is unchanged.",
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.01,
            showarrow=False,
            font=dict(color="#cbd5e1", size=11),
        )

    return fig