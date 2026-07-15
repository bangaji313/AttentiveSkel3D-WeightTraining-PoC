from __future__ import annotations

import inspect
import traceback
from typing import Any, Callable

import streamlit as st


def _supports_stretch_width_param() -> bool:
    """Return True only when st.image width annotation indicates string support."""
    params = inspect.signature(st.image).parameters
    if "width" not in params:
        return False

    width_param = params["width"]
    annotation_repr = str(width_param.annotation).lower()
    default_repr = str(width_param.default).lower()

    # Newer Streamlit supports width="stretch" as string value.
    # Older versions expose width as int|None and should not receive string.
    return "str" in annotation_repr or "stretch" in default_repr


def display_image_compat(
    image: Any,
    *,
    caption: str | list[str] | None = None,
    stretch: bool = True,
    channels: str = "RGB",
    clamp: bool = False,
    output_format: str = "auto",
) -> None:
    """
    Streamlit image wrapper that adapts to API differences across versions.

    This function does NOT swallow errors silently. If Streamlit rejects
    non-width kwargs (e.g., channels), the exception is allowed to surface.
    """
    params = inspect.signature(st.image).parameters

    kwargs: dict[str, Any] = {}
    if "caption" in params:
        kwargs["caption"] = caption
    if "channels" in params:
        kwargs["channels"] = channels
    if "clamp" in params:
        kwargs["clamp"] = clamp
    if "output_format" in params:
        kwargs["output_format"] = output_format

    if stretch:
        if "width" in params and _supports_stretch_width_param():
            kwargs["width"] = "stretch"
        elif "use_container_width" in params:
            kwargs["use_container_width"] = True
        elif "use_column_width" in params:
            kwargs["use_column_width"] = True

    st.image(image, **kwargs)


def render_tab_with_debug(tab_name: str, render_fn: Callable[..., None], *args: Any, **kwargs: Any) -> None:
    """
    Render a tab safely and show local debug output if visualization fails.

    This keeps failure isolated to one tab and preserves visibility of root cause.
    """
    try:
        render_fn(*args, **kwargs)
    except Exception as exc:
        st.error(f"❌ Gagal merender tab '{tab_name}': {exc}")
        with st.expander(f"Debug traceback — {tab_name}"):
            st.code(traceback.format_exc())
