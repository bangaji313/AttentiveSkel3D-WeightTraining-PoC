# web_app/explainer_v2.py
#
# Explanation Engine — AttentiveSkel-3D V2
#
# Menyediakan fungsi-fungsi XAI yang TIDAK mengubah:
#   - arsitektur model, bobot checkpoint, hasil prediksi, pipeline preprocessing
#
# Semua nilai explanation traceable ke perhitungan numerik aktual.

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Nama sendi MediaPipe BlazePose (33 landmark, index 0–32)
# ---------------------------------------------------------------------------
LANDMARK_NAMES: List[str] = [
    "Nose", "L.Eye(in)", "L.Eye", "L.Eye(out)", "R.Eye(in)", "R.Eye",
    "R.Eye(out)", "L.Ear", "R.Ear", "L.Mouth", "R.Mouth",
    "L.Shoulder", "R.Shoulder", "L.Elbow", "R.Elbow",
    "L.Wrist", "R.Wrist", "L.Pinky", "R.Pinky", "L.Index", "R.Index",
    "L.Thumb", "R.Thumb", "L.Hip", "R.Hip", "L.Knee", "R.Knee",
    "L.Ankle", "R.Ankle", "L.Heel", "R.Heel", "L.FootIdx", "R.FootIdx",
]

# Validator Reference Joints per exercise (displayed as OUTLINE only, NOT model attention)
# Squat uses final validator reference: Shoulder 11/12, Hip 23/24, Knee 25/26, Ankle 27/28
BIOMECHANICAL_REFERENCE: Dict[str, List[int]] = {
    "Squat":       [11, 12, 23, 24, 25, 26, 27, 28],
    "Bench Press": [11, 12, 13, 14, 15, 16],
    "Deadlift":    [11, 12, 23, 24, 25, 26],
}

MODEL_SCENARIOS: Dict[str, str] = {
    "S1 — Baseline":      "best_model_baseline.pth",
    "S2 — Full Model":    "best_model_v2.pth",
    "S3a — Hanya BSP":    "best_model_s3a_bsp_holdout_v2.pth",  # holdout resmi seed=42, epoch=52
    "S3b — BSP+LS":       "best_model_ablasi_c.pth",  # best_model_ablasi_c has BSP + LS
    "S3c — BSP+Temporal": "best_model_ablasi_b.pth",  # best_model_ablasi_b has BSP + Temporal
}


# ===========================================================================
# Phase 2A — Forward pass dengan ekstraksi intermediate attention tensors
# ===========================================================================

def forward_with_attention(model: torch.nn.Module, x: torch.Tensor) -> Dict:
    """
    Jalankan forward pass dan ekstraksi intermediate attention tensors
    TANPA mengubah arsitektur atau bobot model.

    Returns dict dengan:
      logits           : (B, 64, 2) — identik dengan model(x)
      bsp_weights      : np.ndarray (33,) atau None
      ls_weights       : np.ndarray (128,) atau None  [channel, BUKAN sendi]
      temporal_weights : np.ndarray (T',) atau None   [T'~32]
    """
    result: Dict = {
        "logits": None, "bsp_weights": None,
        "ls_weights": None, "temporal_weights": None,
    }

    with torch.no_grad():
        if hasattr(model, "biomechanical_spatial_prior"):
            bsp_raw = model.biomechanical_spatial_prior
            result["bsp_weights"] = torch.sigmoid(bsp_raw).detach().cpu().squeeze().numpy()

        ls_captured: List[Optional[torch.Tensor]] = [None]
        def _ls_hook(module, inp, out):
            ls_captured[0] = out.detach().cpu()

        ls_handle = None
        if hasattr(model, "learned_spatial_attention"):
            ls_handle = model.learned_spatial_attention.register_forward_hook(_ls_hook)

        t_captured: List[Optional[torch.Tensor]] = [None]
        def _t_hook(module, inp, out):
            scores = out.detach().cpu()
            scores_m = scores.mean(dim=[3, 4])
            t_w = F.softmax(scores_m, dim=2)
            t_captured[0] = t_w.squeeze().numpy()

        t_handle = None
        if hasattr(model, "temporal_attention"):
            t_handle = model.temporal_attention.register_forward_hook(_t_hook)

        logits = model(x)
        result["logits"] = logits

        if ls_handle is not None: ls_handle.remove()
        if t_handle is not None:  t_handle.remove()

        if ls_captured[0] is not None:
            result["ls_weights"] = ls_captured[0].mean(dim=0).numpy()
        if t_captured[0] is not None:
            result["temporal_weights"] = t_captured[0]

    return result


# ===========================================================================
# Phase 2B — Perturbation-based Joint Influence Attribution
# ===========================================================================

def joint_influence(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hitung Joint Influence Attribution via mean-ablation perturbation.

    Returns:
        influence  : (64, 33)
        delta_prob : (64, 33)
    """
    model.eval()
    x = input_tensor.to(device)

    with torch.no_grad():
        logits_orig = model(x)
        probs_orig  = F.softmax(logits_orig, dim=2)
        margin_orig = (logits_orig[0,:,1] - logits_orig[0,:,0]).cpu().numpy()
        prob_salah_orig = probs_orig[0,:,1].cpu().numpy()

        temporal_mean = x.mean(dim=1, keepdim=True)

        perturbed_list = []
        for v in range(33):
            xp = x.clone()
            xp[:, :, v, :] = temporal_mean[:, :, v, :]
            perturbed_list.append(xp)

        x_batch = torch.cat(perturbed_list, dim=0)

        try:
            logits_abla = model(x_batch)
        except RuntimeError:
            logits_abla = torch.cat([model(xp) for xp in perturbed_list], dim=0)

        probs_abla = F.softmax(logits_abla, dim=2)
        margin_abla     = (logits_abla[:,:,1] - logits_abla[:,:,0]).cpu().numpy()
        prob_salah_abla = probs_abla[:,:,1].cpu().numpy()

    influence_vt  = margin_orig[None, :] - margin_abla
    delta_prob_vt = prob_salah_orig[None, :] - prob_salah_abla

    return influence_vt.T, delta_prob_vt.T


# ===========================================================================
# Phase 5 — Quantitative Attribution Metrics
# ===========================================================================

def sequence_joint_score(influence: np.ndarray) -> np.ndarray:
    """S[v] = mean_t(|influence[t, v]|) — magnitude of influence (importance) (33,)"""
    return np.abs(influence).mean(axis=0)


def signed_mean_influence(influence: np.ndarray) -> np.ndarray:
    """signed_mean[v] = mean_t(influence[t, v]) — net directional tendency (33,)"""
    return influence.mean(axis=0)


def reference_attribution_share(S: np.ndarray, influence: np.ndarray, exercise: str) -> Dict:
    """
    Hitung Reference Attribution Share (RAS) & Attribution Lift.
    RAS = sum(S[ref]) / sum(S)
    uniform = |ref| / 33
    attribution_lift = RAS / uniform
    """
    ref = BIOMECHANICAL_REFERENCE.get(exercise, [])
    total_S = float(S.sum())
    signed_mean = signed_mean_influence(influence)

    if total_S < 1e-10:
        return {
            "exercise": exercise,
            "validator_reference_joints": ref,
            "RAS": 0.0,
            "uniform": len(ref) / 33.0,
            "attribution_lift": 1.0,
            "top5": [],
        }

    ras        = float(S[ref].sum()) / total_S if ref else 0.0
    uniform    = len(ref) / 33.0
    attr_lift  = ras / uniform if uniform > 1e-10 else 0.0

    top5_idx   = np.argsort(S)[::-1][:5].tolist()
    top5 = []
    for i, j in enumerate(top5_idx):
        sm = float(signed_mean[j])
        if sm > 1e-4:
            tendency = "SALAH"
        elif sm < -1e-4:
            tendency = "BENAR"
        else:
            tendency = "Netral"

        top5.append({
            "rank": i + 1,
            "joint_idx": int(j),
            "name": LANDMARK_NAMES[j],
            "importance_score": round(float(S[j]), 6),
            "signed_mean_influence": round(sm, 6),
            "tendency": tendency,
        })

    return {
        "exercise": exercise,
        "validator_reference_joints": ref,
        "RAS": round(ras, 4),
        "uniform": round(uniform, 4),
        "attribution_lift": round(attr_lift, 4),
        "top5": top5,
    }

# Alias for backwards compatibility
biomechanical_focus_mass = reference_attribution_share


# ===========================================================================
# Phase 7 — Perturbation Faithfulness Check
# ===========================================================================

def perturbation_faithfulness_check(
    model: torch.nn.Module,
    x: torch.Tensor,
    S: np.ndarray,
    device: torch.device,
    seed: int = 42,
) -> Dict:
    """Ablasi bersama Top-3 joints vs 3 random joints."""
    rng = random.Random(seed)
    model.eval()
    x = x.to(device)

    top3   = np.argsort(S)[::-1][:3].tolist()
    others = [v for v in range(33) if v not in top3]
    rand3  = rng.sample(others, 3)

    def _ablated_margin(joints):
        tmean = x.mean(dim=1, keepdim=True)
        xabl  = x.clone()
        for v in joints:
            xabl[:, :, v, :] = tmean[:, :, v, :]
        with torch.no_grad():
            logits = model(xabl)
        return float((logits[0,:,1] - logits[0,:,0]).cpu().numpy().mean())

    with torch.no_grad():
        logits_orig = model(x)
    m_orig   = float((logits_orig[0,:,1] - logits_orig[0,:,0]).cpu().numpy().mean())
    m_top3   = _ablated_margin(top3)
    m_rand3  = _ablated_margin(rand3)
    d_top3   = abs(m_orig - m_top3)
    d_rand3  = abs(m_orig - m_rand3)
    faith    = d_top3 / (d_rand3 + 1e-10)

    return {
        "top3_joints":   top3,
        "top3_names":    [LANDMARK_NAMES[v] for v in top3],
        "random3_joints": rand3,
        "random3_names": [LANDMARK_NAMES[v] for v in rand3],
        "margin_original_mean":       round(m_orig, 4),
        "margin_top3_ablated_mean":   round(m_top3, 4),
        "margin_random3_ablated_mean": round(m_rand3, 4),
        "delta_top3":   round(d_top3, 4),
        "delta_random3": round(d_rand3, 4),
        "faithfulness_ratio": round(faith, 4),
    }


# ===========================================================================
# Phase 6 — Cross-Model Comparison
# ===========================================================================

def run_all_models_compare(
    models_dir: Path,
    processed_tensor: torch.Tensor,
    exercise: str,
    device: torch.device,
) -> Dict:
    """Jalankan tensor yang SAMA melalui semua skenario model."""
    import sys, os
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.models.arsitektur_v2 import AttentiveSkel3DPerFrame

    results = {}
    x = processed_tensor.to(device)

    for scenario_name, filename in MODEL_SCENARIOS.items():
        model_path = models_dir / filename
        if not model_path.exists():
            results[scenario_name] = {"available": False}
            continue
        try:
            checkpoint = torch.load(str(model_path), map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            use_sp = any(k.startswith("biomechanical_spatial_prior") for k in state_dict)
            use_ls = any(k.startswith("learned_spatial_attention") for k in state_dict)
            use_ta = any(k.startswith("temporal_attention") for k in state_dict)
            mdl = AttentiveSkel3DPerFrame(2, use_sp, use_ls, use_ta)
            mdl.load_state_dict(state_dict)
            mdl.to(device).eval()

            attn_out = forward_with_attention(mdl, x)
            logits = attn_out["logits"]
            probs  = F.softmax(logits, dim=2)
            preds  = logits.argmax(dim=2)
            preds_np = preds.squeeze().cpu().numpy()
            probs_np = probs.squeeze().cpu().numpy()
            n_benar = int((preds_np == 0).sum())
            n_salah = int((preds_np == 1).sum())

            inf, _ = joint_influence(mdl, x, device)
            S      = sequence_joint_score(inf)
            ras    = reference_attribution_share(S, inf, exercise)
            top_j  = int(np.argmax(S))
            majority = 0 if n_benar >= n_salah else 1
            disagree = [int(t) for t in range(64) if preds_np[t] != majority]

            results[scenario_name] = {
                "available": True, "use_bsp": use_sp, "use_ls": use_ls, "use_temporal": use_ta,
                "preds": preds_np.tolist(), "probs_salah": probs_np[:, 1].tolist(),
                "n_benar": n_benar, "n_salah": n_salah,
                "disagree_frames": disagree,
                "RAS": ras["RAS"], "attribution_lift": ras["attribution_lift"],
                "top_joint_idx": top_j, "top_joint_name": LANDMARK_NAMES[top_j],
                "bsp_weights":      attn_out["bsp_weights"].tolist() if attn_out["bsp_weights"] is not None else None,
                "ls_weights":       attn_out["ls_weights"].tolist() if attn_out["ls_weights"] is not None else None,
                "temporal_weights": attn_out["temporal_weights"].tolist() if attn_out["temporal_weights"] is not None else None,
            }
            del mdl
            if device.type == "cuda": torch.cuda.empty_cache()
        except Exception as e:
            results[scenario_name] = {"available": False, "error": str(e)}

    return results


# ===========================================================================
# Phase 8 — Build explanation JSON output
# ===========================================================================

def build_explanation_json(
    *,
    video_stem: str,
    model_name: str,
    exercise: str,
    preds_np: np.ndarray,
    probs_np: np.ndarray,
    influence: np.ndarray,
    delta_prob: np.ndarray,
    bsp_weights: Optional[np.ndarray],
    ls_weights: Optional[np.ndarray],
    temporal_weights: Optional[np.ndarray],
    ras_result: Dict,
    faithfulness: Optional[Dict],
) -> Dict:
    """Buat explanation JSON yang seluruh nilainya traceable ke perhitungan numerik."""
    S = sequence_joint_score(influence)
    signed_mean = signed_mean_influence(influence)
    ref = ras_result.get("validator_reference_joints", [])
    frames = []
    for t in range(64):
        frames.append({
            "temporal_index": int(t + 1),
            "predicted_class": "SALAH" if int(preds_np[t]) == 1 else "BENAR",
            "predicted_class_id": int(preds_np[t]),
            "P_benar":  round(float(probs_np[t, 0]), 6),
            "P_salah":  round(float(probs_np[t, 1]), 6),
            "joint_influence": {LANDMARK_NAMES[v]: round(float(influence[t, v]), 6) for v in range(33)},
            "delta_probability": {LANDMARK_NAMES[v]: round(float(delta_prob[t, v]), 6) for v in range(33)},
        })

    return {
        "meta": {
            "video": video_stem,
            "model": model_name,
            "exercise": exercise,
            "total_frames": 64,
            "ground_truth_available": False,
        },
        "summary": {
            "n_benar": int((preds_np == 0).sum()),
            "n_salah": int((preds_np == 1).sum()),
            "RAS": ras_result["RAS"],
            "uniform_reference": ras_result["uniform"],
            "attribution_lift": ras_result["attribution_lift"],
            "top5_joints": ras_result["top5"],
            "validator_reference_joints": [{"idx": v, "name": LANDMARK_NAMES[v]} for v in ref],
        },
        "attention_sources": {
            "BSP_weights_33":         bsp_weights.tolist() if bsp_weights is not None else None,
            "BSP_landmark_names":     LANDMARK_NAMES if bsp_weights is not None else None,
            "LS_channel_weights_128": ls_weights.tolist() if ls_weights is not None else None,
            "LS_note":                "Channel attention (128 features) — NOT mapped to 33 joints",
            "temporal_weights_T_prime": temporal_weights.tolist() if temporal_weights is not None else None,
            "temporal_note":          "Temporal attention in downsampled T' domain (~32 frames)",
        },
        "sequence_joint_scores": {
            LANDMARK_NAMES[v]: {
                "importance_score": round(float(S[v]), 6),
                "signed_mean_influence": round(float(signed_mean[v]), 6),
            }
            for v in range(33)
        },
        "faithfulness_check": faithfulness,
        "frames": frames,
    }
