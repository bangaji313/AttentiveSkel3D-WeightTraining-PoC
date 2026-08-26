"""
Comprehensive Audit: Five Scenarios Model Analysis

Objective: Debug why all five scenarios produce identical predictions/visualizations

Scenarios:
1. Baseline 3D-CNN
2. Ablasi A - Tanpa Biomechanical Spatial Prior
3. Ablasi B - Tanpa Learned Spatial Attention
4. Ablasi C - Tanpa Temporal Attention
5. Full AttentiveSkel-3D

Analysis:
- Checkpoint integrity (SHA256, size, tensors)
- Model configuration verification
- Single input inference comparison
- Attention weight similarity
- Potential bug detection
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Any, List
import hashlib
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from scipy import stats

# Setup paths
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.model_3dcnn import AttentiveSkel3D

# Logging setup
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

SCENARIOS = {
    "Full Model": {
        "path": "models/saved_models/AttentiveSkel3D_Final.pth",
        "use_spatial_prior": True,
        "use_learned_spatial": True,
        "use_temporal_attention": True,
    },
    "Baseline 3D-CNN": {
        "path": "models/saved_models/baseline_3dcnn_model.pth",
        "use_spatial_prior": False,
        "use_learned_spatial": False,
        "use_temporal_attention": False,
    },
    "Ablasi A - No Prior": {
        "path": "models/saved_models/ablasi_a_no_prior.pth",
        "use_spatial_prior": False,
        "use_learned_spatial": True,
        "use_temporal_attention": True,
    },
    "Ablasi B - No Learned Spatial": {
        "path": "models/saved_models/ablasi_b_no_learned.pth",
        "use_spatial_prior": True,
        "use_learned_spatial": False,
        "use_temporal_attention": True,
    },
    "Ablasi C - No Temporal": {
        "path": "models/saved_models/ablasi_c_no_temporal.pth",
        "use_spatial_prior": True,
        "use_learned_spatial": True,
        "use_temporal_attention": False,
    },
}

OUTPUT_DIR = ROOT_DIR / "results" / "model_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CHECKPOINT ANALYSIS
# ============================================================================

def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def analyze_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Deeply analyze checkpoint file.
    
    Returns:
    {
        "absolute_path": str,
        "file_exists": bool,
        "file_size_mb": float,
        "sha256": str,
        "can_load": bool,
        "state_dict_keys": int,
        "total_parameters": int,
        "checkpoint_format": str,
        "tensor_info": Dict,
        "error": str or None,
    }
    """
    result = {
        "absolute_path": str(checkpoint_path.resolve()),
        "file_exists": checkpoint_path.exists(),
        "file_size_mb": 0.0,
        "sha256": "",
        "can_load": False,
        "state_dict_keys": 0,
        "total_parameters": 0,
        "checkpoint_format": "unknown",
        "tensor_info": {},
        "error": None,
    }

    if not checkpoint_path.exists():
        result["error"] = "File does not exist"
        return result

    # File size
    result["file_size_mb"] = checkpoint_path.stat().st_size / (1024 ** 2)

    # SHA256
    try:
        result["sha256"] = compute_sha256(checkpoint_path)
    except Exception as e:
        result["error"] = f"SHA256 compute failed: {e}"
        return result

    # Load checkpoint
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except Exception:
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        except Exception as e:
            result["error"] = f"Cannot load checkpoint: {e}"
            return result

    result["can_load"] = True

    # Detect format
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            result["checkpoint_format"] = "dict_with_model_state_dict"
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            result["checkpoint_format"] = "dict_with_state_dict"
        else:
            state_dict = checkpoint
            result["checkpoint_format"] = "raw_state_dict"
    else:
        state_dict = checkpoint
        result["checkpoint_format"] = "direct_state_dict"

    # State dict analysis
    if isinstance(state_dict, dict):
        result["state_dict_keys"] = len(state_dict)
        
        # Tensor info
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                result["tensor_info"][key] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": tensor.numel(),
                }
                result["total_parameters"] += tensor.numel()

    return result


# ============================================================================
# MODEL LOADING & CONFIGURATION VERIFICATION
# ============================================================================

class ModelInstanceTracker:
    """Ensure each scenario loads a separate model instance."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.ids: Dict[str, int] = {}
    
    def load(self, name: str, model: nn.Module):
        """Register model instance."""
        self.models[name] = model
        self.ids[name] = id(model)
        log.info(f"✓ Loaded '{name}' with id={self.ids[name]}")
    
    def check_uniqueness(self):
        """Check if all models are different instances."""
        ids_list = list(self.ids.values())
        unique_ids = len(set(ids_list))
        total = len(ids_list)
        
        log.info(f"\nModel Instance Uniqueness: {unique_ids}/{total} unique")
        
        if unique_ids < total:
            log.warning("⚠️ WARNING: Duplicate model instances detected!")
            for name1, id1 in self.ids.items():
                for name2, id2 in self.ids.items():
                    if name1 < name2 and id1 == id2:
                        log.warning(f"  → '{name1}' and '{name2}' are same object!")
        
        return unique_ids == total


def load_scenario_model(
    scenario_name: str,
    config: Dict,
    checkpoint_path: Path,
    tracker: ModelInstanceTracker,
) -> Tuple[nn.Module | None, Dict[str, Any]]:
    """
    Load model for a specific scenario.
    
    Returns:
        (model, load_info)
    """
    load_info = {
        "scenario": scenario_name,
        "config": config.copy(),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "model_loaded": False,
        "missing_keys": [],
        "unexpected_keys": [],
        "error": None,
    }

    try:
        # Create model with EXPLICIT configuration
        model = AttentiveSkel3D(
            num_classes=2,
            use_attention=True,
            use_spatial_prior=config["use_spatial_prior"],
            use_learned_spatial=config["use_learned_spatial"],
            use_temporal_attention=config["use_temporal_attention"],
        )

        # Load checkpoint
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        except:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

        # Extract state dict
        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint
            )
        else:
            state_dict = checkpoint

        # Load with strict=False (allow missing keys for ablations)
        incompatible = model.load_state_dict(state_dict, strict=False)

        load_info["missing_keys"] = list(incompatible.missing_keys or [])
        load_info["unexpected_keys"] = list(incompatible.unexpected_keys or [])

        # Verify configuration was properly set
        load_info["verified_config"] = {
            "use_spatial_prior": model.use_spatial_prior,
            "use_learned_spatial": model.use_learned_spatial,
            "use_temporal_attention": model.use_temporal_attention,
        }

        # Check if config matches expected (only compare architecture flags, not path)
        expected_flags = {
            k: v for k, v in config.items()
            if k in ("use_spatial_prior", "use_learned_spatial", "use_temporal_attention")
        }
        if load_info["verified_config"] != expected_flags:
            log.warning(f"⚠️ Config mismatch for '{scenario_name}':")
            log.warning(f"  Expected: {expected_flags}")
            log.warning(f"  Actual:   {load_info['verified_config']}")
        else:
            log.info(f"  ✓ Config verified: {load_info['verified_config']}")

        model.eval()
        model_loaded = True

        # Register instance
        tracker.load(scenario_name, model)

    except Exception as e:
        load_info["error"] = str(e)
        model = None
        model_loaded = False
        log.error(f"✗ Failed to load '{scenario_name}': {e}")

    load_info["model_loaded"] = model_loaded
    return model, load_info


# ============================================================================
# INFERENCE & OUTPUT COMPARISON
# ============================================================================

def extract_attention_weights(model: nn.Module) -> np.ndarray:
    """
    Extract spatial attention weights (33-dim) from model.
    
    Logic:
    - If use_spatial_prior: return sigmoid(biomechanical_spatial_prior)
    - Else: return uniform 0.5 array
    """
    if model.use_spatial_prior and hasattr(model, "biomechanical_spatial_prior"):
        prior = model.biomechanical_spatial_prior
        weights = torch.sigmoid(prior).detach().numpy().flatten()
    else:
        weights = np.full(33, 0.5, dtype=np.float32)

    # Min-Max normalization
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min > 1e-8:
        weights = (weights - w_min) / (w_max - w_min)

    return weights


def run_inference(
    models: Dict[str, nn.Module],
    input_tensor: torch.Tensor,
) -> Dict[str, Dict[str, Any]]:
    """
    Run inference on all models with same input.
    
    Returns per-scenario:
    {
        "logits": array,
        "softmax": array,
        "pred_class": int,
        "confidence": float,
        "spatial_attention": array (33,),
        "top5_joints": List[(idx, value)],
        "model_config": dict,
    }
    """
    results = {}

    with torch.no_grad():
        for scenario_name, model in models.items():
            try:
                logits = model(input_tensor)
                softmax = torch.softmax(logits, dim=-1)
                pred_class = int(logits.argmax(dim=-1).item())
                confidence = float(softmax[0, pred_class].item())

                spatial_attn = extract_attention_weights(model)
                top5_idx = np.argsort(spatial_attn)[::-1][:5]
                top5_joints = [(int(idx), float(spatial_attn[idx])) for idx in top5_idx]

                results[scenario_name] = {
                    "logits": logits.cpu().numpy().flatten(),
                    "softmax": softmax.cpu().numpy().flatten(),
                    "pred_class": pred_class,
                    "confidence": confidence,
                    "spatial_attention": spatial_attn,
                    "top5_joints": top5_joints,
                    "model_config": {
                        "use_spatial_prior": model.use_spatial_prior,
                        "use_learned_spatial": model.use_learned_spatial,
                        "use_temporal_attention": model.use_temporal_attention,
                    },
                }

            except Exception as e:
                log.error(f"✗ Inference failed for '{scenario_name}': {e}")
                results[scenario_name] = {"error": str(e)}

    return results


def compute_pairwise_similarity(outputs: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare outputs between scenarios.
    
    Metrics:
    - L1 distance (logits)
    - L2 distance (logits)
    - Cosine similarity (softmax)
    - Pearson correlation (spatial attention)
    - Exact match (logits & softmax)
    """
    scenarios = list(outputs.keys())
    n = len(scenarios)
    
    records = []

    for i, s1 in enumerate(scenarios):
        for s2 in scenarios[i+1:]:
            out1 = outputs[s1]
            out2 = outputs[s2]

            if "error" in out1 or "error" in out2:
                continue

            logits1 = out1["logits"]
            logits2 = out2["logits"]
            softmax1 = out1["softmax"]
            softmax2 = out2["softmax"]
            attn1 = out1["spatial_attention"]
            attn2 = out2["spatial_attention"]

            # L1 distance
            l1_logits = np.abs(logits1 - logits2).mean()

            # L2 distance
            l2_logits = np.sqrt(np.mean((logits1 - logits2) ** 2))

            # Cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            cos_sim = float(cosine_similarity([softmax1], [softmax2])[0, 0])

            # Pearson correlation (spatial attention)
            pearson_corr, _ = stats.pearsonr(attn1, attn2)

            # Exact match
            logits_exact = np.allclose(logits1, logits2, atol=1e-6)
            softmax_exact = np.allclose(softmax1, softmax2, atol=1e-6)
            attn_exact = np.allclose(attn1, attn2, atol=1e-6)

            records.append({
                "Scenario_1": s1,
                "Scenario_2": s2,
                "L1_Distance_Logits": l1_logits,
                "L2_Distance_Logits": l2_logits,
                "Cosine_Similarity_Softmax": cos_sim,
                "Pearson_Corr_Attention": pearson_corr,
                "Logits_Exact_Match": logits_exact,
                "Softmax_Exact_Match": softmax_exact,
                "Attention_Exact_Match": attn_exact,
            })

    df = pd.DataFrame(records)
    return df


# ============================================================================
# BUG DETECTION
# ============================================================================

def detect_bugs(
    checkpoint_analysis: Dict[str, Dict],
    load_infos: Dict[str, Dict],
    model_outputs: Dict[str, Dict],
    similarity_df: pd.DataFrame,
) -> List[str]:
    """
    Check for common bugs that could cause all scenarios to be identical.
    """
    bugs = []

    # Bug 1: All checkpoints point to same file
    file_paths = set()
    for name, analysis in checkpoint_analysis.items():
        if analysis["file_exists"]:
            file_paths.add(analysis["sha256"])

    if len(file_paths) < len(checkpoint_analysis):
        bugs.append("🐛 BUG: Multiple scenarios point to same checkpoint file (by SHA256)!")

    # Bug 2: Missing keys not handled
    for name, load_info in load_infos.items():
        if load_info["missing_keys"] and len(load_info["missing_keys"]) > 10:
            bugs.append(f"🐛 BUG: '{name}' has many missing keys ({len(load_info['missing_keys'])}). "
                       f"Are ablations properly loaded?")

    # Bug 3: Model instance reuse
    model_ids = set()
    for name, output in model_outputs.items():
        # This is hard to detect post-hoc, but we can warn about it
        pass

    # Bug 4: All outputs identical
    identical_pairs = (similarity_df["Logits_Exact_Match"] == True).sum()
    if identical_pairs > 0:
        bugs.append(f"🐛 BUG: {identical_pairs} pairs of scenarios have IDENTICAL logits!")
        identical_list = similarity_df[similarity_df["Logits_Exact_Match"] == True]
        for _, row in identical_list.iterrows():
            bugs.append(f"       → {row['Scenario_1']} == {row['Scenario_2']}")

    # Bug 5: Very high cosine similarity in ALL pairs (skip if just a subset)
    # Note: high cosine similarity alone is not a bug — models can agree on dominant class
    # Only flag if logits are also very similar AND there are many identical pairs
    identical_logit_pairs = (similarity_df["Logits_Exact_Match"] == True).sum()
    if identical_logit_pairs > 0:
        pass  # already flagged above as BUG 4

    # Bug 6: Attention weights identical — only flag if BOTH models have BSP active
    # (no-BSP models correctly return uniform 0.5; that's expected behavior)
    for _, row in similarity_df.iterrows():
        if row["Attention_Exact_Match"]:
            s1, s2 = row["Scenario_1"], row["Scenario_2"]
            s1_has_bsp = model_outputs.get(s1, {}).get("model_config", {}).get("use_spatial_prior", False)
            s2_has_bsp = model_outputs.get(s2, {}).get("model_config", {}).get("use_spatial_prior", False)
            if s1_has_bsp and s2_has_bsp:
                bugs.append(f"🐛 BUG: '{s1}' and '{s2}' have IDENTICAL spatial attention despite both having BSP!")

    # Bug 7: Expected configuration differences not reflected in outputs
    for name, output in model_outputs.items():
        if "error" not in output:
            config = output["model_config"]
            if not config["use_spatial_prior"] and not config["use_learned_spatial"]:
                # Baseline — should have uniform attention
                if not np.allclose(output["spatial_attention"], 0.5, atol=0.1):
                    bugs.append(f"🐛 BUG: '{name}' is baseline but attention not uniform!")

    return bugs


# ============================================================================
# MAIN AUDIT
# ============================================================================

def main():
    """Execute full audit."""
    log.info("=" * 80)
    log.info("FIVE SCENARIOS MODEL AUDIT")
    log.info("=" * 80)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 1: Checkpoint Analysis
    # ────────────────────────────────────────────────────────────────────────
    log.info("\n[STEP 1] Analyzing Checkpoints...")
    checkpoint_analysis = {}

    for scenario_name, config in SCENARIOS.items():
        rel_path = config["path"]
        abs_path = ROOT_DIR / rel_path
        analysis = analyze_checkpoint(abs_path)
        checkpoint_analysis[scenario_name] = analysis

        log.info(f"\n{scenario_name}:")
        log.info(f"  Path: {analysis['absolute_path']}")
        log.info(f"  Exists: {analysis['file_exists']}")
        if analysis["file_exists"]:
            log.info(f"  Size: {analysis['file_size_mb']:.2f} MB")
            log.info(f"  SHA256: {analysis['sha256'][:16]}...")
            log.info(f"  State dict keys: {analysis['state_dict_keys']}")
            log.info(f"  Total parameters: {analysis['total_parameters']:,}")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 2: Model Loading & Configuration Verification
    # ────────────────────────────────────────────────────────────────────────
    log.info("\n[STEP 2] Loading Models & Verifying Configuration...")
    
    tracker = ModelInstanceTracker()
    models = {}
    load_infos = {}

    for scenario_name, config in SCENARIOS.items():
        checkpoint_path = ROOT_DIR / config["path"]
        model, load_info = load_scenario_model(
            scenario_name, config, checkpoint_path, tracker
        )
        models[scenario_name] = model
        load_infos[scenario_name] = load_info

    # Check instance uniqueness
    tracker.check_uniqueness()

    # ────────────────────────────────────────────────────────────────────────
    # STEP 3: Inference with Single Input
    # ────────────────────────────────────────────────────────────────────────
    log.info("\n[STEP 3] Running Inference with Single Input...")
    
    # Create synthetic input (B=1, T=64, L=33, C=3)
    input_tensor = torch.randn(1, 64, 33, 3, dtype=torch.float32)
    log.info(f"Input tensor shape: {input_tensor.shape}")

    model_outputs = run_inference(models, input_tensor)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 4: Output Comparison
    # ────────────────────────────────────────────────────────────────────────
    log.info("\n[STEP 4] Comparing Outputs Between Scenarios...")
    
    similarity_df = compute_pairwise_similarity(model_outputs)
    log.info(f"\nSimilarity Matrix ({len(similarity_df)} comparisons):")
    log.info(similarity_df.to_string(index=False))

    # ────────────────────────────────────────────────────────────────────────
    # STEP 5: Bug Detection
    # ────────────────────────────────────────────────────────────────────────
    log.info("\n[STEP 5] Detecting Potential Bugs...")
    
    bugs = detect_bugs(checkpoint_analysis, load_infos, model_outputs, similarity_df)

    if bugs:
        log.warning("\n🚨 POTENTIAL BUGS FOUND:")
        for bug in bugs:
            log.warning(bug)
    else:
        log.info("\n✓ No obvious bugs detected in this audit layer")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 6: Output Files
    # ────────────────────────────────────────────────────────────────────────
    log.info(f"\n[STEP 6] Generating Output Files...")

    # Checkpoint summary
    checkpoint_summary = []
    for scenario_name, analysis in checkpoint_analysis.items():
        checkpoint_summary.append({
            "Scenario": scenario_name,
            "Absolute_Path": analysis["absolute_path"],
            "File_Exists": analysis["file_exists"],
            "File_Size_MB": analysis["file_size_mb"],
            "SHA256": analysis["sha256"][:16] if analysis["sha256"] else "N/A",
            "State_Dict_Keys": analysis["state_dict_keys"],
            "Total_Parameters": analysis["total_parameters"],
            "Error": analysis["error"] or "None",
        })

    checkpoint_df = pd.DataFrame(checkpoint_summary)
    checkpoint_csv = OUTPUT_DIR / "checkpoint_summary.csv"
    checkpoint_df.to_csv(checkpoint_csv, index=False)
    log.info(f"✓ Saved: {checkpoint_csv}")

    # Prediction comparison
    prediction_records = []
    for scenario_name, output in model_outputs.items():
        if "error" not in output:
            prediction_records.append({
                "Scenario": scenario_name,
                "Logits_Class0": output["logits"][0],
                "Logits_Class1": output["logits"][1],
                "Softmax_Class0": output["softmax"][0],
                "Softmax_Class1": output["softmax"][1],
                "Pred_Class": output["pred_class"],
                "Confidence": output["confidence"],
                "Top5_Joints": str(output["top5_joints"]),
            })

    prediction_df = pd.DataFrame(prediction_records)
    prediction_csv = OUTPUT_DIR / "prediction_comparison.csv"
    prediction_df.to_csv(prediction_csv, index=False)
    log.info(f"✓ Saved: {prediction_csv}")

    # Attention similarity
    similarity_csv = OUTPUT_DIR / "attention_similarity.csv"
    similarity_df.to_csv(similarity_csv, index=False)
    log.info(f"✓ Saved: {similarity_csv}")

    # Save attention arrays
    attention_dir = OUTPUT_DIR / "attention_arrays"
    attention_dir.mkdir(parents=True, exist_ok=True)
    for scenario_name, output in model_outputs.items():
        if "error" not in output:
            attn_array = output["spatial_attention"]
            attn_file = attention_dir / f"{scenario_name.replace(' ', '_')}_attention.npy"
            np.save(attn_file, attn_array)
    log.info(f"✓ Saved attention arrays to: {attention_dir}")

    # Audit report
    report_file = OUTPUT_DIR / "audit_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Five Scenarios Model Audit Report\n\n")
        f.write(f"Generated: {pd.Timestamp.now().isoformat()}\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Total scenarios: {len(SCENARIOS)}\n")
        f.write(f"- Successfully loaded: {sum(1 for o in model_outputs.values() if 'error' not in o)}\n")
        f.write(f"- Bugs detected: {len(bugs)}\n\n")

        f.write("## Checkpoints\n\n")
        f.write("```\n" + checkpoint_df.to_string(index=False) + "\n```\n\n")

        f.write("## Predictions\n\n")
        f.write("```\n" + prediction_df.to_string(index=False) + "\n```\n\n")

        f.write("## Similarity Matrix\n\n")
        f.write("```\n" + similarity_df.to_string(index=False) + "\n```\n\n")

        if bugs:
            f.write("## Bugs Detected\n\n")
            for bug in bugs:
                f.write(f"- {bug}\n")
        else:
            f.write("## No Bugs Detected\n\n")
            f.write("All scenarios loaded successfully with distinct configurations.\n")

    log.info(f"✓ Saved: {report_file}")

    # ────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ────────────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 80)
    log.info("AUDIT COMPLETE")
    log.info("=" * 80)
    log.info(f"\nOutput directory: {OUTPUT_DIR}")
    log.info(f"Generated files:")
    log.info(f"  - checkpoint_summary.csv")
    log.info(f"  - prediction_comparison.csv")
    log.info(f"  - attention_similarity.csv")
    log.info(f"  - attention_arrays/ (npy files)")
    log.info(f"  - audit_report.md")

    return bugs


if __name__ == "__main__":
    bugs = main()
    sys.exit(0 if not bugs else 1)
