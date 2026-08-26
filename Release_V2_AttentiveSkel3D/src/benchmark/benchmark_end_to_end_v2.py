"""
src/benchmark/benchmark_end_to_end_v2.py

Benchmark Latency End-to-End AttentiveSkel-3D V2 (Refined — Single Tensor Save per Run & Correct Stage Names)
Mengevaluasi latensi pipeline lengkap dari video mentah hingga analisis prediksi AI, attribution sendi, dan rendering video heatmap.

Model yang diuji:
  S3b BSP + Learned Spatial (bobot_model/best_model_ablasi_c.pth)

Pipeline yang diukur (per-stage):
  1. Cold-Start: Model Loading
  2. Video Stream Open & Metadata Inspection (Inisialisasi OpenCV VideoCapture & metadata)
  3. Video Decoding & BlazePose Extraction (MediaPipe model_complexity=2 & cap.read frame decoding)
  4. Raw Tensor Saving (.npy ke temp — tepat 1x per run via save_output=False)
  5. Preprocessing (DataPreprocessor: resampling 64 frame, filtering, normalisasi)
  6. Model Inference & Joint Attribution (S3b forward pass + perturbation attribution, 3x model forward)
  7. Time-to-Analysis-Ready (Tahap 2-6: sampai analisis AI & attribution siap, TANPA rendering)
  8. Video Heatmap Rendering (Rendering 64 frame video heatmap dengan overlay)
  9. Total Pipeline with Rendering (Time-to-Analysis-Ready + Rendering)

Keluaran:
  - hasil_evaluasi/Latency_EndToEnd_PerRun_AttentiveSkel3D_V2.csv
  - hasil_evaluasi/Latency_EndToEnd_Summary_AttentiveSkel3D_V2.csv
  - hasil_evaluasi/Latency_EndToEnd_AttentiveSkel3D_V2.json
"""

import os
import sys
import time
import json
import csv
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Root setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Release_V2_AttentiveSkel3D
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WEB_APP_DIR = PROJECT_ROOT / "web_app"
if str(WEB_APP_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_APP_DIR))

import cv2
import numpy as np
import torch
import mediapipe as mp

from src.data.extract_pose import PoseExtractor
from src.data.preprocess import DataPreprocessor
from src.models.arsitektur_v2 import AttentiveSkel3DPerFrame
from explainer_v2 import (
    forward_with_attention,
    joint_influence,
    sequence_joint_score,
    reference_attribution_share,
    perturbation_faithfulness_check,
    build_explanation_json,
    LANDMARK_NAMES,
    BIOMECHANICAL_REFERENCE,
)

# ── Konfigurasi Benchmark ───────────────────────────────────────────────────
MODEL_REL_PATH   = Path("bobot_model/best_model_ablasi_c.pth")  # S3b BSP+LS
OUTPUT_DIR       = PROJECT_ROOT / "hasil_evaluasi"
N_MEASURED_RUNS  = 10  # 10 measured runs per video (+ 1 warm-up)

RAW_VIDEO_CANDIDATES = {
    "Bench Press": PROJECT_ROOT / "data" / "raw" / "BenchPress" / "primer_benchpress_frontal_subjek01_rep1.mp4",
    "Deadlift"   : PROJECT_ROOT / "data" / "raw" / "Deadlift"  / "primer_deadlift_lateral_subjek01_rep1.mp4",
    "Squat"      : PROJECT_ROOT / "data" / "raw" / "Squat"     / "primer_squat_frontal_subjek01_rep1.mp4",
}

DISCLAIMER_METADATA = {
    "pengukuran_menggunakan_video_tersedia"                       : True,
    "tidak_mencakup_network_latency"                              : True,
    "bukan_live_camera_latency"                                   : True,
    "belum_membuktikan_pencegahan_cedera_klinis"                    : True,
    "model_memerlukan_konteks_sekuens_64_frame"                    : True,
    "real_time_threshold_rtf"                                     : 1.0,
    "current_pipeline_meets_real_time"                            : False,
    "effective_pipeline_fps_is_not_model_fps"                     : True,
    "sequential_capture_estimate_assumes_non_overlapping_processing": True,
    "raw_tensor_written_once_per_run"                             : True,
    "video_frame_decoding_included_in_pose_stage"                 : True,
    "time_to_analysis_ready_includes_joint_attribution"          : True,
    "mediapipe_pose_reinitialized_each_run"                       : True,
    "includes_per_run_mediapipe_initialization_overhead"          : True,
    "persistent_mediapipe_steady_state"                           : False,
    "joint_attribution_method"                                    : "per-joint temporal-mean ablation",
    "joint_attribution_primary_batch_size"                        : 33,
    "stage5_forward_pass_executions_normal"                        : 3,
    "stage5_forward_pass_executions_fallback"                      : 35,
    "keterangan_metodologis": (
        "Pengukuran latency end-to-end ini menggunakan video pre-recorded yang sudah tersedia di storage lokal. "
        "Pengukuran tidak mencakup latensi jaringan (network upload/download) maupun latensi antarmuka browser. "
        "Pengukuran ini bukan live-camera real-time latency, belum membuktikan kemampuan pencegahan cedera "
        "secara klinis, dan arsitektur model memerlukan konteks sekuens 64 frame untuk klasifikasi. "
        "Penulisan raw tensor .npy hanya dilakukan 1x per run (save_output=False pada PoseExtractor)."
    )
}


def verify_raw_videos() -> Dict[str, Path]:
    """Memeriksa keberadaan video mentah untuk setiap jenis latihan."""
    found = {}
    missing = []
    print("\n[VERIFIKASI VIDEO MENTAH]")
    for exercise, path in RAW_VIDEO_CANDIDATES.items():
        if path.is_file():
            print(f"  [OK] {exercise:<12}: {path.name}")
            found[exercise] = path
        else:
            print(f"  [MISSING] {exercise:<12}: {path}")
            missing.append((exercise, path))

    if missing:
        print("\n[ERROR] Video mentah tidak lengkap! Kandidat yang diperiksa:")
        for ex, p in RAW_VIDEO_CANDIDATES.items():
            print(f"  - {ex}: {p} (Exists: {p.is_file()})")
        sys.exit(1)

    return found


def measure_model_loading(model_path: Path, device: torch.device) -> Tuple[AttentiveSkel3DPerFrame, float]:
    """Mengukur cold-start latency pemuatan model AI."""
    t0 = time.perf_counter_ns()
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    use_sp = any(k.startswith("biomechanical_spatial_prior") for k in state_dict)
    use_ls = any(k.startswith("learned_spatial_attention")   for k in state_dict)
    use_ta = any(k.startswith("temporal_attention")           for k in state_dict)

    model = AttentiveSkel3DPerFrame(
        num_classes=2,
        use_spatial_prior=use_sp,
        use_learned_spatial=use_ls,
        use_temporal_attention=use_ta,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter_ns()

    load_time_ms = (t1 - t0) / 1e6
    return model, load_time_ms


def run_single_pipeline_benchmark(
    exercise: str,
    video_path: Path,
    model: AttentiveSkel3DPerFrame,
    device: torch.device,
    temp_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    """Menjalankan satu iterasi pipeline end-to-end dengan pengukuran waktu presisi tinggi per tahap."""
    stem = video_path.stem
    extractor = PoseExtractor(model_complexity=2)
    preprocessor = DataPreprocessor(target_frames=64)

    # ── Stage 1: Video Stream Open & Metadata Inspection ──────────────────────
    t0 = time.perf_counter_ns()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Tidak dapat membuka video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    cap.release()
    t1 = time.perf_counter_ns()
    t_video_stream_open_and_metadata_inspection_ms = (t1 - t0) / 1e6

    # ── Stage 2: Video Decoding & BlazePose Extraction ────────────────────────
    # save_output=False agar extractor tidak menulis .npy ke disk (hanya mengembalikan array)
    raw_npy_path = temp_dir / f"{stem}_{run_id}_raw.npy"
    t0 = time.perf_counter_ns()
    raw_array = extractor.extract_video(
        video_path=str(video_path),
        output_npy_path=str(raw_npy_path),
        save_output=False,
    )
    t1 = time.perf_counter_ns()
    t_video_decoding_and_blazepose_extraction_ms = (t1 - t0) / 1e6

    frames_with_pose = int(raw_array.shape[0])
    frames_without_pose = max(0, total_frames - frames_with_pose)

    # ── Stage 3: Raw Tensor Saving (Penulisan 1x per run) ─────────────────────
    t0 = time.perf_counter_ns()
    np.save(str(raw_npy_path), raw_array)
    t1 = time.perf_counter_ns()
    t_raw_tensor_saving_ms = (t1 - t0) / 1e6

    # ── Stage 4: Preprocessing (Resampling + Norm) ───────────────────────────
    tensor_64_path = temp_dir / f"{stem}_{run_id}_64.npy"
    t0 = time.perf_counter_ns()
    tensor_data = preprocessor.process(
        npy_file_path=str(raw_npy_path),
        output_npy_path=str(tensor_64_path),
    )
    input_tensor = torch.tensor(tensor_data, dtype=torch.float32).unsqueeze(0).to(device)
    t1 = time.perf_counter_ns()
    t_preprocessing_ms = (t1 - t0) / 1e6

    # ── Stage 5: Model Inference & Joint Attribution ──────────────────────────
    # Total model forward pass: 3x pada batch normal (1x forward_with_attention + 1x original + 1x batched B=33)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter_ns()

    attn_out = forward_with_attention(model, input_tensor)
    logits = attn_out["logits"]
    preds = logits.argmax(dim=2)
    probs = torch.softmax(logits, dim=2)
    influence, delta_prob = joint_influence(model, input_tensor, device)
    S = sequence_joint_score(influence)
    ras_res = reference_attribution_share(S, influence, exercise)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    t_model_inference_and_joint_attribution_ms = (t1 - t0) / 1e6

    preds_np = preds.squeeze().cpu().numpy()
    probs_np = probs.squeeze().cpu().numpy()

    # ── Time-to-Analysis-Ready ────────────────────────────────────────────────
    t_time_to_analysis_ready_ms = (
        t_video_stream_open_and_metadata_inspection_ms +
        t_video_decoding_and_blazepose_extraction_ms +
        t_raw_tensor_saving_ms +
        t_preprocessing_ms +
        t_model_inference_and_joint_attribution_ms
    )

    # ── Stage 6: Video Heatmap Rendering ──────────────────────────────────────
    out_rendered_video = temp_dir / f"{stem}_{run_id}_demo.mp4"
    t0 = time.perf_counter_ns()

    cap_rend = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_rendered_video), fourcc, fps, (width, height))

    target_indices = np.linspace(0, total_frames - 1, 64).astype(int)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
    ref_joints = BIOMECHANICAL_REFERENCE.get(exercise, [])

    for i in range(64):
        idx_asli = target_indices[i]
        cap_rend.set(cv2.CAP_PROP_POS_FRAMES, idx_asli)
        ret, frame = cap_rend.read()
        if not ret:
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        pred_label = preds_np[i]
        prob_salah = probs_np[i, 1] * 100
        inf_frame = influence[i]
        max_abs = float(np.abs(inf_frame).max()) or 1.0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for conn in mp_pose.POSE_CONNECTIONS:
                p1, p2 = conn
                lm1, lm2 = landmarks[p1], landmarks[p2]
                if lm1.visibility > 0.3 and lm2.visibility > 0.3:
                    cv2.line(frame, (int(lm1.x * width), int(lm1.y * height)),
                             (int(lm2.x * width), int(lm2.y * height)), (120, 120, 120), 2)

            for j, lm in enumerate(landmarks):
                if j >= 33 or lm.visibility < 0.3:
                    continue
                px = int(lm.x * width); py = int(lm.y * height)
                inf_val = float(inf_frame[j])
                norm_abs = abs(inf_val) / max_abs
                radius = 4 + int(norm_abs * 16)
                color = (30, 30, int(norm_abs * 200) + 55) if inf_val > 0.01 * max_abs else (
                    (int(norm_abs * 200) + 55, int(norm_abs * 100) + 27, 30) if inf_val < -0.01 * max_abs else (80, 80, 80)
                )
                cv2.circle(frame, (px, py), radius, color, -1)
                if j in ref_joints:
                    cv2.circle(frame, (px, py), radius + 4, (255, 255, 255), 1)

        writer.write(frame)

    cap_rend.release()
    writer.release()
    pose.close()

    t1 = time.perf_counter_ns()
    t_rendering_ms = (t1 - t0) / 1e6

    # ── Total Pipeline with Rendering ─────────────────────────────────────────
    t_total_with_rendering_ms = t_time_to_analysis_ready_ms + t_rendering_ms

    return {
        "exercise"                                         : exercise,
        "video_file"                                       : video_path.name,
        "resolution"                                       : f"{width}x{height}",
        "source_fps"                                       : round(fps, 2),
        "frame_count"                                      : total_frames,
        "video_duration_sec"                               : round(duration_sec, 3),
        "frames_with_pose"                                 : frames_with_pose,
        "frames_without_pose"                              : frames_without_pose,
        "run_id"                                           : run_id,
        "t_video_stream_open_and_metadata_inspection_ms"   : round(t_video_stream_open_and_metadata_inspection_ms, 3),
        "t_video_decoding_and_blazepose_extraction_ms"     : round(t_video_decoding_and_blazepose_extraction_ms, 3),
        "t_raw_tensor_saving_ms"                           : round(t_raw_tensor_saving_ms, 3),
        "t_preprocessing_ms"                               : round(t_preprocessing_ms, 3),
        "t_model_inference_and_joint_attribution_ms"       : round(t_model_inference_and_joint_attribution_ms, 3),
        "t_time_to_analysis_ready_ms"                      : round(t_time_to_analysis_ready_ms, 3),
        "t_rendering_ms"                                   : round(t_rendering_ms, 3),
        "t_total_with_rendering_ms"                        : round(t_total_with_rendering_ms, 3),
    }


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Menghitung mean, median, std, min, max, dan p95."""
    arr = np.array(values)
    return {
        "mean"  : round(float(np.mean(arr)), 3),
        "median": round(float(np.median(arr)), 3),
        "std"   : round(float(np.std(arr, ddof=1 if len(arr) > 1 else 0)), 3),
        "min"   : round(float(np.min(arr)), 3),
        "max"   : round(float(np.max(arr)), 3),
        "p95"   : round(float(np.percentile(arr, 95)), 3),
    }


def validate_time_sums(
    t_open: float, t_dec_pose: float, t_raw: float, t_prep: float, t_inf_attr: float,
    t_analysis_ready: float, t_rend: float, t_total: float,
    tolerance_pct: float = 1.0,
) -> Dict[str, Any]:
    """
    Validasi persamaan internal pipeline:
      1) time_to_analysis_ready ≈ video_stream_open + video_decoding_and_blazepose + raw_tensor_saving + preprocessing + model_inference_and_joint_attribution
      2) total_with_rendering ≈ time_to_analysis_ready + rendering
    """
    sum_analysis = t_open + t_dec_pose + t_raw + t_prep + t_inf_attr
    diff_analysis = abs(t_analysis_ready - sum_analysis)
    diff_analysis_pct = (diff_analysis / t_analysis_ready * 100) if t_analysis_ready > 0 else 0.0

    sum_total = t_analysis_ready + t_rend
    diff_total = abs(t_total - sum_total)
    diff_total_pct = (diff_total / t_total * 100) if t_total > 0 else 0.0

    valid_analysis = diff_analysis_pct <= tolerance_pct
    valid_total    = diff_total_pct <= tolerance_pct
    overall_valid  = valid_analysis and valid_total

    return {
        "overall_valid"                  : overall_valid,
        "analysis_sum_calculated_ms"     : round(sum_analysis, 3),
        "analysis_diff_ms"               : round(diff_analysis, 3),
        "analysis_diff_pct"              : round(diff_analysis_pct, 4),
        "analysis_valid"                 : valid_analysis,
        "total_sum_calculated_ms"        : round(sum_total, 3),
        "total_diff_ms"                  : round(diff_total, 3),
        "total_diff_pct"                 : round(diff_total_pct, 4),
        "total_valid"                    : valid_total,
        "max_tolerance_pct"              : tolerance_pct,
    }


def run_benchmark(smoke_test: bool = False):
    n_measured = 1 if smoke_test else N_MEASURED_RUNS

    print("=" * 95)
    print("  BENCHMARK LATENCY END-TO-END — ATTENTIVESKEL-3D V2 (SINGLE TENSOR SAVE PER RUN)")
    print("=" * 95)
    print(f"  Model            : S3b BSP + Learned Spatial ({MODEL_REL_PATH.name})")
    print(f"  Measured Runs    : {n_measured} per video (+ 1 warm-up)")
    print(f"  Smoke Test Mode  : {smoke_test}")

    # 1. Verifikasi video mentah
    raw_videos = verify_raw_videos()

    # 2. Device & Cold-start model loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = PROJECT_ROOT / MODEL_REL_PATH

    print(f"\n[COLD-START MODEL LOADING] Device: {device}")
    model, load_time_ms = measure_model_loading(model_path, device)
    print(f"  [OK] Model loaded in {load_time_ms:.2f} ms")

    per_run_results = []
    summary_results = []
    video_summary_dict = {}

    print(f"\n[STEADY-STATE RUNS] (1 warm-up, {n_measured} measured runs per video)")

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        for exercise, video_path in raw_videos.items():
            print(f"\n{'-'*95}")
            print(f"  Latihan : {exercise}")
            print(f"  Video   : {video_path.name}")
            print(f"{'-'*95}")

            # Warm-up
            print("  [Warm-up] Running 1 warm-up run...")
            warmup_res = run_single_pipeline_benchmark(
                exercise, video_path, model, device, temp_dir, run_id="warmup"
            )
            warmup_res["run_type"] = "warmup"
            warmup_res["cold_start_load_ms"] = round(load_time_ms, 3)
            warmup_res["n_measured_runs"] = n_measured
            per_run_results.append(warmup_res)

            # Measured runs
            measured_runs = []
            for i in range(1, n_measured + 1):
                print(f"  [Run {i:>2}/{n_measured}] Measuring full pipeline...")
                res = run_single_pipeline_benchmark(
                    exercise, video_path, model, device, temp_dir, run_id=f"run_{i}"
                )
                res["run_type"] = f"measured_{i}"
                res["cold_start_load_ms"] = round(load_time_ms, 3)
                res["n_measured_runs"] = n_measured
                per_run_results.append(res)
                measured_runs.append(res)

            # Ekstraksi list tiap tahap dari measured runs
            t_open_list     = [r["t_video_stream_open_and_metadata_inspection_ms"] for r in measured_runs]
            t_dec_pose_list = [r["t_video_decoding_and_blazepose_extraction_ms"] for r in measured_runs]
            t_raw_list      = [r["t_raw_tensor_saving_ms"] for r in measured_runs]
            t_prep_list     = [r["t_preprocessing_ms"] for r in measured_runs]
            t_inf_attr_list = [r["t_model_inference_and_joint_attribution_ms"] for r in measured_runs]
            t_analysis_list = [r["t_time_to_analysis_ready_ms"] for r in measured_runs]
            t_rend_list     = [r["t_rendering_ms"] for r in measured_runs]
            t_tot_list      = [r["t_total_with_rendering_ms"] for r in measured_runs]

            stats_open     = compute_statistics(t_open_list)
            stats_dec_pose = compute_statistics(t_dec_pose_list)
            stats_raw      = compute_statistics(t_raw_list)
            stats_prep     = compute_statistics(t_prep_list)
            stats_inf_attr = compute_statistics(t_inf_attr_list)
            stats_analysis = compute_statistics(t_analysis_list)
            stats_rend     = compute_statistics(t_rend_list)
            stats_tot      = compute_statistics(t_tot_list)

            v_meta   = measured_runs[0]
            dur_sec  = v_meta["video_duration_sec"]
            n_frames = v_meta["frame_count"]

            eff_fps_pipeline = round(n_frames / (stats_analysis["mean"] / 1000.0), 2)
            eff_fps_total    = round(n_frames / (stats_tot["mean"] / 1000.0), 2)

            rtf_analysis = round((stats_analysis["mean"] / 1000.0) / dur_sec, 3)
            rtf_tot      = round((stats_tot["mean"] / 1000.0) / dur_sec, 3)

            seq_capture_plus_processing_s = round(dur_sec + (stats_analysis["mean"] / 1000.0), 3)

            # Kontribusi persentase terhadap time_to_analysis_ready
            analysis_m = stats_analysis["mean"]
            pct_open_of_analysis     = round((stats_open["mean"]     / analysis_m) * 100, 2)
            pct_dec_pose_of_analysis = round((stats_dec_pose["mean"] / analysis_m) * 100, 2)
            pct_raw_of_analysis      = round((stats_raw["mean"]      / analysis_m) * 100, 2)
            pct_prep_of_analysis     = round((stats_prep["mean"]     / analysis_m) * 100, 2)
            pct_inf_attr_of_analysis = round((stats_inf_attr["mean"] / analysis_m) * 100, 2)

            # Kontribusi persentase terhadap total_with_rendering
            tot_m = stats_tot["mean"]
            pct_analysis_of_tot = round((analysis_m / tot_m) * 100, 2)
            pct_rend_of_tot     = round((stats_rend["mean"] / tot_m) * 100, 2)

            # Validasi penjumlahan waktu
            validation = validate_time_sums(
                stats_open["mean"], stats_dec_pose["mean"], stats_raw["mean"],
                stats_prep["mean"], stats_inf_attr["mean"], stats_analysis["mean"],
                stats_rend["mean"], stats_tot["mean"],
                tolerance_pct=1.0,
            )

            video_summary = {
                "exercise"                                         : exercise,
                "video_file"                                       : v_meta["video_file"],
                "resolution"                                       : v_meta["resolution"],
                "source_fps"                                       : v_meta["source_fps"],
                "frame_count"                                      : n_frames,
                "video_duration_sec"                               : dur_sec,
                "frames_with_pose"                                 : v_meta["frames_with_pose"],
                "frames_without_pose"                              : v_meta["frames_without_pose"],
                "n_measured_runs"                                  : n_measured,

                # Detailed statistics per stage
                "video_stream_open_and_metadata_inspection_ms"     : stats_open,
                "video_decoding_and_blazepose_extraction_ms"       : stats_dec_pose,
                "raw_tensor_saving_ms"                             : stats_raw,
                "preprocessing_ms"                                 : stats_prep,
                "model_inference_and_joint_attribution_ms"         : stats_inf_attr,
                "time_to_analysis_ready_ms"                        : stats_analysis,
                "rendering_ms"                                     : stats_rend,
                "total_with_rendering_ms"                          : stats_tot,

                # Derived performance metrics
                "effective_pipeline_fps"                           : eff_fps_pipeline,
                "effective_total_fps"                              : eff_fps_total,
                "rtf_analysis_ready"                               : rtf_analysis,
                "real_time_factor_total"                           : rtf_tot,
                "sequential_capture_plus_processing_s"             : seq_capture_plus_processing_s,

                # Stage percentage contribution
                "pct_of_time_to_analysis_ready": {
                    "video_stream_open_and_metadata_inspection": pct_open_of_analysis,
                    "video_decoding_and_blazepose_extraction"  : pct_dec_pose_of_analysis,
                    "raw_tensor_saving"                        : pct_raw_of_analysis,
                    "preprocessing"                            : pct_prep_of_analysis,
                    "model_inference_and_joint_attribution"    : pct_inf_attr_of_analysis,
                },
                "pct_of_total_with_rendering": {
                    "time_to_analysis_ready": pct_analysis_of_tot,
                    "rendering"             : pct_rend_of_tot,
                },

                # Validation status
                "internal_sum_validation": validation,
            }
            summary_results.append(video_summary)
            video_summary_dict[exercise] = video_summary

            val_mark = "[PASSED]" if validation["overall_valid"] else "[WARN]"
            print(f"  [OK] Summary {exercise} (Validation: {val_mark}):")
            print(f"    - Time-to-Analysis-Ready : {stats_analysis['mean']:.2f} ms (p95: {stats_analysis['p95']:.2f} ms, std: {stats_analysis['std']:.2f} ms)")
            print(f"    - Total w/ Rendering     : {stats_tot['mean']:.2f} ms (p95: {stats_tot['p95']:.2f} ms, std: {stats_tot['std']:.2f} ms)")
            print(f"    - Real-Time Factor       : {rtf_analysis:.3f}x (Analysis) | {rtf_tot:.3f}x (Total)")
            print(f"    - Effective Speed        : {eff_fps_pipeline:.1f} FPS (Pipeline) | {eff_fps_total:.1f} FPS (Total)")
            print(f"    - Seq Capture+Proc       : {seq_capture_plus_processing_s:.2f} s")

    # Overall Aggregate Summary across 3 videos
    all_analysis_means = [s["time_to_analysis_ready_ms"]["mean"] for s in summary_results]
    all_tot_means      = [s["total_with_rendering_ms"]["mean"] for s in summary_results]
    overall_summary = {
        "n_measured_runs_per_video"       : n_measured,
        "mean_time_to_analysis_ready_ms"  : compute_statistics(all_analysis_means),
        "mean_total_with_rendering_ms"    : compute_statistics(all_tot_means),
        "cold_start_model_load_ms"        : round(load_time_ms, 3),
        "device"                          : str(device),
    }

    # ── Simpan Hasil ─────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_run_csv = OUTPUT_DIR / "Latency_EndToEnd_PerRun_AttentiveSkel3D_V2.csv"
    summary_csv = OUTPUT_DIR / "Latency_EndToEnd_Summary_AttentiveSkel3D_V2.csv"
    json_path   = OUTPUT_DIR / "Latency_EndToEnd_AttentiveSkel3D_V2.json"

    # CSV Per-Run
    if per_run_results:
        with open(per_run_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=per_run_results[0].keys())
            writer.writeheader()
            writer.writerows(per_run_results)
        print(f"\n[SAVED] CSV Per-Run  : {per_run_csv}")

    # CSV Summary
    summary_rows = []
    for s in summary_results:
        summary_rows.append({
            "Exercise"                             : s["exercise"],
            "Video_File"                           : s["video_file"],
            "Resolution"                           : s["resolution"],
            "Duration_Sec"                         : s["video_duration_sec"],
            "Frame_Count"                          : s["frame_count"],
            "N_Measured_Runs"                      : s["n_measured_runs"],

            # Time-to-Analysis-Ready Stats (ms)
            "Analysis_Ready_Mean_MS"               : s["time_to_analysis_ready_ms"]["mean"],
            "Analysis_Ready_Median_MS"             : s["time_to_analysis_ready_ms"]["median"],
            "Analysis_Ready_Std_MS"                : s["time_to_analysis_ready_ms"]["std"],
            "Analysis_Ready_Min_MS"                : s["time_to_analysis_ready_ms"]["min"],
            "Analysis_Ready_Max_MS"                : s["time_to_analysis_ready_ms"]["max"],
            "Analysis_Ready_P95_MS"                : s["time_to_analysis_ready_ms"]["p95"],

            # Total with Rendering Stats (ms)
            "Total_Mean_MS"                        : s["total_with_rendering_ms"]["mean"],
            "Total_Median_MS"                      : s["total_with_rendering_ms"]["median"],
            "Total_Std_MS"                         : s["total_with_rendering_ms"]["std"],
            "Total_Min_MS"                         : s["total_with_rendering_ms"]["min"],
            "Total_Max_MS"                         : s["total_with_rendering_ms"]["max"],
            "Total_P95_MS"                         : s["total_with_rendering_ms"]["p95"],

            # Stage Means (ms)
            "StreamOpen_Inspection_Mean_MS"        : s["video_stream_open_and_metadata_inspection_ms"]["mean"],
            "Dec_BlazePose_Mean_MS"                : s["video_decoding_and_blazepose_extraction_ms"]["mean"],
            "RawSave_Mean_MS"                      : s["raw_tensor_saving_ms"]["mean"],
            "Prep_Mean_MS"                         : s["preprocessing_ms"]["mean"],
            "Inference_Attribution_Mean_MS"        : s["model_inference_and_joint_attribution_ms"]["mean"],
            "Rendering_Mean_MS"                    : s["rendering_ms"]["mean"],

            # Stage Pct Contribution to Analysis-Ready
            "Pct_StreamOpen_of_Analysis"           : s["pct_of_time_to_analysis_ready"]["video_stream_open_and_metadata_inspection"],
            "Pct_Dec_BlazePose_of_Analysis"        : s["pct_of_time_to_analysis_ready"]["video_decoding_and_blazepose_extraction"],
            "Pct_RawSave_of_Analysis"              : s["pct_of_time_to_analysis_ready"]["raw_tensor_saving"],
            "Pct_Prep_of_Analysis"                 : s["pct_of_time_to_analysis_ready"]["preprocessing"],
            "Pct_Inference_Attribution_of_Analysis": s["pct_of_time_to_analysis_ready"]["model_inference_and_joint_attribution"],

            # Derived Metrics
            "Effective_Pipeline_FPS"               : s["effective_pipeline_fps"],
            "Effective_Total_FPS"                  : s["effective_total_fps"],
            "RTF_Analysis_Ready"                   : s["rtf_analysis_ready"],
            "RTF_Total"                            : s["real_time_factor_total"],
            "Sequential_Capture_Plus_Processing_S" : s["sequential_capture_plus_processing_s"],
            "Sum_Validation_Valid"                 : s["internal_sum_validation"]["overall_valid"],
        })
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[SAVED] CSV Summary : {summary_csv}")

    # JSON Full Metadata
    full_json_data = {
        "benchmark_name"     : "End-to-End Latency Benchmark AttentiveSkel-3D V2 (Refined — Single Save)",
        "practical_model"    : {
            "name"      : "S3b BSP + Learned Spatial",
            "checkpoint": MODEL_REL_PATH.name,
            "path"      : MODEL_REL_PATH.as_posix(),
        },
        "device"             : str(device),
        "n_measured_runs"    : n_measured,
        "cold_start_model_load_ms": round(load_time_ms, 3),
        "disclaimer_metadata": DISCLAIMER_METADATA,
        "per_video_summary"  : video_summary_dict,
        "overall_summary"    : overall_summary,
        "per_run_data"       : per_run_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json_data, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] JSON Metadata: {json_path}")

    # ── Cetak Tabel Ringkasan Final & Stage Breakdown ────────────────────────
    print("\n" + "=" * 95)
    print(f"  RINGKASAN LATENCY END-TO-END ({n_measured} Measured Runs per Video)")
    print("=" * 95)
    print(f" Cold-Start Model Load: {load_time_ms:.2f} ms ({device})\n")

    print(f"{'Latihan':<12} {'Res':<10} {'Dur(s)':<7} {'Analysis Mean':<14} {'Analysis P95':<13} {'Eff. FPS':<10} {'RTF Analysis':<13} {'Seq. Cap+Proc':<14}")
    print("-" * 95)
    for s in summary_results:
        print(
            f"{s['exercise']:<12} "
            f"{s['resolution']:<10} "
            f"{s['video_duration_sec']:<7.2f} "
            f"{s['time_to_analysis_ready_ms']['mean']:<14.2f} "
            f"{s['time_to_analysis_ready_ms']['p95']:<13.2f} "
            f"{s['effective_pipeline_fps']:<10.1f} "
            f"{s['rtf_analysis_ready']:<13.3f} "
            f"{s['sequential_capture_plus_processing_s']:<14.2f}s"
        )
    print("-" * 95)

    print("\n" + "=" * 95)
    print("  TAHAP DEMI TAHAP (STAGE BREAKDOWN & KONTRIBUSI PERSENTASE)")
    print("=" * 95)
    for s in summary_results:
        analysis_m = s["time_to_analysis_ready_ms"]["mean"]
        tot_m      = s["total_with_rendering_ms"]["mean"]
        pcts       = s["pct_of_time_to_analysis_ready"]
        valid      = "[PASSED]" if s["internal_sum_validation"]["overall_valid"] else "[WARN]"

        print(f"\n> Latihan: {s['exercise']} ({s['video_file']}) | Validasi Penjumlahan: {valid}")
        print(f"  {'Tahap Pipeline':<45} {'Mean (ms)':<12} {'Std (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'P95 (ms)':<10} {'% Analysis':<10}")
        print(f"  {'-'*105}")

        stage_items = [
            ("1. Video Stream Open & Metadata Inspection", s["video_stream_open_and_metadata_inspection_ms"], pcts["video_stream_open_and_metadata_inspection"]),
            ("2. Video Decoding & BlazePose Extraction",   s["video_decoding_and_blazepose_extraction_ms"],   pcts["video_decoding_and_blazepose_extraction"]),
            ("3. Raw Tensor Saving (Single Write)",         s["raw_tensor_saving_ms"],                         pcts["raw_tensor_saving"]),
            ("4. Preprocessing",                           s["preprocessing_ms"],                             pcts["preprocessing"]),
            ("5. Model Inference & Joint Attribution",     s["model_inference_and_joint_attribution_ms"],     pcts["model_inference_and_joint_attribution"]),
        ]
        for name, st, pct in stage_items:
            print(f"  {name:<45} {st['mean']:<12.2f} {st['std']:<10.2f} {st['min']:<10.2f} {st['max']:<10.2f} {st['p95']:<10.2f} {pct:>8.2f}%")

        print(f"  {'-'*105}")
        print(f"  {'TIME-TO-ANALYSIS-READY':<45} {analysis_m:<12.2f} {s['time_to_analysis_ready_ms']['std']:<10.2f} {s['time_to_analysis_ready_ms']['min']:<10.2f} {s['time_to_analysis_ready_ms']['max']:<10.2f} {s['time_to_analysis_ready_ms']['p95']:<10.2f}  100.00%")
        print(f"  {'6. Video Heatmap Rendering':<45} {s['rendering_ms']['mean']:<12.2f} {s['rendering_ms']['std']:<10.2f} {s['rendering_ms']['min']:<10.2f} {s['rendering_ms']['max']:<10.2f} {s['rendering_ms']['p95']:<10.2f}  ({s['pct_of_total_with_rendering']['rendering']:.1f}% tot)")
        print(f"  {'TOTAL WITH RENDERING':<45} {tot_m:<12.2f} {s['total_with_rendering_ms']['std']:<10.2f} {s['total_with_rendering_ms']['min']:<10.2f} {s['total_with_rendering_ms']['max']:<10.2f} {s['total_with_rendering_ms']['p95']:<10.2f} ---")

    print("\n" + "=" * 95)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Latency Benchmark AttentiveSkel-3D V2")
    parser.add_argument("--smoke-test", action="store_true", help="Jalankan 1 warm-up + 1 measured run untuk pengujian cepat.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(smoke_test=args.smoke_test)
