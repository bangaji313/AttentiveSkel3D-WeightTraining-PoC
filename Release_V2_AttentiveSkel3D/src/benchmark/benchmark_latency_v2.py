"""Benchmark model-only latency for all AttentiveSkel-3D V2 scenarios.

Place this file at:
    Release_V2_AttentiveSkel3D/src/benchmark/benchmark_latency_v2.py

Run from the Release_V2_AttentiveSkel3D directory:
    python src/benchmark/benchmark_latency_v2.py

The benchmark measures a synchronous forward pass from a preprocessed tensor
with shape (1, 64, 33, 3) to logits with shape (1, 64, 2). Pose extraction,
video decoding, preprocessing, rendering, and the 64-frame acquisition window
are intentionally excluded and must be measured separately as end-to-end
pipeline latency.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.arsitektur_v2 import (  # noqa: E402
    AttentiveSkel3DPerFrame,
    count_parameters,
)


SCENARIOS = {
    "S1 Baseline": "best_model_baseline.pth",
    "S2 Full Model": "best_model_v2.pth",
    "S3a BSP": "best_model_s3a_bsp_holdout_v2.pth",
    # Nama checkpoint b/c mengikuti notebook ablasi awal:
    #   ablasi_b = tanpa Learned Spatial -> BSP + Temporal
    #   ablasi_c = tanpa Temporal        -> BSP + Learned Spatial
    "S3b BSP+Learned Spatial": "best_model_ablasi_c.pth",
    "S3c BSP+Temporal": "best_model_ablasi_b.pth",
}

EXPECTED_MODULES = {
    "S1 Baseline": {"BSP": False, "LearnedSpatial": False, "Temporal": False},
    "S2 Full Model": {"BSP": True, "LearnedSpatial": True, "Temporal": True},
    "S3a BSP": {"BSP": True, "LearnedSpatial": False, "Temporal": False},
    "S3b BSP+Learned Spatial": {
        "BSP": True,
        "LearnedSpatial": True,
        "Temporal": False,
    },
    "S3c BSP+Temporal": {
        "BSP": True,
        "LearnedSpatial": False,
        "Temporal": True,
    },
}

MODELS_DIR = PROJECT_ROOT / "bobot_model"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "hasil_evaluasi"
INPUT_SHAPE = (1, 64, 33, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark model-only latency AttentiveSkel-3D V2."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Jumlah iterasi warm-up per model dan device (default: 50).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Jumlah iterasi terukur per model dan device (default: 500).",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=("cpu", "cuda"),
        default=("cpu", "cuda"),
        help="Device yang diuji (default: cpu cuda).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="Jumlah thread CPU PyTorch. Nilai 0 memakai konfigurasi aktif.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder keluaran CSV dan JSON.",
    )
    return parser.parse_args()


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[AttentiveSkel3DPerFrame, dict[str, bool]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    use_bsp = any(
        key.startswith("biomechanical_spatial_prior") for key in state_dict
    )
    use_learned = any(
        key.startswith("learned_spatial_attention") for key in state_dict
    )
    use_temporal = any(
        key.startswith("temporal_attention") for key in state_dict
    )

    model = AttentiveSkel3DPerFrame(
        num_classes=2,
        use_spatial_prior=use_bsp,
        use_learned_spatial=use_learned,
        use_temporal_attention=use_temporal,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    modules = {
        "BSP": use_bsp,
        "LearnedSpatial": use_learned,
        "Temporal": use_temporal,
    }
    return model, modules


def latency_statistics(latencies_ms: list[float]) -> dict[str, float]:
    values = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(values.mean())
    return {
        "mean_ms": mean_ms,
        "median_ms": float(np.median(values)),
        "std_ms": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "sequences_per_second": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }


def benchmark_model(
    model: AttentiveSkel3DPerFrame,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float], tuple[int, ...], float]:
    input_tensor = torch.randn(INPUT_SHAPE, dtype=torch.float32, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    output = None
    with torch.inference_mode():
        for _ in range(warmup):
            output = model(input_tensor)
        synchronize(device)

        latencies_ms: list[float] = []
        for _ in range(iterations):
            # Synchronisation makes this a blocking wall-clock forward latency.
            synchronize(device)
            start_ns = time.perf_counter_ns()
            output = model(input_tensor)
            synchronize(device)
            end_ns = time.perf_counter_ns()
            latencies_ms.append((end_ns - start_ns) / 1_000_000.0)

    if output is None:
        raise RuntimeError("Benchmark tidak menghasilkan output model.")

    output_shape = tuple(int(value) for value in output.shape)
    if output_shape != (1, 64, 2):
        raise RuntimeError(
            f"Output model tidak sesuai. Diperoleh {output_shape}, "
            "diharapkan (1, 64, 2)."
        )

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    return latency_statistics(latencies_ms), output_shape, peak_memory_mb


def hardware_metadata(args: argparse.Namespace) -> dict[str, Any]:
    gpu_name = None
    gpu_total_memory_mb = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu_name = properties.name
        gpu_total_memory_mb = properties.total_memory / (1024**2)

    return {
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "benchmark_scope": "model-only synchronous forward pass",
        "excluded_from_measurement": [
            "video decoding",
            "64-frame acquisition or buffering",
            "MediaPipe BlazePose extraction",
            "preprocessing and temporal resampling",
            "visualisation and rendering",
            "network and web application overhead",
        ],
        "input_shape": list(INPUT_SHAPE),
        "expected_output_shape": [1, 64, 2],
        "batch_size": 1,
        "precision": "FP32",
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
        "operating_system": platform.platform(),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor(),
        "cpu_threads_pytorch": torch.get_num_threads(),
        "gpu": gpu_name,
        "gpu_total_memory_mb": gpu_total_memory_mb,
    }


def main() -> None:
    args = parse_args()

    if args.warmup < 1:
        raise ValueError("--warmup minimal 1.")
    if args.iterations < 2:
        raise ValueError("--iterations minimal 2.")
    if args.cpu_threads < 0:
        raise ValueError("--cpu-threads tidak boleh negatif.")
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    seed_everything(42)
    torch.backends.cudnn.benchmark = True

    requested_devices = list(dict.fromkeys(args.devices))
    if "cuda" in requested_devices and not torch.cuda.is_available():
        print("[PERINGATAN] CUDA tidak tersedia; pengujian CUDA dilewati.")
        requested_devices.remove("cuda")
    if not requested_devices:
        raise RuntimeError("Tidak ada device yang dapat diuji.")

    missing_checkpoints = [
        str(MODELS_DIR / filename)
        for filename in SCENARIOS.values()
        if not (MODELS_DIR / filename).exists()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(
            "Checkpoint berikut tidak ditemukan:\n- "
            + "\n- ".join(missing_checkpoints)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = hardware_metadata(args)
    rows: list[dict[str, Any]] = []

    print("=" * 79)
    print("MODEL-ONLY LATENCY BENCHMARK - ATTENTIVESKEL-3D V2")
    print("=" * 79)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input        : {INPUT_SHAPE} FP32")
    print(f"Warm-up      : {args.warmup}")
    print(f"Iterations   : {args.iterations}")
    print(f"Devices      : {', '.join(requested_devices)}")
    print()

    for device_name in requested_devices:
        device = torch.device(device_name)
        print(f"[DEVICE] {device}")

        for scenario_name, checkpoint_filename in SCENARIOS.items():
            checkpoint_path = MODELS_DIR / checkpoint_filename
            model, modules = load_model(checkpoint_path, device)
            expected_modules = EXPECTED_MODULES[scenario_name]
            if modules != expected_modules:
                raise RuntimeError(
                    f"Mapping checkpoint salah untuk {scenario_name}. "
                    f"Terdeteksi {modules}, seharusnya {expected_modules}."
                )
            parameter_count = count_parameters(model)

            stats, output_shape, peak_memory_mb = benchmark_model(
                model=model,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
            )

            row: dict[str, Any] = {
                "scenario": scenario_name,
                "device": device_name,
                "checkpoint": checkpoint_filename,
                "parameters": parameter_count,
                "checkpoint_size_mb": checkpoint_path.stat().st_size / (1024**2),
                "bsp_active": modules["BSP"],
                "learned_spatial_active": modules["LearnedSpatial"],
                "temporal_active": modules["Temporal"],
                "input_shape": str(INPUT_SHAPE),
                "output_shape": str(output_shape),
                "warmup_iterations": args.warmup,
                "measured_iterations": args.iterations,
                "peak_cuda_memory_mb": peak_memory_mb,
                **stats,
            }
            rows.append(row)

            print(
                f"  {scenario_name:<27} "
                f"mean={stats['mean_ms']:.4f} ms | "
                f"median={stats['median_ms']:.4f} ms | "
                f"p95={stats['p95_ms']:.4f} ms | "
                f"{stats['sequences_per_second']:.2f} sequence/s"
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print()

    dataframe = pd.DataFrame(rows)
    csv_path = args.output_dir / "Latency_ModelOnly_AttentiveSkel3D_V2.csv"
    json_path = args.output_dir / "Latency_ModelOnly_AttentiveSkel3D_V2.json"

    dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"metadata": metadata, "results": rows},
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 79)
    print("BENCHMARK SELESAI")
    print(f"CSV  : {csv_path}")
    print(f"JSON : {json_path}")
    print("=" * 79)
    print(
        "Catatan: hasil ini hanya model-only latency. Jangan menyebutnya "
        "sebagai end-to-end real-time latency."
    )


if __name__ == "__main__":
    main()
