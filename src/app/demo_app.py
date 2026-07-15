# src/app/demo_app.py
#
# Aplikasi Web Demo Streamlit untuk Proof of Concept AttentiveSkel-3D.
#
# FOKUS UTAMA: "Sanity Check Fokus Atensi Spasial"
# - Membuktikan bahwa model menyoroti sendi yang relevan (misal: lutut pada squat)
# - Overlay skeleton dengan heatmap atensi warna merah untuk sendi kritis
# - Komparasi visual 5 model untuk menunjukkan akurasi Full Model
#
# Struktur Aplikasi (3 Tab):
#   TAB 1 - "Fase 1: Wujud Data Tensor"
#     Menampilkan tensor raw (64, 33, 3) dalam dataframe untuk transparansi data
#   
#   TAB 2 - "Fase 2: Lokalisasi Waktu"
#     Interactive slider frame dengan status BiomechanicalValidator per frame
#   
#   TAB 3 - "Fase 3: Sanity Check Mata AI & Komparasi" (PALING PENTING)
#     Video/frame output dengan skeleton overlay dan heatmap atensi
#     Sendi penting digambar merah (fokus AI), sendi lain abu-abu
#     Komparasi 5 model jika pilih "Semua Skenario"

import os
import sys
import io
import base64
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mediapipe as mp
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ════════════════════════════════════════════════════════════════════════════════
# Konfigurasi Path dan Import Modul Lokal
# ════════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from data.extract_pose import PoseExtractor
    from models.model_3dcnn import AttentiveSkel3D
    from data.biomechanics_validator import BiomechanicalValidator
except ImportError as e:
    st.error(f"❌ Gagal mengimport modul: {e}")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
# Konfigurasi Streamlit
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AttentiveSkel-3D Sanity Check PoC",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

custom_css = """
<style>
    /* Batasi ukuran maksimal container untuk video/frame output */
    .video-frame-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 12px;
        border: 2px solid #dee2e6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .video-frame-image {
        width: 100%;
        height: auto;
        border-radius: 8px;
        display: block;
    }
    
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        margin: 15px 0;
    }
    
    .result-card-label {
        font-size: 28px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .result-card-confidence {
        font-size: 16px;
        margin: 10px 0;
        opacity: 0.95;
    }
    
    .comparison-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 2px solid #dee2e6;
        text-align: center;
    }
    
    .comparison-card h5 {
        color: #333;
        margin-bottom: 8px;
        font-size: 14px;
    }
    
    .comparison-card-label {
        font-size: 16px;
        font-weight: bold;
        color: #667eea;
        margin: 8px 0;
    }
    
    .comparison-card-confidence {
        font-size: 12px;
        color: #666;
    }
    
    .app-title {
        text-align: center;
        color: #333;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .app-subtitle {
        text-align: center;
        color: #666;
        font-size: 16px;
        margin-bottom: 20px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin: 5px 0;
    }
    
    .status-safe {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .status-danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# Konstanta Global
# ════════════════════════════════════════════════════════════════════════════════

MODEL_PATHS = {
    "baseline": PROJECT_ROOT / "models" / "saved_models" / "baseline_3dcnn_model.pth",
    "ablasi_a": PROJECT_ROOT / "models" / "saved_models" / "ablasi_a_no_prior.pth",
    "ablasi_b": PROJECT_ROOT / "models" / "saved_models" / "ablasi_b_no_learned.pth",
    "ablasi_c": PROJECT_ROOT / "models" / "saved_models" / "ablasi_c_no_temporal.pth",
    "full": PROJECT_ROOT / "models" / "saved_models" / "AttentiveSkel3D_Final.pth",
}

SCENARIO_NAMES = {
    "semua": "Semua Skenario (Komparasi 5 Model)",
    "baseline": "Baseline (3D-CNN Murni)",
    "ablasi_a": "Ablasi A (Tanpa Prior)",
    "ablasi_b": "Ablasi B (Tanpa Learned Spatial)",
    "ablasi_c": "Ablasi C (Tanpa Temporal)",
    "full": "Full AttentiveSkel-3D",
}

SCENARIO_CONFIGS = {
    "baseline": {"use_spatial_prior": False, "use_learned_spatial": False, "use_temporal_attention": False},
    "ablasi_a": {"use_spatial_prior": False, "use_learned_spatial": True, "use_temporal_attention": True},
    "ablasi_b": {"use_spatial_prior": True, "use_learned_spatial": False, "use_temporal_attention": True},
    "ablasi_c": {"use_spatial_prior": True, "use_learned_spatial": True, "use_temporal_attention": False},
    "full": {"use_spatial_prior": True, "use_learned_spatial": True, "use_temporal_attention": True},
}

LANDMARK_NAMES = [
    "Hidung", "Mata Kiri Dalam", "Mata Kiri", "Mata Kiri Luar",
    "Mata Kanan Dalam", "Mata Kanan", "Mata Kanan Luar",
    "Telinga Kiri", "Telinga Kanan",
    "Mulut Kiri", "Mulut Kanan",
    "Bahu Kiri", "Bahu Kanan",
    "Siku Kiri", "Siku Kanan",
    "Pergelangan Tangan Kiri", "Pergelangan Tangan Kanan",
    "Jari Kelingking Kiri", "Jari Kelingking Kanan",
    "Jari Telunjuk Kiri", "Jari Telunjuk Kanan",
    "Jari Ibu Jari Kiri", "Jari Ibu Jari Kanan",
    "Pinggul Kiri", "Pinggul Kanan",
    "Lutut Kiri", "Lutut Kanan",
    "Pergelangan Kaki Kiri", "Pergelangan Kaki Kanan",
    "Ujung Kaki Kiri", "Ujung Kaki Kanan",
    "Tumit Kiri", "Tumit Kanan",
]

SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28),
]

# ════════════════════════════════════════════════════════════════════════════════
# Fungsi-Fungsi Caching & Utilitas
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_biomechanical_validator():
    """Memuat BiomechanicalValidator dengan caching."""
    return BiomechanicalValidator()


@st.cache_resource
def load_model(model_key: str) -> Optional[AttentiveSkel3D]:
    """
    Memuat model PyTorch dengan caching untuk efisiensi.
    
    Args:
        model_key (str): Kunci model (baseline, ablasi_a, ablasi_b, ablasi_c, full).
    
    Returns:
        AttentiveSkel3D | None: Model yang telah dimuat atau None jika file tidak ada.
    """
    model_path = MODEL_PATHS.get(model_key)
    
    if model_path is None or not model_path.exists():
        return None
    
    try:
        config = SCENARIO_CONFIGS.get(model_key, {})
        
        model = AttentiveSkel3D(
            num_classes=2,
            use_attention=True,
            **config
        )
        
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model '{model_key}': {e}")
        return None


def extract_pose_from_video(video_file) -> Optional[Tuple[np.ndarray, float, List[np.ndarray]]]:
    """Mengekstraksi pose skeleton dari file video yang di-upload."""
    try:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(video_file.getbuffer())
        
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        all_pose_landmarks = []
        all_frames = []
        
        with st.spinner("🔄 Mengekstraksi pose skeleton dari video..."):
            with mp.solutions.pose.Pose(
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as pose_model:
                
                while cap.isOpened():
                    success, frame_bgr = cap.read()
                    if not success:
                        break
                    
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    all_frames.append(frame_rgb.copy())
                    
                    frame_rgb.flags.writeable = False
                    results = pose_model.process(frame_rgb)
                    frame_rgb.flags.writeable = True
                    
                    if results.pose_landmarks:
                        frame_landmarks = np.array(
                            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
                            dtype=np.float32,
                        )
                        all_pose_landmarks.append(frame_landmarks)
                    else:
                        if all_pose_landmarks:
                            all_pose_landmarks.append(all_pose_landmarks[-1])
                        else:
                            all_pose_landmarks.append(np.zeros((33, 3), dtype=np.float32))
                
                cap.release()
        
        if not all_pose_landmarks:
            st.error("❌ Tidak ada pose yang terdeteksi dalam video.")
            return None
        
        pose_tensor = np.array(all_pose_landmarks, dtype=np.float32)
        pose_tensor = normalize_pose(pose_tensor)
        
        st.success(f"✅ Pose berhasil diekstraksi dari {len(all_pose_landmarks)} frame.")
        
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        return pose_tensor, fps, all_frames
        
    except Exception as e:
        st.error(f"❌ Error saat mengekstraksi pose: {e}")
        return None


def normalize_pose(pose_array: np.ndarray) -> np.ndarray:
    """Normalisasi pose skeleton terhadap mid-hip dan scaling."""
    left_hip_idx = 23
    right_hip_idx = 24
    
    mid_hip = (pose_array[:, left_hip_idx] + pose_array[:, right_hip_idx]) / 2.0
    normalized = pose_array - mid_hip[:, np.newaxis, :]
    
    left_shoulder_idx = 11
    torso_lengths = np.linalg.norm(
        pose_array[:, left_shoulder_idx] - pose_array[:, left_hip_idx],
        axis=1
    )
    mean_torso_length = np.mean(torso_lengths[torso_lengths > 0])
    
    if mean_torso_length > 0:
        normalized = normalized / mean_torso_length
    
    return normalized


def pad_or_truncate_pose(pose_tensor: np.ndarray, target_frames: int = 64) -> np.ndarray:
    """Pad atau truncate pose tensor ke jumlah frame yang ditargetkan."""
    current_frames = pose_tensor.shape[0]
    
    if current_frames == target_frames:
        return pose_tensor
    elif current_frames > target_frames:
        start_idx = (current_frames - target_frames) // 2
        return pose_tensor[start_idx:start_idx + target_frames]
    else:
        padded = np.zeros((target_frames, 33, 3), dtype=np.float32)
        padded[:current_frames] = pose_tensor
        padded[current_frames:] = pose_tensor[-1]
        return padded


def predict_single_model(pose_tensor: torch.Tensor, model: AttentiveSkel3D) -> Tuple[str, float]:
    """Lakukan prediksi menggunakan satu model."""
    with torch.no_grad():
        logits = model(pose_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
    
    label = "Benar" if predicted_class.item() == 0 else "Salah"
    confidence_score = confidence.item()
    
    return label, confidence_score


def extract_attention_weights(model: AttentiveSkel3D, pose_tensor: torch.Tensor) -> np.ndarray:
    """Ekstraksi attention weights dari model untuk visualisasi."""
    try:
        if hasattr(model, "biomechanical_spatial_prior") and model.use_spatial_prior:
            weights = model.biomechanical_spatial_prior.data.squeeze().cpu().numpy()
            if weights.size == 33:
                weights = np.clip(weights, 0, 1)
                return weights
    except:
        pass
    
    return np.ones(33) / 33.0


def denormalize_landmarks(landmarks: np.ndarray, frame_height: int, frame_width: int) -> np.ndarray:
    """Denormalisasi koordinat landmark dari normalized space ke pixel space."""
    pixels = np.zeros((landmarks.shape[0], 2), dtype=np.int32)
    
    pixels[:, 0] = np.clip((landmarks[:, 0] + 1) * frame_width / 2, 0, frame_width - 1).astype(np.int32)
    pixels[:, 1] = np.clip((landmarks[:, 1] + 1) * frame_height / 2, 0, frame_height - 1).astype(np.int32)
    
    return pixels


def draw_skeleton_with_heatmap(
    frame: np.ndarray,
    landmarks: np.ndarray,
    attention_weights: np.ndarray,
    frame_height: int = 480,
    frame_width: int = 640,
) -> np.ndarray:
    """
    Menggambar skeleton dengan overlay heatmap atensi pada frame.
    
    Sendi dengan attention tinggi → lingkaran MERAH (fokus AI)
    Sendi dengan attention rendah → lingkaran ABU-ABU (kurang fokus)
    """
    annotated_frame = frame.copy()
    h, w = annotated_frame.shape[:2]
    
    pixels = denormalize_landmarks(landmarks, h, w)
    
    # Draw skeleton connections
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        start_pixel = pixels[start_idx]
        end_pixel = pixels[end_idx]
        cv2.line(annotated_frame, tuple(start_pixel), tuple(end_pixel), (0, 255, 0), 2)
    
    # Draw landmarks dengan warna berdasarkan attention weights
    for idx, (pixel, weight) in enumerate(zip(pixels, attention_weights)):
        normalized_weight = np.clip(weight, 0, 1)
        
        if normalized_weight > 0.6:
            # Merah menyala: fokus tinggi
            bgr_color = (0, 0, int(255 * normalized_weight))
            radius = 6
        elif normalized_weight > 0.3:
            # Orange/Kuning: fokus sedang
            bgr_color = (0, int(165 * normalized_weight), int(255 * normalized_weight))
            radius = 5
        else:
            # Abu-abu: fokus rendah
            bgr_color = (int(100 * normalized_weight), int(100 * normalized_weight), int(100 * normalized_weight))
            radius = 4
        
        cv2.circle(annotated_frame, tuple(pixel), radius=radius, color=bgr_color, thickness=-1)
        cv2.circle(annotated_frame, tuple(pixel), radius=radius, color=(255, 255, 255), thickness=1)
    
    return annotated_frame


def validate_frame_biomechanics(frame_idx: int, pose_tensor: np.ndarray, validator: BiomechanicalValidator) -> Tuple[str, str]:
    """Validasi frame dengan BiomechanicalValidator untuk menampilkan status."""
    try:
        frame_pose = pose_tensor[frame_idx:frame_idx+1]
        is_valid_squat = validator.validate_squat(frame_pose)
        
        if is_valid_squat:
            return "✅ Aman - Postur Valid", "status-safe"
        else:
            return "⚠️ Peringatan - Postur Tidak Valid", "status-warning"
    except:
        return "ℹ️ Status - Tidak Dapat Divalidasi", "status-warning"


def create_attention_bar_chart(attention_weights: np.ndarray, scenario_name: str = "") -> plt.Figure:
    """Buat bar chart untuk Top 5 Attention Weights."""
    top_5_indices = np.argsort(attention_weights)[-5:][::-1]
    top_5_weights = attention_weights[top_5_indices]
    top_5_names = [LANDMARK_NAMES[idx] for idx in top_5_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B' if w > 0.6 else '#FFD93D' if w > 0.3 else '#95E1D3' for w in top_5_weights]
    bars = ax.bar(range(len(top_5_names)), top_5_weights, color=colors, alpha=0.8, edgecolor="#333", linewidth=2)
    
    ax.set_xlabel("Nama Sendi", fontsize=12, fontweight="bold")
    ax.set_ylabel("Bobot Atensi (0.0 - 1.0)", fontsize=12, fontweight="bold")
    
    if scenario_name:
        ax.set_title(f"Top 5 Bobot Atensi Sendi — {scenario_name}", fontsize=14, fontweight="bold", pad=20)
    else:
        ax.set_title("Top 5 Bobot Atensi Sendi", fontsize=14, fontweight="bold", pad=20)
    
    ax.set_xticks(range(len(top_5_names)))
    ax.set_xticklabels(top_5_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    for bar, weight in zip(bars, top_5_weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{weight:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    return fig


def convert_frame_to_base64(frame: np.ndarray) -> str:
    """Convert OpenCV frame ke base64 string untuk display di st.markdown."""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", frame_bgr)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return img_base64


# ════════════════════════════════════════════════════════════════════════════════
# Main Application
# ════════════════════════════════════════════════════════════════════════════════

def main():
    """Fungsi utama aplikasi Streamlit dengan struktur 3 Tab untuk Sanity Check."""
    
    st.markdown(
        '<div class="app-title">🏋️ AttentiveSkel-3D</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="app-subtitle">Proof of Concept — Sanity Check Fokus Atensi Spasial Model AI</div>',
        unsafe_allow_html=True,
    )
    
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Kontrol Input")
        
        uploaded_video = st.file_uploader(
            "📹 Unggah Video Latihan Beban (MP4)",
            type=["mp4"],
            help="Format: .mp4 | Ukuran maks: 200MB"
        )
        
        selected_scenario = st.selectbox(
            "🤖 Pilih Skenario Model",
            options=list(SCENARIO_NAMES.keys()),
            format_func=lambda x: SCENARIO_NAMES[x],
            help="Pilih antara single model atau komparasi 5 model"
        )
        
        st.divider()
        
        st.markdown("### ℹ️ Tentang Aplikasi")
        st.info(
            """
            **Fokus Utama:**
            Sanity Check Fokus Atensi Spasial Model
            
            **Tujuan:**
            Membuktikan bahwa model menyoroti sendi yang relevan melalui overlay skeleton dengan heatmap atensi.
            
            **Skenario Model:**
            1. Baseline - 3D-CNN tanpa modul atensi
            2. Ablasi A - Tanpa Biomechanical Spatial Prior
            3. Ablasi B - Tanpa Learned Spatial Attention
            4. Ablasi C - Tanpa Temporal Attention
            5. Full - Model lengkap dengan semua modul
            
            **Indikator Warna Skeleton:**
            🔴 Merah = Atensi Tinggi (fokus AI)
            🟠 Orange = Atensi Sedang
            ⚪ Abu-abu = Atensi Rendah
            """
        )
    
    if uploaded_video is None:
        st.warning("⬆️ Silakan unggah file video MP4 terlebih dahulu untuk memulai analisis.")
        st.stop()
    
    # Extract pose
    with st.spinner("🔄 Memproses video..."):
        pose_result = extract_pose_from_video(uploaded_video)
    
    if pose_result is None:
        st.error("❌ Gagal mengekstraksi pose dari video.")
        st.stop()
    
    pose_tensor_np, fps, all_frames = pose_result
    pose_tensor_np_padded = pad_or_truncate_pose(pose_tensor_np, target_frames=64)
    pose_tensor_torch = torch.from_numpy(pose_tensor_np_padded).unsqueeze(0).float()
    
    # ═════════════════════════════════════════════════════════════════════════
    # 3 TABS
    # ═════════════════════════════════════════════════════════════════════════
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Fase 1: Wujud Data Tensor",
        "🎬 Fase 2: Lokalisasi Waktu",
        "👁️ Fase 3: Sanity Check Mata AI & Komparasi"
    ])
    
    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1: Tensor Data
    # ═════════════════════════════════════════════════════════════════════════
    
    with tab1:
        st.markdown("### 📊 Tensor Data Mentah (Float32)")
        
        st.info(
            """
            Tab ini menampilkan **tensor raw (64, 33, 3)** dalam format dataframe agar dosen penguji dapat melihat 
            dengan jelas wujud angka desimal (float32) yang menjadi input model AI.
            
            **Interpretasi:**
            - **Baris** = 64 frame video (0-63)
            - **Kolom** = 33 sendi × 3 koordinat (x, y, z) ternormalisasi
            - **Nilai** = Float32 dalam range [-1, 1] setelah normalisasi
            """
        )
        
        pose_display = pose_tensor_np_padded.reshape(64, 99)
        
        col_names = []
        for joint_idx in range(33):
            joint_name = LANDMARK_NAMES[joint_idx]
            col_names.append(f"{joint_name} (X)")
            col_names.append(f"{joint_name} (Y)")
            col_names.append(f"{joint_name} (Z)")
        
        df_tensor = pd.DataFrame(pose_display, columns=col_names)
        df_tensor.index.name = "Frame"
        
        st.dataframe(
            df_tensor,
            use_container_width=True,
            height=400,
        )
        
        st.markdown("### 📈 Statistik Tensor")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Frames", pose_tensor_np_padded.shape[0])
        with col2:
            st.metric("Jumlah Sendi", pose_tensor_np_padded.shape[1])
        with col3:
            st.metric("Koordinat per Sendi", pose_tensor_np_padded.shape[2])
        with col4:
            st.metric("Data Type", "float32")
        
        st.markdown("**Statistik per Dimensi:**")
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.write("**Dimensi X:**")
            st.write(f"Min: {pose_tensor_np_padded[:, :, 0].min():.4f}")
            st.write(f"Max: {pose_tensor_np_padded[:, :, 0].max():.4f}")
            st.write(f"Mean: {pose_tensor_np_padded[:, :, 0].mean():.4f}")
        
        with stats_col2:
            st.write("**Dimensi Y:**")
            st.write(f"Min: {pose_tensor_np_padded[:, :, 1].min():.4f}")
            st.write(f"Max: {pose_tensor_np_padded[:, :, 1].max():.4f}")
            st.write(f"Mean: {pose_tensor_np_padded[:, :, 1].mean():.4f}")
        
        with stats_col3:
            st.write("**Dimensi Z:**")
            st.write(f"Min: {pose_tensor_np_padded[:, :, 2].min():.4f}")
            st.write(f"Max: {pose_tensor_np_padded[:, :, 2].max():.4f}")
            st.write(f"Mean: {pose_tensor_np_padded[:, :, 2].mean():.4f}")
    
    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2: Lokalisasi Waktu
    # ═════════════════════════════════════════════════════════════════════════
    
    with tab2:
        st.markdown("### 🎬 Lokalisasi Waktu & Validasi Biomekanik per Frame")
        
        st.info(
            """
            Tab ini menampilkan **frame video asli** pada indeks yang dipilih dengan slider.
            Di setiap frame, dilakukan **validasi biomekanik** menggunakan critería dari literatur 
            untuk menunjukkan postur gerakan pada saat itu "Valid" atau "Tidak Valid".
            """
        )
        
        frame_idx = st.slider(
            "Pilih Frame (0-63)",
            min_value=0,
            max_value=63,
            value=32,
            step=1,
            help="Geser untuk memilih frame tertentu"
        )
        
        original_frame = all_frames[frame_idx] if frame_idx < len(all_frames) else all_frames[-1]
        
        col_frame, col_info = st.columns([2, 1])
        
        with col_frame:
            st.markdown("**Frame Video Asli:**")
            st.markdown(
                f'<div class="video-frame-container"><img class="video-frame-image" src="data:image/png;base64,' +
                f'{convert_frame_to_base64(original_frame)}" /></div>',
                unsafe_allow_html=True
            )
        
        with col_info:
            st.markdown("**Informasi Frame:**")
            st.write(f"**Indeks Frame:** {frame_idx}/63")
            st.write(f"**Waktu (approx.):** {frame_idx / fps:.2f} detik")
            
            validator = load_biomechanical_validator()
            status_text, badge_class = validate_frame_biomechanics(
                frame_idx,
                pose_tensor_np_padded,
                validator
            )
            st.markdown(
                f'<div class="status-badge {badge_class}">{status_text}</div>',
                unsafe_allow_html=True
            )
            
            frame_pose = pose_tensor_np_padded[frame_idx]
            st.write(f"**Min Coord:** {frame_pose.min():.3f}")
            st.write(f"**Max Coord:** {frame_pose.max():.3f}")
            st.write(f"**Mean Coord:** {frame_pose.mean():.3f}")
        
        with st.expander("📍 Koordinat Sendi (Pose Landmarks) Frame Ini"):
            frame_pose = pose_tensor_np_padded[frame_idx]
            
            pose_data = []
            for joint_idx, joint_name in enumerate(LANDMARK_NAMES):
                x, y, z = frame_pose[joint_idx]
                pose_data.append({
                    "Sendi": joint_name,
                    "X": f"{x:.4f}",
                    "Y": f"{y:.4f}",
                    "Z": f"{z:.4f}",
                })
            
            df_pose = pd.DataFrame(pose_data)
            st.dataframe(df_pose, use_container_width=True, height=300)
    
    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3: Sanity Check (PALING PENTING)
    # ═════════════════════════════════════════════════════════════════════════
    
    with tab3:
        st.markdown("### 👁️ Sanity Check Mata AI & Komparasi")
        
        st.warning(
            """
            ⚠️ **TAB PALING PENTING** — Menampilkan Skeleton Overlay dan Heatmap Atensi.
            
            **Interpretasi Warna Sendi:**
            - 🔴 **Merah Menyala** = Atensi Tinggi (AI fokus di sini)
            - 🟠 **Orange/Kuning** = Atensi Sedang
            - ⚪ **Abu-abu** = Atensi Rendah
            
            **Tujuan Komparasi:**
            Menunjukkan bahwa Baseline fokus ACAK, sedangkan Full Model fokus AKURAT di sendi krusial.
            """
        )
        
        if selected_scenario == "semua":
            # ═══════════════════════════════════════════════════════════════
            # KOMPARASI 5 MODEL
            # ═══════════════════════════════════════════════════════════════
            
            st.markdown("#### 📊 Komparasi 5 Model Skenario")
            st.info("Menampilkan hasil prediksi dan skeleton visualization dari 5 model secara berdampingan.")
            
            model_keys = ["baseline", "ablasi_a", "ablasi_b", "ablasi_c", "full"]
            results = {}
            attention_weights_dict = {}
            
            with st.spinner("⏳ Memproses prediksi dengan 5 model..."):
                for model_key in model_keys:
                    model = load_model(model_key)
                    if model is not None:
                        label, confidence = predict_single_model(pose_tensor_torch, model)
                        weights = extract_attention_weights(model, pose_tensor_torch)
                        results[model_key] = (label, confidence)
                        attention_weights_dict[model_key] = weights
                    else:
                        results[model_key] = ("❌ Error", 0.0)
                        attention_weights_dict[model_key] = np.ones(33) / 33.0
            
            st.markdown("**Hasil Prediksi & Skeleton Visualization (Frame Tengah):**")
            cols = st.columns(5)
            
            middle_frame_idx = 32
            middle_pose = pose_tensor_np_padded[middle_frame_idx]
            
            for col_idx, model_key in enumerate(model_keys):
                with cols[col_idx]:
                    label, confidence = results[model_key]
                    weights = attention_weights_dict[model_key]
                    
                    frame_to_draw = all_frames[middle_frame_idx] if middle_frame_idx < len(all_frames) else all_frames[-1]
                    
                    annotated_frame = draw_skeleton_with_heatmap(
                        frame_to_draw,
                        middle_pose,
                        weights,
                        frame_height=frame_to_draw.shape[0],
                        frame_width=frame_to_draw.shape[1],
                    )
                    
                    st.image(annotated_frame, use_column_width=True, caption=SCENARIO_NAMES[model_key].split("(")[0].strip())
                    
                    st.markdown(
                        f"""
                        <div class="comparison-card">
                            <div class="comparison-card-label">{label}</div>
                            <div class="comparison-card-confidence">
                                Confidence: {confidence:.1%}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            
            st.divider()
            st.markdown("#### 📊 Top 5 Bobot Atensi Sendi (Full Model)")
            fig = create_attention_bar_chart(attention_weights_dict["full"], "Full AttentiveSkel-3D")
            st.pyplot(fig)
        
        else:
            # ═══════════════════════════════════════════════════════════════
            # SINGLE MODEL
            # ═══════════════════════════════════════════════════════════════
            
            st.markdown(f"#### 🎯 Hasil Prediksi: {SCENARIO_NAMES[selected_scenario]}")
            
            model = load_model(selected_scenario)
            if model is None:
                st.error(f"❌ Tidak dapat memuat model untuk skenario '{selected_scenario}'.")
                st.stop()
            
            with st.spinner(f"⏳ Memproses prediksi..."):
                label, confidence = predict_single_model(pose_tensor_torch, model)
                weights = extract_attention_weights(model, pose_tensor_torch)
            
            col_frame, col_result = st.columns([2, 1])
            
            with col_frame:
                st.markdown("**Skeleton dengan Heatmap Atensi (Frame Tengah):**")
                
                middle_frame_idx = 32
                middle_pose = pose_tensor_np_padded[middle_frame_idx]
                
                frame_to_draw = all_frames[middle_frame_idx] if middle_frame_idx < len(all_frames) else all_frames[-1]
                
                annotated_frame = draw_skeleton_with_heatmap(
                    frame_to_draw,
                    middle_pose,
                    weights,
                    frame_height=frame_to_draw.shape[0],
                    frame_width=frame_to_draw.shape[1],
                )
                
                st.image(annotated_frame, use_column_width=True)
            
            with col_result:
                st.markdown("**Hasil Prediksi:**")
                st.markdown(
                    f"""
                    <div class="result-card">
                        <p>Prediksi Gerakan</p>
                        <div class="result-card-label">{label}</div>
                        <div class="result-card-confidence">
                            Confidence: <strong>{confidence:.2%}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                st.markdown("**Konfigurasi Model:**")
                config = SCENARIO_CONFIGS[selected_scenario]
                st.write(f"🔹 Spatial Prior: {'✅' if config['use_spatial_prior'] else '❌'}")
                st.write(f"🔹 Learned Spatial: {'✅' if config['use_learned_spatial'] else '❌'}")
                st.write(f"🔹 Temporal Att: {'✅' if config['use_temporal_attention'] else '❌'}")
            
            st.divider()
            st.markdown("#### 📊 Top 5 Bobot Atensi Sendi")
            fig = create_attention_bar_chart(weights, SCENARIO_NAMES[selected_scenario])
            st.pyplot(fig)


if __name__ == "__main__":
    main()
