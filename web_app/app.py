from pathlib import Path
import sys
import tempfile

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import streamlit as st
import torch

# =====================================================================================
# Konfigurasi awal halaman Streamlit (WAJIB dipanggil paling pertama)
# =====================================================================================
st.set_page_config(
    page_title="AttentiveSkel-3D Gym Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================================
# Setup sys.path agar import src.models bisa dikenali dari mana pun app dijalankan
# =====================================================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.model_3dcnn import AttentiveSkel3D

# Inisialisasi MediaPipe Pose sekali saja (singleton — hemat memori)
_mp_pose   = mp.solutions.pose
_POSE_CONN = _mp_pose.POSE_CONNECTIONS

# Threshold atensi: sendi dengan nilai > ini akan digambar MERAH
ATTENTION_THR = 0.4


# =====================================================================================
# CSS: Tema Gym Dark Mode Modern
# =====================================================================================
def apply_custom_theme() -> None:
    """Menyuntikkan CSS custom untuk nuansa gym dark mode modern."""
    st.markdown(
        """
        <style>
            :root {
                --bg-main:       #0b0f14;
                --bg-card:       #121a22;
                --line:          #2a3a4a;
                --text-main:     #f5f7fa;
                --text-muted:    #9fb0c0;
                --accent-neon:   #39ff14;
                --accent-yellow: #e8ff34;
                --accent-red:    #ff375f;
            }

            .stApp {
                background:
                    radial-gradient(circle at 20% 15%, rgba(57, 255, 20, 0.10), transparent 35%),
                    radial-gradient(circle at 85% 20%, rgba(232, 255, 52, 0.09), transparent 40%),
                    radial-gradient(circle at 50% 100%, rgba(255, 55, 95, 0.09), transparent 40%),
                    var(--bg-main);
                color: var(--text-main);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0d141d 0%, #0b1118 100%);
                border-right: 1px solid #1f2a36;
            }

            .gym-banner {
                border: 1px solid var(--line);
                background: linear-gradient(135deg,
                    rgba(57, 255, 20, 0.12),
                    rgba(232, 255, 52, 0.06),
                    rgba(255, 55, 95, 0.10));
                border-radius: 18px;
                padding: 1.2rem 1.4rem;
                margin-bottom: 0.8rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
            }

            .gym-banner h1 {
                font-size: 2.1rem;
                margin: 0;
                color: var(--text-main);
                letter-spacing: 0.4px;
            }

            .gym-banner p {
                margin: 0.45rem 0 0 0;
                color: var(--text-muted);
                font-size: 1rem;
            }

            div[data-testid="stMetric"] {
                background: linear-gradient(180deg, rgba(18, 26, 34, 0.95), rgba(18, 26, 34, 0.72));
                border: 1px solid #233344;
                border-radius: 14px;
                padding: 0.8rem;
                box-shadow: 0 8px 18px rgba(0, 0, 0, 0.22);
            }

            div.stButton > button:first-child {
                width: 100%;
                min-height: 3.3rem;
                font-size: 1.15rem;
                font-weight: 800;
                border-radius: 12px;
                border: none;
                color: #0a1116;
                background: linear-gradient(120deg, var(--accent-neon), var(--accent-yellow));
                box-shadow: 0 10px 24px rgba(57, 255, 20, 0.35);
                transition: all 0.2s ease;
            }

            div.stButton > button:first-child:hover {
                transform: translateY(-1px);
                box-shadow: 0 14px 28px rgba(57, 255, 20, 0.45);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================================
# Fungsi Inti (diadopsi dari 06_attention_visualization.ipynb)
# =====================================================================================

def extract_attention_weights(mdl: AttentiveSkel3D) -> np.ndarray:
    """
    Ekstrak array bobot atensi spasial (33,) dari model yang sudah dimuat.

    - Model dengan biomechanical_spatial_prior (Full / Ablasi B / Ablasi C):
        sigmoid(BSP) → di-flatten menjadi (33,) float32
    - Model tanpa BSP (Baseline / Ablasi A):
        Kembalikan array seragam 0.5 — model tidak punya preferensi spasial.

    Nilai output sudah di-normalisasi ke [0, 1] via Min-Max Scaling.
    """
    mdl.eval()
    with torch.no_grad():
        if hasattr(mdl, "biomechanical_spatial_prior"):
            bsp_raw = mdl.biomechanical_spatial_prior          # (1, 1, 1, 33, 1)
            weights = torch.sigmoid(bsp_raw).squeeze().cpu().numpy().astype(np.float32)
        else:
            # Tidak ada BSP → bobot seragam sebagai penanda "model buta spasial"
            weights = np.full(33, 0.5, dtype=np.float32)

    # Min-Max Normalization → [0, 1]
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min > 1e-8:
        weights = (weights - w_min) / (w_max - w_min)
    return weights


def build_inference_tensor(
    video_path: str,
    max_frames: int = 64,
) -> "tuple[torch.Tensor | None, list, list]":
    """
    Baca video dari disk, ekstrak pose MediaPipe per-frame, kembalikan:
      - tensor     : torch.Tensor (1, max_frames, 33, 3) — siap dimasukkan model
      - lm_frames  : list landmark MediaPipe per-frame (untuk visualisasi)
      - raw_frames : list np.ndarray BGR (frame asli untuk latar overlay)

    cv2.VideoCapture membutuhkan path string ke file di disk.
    UploadedFile dari Streamlit WAJIB disimpan ke tempfile dulu sebelum dikirim sini.

    Jika frame < max_frames → padding dengan frame terakhir.
    Jika frame > max_frames → subsampel terdistribusi rata (linspace).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, [], []

    sequence:    list = []
    lm_frames:   list = []
    raw_frames:  list = []

    with _mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result    = pose.process(frame_rgb)
            if result.pose_landmarks is None:
                continue
            lms    = result.pose_landmarks.landmark
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in lms],
                dtype=np.float32,
            )  # (33, 3)
            sequence.append(coords)
            lm_frames.append(lms)
            raw_frames.append(frame_bgr.copy())

    cap.release()

    if len(sequence) == 0:
        return None, [], []

    total = len(sequence)
    if total >= max_frames:
        # Subsampel terdistribusi rata agar distribusi waktu terjaga
        indices    = np.linspace(0, total - 1, max_frames, dtype=int)
        sequence   = [sequence[i]   for i in indices]
        lm_frames  = [lm_frames[i]  for i in indices]
        raw_frames = [raw_frames[i] for i in indices]
    else:
        # Padding dengan frame terakhir jika video lebih pendek dari max_frames
        pad        = max_frames - total
        sequence   = sequence   + [sequence[-1]]   * pad
        lm_frames  = lm_frames  + [lm_frames[-1]]  * pad
        raw_frames = raw_frames + [raw_frames[-1]]  * pad

    tensor = torch.tensor(
        np.stack(sequence, axis=0),   # (max_frames, 33, 3)
        dtype=torch.float32,
    ).unsqueeze(0)                    # → (1, max_frames, 33, 3)

    return tensor, lm_frames, raw_frames


def draw_attention_skeleton(
    frame: np.ndarray,
    landmarks,
    attention_weights: np.ndarray,
    threshold: float = ATTENTION_THR,
) -> np.ndarray:
    """
    Gambar overlay skeleton berbasis atensi pada satu frame BGR.

    Aturan visual:
      attention_weights[i] > threshold  →  MERAH TERANG (0,0,255) + Glow, radius 14 px
      attention_weights[i] ≤ threshold  →  ABU-ABU REDUP (80,80,80),        radius  5 px

    Tulang:
      Rata-rata kedua ujung > threshold  →  Merah tebal (lebar 3 px)
      Sebaliknya                         →  Abu-abu sangat tipis (1 px)

    Label nilai atensi per sendi ditampilkan dalam warna CYAN.
    """
    out  = frame.copy()
    h, w = out.shape[:2]
    n_lm = min(33, len(landmarks))

    COLOR_KRITIS  = (0,   0, 255)   # BGR — Merah terang
    COLOR_NON     = (80,  80,  80)  # BGR — Abu-abu redup
    COLOR_BONE_HI = (0,   0, 200)   # BGR — Merah untuk tulang kritis
    COLOR_BONE_LO = (50,  50,  50)  # BGR — Abu-abu gelap untuk tulang non-kritis
    RADIUS_HI     = 14
    RADIUS_LO     = 5

    # Gelapkan latar belakang agar overlay lebih kontras
    out = (out.astype(np.float32) * 0.45).astype(np.uint8)

    # ── Pass 1: Efek Glow melingkari sendi kritis ────────────────────────────
    glow_layer = np.zeros_like(out)
    for i in range(n_lm):
        lm = landmarks[i]
        if getattr(lm, "visibility", 1.0) < 0.3:
            continue
        if attention_weights[i] > threshold:
            px = int(lm.x * w)
            py = int(lm.y * h)
            cv2.circle(glow_layer, (px, py), RADIUS_HI + 18, COLOR_KRITIS, -1)
    glow_blur = cv2.GaussianBlur(glow_layer, (51, 51), 20)
    out       = cv2.addWeighted(out, 1.0, glow_blur, 0.55, 0)

    # ── Pass 2: Garis tulang (koneksi BlazePose standar) ────────────────────
    for connection in _POSE_CONN:
        s_idx, e_idx = connection
        if s_idx >= n_lm or e_idx >= n_lm:
            continue
        lm_s = landmarks[s_idx]
        lm_e = landmarks[e_idx]
        if getattr(lm_s, "visibility", 1.0) < 0.3 or getattr(lm_e, "visibility", 1.0) < 0.3:
            continue
        x1 = int(lm_s.x * w);  y1 = int(lm_s.y * h)
        x2 = int(lm_e.x * w);  y2 = int(lm_e.y * h)
        avg_w = (attention_weights[s_idx] + attention_weights[e_idx]) / 2.0
        if avg_w > threshold:
            cv2.line(out, (x1, y1), (x2, y2), COLOR_BONE_HI, 3, cv2.LINE_AA)
        else:
            cv2.line(out, (x1, y1), (x2, y2), COLOR_BONE_LO, 1, cv2.LINE_AA)

    # ── Pass 3: Titik sendi + label nilai atensi ────────────────────────────
    for i in range(n_lm):
        lm = landmarks[i]
        if getattr(lm, "visibility", 1.0) < 0.3:
            continue
        px = int(lm.x * w)
        py = int(lm.y * h)
        if attention_weights[i] > threshold:
            # MERAH TERANG — sendi kritis
            cv2.circle(out, (px, py), RADIUS_HI, COLOR_KRITIS, -1, cv2.LINE_AA)
            cv2.circle(out, (px, py), RADIUS_HI, (255, 255, 255), 2,  cv2.LINE_AA)
            cv2.circle(out, (px, py), 3,          (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                out, f"{i}:{attention_weights[i]:.2f}",
                (px + RADIUS_HI + 4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 255, 255), 2, cv2.LINE_AA,
            )
        else:
            # ABU-ABU REDUP — sendi non-kritis
            cv2.circle(out, (px, py), RADIUS_LO, COLOR_NON, -1, cv2.LINE_AA)
            cv2.putText(
                out, f"{i}:{attention_weights[i]:.2f}",
                (px + RADIUS_LO + 2, py - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (0, 160, 160), 1, cv2.LINE_AA,
            )

    return out


# =====================================================================================
# Pemuatan model (di-cache agar tidak dimuat ulang setiap kali Streamlit rerun)
# =====================================================================================
@st.cache_resource(show_spinner=False)
def load_model(model_relative_path: str):
    """
    Memuat checkpoint model AttentiveSkel3D.

    strict=False    → agar model ablasi (tanpa BSP) tidak crash (Missing keys).
    weights_only=True → keamanan: hindari deserialisasi kode arbitrer dari .pth
                        (fallback ke weights_only=False untuk format checkpoint lama).
    """
    model_path = ROOT_DIR / model_relative_path
    if not model_path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")

    mdl = AttentiveSkel3D(num_classes=2)

    # Coba muat dengan weights_only=True (aman); fallback jika format lama
    try:
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)

    # Deteksi format checkpoint secara otomatis
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    # KRUSIAL: strict=False — model ablasi tidak memiliki semua key
    incompatible = mdl.load_state_dict(state_dict, strict=False)
    mdl.eval()

    return mdl, model_path, incompatible


# =====================================================================================
# Konstanta pilihan model
# =====================================================================================
MODEL_OPTIONS = {
    "Full Model - Pelatih Sempurna":    "models/saved_models/AttentiveSkel3D_Final.pth",
    "Baseline 3D-CNN":                  "models/saved_models/baseline_3dcnn_model.pth",
    "Ablasi A - Tanpa Prior":           "models/saved_models/ablasi_a_no_prior.pth",
    "Ablasi B - Tanpa Learned Spatial": "models/saved_models/ablasi_b_no_learned.pth",
    "Ablasi C - Tanpa Temporal":        "models/saved_models/ablasi_c_no_temporal.pth",
}

CLASS_NAMES = {0: "Form Benar ✔", 1: "Form Salah ✗"}


# =====================================================================================
# Render halaman
# =====================================================================================
apply_custom_theme()

# ── Sidebar: konfigurasi model ────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Konfigurasi Pelatih AI")
    st.caption("Pilih varian model AttentiveSkel-3D untuk evaluasi gerakan.")

    selected_label = st.selectbox(
        "Pilih skenario model:",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
    )
    selected_model_path = MODEL_OPTIONS[selected_label]
    st.caption(f"📂 Path: `{selected_model_path}`")

model_ready       = False
active_model      = None
load_error        = ""
incompatible_keys = None

try:
    with st.sidebar:
        with st.spinner("Memuat model terpilih..."):
            active_model, resolved_path, incompatible_keys = load_model(selected_model_path)
        st.success("Model berhasil dimuat! 💪")
        st.caption(f"✔ `{resolved_path.name}`")

        missing_keys    = list(getattr(incompatible_keys, "missing_keys",   []))
        unexpected_keys = list(getattr(incompatible_keys, "unexpected_keys", []))
        if missing_keys or unexpected_keys:
            with st.expander("ℹ️ Info kompatibilitas state_dict"):
                st.write(f"Missing keys    : **{len(missing_keys)}** (strict=False OK)")
                st.write(f"Unexpected keys : **{len(unexpected_keys)}**")
                if missing_keys:
                    st.code("\n".join(missing_keys[:10]))

    model_ready = True
except Exception as exc:
    load_error = str(exc)
    with st.sidebar:
        st.error("Gagal memuat model. Periksa path/checkpoint.")
        st.caption(load_error)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="gym-banner">
        <h1>🏋️ AttentiveSkel-3D AI Gym Coach</h1>
        <p>
            Analisis form latihan beban secara real-time dengan deep learning.
            Jaga performa, minimalkan cedera, dan latih teknik seperti atlet! 🔥
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
kpi_col1.metric("Model Aktif",      selected_label)
kpi_col2.metric("Status Sistem",    "READY 🟢" if model_ready else "ERROR 🔴")
kpi_col3.metric("Threshold Atensi", f"> {ATTENTION_THR:.1f}")

st.markdown("---")
st.markdown("### 🎥 Upload Video Latihan")
uploaded_video = st.file_uploader(
    "Unggah video latihan (.mp4)",
    type=["mp4"],
    accept_multiple_files=False,
)

if uploaded_video is not None:
    # Tampilkan pratinjau video asli yang diunggah
    st.video(uploaded_video)

    start_analysis = st.button(
        "🔥 Mulai Analisis Gerakan",
        type="primary",
        use_container_width=True,
    )

    if start_analysis:
        if not model_ready:
            st.error("⚠️ Model belum siap. Pilih model lain atau cek file checkpoint.")
            st.caption(load_error)
            st.stop()

        # ── Simpan UploadedFile ke file sementara ─────────────────────────────
        # cv2.VideoCapture tidak bisa membaca objek UploadedFile Streamlit
        # secara langsung — file wajib ditulis ke disk dulu via tempfile.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.getbuffer())
            tmp_path = tmp.name

        try:
            # ── Fase 1: Ekstraksi pose & bangun tensor inferensi ───────────────
            with st.spinner("🧠 AI mengobservasi gerakanmu... Mohon tunggu."):
                tensor_input, lm_frames, raw_frames = build_inference_tensor(
                    tmp_path, max_frames=64
                )

            if tensor_input is None:
                st.error(
                    "❌ Tidak ada pose tubuh yang terdeteksi di video ini. "
                    "Pastikan seluruh tubuh terlihat jelas di kamera."
                )
                st.stop()

            # ── Fase 2: Inferensi model ────────────────────────────────────────
            with torch.no_grad():
                logits     = active_model(tensor_input)       # (1, num_classes)
                probs      = torch.softmax(logits, dim=-1)    # (1, num_classes)
                pred_class = int(logits.argmax(dim=-1).item())
                confidence = float(probs[0, pred_class].item()) * 100.0

            # ── Fase 3: Ekstraksi bobot atensi ────────────────────────────────
            attn_weights = extract_attention_weights(active_model)
            top5_idx     = np.argsort(attn_weights)[::-1][:5]
            has_bsp      = hasattr(active_model, "biomechanical_spatial_prior")

            # ── Tampilkan hasil inferensi ──────────────────────────────────────
            st.markdown("### 📊 Hasil Evaluasi Model")
            pred_label = CLASS_NAMES.get(pred_class, f"Kelas {pred_class}")
            if pred_class == 0:
                st.success(f"🏆 Form Sempurna! Lanjutkan! 💪🔥 — {pred_label}")
            else:
                st.error(f"🚨 Awas Cedera! Perbaiki Posturmu! — {pred_label}")

            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Prediksi",   pred_label)
            res_col2.metric("Confidence", f"{confidence:.1f}%")
            res_col3.metric(
                "BSP Layer",
                "Ada ✔" if has_bsp else "Tidak (Ablasi/Baseline)",
            )
            st.progress(confidence / 100.0)

            # ── Fase 4: Pemutaran frame dengan overlay skeleton atensi ─────────
            st.markdown("### 🔴 Visualisasi Biomechanical Attention")
            top5_label = "  |  ".join(
                f"Sendi #{int(i)}: {attn_weights[i]:.3f}" for i in top5_idx
            )
            st.caption(
                f"⬤ **MERAH** = atensi > {ATTENTION_THR}  |  "
                f"● **abu-abu** = atensi ≤ {ATTENTION_THR}  |  "
                f"Top-5: {top5_label}"
            )

            # Placeholder tunggal yang diperbarui setiap frame (efek video live)
            frame_window = st.empty()

            for idx, (raw_bgr, lm) in enumerate(zip(raw_frames, lm_frames)):
                # Gambar skeleton berbasis atensi di atas frame asli
                annotated_bgr = draw_attention_skeleton(
                    frame             = raw_bgr,
                    landmarks         = lm,
                    attention_weights = attn_weights,
                    threshold         = ATTENTION_THR,
                )

                # HUD bawah: nomor frame + hasil prediksi
                h_f, w_f = annotated_bgr.shape[:2]
                cv2.putText(
                    annotated_bgr,
                    f"Frame {idx + 1}/{len(raw_frames)}  |  {pred_label}  ({confidence:.0f}%)",
                    (10, h_f - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (200, 200, 200), 1, cv2.LINE_AA,
                )

                # Konversi BGR → RGB sebelum dikirim ke Streamlit
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                frame_window.image(annotated_rgb, channels="RGB", use_container_width=True)

            st.success(f"✅ Selesai! {len(raw_frames)} frame diproses dengan overlay atensi.")

            # ── Panel detail bobot atensi 33 sendi ────────────────────────────
            with st.expander("📈 Detail Bobot Atensi 33 Sendi BlazePose"):
                fig, ax = plt.subplots(figsize=(12, 3))
                fig.patch.set_facecolor("#0b0f14")
                ax.set_facecolor("#121a22")
                bar_colors = [
                    "#ff375f" if attn_weights[i] > ATTENTION_THR else "#334d66"
                    for i in range(33)
                ]
                ax.bar(range(33), attn_weights, color=bar_colors, edgecolor="none")
                ax.axhline(
                    ATTENTION_THR, color="#e8ff34", linestyle="--",
                    linewidth=1.2, label=f"Threshold = {ATTENTION_THR}",
                )
                ax.set_xlabel("Indeks Sendi BlazePose", color="#9fb0c0")
                ax.set_ylabel("Bobot Atensi (0–1)",     color="#9fb0c0")
                ax.set_title(
                    f"Distribusi Bobot Atensi Spasial — {selected_label}",
                    color="#f5f7fa", fontweight="bold",
                )
                ax.tick_params(colors="#9fb0c0")
                ax.legend(facecolor="#1a2535", labelcolor="#e8ff34")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2a3a4a")
                st.pyplot(fig)
                plt.close(fig)

                st.caption(
                    "🔴 Merah = atensi di atas threshold (sendi kritis)  |  "
                    "🔵 Biru gelap = atensi rendah  |  "
                    f"Sumber BSP: {'sigmoid(biomechanical_spatial_prior)' if has_bsp else 'Seragam 0.5 (tidak ada BSP)'}"
                )

        finally:
            # Bersihkan file sementara agar tidak menumpuk di disk
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

else:
    st.info(
        "📂 Silakan unggah video latihan (.mp4) untuk memulai analisis gerakan. "
        "Pastikan seluruh tubuh terlihat jelas di kamera. 🏃"
    )
