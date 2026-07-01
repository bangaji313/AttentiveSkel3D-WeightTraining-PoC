# src/models/model_3dcnn.py
#
# Implementasi arsitektur AttentiveSkel-3D untuk deteksi kesalahan gerakan
# latihan beban (Squat, Deadlift, Bench Press).
#
# Arsitektur menggabungkan:
#   - 3D Convolutional Neural Network (3D-CNN) ringan sebagai ekstraktor fitur
#     spatio-temporal dari sekuens pose skeleton.
#   - Biomechanical Spatial Prior (BSP): parameter bobot yang dapat dipelajari
#     (learnable attention weight) per landmark sendi, yang berfungsi sebagai
#     inductive bias biomekanikal — sendi yang lebih relevan untuk gerakan
#     tertentu secara otomatis mendapatkan bobot lebih tinggi saat pelatihan.
#
# Format Tensor:
#   Input  : (B, T=64, L=33, C=3)   → (Batch, Frame, Landmark, Koordinat xyz)
#   Output : (B, num_classes)        → Logit untuk setiap kelas gerakan

import torch
import torch.nn as nn


class AttentiveSkel3D(nn.Module):
    """
    AttentiveSkel-3D: 3D-CNN dengan modul atensi biomekanikal granular.

    Alur forward pass (mode penuh):
        Input (B, 64, 33, 3)
            → Reshape ke (B, 3, 64, 33, 1)              # Sesuaikan ke format Conv3d
            → Biomechanical Spatial Prior (× BSP)        # Bobot per-sendi learnable [opsional]
            → Conv Block 1 (3→32 channels)               # Ekstraksi fitur level rendah
            → Conv Block 2 (32→64 channels)              # Ekstraksi fitur level menengah
            → Conv Block 3 (64→128 channels)             # Ekstraksi fitur level tinggi
            → Learned Spatial Attention (× channel-wise) # Atensi spasial adaptif [opsional]
            → Temporal Attention (× softmax per waktu)   # Pembobotan temporal [opsional]
            → Global Average Pooling + Flatten           # Agregasi global
            → Classifier Head (128→64→num_classes)       # Klasifikasi akhir

    Args:
        num_classes           (int)  : Jumlah kelas output. Default = 2.
        use_attention         (bool) : Alias kompatibilitas mundur. Jika False, ketiga
                                       flag di bawah diset False secara otomatis.
                                       Default = True.
        use_spatial_prior     (bool) : Aktifkan Biomechanical Spatial Prior (BSP) —
                                       perkalian bobot per-sendi sebelum conv blocks.
                                       Default = True.
        use_learned_spatial   (bool) : Aktifkan Learned Spatial Attention — bobot
                                       adaptif per-channel setelah conv_block_3.
                                       Default = True.
        use_temporal_attention (bool): Aktifkan Temporal Attention — pembobotan frame
                                       berbasis softmax sepanjang dimensi waktu.
                                       Default = True.
    """

    def __init__(
        self,
        num_classes: int = 2,
        use_attention: bool = True,
        use_spatial_prior: bool = True,
        use_learned_spatial: bool = True,
        use_temporal_attention: bool = True,
    ):
        super(AttentiveSkel3D, self).__init__()

        # ----------------------------------------------------------------
        # Simpan flag kontrol atensi
        # ----------------------------------------------------------------
        # use_attention=False sebagai override global (kompatibilitas mundur)
        if not use_attention:
            use_spatial_prior      = False
            use_learned_spatial    = False
            use_temporal_attention = False

        self.use_attention          = use_attention
        self.use_spatial_prior      = use_spatial_prior
        self.use_learned_spatial    = use_learned_spatial
        self.use_temporal_attention = use_temporal_attention

        # ----------------------------------------------------------------
        # Biomechanical Spatial Prior (BSP)
        # ----------------------------------------------------------------
        # Parameter learnable berukuran (1, 1, 1, 33, 1) yang akan di-broadcast
        # dan dikalikan element-wise dengan tensor (B, C, T, 33, 1).
        # Sigmoid memastikan bobot selalu positif (0–1).
        # Hanya dialokasikan jika use_spatial_prior=True.
        if self.use_spatial_prior:
            self.biomechanical_spatial_prior = nn.Parameter(
                torch.ones(1, 1, 1, 33, 1)  # Inisialisasi semua bobot = 1.0
            )

        # ----------------------------------------------------------------
        # Learned Spatial Attention
        # ----------------------------------------------------------------
        # Channel Attention (SE-style) ringan: merangkum konteks global tiap channel
        # via GAP lalu memprediksi bobot tiap channel dengan MLP kecil.
        # Input : (B, 128, T', L', 1) → GAP → (B, 128) → MLP → (B, 128) → sigmoid
        # Bobot di-broadcast kembali ke (B, 128, 1, 1, 1) untuk scaling.
        if self.use_learned_spatial:
            self.learned_spatial_attention = nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 128),
                nn.Sigmoid(),
            )

        # ----------------------------------------------------------------
        # Temporal Attention
        # ----------------------------------------------------------------
        # Menghasilkan skor kepentingan per-frame dari representasi global tiap frame.
        # Input : (B, 128, T', 1, 1) → conv1d-like → (B, 1, T') → softmax → weights
        # Pembobotan: x = x * weights → aggregasi penting temporal.
        if self.use_temporal_attention:
            self.temporal_attention = nn.Sequential(
                nn.Conv3d(128, 1, kernel_size=(1, 1, 1), bias=True),
            )

        # ----------------------------------------------------------------
        # Conv Block 1 — Ekstraksi fitur level rendah
        # ----------------------------------------------------------------
        # Input : (B, 3,  64, 33, 1)
        # Output: (B, 32, 64, 16, 1)  ← MaxPool3d(1,2,1) mengecilkan dimensi landmark
        self.conv_block_1 = nn.Sequential(
            # Konvolusi 3D: menangkap pola spatio-temporal lokal
            # kernel (T=3, H=3, W=1): 3 frame waktu × 3 landmark bertetangga
            nn.Conv3d(
                in_channels=3,
                out_channels=32,
                kernel_size=(3, 3, 1),
                padding=(1, 1, 0),   # Same-padding pada dimensi waktu & landmark
                bias=False,          # Bias dinonaktifkan karena ada BatchNorm
            ),
            nn.BatchNorm3d(32),      # Normalisasi per-channel untuk stabilitas pelatihan
            nn.ReLU(inplace=True),
            # Pooling hanya pada dimensi spasial landmark (H): 33 → 16
            # Dimensi waktu (T) dibiarkan utuh pada tahap ini
            nn.MaxPool3d(kernel_size=(1, 2, 1)),
        )

        # ----------------------------------------------------------------
        # Conv Block 2 — Ekstraksi fitur level menengah
        # ----------------------------------------------------------------
        # Input : (B, 32, 64, 16, 1)
        # Output: (B, 64, 32,  8, 1)  ← MaxPool3d(2,2,1) mengecilkan waktu & landmark
        self.conv_block_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3, 1),
                padding=(1, 1, 0),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # Pooling pada dimensi waktu (64→32) DAN landmark (16→8)
            nn.MaxPool3d(kernel_size=(2, 2, 1)),
        )

        # ----------------------------------------------------------------
        # Conv Block 3 — Ekstraksi fitur level tinggi
        # ----------------------------------------------------------------
        # Input : (B, 64,  32, 8, 1)
        # Output: (B, 128, 32, 8, 1)  ← tidak ada pooling, hanya perluasan channel
        self.conv_block_3 = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3, 1),
                padding=(1, 1, 0),
                bias=False,
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        # ----------------------------------------------------------------
        # Global Average Pooling
        # ----------------------------------------------------------------
        # Agregasi seluruh dimensi spatio-temporal menjadi satu vektor per channel.
        # Input : (B, 128, 32, 8, 1)
        # Output: (B, 128, 1,  1, 1) → setelah flatten → (B, 128)
        self.global_avg_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

        # ----------------------------------------------------------------
        # Classifier Head
        # ----------------------------------------------------------------
        # Input : (B, 128)
        # Output: (B, num_classes) → logit mentah (gunakan CrossEntropyLoss)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),       # Regularisasi untuk mencegah overfitting
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alur forward pass AttentiveSkel-3D.

        Args:
            x (torch.Tensor): Tensor input dengan bentuk (B, T=64, L=33, C=3).

        Returns:
            torch.Tensor: Logit output dengan bentuk (B, num_classes).
        """
        # ------------------------------------------------------------------
        # Langkah 1: Reshape tensor ke format yang dibutuhkan oleh nn.Conv3d
        # ------------------------------------------------------------------
        # Format Conv3d: (Batch, Channels, Depth/Time, Height, Width)
        #
        # Input awal : (B, T=64, L=33, C=3)
        # permute    : (B, C=3, T=64, L=33) → pindahkan channel ke posisi 1
        # unsqueeze  : (B, C=3, T=64, L=33, W=1) → tambah dimensi Width semu

        x = x.permute(0, 3, 1, 2)   # (B, 64, 33, 3) → (B, 3, 64, 33)
        x = x.unsqueeze(-1)          # (B, 3, 64, 33) → (B, 3, 64, 33, 1)

        # ------------------------------------------------------------------
        # Langkah 2: Biomechanical Spatial Prior (BSP)
        #            — hanya aktif jika use_spatial_prior=True
        # ------------------------------------------------------------------
        if self.use_spatial_prior:
            # Terapkan bobot per-sendi BSP (learnable).
            # BSP shape : (1, 1,  1, 33, 1)
            # x shape   : (B, 3, 64, 33, 1)
            # Broadcast : (B, 3, 64, 33, 1) — tiap sendi diskalakan.
            # Sigmoid memastikan bobot selalu positif (0–1).
            bsp_weights = torch.sigmoid(self.biomechanical_spatial_prior)
            x = x * bsp_weights      # Element-wise multiplication
        # Jika use_spatial_prior=False: tensor x mengalir tanpa modifikasi BSP.

        # ------------------------------------------------------------------
        # Langkah 3: Ekstraksi fitur bertahap melalui 3 Conv Block
        # ------------------------------------------------------------------
        x = self.conv_block_1(x)     # (B, 32, 64, 16, 1)
        x = self.conv_block_2(x)     # (B, 64, 32,  8, 1)
        x = self.conv_block_3(x)     # (B, 128, 32, 8, 1)

        # ------------------------------------------------------------------
        # Langkah 4: Learned Spatial Attention (Channel Attention SE-style)
        #            — hanya aktif jika use_learned_spatial=True
        # ------------------------------------------------------------------
        if self.use_learned_spatial:
            # GAP di seluruh dimensi spasial & temporal → vektor channel (B, 128)
            # lalu MLP menghasilkan bobot per-channel, di-broadcast kembali.
            # x shape : (B, 128, T', L', 1)
            gap_feat = x.mean(dim=[2, 3, 4])              # (B, 128)
            ch_weights = self.learned_spatial_attention(gap_feat)  # (B, 128) sigmoid
            x = x * ch_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # (B, 128, T', L', 1) × (B, 128, 1, 1, 1) → broadcast scaling
        # Jika use_learned_spatial=False: lewati sepenuhnya.

        # ------------------------------------------------------------------
        # Langkah 5: Temporal Attention
        #            — hanya aktif jika use_temporal_attention=True
        # ------------------------------------------------------------------
        if self.use_temporal_attention:
            # Prediksi skor kepentingan per-frame menggunakan konvolusi 1×1×1.
            # x shape    : (B, 128, T', L', 1)
            # scores     : (B, 1,   T', L', 1) → rata-rata spasial → (B, 1, T', 1, 1)
            # softmax    : normalkan sepanjang dimensi waktu T'
            scores = self.temporal_attention(x)            # (B, 1, T', L', 1)
            scores = scores.mean(dim=[3, 4], keepdim=True) # (B, 1, T', 1, 1)
            t_weights = torch.softmax(scores, dim=2)       # (B, 1, T', 1, 1) softmax T
            x = x * t_weights                              # broadcast: (B, 128, T', L', 1)
        # Jika use_temporal_attention=False: lewati pembobotan temporal.

        # ------------------------------------------------------------------
        # Langkah 6: Global Average Pooling + Flatten
        # ------------------------------------------------------------------
        x = self.global_avg_pool(x)  # (B, 128, 1, 1, 1)
        x = x.flatten(start_dim=1)   # (B, 128)

        # ------------------------------------------------------------------
        # Langkah 7: Classifier Head → logit output
        # ------------------------------------------------------------------
        x = self.classifier(x)       # (B, num_classes)

        return x


# ============================================================
# Utilitas: hitung jumlah total parameter model
# ============================================================
def count_parameters(model: nn.Module) -> int:
    """Mengembalikan jumlah parameter yang dapat dilatih (trainable parameters)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
