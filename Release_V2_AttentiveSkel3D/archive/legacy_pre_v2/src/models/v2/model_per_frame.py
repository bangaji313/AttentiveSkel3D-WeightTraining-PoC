# src/models/v2/model_per_frame.py
#
# AttentiveSkel3D-v2: Arsitektur per-frame classifier.
#
# PERUBAHAN UTAMA dari v1:
#   - Global Average Pooling (GAP) DIHAPUS sepenuhnya.
#   - Setelah conv blocks + atensi, dimensi temporal DIPERTAHANKAN (tidak diagregasi).
#   - Tambahkan AdaptiveAvgPool3d pada dimensi LANDMARK saja → output (B, 128, T', 1, 1).
#   - Gunakan Upsample / interpolate untuk mengembalikan T' → 64 frame.
#   - Classifier head diterapkan per-frame dengan nn.Linear → output (B, 64, num_classes).
#
# Format Tensor:
#   Input  : (B, T=64, L=33, C=3)
#   Output : (B, T=64, num_classes=2)   → logit per-frame

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveSkel3DPerFrame(nn.Module):
    """
    AttentiveSkel-3D v2: klasifikasi per-frame dengan output (B, 64, 2).

    Alur forward pass:
        Input (B, 64, 33, 3)
            → Reshape   → (B, 3, 64, 33, 1)           # Format Conv3d
            → BSP (×)   → (B, 3, 64, 33, 1)           # Biomechanical Spatial Prior [opsional]
            → Block 1   → (B, 32, 64, 16, 1)           # MaxPool hanya pada dimensi landmark
            → Block 2   → (B, 64, 32,  8, 1)           # MaxPool pada landmark (waktu dibiarkan 32)
            → Block 3   → (B, 128, 32, 8, 1)           # Tidak ada pooling
            → Ch-Attn   → (B, 128, 32, 8, 1)           # Learned Spatial Attention [opsional]
            → T-Attn    → (B, 128, 32, 8, 1)           # Temporal Attention [opsional]
            → LandPool  → (B, 128, 32, 1, 1)           # Pool hanya dimensi landmark → (B,128,T',1,1)
            → Upsample  → (B, 128, 64, 1, 1)           # Kembalikan T' ke 64 frame
            → Squeeze   → (B, 128, 64)                  # Hilangkan dimensi W semu
            → Permute   → (B, 64, 128)                  # Siapkan untuk Linear per-frame
            → Linear    → (B, 64, num_classes)          # Logit per-frame

    Args:
        num_classes            (int)  : Jumlah kelas. Default 2.
        use_spatial_prior      (bool) : Aktifkan Biomechanical Spatial Prior. Default True.
        use_learned_spatial    (bool) : Aktifkan Channel Attention (SE-style). Default True.
        use_temporal_attention (bool) : Aktifkan Temporal Attention. Default True.
    """

    def __init__(
        self,
        num_classes: int = 2,
        use_spatial_prior: bool = True,
        use_learned_spatial: bool = True,
        use_temporal_attention: bool = True,
    ):
        super(AttentiveSkel3DPerFrame, self).__init__()

        # Simpan flag kontrol atensi
        self.use_spatial_prior      = use_spatial_prior
        self.use_learned_spatial    = use_learned_spatial
        self.use_temporal_attention = use_temporal_attention

        # ----------------------------------------------------------------
        # Biomechanical Spatial Prior (BSP)
        # ----------------------------------------------------------------
        # Parameter learnable (1, 1, 1, 33, 1) — bobot per sendi.
        # Sigmoid memastikan bobot selalu dalam rentang 0–1.
        if self.use_spatial_prior:
            self.biomechanical_spatial_prior = nn.Parameter(
                torch.ones(1, 1, 1, 33, 1)
            )

        # ----------------------------------------------------------------
        # Learned Spatial Attention (Channel Attention / SE-style)
        # ----------------------------------------------------------------
        # Merangkum seluruh konteks spatial → vektor channel (B, 128) →
        # MLP kecil → sigmoid → bobot per-channel → scaling.
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
        # Konvolusi 1×1×1 menghasilkan skor kepentingan per-frame.
        # Skor di-softmax sepanjang dimensi waktu.
        if self.use_temporal_attention:
            self.temporal_attention = nn.Sequential(
                nn.Conv3d(128, 1, kernel_size=(1, 1, 1), bias=True),
            )

        # ----------------------------------------------------------------
        # Conv Block 1 — Ekstraksi fitur level rendah
        # ----------------------------------------------------------------
        # Input : (B, 3,  64, 33, 1)
        # Output: (B, 32, 64, 16, 1) — MaxPool hanya pada dimensi landmark (33→16)
        # Dimensi waktu (64) TIDAK diperkecil agar resolusi temporal terjaga.
        self.conv_block_1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 1), padding=(1, 1, 0), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 1)),  # Hanya landmark: 33 → 16
        )

        # ----------------------------------------------------------------
        # Conv Block 2 — Ekstraksi fitur level menengah
        # ----------------------------------------------------------------
        # Input : (B, 32, 64, 16, 1)
        # Output: (B, 64, 32,  8, 1) — MaxPool pada waktu (64→32) DAN landmark (16→8)
        # Waktu diperkecil setengah (64→32) agar beban komputasi terjaga.
        self.conv_block_2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 1), padding=(1, 1, 0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 1)),  # Waktu: 64→32, Landmark: 16→8
        )

        # ----------------------------------------------------------------
        # Conv Block 3 — Ekstraksi fitur level tinggi
        # ----------------------------------------------------------------
        # Input : (B, 64,  32, 8, 1)
        # Output: (B, 128, 32, 8, 1) — Tidak ada pooling, hanya perluasan channel
        self.conv_block_3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 1), padding=(1, 1, 0), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        # ----------------------------------------------------------------
        # Landmark Pooling (menggantikan Global Average Pooling v1)
        # ----------------------------------------------------------------
        # Pool HANYA pada dimensi landmark dan Width semu → (B, 128, T', 1, 1).
        # Dimensi temporal T' (=32 setelah Block 2) TETAP dipertahankan.
        self.landmark_pool = nn.AdaptiveAvgPool3d(output_size=(None, 1, 1))

        # ----------------------------------------------------------------
        # Classifier Head (per-frame)
        # ----------------------------------------------------------------
        # Diterapkan pada setiap frame secara independen.
        # Input per frame  : vektor 128-dimensi
        # Output per frame : logit mentah untuk num_classes kelas
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alur forward pass AttentiveSkel-3D v2 (per-frame).

        Args:
            x (torch.Tensor): Input shape (B, T=64, L=33, C=3).

        Returns:
            torch.Tensor: Logit per-frame shape (B, T=64, num_classes=2).
        """
        # ------------------------------------------------------------------
        # Langkah 1: Reshape ke format Conv3d (B, C, T, L, W)
        # ------------------------------------------------------------------
        # (B, 64, 33, 3) → permute → (B, 3, 64, 33) → unsqueeze → (B, 3, 64, 33, 1)
        x = x.permute(0, 3, 1, 2)   # (B, 3, 64, 33)
        x = x.unsqueeze(-1)          # (B, 3, 64, 33, 1)

        # ------------------------------------------------------------------
        # Langkah 2: Biomechanical Spatial Prior (BSP)
        # ------------------------------------------------------------------
        if self.use_spatial_prior:
            # Broadcast (1, 1, 1, 33, 1) × (B, 3, 64, 33, 1)
            bsp_weights = torch.sigmoid(self.biomechanical_spatial_prior)
            x = x * bsp_weights

        # ------------------------------------------------------------------
        # Langkah 3: Ekstraksi fitur bertahap (3 Conv Blocks)
        # ------------------------------------------------------------------
        x = self.conv_block_1(x)     # → (B, 32, 64, 16, 1)
        x = self.conv_block_2(x)     # → (B, 64, 32,  8, 1)
        x = self.conv_block_3(x)     # → (B, 128, 32, 8, 1)

        # ------------------------------------------------------------------
        # Langkah 4: Learned Spatial Attention (Channel Attention SE-style)
        # ------------------------------------------------------------------
        if self.use_learned_spatial:
            # GAP di seluruh dimensi spasial & temporal → (B, 128)
            gap_feat   = x.mean(dim=[2, 3, 4])
            ch_weights = self.learned_spatial_attention(gap_feat)  # (B, 128) sigmoid
            # Broadcast kembali: (B, 128, 1, 1, 1) × (B, 128, T', L', 1)
            x = x * ch_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # ------------------------------------------------------------------
        # Langkah 5: Temporal Attention
        # ------------------------------------------------------------------
        if self.use_temporal_attention:
            scores   = self.temporal_attention(x)            # (B, 1, T', L', 1)
            scores   = scores.mean(dim=[3, 4], keepdim=True) # (B, 1, T', 1,  1)
            t_weights = torch.softmax(scores, dim=2)          # Normalisasi per waktu
            x = x * t_weights                                 # (B, 128, T', L', 1)

        # ------------------------------------------------------------------
        # Langkah 6: Landmark Pooling — pool hanya dimensi L dan W
        # ------------------------------------------------------------------
        # (B, 128, 32, 8, 1) → (B, 128, 32, 1, 1)
        x = self.landmark_pool(x)

        # ------------------------------------------------------------------
        # Langkah 7: Upsample T' (=32) kembali ke T=64
        # ------------------------------------------------------------------
        # Menggunakan interpolasi linear 1D pada dimensi waktu agar representasi
        # tiap frame ke-i pada output selaras dengan frame ke-i pada input.
        # Bentuk kerja: squeeze W → (B, 128, 32, 1) → interpolate → (B, 128, 64, 1)
        x = x.squeeze(-1)            # (B, 128, 32, 1)
        x = F.interpolate(
            x,
            size=(64, 1),            # Target: T=64 frame, L=1
            mode="bilinear",
            align_corners=False,
        )                            # (B, 128, 64, 1)
        x = x.squeeze(-1)            # (B, 128, 64)

        # ------------------------------------------------------------------
        # Langkah 8: Permute → (B, 64, 128) lalu apply Classifier per-frame
        # ------------------------------------------------------------------
        x = x.permute(0, 2, 1)       # (B, 64, 128)

        # nn.Linear diterapkan pada dimensi terakhir (128) secara otomatis
        # untuk setiap posisi frame → output (B, 64, num_classes)
        x = self.classifier(x)       # (B, 64, num_classes)

        return x


# ============================================================
# Utilitas: hitung jumlah total parameter model
# ============================================================
def count_parameters(model: nn.Module) -> int:
    """Mengembalikan jumlah parameter yang dapat dilatih."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
