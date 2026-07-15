# src/analysis/biomechanical_validator_audit.py
#
# Modul audit dan dokumentasi matematis Biomechanical Validator.
#
# Tujuan:
#   - Menjelaskan rumus matematis setiap aturan biomekanik
#   - Implementasi stabil untuk sudut 3D dengan epsilon safety
#   - Analisis scale invariance: membuktikan bahwa sudut/rasio tidak bergantung pada skala
#   - Eksperimen validasi dengan koordinat contoh
#   - Per-frame scoring untuk video nyata
#
# Struktur:
#   1. BiomechanicalRulesDocumentation — katalog lengkap setiap aturan
#   2. StableAngleCalculator — fungsi sudut 3D yang robust
#   3. ScaleInvarianceAnalyzer — eksperimen scaling dengan bukti matematis
#   4. PerFrameAnalyzer — hitung skor per-frame untuk video nyata

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("biomech_audit")


# ── Dokumentasi Aturan Biomekanik ──────────────────────────────────────────

@dataclass
class BiomechanicalRule:
    """Dokumentasi satu aturan biomekanik."""
    
    exercise: str                    # Nama gerakan: "Squat", "Bench Press", "Deadlift"
    rule_name: str                   # Nama aturan detail
    criteria_number: int             # Kriteria ke-berapa dalam gerakan
    
    landmark_a_idx: int
    landmark_a_name: str
    
    landmark_b_idx: int              # Vertex / pivot point
    landmark_b_name: str
    
    landmark_c_idx: Optional[int] = None
    landmark_c_name: Optional[str] = None
    
    # Untuk aturan yang melibatkan jarak (rasio)
    landmark_ref_idx: Optional[int] = None
    landmark_ref_name: Optional[str] = None
    
    rule_type: str = "angle"  # "angle" atau "ratio" atau "inclination"
    threshold_value: float = 0.0
    threshold_condition: str = ""  # "<=" atau ">=" atau "between"
    
    formula_latex: str = ""
    explanation_id: str = ""


class BiomechanicalRulesDocumentation:
    """Katalog lengkap aturan biomekanik dengan dokumentasi matematis."""
    
    LANDMARK_NAMES = {
        11: "Left Shoulder", 12: "Right Shoulder",
        13: "Left Elbow", 14: "Right Elbow",
        15: "Left Wrist", 16: "Right Wrist",
        23: "Left Hip", 24: "Right Hip",
        25: "Left Knee", 26: "Right Knee",
        27: "Left Ankle", 28: "Right Ankle",
    }
    
    # MediaPipe BlazePose reference frame:
    # X-axis: horizontal, left-right (positive = right)
    # Y-axis: vertical, top-bottom (positive = down in image coordinate)
    # Z-axis: depth (positive = away from camera)
    #
    # Setelah preprocessing:
    # - Mid-hip set to origin (0, 0, 0)
    # - Semua koordinat diScale dengan panjang torso
    # - Hal ini membuat sudut dan rasio SCALE INVARIANT
    
    @staticmethod
    def get_squat_rules() -> list[BiomechanicalRule]:
        """Dokumentasi aturan Squat (3 kriteria)."""
        return [
            BiomechanicalRule(
                exercise="Squat",
                rule_name="Criterion 1: Knee Valgus Prevention",
                criteria_number=1,
                landmark_a_idx=25,
                landmark_a_name="Left Knee",
                landmark_b_idx=27,
                landmark_b_name="Left Ankle",
                landmark_c_idx=26,
                landmark_c_name="Right Knee",
                landmark_ref_idx=28,
                landmark_ref_name="Right Ankle",
                rule_type="ratio",
                threshold_value=0.85,
                threshold_condition=">=",
                formula_latex=r"$\frac{\text{knee\_width}}{\text{ankle\_width}} \geq 0.85$",
                explanation_id="squat_valgus_ratio",
            ),
            BiomechanicalRule(
                exercise="Squat",
                rule_name="Criterion 2: Insufficient Squat Depth (Hip Flexion)",
                criteria_number=2,
                landmark_a_idx=11,
                landmark_a_name="Left Shoulder",
                landmark_b_idx=23,
                landmark_b_name="Left Hip",
                landmark_c_idx=25,
                landmark_c_name="Left Knee",
                rule_type="angle",
                threshold_value=137.0,
                threshold_condition="<=",
                formula_latex=r"$\theta_{\text{hip}} = \arccos\left(\frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}||\vec{BC}|}\right) \leq 137°$",
                explanation_id="squat_hip_flexion",
            ),
            BiomechanicalRule(
                exercise="Squat",
                rule_name="Criterion 3: Squat Depth via Knee Flexion",
                criteria_number=3,
                landmark_a_idx=23,
                landmark_a_name="Left Hip",
                landmark_b_idx=25,
                landmark_b_name="Left Knee",
                landmark_c_idx=27,
                landmark_c_name="Left Ankle",
                rule_type="angle",
                threshold_value=100.0,
                threshold_condition="<=",
                formula_latex=r"$\theta_{\text{knee}} = \arccos\left(\frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}||\vec{BC}|}\right) \leq 100°$",
                explanation_id="squat_knee_depth",
            ),
        ]
    
    @staticmethod
    def get_benchpress_rules() -> list[BiomechanicalRule]:
        """Dokumentasi aturan Bench Press (1 kriteria)."""
        return [
            BiomechanicalRule(
                exercise="Bench Press",
                rule_name="Criterion 1: Full Range of Motion (Elbow)",
                criteria_number=1,
                landmark_a_idx=11,
                landmark_a_name="Left Shoulder",
                landmark_b_idx=13,
                landmark_b_name="Left Elbow",
                landmark_c_idx=15,
                landmark_c_name="Left Wrist",
                rule_type="angle",
                threshold_value=85.0,
                threshold_condition="<=",
                formula_latex=r"$\theta_{\text{elbow}} = \arccos\left(\frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}||\vec{BC}|}\right) \leq 85°$",
                explanation_id="benchpress_elbow_rom",
            ),
        ]
    
    @staticmethod
    def get_deadlift_rules() -> list[BiomechanicalRule]:
        """Dokumentasi aturan Deadlift (1 kriteria dengan 2 sub-kondisi)."""
        return [
            BiomechanicalRule(
                exercise="Deadlift",
                rule_name="Criterion 1: Spine Inclination (Hip Hinge Pattern) — Min",
                criteria_number=1,
                landmark_a_idx=11,
                landmark_a_name="Left Shoulder (mid-shoulder)",
                landmark_b_idx=23,
                landmark_b_name="Left Hip (mid-hip, origin)",
                landmark_c_idx=None,
                landmark_c_name=None,
                rule_type="inclination",
                threshold_value=20.0,
                threshold_condition=">=",
                formula_latex=r"$\theta_{\text{inclination}} = \arccos\left(\frac{\vec{spine} \cdot \vec{vertical}}{|\vec{spine}||\vec{vertical}|}\right) \geq 20°$",
                explanation_id="deadlift_spine_min",
            ),
            BiomechanicalRule(
                exercise="Deadlift",
                rule_name="Criterion 1: Spine Inclination (Hip Hinge Pattern) — Max",
                criteria_number=1,
                landmark_a_idx=11,
                landmark_a_name="Left Shoulder (mid-shoulder)",
                landmark_b_idx=23,
                landmark_b_name="Left Hip (mid-hip, origin)",
                landmark_c_idx=None,
                landmark_c_name=None,
                rule_type="inclination",
                threshold_value=60.0,
                threshold_condition="<=",
                formula_latex=r"$\theta_{\text{inclination}} \leq 60°$",
                explanation_id="deadlift_spine_max",
            ),
        ]
    
    @classmethod
    def get_all_rules(cls) -> list[BiomechanicalRule]:
        """Kembalikan semua aturan dari ketiga gerakan."""
        return (
            cls.get_squat_rules()
            + cls.get_benchpress_rules()
            + cls.get_deadlift_rules()
        )


class StableAngleCalculator:
    """
    Implementasi stabil untuk menghitung sudut 3D dengan safety epsilon.
    
    Rumus matematis:
        Diberikan tiga titik A, B, C dalam ruang 3D, sudut yang dibentuk di vertex B adalah:
        
        θ = arccos( (BA · BC) / (|BA| × |BC|) )
        
        Di mana:
        - BA = A - B (vektor dari vertex ke titik A)
        - BC = C - B (vektor dari vertex ke titik C)
        - BA · BC = dot product
        - |BA|, |BC| = norma Euclidean
        
    Stabilitas numerik:
        1. Epsilon untuk pembagian nol: jika norm < 1e-8, return 180° (titik identik)
        2. Clipping cosine ke [-1, 1]: mencegah floating point error pada arccos
        3. Hasil dalam degree, bukan radian
    
    Invariansi skala:
        Sudut NOT bergantung pada skala karena:
        - cos(θ) = (BA · BC) / (|BA| × |BC|)
        - Jika scaled: BA' = k×BA, BC' = k×BC
        - cos(θ') = (k×BA · k×BC) / (|k×BA| × |k×BC|)
        -          = (k²×(BA·BC)) / (k×|BA| × k×|BC|)
        -          = (BA · BC) / (|BA| × |BC|)
        -          = cos(θ) ✓
        - Maka θ' = θ (sudut invariant terhadap scaling)
    """
    
    EPSILON = 1e-8  # Threshold untuk numerik zero
    
    @staticmethod
    def calculate_angle(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
    ) -> float:
        """
        Hitung sudut 3D (derajat) dengan vertex di B.
        
        Args:
            a: Koordinat 3D (shape (3,))
            b: Koordinat 3D vertex (shape (3,))
            c: Koordinat 3D (shape (3,))
        
        Returns:
            float: Sudut dalam derajat [0, 180]
        """
        ba = a - b
        bc = c - b
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba < StableAngleCalculator.EPSILON or norm_bc < StableAngleCalculator.EPSILON:
            return 180.0
        
        cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return float(np.degrees(np.arccos(cos_angle)))
    
    @staticmethod
    def calculate_angle_batch(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
    ) -> np.ndarray:
        """
        Hitung sudut batch (vectorized) untuk F frame.
        
        Args:
            a: Array (F, 3)
            b: Array (F, 3) — vertex untuk setiap frame
            c: Array (F, 3)
        
        Returns:
            Array (F,) — sudut per frame
        """
        ba = a - b  # (F, 3)
        bc = c - b  # (F, 3)
        
        norm_ba = np.linalg.norm(ba, axis=1)  # (F,)
        norm_bc = np.linalg.norm(bc, axis=1)  # (F,)
        
        dot_prod = np.einsum("fi,fi->f", ba, bc)  # (F,)
        
        valid = (norm_ba > StableAngleCalculator.EPSILON) & (norm_bc > StableAngleCalculator.EPSILON)
        
        cos_angles = np.ones(len(a))  # default cos=1 → 0 derajat
        cos_angles[valid] = dot_prod[valid] / (norm_ba[valid] * norm_bc[valid])
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        
        return np.degrees(np.arccos(cos_angles))


class ScaleInvarianceAnalyzer:
    """
    Analisis dan pembuktian scale invariance untuk sudut dan rasio.
    
    Tesis:
        Sudut dan rasio adalah besaran yang SCALE INVARIANT. Ini berarti bahwa
        jika semua koordinat dikalikan dengan faktor skala k > 0, nilai sudut
        dan rasio tetap sama.
    
    Implikasi praktis:
        Perbedaan tinggi badan dan panjang tulang tidak mempengaruhi hasil validasi
        biomekanik, karena yang diukur adalah PROPORSI dan SUDUT, bukan nilai absolut.
    """
    
    @staticmethod
    def scale_coordinates(
        tensor: np.ndarray,
        scale_factor: float,
    ) -> np.ndarray:
        """
        Kalikan semua koordinat dengan faktor skala.
        
        Args:
            tensor: Array (F, 33, 3)
            scale_factor: Faktor skala (mis. 0.5, 1.0, 2.0)
        
        Returns:
            Array (F, 33, 3) dengan koordinat scaled
        """
        return tensor * scale_factor
    
    @staticmethod
    def test_angle_scale_invariance(
        tensor: np.ndarray,
        landmark_a_idx: int,
        landmark_b_idx: int,
        landmark_c_idx: int,
        scale_factors: list[float],
    ) -> pd.DataFrame:
        """
        Buktikan bahwa sudut tidak berubah saat scaling.
        
        Args:
            tensor: Array (F, 33, 3)
            landmark_a_idx, landmark_b_idx, landmark_c_idx: Indeks landmark
            scale_factors: List faktor skala untuk diuji [0.5, 1.0, 2.0, ...]
        
        Returns:
            DataFrame dengan perbandingan sudut pada berbagai skala
        """
        results = []
        
        calc = StableAngleCalculator()
        
        for scale in scale_factors:
            scaled = ScaleInvarianceAnalyzer.scale_coordinates(tensor, scale)
            
            a = scaled[:, landmark_a_idx, :]
            b = scaled[:, landmark_b_idx, :]
            c = scaled[:, landmark_c_idx, :]
            
            angles = calc.calculate_angle_batch(a, b, c)
            
            for frame_idx, angle in enumerate(angles):
                results.append({
                    "scale_factor": scale,
                    "frame_index": frame_idx,
                    "angle_deg": round(angle, 6),
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def test_ratio_scale_invariance(
        tensor: np.ndarray,
        landmark_a_idx: int,
        landmark_b_idx: int,
        scale_factors: list[float],
    ) -> pd.DataFrame:
        """
        Buktikan bahwa rasio jarak tidak berubah saat scaling.
        
        Args:
            tensor: Array (F, 33, 3)
            landmark_a_idx, landmark_b_idx: Indeks landmark untuk dua pasangan
            scale_factors: List faktor skala untuk diuji
        
        Returns:
            DataFrame dengan perbandingan rasio pada berbagai skala
        """
        results = []
        
        for scale in scale_factors:
            scaled = ScaleInvarianceAnalyzer.scale_coordinates(tensor, scale)
            
            a = scaled[:, landmark_a_idx, :]
            b = scaled[:, landmark_b_idx, :]
            
            distances = np.linalg.norm(a - b, axis=1)
            
            for frame_idx, dist in enumerate(distances):
                results.append({
                    "scale_factor": scale,
                    "frame_index": frame_idx,
                    "distance": round(dist, 6),
                })
        
        return pd.DataFrame(results)


class PerFrameAnalyzer:
    """
    Analisis per-frame untuk menghitung skor validasi pada setiap frame.
    Berguna untuk melacak kapan subjek memenuhi kriteria selama gerakan.
    """
    
    @staticmethod
    def analyze_squat_per_frame(
        tensor: np.ndarray,
    ) -> pd.DataFrame:
        """
        Hitung metrik biomekanik Squat untuk setiap frame.
        
        Returns:
            DataFrame dengan kolom:
            - frame_index
            - angle_knee_left, angle_knee_right (sudut lutut)
            - angle_hip_left, angle_hip_right (sudut pinggul)
            - valgus_ratio
            - criteria_1_pass (knee valgus)
            - criteria_2_pass (hip depth)
            - criteria_3_pass (knee depth)
            - all_criteria_pass
        """
        IDX_L_SHOULDER, IDX_R_SHOULDER = 11, 12
        IDX_L_ELBOW, IDX_R_ELBOW = 13, 14
        IDX_L_HIP, IDX_R_HIP = 23, 24
        IDX_L_KNEE, IDX_R_KNEE = 25, 26
        IDX_L_ANKLE, IDX_R_ANKLE = 27, 28
        
        calc = StableAngleCalculator()
        F = tensor.shape[0]
        
        records = []
        
        for f in range(F):
            # Sudut lutut
            angle_knee_left = calc.calculate_angle(
                tensor[f, IDX_L_HIP, :],
                tensor[f, IDX_L_KNEE, :],
                tensor[f, IDX_L_ANKLE, :],
            )
            angle_knee_right = calc.calculate_angle(
                tensor[f, IDX_R_HIP, :],
                tensor[f, IDX_R_KNEE, :],
                tensor[f, IDX_R_ANKLE, :],
            )
            angle_knee_avg = (angle_knee_left + angle_knee_right) / 2.0
            
            # Sudut pinggul
            angle_hip_left = calc.calculate_angle(
                tensor[f, IDX_L_SHOULDER, :],
                tensor[f, IDX_L_HIP, :],
                tensor[f, IDX_L_KNEE, :],
            )
            angle_hip_right = calc.calculate_angle(
                tensor[f, IDX_R_SHOULDER, :],
                tensor[f, IDX_R_HIP, :],
                tensor[f, IDX_R_KNEE, :],
            )
            angle_hip_avg = (angle_hip_left + angle_hip_right) / 2.0
            
            # Rasio lebar lutut/kaki
            knee_width = abs(tensor[f, IDX_L_KNEE, 0] - tensor[f, IDX_R_KNEE, 0])
            ankle_width = abs(tensor[f, IDX_L_ANKLE, 0] - tensor[f, IDX_R_ANKLE, 0])
            valgus_ratio = knee_width / max(ankle_width, 1e-8)
            
            # Evaluasi kriteria
            crit_1 = valgus_ratio >= 0.85
            crit_2 = angle_hip_avg <= 137.0
            crit_3 = angle_knee_avg <= 100.0
            all_pass = crit_1 and crit_2 and crit_3
            
            records.append({
                "frame_index": f,
                "angle_knee_left": round(angle_knee_left, 2),
                "angle_knee_right": round(angle_knee_right, 2),
                "angle_knee_avg": round(angle_knee_avg, 2),
                "angle_hip_left": round(angle_hip_left, 2),
                "angle_hip_right": round(angle_hip_right, 2),
                "angle_hip_avg": round(angle_hip_avg, 2),
                "valgus_ratio": round(valgus_ratio, 4),
                "criteria_1_pass": crit_1,
                "criteria_2_pass": crit_2,
                "criteria_3_pass": crit_3,
                "all_criteria_pass": all_pass,
                "landmarks": "L/R Shoulder, Hip, Knee, Ankle",
            })
        
        return pd.DataFrame(records)
    
    @staticmethod
    def analyze_benchpress_per_frame(
        tensor: np.ndarray,
    ) -> pd.DataFrame:
        """
        Hitung metrik biomekanik Bench Press untuk setiap frame.
        """
        IDX_L_SHOULDER, IDX_R_SHOULDER = 11, 12
        IDX_L_ELBOW, IDX_R_ELBOW = 13, 14
        IDX_L_WRIST, IDX_R_WRIST = 15, 16
        
        calc = StableAngleCalculator()
        F = tensor.shape[0]
        
        records = []
        
        for f in range(F):
            angle_elbow_left = calc.calculate_angle(
                tensor[f, IDX_L_SHOULDER, :],
                tensor[f, IDX_L_ELBOW, :],
                tensor[f, IDX_L_WRIST, :],
            )
            angle_elbow_right = calc.calculate_angle(
                tensor[f, IDX_R_SHOULDER, :],
                tensor[f, IDX_R_ELBOW, :],
                tensor[f, IDX_R_WRIST, :],
            )
            angle_elbow_avg = (angle_elbow_left + angle_elbow_right) / 2.0
            
            crit_1 = angle_elbow_avg <= 85.0
            
            records.append({
                "frame_index": f,
                "angle_elbow_left": round(angle_elbow_left, 2),
                "angle_elbow_right": round(angle_elbow_right, 2),
                "angle_elbow_avg": round(angle_elbow_avg, 2),
                "criteria_1_pass": crit_1,
                "landmarks": "L/R Shoulder, Elbow, Wrist",
            })
        
        return pd.DataFrame(records)
    
    @staticmethod
    def analyze_deadlift_per_frame(
        tensor: np.ndarray,
    ) -> pd.DataFrame:
        """
        Hitung metrik biomekanik Deadlift untuk setiap frame.
        """
        IDX_L_SHOULDER, IDX_R_SHOULDER = 11, 12
        IDX_L_HIP, IDX_R_HIP = 23, 24
        
        F = tensor.shape[0]
        
        records = []
        
        for f in range(F):
            # Mid-shoulder dan mid-hip
            mid_shoulder = (
                tensor[f, IDX_L_SHOULDER, :] + tensor[f, IDX_R_SHOULDER, :]
            ) / 2.0
            mid_hip = (
                tensor[f, IDX_L_HIP, :] + tensor[f, IDX_R_HIP, :]
            ) / 2.0
            
            # Spine vector (mid-hip assumed to be origin after normalization)
            spine_vec = mid_shoulder - mid_hip
            
            # Vertical vector (Y down in MediaPipe, so Y negative = up)
            vertical = np.array([0.0, -1.0, 0.0])
            
            norm_spine = np.linalg.norm(spine_vec)
            norm_vert = np.linalg.norm(vertical)
            
            if norm_spine > 1e-8 and norm_vert > 1e-8:
                cos_angle = np.dot(spine_vec, vertical) / (norm_spine * norm_vert)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                inclination = float(np.degrees(np.arccos(cos_angle)))
            else:
                inclination = 0.0
            
            crit_1_min = inclination >= 20.0
            crit_1_max = inclination <= 60.0
            all_pass = crit_1_min and crit_1_max
            
            records.append({
                "frame_index": f,
                "spine_inclination_deg": round(inclination, 2),
                "criteria_1_min_pass": crit_1_min,
                "criteria_1_max_pass": crit_1_max,
                "all_criteria_pass": all_pass,
                "landmarks": "L/R Shoulder, L/R Hip",
            })
        
        return pd.DataFrame(records)


def export_rules_documentation() -> pd.DataFrame:
    """
    Export dokumentasi lengkap semua aturan sebagai DataFrame untuk CSV.
    """
    rules = BiomechanicalRulesDocumentation.get_all_rules()
    
    records = []
    for rule in rules:
        records.append({
            "exercise": rule.exercise,
            "rule_name": rule.rule_name,
            "criteria_number": rule.criteria_number,
            "landmark_a_idx": rule.landmark_a_idx,
            "landmark_a_name": rule.landmark_a_name,
            "landmark_b_idx": rule.landmark_b_idx,
            "landmark_b_name": rule.landmark_b_name,
            "landmark_c_idx": rule.landmark_c_idx if rule.landmark_c_idx else "N/A",
            "landmark_c_name": rule.landmark_c_name if rule.landmark_c_name else "N/A",
            "rule_type": rule.rule_type,
            "threshold_value": rule.threshold_value,
            "threshold_condition": rule.threshold_condition,
            "formula_latex": rule.formula_latex,
            "explanation_id": rule.explanation_id,
        })
    
    return pd.DataFrame(records)
