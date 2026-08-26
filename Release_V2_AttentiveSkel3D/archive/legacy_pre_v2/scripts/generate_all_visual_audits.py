"""
Batch script to generate MediaPipe + Biomechanics visual audits for all raw videos.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import traceback

# Setup paths to ensure we can import src modules
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.analysis.cvat_mediapipe_pilot import (
    extract_mediapipe_landmarks,
    CVAT_12_JOINTS,
    LANDMARK_INDEX
)
from src.analysis.biomechanical_validator_audit import PerFrameAnalyzer

RAW_DATA_DIR = project_root / 'data' / 'raw'
OUTPUT_DIR = project_root / 'results' / 'dataset_visual_audit'

SKELETON_EDGES_12 = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
]

def draw_skeleton(frame_bgr, pose_frame, w, h):
    overlay = frame_bgr.copy()
    joints = {}
    for jn in CVAT_12_JOINTS:
        li = LANDMARK_INDEX[jn]
        x_norm, y_norm = float(pose_frame[li, 0]), float(pose_frame[li, 1])
        if np.isfinite(x_norm) and np.isfinite(y_norm):
            joints[jn] = (x_norm * w, y_norm * h)
            
    # Edges
    edge_color = (30, 200, 80) # Hijau
    joint_color = (50, 220, 100)
    for jn_a, jn_b in SKELETON_EDGES_12:
        if jn_a in joints and jn_b in joints:
            pt_a = (int(round(joints[jn_a][0])), int(round(joints[jn_a][1])))
            pt_b = (int(round(joints[jn_b][0])), int(round(joints[jn_b][1])))
            cv2.line(overlay, pt_a, pt_b, edge_color, 3, cv2.LINE_AA)
            
    # Points
    for jn, (x, y) in joints.items():
        pt = (int(round(x)), int(round(y)))
        cv2.circle(overlay, pt, 7, joint_color, -1, cv2.LINE_AA)
        cv2.circle(overlay, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)
        
    return overlay

def process_video(video_path: Path):
    try:
        exercise_type = video_path.parent.name
        out_subdir = OUTPUT_DIR / exercise_type
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_file = out_subdir / video_path.name
        
        if out_file.exists():
            return (video_path.name, "SKIPPED")
            
        # 1. Extract MediaPipe
        # Suppress MediaPipe output by redirecting stdout/stderr inside the process?
        # Actually extract_mediapipe_landmarks is quite clean.
        bundle = extract_mediapipe_landmarks(video_path)
        
        if len(bundle.frames_bgr) == 0:
            return (video_path.name, "ERROR: No frames extracted")
            
        # 2. Analyze Biomechanics (Only X, Y, Z - discard visibility for math)
        analyzer = PerFrameAnalyzer()
        pose_xyz = bundle.pose_array[..., :3]
        if exercise_type == "Squat":
            per_frame_df = analyzer.analyze_squat_per_frame(pose_xyz)
        elif exercise_type == "BenchPress":
            per_frame_df = analyzer.analyze_benchpress_per_frame(pose_xyz)
        elif exercise_type == "Deadlift":
            per_frame_df = analyzer.analyze_deadlift_per_frame(pose_xyz)
        else:
            return (video_path.name, f"ERROR: Unknown exercise {exercise_type}")
            
        # 3. Generate Video
        w, h = bundle.frame_width, bundle.frame_height
        info_h = 160
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_file), fourcc, bundle.fps, (w, h + info_h))
        
        for fidx in range(len(bundle.frames_bgr)):
            frame_bgr = bundle.frames_bgr[fidx]
            pose_frame = bundle.pose_array[fidx]
            
            # Draw skeleton
            overlay = draw_skeleton(frame_bgr, pose_frame, w, h)
            
            # Info bar
            info_bar = np.zeros((info_h, w, 3), dtype=np.uint8)
            info_bar[:] = (30, 30, 30)
            
            row = per_frame_df.loc[per_frame_df['frame_index'] == fidx]
            if not row.empty:
                row = row.iloc[0]
                
                # Format string and evaluate label depends on exercise type
                if exercise_type == "Squat":
                    stats = f"Knee={row['angle_knee_avg']:.1f} Hip={row['angle_hip_avg']:.1f} Valgus={row['valgus_ratio']:.2f}"
                    is_valid = row.get('all_criteria_pass', False)
                    status_text = "Squat Valid" if is_valid else "Squat Invalid"
                elif exercise_type == "BenchPress":
                    elbow = row.get('angle_elbow_avg', row.get('angle_elbow_left', 0.0))
                    stats = f"Elbow={elbow:.1f}"
                    is_valid = row.get('criteria_1_pass', False)
                    status_text = "BenchPress Valid" if is_valid else "BenchPress Invalid"
                else: # Deadlift
                    stats = f"BackIncl={row['spine_inclination_deg']:.1f}"
                    is_valid = row.get('all_criteria_pass', False)
                    status_text = "Deadlift Valid" if is_valid else "Deadlift Invalid"
                    
                label_val = 0 if is_valid else 1
            else:
                label_val = -1
                stats = "N/A"
                status_text = "N/A"
                
            if label_val == 0:
                label_color = (87, 139, 46) # Green
                label_str = 'BENAR'
            elif label_val == 1:
                label_color = (39, 39, 214) # Red
                label_str = 'SALAH'
            else:
                label_color = (128, 128, 128)
                label_str = 'N/A'
                
            cv2.putText(info_bar, f"Frame {fidx} | {exercise_type} | Label: {label_str}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3, cv2.LINE_AA)
            cv2.putText(info_bar, stats, (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(info_bar, status_text[:90], (20, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
            
            canvas = np.vstack([overlay, info_bar])
            writer.write(canvas)
            
        writer.release()
        return (video_path.name, "SUCCESS")
        
    except Exception as e:
        err = traceback.format_exc()
        return (video_path.name, f"ERROR: {str(e)}\n{err}")

def main():
    videos = list(RAW_DATA_DIR.rglob("*.mp4"))
    print(f"Ditemukan {len(videos)} video di {RAW_DATA_DIR}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run multiprocessing
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    workers = 4
    print(f"Menggunakan {workers} worker process.")
    
    with multiprocessing.Pool(workers) as pool:
        for fname, status in tqdm(pool.imap_unordered(process_video, videos), total=len(videos)):
            if status == "SUCCESS":
                success_count += 1
            elif status == "SKIPPED":
                skipped_count += 1
            else:
                error_count += 1
                print(f"\n[!] Failed {fname}: {status}")
                
    print("\n--- Selesai ---")
    print(f"Berhasil : {success_count}")
    print(f"Dilewati : {skipped_count} (sudah ada)")
    print(f"Gagal    : {error_count}")

if __name__ == "__main__":
    main()
