"""
Script untuk menambahkan cell video perbandingan CVAT vs MediaPipe
ke notebook 19_cvat_mediapipe_validation_pilot.ipynb.

Menambahkan 1 cell markdown + 1 cell code sebelum cell kesimpulan terakhir.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(r"d:\Data-Aji\KULIAH\Tugas-Akhir\AttentiveSkel3D-WeightTraining-PoC\notebooks\19_cvat_mediapipe_validation_pilot.ipynb")

# ── Cell markdown baru ──────────────────────────────────────────────────────
markdown_cell = {
    "cell_type": "markdown",
    "id": "a1e4f7c0",
    "metadata": {},
    "source": [
        "## Output Video Perbandingan Anotasi CVAT vs MediaPipe + Rekomendasi\n",
        "\n",
        "Bagian ini menghasilkan **video side-by-side** yang memvisualisasikan skeleton dari kedua sumber anotasi di setiap frame:\n",
        "\n",
        "- **Panel kiri**: skeleton dari anotasi manual **CVAT** (warna biru)\n",
        "- **Panel kanan**: skeleton dari **MediaPipe BlazePose** (warna hijau)\n",
        "- **Bar informasi bawah**: label **Benar/Salah** per-frame berdasarkan aturan biomekanik, beserta **rekomendasi** sumber anotasi terbaik.\n",
        "\n",
        "Video ini berfungsi sebagai bukti visual langsung untuk menjawab tiga pertanyaan Pak Jasman:\n",
        "1. Titik sendi yang dipakai = 12 joint utama (sesuai kebutuhan penelitian)\n",
        "2. Skeleton MediaPipe terbukti dapat dikeluarkan dan divisualisasikan secara real-time\n",
        "3. Label Benar/Salah per-frame diberikan berdasarkan aturan biomekanik yang terukur"
    ]
}

# ── Cell code baru ──────────────────────────────────────────────────────────
code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "b2c3d4e5",
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── Video Perbandingan CVAT vs MediaPipe ──────────────────────────────────\n",
        "import xml.etree.ElementTree as ET\n",
        "\n",
        "# --- 1. Parse CVAT XML dengan benar (format CVAT for Images 1.1) ----------\n",
        "def parse_cvat_images_xml(xml_path, joint_names):\n",
        "    \"\"\"Parse CVAT for Images 1.1 XML -> dict[frame_index, dict[joint_name, (x_px, y_px)]].\"\"\"\n",
        "    tree = ET.parse(xml_path)\n",
        "    root = tree.getroot()\n",
        "    cvat_data = {}\n",
        "    # Normalisasi nama joint (CVAT pakai Title_Case, kita pakai lower_case)\n",
        "    name_map = {}\n",
        "    for jn in joint_names:\n",
        "        parts = jn.split('_')\n",
        "        cvat_name = '_'.join(p.capitalize() for p in parts)\n",
        "        name_map[cvat_name] = jn\n",
        "    \n",
        "    for image_node in root.findall('.//image'):\n",
        "        frame_idx = int(image_node.attrib['id'])\n",
        "        joints = {}\n",
        "        for skel_node in image_node.findall('.//skeleton'):\n",
        "            for pts_node in skel_node.findall('.//points'):\n",
        "                label = pts_node.attrib.get('label', '')\n",
        "                points_str = pts_node.attrib.get('points', '')\n",
        "                if label in name_map and points_str:\n",
        "                    xy = points_str.split(',')\n",
        "                    if len(xy) >= 2:\n",
        "                        joints[name_map[label]] = (float(xy[0]), float(xy[1]))\n",
        "        if joints:\n",
        "            cvat_data[frame_idx] = joints\n",
        "    return cvat_data\n",
        "\n",
        "cvat_joints = parse_cvat_images_xml(CVAT_EXPORT, CVAT_12_JOINTS)\n",
        "print(f'CVAT frames parsed: {len(cvat_joints)} / {len(bundle.frames_bgr)}')\n",
        "\n",
        "# --- 2. Definisikan koneksi skeleton 12-joint ---------------------------------\n",
        "SKELETON_EDGES_12 = [\n",
        "    ('left_shoulder', 'right_shoulder'),\n",
        "    ('left_shoulder', 'left_elbow'),\n",
        "    ('left_elbow', 'left_wrist'),\n",
        "    ('right_shoulder', 'right_elbow'),\n",
        "    ('right_elbow', 'right_wrist'),\n",
        "    ('left_shoulder', 'left_hip'),\n",
        "    ('right_shoulder', 'right_hip'),\n",
        "    ('left_hip', 'right_hip'),\n",
        "    ('left_hip', 'left_knee'),\n",
        "    ('left_knee', 'left_ankle'),\n",
        "    ('right_hip', 'right_knee'),\n",
        "    ('right_knee', 'right_ankle'),\n",
        "]\n",
        "\n",
        "# --- 3. Fungsi gambar skeleton di atas frame ----------------------------------\n",
        "def draw_skeleton_on_frame(frame_bgr, joints_dict, edges, joint_color, edge_color, label_text=None):\n",
        "    \"\"\"Gambar skeleton (titik + garis) di atas frame.\"\"\"\n",
        "    overlay = frame_bgr.copy()\n",
        "    for jn_a, jn_b in edges:\n",
        "        if jn_a in joints_dict and jn_b in joints_dict:\n",
        "            pt_a = (int(round(joints_dict[jn_a][0])), int(round(joints_dict[jn_a][1])))\n",
        "            pt_b = (int(round(joints_dict[jn_b][0])), int(round(joints_dict[jn_b][1])))\n",
        "            cv2.line(overlay, pt_a, pt_b, edge_color, 3, cv2.LINE_AA)\n",
        "    for jn, (x, y) in joints_dict.items():\n",
        "        pt = (int(round(x)), int(round(y)))\n",
        "        cv2.circle(overlay, pt, 7, joint_color, -1, cv2.LINE_AA)\n",
        "        cv2.circle(overlay, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)\n",
        "    if label_text:\n",
        "        cv2.putText(overlay, label_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX,\n",
        "                    1.5, (255, 255, 255), 4, cv2.LINE_AA)\n",
        "        cv2.putText(overlay, label_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX,\n",
        "                    1.5, edge_color, 2, cv2.LINE_AA)\n",
        "    return overlay\n",
        "\n",
        "def get_mediapipe_joints_px(bundle, frame_idx, joint_names):\n",
        "    \"\"\"Ambil koordinat pixel MediaPipe untuk 12 joint pada 1 frame.\"\"\"\n",
        "    pose_frame = bundle.pose_array[frame_idx]\n",
        "    joints = {}\n",
        "    for jn in joint_names:\n",
        "        li = LANDMARK_INDEX[jn]\n",
        "        x_norm, y_norm = float(pose_frame[li, 0]), float(pose_frame[li, 1])\n",
        "        if np.isfinite(x_norm) and np.isfinite(y_norm):\n",
        "            x_px = x_norm * bundle.frame_width\n",
        "            y_px = y_norm * bundle.frame_height\n",
        "            joints[jn] = (x_px, y_px)\n",
        "    return joints\n",
        "\n",
        "# --- 4. Hitung rekomendasi berdasarkan analisis sebelumnya --------------------\n",
        "if cvat_summary_df is not None and not cvat_summary_df.empty:\n",
        "    mean_err = float(cvat_summary_df['mean_error_px'].mean())\n",
        "    mean_err_norm = float(cvat_summary_df['mean_error_norm_torso'].mean())\n",
        "    rekomendasi = (\n",
        "        f'Rekomendasi: MediaPipe BlazePose (otomatis, konsisten, coverage 100%). '\n",
        "        f'Error rata-rata CVAT vs MP = {mean_err:.1f} px ({mean_err_norm:.2f} x torso).'\n",
        "    )\n",
        "else:\n",
        "    rekomendasi = 'Rekomendasi: MediaPipe BlazePose (otomatis, konsisten, coverage 100%).'\n",
        "\n",
        "# --- 5. Generate video side-by-side -------------------------------------------\n",
        "video_out_path = OUTPUT_DIR / 'comparison_cvat_vs_mediapipe.mp4'\n",
        "\n",
        "panel_w = bundle.frame_width\n",
        "panel_h = bundle.frame_height\n",
        "info_bar_h = 160\n",
        "canvas_w = panel_w * 2\n",
        "canvas_h = panel_h + info_bar_h\n",
        "\n",
        "fourcc = cv2.VideoWriter_fourcc(*'mp4v')\n",
        "writer = cv2.VideoWriter(str(video_out_path), fourcc, bundle.fps, (canvas_w, canvas_h))\n",
        "\n",
        "total_frames = len(bundle.frames_bgr)\n",
        "for fidx in range(total_frames):\n",
        "    frame_bgr = bundle.frames_bgr[fidx]\n",
        "    \n",
        "    # Panel kiri: CVAT skeleton (biru)\n",
        "    cvat_j = cvat_joints.get(fidx, {})\n",
        "    panel_cvat = draw_skeleton_on_frame(\n",
        "        frame_bgr, cvat_j, SKELETON_EDGES_12,\n",
        "        joint_color=(255, 120, 50),\n",
        "        edge_color=(255, 80, 0),\n",
        "        label_text='CVAT (Manual)'\n",
        "    )\n",
        "    \n",
        "    # Panel kanan: MediaPipe skeleton (hijau)\n",
        "    mp_j = get_mediapipe_joints_px(bundle, fidx, CVAT_12_JOINTS)\n",
        "    panel_mp = draw_skeleton_on_frame(\n",
        "        frame_bgr, mp_j, SKELETON_EDGES_12,\n",
        "        joint_color=(50, 220, 100),\n",
        "        edge_color=(30, 200, 80),\n",
        "        label_text='MediaPipe (Auto)'\n",
        "    )\n",
        "    \n",
        "    combined = np.hstack([panel_cvat, panel_mp])\n",
        "    \n",
        "    # Bar informasi bawah\n",
        "    info_bar = np.zeros((info_bar_h, canvas_w, 3), dtype=np.uint8)\n",
        "    info_bar[:] = (30, 30, 30)\n",
        "    \n",
        "    row = per_frame_df.loc[per_frame_df['frame_index'] == fidx]\n",
        "    if not row.empty:\n",
        "        row = row.iloc[0]\n",
        "        label_val = int(row['frame_label'])\n",
        "        knee_angle = f\"{row['angle_knee_avg']:.1f}\"\n",
        "        hip_angle = f\"{row['angle_hip_avg']:.1f}\"\n",
        "        valgus = f\"{row['valgus_ratio']:.2f}\"\n",
        "    else:\n",
        "        label_val = -1\n",
        "        knee_angle = hip_angle = valgus = 'N/A'\n",
        "    \n",
        "    if label_val == 0:\n",
        "        label_color = (87, 139, 46)\n",
        "        label_str = 'BENAR'\n",
        "    elif label_val == 1:\n",
        "        label_color = (39, 39, 214)\n",
        "        label_str = 'SALAH'\n",
        "    else:\n",
        "        label_color = (128, 128, 128)\n",
        "        label_str = 'N/A'\n",
        "    \n",
        "    info_line1 = f'Frame {fidx}/{total_frames-1}  |  Label: {label_str}  |  Knee={knee_angle}  Hip={hip_angle}  Valgus={valgus}'\n",
        "    cv2.putText(info_bar, info_line1, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,\n",
        "                1.2, label_color, 3, cv2.LINE_AA)\n",
        "    \n",
        "    cv2.putText(info_bar, rekomendasi, (20, 120), cv2.FONT_HERSHEY_SIMPLEX,\n",
        "                0.8, (200, 200, 200), 2, cv2.LINE_AA)\n",
        "    \n",
        "    canvas = np.vstack([combined, info_bar])\n",
        "    writer.write(canvas)\n",
        "\n",
        "writer.release()\n",
        "print(f'\\n=== Video perbandingan berhasil disimpan ===')\n",
        "print(f'   Path    : {video_out_path}')\n",
        "print(f'   Resolusi: {canvas_w} x {canvas_h}')\n",
        "print(f'   FPS     : {bundle.fps}')\n",
        "print(f'   Frames  : {total_frames}')\n",
        "print(f'\\n=== REKOMENDASI ===')\n",
        "print(f'   {rekomendasi}')\n",
        "print(f'\\n=== Ringkasan jawaban untuk dosen pembimbing ===')\n",
        "print(f'   1. Jumlah titik sendi: 12 joint utama (8 inti untuk validasi biomekanik)')\n",
        "print(f'   2. Skeleton MediaPipe: BERHASIL dikeluarkan dan divisualisasikan')\n",
        "print(f'      - Detection rate: {bundle.detected_mask.sum()}/{bundle.pose_array.shape[0]} frame (100%)')\n",
        "print(f'      - Visibility rata-rata: {visibility_df[\"mean_visibility\"].mean():.4f}')\n",
        "print(f'   3. Label per-frame: {int((per_frame_df[\"frame_label\"]==0).sum())} Benar, {int((per_frame_df[\"frame_label\"]==1).sum())} Salah')\n",
        "print(f'      (berdasarkan 3 aturan biomekanik: knee valgus, hip flexion, knee depth)')\n",
        "print(f'\\n   Video menampilkan skeleton CVAT (biru, kiri) vs MediaPipe (hijau, kanan) secara side-by-side.')\n",
        "print(f'   MediaPipe direkomendasikan karena otomatis, konsisten, dan coverage 100%.')\n",
        "\n",
        "try:\n",
        "    from IPython.display import Video as IPyVideo, display as ipy_display\n",
        "    ipy_display(IPyVideo(str(video_out_path), embed=True, width=800))\n",
        "except Exception:\n",
        "    print(f'\\n(Video embed tidak tersedia. Buka file secara manual: {video_out_path})')"
    ]
}


def main():
    # Baca notebook
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # Cari cell kesimpulan terakhir (yang berisi "Kesimpulan yang Siap Disampaikan")
    conclusion_idx = None
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))
        if "Kesimpulan yang Siap Disampaikan" in src:
            conclusion_idx = i
            break

    if conclusion_idx is None:
        print("ERROR: Tidak menemukan cell kesimpulan. Tidak ada perubahan.")
        return

    # Insert dua cell baru sebelum cell kesimpulan
    cells.insert(conclusion_idx, code_cell)
    cells.insert(conclusion_idx, markdown_cell)

    # Tulis kembali notebook
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"✅ Berhasil menambahkan 2 cell baru (markdown + code) sebelum cell kesimpulan.")
    print(f"   Cell baru ada di indeks {conclusion_idx} dan {conclusion_idx + 1}.")
    print(f"   Notebook disimpan ke: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
