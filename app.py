import os
import uuid

import cv2
import matplotlib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

matplotlib.use("Agg")

# ---- Page Config ---- #
st.set_page_config(
    page_title="Road Vision AI",
    layout="wide",
    page_icon="🛣️",
    initial_sidebar_state="expanded",
)

# ---- Constants ---- #
MODEL_INPUT_H, MODEL_INPUT_W = 256, 256
UPLOADED_FOLDER = "static/uploaded"
PREDICTED_FOLDER = "static/predicted"
SAMPLE_IMAGE_PATH = "upload_image/road.webp"

os.makedirs(UPLOADED_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_FOLDER, exist_ok=True)

# ---- Optional: Disable GPU ---- #
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


# ---- Load Model ---- #
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_for_lane.keras", compile=False)


model = load_model()


# ---- YOLO Model ---- #

DETECT_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@st.cache_resource
def load_yolo():
    from ultralytics import YOLO

    return YOLO("yolov8n.pt")


def draw_detections(bgr_frame, conf_threshold=0.40):
    """Draw YOLO bounding boxes for vehicles/persons on a BGR frame."""
    yolo = load_yolo()
    results = yolo(bgr_frame, verbose=False, conf=conf_threshold)[0]
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in DETECT_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{DETECT_CLASSES[cls_id]} {float(box.conf[0]):.2f}"
        cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            bgr_frame,
            label,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
        )
    return bgr_frame


# ---- Inference constants ---- #
INFERENCE_BATCH = 8  # frames per model.predict() call — amortises TF overhead


# ---- Core Prediction Helpers ---- #


def _preprocess_roi(rgb_array, crop_top):
    """Crop + resize one RGB frame to model input size."""
    roi = rgb_array[crop_top:, :, :]
    return (
        cv2.resize(
            roi, (MODEL_INPUT_W, MODEL_INPUT_H), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        / 255.0
    )


def _pred_to_mask(pred, orig_h, orig_w, crop_top):
    """Convert a single model prediction to a full-frame binary mask."""
    mask_256 = np.squeeze((pred > 0.5).astype(np.uint8) * 255)
    roi_h, roi_w = orig_h - crop_top, orig_w
    mask_roi = cv2.resize(mask_256, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((orig_h, orig_w), dtype=np.uint8)
    full[crop_top:, :] = mask_roi
    return full


def predict_masks_batch(rgb_list, road_start_ratio=0.0):
    """
    Run lane segmentation on a list of RGB frames in ONE batched predict() call.
    Returns a list of full-frame binary masks.
    """
    if not rgb_list:
        return []
    orig_h, orig_w = rgb_list[0].shape[:2]
    crop_top = int(orig_h * road_start_ratio)
    batch = np.stack([_preprocess_roi(f, crop_top) for f in rgb_list])  # (N,256,256,3)
    preds = model.predict(batch, verbose=0)  # (N,256,256,1)
    return [_pred_to_mask(p, orig_h, orig_w, crop_top) for p in preds]


# kept for image-tab (single-frame) use
def predict_mask_for_frame(rgb_array, road_start_ratio=0.0):
    return predict_masks_batch([rgb_array], road_start_ratio)[0]


def draw_detections_batch(bgr_frames, conf_threshold=0.40):
    """Run YOLO on a list of BGR frames in one batched call."""
    yolo = load_yolo()
    results = yolo(bgr_frames, verbose=False, conf=conf_threshold)
    out = []
    for bgr, res in zip(bgr_frames, results):
        for box in res.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in DETECT_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{DETECT_CLASSES[cls_id]} {float(box.conf[0]):.2f}"
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(
                bgr,
                label,
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
            )
        out.append(bgr)
    return out


# kept for single-frame use
def draw_detections(bgr_frame, conf_threshold=0.40):
    return draw_detections_batch([bgr_frame], conf_threshold)[0]


def overlay_mask_on_frame(bgr_frame, mask, color=(0, 255, 80), alpha=0.45):
    colored = np.zeros_like(bgr_frame)
    colored[mask > 0] = color
    return cv2.addWeighted(bgr_frame, 1.0, colored, alpha, 0)


def predict_image(image_path, road_start_ratio=0.0):
    img_pil = Image.open(image_path).convert("RGB")
    rgb = np.array(img_pil)
    mask = predict_mask_for_frame(rgb, road_start_ratio=road_start_ratio)
    return img_pil, Image.fromarray(mask, mode="L")


def _should_skip(frame_idx, frame_skip):
    """Return True if this frame should reuse the last mask instead of inferring."""
    if frame_skip <= 1:
        return False
    if frame_skip == 1.5:
        return frame_idx % 3 == 2  # skip 1 in 3 → 1.5× speedup
    return frame_idx % int(frame_skip) != 0


def _open_writer(video_path, suffix):
    """Open VideoCapture + VideoWriter pair, return (cap, writer, meta)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(PREDICTED_FOLDER, f"{suffix}_{uuid.uuid4().hex}.mp4")
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (orig_w, orig_h)
    )
    return cap, writer, out_path, fps, total, orig_w, orig_h


def process_video(video_path, road_start_ratio=0.0, test_seconds=0, frame_skip=1):
    """
    Lane-only processing with batched inference and optional frame skipping.
    frame_skip=1 → every frame, =2 → every other frame (reuse last mask), etc.
    """
    cap, writer, out_path, fps, total, orig_w, orig_h = _open_writer(video_path, "lane")
    max_frames = int(fps * test_seconds) if test_seconds > 0 else 0

    bar = st.progress(0, text="Detecting lanes…")
    frame_idx = 0
    last_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    rgb_buf, bgr_buf = [], []

    def flush_lane(rgb_b, bgr_b):
        nonlocal last_mask
        masks = predict_masks_batch(rgb_b, road_start_ratio)
        for bgr_f, m in zip(bgr_b, masks):
            last_mask = m
            writer.write(overlay_mask_on_frame(bgr_f, m))

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if _should_skip(frame_idx, frame_skip):
            # reuse last known mask — no inference needed
            writer.write(overlay_mask_on_frame(bgr, last_mask))
        else:
            rgb_buf.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            bgr_buf.append(bgr)
            if len(rgb_buf) == INFERENCE_BATCH:
                flush_lane(rgb_buf, bgr_buf)
                rgb_buf, bgr_buf = [], []

        frame_idx += 1
        bar.progress(
            min(frame_idx / total, 1.0) if total else 0,
            text=f"Frame {frame_idx} / {total}",
        )
        if max_frames and frame_idx >= max_frames:
            break

    if rgb_buf:  # flush remainder
        flush_lane(rgb_buf, bgr_buf)

    cap.release()
    writer.release()
    bar.empty()
    return out_path


def process_video_advanced(
    video_path, road_start_ratio=0.0, test_seconds=0, conf=0.40, frame_skip=1
):
    """
    Lane + YOLO processing with batched inference and optional frame skipping.
    """
    cap, writer, out_path, fps, total, orig_w, orig_h = _open_writer(
        video_path, "advanced"
    )
    max_frames = int(fps * test_seconds) if test_seconds > 0 else 0

    bar = st.progress(0, text="Running lane + object detection…")
    frame_idx = 0
    last_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    rgb_buf, bgr_buf = [], []

    def flush_advanced(rgb_b, bgr_b):
        nonlocal last_mask
        masks = predict_masks_batch(rgb_b, road_start_ratio)
        frames = [overlay_mask_on_frame(b, m) for b, m in zip(bgr_b, masks)]
        frames = draw_detections_batch(frames, conf)
        for bgr_f, m, result in zip(bgr_b, masks, frames):
            last_mask = m
            writer.write(result)

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if _should_skip(frame_idx, frame_skip):
            result = overlay_mask_on_frame(bgr, last_mask)
            result = draw_detections(result, conf)
            writer.write(result)
        else:
            rgb_buf.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            bgr_buf.append(bgr)
            if len(rgb_buf) == INFERENCE_BATCH:
                flush_advanced(rgb_buf, bgr_buf)
                rgb_buf, bgr_buf = [], []

        frame_idx += 1
        bar.progress(
            min(frame_idx / total, 1.0) if total else 0,
            text=f"Frame {frame_idx} / {total}",
        )
        if max_frames and frame_idx >= max_frames:
            break

    if rgb_buf:
        flush_advanced(rgb_buf, bgr_buf)

    cap.release()
    writer.release()
    bar.empty()
    return out_path


# ══════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════

# ---- Global CSS ---- #
st.markdown(
    """
<style>
/* Hide default Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}

/* Tighter top padding */
.block-container {padding-top: 1.5rem;}

/* Tab font size */
.stTabs [data-baseweb="tab"] {font-size: 0.95rem; font-weight: 600; padding: 0.5rem 1.2rem;}

/* Subtle card containers */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}

/* Legend badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 6px;
}
.badge-green  { background: rgba(0,255,80,0.18);  color: #00ff50; border: 1px solid #00ff50; }
.badge-orange { background: rgba(255,165,0,0.18); color: #ffa500; border: 1px solid #ffa500; }
</style>
""",
    unsafe_allow_html=True,
)


# ---- Sidebar ---- #
with st.sidebar:
    st.markdown("## 🛣️ Road Vision AI")
    st.markdown("Cascaded U-Net lane segmentation + YOLOv8 object detection.")
    st.divider()

    st.markdown("**Detection legend**")
    st.markdown(
        '<span class="badge badge-green">● Lanes</span>'
        '<span class="badge badge-orange">● Objects</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Models**")
    st.markdown("- Lane: Cascaded U-Net (custom trained)\n- Objects: YOLOv8n (COCO)")
    st.divider()

    st.caption("Built by Akshwin T · akshwint.2003@gmail.com")


# ---- Header ---- #
st.markdown("## 🛣️ Road Vision AI")
st.markdown(
    "Lane segmentation and object detection for road videos — portrait & landscape."
)
st.divider()

# ---- Session state init ---- #
for _key in ("simple_out", "simple_name", "adv_out", "adv_name", "script_results"):
    if _key not in st.session_state:
        st.session_state[_key] = None

# ---- Tabs ---- #
tab_image, tab_simple, tab_advanced, tab_script = st.tabs(
    ["🖼️  Image", "🛣️  Lane Detection", "🚗  Advanced", "📋  Personal Script"]
)


# ══════════════════════════════════════════════════════
# IMAGE TAB
# ══════════════════════════════════════════════════════
with tab_image:
    col_upload, col_controls = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a road image",
            type=["jpg", "jpeg", "png", "webp"],
            key="img_upload",
            label_visibility="collapsed",
        )
        use_sample = st.button("Use sample image", key="use_sample")

    with col_controls:
        img_road_start = (
            st.slider(
                "Skip top %",
                0,
                70,
                0,
                5,
                help="Slide up for portrait images with sky at the top.",
                key="img_road_start",
            )
            / 100.0
        )

    image_source_path = None
    source_label = ""

    if uploaded_file and not use_sample:
        temp_name = f"{uuid.uuid4().hex}.png"
        image_source_path = os.path.join(UPLOADED_FOLDER, temp_name)
        with open(image_source_path, "wb") as f:
            f.write(uploaded_file.read())
        source_label = "Original"
    elif use_sample and not uploaded_file:
        image_source_path = SAMPLE_IMAGE_PATH
        source_label = "Sample"
    elif uploaded_file and use_sample:
        st.warning("Pick one — upload or sample, not both.")
    else:
        st.info("Upload an image or use the sample to get started.")

    if image_source_path and os.path.exists(image_source_path):
        try:
            with st.spinner("Detecting lanes…"):
                orig_pil, mask_pil = predict_image(
                    image_source_path, road_start_ratio=img_road_start
                )

            c1, c2 = st.columns(2)
            with c1:
                st.image(orig_pil, caption=source_label, use_container_width=True)
            with c2:
                st.image(mask_pil, caption="Lane mask", use_container_width=True)
            st.success("Done.")
        except Exception as e:
            st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════
# SIMPLE TAB — Lane detection only
# ══════════════════════════════════════════════════════
with tab_simple:
    st.markdown(
        '<span class="badge badge-green">● Lane mask</span> overlaid on every frame at original resolution.',
        unsafe_allow_html=True,
    )
    st.markdown("")

    video_file = st.file_uploader(
        "Upload a road video",
        type=["mp4", "mov", "avi", "mkv"],
        key="vid_upload",
        label_visibility="collapsed",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        test_seconds = st.slider(
            "Seconds to process",
            0,
            60,
            6,
            1,
            help="0 = full video. Use 6 s for a quick test.",
            key="test_seconds",
        )
    with c2:
        vid_road_start = (
            st.slider(
                "Skip top %",
                0,
                70,
                40,
                5,
                help="Skip sky/signs at top of portrait frames.",
                key="vid_road_start",
            )
            / 100.0
        )
    with c3:
        vid_frame_skip = st.select_slider(
            "Speed",
            options=[1, 1.5, 2, 3],
            value=1,
            format_func=lambda x: {
                1: "Quality (1×)",
                1.5: "Smooth (1.5×)",
                2: "Fast (2×)",
                3: "Fastest (3×)",
            }[x],
            help="Reuse last mask for skipped frames. 1.5× skips 1 in 3 frames.",
            key="vid_frame_skip",
        )

    if video_file is not None:
        # Clear old result if a new file is uploaded
        if st.session_state.simple_name != video_file.name:
            st.session_state.simple_out = None
            st.session_state.simple_name = video_file.name

        vid_in_path = os.path.join(
            UPLOADED_FOLDER, f"{uuid.uuid4().hex}_{video_file.name}"
        )
        with open(vid_in_path, "wb") as f:
            f.write(video_file.read())

        st.video(vid_in_path)

        if st.button("▶  Run Lane Detection", key="run_simple"):
            try:
                out_path = process_video(
                    vid_in_path,
                    road_start_ratio=vid_road_start,
                    test_seconds=test_seconds,
                    frame_skip=vid_frame_skip,
                )
                st.session_state.simple_out = out_path
                st.success("Done — download ready below.")
            except Exception as e:
                st.error(f"Error: {e}")

        # Download section — persists across reruns via session state
        if st.session_state.simple_out and os.path.exists(st.session_state.simple_out):
            cap_p = cv2.VideoCapture(st.session_state.simple_out)
            ret, frame = cap_p.read()
            cap_p.release()
            if ret:
                st.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    caption="Preview — first frame",
                    use_container_width=True,
                )
            with open(st.session_state.simple_out, "rb") as f:
                st.download_button(
                    "⬇  Download video",
                    f,
                    "lane_output.mp4",
                    "video/mp4",
                    key="dl_simple",
                )
    else:
        st.info("Upload a video to get started.")


# ══════════════════════════════════════════════════════
# ADVANCED TAB — Lanes + YOLO
# ══════════════════════════════════════════════════════
with tab_advanced:
    st.markdown(
        '<span class="badge badge-green">● Lane mask</span>'
        '<span class="badge badge-orange">● Vehicles &amp; persons</span>'
        "&nbsp; Both overlaid on every frame.",
        unsafe_allow_html=True,
    )
    st.markdown("")

    adv_video_file = st.file_uploader(
        "Upload a road video",
        type=["mp4", "mov", "avi", "mkv"],
        key="adv_vid_upload",
        label_visibility="collapsed",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        adv_test_seconds = st.slider(
            "Seconds to process",
            0,
            60,
            6,
            1,
            help="0 = full video.",
            key="adv_test_seconds",
        )
    with c2:
        adv_road_start = (
            st.slider(
                "Skip top %",
                0,
                70,
                40,
                5,
                help="Skip sky/signs at top of portrait frames.",
                key="adv_road_start",
            )
            / 100.0
        )
    with c3:
        adv_conf = (
            st.slider(
                "Detection confidence %",
                10,
                90,
                40,
                5,
                help="Higher = stricter, fewer false positives.",
                key="adv_conf",
            )
            / 100.0
        )
    with c4:
        adv_frame_skip = st.select_slider(
            "Speed",
            options=[1, 1.5, 2, 3],
            value=1,
            format_func=lambda x: {
                1: "Quality (1×)",
                1.5: "Smooth (1.5×)",
                2: "Fast (2×)",
                3: "Fastest (3×)",
            }[x],
            help="Reuse last mask for skipped frames. 1.5× skips 1 in 3 frames.",
            key="adv_frame_skip",
        )

    if adv_video_file is not None:
        # Clear old result if a new file is uploaded
        if st.session_state.adv_name != adv_video_file.name:
            st.session_state.adv_out = None
            st.session_state.adv_name = adv_video_file.name

        adv_in_path = os.path.join(
            UPLOADED_FOLDER, f"{uuid.uuid4().hex}_{adv_video_file.name}"
        )
        with open(adv_in_path, "wb") as f:
            f.write(adv_video_file.read())

        st.video(adv_in_path)

        if st.button("▶  Run Advanced Detection", key="run_advanced"):
            try:
                adv_out = process_video_advanced(
                    adv_in_path,
                    road_start_ratio=adv_road_start,
                    test_seconds=adv_test_seconds,
                    conf=adv_conf,
                    frame_skip=adv_frame_skip,
                )
                st.session_state.adv_out = adv_out
                st.success("Done — download ready below.")
            except Exception as e:
                st.error(f"Error: {e}")

        # Download section — persists across reruns via session state
        if st.session_state.adv_out and os.path.exists(st.session_state.adv_out):
            cap_p = cv2.VideoCapture(st.session_state.adv_out)
            ret, frame = cap_p.read()
            cap_p.release()
            if ret:
                st.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    caption="Preview — first frame (green = lanes · orange = objects)",
                    use_container_width=True,
                )
            with open(st.session_state.adv_out, "rb") as f:
                st.download_button(
                    "⬇  Download video",
                    f,
                    "advanced_output.mp4",
                    "video/mp4",
                    key="dl_adv",
                )
    else:
        st.info("Upload a video to get started.")


# ══════════════════════════════════════════════════════
# PERSONAL SCRIPT TAB
# ══════════════════════════════════════════════════════
with tab_script:
    st.markdown(
        "Upload up to **10 videos**. "
        '<span class="badge badge-green">● Videos 1–5</span> get lane detection only. '
        '<span class="badge badge-orange">● Videos 6–10</span> get lane + object detection. '
        "All videos are processed in full — no time limit.",
        unsafe_allow_html=True,
    )
    st.markdown("")

    script_files = st.file_uploader(
        "Upload videos",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=True,
        key="script_upload",
        label_visibility="collapsed",
    )

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        script_road_start = (
            st.slider(
                "Skip top %",
                0,
                70,
                40,
                5,
                help="Skip sky/signs at the top of portrait frames.",
                key="script_road_start",
            )
            / 100.0
        )
    with sc2:
        script_conf = (
            st.slider(
                "Detection confidence % — Advanced only",
                10,
                90,
                40,
                5,
                help="Applies to videos 6–10 only.",
                key="script_conf",
            )
            / 100.0
        )
    with sc3:
        script_frame_skip = st.select_slider(
            "Speed",
            options=[1, 1.5, 2, 3],
            value=2,
            format_func=lambda x: {
                1: "Quality (1×)",
                1.5: "Smooth (1.5×)",
                2: "Fast (2×)",
                3: "Fastest (3×)",
            }[x],
            help="Default is Fast (2×) for batch jobs. 1.5× skips 1 in 3 frames.",
            key="script_frame_skip",
        )

    if script_files:
        script_files = script_files[:10]
        total = len(script_files)

        st.markdown("**Queue**")
        for i, f in enumerate(script_files):
            badge = (
                '<span class="badge badge-green">● Simple</span>'
                if i < 5
                else '<span class="badge badge-orange">● Advanced</span>'
            )
            st.markdown(f"{i + 1}. `{f.name}` &nbsp; {badge}", unsafe_allow_html=True)

        st.markdown("")

        if st.button("▶  Run Personal Script", key="run_script"):
            results = []
            simple_count = 0
            advanced_count = 0

            for i, vid_file in enumerate(script_files):
                mode = "simple" if i < 5 else "advanced"
                if mode == "simple":
                    simple_count += 1
                    video_label = f"Simple Video {simple_count}"
                else:
                    advanced_count += 1
                    video_label = f"Advanced Video {advanced_count}"

                st.markdown(f"**[{i + 1}/{total}] {video_label}** — `{vid_file.name}`")

                in_path = os.path.join(
                    UPLOADED_FOLDER, f"{uuid.uuid4().hex}_{vid_file.name}"
                )
                with open(in_path, "wb") as f_:
                    f_.write(vid_file.read())

                try:
                    if mode == "simple":
                        out_path = process_video(
                            in_path,
                            road_start_ratio=script_road_start,
                            test_seconds=0,
                            frame_skip=script_frame_skip,
                        )
                    else:
                        out_path = process_video_advanced(
                            in_path,
                            road_start_ratio=script_road_start,
                            test_seconds=0,
                            conf=script_conf,
                            frame_skip=script_frame_skip,
                        )
                    results.append((video_label, out_path, mode))
                    st.success("✓ Done")
                except Exception as e:
                    st.error(f"✗ Failed: {e}")
                    results.append((video_label, None, mode))

            # Save to session state so downloads survive reruns
            st.session_state.script_results = results

        # Downloads — always rendered from session state
        if st.session_state.script_results:
            st.divider()
            st.markdown("### ⬇  Downloads")
            for video_label, out_path, mode in st.session_state.script_results:
                if out_path is None or not os.path.exists(out_path):
                    st.markdown(f"❌ `{video_label}` — processing failed")
                    continue
                badge = (
                    '<span class="badge badge-green">● Simple</span>'
                    if mode == "simple"
                    else '<span class="badge badge-orange">● Advanced</span>'
                )
                st.markdown(f"**{video_label}** &nbsp; {badge}", unsafe_allow_html=True)
                out_filename = video_label.lower().replace(" ", "_") + ".mp4"
                with open(out_path, "rb") as f_:
                    st.download_button(
                        f"⬇  Download {video_label}",
                        data=f_,
                        file_name=out_filename,
                        mime="video/mp4",
                        key=f"dl_{out_path}",
                    )
    else:
        st.info(
            "Upload up to 10 videos. First 5 → lane detection, next 5 → lane + object detection."
        )
