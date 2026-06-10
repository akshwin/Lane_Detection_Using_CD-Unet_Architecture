import os
import uuid

import cv2
import matplotlib
import numpy as np
import streamlit as st
from PIL import Image

matplotlib.use("Agg")

# ---- Page Config ---- #
st.set_page_config(
    page_title="Object Detection AI",
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="expanded",
)

# ---- Folders ---- #
UPLOADED_FOLDER = "static/uploaded"
PREDICTED_FOLDER = "static/predicted"
os.makedirs(UPLOADED_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_FOLDER, exist_ok=True)

# ---- COCO class groups ---- #
ALL_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

CLASS_GROUPS = {
    "All objects": set(ALL_CLASSES.keys()),
    "Vehicles": {1, 2, 3, 4, 5, 6, 7, 8},
    "People": {0},
    "Animals": {14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
    "Electronics": {62, 63, 64, 65, 66, 67, 68, 69, 70},
}

INFERENCE_BATCH = 8


# ---- Model loader ---- #
@st.cache_resource
def load_yolo(model_size="n"):
    from ultralytics import YOLO

    return YOLO(f"yolov8{model_size}.pt")


# ---- Inference helpers ---- #


def _should_skip(frame_idx, frame_skip):
    if frame_skip <= 1:
        return False
    if frame_skip == 1.5:
        return frame_idx % 3 == 2
    return frame_idx % int(frame_skip) != 0


def draw_boxes(bgr_frame, results, active_classes):
    """Draw bounding boxes and labels on a BGR frame."""
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in active_classes:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{ALL_CLASSES.get(cls_id, cls_id)} {conf:.2f}"
        cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(
            bgr_frame,
            (x1, max(y1 - th - 6, 0)),
            (x1 + tw + 4, max(y1, th + 6)),
            (0, 165, 255),
            -1,
        )
        cv2.putText(
            bgr_frame,
            label,
            (x1 + 2, max(y1 - 4, th + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
    return bgr_frame


def detect_image(image_path, model_size, conf, active_classes):
    """Run detection on a single image. Returns annotated PIL image + detection count."""
    yolo = load_yolo(model_size)
    bgr = cv2.imread(image_path)
    results = yolo(bgr, verbose=False, conf=conf)[0]
    count = sum(1 for b in results.boxes if int(b.cls[0]) in active_classes)
    annotated = draw_boxes(bgr.copy(), results, active_classes)
    return Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)), count


def detect_video(
    video_path, model_size, conf, active_classes, test_seconds=0, frame_skip=1
):
    """Run detection on every frame of a video. Returns output path."""
    from ultralytics import YOLO

    yolo = load_yolo(model_size)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(fps * test_seconds) if test_seconds > 0 else 0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(PREDICTED_FOLDER, f"detect_{uuid.uuid4().hex}.mp4")
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (orig_w, orig_h)
    )

    bar = st.progress(0, text="Running object detection…")
    frame_idx = 0
    last_frame = None  # last annotated frame, reused on skipped frames

    bgr_buf = []

    def flush(buf):
        nonlocal last_frame
        results_batch = yolo(buf, verbose=False, conf=conf)
        for bgr_f, res in zip(buf, results_batch):
            annotated = draw_boxes(bgr_f.copy(), res, active_classes)
            last_frame = annotated
            writer.write(annotated)

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if _should_skip(frame_idx, frame_skip) and last_frame is not None:
            writer.write(last_frame)
        else:
            bgr_buf.append(bgr)
            if len(bgr_buf) == INFERENCE_BATCH:
                flush(bgr_buf)
                bgr_buf = []

        frame_idx += 1
        bar.progress(
            min(frame_idx / total, 1.0) if total else 0,
            text=f"Frame {frame_idx} / {total}",
        )
        if max_frames and frame_idx >= max_frames:
            break

    if bgr_buf:
        flush(bgr_buf)

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
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem;}
.stTabs [data-baseweb="tab"] {font-size: 0.95rem; font-weight: 600; padding: 0.5rem 1.2rem;}
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.82rem; font-weight: 600; margin-right: 6px;
}
.badge-orange { background: rgba(255,165,0,0.18); color: #ffa500; border: 1px solid #ffa500; }
.badge-blue   { background: rgba(0,120,255,0.18); color: #4499ff; border: 1px solid #4499ff; }
</style>
""",
    unsafe_allow_html=True,
)

# ---- Sidebar ---- #
with st.sidebar:
    st.markdown("## 🎯 Object Detection AI")
    st.markdown("YOLOv8 real-time object detection for images and videos.")
    st.divider()

    st.markdown("**Model size**")
    model_size = st.radio(
        "model_size",
        ["n", "s", "m"],
        format_func=lambda x: {
            "n": "Nano (fastest)",
            "s": "Small",
            "m": "Medium (most accurate)",
        }[x],
        index=0,
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown("**Detect**")
    group = st.selectbox(
        "Class group", list(CLASS_GROUPS.keys()), label_visibility="collapsed"
    )
    active_classes = CLASS_GROUPS[group]
    st.divider()

    st.markdown("**Confidence threshold**")
    conf_thresh = st.slider("conf", 10, 90, 40, 5, label_visibility="collapsed") / 100.0
    st.divider()

    st.caption("YOLOv8 · ultralytics · COCO 80 classes")


# ---- Header ---- #
st.markdown("## 🎯 Object Detection AI")
st.markdown(
    "YOLOv8-powered detection for images and videos — portrait & landscape supported."
)
st.divider()

# ---- Session state ---- #
for _k in ("img_result", "img_name", "vid_out", "vid_name", "script_results"):
    if _k not in st.session_state:
        st.session_state[_k] = None

# ---- Tabs ---- #
tab_img, tab_vid, tab_script = st.tabs(["🖼️  Image", "🎬  Video", "📋  Personal Script"])


# ══════════════════════════════════════════════════════
# IMAGE TAB
# ══════════════════════════════════════════════════════
with tab_img:
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="img_upload",
        label_visibility="collapsed",
    )

    if uploaded:
        if st.session_state.img_name != uploaded.name:
            st.session_state.img_result = None
            st.session_state.img_name = uploaded.name

        img_path = os.path.join(UPLOADED_FOLDER, f"{uuid.uuid4().hex}_{uploaded.name}")
        with open(img_path, "wb") as f:
            f.write(uploaded.read())

        orig_pil = Image.open(img_path).convert("RGB")

        if st.button("▶  Detect Objects", key="run_img"):
            try:
                with st.spinner("Running detection…"):
                    result_pil, count = detect_image(
                        img_path, model_size, conf_thresh, active_classes
                    )
                st.session_state.img_result = (result_pil, count)
                st.success(f"Done — {count} object(s) detected.")
            except Exception as e:
                st.error(f"Error: {e}")

        if st.session_state.img_result:
            result_pil, count = st.session_state.img_result
            c1, c2 = st.columns(2)
            with c1:
                st.image(orig_pil, caption="Original", use_container_width=True)
            with c2:
                st.image(
                    result_pil,
                    caption=f"Detected — {count} object(s)",
                    use_container_width=True,
                )

            # Save annotated image for download
            ann_path = os.path.join(
                PREDICTED_FOLDER, f"annotated_{uuid.uuid4().hex}.jpg"
            )
            result_pil.save(ann_path)
            with open(ann_path, "rb") as f:
                st.download_button(
                    "⬇  Download annotated image",
                    f,
                    "detected.jpg",
                    "image/jpeg",
                    key="dl_img",
                )
    else:
        st.info("Upload an image to get started.")


# ══════════════════════════════════════════════════════
# VIDEO TAB
# ══════════════════════════════════════════════════════
with tab_vid:
    vid_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv"],
        key="vid_upload",
        label_visibility="collapsed",
    )

    c1, c2 = st.columns(2)
    with c1:
        test_secs = st.slider(
            "Seconds to process",
            0,
            60,
            6,
            1,
            help="0 = full video. Use 6 s to verify first.",
            key="vid_secs",
        )
    with c2:
        vid_skip = st.select_slider(
            "Speed",
            options=[1, 1.5, 2, 3],
            value=1,
            format_func=lambda x: {
                1: "Quality (1×)",
                1.5: "Smooth (1.5×)",
                2: "Fast (2×)",
                3: "Fastest (3×)",
            }[x],
            key="vid_skip",
        )

    if vid_file:
        if st.session_state.vid_name != vid_file.name:
            st.session_state.vid_out = None
            st.session_state.vid_name = vid_file.name

        vid_path = os.path.join(UPLOADED_FOLDER, f"{uuid.uuid4().hex}_{vid_file.name}")
        with open(vid_path, "wb") as f:
            f.write(vid_file.read())

        st.video(vid_path)

        if st.button("▶  Detect Objects in Video", key="run_vid"):
            try:
                out = detect_video(
                    vid_path,
                    model_size,
                    conf_thresh,
                    active_classes,
                    test_seconds=test_secs,
                    frame_skip=vid_skip,
                )
                st.session_state.vid_out = out
                st.success("Done — download ready below.")
            except Exception as e:
                st.error(f"Error: {e}")

        if st.session_state.vid_out and os.path.exists(st.session_state.vid_out):
            cap_p = cv2.VideoCapture(st.session_state.vid_out)
            ret, frame = cap_p.read()
            cap_p.release()
            if ret:
                st.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    caption="Preview — first processed frame",
                    use_container_width=True,
                )
            with open(st.session_state.vid_out, "rb") as f:
                st.download_button(
                    "⬇  Download video",
                    f,
                    "detected_video.mp4",
                    "video/mp4",
                    key="dl_vid",
                )
    else:
        st.info("Upload a video to get started.")


# ══════════════════════════════════════════════════════
# PERSONAL SCRIPT TAB
# ══════════════════════════════════════════════════════
with tab_script:
    st.markdown(
        "Upload up to **10 videos** for batch detection. "
        "All settings from the sidebar apply to every video.",
    )
    st.markdown("")

    script_files = st.file_uploader(
        "Upload videos",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=True,
        key="script_upload",
        label_visibility="collapsed",
    )

    sc1, sc2 = st.columns(2)
    with sc1:
        script_skip = st.select_slider(
            "Speed",
            options=[1, 1.5, 2, 3],
            value=2,
            format_func=lambda x: {
                1: "Quality (1×)",
                1.5: "Smooth (1.5×)",
                2: "Fast (2×)",
                3: "Fastest (3×)",
            }[x],
            help="Default Fast (2×) for batch jobs.",
            key="script_skip",
        )
    with sc2:
        script_secs = st.slider(
            "Seconds per video (0 = full)", 0, 60, 0, 1, key="script_secs"
        )

    if script_files:
        script_files = script_files[:10]
        total = len(script_files)

        st.markdown("**Queue**")
        for i, f in enumerate(script_files):
            st.markdown(f"{i + 1}. `{f.name}`")

        st.markdown("")

        if st.button("▶  Run Personal Script", key="run_script"):
            results = []
            for i, vid_file in enumerate(script_files):
                label = f"Video {i + 1}"
                st.markdown(f"**[{i + 1}/{total}] {label}** — `{vid_file.name}`")

                in_path = os.path.join(
                    UPLOADED_FOLDER, f"{uuid.uuid4().hex}_{vid_file.name}"
                )
                with open(in_path, "wb") as f_:
                    f_.write(vid_file.read())

                try:
                    out_path = detect_video(
                        in_path,
                        model_size,
                        conf_thresh,
                        active_classes,
                        test_seconds=script_secs,
                        frame_skip=script_skip,
                    )
                    results.append((label, vid_file.name, out_path))
                    st.success("✓ Done")
                except Exception as e:
                    st.error(f"✗ Failed: {e}")
                    results.append((label, vid_file.name, None))

            st.session_state.script_results = results

        if st.session_state.script_results:
            st.divider()
            st.markdown("### ⬇  Downloads")
            for label, orig_name, out_path in st.session_state.script_results:
                if out_path is None or not os.path.exists(out_path):
                    st.markdown(f"❌ `{label}` — failed")
                    continue
                out_filename = f"detected_{label.lower().replace(' ', '_')}.mp4"
                st.markdown(f"`{label}` — `{orig_name}`")
                with open(out_path, "rb") as f_:
                    st.download_button(
                        f"⬇  Download {label}",
                        data=f_,
                        file_name=out_filename,
                        mime="video/mp4",
                        key=f"dl_{out_path}",
                    )
    else:
        st.info("Upload up to 10 videos to run batch detection.")
