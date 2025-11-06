import os
import tempfile
import shutil
import time
from typing import List, Dict, Tuple

import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

# CLIP (OpenAI) - for embeddings
import clip
from sklearn.metrics.pairwise import cosine_similarity

# LangChain - used to generate a short textual summary of results
from langchain_classic.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate

# -----------------------------
# Helper utilities
# -----------------------------

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def extract_frames(video_path: str, dest_dir: str, frame_step: int = 2) -> List[str]:
    """Extract frames from video and save as images. Returns list of frame paths."""
    ensure_dir(dest_dir)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    saved = []
    p = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if p % frame_step == 0:
            fname = os.path.join(dest_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(fname, frame)
            saved.append(fname)
            idx += 1
        p += 1
    cap.release()
    return saved


def crop_and_save(frame_path: str, box: List[int], out_path: str):
    img = cv2.imread(frame_path)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    cv2.imwrite(out_path, crop)
    return out_path


# -----------------------------
# Core pipeline functions
# -----------------------------

def detect_and_track_with_yolo(video_path: str, model: YOLO, work_dir: str, conf: float = 0.25, frame_step: int = 2) -> Dict[int, List[Tuple[str, List[int]]]]:
    """
    Runs YOLOv8 track on the video and returns a dictionary mapping track_id -> list of (frame_image_path, bbox)
    Uses ultralytics YOLO.track to produce tracking results. If that isn't available in your ultralytics version,
    this function falls back to per-frame detection (no tracking) and assigns pseudo-track ids by IoU linking.
    """
    ensure_dir(work_dir)
    temp_frames = os.path.join(work_dir, "frames")
    ensure_dir(temp_frames)
    frames = extract_frames(video_path, temp_frames, frame_step=frame_step)

    # Try to use model.track API which can leverage ByteTrack
    tracks_out = {}
    try:
        # ultralytics provides model.track(source=...). We'll call it and parse results.
        results = model.track(source=video_path, conf=conf, persist=True, save=False)
        # results is an iterable of Results for each frame
        # Each result should contain .boxes and optionally .masks and .orig_img
        # We'll iterate frames and collect track ids
        frame_idx = 0
        for r in results:
            if len(r.boxes) == 0:
                frame_idx += 1
                continue
            for box in r.boxes:
                # ultralytics Box object: .xyxy, .conf, .cls, maybe .id
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf_score = float(box.conf[0])
                cls_id = int(box.cls[0])
                track_id = None
                # Some tracker plugins attach .id property
                try:
                    track_id = int(box.id[0])
                except Exception:
                    # not present
                    track_id = None
                # If frame was extracted earlier, try to map
                frame_path = None
                # map by frame_idx if we have saved frames
                if frame_idx < len(frames):
                    frame_path = frames[frame_idx]
                else:
                    # fallback: save original frame from r.orig_img
                    img = r.orig_img
                    frame_path = os.path.join(temp_frames, f"frame_extra_{frame_idx:06d}.jpg")
                    cv2.imwrite(frame_path, img)
                if track_id is None:
                    # assign temporary negative unique id by hashing frame_idx+box coords
                    track_id = hash((frame_idx, int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))) & 0xFFFF
                tracks_out.setdefault(track_id, []).append((frame_path, [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]))
            frame_idx += 1
        return tracks_out
    except Exception as e:
        st.warning("YOLO.track unavailable or failed; falling back to per-frame detection + simple IoU linking. Error: %s" % str(e))
        # Fallback: detect on saved frames and perform greedy IoU linking
        detections_per_frame = []
        for f in frames:
            res = model(f, imgsz=640, conf=conf)
            boxes = []
            for r in res:
                for b in r.boxes:
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    score = float(b.conf[0])
                    clsid = int(b.cls[0])
                    boxes.append((xyxy, score, clsid))
            detections_per_frame.append((f, boxes))

        # Simple greedy IoU tracker
        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            if boxAArea + boxBArea - interArea == 0:
                return 0
            return interArea / float(boxAArea + boxBArea - interArea)

        next_id = 1
        active = []  # list of tuples (track_id, last_box)
        tracks = {}
        for fpath, boxes in detections_per_frame:
            matched = set()
            for b, score, clsid in boxes:
                best_id = None
                best_iou = 0.0
                for tid, last_box in active:
                    i = iou(b, last_box)
                    if i > best_iou and i > 0.3:
                        best_iou = i
                        best_id = tid
                if best_id is None:
                    tid = next_id
                    next_id += 1
                    active.append((tid, b))
                    tracks.setdefault(tid, []).append((fpath, [int(bb) for bb in b]))
                else:
                    # update last_box for that tid
                    for idx_a, (tid_a, last_box_a) in enumerate(active):
                        if tid_a == best_id:
                            active[idx_a] = (tid_a, b)
                            break
                    tracks.setdefault(best_id, []).append((fpath, [int(bb) for bb in b]))
            # Optionally remove stale tracks (not implemented here)
        return tracks


# CLIP embedding helpers

def load_clip(device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def embed_crop(model, preprocess, device: str, crop_path: str):
    img = Image.open(crop_path).convert("RGB")
    inp = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(inp)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]


def aggregate_tracklet_embeddings(model, preprocess, device: str, track_frames: List[Tuple[str, List[int]]], out_dir: str, max_samples: int = 10) -> Tuple[np.ndarray, List[str]]:
    """
    For a tracklet (list of (frame_path, bbox)), crop up to max_samples frames, embed them and aggregate (mean).
    Returns (embedding_vector, list_of_crop_paths)
    """
    ensure_dir(out_dir)
    emb_list = []
    crop_paths = []
    n = len(track_frames)
    if n == 0:
        return None, []
    indices = np.linspace(0, n - 1, min(n, max_samples)).astype(int)
    for i in indices:
        frame_path, bbox = track_frames[i]
        crop_path = os.path.join(out_dir, f"crop_{os.path.basename(frame_path)}_{i}.jpg")
        outp = crop_and_save(frame_path, bbox, crop_path)
        if outp is None:
            continue
        crop_paths.append(outp)
        try:
            e = embed_crop(model, preprocess, device, outp)
            emb_list.append(e)
        except Exception as ex:
            continue
    if len(emb_list) == 0:
        return None, crop_paths
    mat = np.vstack(emb_list)
    agg = mat.mean(axis=0)
    agg = agg / np.linalg.norm(agg)
    return agg, crop_paths


# Compare embeddings

def compare_track_embeddings(embA: Dict[int, np.ndarray], embB: Dict[int, np.ndarray], threshold: float = 0.82) -> List[Tuple[int, int, float]]:
    matches = []
    for idA, vA in embA.items():
        for idB, vB in embB.items():
            sim = float(cosine_similarity(vA.reshape(1, -1), vB.reshape(1, -1))[0, 0])
            if sim >= threshold:
                matches.append((idA, idB, sim))
    matches = sorted(matches, key=lambda x: -x[2])
    return matches


# -----------------------------
# Streamlit UI + orchestration
# -----------------------------

st.set_page_config(page_title="Video Object Matcher", layout="wide")
st.title("🎯 Video Object Matcher")

with st.sidebar:
    st.header("Settings")
    conf_th = st.slider("YOLO confidence threshold", 0.1, 0.9, 0.25, 0.01)
    frame_step = st.slider("Frame step (process every Nth frame)", 1, 10, 3)
    similarity_threshold = st.slider("Embedding similarity threshold", 0.5, 0.99, 0.82, 0.01)
    max_samples = st.slider("Max crops per tracklet", 1, 30, 8)
    run_button = st.button("Run matching")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Video A")
    uploaded_a = st.file_uploader("Upload video A", type=["mp4", "mov", "avi"], key="a")
with col2:
    st.subheader("Video B")
    uploaded_b = st.file_uploader("Upload video B", type=["mp4", "mov", "avi"], key="b")

if uploaded_a is not None and uploaded_b is not None and run_button:
    # create a working directory
    work_root = tempfile.mkdtemp(prefix="video_match_")
    try:
        path_a = os.path.join(work_root, "A.mp4")
        path_b = os.path.join(work_root, "B.mp4")
        with open(path_a, "wb") as f:
            f.write(uploaded_a.getbuffer())
        with open(path_b, "wb") as f:
            f.write(uploaded_b.getbuffer())

        st.info("Loading YOLOv8 model (this may take a few seconds)...")
        yolo_model = YOLO("yolov8n.pt")  # small model; change to yolov8m.pt or larger if you have GPU

        st.info("Running detection + tracking on Video A...")
        tracksA = detect_and_track_with_yolo(path_a, yolo_model, os.path.join(work_root, "A"), conf=conf_th, frame_step=frame_step)
        st.success(f"Found {len(tracksA)} tracklets in Video A")

        st.info("Running detection + tracking on Video B...")
        tracksB = detect_and_track_with_yolo(path_b, yolo_model, os.path.join(work_root, "B"), conf=conf_th, frame_step=frame_step)
        st.success(f"Found {len(tracksB)} tracklets in Video B")

        st.info("Loading CLIP model...")
        clip_model, clip_preprocess, clip_device = load_clip()

        st.info("Embedding tracklets from Video A...")
        embA = {}
        cropsA = {}
        a_out = os.path.join(work_root, "A_crops")
        ensure_dir(a_out)
        for tid, frames in tracksA.items():
            vec, crop_paths = aggregate_tracklet_embeddings(clip_model, clip_preprocess, clip_device, frames, a_out, max_samples=max_samples)
            if vec is not None:
                embA[tid] = vec
                cropsA[tid] = crop_paths
        st.success(f"Embedded {len(embA)} tracklets from A")

        st.info("Embedding tracklets from Video B...")
        embB = {}
        cropsB = {}
        b_out = os.path.join(work_root, "B_crops")
        ensure_dir(b_out)
        for tid, frames in tracksB.items():
            vec, crop_paths = aggregate_tracklet_embeddings(clip_model, clip_preprocess, clip_device, frames, b_out, max_samples=max_samples)
            if vec is not None:
                embB[tid] = vec
                cropsB[tid] = crop_paths
        st.success(f"Embedded {len(embB)} tracklets from B")

        if len(embA) == 0 or len(embB) == 0:
            st.error("No valid tracklet embeddings found in one of the videos. Try lowering the YOLO confidence or increasing frame sampling.")
        else:
            st.info("Comparing embeddings...")
            matches = compare_track_embeddings(embA, embB, threshold=similarity_threshold)

            if len(matches) == 0:
                st.warning("No matching objects found above the similarity threshold — videos likely contain different objects.")
            else:
                st.success(f"Found {len(matches)} matching track pairs (above threshold)")
                # show top matches
                for tidA, tidB, sim in matches[:10]:
                    st.markdown(f"**Match:** Video A track `{tidA}` ⇄ Video B track `{tidB}`  — similarity **{sim:.3f}**")
                    rowA = cropsA.get(tidA, [])
                    rowB = cropsB.get(tidB, [])
                    cols = st.columns(2)
                    with cols[0]:
                        st.write(f"Representative crops — Video A (track {tidA})")
                        for imgp in (rowA[:6] if len(rowA)>0 else []):
                            st.image(imgp, use_container_width=True)
                    with cols[1]:
                        st.write(f"Representative crops — Video B (track {tidB})")
                        for imgp in (rowB[:6] if len(rowB)>0 else []):
                            st.image(imgp, use_container_width=True)

            # Use LangChain to generate a short summary (requires OPENAI_API_KEY in env for OpenAI LLM)
            try:
                llm = OpenAI(temperature=0)
                prompt = PromptTemplate(
                    input_variables=["num_matches", "top_matches"],
                    template="""
You are an assistant that summarizes object matching results.
Number of matches: {num_matches}
Top matches text: {top_matches}
Provide a 2-3 sentence summary explaining whether the two videos likely contain the same object and the confidence reasoning.
""",
                )
                chain = LLMChain(llm=llm, prompt=prompt)
                top_text = "\n".join([f"A:{a} <-> B:{b} sim={s:.3f}" for a,b,s in matches[:5]])
                summary = chain.run({"num_matches": str(len(matches)), "top_matches": top_text})
                st.info("Summary (generated)")
                st.write(summary)
            except Exception as e:
                st.info("LangChain summary skipped (OpenAI LLM not configured or failed).")

    finally:
        # cleanup large temp files to free disk space
        try:
            shutil.rmtree(work_root)
        except Exception:
            pass

else:
    st.info("Upload two videos and press 'Run matching' to start.")

# Footer
st.markdown("---")
st.write("Built with YOLOv8 (ultralytics), CLIP, Streamlit, and optional LangChain summary.")
