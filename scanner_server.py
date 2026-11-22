# scanner_server.py
from __future__ import annotations

import os
import json
import base64
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from build_index import CACHE_DIR, compute_card_descriptor, _get_ort_session

app = FastAPI(title="ManaStat Vision Scanner API (ONNX)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTORS: np.ndarray | None = None  # int8 (N, 576) oppure float32
METADATA: list[dict[str, Any]] = []
SET_TO_INDICES: dict[str, np.ndarray] = {}
AVAILABLE_SETS: list[dict[str, Any]] = []


def load_visual_index():
    global VECTORS, METADATA, SET_TO_INDICES, AVAILABLE_SETS
    local_dir = os.path.dirname(os.path.abspath(__file__))

    # Pre-carica il modello ONNX
    try:
        _get_ort_session()
        print("[Scanner] Modello MobileNetV3 ONNX caricato.")
    except Exception as e:
        print(f"[Scanner] AVVISO modello ONNX: {e}")

    for base_dir in [local_dir, CACHE_DIR]:
        vec_path = os.path.join(base_dir, "vectors.npy")
        meta_path = os.path.join(base_dir, "metadata.json")
        if os.path.exists(vec_path) and os.path.exists(meta_path):
            VECTORS = np.load(vec_path)  # mantiene int8 per risparmiare RAM su Render
            with open(meta_path, "r", encoding="utf-8") as f:
                METADATA = json.load(f)

            # Indice rapido per filtro espansioni
            by_set: dict[str, list[int]] = {}
            set_names: dict[str, str] = {}
            for idx, m in enumerate(METADATA):
                code = (m.get("set_code") or "").upper()
                if not code:
                    continue
                by_set.setdefault(code, []).append(idx)
                if code not in set_names and m.get("set_name"):
                    set_names[code] = m["set_name"]

            SET_TO_INDICES = {k: np.array(v, dtype=np.int32) for k, v in by_set.items()}
            AVAILABLE_SETS = sorted(
                [{"code": k, "name": set_names.get(k, k), "count": len(v)} for k, v in by_set.items()],
                key=lambda x: x["name"].lower(),
            )
            print(f"[Scanner] Indice caricato da {base_dir}: {len(METADATA)} carte ({VECTORS.dtype}, {VECTORS.shape}).")
            return

    print("[Scanner] ATTENZIONE: Indice non trovato. Esegui prima: python build_index.py")


@app.on_event("startup")
def on_startup():
    load_visual_index()


def normalize_lighting_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Bilanciamento del bianco Gray World leggero per ridurre dominanti di luce artificiale."""
    img = img_bgr.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    if avg_gray < 1e-3:
        return img_bgr
    img[:, :, 0] *= np.clip(avg_gray / max(avg_b, 1.0), 0.75, 1.35)
    img[:, :, 1] *= np.clip(avg_gray / max(avg_g, 1.0), 0.75, 1.35)
    img[:, :, 2] *= np.clip(avg_gray / max(avg_r, 1.0), 0.75, 1.35)
    return np.clip(img, 0, 255).astype(np.uint8)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _quad_contains(outer: np.ndarray, inner: np.ndarray) -> bool:
    inside_count = 0
    for pt in inner:
        if cv2.pointPolygonTest(outer, (float(pt[0]), float(pt[1])), False) >= 0:
            inside_count += 1
    return inside_count >= 3


def detect_and_warp_card(frame_bgr: np.ndarray) -> tuple[np.ndarray | None, list[list[float]] | None]:
    fh, fw = frame_bgr.shape[:2]
    frame_area = float(fh * fw)

    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    b_ch, g_ch, r_ch = cv2.split(blurred)
    edges = cv2.bitwise_or(
        cv2.Canny(b_ch, 35, 110),
        cv2.bitwise_or(cv2.Canny(g_ch, 35, 110), cv2.Canny(r_ch, 35, 110)),
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    candidates: list[tuple[float, float, np.ndarray]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.12 or area > frame_area * 0.90:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        ordered = order_quad_points(approx.reshape(4, 2).astype(np.float32))
        w_top = np.linalg.norm(ordered[1] - ordered[0])
        w_bot = np.linalg.norm(ordered[2] - ordered[3])
        h_left = np.linalg.norm(ordered[3] - ordered[0])
        h_right = np.linalg.norm(ordered[2] - ordered[1])

        avg_w = (w_top + w_bot) * 0.5
        avg_h = (h_left + h_right) * 0.5
        if avg_w < 50 or avg_h < 50:
            continue

        ratio = max(avg_h, avg_w) / max(1.0, min(avg_h, avg_w))
        if not (1.22 <= ratio <= 1.62):
            continue

        w_diff = abs(w_top - w_bot) / max(1.0, avg_w)
        h_diff = abs(h_left - h_right) / max(1.0, avg_h)
        if w_diff > 0.25 or h_diff > 0.25:
            continue

        ratio_err = abs(ratio - 1.397)
        score = area / (1.0 + ratio_err * 4.0)
        candidates.append((score, area, ordered))

    if not candidates:
        return None, None

    outer_candidates = []
    for i, (score_i, area_i, quad_i) in enumerate(candidates):
        is_inner = False
        for j, (_, area_j, quad_j) in enumerate(candidates):
            if i != j and area_j > area_i * 1.25 and _quad_contains(quad_j, quad_i):
                is_inner = True
                break
        if not is_inner:
            outer_candidates.append((score_i, area_i, quad_i))

    pool = outer_candidates if outer_candidates else candidates
    pool.sort(key=lambda x: x[0], reverse=True)
    best_quad = pool[0][2]

    w_est = np.linalg.norm(best_quad[1] - best_quad[0])
    h_est = np.linalg.norm(best_quad[3] - best_quad[0])
    if w_est > h_est:
        best_quad = np.roll(best_quad, -1, axis=0)

    dst_pts = np.array([[0, 0], [251, 0], [251, 351], [0, 351]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(best_quad, dst_pts)
    warped = cv2.warpPerspective(frame_bgr, M, (252, 352))

    norm_quad = [
        [round(float(pt[0]) / fw, 4), round(float(pt[1]) / fh, 4)]
        for pt in best_quad
    ]
    return warped, norm_quad


def _chunked_max_similarity(
    queries: np.ndarray,  # (Q, D) float32
    subset_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Calcola la similarità coseno massima tra Q varianti della carta e l'indice int8 senza picchi di RAM."""
    assert VECTORS is not None
    is_int8 = VECTORS.dtype == np.int8
    scale = (1.0 / 127.0) if is_int8 else 1.0
    q_t = (queries * scale).T  # (D, Q)

    if subset_indices is not None:
        sub = VECTORS[subset_indices].astype(np.float32)
        sims = np.max(sub @ q_t, axis=1)
        return sims, subset_indices

    n = VECTORS.shape[0]
    sims = np.empty(n, dtype=np.float32)
    chunk_size = 16384
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = VECTORS[start:end].astype(np.float32)
        sims[start:end] = np.max(block @ q_t, axis=1)
    return sims, None


def match_warped_card(
    card_bgr: np.ndarray,
    top_k: int = 5,
    allowed_sets: list[str] | None = None,
) -> list[dict[str, Any]]:
    if VECTORS is None or len(METADATA) == 0:
        return []

    subset_indices: np.ndarray | None = None
    if allowed_sets:
        idx_parts = [SET_TO_INDICES[s.upper()] for s in allowed_sets if s.upper() in SET_TO_INDICES]
        if idx_parts:
            subset_indices = np.concatenate(idx_parts)

    norm_card = normalize_lighting_bgr(card_bgr)

    # 1° passaggio: solo dritta standard (1 inferenza + ricerca 1D)
    q_main = compute_card_descriptor(norm_card, crop_margin=0.0).astype(np.float32)
    sims, idx_map = _chunked_max_similarity(q_main, subset_indices)
    if len(sims) == 0:
        return []

    # Fallback leggero solo per carte in bustina/sleeve se il primo score è basso
    if float(np.max(sims)) < 0.72:
        q_sleeve = compute_card_descriptor(norm_card, crop_margin=0.04).astype(np.float32)
        sims_sleeve, _ = _chunked_max_similarity(q_sleeve, subset_indices)
        sims = np.maximum(sims, sims_sleeve)

    k = min(top_k, len(sims))
    top_local = np.argpartition(-sims, k - 1)[:k]
    top_local = top_local[np.argsort(-sims[top_local])]

    results = []
    for loc_idx in top_local:
        score = float(sims[loc_idx])
        if score < 0.58:
            continue
        global_idx = int(idx_map[loc_idx]) if idx_map is not None else int(loc_idx)
        item = dict(METADATA[global_idx])
        item["confidence"] = round(score, 4)
        results.append(item)
    return results


def analyze_frame_bytes(
    img_bytes: bytes,
    allowed_sets: list[str] | None = None,
) -> dict[str, Any]:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"detected": False, "quad": None, "matches": []}

    warped, quad = detect_and_warp_card(frame)
    if warped is None:
        return {"detected": False, "quad": None, "matches": []}

    matches = match_warped_card(warped, top_k=5, allowed_sets=allowed_sets)
    return {
        "detected": True,
        "quad": quad,
        "matches": matches,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_cards": len(METADATA),
        "model": "mobilenet_v3_small_onnx",
    }


@app.get("/sets")
def list_sets():
    return {"sets": AVAILABLE_SETS}


@app.post("/scan-frame")
async def scan_frame_http(file: UploadFile = File(...)):
    content = await file.read()
    return analyze_frame_bytes(content)


@app.websocket("/ws/scan")
async def scan_websocket(websocket: WebSocket):
    await websocket.accept()
    allowed_sets: list[str] | None = None
    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                res = analyze_frame_bytes(msg["bytes"], allowed_sets=allowed_sets)
                await websocket.send_json(res)
            elif "text" in msg and msg["text"] is not None:
                raw = msg["text"].strip()
                if raw.startswith("{"):
                    try:
                        cfg = json.loads(raw)
                        if cfg.get("type") == "config":
                            sets = cfg.get("sets")
                            allowed_sets = [str(s).upper() for s in sets if s] if sets else None
                            await websocket.send_json({"type": "config_ack", "sets": allowed_sets or []})
                            continue
                    except Exception:
                        pass
                if "," in raw:
                    raw = raw.split(",", 1)[1]
                img_bytes = base64.b64decode(raw)
                res = analyze_frame_bytes(img_bytes, allowed_sets=allowed_sets)
                await websocket.send_json(res)
    except WebSocketDisconnect:
        pass