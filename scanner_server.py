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

from build_index import CACHE_DIR, compute_card_descriptor

app = FastAPI(title="ManaStat Vision Scanner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTORS: np.ndarray | None = None
METADATA: list[dict[str, Any]] = []


def load_visual_index():
    global VECTORS, METADATA
    local_dir = os.path.dirname(os.path.abspath(__file__))
    for base_dir in [local_dir, CACHE_DIR]:
        vec_path = os.path.join(base_dir, "vectors.npy")
        meta_path = os.path.join(base_dir, "metadata.json")
        if os.path.exists(vec_path) and os.path.exists(meta_path):
            VECTORS = np.load(vec_path).astype(np.float32)
            with open(meta_path, "r", encoding="utf-8") as f:
                METADATA = json.load(f)
            print(f"[Scanner] Indice caricato da {base_dir}: {len(METADATA)} carte pronte in RAM.")
            return
    print("[Scanner] ATTENZIONE: Indice non trovato. Esegui prima: python build_index.py")
    
@app.on_event("startup")
def on_startup():
    load_visual_index()


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Ordina 4 punti nel formato: [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _quad_contains(outer: np.ndarray, inner: np.ndarray) -> bool:
    """Restituisce True se almeno 3 vertici di inner cadono dentro il poligono outer."""
    inside_count = 0
    for pt in inner:
        if cv2.pointPolygonTest(outer, (float(pt[0]), float(pt[1])), False) >= 0:
            inside_count += 1
    return inside_count >= 3


def detect_and_warp_card(frame_bgr: np.ndarray) -> tuple[np.ndarray | None, list[list[float]] | None]:
    """
    Cerca il contorno esterno della carta Magic nel fotogramma, ignorando
    i riquadri interni (box del testo o dell'artwork).
    """
    fh, fw = frame_bgr.shape[:2]
    frame_area = float(fh * fw)

    # Sfocatura leggera + gradiente su tutti e 3 i canali BGR (non solo grigio,
    # così rileva anche bordi neri su superfici scure o colorate)
    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    b_ch, g_ch, r_ch = cv2.split(blurred)
    edges_b = cv2.Canny(b_ch, 35, 110)
    edges_g = cv2.Canny(g_ch, 35, 110)
    edges_r = cv2.Canny(r_ch, 35, 110)
    edges = cv2.bitwise_or(edges_b, cv2.bitwise_or(edges_g, edges_r))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    candidates: list[tuple[float, float, np.ndarray]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Una carta inquadrata occupa in genere almeno il 12% del fotogramma;
        # i box interni del testo sono molto più piccoli a parità di inquadratura.
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

        # Rapporto carta Magic: 63x88 mm -> ~1.397
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

    # Scarta qualsiasi quadrilatero che sia contenuto dentro un candidato più grande
    # (es. il riquadro del testo delle regole dentro il bordo esterno della carta)
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

    # Se la carta è inquadrata in orizzontale, ruota l'ordine dei vertici per mantenerla verticale
    w_est = np.linalg.norm(best_quad[1] - best_quad[0])
    h_est = np.linalg.norm(best_quad[3] - best_quad[0])
    if w_est > h_est:
        best_quad = np.roll(best_quad, -1, axis=0)

    dst_pts = np.array(
        [[0, 0], [251, 0], [251, 351], [0, 351]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(best_quad, dst_pts)
    warped = cv2.warpPerspective(frame_bgr, M, (252, 352))

    norm_quad = [
        [round(float(pt[0]) / fw, 4), round(float(pt[1]) / fh, 4)]
        for pt in best_quad
    ]
    return warped, norm_quad


def match_warped_card(card_bgr: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
    if VECTORS is None or len(METADATA) == 0:
        return []

    # Calcola firma sia dritta sia ruotata di 180° (se la carta è capovolta sul tavolo)
    vec_0 = compute_card_descriptor(card_bgr)
    vec_180 = compute_card_descriptor(cv2.rotate(card_bgr, cv2.ROTATE_180))

    sims_0 = VECTORS @ vec_0
    sims_180 = VECTORS @ vec_180
    sims = np.maximum(sims_0, sims_180)

    k = min(top_k, len(METADATA))
    top_indices = np.argpartition(-sims, k - 1)[:k]
    top_indices = top_indices[np.argsort(-sims[top_indices])]

    results = []
    for idx in top_indices:
        score = float(sims[idx])
        if score < 0.62:
            continue
        item = dict(METADATA[int(idx)])
        item["confidence"] = round(score, 4)
        results.append(item)
    return results


def analyze_frame_bytes(img_bytes: bytes) -> dict[str, Any]:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"detected": False, "quad": None, "matches": []}

    warped, quad = detect_and_warp_card(frame)
    if warped is None:
        return {"detected": False, "quad": None, "matches": []}

    matches = match_warped_card(warped, top_k=5)
    return {
        "detected": True,
        "quad": quad,
        "matches": matches,
    }


@app.get("/health")
def health():
    return {"status": "ok", "indexed_cards": len(METADATA)}


@app.post("/scan-frame")
async def scan_frame_http(file: UploadFile = File(...)):
    content = await file.read()
    return analyze_frame_bytes(content)


@app.websocket("/ws/scan")
async def scan_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                res = analyze_frame_bytes(msg["bytes"])
                await websocket.send_json(res)
            elif "text" in msg and msg["text"] is not None:
                raw = msg["text"]
                if "," in raw:
                    raw = raw.split(",", 1)[1]
                img_bytes = base64.b64decode(raw)
                res = analyze_frame_bytes(img_bytes)
                await websocket.send_json(res)
    except WebSocketDisconnect:
        pass