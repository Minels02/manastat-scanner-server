from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

try:
    from usearch.index import Index as USearchIndex
except Exception:
    USearchIndex = None

SERVER_VERSION = "v2.1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mobilenet_v3_small.onnx")
VECTORS_PATH = os.path.join(BASE_DIR, "vectors.npy")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.json")

ORT_SESSION: ort.InferenceSession | None = None
ORT_INPUT_NAME: str = ""
VECTORS: np.ndarray | None = None
METADATA: list[dict[str, Any]] = []
SET_TO_INDICES: dict[str, np.ndarray] = {}
AVAILABLE_SETS: list[dict[str, Any]] = []

USEARCH_INDEX: Any = None
USEARCH_READY: bool = False

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _build_usearch_background() -> None:
    global USEARCH_INDEX, USEARCH_READY
    if USearchIndex is None or VECTORS is None or len(VECTORS) == 0:
        return
    try:
        n, dim = VECTORS.shape
        is_int8 = VECTORS.dtype == np.int8
        idx = USearchIndex(
            ndim=dim,
            metric="cos",
            dtype="i8" if is_int8 else "f32",
            connectivity=24,
            expansion_add=96,
            expansion_search=96,
        )
        keys = np.arange(n, dtype=np.uint64)
        batch_size = 16384
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx.add(keys[start:end], VECTORS[start:end], threads=1)
        USEARCH_INDEX = idx
        USEARCH_READY = True
        print(f"[Scanner {SERVER_VERSION}] Indice USearch pronto ({n} carte).")
    except Exception as exc:
        print(f"[Scanner] USearch non disponibile, uso fallback NumPy: {exc}")
        USEARCH_INDEX = None
        USEARCH_READY = False


def load_resources() -> None:
    global ORT_SESSION, ORT_INPUT_NAME, VECTORS, METADATA, SET_TO_INDICES, AVAILABLE_SETS

    if os.path.exists(MODEL_PATH):
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ORT_SESSION = ort.InferenceSession(
            MODEL_PATH,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        ORT_INPUT_NAME = ORT_SESSION.get_inputs()[0].name
        print(f"[Scanner {SERVER_VERSION}] Modello MobileNetV3 ONNX caricato (1 thread).")
    else:
        print(f"[Scanner] ATTENZIONE: modello non trovato in {MODEL_PATH}")

    if os.path.exists(VECTORS_PATH) and os.path.exists(METADATA_PATH):
        VECTORS = np.load(VECTORS_PATH, mmap_mode="r")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            METADATA = json.load(f)

        by_set: dict[str, list[int]] = defaultdict(list)
        set_names: dict[str, str] = {}
        for i, item in enumerate(METADATA):
            sc = str(item.get("set_code") or "").strip().upper()
            if not sc:
                continue
            by_set[sc].append(i)
            if sc not in set_names:
                set_names[sc] = str(item.get("set_name") or sc).strip()

        SET_TO_INDICES = {
            sc: np.array(idxs, dtype=np.int32) for sc, idxs in by_set.items()
        }
        AVAILABLE_SETS = sorted(
            [
                {"code": sc, "name": set_names.get(sc, sc), "count": len(idxs)}
                for sc, idxs in by_set.items()
            ],
            key=lambda x: x["name"].lower(),
        )
        print(f"[Scanner {SERVER_VERSION}] Indice caricato: {len(METADATA)} carte, {len(AVAILABLE_SETS)} set.")

        threading.Thread(target=_build_usearch_background, daemon=True).start()
    else:
        print("[Scanner] ATTENZIONE: vectors.npy o metadata.json mancanti.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_resources()
    yield


app = FastAPI(title="ManaStat Scanner Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": SERVER_VERSION,
        "model": "mobilenet_v3_small_onnx" if ORT_SESSION is not None else "missing",
        "indexed_cards": len(METADATA),
        "sets_count": len(AVAILABLE_SETS),
        "usearch_ready": USEARCH_READY,
    }


@app.get("/sets")
def list_sets() -> list[dict[str, Any]]:
    return AVAILABLE_SETS


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
    for pt in inner:
        if cv2.pointPolygonTest(outer, (float(pt[0]), float(pt[1])), False) < 0:
            return False
    return True


def detect_and_warp_card(
    frame_bgr: np.ndarray,
) -> tuple[np.ndarray | None, list[list[float]] | None]:
    fh, fw = frame_bgr.shape[:2]
    frame_area = float(fh * fw)

    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    b_ch, g_ch, r_ch = cv2.split(blurred)
    edges = cv2.bitwise_or(
        cv2.Canny(b_ch, 25, 90),
        cv2.bitwise_or(cv2.Canny(g_ch, 25, 90), cv2.Canny(r_ch, 25, 90)),
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(
        closed, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
    )

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    candidates: list[tuple[float, float, np.ndarray]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.06 or area > frame_area * 0.92:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        quad = None
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            rect_area = rw * rh
            if rect_area > 0 and (area / rect_area) >= 0.78:
                quad = cv2.boxPoints(rect).astype(np.float32)

        if quad is None:
            continue

        ordered = order_quad_points(quad)
        w_top = np.linalg.norm(ordered[1] - ordered[0])
        w_bot = np.linalg.norm(ordered[2] - ordered[3])
        h_left = np.linalg.norm(ordered[3] - ordered[0])
        h_right = np.linalg.norm(ordered[2] - ordered[1])

        avg_w = (w_top + w_bot) * 0.5
        avg_h = (h_left + h_right) * 0.5
        if avg_w < 40 or avg_h < 40:
            continue

        ratio = max(avg_h, avg_w) / max(1.0, min(avg_h, avg_w))
        if not (1.18 <= ratio <= 1.65):
            continue

        w_diff = abs(w_top - w_bot) / max(1.0, avg_w)
        h_diff = abs(h_left - h_right) / max(1.0, avg_h)
        if w_diff > 0.30 or h_diff > 0.30:
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


def normalize_lighting_bgr(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge((l_eq, a_ch, b_ch))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def compute_card_descriptor(card_bgr: np.ndarray, crop_margin: float = 0.0) -> np.ndarray:
    """Identico a build_index.py: ritaglia la zona artwork MTG (y: 49..197, x: 20..232)."""
    assert ORT_SESSION is not None
    resized = cv2.resize(card_bgr, (252, 352), interpolation=cv2.INTER_AREA)

    y1, y2 = 49, 197
    x1, x2 = 20, 232
    if crop_margin > 0.0:
        dy = int((y2 - y1) * crop_margin)
        dx = int((x2 - x1) * crop_margin)
        y1, y2 = y1 + dy, y2 - dy
        x1, x2 = x1 + dx, x2 - dx

    art = resized[y1:y2, x1:x2]
    rgb = cv2.cvtColor(art, cv2.COLOR_BGR2RGB)
    img_224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    img_224 = (img_224 - IMAGENET_MEAN) / IMAGENET_STD
    inp = np.transpose(img_224, (2, 0, 1))[None, ...].astype(np.float32)

    out = ORT_SESSION.run(None, {ORT_INPUT_NAME: inp})[0]
    vec = out.reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-6:
        vec /= norm
    return vec


def _search_single_vector(
    q_vec: np.ndarray,
    top_k: int,
    subset_indices: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    assert VECTORS is not None
    is_int8 = VECTORS.dtype == np.int8
    scale = (1.0 / 127.0) if is_int8 else 1.0

    # 1) Ricerca esatta sul sottoinsieme di espansioni filtrate
    if subset_indices is not None:
        if len(subset_indices) == 0:
            return []
        sub = VECTORS[subset_indices].astype(np.float32)
        sims = sub @ (q_vec * scale)
        k = min(top_k, len(sims))
        top_local = np.argpartition(-sims, k - 1)[:k]
        top_local = top_local[np.argsort(-sims[top_local])]
        return [(int(subset_indices[i]), float(sims[i])) for i in top_local]

    # 2) Ricerca globale USearch + ri-ordinamento esatto sui top 50
    if USEARCH_READY and USEARCH_INDEX is not None:
        q_in = np.clip(np.round(q_vec * 127.0), -127, 127).astype(np.int8) if is_int8 else q_vec
        fetch_k = max(top_k, 50)
        matches = USEARCH_INDEX.search(q_in, fetch_k)
        cand_keys = np.array([int(m.key) for m in matches], dtype=np.int64)
        if len(cand_keys) == 0:
            return []
        sub = VECTORS[cand_keys].astype(np.float32)
        sims = sub @ (q_vec * scale)
        k = min(top_k, len(sims))
        top_local = np.argpartition(-sims, k - 1)[:k]
        top_local = top_local[np.argsort(-sims[top_local])]
        return [(int(cand_keys[i]), float(sims[i])) for i in top_local]

    # 3) Fallback NumPy 1D
    q_scaled = q_vec * scale
    n = VECTORS.shape[0]
    sims = np.empty(n, dtype=np.float32)
    chunk_size = 32768
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims[start:end] = VECTORS[start:end].astype(np.float32) @ q_scaled

    k = min(top_k, n)
    top_local = np.argpartition(-sims, k - 1)[:k]
    top_local = top_local[np.argsort(-sims[top_local])]
    return [(int(i), float(sims[i])) for i in top_local]


def match_warped_card(
    card_bgr: np.ndarray,
    top_k: int = 5,
    allowed_sets: list[str] | None = None,
) -> list[dict[str, Any]]:
    if VECTORS is None or len(METADATA) == 0 or ORT_SESSION is None:
        return []

    subset_indices: np.ndarray | None = None
    if allowed_sets:
        idx_parts = [
            SET_TO_INDICES[s.upper()]
            for s in allowed_sets
            if s.upper() in SET_TO_INDICES
        ]
        if idx_parts:
            subset_indices = np.concatenate(idx_parts)
        else:
            return []

    norm_card = normalize_lighting_bgr(card_bgr)
    internal_k = max(top_k, 10)

    # 1° passaggio veloce: carta dritta standard
    q_main = compute_card_descriptor(norm_card, crop_margin=0.0)
    hits = _search_single_vector(q_main, top_k=internal_k, subset_indices=subset_indices)
    best_score = hits[0][1] if hits else 0.0

    merged: dict[int, float] = {idx: sc for idx, sc in hits}

    # 2° passaggio leggero solo per carte in bustina/sleeve se il punteggio è basso
    if best_score < 0.72:
        q_sleeve = compute_card_descriptor(norm_card, crop_margin=0.04)
        for idx, sc in _search_single_vector(q_sleeve, top_k=internal_k, subset_indices=subset_indices):
            if idx not in merged or sc > merged[idx]:
                merged[idx] = sc

    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)

    results = []
    for global_idx, score in ranked:
        if score < 0.62:
            continue
        item = dict(METADATA[global_idx])
        item["confidence"] = round(score, 4)
        results.append(item)
        if len(results) >= top_k:
            break
    return results


def analyze_frame_bytes(
    jpeg_bytes: bytes,
    allowed_sets: list[str] | None = None,
) -> dict[str, Any]:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"detected": False, "quad": None, "matches": []}

    warped, norm_quad = detect_and_warp_card(frame)
    if warped is None:
        return {"detected": False, "quad": None, "matches": []}

    matches = match_warped_card(warped, top_k=5, allowed_sets=allowed_sets)
    return {
        "detected": True,
        "quad": norm_quad,
        "matches": matches,
    }


@app.websocket("/ws/scan")
async def scan_websocket(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"type": "hello", "server_version": SERVER_VERSION})
    allowed_sets: list[str] | None = None
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                    if isinstance(data, dict) and data.get("type") == "set_filter":
                        raw_sets = data.get("sets") or []
                        cleaned = [
                            str(s).strip().upper()
                            for s in raw_sets
                            if str(s).strip()
                        ]
                        allowed_sets = cleaned if cleaned else None
                        await websocket.send_json(
                            {"type": "filter_ack", "sets": allowed_sets or []}
                        )
                except Exception:
                    pass
                continue

            if msg.get("bytes") is not None:
                res = analyze_frame_bytes(msg["bytes"], allowed_sets=allowed_sets)
                await websocket.send_json(res)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[Scanner] Errore WebSocket: {exc}")