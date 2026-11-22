# build_index.py
from __future__ import annotations

import os
import sys
import gzip
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests
import onnxruntime as ort

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".gestione_mazzi_cache", "scanner_index")
MODEL_PATH = os.path.join(LOCAL_DIR, "mobilenet_v3_small.onnx")

SCRYFALL_HEADERS = {
    "User-Agent": "ManaStatDesktopApp/3.0",
    "Accept": "application/json",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_ORT_SESSION: ort.InferenceSession | None = None


def _get_ort_session() -> ort.InferenceSession:
    global _ORT_SESSION
    if _ORT_SESSION is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Modello ONNX non trovato in {MODEL_PATH}. Esegui prima: python export_model.py"
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        _ORT_SESSION = ort.InferenceSession(
            MODEL_PATH,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    return _ORT_SESSION


def compute_card_descriptor(card_bgr: np.ndarray, crop_margin: float = 0.0) -> np.ndarray:
    """
    Calcola un embedding visivo AI (576 float32 L2-normalizzati) dall'artwork della carta
    usando MobileNetV3-Small via ONNX Runtime.
    crop_margin > 0 restringe leggermente il ritaglio (utile per carte in bustina/sleeve).
    """
    resized = cv2.resize(card_bgr, (252, 352), interpolation=cv2.INTER_AREA)

    # Zona artwork standard MTG (y: 14%..56%, x: 8%..92%)
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

    sess = _get_ort_session()
    vec = sess.run(["embedding"], {"input": inp})[0][0].astype(np.float32)

    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec /= norm
    return vec


def _normalize_card_entry(c: dict) -> dict | None:
    if c.get("layout") in ("token", "emblem", "art_series"):
        return None
    set_code = (c.get("set") or "").lower()
    uris = c.get("image_uris") or (
        c.get("card_faces", [{}])[0].get("image_uris") if c.get("card_faces") else None
    )
    if not uris:
        return None
    img_url = uris.get("small") or uris.get("normal")
    if not img_url:
        return None

    return {
        "scryfall_id": c.get("id"),
        "name": c.get("name"),
        "set_code": set_code.upper(),
        "set_name": c.get("set_name") or "",
        "collector_number": c.get("collector_number") or "",
        "released_at": c.get("released_at") or "",
        "rarity": c.get("rarity") or "",
        "type_line": c.get("type_line") or "",
        "mana_cost": c.get("mana_cost") or "",
        "cmc": c.get("cmc") or 0,
        "colors": c.get("colors") or [],
        "image_url": uris.get("normal") or img_url,
        "thumb_url": img_url,
        "price_eur": float(c["prices"]["eur"]) if (c.get("prices") or {}).get("eur") else None,
        "price_eur_foil": float(c["prices"]["eur_foil"]) if (c.get("prices") or {}).get("eur_foil") else None,
    }


def fetch_specific_sets(set_codes: set[str]) -> list[dict]:
    filtered: list[dict] = []
    for code in sorted(set_codes):
        print(f"Recupero metadati set [{code.upper()}] da Scryfall...")
        url = f"https://api.scryfall.com/cards/search?q=e%3A{code}&unique=prints"
        while url:
            resp = requests.get(url, headers=SCRYFALL_HEADERS, timeout=20)
            if resp.status_code != 200:
                print(f"Avviso set {code}: HTTP {resp.status_code} - {resp.text[:160]}")
                break
            body = resp.json()
            for raw_c in body.get("data", []):
                entry = _normalize_card_entry(raw_c)
                if entry:
                    filtered.append(entry)
            url = body.get("next_page") if body.get("has_more") else None
            if url:
                time.sleep(0.12)
    return filtered


def download_bulk_metadata() -> list[dict]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    bulk_meta_path = os.path.join(CACHE_DIR, "default_cards.jsonl.gz")

    need_download = True
    if os.path.exists(bulk_meta_path):
        age_hours = (time.time() - os.path.getmtime(bulk_meta_path)) / 3600.0
        if age_hours < 48:
            need_download = False

    if need_download:
        print("Scaricamento catalogo completo Scryfall (default_cards)...")
        r = requests.get(
            "https://api.scryfall.com/bulk-data/default-cards",
            headers=SCRYFALL_HEADERS,
            timeout=30,
        )
        data = r.json()
        download_uri = data.get("jsonl_download_uri") or data.get("download_uri")
        if r.status_code != 200 or not download_uri:
            raise RuntimeError(f"Risposta inattesa da Scryfall ({r.status_code}): {data}")

        with requests.get(download_uri, headers=SCRYFALL_HEADERS, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            with open(bulk_meta_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        print("Catalogo salvato in cache.")

    filtered = []
    if bulk_meta_path.endswith(".gz"):
        with gzip.open(bulk_meta_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = _normalize_card_entry(json.loads(line))
                if entry:
                    filtered.append(entry)
    else:
        with open(bulk_meta_path, "r", encoding="utf-8") as f:
            cards = json.load(f)
        for c in cards:
            entry = _normalize_card_entry(c)
            if entry:
                filtered.append(entry)
    return filtered


def _process_single_card(card: dict) -> tuple[dict, np.ndarray] | None:
    try:
        resp = requests.get(
            card["thumb_url"],
            headers={"User-Agent": "ManaStatDesktopApp/3.0"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        vec = compute_card_descriptor(bgr)
        # Quantizzazione compatta int8 (1 byte per valore -> ~65 MB per 114k carte)
        vec_q = np.clip(np.round(vec * 127.0), -127, 127).astype(np.int8)
        return card, vec_q
    except Exception:
        return None


def build_index(set_codes: list[str] | None = None, workers: int = 8):
    # Pre-carica la sessione ONNX sul thread principale
    _get_ort_session()

    set_filter = {s.strip().lower() for s in set_codes if s.strip()} if set_codes else None
    if set_filter:
        cards = fetch_specific_sets(set_filter)
    else:
        cards = download_bulk_metadata()

    print(f"Carte da indicizzare (MobileNetV3 ONNX): {len(cards)}")
    if not cards:
        print("Nessuna carta trovata.")
        sys.exit(1)

    meta_out: list[dict] = []
    vecs_out: list[np.ndarray] = []

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_single_card, c) for c in cards]
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            if res is not None:
                c_meta, vec_q = res
                c_meta.pop("thumb_url", None)
                meta_out.append(c_meta)
                vecs_out.append(vec_q)
            if done % 250 == 0 or done == len(cards):
                print(f"Progresso: {done}/{len(cards)} (indicizzate: {len(meta_out)})")

    if not vecs_out:
        print("Nessuna carta indicizzata.")
        sys.exit(1)

    matrix = np.vstack(vecs_out).astype(np.int8)
    np.save(os.path.join(LOCAL_DIR, "vectors.npy"), matrix)
    with open(os.path.join(LOCAL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False)

    print(f"Indice AI completato! {len(meta_out)} carte salvate in {LOCAL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Costruisce l'indice visivo AI (MobileNetV3) delle carte Scryfall.")
    parser.add_argument(
        "--sets",
        nargs="*",
        help="Codici set opzionali per un test veloce. Se omesso indicizza tutto.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Thread paralleli di download/inferenza.")
    args = parser.parse_args()
    build_index(set_codes=args.sets, workers=args.workers)