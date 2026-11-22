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

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".gestione_mazzi_cache", "scanner_index")
SCRYFALL_HEADERS = {
    "User-Agent": "ManaStatDesktopApp/3.0",
    "Accept": "application/json",
}


def compute_card_descriptor(card_bgr: np.ndarray) -> np.ndarray:
    """
    Calcola una firma visiva compatta (96 float32 L2-normalizzati) da una carta 252x352:
    - 64 valori: gradienti DCT / luminanza dell'artwork centrale (struttura resistente a luce/lingua)
    - 32 valori: distribuzione colore LAB normalizzata di artwork + cornice
    """
    resized = cv2.resize(card_bgr, (252, 352), interpolation=cv2.INTER_AREA)

    # Zona artwork standard MTG (y: 14%..56%, x: 8%..92%)
    art = resized[49:197, 20:232]

    # 1) Struttura luminanza (pHash continuo 8x8 via DCT)
    gray = cv2.cvtColor(art, cv2.COLOR_BGR2GRAY)
    gray_32 = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(gray_32)
    low_freq = dct[:8, :8].flatten()
    low_freq[0] = 0.0
    norm_dct = np.linalg.norm(low_freq)
    if norm_dct > 1e-6:
        low_freq = low_freq / norm_dct

    # 2) Istogramma colore percettivo LAB (canali a e b, 4x4 bin) + bordo carta
    lab_art = cv2.cvtColor(art, cv2.COLOR_BGR2LAB)
    hist_ab = cv2.calcHist([lab_art], [1, 2], None, [4, 4], [0, 256, 0, 256]).flatten()
    norm_ab = np.linalg.norm(hist_ab)
    if norm_ab > 1e-6:
        hist_ab = hist_ab / norm_ab

    frame_top = resized[8:42, 14:238]
    lab_frame = cv2.cvtColor(frame_top, cv2.COLOR_BGR2LAB)
    hist_frame = cv2.calcHist([lab_frame], [1, 2], None, [4, 4], [0, 256, 0, 256]).flatten()
    norm_fr = np.linalg.norm(hist_frame)
    if norm_fr > 1e-6:
        hist_frame = hist_frame / norm_fr

    vec = np.concatenate([low_freq * 1.4, hist_ab * 0.8, hist_frame * 0.6]).astype(np.float32)
    total_norm = np.linalg.norm(vec)
    if total_norm > 1e-6:
        vec /= total_norm
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
    """Scarica rapidamente solo i set richiesti via /cards/search senza dover scaricare l'intero bulk JSON."""
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
        return card, vec
    except Exception:
        return None


def build_index(set_codes: list[str] | None = None, workers: int = 8):
    os.makedirs(CACHE_DIR, exist_ok=True)
    set_filter = {s.strip().lower() for s in set_codes if s.strip()} if set_codes else None

    if set_filter:
        cards = fetch_specific_sets(set_filter)
    else:
        cards = download_bulk_metadata()

    print(f"Carte da indicizzare: {len(cards)}")
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
                c_meta, vec = res
                c_meta.pop("thumb_url", None)
                meta_out.append(c_meta)
                vecs_out.append(vec)
            if done % 250 == 0 or done == len(cards):
                print(f"Progresso: {done}/{len(cards)} (indicizzate: {len(meta_out)})")

    if not vecs_out:
        print("Nessuna carta indicizzata.")
        sys.exit(1)

    matrix = np.vstack(vecs_out).astype(np.float32)
    np.save(os.path.join(CACHE_DIR, "vectors.npy"), matrix)
    with open(os.path.join(CACHE_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False)

    print(f"Indice completato! {len(meta_out)} carte salvate in {CACHE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Costruisce l'indice visivo delle carte Scryfall.")
    parser.add_argument(
        "--sets",
        nargs="*",
        help="Codici set opzionali per un test veloce (es. --sets mh3 fdn blb). Se omesso indicizza tutto.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Thread paralleli di download.")
    args = parser.parse_args()
    build_index(set_codes=args.sets, workers=args.workers)