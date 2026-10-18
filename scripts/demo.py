"""Demo: tek plan görüntüsü → katmanlı sonuç görüntüsü + graf JSON.

Sunumda canlı çalıştırmak için; ekran çıktısı ve şekil metinleri İngilizce.
CVC-FP planı adıyla verilirse cevapla yan yana görüntü ve plan başına
doğru/fazla/eksik sayıları da yazılır.

Kullanım:
    uv run python scripts/demo.py yol/plan.png [--cikti results/demo]
    uv run python scripts/demo.py --cvcfp IIa_AP3301
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from floorplan_parser.akis import plani_coz
from floorplan_parser.cvcfp import ornekler, yukle
from floorplan_parser.degerlendirme import bolge_eslesmeleri, poligon_eslesmeleri
from floorplan_parser.gorsel import (
    VARSAYILAN_LEJANT,
    cevap_katmanlari,
    graf_sozluk,
    katmanlar,
    lejant,
)
from floorplan_parser.goruntu import oku, yaz

KOK = Path(__file__).resolve().parents[1]


def baslik(goruntu: np.ndarray, metin: str) -> np.ndarray:
    serit = np.full((90, goruntu.shape[1], 3), 255, np.uint8)
    cv2.putText(serit, metin, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4, cv2.LINE_AA)
    return np.vstack([serit, goruntu])


def main() -> None:
    ap = argparse.ArgumentParser(description="Rule-based floor plan parser: demo on one plan.")
    ap.add_argument("plan", nargs="?", help="floor plan image (png/jpg)")
    ap.add_argument("--cvcfp", help="CVC-FP plan name instead of a file, e.g. IIa_AP3301")
    ap.add_argument("--cikti", default=str(KOK / "results" / "demo"), help="output folder")
    a = ap.parse_args()
    if not a.plan and not a.cvcfp:
        ap.error("give an image path or --cvcfp NAME")

    cevap = None
    if a.cvcfp:
        o = next((o for o in ornekler() if o.ad == a.cvcfp), None)
        if o is None:
            ap.error(f"unknown CVC-FP plan: {a.cvcfp}")
        gri, cevap = yukle(o)
        ad = o.ad
    else:
        gri = oku(a.plan)
        ad = Path(a.plan).stem

    t0 = time.perf_counter()
    s = plani_coz(gri)
    sure = time.perf_counter() - t0
    n_oda = s.graf.number_of_nodes() - 1
    print(f"plan: {ad}  ({gri.shape[1]} x {gri.shape[0]} px)  parsed in {sure:.1f} s")
    print(
        f"  walls: {len(s.kalin_parcalar)} thick + {len(s.ince_parcalar)} thin segments"
        f"   doors: {len(s.kapilar)}   windows: {len(s.pencereler)}"
        f"   rooms: {n_oda}   door edges: {s.graf.number_of_edges()}"
    )

    cikti = Path(a.cikti)
    sonuc = katmanlar(gri, s)
    lejant(sonuc, VARSAYILAN_LEJANT, olcek=1.6)
    yaz(cikti / f"{ad}_result.png", sonuc)
    (cikti / f"{ad}_graph.json").write_text(json.dumps(graf_sozluk(s), indent=1), encoding="utf-8")
    print(f"  -> {cikti / (ad + '_result.png')}")
    print(f"  -> {cikti / (ad + '_graph.json')}")

    if cevap is not None:
        for ad_t, tahmin, cv in (
            ("doors", [k.poligon() for k in s.kapilar], cevap.sinif("Door")),
            ("windows", [p.poligon() for p in s.pencereler], cevap.sinif("Window")),
        ):
            d = len(poligon_eslesmeleri(tahmin, [n.noktalar for n in cv]))
            print(f"  {ad_t:8} correct {d:3}  extra {len(tahmin) - d:3}  missed {len(cv) - d:3}")
        oda_c = cevap.sinif("Room")
        d = len(bolge_eslesmeleri(s.oda_etiket, [n.noktalar for n in oda_c]))
        print(f"  {'rooms':8} correct {d:3}  extra {n_oda - d:3}  missed {len(oda_c) - d:3}")
        yan = np.hstack(
            [
                baslik(cevap_katmanlari(gri, cevap), "ground truth (CVC-FP)"),
                baslik(katmanlar(gri, s, graf=False), "this method"),
            ]
        )
        lejant(yan, VARSAYILAN_LEJANT[:3], olcek=1.6, konum=(10, 100))
        yaz(cikti / f"{ad}_vs_truth.png", yan)
        print(f"  -> {cikti / (ad + '_vs_truth.png')}")


if __name__ == "__main__":
    main()
