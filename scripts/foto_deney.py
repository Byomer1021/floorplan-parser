"""Taramadan fotoğrafa: akış açılı telefon fotoğrafında ne kadar kaybediyor?

Geliştirme bölümünün dolu duvarlı planlarından sentetik fotoğraf üretilir
(``foto_benzetim``), cevap aynı dönüşümle taşınır ve ``degerlendir.py`` ile
aynı ölçüler alınır (``degerlendirme.plan_puani``). Test bölümüne dokunulmaz
(kararlar K-005).

Koşullar (kamera eğimi derece, kırışık sayısı):

- ``tarama``: orijinal tarama (geliştirme sayılarını birebir vermeli).
- ``ham``: fotoğraf doğrudan akışa.
- ``kahin``: gerçek köşelerle perspektif düzeltme + ışık düzeltme (ön işlemenin
  ulaşabileceği en iyi, arayüzde köşeleri kullanıcının işaretlemesine denk).
- ``oto``: köşeler otomatik (``foto.kagit_bul``) + ışık düzeltme.
- ``oto_isiksiz``: otomatik köşe, ışık düzeltme yok.

``--ikile otsu``: düzeltilmiş sayfa doğrudan akışa (akışın Otsu'su).
``--ikile kagit``: ``foto.foto_ikile`` (kâğıda göre eşik, duvar maskesi Otsu
ile kırpılır), uygulamadaki ``foto.fotograftan_coz`` ile aynı.

Kullanım: uv run python scripts/foto_deney.py [--plan 10] [--is 4] [--ikile kagit]
"""

import argparse
import csv
import math
import time
import zlib
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from floorplan_parser.akis import plani_coz
from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.degerlendirme import plan_puani
from floorplan_parser.foto import (
    duzelt,
    foto_ikile,
    homografi_uygula,
    isik_duzelt,
    kagit_bul,
    kose_hatasi,
)
from floorplan_parser.foto_benzetim import FotoAyar, cevap_tasi, foto_uret
from floorplan_parser.goruntu import yaz
from floorplan_parser.onisleme import ikile

SONUC = Path(__file__).resolve().parents[1] / "results" / "foto"

# (eğim, kırışık) → o fotoğrafta denenen kipler
FOTOLAR = {
    (0, 0): ("ham", "kahin", "oto"),
    (20, 0): ("ham", "kahin", "oto"),
    (40, 0): ("ham", "kahin", "oto", "oto_isiksiz"),
    (20, 4): ("kahin", "oto"),
    (20, 10): ("kahin", "oto"),
}
ORNEK_PLANLAR = 2  # ilk iki plan için görseller kaydedilir


def dolu_planlar():
    out = []
    for o in ornekler(bolum="gelistirme"):
        if o.grup != "I-II":
            continue
        g, c = yukle(o)
        if float((ikile(g)[maske(c, "Wall") > 0] > 0).mean()) >= 0.8:
            out.append(o)
    return out


def _puanla(sayfa, c, ek: dict, ikile: str = "otsu") -> dict:
    t0 = time.perf_counter()
    if ikile == "kagit":
        duz, siki = foto_ikile(sayfa)
        s = plani_coz(duz)
        s.duvar_maske = cv2.bitwise_and(s.duvar_maske, siki)
    else:
        s = plani_coz(sayfa)
    return {
        **ek,
        **plan_puani(s, c, maske(c, "Wall")),
        "sure_s": round(time.perf_counter() - t0, 2),
    }


def _kucuk(g: np.ndarray, uzun: int = 1200) -> np.ndarray:
    k = uzun / max(g.shape)
    return cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)


def is_yap(gorev, ikile: str = "otsu", ham: bool = True) -> list[dict]:
    sira, o, egim, buruk = gorev
    g, c = yukle(o)
    if egim is None:
        return [_puanla(g, c, {"kosul": "tarama", "ad": o.ad, "egim": "", "buruk": ""})]
    tohum = (zlib.crc32(o.ad.encode()) + 1009 * egim + 7919 * buruk) % (2**31)
    f = foto_uret(g, FotoAyar(egim=egim, buruk=buruk, tohum=tohum))
    sik = buruk > 0
    satirlar = []
    otomatik = None
    for kip in FOTOLAR[(egim, buruk)]:
        ad = f"{kip}_e{egim}" + (f"_k{buruk}" if buruk else "")
        ek = {"kosul": ad, "ad": o.ad, "egim": egim, "buruk": buruk}
        if kip == "ham":
            if not ham:
                continue
            ct = cevap_tasi(c, f.tasi, f.goruntu.shape[::-1], sik=sik)
            satirlar.append(_puanla(f.goruntu, ct, ek))
            continue
        if kip == "kahin":
            kose, hata = f.kose, 0.0
        else:
            if otomatik is None:
                otomatik = kagit_bul(f.goruntu)
            kose = otomatik
            hata = kose_hatasi(kose, f.kose) if kose is not None else float("nan")
            if kose is None:  # arayüzde kullanıcı köşeleri işaretler
                kose = f.kose
        sayfa, hom = duzelt(f.goruntu, kose)
        if kip != "oto_isiksiz":
            sayfa = isik_duzelt(sayfa)
        ct = cevap_tasi(
            c, lambda p, hom=hom: homografi_uygula(hom, f.tasi(p)), sayfa.shape[::-1], sik=sik
        )
        satirlar.append(_puanla(sayfa, ct, {**ek, "kose_hatasi_px": round(hata, 2)}, ikile))
        if sira < ORNEK_PLANLAR and kip == "oto" and ikile == "otsu":
            yaz(SONUC / "ornek" / f"{o.ad}_{ad}_foto.jpg", _kucuk(f.goruntu))
            yaz(SONUC / "ornek" / f"{o.ad}_{ad}_sayfa.png", _kucuk(sayfa))
    return satirlar


def f1(d: int, fz: int, e: int) -> float:
    k, duy = d / max(d + fz, 1), d / max(d + e, 1)
    return 2 * k * duy / max(k + duy, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=int, default=0, help="ilk N plan (0: hepsi)")
    ap.add_argument("--is", dest="isci", type=int, default=4)
    ap.add_argument("--etiket", default="")
    ap.add_argument("--ikile", default="otsu", choices=["otsu", "kagit"])
    ap.add_argument(
        "--ham", type=int, default=1, help="0: ham kipi atla (yavaş, ikilemeden bağımsız)"
    )
    a = ap.parse_args()
    planlar = dolu_planlar()
    if a.plan:
        planlar = planlar[: a.plan]
    gorevler = [(i, o, None, None) for i, o in enumerate(planlar)]
    gorevler += [(i, o, e, b) for (e, b) in FOTOLAR for i, o in enumerate(planlar)]
    t0 = time.perf_counter()
    satirlar = []
    with Pool(a.isci) as havuz:
        is_ = partial(is_yap, ikile=a.ikile, ham=bool(a.ham))
        for n, sonuc in enumerate(havuz.imap_unordered(is_, gorevler), 1):
            satirlar.extend(sonuc)
            print(f"{n}/{len(gorevler)} {sonuc[0]['ad']} {sonuc[0]['kosul']}", flush=True)

    SONUC.mkdir(parents=True, exist_ok=True)
    ad = f"foto_deney{'_' + a.etiket if a.etiket else ''}.csv"
    alanlar = list(dict.fromkeys(k for s in satirlar for k in s))
    with (SONUC / ad).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=alanlar)
        w.writeheader()
        w.writerows(sorted(satirlar, key=lambda s: (s["kosul"], s["ad"])))
    print(f"\n{len(planlar)} plan, {time.perf_counter() - t0:.0f} s -> results/foto/{ad}\n")

    sira = ["tarama"] + list(dict.fromkeys(s["kosul"] for s in satirlar if s["kosul"] != "tarama"))
    print(
        f"{'koşul':22} {'n':>3} {'duvar':>6} {'kapı':>6} {'pencere':>7} {'oda':>6} {'kenar':>6}"
        f" {'köşe px':>8} {'süre':>5}"
    )
    for k in sira:
        r = [s for s in satirlar if s["kosul"] == k]
        duvar = np.mean([s["duvar_maske_iou"] for s in r])
        f1ler = [
            f1(*(sum(s[f"{t}_{x}"] for s in r) for x in ("dogru", "fazla", "eksik")))
            for t in ("kapi", "pencere", "oda", "kenar")
        ]
        kose = [float(s.get("kose_hatasi_px", "nan")) for s in r]
        kose = [x for x in kose if not math.isnan(x)]
        kose_m = f"{np.median(kose):.2f}" if kose and "oto" in k else ""
        sure = np.median([s["sure_s"] for s in r])
        kapi, pencere, oda, kenar = f1ler
        print(
            f"{k:22} {len(r):3} {duvar:6.3f} {kapi:6.3f} {pencere:7.3f} {oda:6.3f} {kenar:6.3f}"
            f" {kose_m:>8} {sure:5.1f}"
        )


if __name__ == "__main__":
    main()
