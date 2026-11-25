"""Fotoğraf sayfasında ikileme seçimi (olcumler §18).

Geliştirme bölümünden 8 dolu planın 0 ve 40 derece eğimli sentetik
fotoğrafı, gerçek köşelerle düzeltilip ışığı düzeltilir. Sonra ikileme
yöntemleri karşılaştırılır:

- ``otsu``: akışın kendi Otsu'su (taramada kullanılan).
- ``kagitN``: kâğıdın parlaklığından (sayfanın medyanı) N gri seviye koyu
  olan her piksel mürekkep.
- ``kagitN_kirp``: aynısı, çıktıdaki duvar maskesi Otsu mürekkebiyle kırpılır
  (``foto.foto_ikile`` ve ``foto.fotograftan_coz`` bu, N = 50).
- ``sauvola``: yerel eşik (pencere 31, k 0.2).
- ``karma40_H``: Otsu mürekkebi + düşük eşikten gelen pikseller, ama kalın
  mürekkebin H piksellik çevresi hariç (bulanıklık payı).

Kullanım: uv run python scripts/foto_esik_deney.py
"""

import zlib
from multiprocessing import Pool

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

from floorplan_parser.akis import plani_coz
from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.degerlendirme import plan_puani
from floorplan_parser.foto import duzelt, homografi_uygula, isik_duzelt
from floorplan_parser.foto_benzetim import FotoAyar, cevap_tasi, foto_uret
from floorplan_parser.onisleme import ikile

YONTEMLER = [
    "otsu",
    "kagit25",
    "kagit40",
    "kagit60",
    "kagit30_kirp",
    "kagit40_kirp",
    "kagit50_kirp",
    "sauvola",
    "karma40_1",
    "karma40_3",
]
PLAN_SAYISI = 8


def ikile_foto(sayfa: np.ndarray, yontem: str) -> np.ndarray:
    """Akışa verilecek sayfa: 0 mürekkep, 255 kâğıt (``otsu``: olduğu gibi)."""
    if yontem == "otsu":
        return sayfa
    kagit = float(np.median(sayfa))
    if yontem == "sauvola":
        murekkep = sayfa < threshold_sauvola(sayfa, window_size=31, k=0.2)
    elif yontem.startswith("karma"):
        fark, hale = (int(x) for x in yontem[5:].split("_"))
        _, otsu = cv2.threshold(sayfa, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kalin = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        bant = cv2.dilate(kalin, np.ones((2 * hale + 1, 2 * hale + 1), np.uint8)) > 0
        murekkep = (otsu > 0) | ((sayfa < kagit - fark) & ~(bant & (kalin == 0)))
    else:
        murekkep = sayfa < kagit - int(yontem[5:].split("_")[0])
    return np.where(murekkep, 0, 255).astype(np.uint8)


def is_yap(gorev):
    ad, egim = gorev
    o = next(o for o in ornekler(bolum="gelistirme") if o.ad == ad)
    g, c = yukle(o)
    tohum = (zlib.crc32(o.ad.encode()) + 1009 * egim) % (2**31)  # foto_deney ile aynı
    f = foto_uret(g, FotoAyar(egim=egim, tohum=tohum))
    sayfa, hom = duzelt(f.goruntu, f.kose)
    sayfa = isik_duzelt(sayfa)
    ct = cevap_tasi(c, lambda p: homografi_uygula(hom, f.tasi(p)), sayfa.shape[::-1])
    _, otsu = cv2.threshold(sayfa, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    out = []
    for y in YONTEMLER:
        s = plani_coz(ikile_foto(sayfa, y))
        if y.endswith("_kirp"):
            s.duvar_maske = cv2.bitwise_and(s.duvar_maske, otsu)
        out.append((egim, y, plan_puani(s, ct, maske(ct, "Wall"))))
    return out


def f1(d: int, fz: int, e: int) -> float:
    k, duy = d / max(d + fz, 1), d / max(d + e, 1)
    return 2 * k * duy / max(k + duy, 1e-9)


def main() -> None:
    planlar = []
    for o in ornekler(bolum="gelistirme"):
        if o.grup != "I-II":
            continue
        g, c = yukle(o)
        if float((ikile(g)[maske(c, "Wall") > 0] > 0).mean()) >= 0.8:
            planlar.append(o.ad)
    planlar = planlar[:: len(planlar) // PLAN_SAYISI][:PLAN_SAYISI]
    print("planlar:", " ".join(planlar))
    with Pool(4) as havuz:
        sonuc = [
            r for rs in havuz.map(is_yap, [(a, e) for a in planlar for e in (0, 40)]) for r in rs
        ]
    print(
        f"{'eğim':4} {'yöntem':13} {'duvar':>6} {'kapı':>6} {'pencere':>7} {'oda':>6} {'kenar':>6}"
    )
    for e in (0, 40):
        for y in YONTEMLER:
            r = [p for (eg, yy, p) in sonuc if eg == e and yy == y]
            kapi, pencere, oda, kenar = (
                f1(*(sum(p[f"{t}_{x}"] for p in r) for x in ("dogru", "fazla", "eksik")))
                for t in ("kapi", "pencere", "oda", "kenar")
            )
            duvar = np.mean([p["duvar_maske_iou"] for p in r])
            print(
                f"{e:4} {y:13} {duvar:6.3f} {kapi:6.3f} {pencere:7.3f} {oda:6.3f} {kenar:6.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
