"""Renkli kat planı akışı bozar mı (olcumler §19)?

Geliştirme bölümünden 8 dolu plan (``foto_esik_deney`` ile aynı). Odalar
cevaptaki oda poligonlarıyla boyanır, çizgiler üstte kalır (çarpma
karışımı, basılı çizginin renkli dolgu üstündeki hâli). Dolgular:

- ``pastel``: açık tonlar (gri karşılığı yaklaşık 230-240).
- ``orta``: bej, mavi, yeşil (gri karşılığı yaklaşık 190-215).
- ``ahsap``: kahverengi, çizgili doku (gri karşılığı yaklaşık 130-170).

İkileme: ``otsu`` (taramada kullanılan), ``kagit`` (fotoğraf yolunun
``foto_ikile``'si) ve ``doygun``: önce renkli pikseller (doygunluk > 60,
parlaklık > 60) kâğıt sayılır, sonra ``kagit``. Mürekkep siyah ya da gri,
dolgu renkli varsayımı.

Kullanım: uv run python scripts/renk_deney.py
"""

from multiprocessing import Pool

import cv2
import numpy as np

from floorplan_parser.akis import plani_coz
from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.degerlendirme import plan_puani
from floorplan_parser.foto import foto_ikile

PLANLAR = [
    "IIa_AP3301",
    "IIa_BS0601",
    "IIa_GZ0501",
    "IIa_SR1101",
    "IIc_UO0701",
    "IId_OZ0501",
    "Ia_GO0601_sommaire",
    "Ib_CT0601_sommaire",
]
RENKLER = {
    "pastel": [(250, 236, 216), (222, 238, 250), (230, 248, 224), (248, 226, 232)],
    "orta": [(238, 204, 156), (172, 208, 238), (182, 224, 172), (230, 186, 196)],
    "ahsap": [(186, 140, 92), (170, 122, 78), (198, 152, 104)],
}
YONTEMLER = ["otsu", "kagit", "doygun"]


def boya(gri: np.ndarray, c, dolgu: str, tohum: int) -> np.ndarray:
    """RGB plan: odalar renkli, çizgiler çarpma karışımıyla üstte."""
    rng = np.random.default_rng(tohum)
    zemin = np.full((*gri.shape, 3), 255, np.float32)
    renkler = RENKLER[dolgu]
    for i, n in enumerate(c.sinif("Room")):
        m = np.zeros(gri.shape, np.uint8)
        cv2.fillPoly(m, [np.round(n.noktalar).astype(np.int32)], 255)
        zemin[m > 0] = renkler[i % len(renkler)]
    if dolgu == "ahsap":
        cizgi = cv2.resize(rng.normal(0, 1, (gri.shape[0] // 12, 1)), gri.shape[::-1])
        doku = 1 + 0.12 * cizgi.astype(np.float32)
        oda = zemin.min(2) < 250
        zemin[oda] *= doku[oda][:, None]
    rgb = zemin * (gri.astype(np.float32)[..., None] / 255)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def ikile_renkli(rgb: np.ndarray, yontem: str):
    """(akışa verilecek gri sayfa, duvar maskesini kırpacak sıkı mürekkep ya da None)."""
    gri = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if yontem == "otsu":
        return gri, None
    if yontem == "doygun":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        renkli = (hsv[..., 1] > 60) & (hsv[..., 2] > 60)
        gri = np.where(renkli, 255, gri).astype(np.uint8)
    return foto_ikile(gri)


def is_yap(ad: str):
    o = next(o for o in ornekler(bolum="gelistirme") if o.ad == ad)
    g, c = yukle(o)
    cevap_duvar = maske(c, "Wall")
    out = []
    for dolgu in ["yok", *RENKLER]:
        rgb = np.repeat(g[..., None], 3, 2) if dolgu == "yok" else boya(g, c, dolgu, 1)
        for y in YONTEMLER:
            sayfa, siki = ikile_renkli(rgb, y)
            s = plani_coz(sayfa)
            if siki is not None:
                s.duvar_maske = cv2.bitwise_and(s.duvar_maske, siki)
            out.append((dolgu, y, plan_puani(s, c, cevap_duvar)))
    return out


def f1(d: int, fz: int, e: int) -> float:
    k, duy = d / max(d + fz, 1), d / max(d + e, 1)
    return 2 * k * duy / max(k + duy, 1e-9)


def main() -> None:
    with Pool(4) as havuz:
        sonuc = [r for rs in havuz.map(is_yap, PLANLAR) for r in rs]
    print(
        f"{'dolgu':7} {'ikileme':7} {'duvar':>6} {'kapı':>6} {'pencere':>7} {'oda':>6} {'kenar':>6}"
    )
    for dolgu in ["yok", *RENKLER]:
        for y in YONTEMLER:
            r = [p for (d, yy, p) in sonuc if d == dolgu and yy == y]
            kapi, pencere, oda, kenar = (
                f1(*(sum(p[f"{t}_{x}"] for p in r) for x in ("dogru", "fazla", "eksik")))
                for t in ("kapi", "pencere", "oda", "kenar")
            )
            duvar = np.mean([p["duvar_maske_iou"] for p in r])
            print(
                f"{dolgu:7} {y:7} {duvar:6.3f} {kapi:6.3f} {pencere:7.3f} {oda:6.3f} {kenar:6.3f}",
                flush=True,
            )
    # Gri karşılıkları (dolgunun tonu, akış griyle çalışıyor).
    for dolgu, renkler in RENKLER.items():
        gri = [round(0.299 * r + 0.587 * g + 0.114 * b) for r, g, b in renkler]
        print(f"{dolgu}: gri {gri}")


if __name__ == "__main__":
    main()
