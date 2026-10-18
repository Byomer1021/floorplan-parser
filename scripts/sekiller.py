"""Rapor ve sunum şekilleri → rapor/sekiller/ (İngilizce etiketler).

1. pipeline.png   akışın adımları, bir geliştirme planının kırpıntısında
2. door_radius.png  kapı yayı testi: kaba yarıçap taraması doğru yarıçapı
                    atlıyor (olcumler §15, BK0701)
3. test_example.png bir test planı (oda F1'e göre medyan), cevap ve sonuç
4. failures.png   hata tipleri: duvarsız ayrım, garaj kapısı, taralı stil

Kullanım: uv run python scripts/sekiller.py [--yalniz pipeline,failures]
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from floorplan_parser.aciklik import (
    _TAM_TETA,
    _isabet,
    bosluklar,
    ince_murekkep,
    kose_bosluklari,
)
from floorplan_parser.akis import plani_coz
from floorplan_parser.cvcfp import ornekler, yukle
from floorplan_parser.duvar import ince_aday_maskesi
from floorplan_parser.gorsel import (
    DUVAR,
    cevap_katmanlari,
    katmanlar,
    taban,
)
from floorplan_parser.onisleme import delik_doldur, ikile

KOK = Path(__file__).resolve().parents[1]
CIKTI = KOK / "rapor" / "sekiller"
TURUNCU = (0, 140, 255)

plt.rcParams.update({"font.family": "serif", "font.size": 9})


def plan(ad: str):
    o = next(o for o in ornekler() if o.ad == ad)
    return yukle(o)


def kes(g: np.ndarray, kutu) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in kutu)
    h, w = g.shape[:2]
    return g[max(y0, 0) : min(y1, h), max(x0, 0) : min(x1, w)]


def rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else bgr


def kaydet(fig, ad: str) -> None:
    CIKTI.mkdir(parents=True, exist_ok=True)
    fig.savefig(CIKTI / ad, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("->", CIKTI / ad)


def bina_kutusu(c, pay: int = 60) -> tuple[int, int, int, int]:
    p = np.vstack([n.noktalar for n in c.sinif("Wall")])
    x0, y0 = np.floor(p.min(0)).astype(int) - pay
    x1, y1 = np.ceil(p.max(0)).astype(int) + pay
    return max(x0, 0), max(y0, 0), x1, y1


def sekil_akis() -> None:
    g, _ = plan("IIa_AP3301")
    kutu = (940, 1340, 1885, 2130)
    ikili = delik_doldur(ikile(g), 4)
    s = plani_coz(g)

    # (c) iki ölçek: kalın maske, ince adaylar, seçilen ince duvarlar
    kalin, aday = ince_aday_maskesi(ikili, 13, 7)
    c = np.full((*g.shape, 3), 255, np.uint8)
    c[ikili > 0] = (225, 225, 225)
    c[aday > 0] = (150, 150, 150)
    secilen = np.zeros(g.shape, np.uint8)
    from floorplan_parser.degerlendirme import parca_maskesi

    secilen = parca_maskesi(s.ince_parcalar, g.shape)
    c[secilen > 0] = TURUNCU
    c[kalin > 0] = DUVAR

    # (d) vektör duvarlar: orta çizgiler ve uçlar
    d = taban(g, 0.3)
    for p in s.kalin_parcalar + s.ince_parcalar:
        a, b = (round(p.x1), round(p.y1)), (round(p.x2), round(p.y2))
        renk = DUVAR if p in s.kalin_parcalar else TURUNCU
        cv2.line(d, a, b, renk, 5, cv2.LINE_AA)
        for q in (a, b):
            cv2.circle(d, q, 8, (0, 0, 0), -1, cv2.LINE_AA)

    # (e) boşluk adayları + kapı ve pencereler
    e = taban(g, 0.45)
    parcalar = s.kalin_parcalar + s.ince_parcalar
    for b in bosluklar(parcalar, s.duvar_maske) + kose_bosluklari(s.duvar_maske):
        cv2.line(e, tuple(map(round, b.a)), tuple(map(round, b.b)), (160, 60, 160), 3, cv2.LINE_AA)
    e2 = katmanlar(g, s, oda=False, duvar=False, graf=False, kalinlik=4)
    e = np.where((e2 != taban(g)).any(axis=2)[..., None], e2, e)

    f = katmanlar(g, s, kalinlik=4)
    paneller = [
        ("(a) input", cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)),
        ("(b) binarized", cv2.cvtColor(255 - ikili, cv2.COLOR_GRAY2BGR)),
        ("(c) thick / thin walls", c),
        ("(d) wall segments", d),
        ("(e) gaps, doors, windows", e),
        ("(f) rooms and graph", f),
    ]
    fig, eksen = plt.subplots(2, 3, figsize=(7.16, 4.4))
    for ax, (baslik, im) in zip(eksen.ravel(), paneller):
        ax.imshow(rgb(kes(im, kutu)))
        ax.set_title(baslik, fontsize=7.5)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.04, hspace=0.18)
    kaydet(fig, "pipeline.png")


def sekil_yaricap() -> None:
    g, _ = plan("IIa_BK0701")
    s = plani_coz(g)
    m = ince_murekkep(delik_doldur(ikile(g), 4), s.duvar_maske, 3.0)
    h = np.array([1381.0, 1413.0])
    u, n = np.array([1.0, 0.0]), np.array([0.0, -1.0])
    kutu = (1330, 1280, 1500, 1440)
    fig, eksen = plt.subplots(1, 2, figsize=(3.5, 1.9))
    for ax, r in zip(eksen, (91.0, 88.0)):
        yay = h + r * (np.cos(_TAM_TETA)[:, None] * u + np.sin(_TAM_TETA)[:, None] * n)
        isabet = _isabet(m, yay)
        im = taban(g, 0.5)
        im[m > 0] = (0.5 * im[m > 0] + 0.5 * np.array([120, 200, 255])).astype(np.uint8)
        ax.imshow(rgb(kes(im, kutu)))
        for (x, y), v in zip(yay, isabet):
            ax.plot(x - kutu[0], y - kutu[1], "o", ms=2.2, color="#1a9641" if v else "#d7191c")
        ax.plot(h[0] - kutu[0], h[1] - kutu[1], "k+", ms=6)
        ax.set_title(f"r = {r:.0f} px: score {isabet.mean():.2f}", fontsize=8)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.05)
    kaydet(fig, "door_radius.png")


def sekil_test() -> None:
    g, c = plan("IIa_LZ1101")
    s = plani_coz(g)
    kutu = bina_kutusu(c)
    fig, eksen = plt.subplots(1, 2, figsize=(7.16, 3.6))
    for ax, (baslik, im) in zip(
        eksen,
        [
            ("ground truth (CVC-FP)", cevap_katmanlari(g, c, kalinlik=4)),
            ("this method", katmanlar(g, s, kalinlik=4)),
        ],
    ):
        ax.imshow(rgb(kes(im, kutu)))
        ax.set_title(baslik, fontsize=9)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.03)
    kaydet(fig, "test_example.png")


def sekil_hatalar() -> None:
    paneller = []
    # (a) duvarsız ayrım: cevapta Separation, bizde tek bölge (test planı)
    g, c = plan("IIa_LZ1101")
    s = plani_coz(g)
    im = katmanlar(g, s, graf=False, kalinlik=4)
    for n in c.sinif("Room"):
        cv2.polylines(im, [np.round(n.noktalar).astype(np.int32)], True, (200, 0, 200), 5)
    ayrim = c.sinif("Separation")[0].noktalar
    (cx, cy), w = ayrim.mean(0).astype(int), 420
    paneller.append(
        (
            "(a) rooms separated without a wall\n(truth rooms: magenta)",
            kes(im, (cx - w, cy - w, cx + w, cy + w)),
        )
    )
    # (b) garaj kapısı pencere sanılıyor (cevapta ayrı sınıf: Parking)
    g, c = plan("IIa_AP3401")
    s = plani_coz(g)
    im = katmanlar(g, s, oda=False, graf=False, kalinlik=4)
    pk = c.sinif("Parking")
    for n in pk:
        cv2.polylines(im, [np.round(n.noktalar).astype(np.int32)], True, (200, 0, 200), 5)
    (cx, cy), w = pk[0].noktalar.mean(0).astype(int), 420
    paneller.append(
        (
            "(b) garage door taken as window\n(truth class: Parking, magenta)",
            kes(im, (cx - w, cy - w, cx + w, cy + w)),
        )
    )
    # (c) taralı duvar stili: duvar bulunmuyor
    ad = next(o.ad for o in ornekler(bolum="gelistirme") if o.grup == "sayi")
    g, c = plan(ad)
    s = plani_coz(g)
    im = katmanlar(g, s, oda=False, graf=False, kalinlik=3)
    for n in c.sinif("Wall"):
        cv2.polylines(im, [np.round(n.noktalar).astype(np.int32)], True, (200, 0, 200), 2)
    x0, y0, x1, y1 = bina_kutusu(c)
    w = min(x1 - x0, y1 - y0) // 2
    paneller.append(
        (
            "(c) hatched walls: mostly missed\n(truth walls: magenta)",
            kes(im, (x0, y0, x0 + w, y0 + w)),
        )
    )
    fig, eksen = plt.subplots(1, len(paneller), figsize=(7.16, 2.7))
    for ax, (baslik, im) in zip(eksen, paneller):
        ax.imshow(rgb(im))
        ax.set_title(baslik, fontsize=7.5)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.05)
    kaydet(fig, "failures.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yalniz", default="", help="pipeline,radius,test,failures")
    a = ap.parse_args()
    hepsi = {
        "pipeline": sekil_akis,
        "radius": sekil_yaricap,
        "test": sekil_test,
        "failures": sekil_hatalar,
    }
    for ad in a.yalniz.split(",") if a.yalniz else hepsi:
        hepsi[ad]()


if __name__ == "__main__":
    main()
