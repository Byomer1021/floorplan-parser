"""Gerçek fotoğraf testi için basılacak planlar: seçim ve A4 PDF.

Kural (kiraz toplamamak için): beş alt gruptan (IIa, IId, Ia, Ib, Ic) birer
geliştirme planı. İkileme eşiğinin seçildiği 8 plan (olcumler §18) dışarıda.
Her grupta, taramadaki kapı F1 + pencere F1 toplamı grup medyanına en yakın
plan. Test bölümüne dokunulmaz.

Etiket (P1 ... P5 ve plan adı) çizimin dışında, ilk boş köşeye yazılır.
PDF 300 dpi, her sayfa A4. Yüzde 100 ölçekle basılmalı.

Kullanım: uv run python scripts/foto_baski.py ÇIKTI_KLASÖRÜ
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from floorplan_parser.cvcfp import ornekler, yukle

ESIK_PLANLARI = {
    "IIa_AP3301",
    "IIa_BS0601",
    "IIa_GZ0501",
    "IIa_SR1101",
    "IIc_UO0701",
    "IId_OZ0501",
    "Ia_GO0601_sommaire",
    "Ib_CT0601_sommaire",
}
GRUPLAR = ["IIa", "IId", "Ia", "Ib", "Ic"]


def f1(d, fz, e):
    k, du = d / max(d + fz, 1), d / max(d + e, 1)
    return 2 * k * du / max(k + du, 1e-9)


out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
gel = {o.ad: o for o in ornekler(bolum="gelistirme")}
with open(
    Path(__file__).resolve().parents[1] / "results/foto/foto_deney.csv", encoding="utf-8"
) as f:
    satir = [s for s in csv.DictReader(f) if s["kosul"] == "tarama"]
secilen = []
for gr in GRUPLAR:
    aday = [s for s in satir if s["ad"].split("_")[0] == gr and s["ad"] not in ESIK_PLANLARI]
    puan = {
        s["ad"]: f1(*(int(s[f"kapi_{x}"]) for x in ("dogru", "fazla", "eksik")))
        + f1(*(int(s[f"pencere_{x}"]) for x in ("dogru", "fazla", "eksik")))
        for s in aday
    }
    med = float(np.median(list(puan.values())))
    ad = min(sorted(puan), key=lambda a: abs(puan[a] - med))
    secilen.append(ad)

sayfalar = []
for i, ad in enumerate(secilen, 1):
    g, c = yukle(gel[ad])
    h, w = g.shape
    etiket = f"P{i}  {ad}"
    # Köşelerden biri, kenardan 1 cm (118 px). İlk boş köşeye etiket yazılır.
    yer = None
    for x0, y0 in ((w - 1018, h - 163), (118, h - 163), (118, 118), (w - 1018, 118)):
        if (g[y0 - 10 : y0 + 55, x0 - 10 : x0 + 910] > 200).mean() > 0.999:
            yer = (x0, y0)
            break
    if yer:
        cv2.putText(
            g, etiket, (yer[0], yer[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA
        )
    kapi, pencere, oda = (len(c.sinif(s)) for s in ("Door", "Window", "Room"))
    yon = "yatay" if w > h else "dikey"
    print(
        f"P{i} {ad:22} {w}x{h} {yon:5} kapı {kapi:2} pencere {pencere:2} oda {oda:2} "
        f"etiket {yer if yer else 'YOK'}"
    )
    Image.fromarray(g).save(out / f"P{i}_{ad}.png", dpi=(300, 300))
    sayfalar.append(Image.fromarray(g).convert("L"))

sayfalar[0].save(out / "planlar_A4.pdf", save_all=True, append_images=sayfalar[1:], resolution=300)
print("->", out / "planlar_A4.pdf")
