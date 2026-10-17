"""Kapı hatalarının sınıflandırması (geliştirme, I-II dolu duvarlı planlar).

Akış ``aciklik_bul`` ile aynı (parça + köşe boşlukları, yay, yarıçap sınırı).

Fazla kapı (eşleşmeyen bulunan):
- pencere: cevaptaki bir pencereyle IoU ≥ 0.3
- geometri: cevaptaki bir kapıyla 0.1 ≤ IoU < 0.5 (yer doğru, şekil tutmuyor)
- bos: ikisi de değil

Eksik kapı (eşleşmeyen cevap):
- aday_yok: çevresinde (10 px) parça ya da köşe boşluğu yok
- yay_zayif: çevresindeki boşlukların en iyi yay payı ``yay_esik``'in altında
- kucuk_yaricap: yay var ama yarıçap ``kapi_min_yaricap``'tan küçük (kanatlı
  pencere sayıldı)
- geometri: bir bulunan kapıyla 0.1 ≤ IoU < 0.5
- diger: yay ve yarıçap tamam ama eşleşmedi (bastırma, menteşe tarafı)

Çıktı: results/aciklik_hata.csv (hata başına satır).
Kullanım: uv run python scripts/aciklik_hata.py
"""

import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from floorplan_parser.aciklik import (
    VARSAYILAN_ACIKLIK_KURAL,
    aciklik_bul,
    bosluklar,
    ince_murekkep,
    kapi_dene,
    kose_bosluklari,
)
from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.degerlendirme import parca_maskesi, poligon_iou
from floorplan_parser.duvar import iki_olcekli_duvarlar
from floorplan_parser.onisleme import delik_doldur, ikile

SONUC = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    kural = VARSAYILAN_ACIKLIK_KURAL
    satirlar = []
    for o in ornekler(bolum="gelistirme"):
        if o.grup != "I-II":
            continue
        g, c = yukle(o)
        ham = ikile(g)
        if (ham[maske(c, "Wall") > 0] > 0).mean() < 0.8:
            continue
        ikili = delik_doldur(ham, 4)
        kalin, kalin_p, ince_p = iki_olcekli_duvarlar(ikili)
        parcalar = kalin_p + ince_p
        duvar = np.maximum(kalin, parca_maskesi(ince_p, g.shape))
        kapilar, _, _ = aciklik_bul(ikili, parcalar, duvar, kural)
        murekkep = ince_murekkep(ikili, duvar, kural.yay_tol)
        adaylar = bosluklar(parcalar, duvar, kural) + kose_bosluklari(duvar, kural)
        yaylar = [(b, kapi_dene(b, murekkep)) for b in adaylar]
        cevap_kapi = [n.noktalar for n in c.sinif("Door")]
        cevap_pencere = [n.noktalar for n in c.sinif("Window")]

        iou = np.array(
            [[poligon_iou(k.poligon(), cv) for cv in cevap_kapi] for k in kapilar]
        ).reshape(len(kapilar), len(cevap_kapi))
        eslesen_k, eslesen_c = set(), set()
        for i, j in sorted(np.argwhere(iou >= 0.5), key=lambda ij: -iou[ij[0], ij[1]]):
            if i not in eslesen_k and j not in eslesen_c:
                eslesen_k.add(i)
                eslesen_c.add(j)

        for i, k in enumerate(kapilar):
            if i in eslesen_k:
                continue
            pen = max((poligon_iou(k.poligon(), cp) for cp in cevap_pencere), default=0.0)
            kap = iou[i].max(initial=0.0)
            tur = "pencere" if pen >= 0.3 else "geometri" if kap >= 0.1 else "bos"
            satirlar.append(
                {
                    "ad": o.ad,
                    "hata": "fazla",
                    "tur": tur,
                    "pay": round(k.skor, 2),
                    "yaricap": round(k.yaricap),
                    "x": round(k.mentese[0]),
                    "y": round(k.mentese[1]),
                }
            )

        for j, cv in enumerate(cevap_kapi):
            if j in eslesen_c:
                continue
            cv32 = cv.astype(np.float32)
            yakin = [
                (b, k)
                for b, k in yaylar
                if cv2.pointPolygonTest(cv32, ((b.a[0] + b.b[0]) / 2, (b.a[1] + b.b[1]) / 2), True)
                >= -10
            ]
            iyi = [k for _, k in yakin if k is not None and k.skor >= kural.yay_esik]
            en_iyi = max((k.skor for _, k in yakin if k is not None), default=0.0)
            if not yakin:
                tur = "aday_yok"
            elif not iyi:
                tur = "yay_zayif"
            elif all(k.yaricap < kural.kapi_min_yaricap for k in iyi):
                tur = "kucuk_yaricap"
            elif iou[:, j].max(initial=0.0) >= 0.1:
                tur = "geometri"
            else:
                tur = "diger"
            x, y = cv.mean(0)
            satirlar.append(
                {
                    "ad": o.ad,
                    "hata": "eksik",
                    "tur": tur,
                    "pay": round(en_iyi, 2),
                    "yaricap": round(float(np.ptp(cv, 0).max())),
                    "x": round(x),
                    "y": round(y),
                }
            )
        print(o.ad, flush=True)

    with (SONUC / "aciklik_hata.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)
    for hata in ("fazla", "eksik"):
        print(hata, dict(Counter(s["tur"] for s in satirlar if s["hata"] == hata)))


if __name__ == "__main__":
    main()
