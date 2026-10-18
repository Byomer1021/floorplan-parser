"""Oda bölmede kapatılan sınıflanmamış parça boşlukları: hangisi gerçekten
iki alanı ayırıyor, hangisi bir odanın ortasından geçiyor (geliştirme, I-II
dolu duvarlı planlar).

Kapı ya da pencere olmayan her parça boşluğu için kirişin iki yanında, orta
noktadan duvar yarım kalınlığı + 20 px ötede cevaptaki oda:
- ic: iki yan aynı cevap odası (kiriş odayı böler, zararlı)
- ayirici: iki farklı cevap odası
- dis: bir yan oda değil (dış, duvar, cevapsız alan)

Özellikler (boşluğun nasıl bulunduğu, ``aciklik.bosluklar`` ile aynı ışın):
- kaynak: ışının çıktığı uç kavşakta mı (``kavsak``) serbest mi (``serbest``)
- hedef: çarpılan duvar ``hizali`` (≤ 10°, orta çizgi ≤ yarım kalınlık + 4),
  ``dik`` (başka açıda parça) ya da ``parcasiz`` (parçaya dönüşmemiş duvar)
- murekkep: kiriş boyunca ince mürekkep payı (kapı kanadı, pencere çizgisi)

Çıktı: results/bosluk_hata.csv.
Kullanım: uv run python scripts/bosluk_hata.py
"""

import csv
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from floorplan_parser.aciklik import (
    VARSAYILAN_ACIKLIK_KURAL,
    aciklik_bul,
    bosluk_isini,
    duvar_etiketi,
    ince_murekkep,
    yan_durumu,
)
from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.degerlendirme import parca_maskesi
from floorplan_parser.duvar import aci_farki, iki_olcekli_duvarlar
from floorplan_parser.onisleme import delik_doldur, ikile

SONUC = Path(__file__).resolve().parents[1] / "results"


def bosluk_ozellikleri(parcalar, duvar_maske, kural):
    """``aciklik.bosluklar`` ile aynı adaylar, (a, b) anahtarıyla özellikler."""
    etiket = duvar_etiketi(parcalar, duvar_maske)
    h, w = duvar_maske.shape
    en_kalin = max((p.kalinlik for p in parcalar), default=0.0)
    ozellik = {}
    for i, p in enumerate(parcalar):
        k = np.array(p.kapsam or (p.x1, p.y1, p.x2, p.y2))
        a, b = k[:2], k[2:]
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n = np.array([-d[1], d[0]])
        graf = (np.array([p.x1, p.y1]), np.array([p.x2, p.y2]))
        for uc, yon, g_uc in ((a, -d, graf[0]), (b, d, graf[1])):
            kavsak = np.linalg.norm(g_uc - uc) > 1.0
            r = bosluk_isini(duvar_maske, uc, yon, kural, en_kalin + 4 if kavsak else 4)
            if r is None:
                continue
            ga, gb = uc + r[0] * yon, uc + r[1] * yon
            xi = min(max(int(np.floor(gb[0] + 0.5 + yon[0])), 0), w - 1)
            yi = min(max(int(np.floor(gb[1] + 0.5 + yon[1])), 0), h - 1)
            e = etiket[yi, xi]
            if e <= 0:
                hedef = "parcasiz"
            else:
                q = parcalar[e - 1]
                q_orta = (np.array([q.x1, q.y1]) + np.array([q.x2, q.y2])) / 2
                hizali = (
                    aci_farki(q.aci, p.aci) <= math.radians(10)
                    and abs((q_orta - a) @ n) <= max(p.kalinlik, q.kalinlik) / 2 + 4
                )
                hedef = "hizali" if hizali else "dik"
            anahtar = (round(float(ga[0]), 1), round(float(ga[1]), 1))
            yanlar = sorted(yan_durumu(duvar_maske, ga, yon, p.kalinlik, s) for s in (n, -n))
            ozellik.setdefault(
                anahtar,
                {
                    "kaynak": "kavsak" if kavsak else "serbest",
                    "hedef": hedef,
                    "uzunluk": p.uzunluk,
                    "yan": "+".join(yanlar),
                },
            )
    return ozellik


def oda(oda_lab: np.ndarray, q: np.ndarray) -> int:
    """Noktadaki cevap odası (0: oda değil ya da resim dışı)."""
    x, y = round(q[0]), round(q[1])
    if 0 <= x < oda_lab.shape[1] and 0 <= y < oda_lab.shape[0]:
        return int(oda_lab[y, x])
    return 0


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
        _, _, kalan = aciklik_bul(ikili, parcalar, duvar, kural)
        ozellik = bosluk_ozellikleri(parcalar, duvar, kural)
        murekkep = ince_murekkep(ikili, duvar, 2.0)

        oda_lab = np.zeros(g.shape, np.int32)
        for j, n in enumerate(c.sinif("Room"), start=1):
            cv2.fillPoly(oda_lab, [np.round(n.noktalar).astype(np.int32)], j)

        for bo in kalan:
            a, b = np.array(bo.a), np.array(bo.b)
            d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            n = np.array([-d[1], d[0]])
            m = (a + b) / 2
            off = bo.kalinlik / 2 + 20
            y1, y2 = oda(oda_lab, m + off * n), oda(oda_lab, m - off * n)
            tur = "dis" if y1 == 0 or y2 == 0 else "ic" if y1 == y2 else "ayirici"
            orn = a + np.linspace(0.1, 0.9, 20)[:, None] * (b - a)
            x = np.clip(np.round(orn[:, 0]).astype(int), 0, g.shape[1] - 1)
            y = np.clip(np.round(orn[:, 1]).astype(int), 0, g.shape[0] - 1)
            oz = ozellik.get((round(bo.a[0], 1), round(bo.a[1], 1)), {})
            satirlar.append(
                {
                    "ad": o.ad,
                    "tur": tur,
                    "genislik": round(bo.genislik),
                    "kalinlik": round(bo.kalinlik, 1),
                    "kaynak": oz.get("kaynak", "?"),
                    "hedef": oz.get("hedef", "?"),
                    "parca_uzunluk": round(oz.get("uzunluk", 0)),
                    "yan": oz.get("yan", "?"),
                    "murekkep": round(float((murekkep[y, x] > 0).mean()), 2),
                    "x": round(m[0]),
                    "y": round(m[1]),
                }
            )
        print(o.ad, flush=True)

    with (SONUC / "bosluk_hata.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)
    print("tur", dict(Counter(s["tur"] for s in satirlar)))
    for alan in ("kaynak", "hedef"):
        print(alan, dict(Counter((s[alan], s["tur"]) for s in satirlar)))
    for hedef in ("dik", "hizali", "parcasiz"):
        print(f"yan, hedef {hedef}:")
        sayac = Counter((s["yan"], s["tur"]) for s in satirlar if s["hedef"] == hedef)
        for yan in sorted({y for y, _ in sayac}):
            print(f"  {yan:20}", {t: sayac[(yan, t)] for t in ("ic", "ayirici", "dis")})


if __name__ == "__main__":
    main()
