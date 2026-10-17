"""Kaçan odaların sınıflandırması (geliştirme, I-II dolu duvarlı planlar).

Eşleşmeyen her cevap odası için, içinde en çok piksel kaplayan bulunan bölge:
- dis: o bölge bina dışı (DIS)
- duvar: odanın çoğu duvar / kiriş pikseli (bölge yok)
- birlesti_ayrim: bölge başka cevap odalarını da kaplıyor ve aralarından
  biriyle cevapta duvarsız ayrım (Separation) ilişkisi var
- birlesti_kapi: bölge başka cevap odalarını da kaplıyor, aralarında kapılı
  ilişki var (kapı bulunamadı ya da kirişi açıklığı kapatmadı)
- birlesti_diger: başka odaları kaplıyor, cevapta ikisi de yok
- kismi: bölge tek odaya ait ama IoU < 0.5

"Kaplıyor": bölgenin o odayla kesişimi odanın %30'undan fazla.
Kullanım: uv run python scripts/oda_hata.py [--varyant duvar:yok:0:151:0]
(varyant biçimi oda_deney.py ile aynı; INCE yok sayılır)
"""

import argparse
import csv
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from floorplan_parser.aciklik import aciklik_bul, bosluklar, genis_bosluklar, kose_bosluklari
from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.degerlendirme import bolge_eslesmeleri, parca_maskesi
from floorplan_parser.duvar import iki_olcekli_duvarlar
from floorplan_parser.oda import DIS, VARSAYILAN_ODA_KURAL, oda_grafi
from floorplan_parser.onisleme import delik_doldur, ikile

SONUC = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--varyant", default="genis:pg:0:301:0.8:45:0.8")
    a = ap.parse_args()
    ayak_k, oda_k, _, cek, pay, *havza = a.varyant.split(":")
    kural = replace(VARSAYILAN_ODA_KURAL, ayak_izi_cekirdek=int(cek), min_duvar_payi=float(pay))
    if havza:
        kural = replace(kural, havza_cekirdek=float(havza[0]), havza_birlesme=float(havza[1]))
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
        duvar = np.maximum(kalin, parca_maskesi(ince_p, g.shape))
        parcalar = kalin_p + ince_p
        kapilar, pencereler, _ = aciklik_bul(ikili, parcalar, duvar)
        parca_b = bosluklar(parcalar, duvar)
        secenek = {
            "yok": [],
            "parca": parca_b,
            "hepsi": parca_b + kose_bosluklari(duvar),
            "genis": genis_bosluklar(parcalar, duvar),
        }
        secenek["pg"] = secenek["parca"] + secenek["genis"]
        secenek["hg"] = secenek["hepsi"] + secenek["genis"]
        _, etiket = oda_grafi(
            duvar,
            kapilar,
            pencereler,
            kural,
            bosluklar=secenek[oda_k],
            ayak_bosluklari=None if ayak_k == "duvar" else secenek[ayak_k],
        )

        odalar = c.sinif("Room")
        eslesen = {j for _, j, _ in bolge_eslesmeleri(etiket, [n.noktalar for n in odalar])}
        sinif = {n.id: n.sinif for n in c.nesneler}
        iliski: dict[frozenset, str] = {}
        for tip, ids in c.iliskiler:
            if tip != "access":
                continue
            oda_ids = [i for i in ids if sinif.get(i) == "Room"]
            tur = "ayrim" if any(sinif.get(i) == "Separation" for i in ids) else "kapi"
            for x in oda_ids:
                for y in oda_ids:
                    if x != y:
                        iliski.setdefault(frozenset((x, y)), tur)

        # Her odanın içindeki etiket sayımları.
        sayim = []
        for n in odalar:
            m = np.zeros(etiket.shape, np.uint8)
            cv2.fillPoly(m, [np.round(n.noktalar * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
            v, s = np.unique(etiket[m > 0], return_counts=True)
            sayim.append((dict(zip(v.tolist(), s.tolist())), int(m.sum())))

        for j, n in enumerate(odalar):
            if j in eslesen:
                continue
            say, alan = sayim[j]
            bolgeler = {k: v for k, v in say.items() if k != 0}
            if not bolgeler or sum(bolgeler.values()) < 0.5 * alan:
                tur = "duvar"
            else:
                e = max(bolgeler, key=bolgeler.get)
                if e == DIS:
                    tur = "dis"
                else:
                    ortak = [
                        odalar[k].id
                        for k, (s2, a2) in enumerate(sayim)
                        if k != j and s2.get(e, 0) > 0.3 * a2
                    ]
                    if not ortak:
                        tur = "kismi"
                    else:
                        turler = {iliski.get(frozenset((n.id, x))) for x in ortak}
                        tur = (
                            "birlesti_ayrim"
                            if "ayrim" in turler
                            else "birlesti_kapi"
                            if "kapi" in turler
                            else "birlesti_diger"
                        )
            satirlar.append({"ad": o.ad, "oda": n.id, "tur": tur, "alan": alan})
        print(o.ad, flush=True)

    with (SONUC / "oda_hata.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)
    print("kacan oda", len(satirlar), dict(Counter(s["tur"] for s in satirlar)))


if __name__ == "__main__":
    main()
