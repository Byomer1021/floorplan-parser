"""Bulunamayan kapı kenarlarının sınıflandırması (geliştirme, I-II dolu).

Cevaptaki her kapılı ``access`` kenarı (kapı + iki farklı oda) için, ilk
uyan neden:
- kapi_yok: kapı bulunamadı (poligon IoU ≥ 0.5 eşleşmesi yok)
- oda_yok: iki odadan biri bulunamadı (bölge IoU ≥ 0.5 eşleşmesi yok)
- ayni_bolge: kapı ve odalar bulundu ama kapının iki yanı aynı bölgede
  ya da bir yanı bulunamadı (kenar oluşmadı)
- yanlis_oda: kenar oluştu ama başka oda çiftine bağlandı
- dogru

Kullanım: uv run python scripts/kenar_hata.py --varyant duvar:yok:0:151:0
(varyant biçimi oda_deney.py ile aynı)
"""

import argparse
from collections import Counter
from dataclasses import replace

import numpy as np

from floorplan_parser.aciklik import aciklik_bul, bosluklar, genis_bosluklar, kose_bosluklari
from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.degerlendirme import bolge_eslesmeleri, parca_maskesi, poligon_eslesmeleri
from floorplan_parser.duvar import iki_olcekli_duvarlar
from floorplan_parser.oda import VARSAYILAN_ODA_KURAL, kapi_kenarlari, oda_grafi
from floorplan_parser.onisleme import delik_doldur, ikile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--varyant", nargs="+", default=["duvar:yok:0:151:0"])
    a = ap.parse_args()
    sayac = {v: Counter() for v in a.varyant}
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
        sinif = {n.id: n.sinif for n in c.nesneler}
        kapilar_c = c.sinif("Door")
        odalar = c.sinif("Room")
        kapi_es = {
            kapilar_c[j].id: i
            for i, j in poligon_eslesmeleri(
                [k.poligon() for k in kapilar], [n.noktalar for n in kapilar_c]
            )
        }
        for v in a.varyant:
            ayak_k, oda_k, _, cek, pay = v.split(":")
            kural = replace(
                VARSAYILAN_ODA_KURAL, ayak_izi_cekirdek=int(cek), min_duvar_payi=float(pay)
            )
            _, etiket = oda_grafi(
                duvar,
                kapilar,
                pencereler,
                kural,
                bosluklar=secenek[oda_k],
                ayak_bosluklari=None if ayak_k == "duvar" else secenek[ayak_k],
            )
            oda_es = {
                odalar[j].id: e
                for e, j, _ in bolge_eslesmeleri(etiket, [n.noktalar for n in odalar])
            }
            kenar = {i: frozenset((x, y)) for x, y, i in kapi_kenarlari(etiket, kapilar, kural)}
            gorulen = set()
            for tip, ids in c.iliskiler:
                if tip != "access" or "Door" not in [sinif.get(i) for i in ids]:
                    continue
                kapi = next(i for i in ids if sinif.get(i) == "Door")
                iki = frozenset(i for i in ids if sinif.get(i) == "Room")
                if len(iki) != 2 or kapi in gorulen:
                    continue
                gorulen.add(kapi)
                if kapi not in kapi_es:
                    tur = "kapi_yok"
                elif not all(r in oda_es for r in iki):
                    tur = "oda_yok"
                elif kapi_es[kapi] not in kenar:
                    tur = "ayni_bolge"
                elif kenar[kapi_es[kapi]] != frozenset(oda_es[r] for r in iki):
                    tur = "yanlis_oda"
                else:
                    tur = "dogru"
                sayac[v][tur] += 1
        print(o.ad, flush=True)
    for v, s in sayac.items():
        print(f"{v:22} {dict(s.most_common())}")


if __name__ == "__main__":
    main()
