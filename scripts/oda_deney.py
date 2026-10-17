"""CVC-FP'de oda grafı deneyi: odalar ve kapı kenarları, cevaptaki Room ve
``access`` ilişkileriyle.

Duvar ve açıklık adımları plan başına bir kez çalışır; oda adımı her
``--varyant`` için ayrı. Varyant ``AYAK:ODA:INCE[:CEKIRDEK:DUVAR_PAYI]``:
- AYAK: ayak izinin kaynağı. duvar = ham duvar maskesi; parca = parça
  boşlukları da kapanmış; hepsi = parça + köşe boşlukları kapanmış; genis =
  aynı doğrultuda 150-450 px açıklıklar (``genis_bosluklar``) kapanmış
- ODA: oda bölmede kapı/pencere dışında kapatılan boşluklar (yok/parca/hepsi)
- INCE: 1 ise iki ucu duvara değen uzun ince çizgiler oda sınırı sayılır
- CEKIRDEK: ayak izi kapama çekirdeği (px); DUVAR_PAYI: ayak izi içindeki
  bölgenin çevresinde duvar/kiriş payı bunun altındaysa dış (0 = kapalı)

Plan başına:
- oda_dogru / oda_fazla / oda_eksik: bulunan oda bölgesi ile cevaptaki Room
  poligonu IoU ≥ 0.5, bire bir.
- kenar_dogru / kenar_fazla / kenar_eksik: oda-oda kapı kenarları. Doğru:
  kenarın kapısı bir cevap kapısıyla eşleşmiş (poligon IoU ≥ 0.5) ve kenarın
  iki odası, o kapının ``access`` ilişkisindeki iki odayla eşleşmiş. Cevap
  kenarları: iki farklı odası olan kapılı ``access`` ilişkileri, kapı başına.
- ayrim: cevaptaki duvarsız (Separation) ``access`` sayısı.
- dis_kapi: bir yanı bina dışı olan kapı kenarı sayısı (yalnız sayılır).

Yalnız I-II, dolu duvarlı planlar. Varsayılan bölüm geliştirme.
Kullanım: uv run python scripts/oda_deney.py --varyant duvar:yok:0 hepsi:parca:1
"""

import argparse
import csv
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from floorplan_parser.aciklik import aciklik_bul, bosluklar, genis_bosluklar, kose_bosluklari
from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.degerlendirme import bolge_eslesmeleri, parca_maskesi, poligon_eslesmeleri
from floorplan_parser.duvar import iki_olcekli_duvarlar
from floorplan_parser.oda import DIS, VARSAYILAN_ODA_KURAL, ince_sinirlar, oda_grafi
from floorplan_parser.onisleme import delik_doldur, ikile

SONUC = Path(__file__).resolve().parents[1] / "results"
VARSAYILAN_VARYANT = "genis:pg:0:301:0.8:45:0.8"  # akis.plani_coz ile aynı


def cevap_kenarlari(c) -> tuple[dict, int]:
    """(kapı id → oda id çifti, duvarsız ayrım sayısı)."""
    sinif = {n.id: n.sinif for n in c.nesneler}
    kenar, ayrim = {}, 0
    for tip, ids in c.iliskiler:
        if tip != "access":
            continue
        turler = [sinif.get(i) for i in ids]
        oda_ids = frozenset(i for i in ids if sinif.get(i) == "Room")
        if "Door" in turler and len(oda_ids) == 2:
            kenar[next(i for i in ids if sinif.get(i) == "Door")] = oda_ids
        elif "Separation" in turler:
            ayrim += 1
    return kenar, ayrim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bolum", default="gelistirme", choices=["gelistirme", "test"])
    ap.add_argument("--varyant", nargs="+", default=[VARSAYILAN_VARYANT])
    ap.add_argument("--etiket", default="")
    a = ap.parse_args()
    kural = VARSAYILAN_ODA_KURAL

    satirlar = []
    t_basla = time.perf_counter()
    for o in ornekler(bolum=a.bolum):
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
        genis_b = genis_bosluklar(parcalar, duvar)
        secenek = {
            "yok": [],
            "parca": parca_b,
            "hepsi": parca_b + kose_bosluklari(duvar),
            "genis": genis_b,
            "pg": parca_b + genis_b,
            "hg": parca_b + kose_bosluklari(duvar) + genis_b,
        }
        sinir = None

        odalar = c.sinif("Room")
        kapilar_c = c.sinif("Door")
        kapi_es = {
            i: kapilar_c[j].id
            for i, j in poligon_eslesmeleri(
                [k.poligon() for k in kapilar], [n.noktalar for n in kapilar_c]
            )
        }
        cevap_kenar, ayrim = cevap_kenarlari(c)

        for v in a.varyant:
            ayak_k, oda_k, ince_k, *ek_k = v.split(":")
            # İsteğe bağlı: CEKIRDEK:DUVAR_PAYI[:HAVZA_ESIK:HAVZA_BIRLESME]
            # (ör. genis:pg:0:301:0.8:45:0.8)
            kural = VARSAYILAN_ODA_KURAL
            if ek_k:
                kural = replace(
                    kural, ayak_izi_cekirdek=int(ek_k[0]), min_duvar_payi=float(ek_k[1])
                )
            if len(ek_k) >= 4:
                kural = replace(kural, havza_cekirdek=float(ek_k[2]), havza_birlesme=float(ek_k[3]))
            if ince_k == "1" and sinir is None:
                sinir = ince_sinirlar(ikili, duvar)
            t0 = time.perf_counter()
            graf, etiket = oda_grafi(
                duvar,
                kapilar,
                pencereler,
                kural,
                bosluklar=secenek[oda_k],
                ayak_bosluklari=None if ayak_k == "duvar" else secenek[ayak_k],
                sinir_maske=sinir if ince_k == "1" else None,
            )
            sure = time.perf_counter() - t0
            oda_es = {
                e: odalar[j].id
                for e, j, _ in bolge_eslesmeleri(etiket, [n.noktalar for n in odalar])
            }
            oda_oda = [(u, w, d["kapi"]) for u, w, d in graf.edges(data=True) if DIS not in (u, w)]
            dogru = sum(
                1
                for u, w, i in oda_oda
                if i in kapi_es
                and u in oda_es
                and w in oda_es
                and cevap_kenar.get(kapi_es[i]) == frozenset((oda_es[u], oda_es[w]))
            )
            n_oda = graf.number_of_nodes() - 1
            satirlar.append(
                {
                    "ad": o.ad,
                    "varyant": v,
                    "oda_dogru": len(oda_es),
                    "oda_fazla": n_oda - len(oda_es),
                    "oda_eksik": len(odalar) - len(oda_es),
                    "kenar_dogru": dogru,
                    "kenar_fazla": len(oda_oda) - dogru,
                    "kenar_eksik": len(cevap_kenar) - dogru,
                    "ayrim": ayrim,
                    "dis_kapi": graf.number_of_edges() - len(oda_oda),
                    "oda_sure_s": round(sure, 2),
                }
            )
        s = satirlar[-1]
        print(
            f"{o.ad:24} oda {s['oda_dogru']}/{s['oda_fazla']}/{s['oda_eksik']}  "
            f"kenar {s['kenar_dogru']}/{s['kenar_fazla']}/{s['kenar_eksik']}  ({s['varyant']})",
            flush=True,
        )

    ad = f"oda_deney_{a.bolum}{'_' + a.etiket if a.etiket else ''}.csv"
    with (SONUC / ad).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)
    gruplar = defaultdict(list)
    for s in satirlar:
        gruplar[s["varyant"]].append(s)
    for v, ss in gruplar.items():
        top = {k: sum(s[k] for s in ss) for k in ss[0] if k not in ("ad", "varyant", "oda_sure_s")}
        parca = []
        for tur in ("oda", "kenar"):
            d, fz, e = top[f"{tur}_dogru"], top[f"{tur}_fazla"], top[f"{tur}_eksik"]
            kes, duy = d / max(d + fz, 1), d / max(d + e, 1)
            parca.append(
                f"{tur} {d}/{fz}/{e} K {kes:.3f} D {duy:.3f} F1 {2 * kes * duy / max(kes + duy, 1e-9):.3f}"
            )
        print(f"{v:16} " + " | ".join(parca) + f" | dis kapi {top['dis_kapi']}")
    print(f"{len(satirlar)} satir, {time.perf_counter() - t_basla:.0f} s -> results/{ad}")


if __name__ == "__main__":
    main()
