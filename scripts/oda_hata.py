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

Fazla oda (eşleşmeyen bulunan bölge):
- cevapsiz: bölgenin yarısından azı cevaptaki odaların içinde (balkon,
  teras, merdiven boşluğu, dış alan)
- birlesik: iki ya da daha çok cevap odasını kaplıyor (eksik tarafta da
  sayılır)
- kirinti: ağırlıkla içinde olduğu cevap odası başka bir bölgeyle eşleşmiş
- bolunmus: o cevap odası eşleşmemiş ve başka bölgeler de onu kaplıyor
- kismi: tek bölge, IoU < 0.5

Kullanım: uv run python scripts/oda_hata.py [--varyant duvar:yok:0:151:0] [--kirp N]
(varyant biçimi oda_deney.py ile aynı; INCE yok sayılır). ``--kirp`` her
türden N planın etiket görüntüsünü results/ara/oda_hata/ altına yazar.
"""

import argparse
import csv
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from floorplan_parser.aciklik import aciklik_bul, bosluklar, genis_bosluklar, kose_bosluklari
from floorplan_parser.akis import plani_coz
from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.degerlendirme import (
    bolge_eslesmeleri,
    parca_maskesi,
    poligon_eslesmeleri,
)
from floorplan_parser.duvar import iki_olcekli_duvarlar
from floorplan_parser.oda import DIS, VARSAYILAN_ODA_KURAL, oda_grafi
from floorplan_parser.onisleme import delik_doldur, ikile

SONUC = Path(__file__).resolve().parents[1] / "results"


def boya(gri, etiket, odalar, vurgu, yol: Path, kenar: int = 250) -> None:
    """Bölgeler rastgele renkte, dış gri; cevap odaları yeşil, vurgulanan
    cevap odası kalın kırmızı; vurgunun çevresi kırpılır."""
    rng = np.random.default_rng(0)
    renk = rng.integers(60, 255, (int(etiket.max()) + 2, 3)).astype(np.uint8)
    goruntu = cv2.cvtColor(gri, cv2.COLOR_GRAY2BGR)
    m = etiket > 0
    goruntu[m] = (0.5 * goruntu[m] + 0.5 * renk[etiket[m]]).astype(np.uint8)
    goruntu[etiket == DIS] = (0.6 * goruntu[etiket == DIS] + 0.4 * 128).astype(np.uint8)
    for n in odalar:
        cv2.polylines(goruntu, [np.round(n.noktalar).astype(np.int32)], True, (0, 150, 0), 2)
    cv2.polylines(goruntu, [np.round(vurgu).astype(np.int32)], True, (0, 0, 230), 4)
    h, w = gri.shape
    x0, y0 = np.maximum(np.floor(vurgu.min(0)).astype(int) - kenar, 0)
    x1, y1 = np.ceil(vurgu.max(0)).astype(int) + kenar
    kesit = goruntu[y0 : min(y1, h), x0 : min(x1, w)]
    olcek = 800 / max(kesit.shape[:2])
    kesit = cv2.resize(kesit, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)
    cv2.imencode(".png", kesit)[1].tofile(str(yol))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--varyant", default="", help="boşsa akis.plani_coz (varsayılan ayarlar)")
    ap.add_argument("--kirp", type=int, default=0)
    a = ap.parse_args()
    kirp_dizin = SONUC / "ara" / "oda_hata"
    if a.kirp:
        kirp_dizin.mkdir(parents=True, exist_ok=True)
    kirpilan: Counter = Counter()
    if a.varyant:
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
        if not a.varyant:
            s_ = plani_coz(g)
            etiket, kapilar = s_.oda_etiket, s_.kapilar
        else:
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
        ciftler = bolge_eslesmeleri(etiket, [n.noktalar for n in odalar])
        eslesen = {j for _, j, _ in ciftler}
        eslesen_bolge = {e for e, _, _ in ciftler}
        sinif = {n.id: n.sinif for n in c.nesneler}
        iliski: dict[frozenset, str] = {}
        cift_kapi: dict[frozenset, set] = {}
        for tip, ids in c.iliskiler:
            if tip != "access":
                continue
            oda_ids = [i for i in ids if sinif.get(i) == "Room"]
            tur = "ayrim" if any(sinif.get(i) == "Separation" for i in ids) else "kapi"
            for x in oda_ids:
                for y in oda_ids:
                    if x != y:
                        iliski.setdefault(frozenset((x, y)), tur)
                        cift_kapi.setdefault(frozenset((x, y)), set()).update(
                            i for i in ids if sinif.get(i) == "Door"
                        )
        kapi_c = c.sinif("Door")
        bulunan_kapi = {
            kapi_c[j].id
            for _, j in poligon_eslesmeleri(
                [k.poligon() for k in kapilar], [n.noktalar for n in kapi_c]
            )
        }

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
            kapi = ""
            if tur == "birlesti_kapi":
                kapilari = set().union(*(cift_kapi.get(frozenset((n.id, x)), set()) for x in ortak))
                kapi = "bulundu" if kapilari & bulunan_kapi else "yok"
            satirlar.append(
                {"ad": o.ad, "hata": "eksik", "oda": n.id, "tur": tur, "alan": alan, "kapi": kapi}
            )
            if kirpilan[tur] < a.kirp:
                kirpilan[tur] += 1
                boya(
                    g, etiket, odalar, n.noktalar, kirp_dizin / f"{tur}_{kirpilan[tur]}_{o.ad}.png"
                )

        # Fazla bölgeler: her bölgenin cevap odalarıyla kesişimi.
        bolge_alan = np.bincount(etiket[etiket > 0].ravel(), minlength=int(etiket.max()) + 1)
        for e in np.flatnonzero(bolge_alan):
            if e in eslesen_bolge:
                continue
            kesisim = np.array([say.get(int(e), 0) for say, _ in sayim])
            alanlar = np.array([alan_j for _, alan_j in sayim])
            if kesisim.sum() < 0.5 * bolge_alan[e]:
                tur = "cevapsiz"
            elif (kesisim > 0.3 * alanlar).sum() >= 2:
                tur = "birlesik"
            else:
                j = int(kesisim.argmax())
                baskasi = sum(
                    1 for e2 in sayim[j][0] if e2 not in (0, DIS, int(e)) and sayim[j][0][e2] > 0
                )
                tur = "kirinti" if j in eslesen else "bolunmus" if baskasi else "kismi"
            ys, xs = np.nonzero(etiket == e)
            satirlar.append(
                {
                    "ad": o.ad,
                    "hata": "fazla",
                    "oda": int(e),
                    "tur": tur,
                    "alan": int(bolge_alan[e]),
                    "kapi": "",
                }
            )
            if kirpilan["fazla_" + tur] < a.kirp:
                kirpilan["fazla_" + tur] += 1
                x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
                kutu = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
                yol = kirp_dizin / f"fazla_{tur}_{kirpilan['fazla_' + tur]}_{o.ad}.png"
                boya(g, etiket, odalar, kutu, yol)
        print(o.ad, flush=True)

    with (SONUC / "oda_hata.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)
    for hata in ("eksik", "fazla"):
        ss = [s for s in satirlar if s["hata"] == hata]
        print(hata, len(ss), dict(Counter(s["tur"] for s in ss)))
    print(
        "birlesti_kapi, cevaptaki kapı:",
        dict(Counter(s["kapi"] for s in satirlar if s["tur"] == "birlesti_kapi")),
    )


if __name__ == "__main__":
    main()
