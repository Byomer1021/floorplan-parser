"""CVC-FP'de kapı ve pencere deneyi.

Duvarlar iki ölçekli akıştan (``iki_olcekli_duvarlar``), açıklıklar
``aciklik_bul``'dan. Eşleme: bulunan poligon ile cevaptaki Door / Window
poligonunun IoU'su ≥ --esik, bire bir.

Plan başına ayrıca hata ayrışımı için:
- kapi_bosluklu: çevresinde (poligonu 10 px genişletilmiş) en az bir boşluk
  adayının (parça ya da köşe) orta noktası bulunan cevap kapısı sayısı.
  Boşluk bulunmadıysa yay testi hiç denenmedi demektir.

Yalnız I-II, dolu duvarlı planlar (cevap dolgusu ≥ 0.8; değerlendirme
alt kümesi, sistem stili bilmez). Varsayılan bölüm geliştirme.

Kullanım: uv run python scripts/aciklik_deney.py [--etiket ad]
"""

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from floorplan_parser.aciklik import (
    VARSAYILAN_ACIKLIK_KURAL,
    aciklik_bul,
    bosluklar,
    kose_bosluklari,
)
from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.degerlendirme import parca_maskesi, poligon_eslestir
from floorplan_parser.duvar import iki_olcekli_duvarlar
from floorplan_parser.onisleme import delik_doldur, ikile

SONUC = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bolum", default="gelistirme", choices=["gelistirme", "test"])
    ap.add_argument("--esik", type=float, default=0.5)
    ap.add_argument("--yay-esik", type=float, default=VARSAYILAN_ACIKLIK_KURAL.yay_esik)
    ap.add_argument("--kismi-esik", type=float, default=VARSAYILAN_ACIKLIK_KURAL.kismi_esik)
    ap.add_argument("--kismi-acilar", default="30,45,60")
    ap.add_argument("--kismi-pencere", type=int, default=1)
    ap.add_argument("--kismi-kanat-yan", type=float, default=0.3)
    ap.add_argument("--kapi-yontemi", default="bosluk_kose", choices=["bosluk", "bosluk_kose"])
    ap.add_argument("--cift-kanat", type=int, default=int(VARSAYILAN_ACIKLIK_KURAL.cift_kanat))
    ap.add_argument(
        "--kanat-dikdortgen", type=int, default=int(VARSAYILAN_ACIKLIK_KURAL.kanat_dikdortgen)
    )
    ap.add_argument(
        "--genis-pencere", type=int, default=int(VARSAYILAN_ACIKLIK_KURAL.genis_pencere)
    )
    ap.add_argument(
        "--genis-kisa-duvar", type=int, default=int(VARSAYILAN_ACIKLIK_KURAL.genis_kisa_duvar)
    )
    ap.add_argument(
        "--olcekler",
        default="",
        help="yarıçap/genişlik taraması, ör. 0.83:1.07:0.04 (baş:son:adım) ya da 0.85,0.92",
    )
    ap.add_argument("--ek-olcekler", default="", help="kapı boyunda ek ölçekler, --olcekler gibi")
    ap.add_argument("--kavsak-aday", type=int, default=int(VARSAYILAN_ACIKLIK_KURAL.kavsak_aday))
    ap.add_argument("--etiket", default="")
    a = ap.parse_args()
    kural = replace(
        VARSAYILAN_ACIKLIK_KURAL,
        yay_esik=a.yay_esik,
        kismi_esik=a.kismi_esik,
        kismi_acilar=tuple(float(x) for x in a.kismi_acilar.split(",")),
        kismi_pencerede=bool(a.kismi_pencere),
        kismi_kanat_yan=a.kismi_kanat_yan,
        cift_kanat=bool(a.cift_kanat),
        kanat_dikdortgen=bool(a.kanat_dikdortgen),
        genis_pencere=bool(a.genis_pencere),
        genis_kisa_duvar=bool(a.genis_kisa_duvar),
        kavsak_aday=bool(a.kavsak_aday),
    )

    def olcek_oku(metin: str) -> tuple[float, ...]:
        if ":" in metin:
            bas, son, adim = (float(x) for x in metin.split(":"))
            return tuple(round(v, 3) for v in np.arange(bas, son + adim / 2, adim))
        return tuple(float(x) for x in metin.split(","))

    if a.olcekler:
        kural = replace(kural, yaricap_olcekleri=olcek_oku(a.olcekler))
    if a.ek_olcekler:
        kural = replace(kural, kapi_ek_olcekleri=olcek_oku(a.ek_olcekler))
    print("yarıçap ölçekleri:", kural.yaricap_olcekleri, "ek:", kural.kapi_ek_olcekleri, flush=True)

    satirlar = []
    t_basla = time.perf_counter()
    for o in ornekler(bolum=a.bolum):
        if o.grup != "I-II":
            continue
        g, c = yukle(o)
        ham = ikile(g)
        if (ham[maske(c, "Wall") > 0] > 0).mean() < 0.8:
            continue
        t0 = time.perf_counter()
        ikili = delik_doldur(ham, 4)
        kalin, kalin_p, ince_p = iki_olcekli_duvarlar(ikili)
        parcalar = kalin_p + ince_p
        duvar = np.maximum(kalin, parca_maskesi(ince_p, g.shape))
        kapilar, pencereler, kalan = aciklik_bul(ikili, parcalar, duvar, kural, a.kapi_yontemi)
        sure = time.perf_counter() - t0

        cevap_kapi = [n.noktalar for n in c.sinif("Door")]
        cevap_pencere = [n.noktalar for n in c.sinif("Window")]
        kd, kf, ke = poligon_eslestir([k.poligon() for k in kapilar], cevap_kapi, a.esik)
        pd, pf, pe = poligon_eslestir([p.poligon() for p in pencereler], cevap_pencere, a.esik)
        ortalar = [
            ((b.a[0] + b.b[0]) / 2, (b.a[1] + b.b[1]) / 2)
            for b in bosluklar(parcalar, duvar, kural) + kose_bosluklari(duvar, kural)
        ]
        bosluklu = sum(
            any(cv2.pointPolygonTest(n.astype(np.float32), o_, True) >= -10 for o_ in ortalar)
            for n in cevap_kapi
        )
        satirlar.append(
            {
                "ad": o.ad,
                "kapi_dogru": kd,
                "kapi_fazla": kf,
                "kapi_eksik": ke,
                "kapi_bosluklu": bosluklu,
                "pencere_dogru": pd,
                "pencere_fazla": pf,
                "pencere_eksik": pe,
                "bosluk_kalan": len(kalan),
                "sure_s": round(sure, 2),
            }
        )
        print(
            f"{o.ad:24} kapı {kd}/{kf}/{ke} (boşluklu {bosluklu}/{len(cevap_kapi)})  pencere {pd}/{pf}/{pe}",
            flush=True,
        )

    ad = f"aciklik_deney_{a.bolum}{'_' + a.etiket if a.etiket else ''}.csv"
    with (SONUC / ad).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)
    top = {k: sum(s[k] for s in satirlar) for k in satirlar[0] if k not in ("ad", "sure_s")}
    for tur in ("kapi", "pencere"):
        d, fz, e = top[f"{tur}_dogru"], top[f"{tur}_fazla"], top[f"{tur}_eksik"]
        kes, duy = d / max(d + fz, 1), d / max(d + e, 1)
        print(
            f"{tur}: doğru {d} fazla {fz} eksik {e}  kesinlik {kes:.3f} duyarlılık {duy:.3f} F1 {2 * kes * duy / max(kes + duy, 1e-9):.3f}"
        )
    print(
        f"boşluk adayı olan cevap kapısı: {top['kapi_bosluklu']}/{top['kapi_dogru'] + top['kapi_eksik']}"
    )
    print(f"{len(satirlar)} plan, {time.perf_counter() - t_basla:.0f} s -> results/{ad}")


if __name__ == "__main__":
    main()
