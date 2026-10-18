"""Pencere hatalarının sınıflandırması (geliştirme, I-II dolu duvarlı planlar).

Akış ``akis.plani_coz``'un açıklık kısmıyla aynı.

Eksik pencere (eşleşmeyen cevap), sırayla ilk tutan:
- duvar_sayildi: cevap poligonunun ≥ %50'si bizim duvar maskemizde
- geometri: bulunan bir pencereyle 0.1 ≤ IoU < 0.5
- aday_yok: çevresinde (10 px) parça, köşe ya da geniş boşluk yok
- yay: parça boşluğu var ama bir tek yay (kapı ya da kanatlı pencere) tutmuş
- yalniz_kose: yalnız köşe boşluğu var (pencere orada denenmiyor)
- bulunamadi: parça ya da geniş boşlukta ne çizgi ne çift kanat testi geçti
- diger

Fazla pencere (eşleşmeyen bulunan): kapi (cevaptaki kapıyla IoU ≥ 0.3),
geometri (cevaptaki pencereyle 0.1 ≤ IoU < 0.5), bos. Eşleşen bulunanlar da
``hata=dogru`` satırı olarak yazılır (çeşit ve genişlik karşılaştırması için).

Çıktı: results/pencere_hata.csv; ``--kirp N`` her eksik türünden N örneği
results/ara/pencere_hata/ altına kırpılmış görüntü olarak yazar.
Kullanım: uv run python scripts/pencere_hata.py [--kirp 6]
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from floorplan_parser.aciklik import (
    VARSAYILAN_ACIKLIK_KURAL,
    aciklik_bul,
    bosluklar,
    genis_bosluklar,
    ince_murekkep,
    kapi_dene,
    kose_bosluklari,
    pencere_dene,
)
from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.degerlendirme import parca_maskesi, poligon_eslesmeleri, poligon_iou
from floorplan_parser.duvar import iki_olcekli_duvarlar
from floorplan_parser.onisleme import delik_doldur, ikile

SONUC = Path(__file__).resolve().parents[1] / "results"


def yakin(bosluk_listesi, poligon: np.ndarray, pay: float = 10.0) -> list:
    p32 = poligon.astype(np.float32)
    return [
        b
        for b in bosluk_listesi
        if cv2.pointPolygonTest(p32, ((b.a[0] + b.b[0]) / 2, (b.a[1] + b.b[1]) / 2), True) >= -pay
    ]


def yayda(b, yay_poligon: list[np.ndarray]) -> bool:
    """Boşluğun ortası bir yayın açılma alanında (aciklik_bul'daki gibi)."""
    orta = ((b.a[0] + b.b[0]) / 2, (b.a[1] + b.b[1]) / 2)
    return any(cv2.pointPolygonTest(yp, orta, True) >= -(b.kalinlik / 2 + 2) for yp in yay_poligon)


def kirp(gri, duvar, cevap, bulunan, yer, yol: Path, kenar: int = 60) -> None:
    """Gri görüntü üstüne: duvar maskesi mavi, cevap yeşil, bulunan pencere kırmızı."""
    x0, y0 = np.floor(yer.min(0)).astype(int) - kenar
    x1, y1 = np.ceil(yer.max(0)).astype(int) + kenar
    h, w = gri.shape
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    renk = cv2.cvtColor(gri[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    mavi = duvar[y0:y1, x0:x1] > 0
    renk[mavi] = (0.6 * renk[mavi] + 0.4 * np.array([255, 120, 0])).astype(np.uint8)
    for p, r in [(c, (0, 170, 0)) for c in cevap] + [(b, (0, 0, 230)) for b in bulunan]:
        cv2.polylines(renk, [np.round(p - [x0, y0]).astype(np.int32)], True, r, 1)
    renk = cv2.resize(renk, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imencode(".png", renk)[1].tofile(str(yol))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kirp", type=int, default=0)
    a = ap.parse_args()
    kural = VARSAYILAN_ACIKLIK_KURAL
    kirp_dizin = SONUC / "ara" / "pencere_hata"
    if a.kirp:
        kirp_dizin.mkdir(parents=True, exist_ok=True)
    kirpilan: dict[str, int] = defaultdict(int)
    satirlar = []
    toplam = Counter()
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
        _, pencereler, _ = aciklik_bul(ikili, parcalar, duvar, kural)
        murekkep = ince_murekkep(ikili, duvar, kural.yay_tol)
        parca_b = bosluklar(parcalar, duvar, kural)
        kose_b = kose_bosluklari(duvar, kural)
        genis_b = genis_bosluklar(parcalar, duvar, kisa_duvar=kural.genis_kisa_duvar)
        parca_kume, genis_kume = set(parca_b), set(genis_b)
        yay_poligon = []
        for b in parca_b + kose_b:
            k = kapi_dene(b, murekkep)
            if k is not None and k.skor >= kural.yay_esik:
                yay_poligon.append(k.poligon().astype(np.float32))

        cevap_p = [n.noktalar for n in c.sinif("Window")]
        cevap_k = [n.noktalar for n in c.sinif("Door")]
        bulunan = [p.poligon() for p in pencereler]
        ciftler = poligon_eslesmeleri(bulunan, cevap_p)
        eb, ec = {i for i, _ in ciftler}, {j for _, j in ciftler}
        toplam.update(dogru=len(ciftler), fazla=len(bulunan) - len(ciftler))
        toplam.update(eksik=len(cevap_p) - len(ciftler))

        for i, (p, bp) in enumerate(zip(pencereler, bulunan)):
            kap = max((poligon_iou(bp, ck) for ck in cevap_k), default=0.0)
            pen = max((poligon_iou(bp, cp) for cp in cevap_p), default=0.0)
            if i in eb:
                tur = "dogru"
            else:
                tur = "kapi" if kap >= 0.3 else "geometri" if pen >= 0.1 else "bos"
            x, y = bp.mean(0)
            satirlar.append(
                {
                    "ad": o.ad,
                    "hata": "dogru" if i in eb else "fazla",
                    "tur": tur,
                    "cesit": "cift"
                    if p.kanatlar
                    else "kanat"
                    if p.kanat is not None
                    else f"cizgi{p.cizgi}",
                    "kaynak": "parca"
                    if p.bosluk in parca_kume
                    else "genis"
                    if p.bosluk in genis_kume
                    else "kose",
                    "genislik": round(p.bosluk.genislik),
                    "kalinlik": round(p.bosluk.kalinlik, 1),
                    "duvar_orani": "",
                    "genis": "",
                    "en_iyi_iou": round(pen, 2),
                    "x": round(x),
                    "y": round(y),
                }
            )
            if i not in eb and kirpilan["fazla_" + tur] < a.kirp:
                kirpilan["fazla_" + tur] += 1
                yol = kirp_dizin / f"fazla_{tur}_{kirpilan['fazla_' + tur]}_{o.ad}.png"
                kirp(g, duvar, cevap_p, [bp], bp, yol)

        for j, cp in enumerate(cevap_p):
            if j in ec:
                continue
            m = np.zeros(g.shape, np.uint8)
            cv2.fillPoly(m, [np.round(cp).astype(np.int32)], 1)
            duvar_orani = float((duvar[m > 0] > 0).mean()) if m.any() else 0.0
            (_, _), (rw, rh), _ = cv2.minAreaRect(cp.astype(np.float32))
            pb, kb, gb = yakin(parca_b, cp), yakin(kose_b, cp), yakin(genis_b, cp)
            iou_bulunan = max((poligon_iou(bp, cp) for bp in bulunan), default=0.0)

            serbest = [b for b in pb if not yayda(b, yay_poligon)]
            if duvar_orani >= 0.5:
                tur = "duvar_sayildi"
            elif iou_bulunan >= 0.1:
                tur = "geometri"
            elif not pb and not kb and not gb:
                tur = "aday_yok"
            elif pb and not serbest and not gb:
                tur = "yay"
            elif not pb and not gb:
                tur = "yalniz_kose"
            elif all(pencere_dene(b, ikili, kural) is None for b in serbest + gb):
                tur = "bulunamadi"
            else:
                tur = "diger"
            x, y = cp.mean(0)
            satirlar.append(
                {
                    "ad": o.ad,
                    "hata": "eksik",
                    "tur": tur,
                    "cesit": "",
                    "kaynak": "",
                    "genislik": round(max(rw, rh)),
                    "kalinlik": round(min(rw, rh), 1),
                    "duvar_orani": round(duvar_orani, 2),
                    "genis": int(bool(gb)),
                    "en_iyi_iou": round(iou_bulunan, 2),
                    "x": round(x),
                    "y": round(y),
                }
            )
            if kirpilan[tur] < a.kirp:
                kirpilan[tur] += 1
                kirp(g, duvar, [cp], bulunan, cp, kirp_dizin / f"{tur}_{kirpilan[tur]}_{o.ad}.png")
        print(o.ad, flush=True)

    with (SONUC / "pencere_hata.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)
    print("toplam", dict(toplam))
    for hata in ("fazla", "eksik"):
        print(hata, dict(Counter(s["tur"] for s in satirlar if s["hata"] == hata)))


if __name__ == "__main__":
    main()
