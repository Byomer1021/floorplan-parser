"""Kapı hatalarının sınıflandırması (geliştirme, I-II dolu duvarlı planlar).

Akış ``akis.plani_coz`` ile aynı.

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

Eksik kapı için ayrıca:
- kismi_pay: yarı açık varsayımı da açıkken (eşik 0) en iyi pay
- birlesik: kapının iki yanındaki cevap odaları bizde aynı bölgede (kapı
  bulunamadığı için oda birleşmiş olabilir)
- aci: cevap kapısının duvarının yatayla açısı (derece, 0-90)

Çıktı: results/aciklik_hata.csv (hata başına satır); ``--kirp N`` her eksik
türünden, önce birleşik olanlardan N görüntü results/ara/kapi_hata/ altına.
Kullanım: uv run python scripts/aciklik_hata.py [--kirp 8]
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from floorplan_parser.aciklik import (
    VARSAYILAN_ACIKLIK_KURAL,
    bosluklar,
    ince_murekkep,
    kapi_dene,
    kose_bosluklari,
)
from floorplan_parser.akis import plani_coz
from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.degerlendirme import poligon_eslesmeleri, poligon_iou
from floorplan_parser.onisleme import delik_doldur, ikile

SONUC = Path(__file__).resolve().parents[1] / "results"


def oda_bolgesi(etiket: np.ndarray, poligon: np.ndarray) -> int:
    """Cevap odasının içinde en çok piksel kaplayan bölge (0: yok)."""
    m = np.zeros(etiket.shape, np.uint8)
    cv2.fillPoly(m, [np.round(poligon).astype(np.int32)], 1)
    v = etiket[(m > 0) & (etiket > 0)]
    return int(np.bincount(v).argmax()) if v.size else 0


def kirp(gri, duvar, cevap, bulunan, yer, yol: Path, kenar: int = 120) -> None:
    """Gri plan; duvar maskesi mavi, cevap kapısı yeşil, bulunan kapılar kırmızı."""
    x0, y0 = np.floor(yer.min(0)).astype(int) - kenar
    x1, y1 = np.ceil(yer.max(0)).astype(int) + kenar
    h, w = gri.shape
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    renk = cv2.cvtColor(gri[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    mavi = duvar[y0:y1, x0:x1] > 0
    renk[mavi] = (0.6 * renk[mavi] + 0.4 * np.array([255, 120, 0])).astype(np.uint8)
    for p, r in [(cevap, (0, 170, 0))] + [(b, (0, 0, 230)) for b in bulunan]:
        cv2.polylines(renk, [np.round(p - [x0, y0]).astype(np.int32)], True, r, 2)
    olcek = 600 / max(renk.shape[:2])
    renk = cv2.resize(renk, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)
    cv2.imencode(".png", renk)[1].tofile(str(yol))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kirp", type=int, default=0)
    a = ap.parse_args()
    kural = VARSAYILAN_ACIKLIK_KURAL
    kirp_dizin = SONUC / "ara" / "kapi_hata"
    if a.kirp:
        kirp_dizin.mkdir(parents=True, exist_ok=True)
    kirpilan: Counter = Counter()
    satirlar = []
    for o in ornekler(bolum="gelistirme"):
        if o.grup != "I-II":
            continue
        g, c = yukle(o)
        ham = ikile(g)
        if (ham[maske(c, "Wall") > 0] > 0).mean() < 0.8:
            continue
        s = plani_coz(g)
        ikili = delik_doldur(ham, 4)
        parcalar = s.kalin_parcalar + s.ince_parcalar
        duvar, kapilar = s.duvar_maske, s.kapilar
        murekkep = ince_murekkep(ikili, duvar, kural.yay_tol)
        adaylar = bosluklar(parcalar, duvar, kural) + kose_bosluklari(duvar, kural)
        yaylar = [
            (b, kapi_dene(b, murekkep), kapi_dene(b, murekkep, kismi_esik=0.0)) for b in adaylar
        ]
        kapi_c = c.sinif("Door")
        cevap_kapi = [n.noktalar for n in kapi_c]
        cevap_pencere = [n.noktalar for n in c.sinif("Window")]
        bulunan = [k.poligon() for k in kapilar]

        # Kapı → iki yanındaki cevap odaları (access ilişkisi).
        sinif = {n.id: n for n in c.nesneler}
        kapi_odalari: dict[int, list] = {}
        for tip, ids in c.iliskiler:
            if tip != "access":
                continue
            odalar = [sinif[i] for i in ids if i in sinif and sinif[i].sinif == "Room"]
            for i in ids:
                if i in sinif and sinif[i].sinif == "Door" and len(odalar) == 2:
                    kapi_odalari[i] = odalar

        ciftler = poligon_eslesmeleri(bulunan, cevap_kapi)
        eslesen_k, eslesen_c = {i for i, _ in ciftler}, {j for _, j in ciftler}

        for i, k in enumerate(kapilar):
            if i in eslesen_k:
                continue
            pen = max((poligon_iou(bulunan[i], cp) for cp in cevap_pencere), default=0.0)
            kap = max((poligon_iou(bulunan[i], cv) for cv in cevap_kapi), default=0.0)
            tur = "pencere" if pen >= 0.3 else "geometri" if kap >= 0.1 else "bos"
            satirlar.append(
                {
                    "ad": o.ad,
                    "hata": "fazla",
                    "tur": tur,
                    "pay": round(k.skor, 2),
                    "kismi_pay": "",
                    "birlesik": "",
                    "aci": "",
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
                (b, k, kk)
                for b, k, kk in yaylar
                if cv2.pointPolygonTest(cv32, ((b.a[0] + b.b[0]) / 2, (b.a[1] + b.b[1]) / 2), True)
                >= -10
            ]
            iyi = [k for _, k, _ in yakin if k is not None and k.skor >= kural.yay_esik]
            en_iyi = max((k.skor for _, k, _ in yakin if k is not None), default=0.0)
            kismi = max((kk.skor for _, _, kk in yakin if kk is not None), default=0.0)
            iou_bulunan = max((poligon_iou(b, cv) for b in bulunan), default=0.0)
            if not yakin:
                tur = "aday_yok"
            elif not iyi:
                tur = "yay_zayif"
            elif all(k.yaricap < kural.kapi_min_yaricap for k in iyi):
                tur = "kucuk_yaricap"
            elif iou_bulunan >= 0.1:
                tur = "geometri"
            else:
                tur = "diger"
            odalar = kapi_odalari.get(kapi_c[j].id)
            birlesik = ""
            if odalar:
                r1, r2 = (oda_bolgesi(s.oda_etiket, n.noktalar) for n in odalar)
                birlesik = int(r1 > 0 and r1 == r2)
            aci = cv2.minAreaRect(cv32)[2]
            x, y = cv.mean(0)
            satirlar.append(
                {
                    "ad": o.ad,
                    "hata": "eksik",
                    "tur": tur,
                    "pay": round(en_iyi, 2),
                    "kismi_pay": round(kismi, 2),
                    "birlesik": birlesik,
                    "aci": round(min(abs(aci) % 90, 90 - abs(aci) % 90)),
                    "yaricap": round(float(np.ptp(cv, 0).max())),
                    "x": round(x),
                    "y": round(y),
                }
            )
            anahtar = f"{tur}_{'b' if birlesik == 1 else 'd'}"
            if birlesik == 1 and kirpilan[anahtar] < a.kirp:
                kirpilan[anahtar] += 1
                yol = kirp_dizin / f"{anahtar}_{kirpilan[anahtar]}_{o.ad}.png"
                kirp(g, duvar, cv, bulunan, cv, yol)
        print(o.ad, flush=True)

    with (SONUC / "aciklik_hata.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)
    for hata in ("fazla", "eksik"):
        print(hata, dict(Counter(s["tur"] for s in satirlar if s["hata"] == hata)))
    eksik = [s for s in satirlar if s["hata"] == "eksik"]
    print("eksik, birlesik:", dict(Counter((s["tur"], s["birlesik"]) for s in eksik)))
    print(
        "eksik, açı > 10 derece:",
        dict(Counter(s["tur"] for s in eksik if s["aci"] != "" and s["aci"] > 10)),
    )
    for esik in (0.95, 0.9, 0.85, 0.8):
        print(
            f"eksik yay_zayif, kismi_pay >= {esik}:",
            sum(1 for s in eksik if s["tur"] == "yay_zayif" and s["kismi_pay"] >= esik),
        )


if __name__ == "__main__":
    main()
