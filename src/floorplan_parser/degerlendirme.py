"""Değerlendirme: piksel IoU ve duvar parçası eşleme."""

import math
from collections.abc import Sequence

import cv2
import numpy as np

from floorplan_parser.duvar import Parca, aci_farki, dortgen_ciz


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    birlesim = (a | b).sum()
    return float((a & b).sum() / birlesim) if birlesim else 1.0


def parca_maskesi(parcalar: Sequence[Parca], boyut: tuple[int, int]) -> np.ndarray:
    """Her parçayı ``kapsam`` uçları arasında kalınlığı kadar dikdörtgen çizer
    (kapsam yoksa graf uçları). Köşeyi dolduran, öbür duvarın kapsamı.
    Tek tek: fillPoly birden çok poligonu tek-çift kuralıyla doldurur."""
    m = np.zeros(boyut, np.uint8)
    for p in parcalar:
        dortgen_ciz(m, p, 255)
    return m


def parca_eslestir(
    tahmin: Sequence[Parca],
    cevap: Sequence[Sequence[float]],
    uc_tol: float,
    aci_tol_derece: float = 5.0,
) -> tuple[int, int, int]:
    """Bire bir eşleme: açı farkı ve iki ucun (sıradan bağımsız) uzaklığı
    tolerans içinde. (doğru, fazla, eksik)."""
    kullanildi = set()
    dogru = 0
    for c in cevap:
        c1, c2 = np.array(c[:2], float), np.array(c[2:4], float)
        c_aci = math.atan2(c2[1] - c1[1], c2[0] - c1[0]) % math.pi
        en_iyi, en_iyi_hata = None, math.inf
        for i, t in enumerate(tahmin):
            if i in kullanildi or aci_farki(t.aci, c_aci) > math.radians(aci_tol_derece):
                continue
            t1, t2 = np.array([t.x1, t.y1]), np.array([t.x2, t.y2])
            hata = min(
                max(np.linalg.norm(t1 - c1), np.linalg.norm(t2 - c2)),
                max(np.linalg.norm(t1 - c2), np.linalg.norm(t2 - c1)),
            )
            if hata <= uc_tol and hata < en_iyi_hata:
                en_iyi, en_iyi_hata = i, hata
        if en_iyi is not None:
            kullanildi.add(en_iyi)
            dogru += 1
    return dogru, len(tahmin) - dogru, len(cevap) - dogru


def poligon_iou(p: np.ndarray, q: np.ndarray) -> float:
    """İki poligonun IoU'su, ortak sınır kutusunda 1/4 piksel raster ile."""
    hepsi = np.vstack([p, q])
    x0, y0 = np.floor(hepsi.min(0)) - 1
    x1, y1 = np.ceil(hepsi.max(0)) + 1
    olcek = 4
    boyut = (int((y1 - y0) * olcek) + 1, int((x1 - x0) * olcek) + 1)
    a, b = np.zeros(boyut, np.uint8), np.zeros(boyut, np.uint8)
    cv2.fillPoly(a, [np.round((p - [x0, y0]) * olcek).astype(np.int32)], 1)
    cv2.fillPoly(b, [np.round((q - [x0, y0]) * olcek).astype(np.int32)], 1)
    birlesim = (a | b).sum()
    return float((a & b).sum() / birlesim) if birlesim else 0.0


def poligon_eslesmeleri(
    tahmin: Sequence[np.ndarray], cevap: Sequence[np.ndarray], esik: float = 0.5
) -> list[tuple[int, int]]:
    """IoU'su ``esik``'ten büyük çiftleri en iyiden başlayarak bire bir eşler:
    (tahmin sırası, cevap sırası) listesi."""
    adaylar = []
    for i, t in enumerate(tahmin):
        tk = (t.min(0), t.max(0))
        for j, c in enumerate(cevap):
            ck = (c.min(0), c.max(0))
            if (tk[0] > ck[1]).any() or (ck[0] > tk[1]).any():
                continue
            v = poligon_iou(t, c)
            if v >= esik:
                adaylar.append((v, i, j))
    ti, cj, ciftler = set(), set(), []
    for _, i, j in sorted(adaylar, reverse=True):
        if i not in ti and j not in cj:
            ti.add(i)
            cj.add(j)
            ciftler.append((i, j))
    return ciftler


def poligon_eslestir(
    tahmin: Sequence[np.ndarray], cevap: Sequence[np.ndarray], esik: float = 0.5
) -> tuple[int, int, int]:
    """(doğru, fazla, eksik), ``poligon_eslesmeleri`` ile."""
    d = len(poligon_eslesmeleri(tahmin, cevap, esik))
    return d, len(tahmin) - d, len(cevap) - d


def bolge_eslesmeleri(
    etiket: np.ndarray, cevap: Sequence[np.ndarray], esik: float = 0.5
) -> list[tuple[int, int, float]]:
    """Etiket görüntüsündeki bölgeler (> 0) ile cevap poligonları arasında bire
    bir eşleme, IoU ≥ ``esik``: (etiket, cevap sırası, IoU)."""
    n = int(etiket.max()) + 1 if etiket.size else 1
    alan = np.bincount(etiket[etiket > 0].ravel(), minlength=n)
    adaylar = []
    for j, c in enumerate(cevap):
        x0, y0 = np.floor(c.min(0)).astype(int)
        x1, y1 = np.ceil(c.max(0)).astype(int) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        kes = etiket[y0:y1, x0:x1]
        m = np.zeros(kes.shape, np.uint8)
        cv2.fillPoly(m, [np.round((c - [x0, y0]) * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
        icerde = kes[(m > 0) & (kes > 0)]
        if icerde.size == 0:
            continue
        kesisim = np.bincount(icerde, minlength=n)
        cevap_alan = int(m.sum())
        for e in np.flatnonzero(kesisim):
            v = kesisim[e] / (alan[e] + cevap_alan - kesisim[e])
            if v >= esik:
                adaylar.append((float(v), int(e), j))
    te, cj, ciftler = set(), set(), []
    for v, e, j in sorted(adaylar, reverse=True):
        if e not in te and j not in cj:
            te.add(e)
            cj.add(j)
            ciftler.append((e, j, v))
    return ciftler
