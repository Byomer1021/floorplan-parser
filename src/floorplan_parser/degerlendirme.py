"""Değerlendirme: piksel IoU ve duvar parçası eşleme."""

import math
from collections.abc import Sequence

import cv2
import numpy as np

from floorplan_parser.duvar import Parca, aci_farki


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    birlesim = (a | b).sum()
    return float((a & b).sum() / birlesim) if birlesim else 1.0


_ALT_PIKSEL = 4


def parca_maskesi(parcalar: Sequence[Parca], boyut: tuple[int, int]) -> np.ndarray:
    """Her parçayı ``kapsam`` uçları arasında kalınlığı kadar dikdörtgen çizer
    (kapsam yoksa graf uçları). Köşeyi dolduran, öbür duvarın kapsamı."""
    m = np.zeros(boyut, np.uint8)
    for p in parcalar:
        x1, y1, x2, y2 = p.kapsam or (p.x1, p.y1, p.x2, p.y2)
        a, b = np.array([x1, y1]), np.array([x2, y2])
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        # fillPoly kenar piksellerini de boyar: köşeler sınır piksellerinin
        # merkezinde, yani yarım genişlik (kalınlık - 1) / 2.
        n = np.array([-d[1], d[0]]) * (p.kalinlik - 1) / 2
        kose = np.round(np.array([a + n, b + n, b - n, a - n]) * (1 << _ALT_PIKSEL))
        # Tek tek: fillPoly birden çok poligonu tek-çift kuralıyla doldurur,
        # köşede üst üste binen iki duvar delik olur.
        cv2.fillConvexPoly(m, kose.astype(np.int32), 255, cv2.LINE_8, _ALT_PIKSEL)
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
