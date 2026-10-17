"""Cevabı bilinen sentetik kat planı: iki oda, bir kapı, bir pencere.

Gerçek planda doğru cevabı elle işaretlemek gerekir; burada çizen kod cevabı
da verir. Her adımın testi önce burada geçmeli. Bilinen sınır: duvarlar yalnız
yatay/dikey, çizgiler kenar yumuşatmasız, tek çizim stili.

Yerleşim (varsayılan, piksel)::

    (45,45) ───────────────────────────── (555,45)
      │              │                       │
      │     oda 1    │        oda 2          │
      │              ╰╮ kapı (300,180-240)   │
      │              │                       │
      └──═══──────────────────────────────── ┘
         pencere (100-200, 350)
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

MUREKKEP = 0
KAGIT = 255


@dataclass(frozen=True)
class Duvar:
    """Orta çizgi (x1, y1)-(x2, y2), yatay ya da dikey."""

    x1: int
    y1: int
    x2: int
    y2: int
    kalinlik: int


@dataclass(frozen=True)
class Kapi:
    """Menteşe noktası, kanat uzunluğu (= boşluk genişliği); boşluk menteşeden aşağı."""

    mentese: tuple[int, int]
    genislik: int


@dataclass(frozen=True)
class Pencere:
    """Yatay duvardaki boşluk, x1..x2 aralığı, y orta çizgi."""

    x1: int
    x2: int
    y: int


@dataclass
class SentetikPlan:
    goruntu: np.ndarray
    duvar_maskesi: np.ndarray  # yalnız kalın duvar pikselleri, 255
    duvarlar: list[Duvar] = field(default_factory=list)
    kapilar: list[Kapi] = field(default_factory=list)
    pencereler: list[Pencere] = field(default_factory=list)


def _duvar_ciz(maske: np.ndarray, d: Duvar) -> None:
    h = d.kalinlik // 2
    if d.y1 == d.y2:
        maske[d.y1 - h : d.y1 + h, min(d.x1, d.x2) : max(d.x1, d.x2)] = 255
    elif d.x1 == d.x2:
        maske[min(d.y1, d.y2) : max(d.y1, d.y2), d.x1 - h : d.x1 + h] = 255
    else:
        raise ValueError("yalnız yatay/dikey duvar")


def plan_ciz(duvar: int = 10, ince: int = 2, gurultu: float = 0.0, tohum: int = 0) -> SentetikPlan:
    """600x400 plan. ``gurultu``: her pikselin rastgele ters çevrilme olasılığı."""
    h = duvar // 2
    kapi = Kapi(mentese=(300, 180), genislik=60)
    pencere = Pencere(x1=100, x2=200, y=350)
    duvarlar = [
        Duvar(45, 50, 555, 50, duvar),  # üst, köşeleri de kaplar
        Duvar(50, 50, 50, 350, duvar),  # sol
        Duvar(550, 50, 550, 350, duvar),  # sağ
        Duvar(45, 350, pencere.x1, 350, duvar),  # alt, pencereye kadar
        Duvar(pencere.x2, 350, 555, 350, duvar),  # alt, pencereden sonra
        Duvar(300, 50 + h, 300, kapi.mentese[1], duvar),  # iç duvar, kapıya kadar
        Duvar(300, kapi.mentese[1] + kapi.genislik, 300, 350 - h, duvar),  # kapıdan sonra
    ]

    maske = np.zeros((400, 600), np.uint8)
    for d in duvarlar:
        _duvar_ciz(maske, d)
    goruntu = np.where(maske > 0, MUREKKEP, KAGIT).astype(np.uint8)

    # Kapı: açık kanat (menteşeden sağa) ve kapalı konumdan açığa çeyrek yay.
    mx, my = kapi.mentese
    r = kapi.genislik
    cv2.line(goruntu, (mx, my), (mx + r, my), MUREKKEP, ince)
    cv2.ellipse(goruntu, (mx, my), (r, r), 0, 0, 90, MUREKKEP, ince)

    # Pencere: boşlukta duvarın iki yüzü ve ortası boyunca üç ince çizgi.
    for y in (pencere.y - h, pencere.y, pencere.y + h - 1):
        cv2.line(goruntu, (pencere.x1, y), (pencere.x2 - 1, y), MUREKKEP, ince)

    if gurultu > 0:
        rng = np.random.default_rng(tohum)
        ters = rng.random(goruntu.shape) < gurultu
        goruntu[ters] = KAGIT - goruntu[ters]

    return SentetikPlan(goruntu, maske, duvarlar, [kapi], [pencere])
