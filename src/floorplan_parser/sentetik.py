"""Cevabı bilinen sentetik kat planı: iki oda, bir kapı, bir pencere.

Gerçek planda doğru cevabı elle işaretlemek gerekir; burada çizen kod cevabı
da verir. Her adımın testi önce burada geçmeli. Bilinen sınır: duvarlar yalnız
yatay/dikey, çizgiler kenar yumuşatmasız, tek çizim stili.

Yerleşim (varsayılan, piksel, orta çizgiler)::

    (50,50) ──────────────(300,50)────────── (550,50)
      │              │                       │
      │     oda 1    │        oda 2          │
      │              ╰╮ kapı (300,180-240)   │
      │              │                       │
    (50,350)─═══─────(300,350)────────────── (550,350)
         pencere (100-200)

Duvar cevabı en uzun düz parçalar (T-kavşakta bölünmez; bölmek graf adımının
işi). Başka duvara dayanan uç, orta çizgilerin kesişme noktası; kapı/pencere
boşluğuna bakan uç, boşluğun kenarı.
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
    # Uçlar yarım kalınlık uzatılır ki köşeler dolsun; boşluklar sonra oyulur.
    h = d.kalinlik // 2
    if d.y1 == d.y2:
        maske[d.y1 - h : d.y1 + h, min(d.x1, d.x2) - h : max(d.x1, d.x2) + h] = 255
    elif d.x1 == d.x2:
        maske[min(d.y1, d.y2) - h : max(d.y1, d.y2) + h, d.x1 - h : d.x1 + h] = 255
    else:
        raise ValueError("yalnız yatay/dikey duvar")


def plan_ciz(
    duvar: int = 10,
    ince: int = 2,
    gurultu: float = 0.0,
    tohum: int = 0,
    ic_duvar: int | None = None,
    merdiven: bool = False,
    serbest_cizgi: bool = False,
    kapi_genislik: int = 60,
) -> SentetikPlan:
    """600x400 plan. ``gurultu``: her pikselin rastgele ters çevrilme olasılığı.

    ``ic_duvar``: iç (bölme) duvarın kalınlığı, verilmezse ``duvar``.
    ``merdiven``: oda 2'ye altı paralel basamak çizgisi, iç duvar
    kalınlığında. Duvar değil; tek açma çekirdeği bölme duvarla ayıramaz.
    ``serbest_cizgi``: oda 1'e hiçbir duvara değmeyen, iç duvar kalınlığında
    uzun tek çizgi (merdiven taban çizgisi gibi). Duvar değil.
    """
    h = duvar // 2
    ic = ic_duvar or duvar
    kapi = Kapi(mentese=(300, 180), genislik=kapi_genislik)
    pencere = Pencere(x1=100, x2=200, y=350)
    mx, my = kapi.mentese
    r = kapi.genislik
    duvarlar = [
        Duvar(50, 50, 550, 50, duvar),  # üst
        Duvar(50, 50, 50, 350, duvar),  # sol
        Duvar(550, 50, 550, 350, duvar),  # sağ
        Duvar(50, 350, pencere.x1, 350, duvar),  # alt, pencereye kadar
        Duvar(pencere.x2, 350, 550, 350, duvar),  # alt, pencereden sonra
        Duvar(300, 50, 300, my, ic),  # iç duvar, kapıya kadar
        Duvar(300, my + r, 300, 350, ic),  # iç duvar, kapıdan sonra
    ]

    maske = np.zeros((400, 600), np.uint8)
    for d in duvarlar:
        _duvar_ciz(maske, d)
    maske[my : my + r, mx - ic // 2 : mx + ic // 2] = 0
    maske[pencere.y - h : pencere.y + h, pencere.x1 : pencere.x2] = 0
    goruntu = np.where(maske > 0, MUREKKEP, KAGIT).astype(np.uint8)
    if merdiven:
        for y in range(110, 230, 20):
            goruntu[y - ic // 2 : y + ic // 2, 420:500] = MUREKKEP
    if serbest_cizgi:
        goruntu[260 - ic // 2 : 260 + ic // 2, 100:240] = MUREKKEP

    # Kapı: açık kanat (menteşeden sağa) ve kapalı konumdan açığa çeyrek yay.
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


@dataclass
class EvPlani:
    """Arayüzün örnek planı: A4 yatay sayfa, 300 dpi (CVC-FP ölçeği)."""

    goruntu: np.ndarray
    duvar_maskesi: np.ndarray
    kapi_sayisi: int
    pencere_sayisi: int
    oda_sayisi: int


# Sekiz oda: üstte yatak odası 1, banyo, yatak odası 2 ve 3. Altta salon,
# koridor (sağa uzanan), mutfak, çalışma odası. Dış duvar 28 px, iç 14 px.
_DIS, _IC = 28, 14
_X0, _Y0, _X1, _Y1 = 550, 420, 2950, 1920
# (yön, sabit, başlangıç, bitiş, kalınlık): "y" yatay duvar y=sabit, "x" dikey.
_EV_DUVARLARI = [
    ("y", _Y0, _X0, _X1, _DIS),
    ("y", _Y1, _X0, _X1, _DIS),
    ("x", _X0, _Y0, _Y1, _DIS),
    ("x", _X1, _Y0, _Y1, _DIS),
    ("y", 1120, _X0, _X1, _IC),
    ("y", 1320, 1650, _X1, _IC),
    ("x", 1250, _Y0, 1120, _IC),
    ("x", 1650, _Y0, _Y1, _IC),
    ("x", 2300, _Y0, 1120, _IC),
    ("x", 2300, 1320, _Y1, _IC),
]
# Kapı: (duvar yönü, sabit, boşluk başı, genişlik, menteşe başta mı, açılış yönü ±1, kalınlık)
_EV_KAPILARI = [
    ("y", 1120, 900, 95, True, -1, _IC),  # yatak 1 → salon
    ("y", 1120, 1400, 90, True, -1, _IC),  # banyo → salon
    ("y", 1120, 1850, 95, True, -1, _IC),  # yatak 2 → koridor
    ("y", 1120, 2500, 95, False, -1, _IC),  # yatak 3 → koridor
    ("x", 1650, 1170, 95, True, 1, _IC),  # salon → koridor
    ("y", 1320, 1850, 95, True, 1, _IC),  # mutfak → koridor
    ("y", 1320, 2500, 95, False, 1, _IC),  # çalışma → koridor
    ("x", _X1, 1170, 105, True, -1, _DIS),  # giriş
]
# Pencere: (duvar yönü, sabit, başlangıç, bitiş)
_EV_PENCERELERI = [
    ("y", _Y0, 760, 960),
    ("y", _Y0, 1380, 1520),
    ("y", _Y0, 1880, 2080),
    ("y", _Y0, 2520, 2720),
    ("x", _X0, 650, 850),
    ("x", _X0, 1400, 1640),
    ("y", _Y1, 900, 1200),
    ("y", _Y1, 1880, 2080),
    ("y", _Y1, 2520, 2720),
    ("x", _X1, 650, 850),
    ("x", _X1, 1500, 1700),
    ("y", _Y0, 2170, 2260),
    ("x", _X0, 950, 1080),
]
_EV_ODALARI = [
    ("BEDROOM 1", 900, 770),
    ("BATH", 1450, 770),
    ("BEDROOM 2", 1975, 770),
    ("BEDROOM 3", 2625, 770),
    ("LIVING ROOM", 1100, 1520),
    ("HALL", 2300, 1220),
    ("KITCHEN", 1975, 1620),
    ("STUDY", 2625, 1620),
]


def _bant(yon: str, sabit: int, a: int, b: int, h: int) -> tuple[slice, slice]:
    """Duvar boyunca [a, b) aralığında, orta çizgiden ±h bant (satır, sütun)."""
    if yon == "y":
        return slice(sabit - h, sabit + h), slice(a, b)
    return slice(a, b), slice(sabit - h, sabit + h)


def _nokta(yon: str, sabit: int, t: float) -> tuple[float, float]:
    return (t, sabit) if yon == "y" else (sabit, t)


def ev_ciz(ince: int = 2) -> EvPlani:
    """Sekiz odalı örnek ev, cevabı bilinen sayılarla. Kapı çeyrek yayla,
    pencere boşlukta üç paralel ince çizgiyle, oda adları yazıyla."""
    yuk, gen = 2480, 3508
    maske = np.zeros((yuk, gen), np.uint8)
    for yon, sabit, a, b, kal in _EV_DUVARLARI:
        h = kal // 2
        satir, sutun = _bant(yon, sabit, a - h, b + h, h)
        maske[satir, sutun] = 255
    for yon, sabit, a, w, _, _, kal in _EV_KAPILARI:
        maske[_bant(yon, sabit, a, a + w, kal // 2 + 1)] = 0
    for yon, sabit, a, b in _EV_PENCERELERI:
        maske[_bant(yon, sabit, a, b, _DIS // 2 + 1)] = 0
    goruntu = np.where(maske > 0, MUREKKEP, KAGIT).astype(np.uint8)

    for yon, sabit, a, w, bastan, acilis, _ in _EV_KAPILARI:
        mentese = np.array(_nokta(yon, sabit, a if bastan else a + w), float)
        boy = np.array((1.0, 0.0) if yon == "y" else (0.0, 1.0)) * (1 if bastan else -1)
        normal = np.array((0.0, 1.0) if yon == "y" else (1.0, 0.0)) * acilis
        kanat = mentese + w * normal
        cv2.line(goruntu, tuple(map(round, mentese)), tuple(map(round, kanat)), MUREKKEP, ince)
        a1 = np.degrees(np.arctan2(boy[1], boy[0])) % 360
        a2 = np.degrees(np.arctan2(normal[1], normal[0])) % 360
        bas, son = sorted((a1, a2))
        if son - bas > 180:
            bas, son = son, bas + 360
        merkez = tuple(map(round, mentese))
        cv2.ellipse(goruntu, merkez, (w, w), 0, bas, son, MUREKKEP, ince)

    h = _DIS // 2
    for yon, sabit, a, b in _EV_PENCERELERI:
        for d in (-h, 0, h - 1):
            p = _nokta(yon, sabit + d, a)
            q = _nokta(yon, sabit + d, b - 1)
            cv2.line(goruntu, tuple(map(round, p)), tuple(map(round, q)), MUREKKEP, ince)

    for ad, x, y in _EV_ODALARI:
        (tw, th), _ = cv2.getTextSize(ad, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.putText(
            goruntu,
            ad,
            (x - tw // 2, y + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            MUREKKEP,
            3,
            cv2.LINE_AA,
        )
    cv2.putText(
        goruntu,
        "Sample plan (synthetic)",
        (_X0, 2200),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.4,
        MUREKKEP,
        5,
        cv2.LINE_AA,
    )
    return EvPlani(goruntu, maske, len(_EV_KAPILARI), len(_EV_PENCERELERI), len(_EV_ODALARI))
