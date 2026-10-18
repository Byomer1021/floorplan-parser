"""Görselleştirme: plan sonucunun katmanları ve graf dışa aktarımı.

Demo, rapor ve sunum şekilleri bunu kullanır. Teslimatlar İngilizce olduğu
için şekil üstündeki metinler ve JSON alan adları İngilizce.
"""

import cv2
import numpy as np

from floorplan_parser.akis import PlanSonucu
from floorplan_parser.cvcfp import Cevap
from floorplan_parser.oda import DIS

# BGR
DUVAR = (180, 90, 20)
KAPI = (40, 40, 220)
PENCERE = (40, 160, 40)
GRAF = (20, 20, 20)
DIS_RENK = (110, 110, 110)


def oda_renkleri(n: int, tohum: int = 3) -> np.ndarray:
    """``n`` oda için açık (pastel) renkler; 0 ve DIS boyanmaz."""
    rng = np.random.default_rng(tohum)
    return rng.integers(150, 250, (max(n, 1) + 1, 3)).astype(np.uint8)


def taban(gri: np.ndarray, soluk: float = 0.55) -> np.ndarray:
    """Gri planın soluklaştırılmış renkli kopyası (katmanlar öne çıksın)."""
    renk = cv2.cvtColor(gri, cv2.COLOR_GRAY2BGR).astype(np.float32)
    return (255 - (255 - renk) * soluk).astype(np.uint8)


def _boya(goruntu: np.ndarray, maske: np.ndarray, renk, alfa: float) -> None:
    m = maske > 0
    goruntu[m] = ((1 - alfa) * goruntu[m] + alfa * np.array(renk)).astype(np.uint8)


def _poligon(goruntu, p: np.ndarray, renk, kalinlik: int, dolgu: float = 0.0) -> None:
    nokta = np.round(p).astype(np.int32)
    if dolgu > 0:
        m = np.zeros(goruntu.shape[:2], np.uint8)
        cv2.fillPoly(m, [nokta], 255)
        _boya(goruntu, m, renk, dolgu)
    cv2.polylines(goruntu, [nokta], True, renk, kalinlik, cv2.LINE_AA)


def kapi_ortasi(k) -> tuple[float, float]:
    return ((k.mentese[0] + k.kapali_uc[0]) / 2, (k.mentese[1] + k.kapali_uc[1]) / 2)


def katmanlar(
    gri: np.ndarray,
    s: PlanSonucu,
    oda: bool = True,
    duvar: bool = True,
    acikliklar: bool = True,
    graf: bool = True,
    kalinlik: int = 3,
) -> np.ndarray:
    """Sonucun katmanlı çizimi (BGR): odalar pastel, duvar mavi, kapı
    kırmızı, pencere yeşil, graf siyah (oda merkezi → kapı → oda merkezi)."""
    g = taban(gri)
    if oda:
        renk = oda_renkleri(int(s.oda_etiket.max()))
        m = s.oda_etiket > 0
        g[m] = (0.45 * g[m] + 0.55 * renk[s.oda_etiket[m]]).astype(np.uint8)
    if duvar:
        _boya(g, s.duvar_maske, DUVAR, 0.75)
    if acikliklar:
        for p in s.pencereler:
            _poligon(g, p.poligon(), PENCERE, kalinlik, 0.25)
        for k in s.kapilar:
            _poligon(g, k.poligon(), KAPI, kalinlik, 0.25)
    if graf:
        merkez = {u: d["merkez"] for u, d in s.graf.nodes(data=True) if u != DIS}
        for u, v, d in s.graf.edges(data=True):
            orta = kapi_ortasi(s.kapilar[d["kapi"]])
            for w in (u, v):
                if w in merkez:
                    a = tuple(map(round, merkez[w]))
                    cv2.line(g, a, tuple(map(round, orta)), GRAF, kalinlik, cv2.LINE_AA)
            cv2.circle(g, tuple(map(round, orta)), 3 * kalinlik, GRAF, -1, cv2.LINE_AA)
            if DIS in (u, v):
                cv2.circle(g, tuple(map(round, orta)), 5 * kalinlik, DIS_RENK, kalinlik)
        for w, c in merkez.items():
            cv2.circle(g, tuple(map(round, c)), 6 * kalinlik, GRAF, -1, cv2.LINE_AA)
            cv2.circle(g, tuple(map(round, c)), 4 * kalinlik, (255, 255, 255), -1, cv2.LINE_AA)
    return g


def cevap_katmanlari(gri: np.ndarray, c: Cevap, kalinlik: int = 3) -> np.ndarray:
    """CVC-FP cevabı aynı renklerle: oda pastel, duvar mavi, kapı kırmızı,
    pencere yeşil."""
    g = taban(gri)
    odalar = c.sinif("Room")
    renk = oda_renkleri(len(odalar))
    for i, n in enumerate(odalar, start=1):
        m = np.zeros(gri.shape, np.uint8)
        cv2.fillPoly(m, [np.round(n.noktalar).astype(np.int32)], 255)
        _boya(g, m, renk[i].tolist(), 0.55)
    for n in c.sinif("Wall"):
        m = np.zeros(gri.shape, np.uint8)
        cv2.fillPoly(m, [np.round(n.noktalar).astype(np.int32)], 255)
        _boya(g, m, DUVAR, 0.75)
    for n in c.sinif("Window"):
        _poligon(g, n.noktalar, PENCERE, kalinlik, 0.25)
    for n in c.sinif("Door"):
        _poligon(g, n.noktalar, KAPI, kalinlik, 0.25)
    return g


def lejant(
    goruntu: np.ndarray,
    ogeler: list[tuple[str, tuple]],
    olcek: float = 1.0,
    konum: tuple[int, int] = (10, 10),
) -> None:
    """``konum`` (sol üst) köşesine renk kutulu lejant (İngilizce etiketler)."""
    x0, y0 = konum
    yazi = 0.9 * olcek
    satir = int(40 * olcek)
    gen = int(260 * olcek)
    alt = y0 + 10 + satir * len(ogeler)
    cv2.rectangle(goruntu, (x0, y0), (x0 + gen, alt), (255, 255, 255), -1)
    cv2.rectangle(goruntu, (x0, y0), (x0 + gen, alt), (0, 0, 0), 1)
    for i, (ad, renk) in enumerate(ogeler):
        y = y0 + 10 + i * satir
        cv2.rectangle(
            goruntu, (x0 + 10, y + 5), (x0 + 10 + int(28 * olcek), y + satir - 10), renk, -1
        )
        cv2.putText(
            goruntu,
            ad,
            (x0 + 20 + int(36 * olcek), y + satir - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            yazi,
            (0, 0, 0),
            max(1, round(2 * olcek)),
            cv2.LINE_AA,
        )


VARSAYILAN_LEJANT = [
    ("wall", DUVAR),
    ("door", KAPI),
    ("window", PENCERE),
    ("room graph", GRAF),
]


def graf_sozluk(s: PlanSonucu) -> dict:
    """Sonucun JSON'a yazılabilir hâli (piksel koordinatları)."""

    def nokta(p):
        return [round(float(p[0]), 1), round(float(p[1]), 1)]

    return {
        "rooms": [
            {"id": int(u), "area_px": int(d["alan"]), "centroid": nokta(d["merkez"])}
            for u, d in s.graf.nodes(data=True)
            if u != DIS
        ],
        "doors": [
            {
                "id": i,
                "hinge": nokta(k.mentese),
                "closed_end": nokta(k.kapali_uc),
                "open_end": nokta(k.acik_uc),
                "radius_px": round(k.yaricap, 1),
                "score": round(k.skor, 3),
            }
            for i, k in enumerate(s.kapilar)
        ],
        "windows": [
            {
                "id": i,
                "type": "double_casement"
                if p.kanatlar
                else "casement"
                if p.kanat is not None
                else "parallel_lines",
                "polygon": [nokta(q) for q in p.poligon()],
            }
            for i, p in enumerate(s.pencereler)
        ],
        "edges": [
            {
                "door": int(d["kapi"]),
                "rooms": ["outside" if w == DIS else int(w) for w in (u, v)],
            }
            for u, v, d in s.graf.edges(data=True)
        ],
        "walls": [
            {"centerline": [nokta((p.x1, p.y1)), nokta((p.x2, p.y2))], "thickness_px": p.kalinlik}
            for p in s.kalin_parcalar + s.ince_parcalar
        ],
    }
