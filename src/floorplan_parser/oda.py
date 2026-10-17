"""Oda grafı: duvar maskesi + açıklıklar → odalar (düğüm), kapılar (kenar).

1. Boşluklar kirişle kapatılır: kapı ve pencerelerin boşluğu (iki yandan 3 px
   uzatılmış; kapıda ayrıca menteşeden kapalı uca) ve verilirse sınıflanmamış
   bütün boşluk adayları (kapısız geçiş, bulunamamış pencere).
2. Dış: bina ayak izinin dışı. Ayak izi = duvar maskesinin büyük bir kareyle
   kapanmışının delikleri doldurulmuşu; kapanmamış tek bir açıklık iç mekânı
   dışarı akıtmasın diye dış bağlantıyla değil ayak iziyle tanımlanır
   (``ayak_izi_cekirdek`` 0 ise eski tanım: resmin kenarına değen bileşen).
   Ayak izinin içindeki boş alanın 4-bağlı bileşenlerinden ``min_alan``'dan
   büyükler oda.
3. Her kapının kirişinin iki yanında, kirişe dik birkaç noktadaki bileşen
   etiketinin çoğunluğu: iki farklı oda → oda-oda kenarı; bir yan dış → dış kapı.

Duvarı olmayan (çizimde mürekkebi olmayan) oda ayrımları bulunamaz: iki oda
tek bileşen olur.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import networkx as nx
import numpy as np

from floorplan_parser.aciklik import Bosluk, Kapi, Pencere

DIS = -1  # graf düğümü: bina dışı


@dataclass(frozen=True)
class OdaKural:
    min_alan: int = 5000  # px; CVC-FP I-II cevabında oda alanı %1'lik dilim 13 391
    kiris_kalinlik: int = 5
    yan_uzaklik: tuple[int, ...] = (8, 16, 24, 32, 40)
    ayak_izi_cekirdek: int = 151  # px, 0 = kapalı


VARSAYILAN_ODA_KURAL = OdaKural()


@dataclass(frozen=True)
class Oda:
    etiket: int
    alan: int
    merkez: tuple[float, float]


def kapali_maske(
    duvar_maske: np.ndarray,
    kapilar: list[Kapi],
    pencereler: list[Pencere],
    kural: OdaKural = VARSAYILAN_ODA_KURAL,
    bosluklar: Sequence[Bosluk] = (),
) -> np.ndarray:
    """Duvar maskesi + kapı, pencere ve verilen boşlukların kirişleri."""
    m = duvar_maske.copy()

    def ciz(a, b):
        # Boşluğun iki ucu arası, iki yandan 3 px uzatılmış: yarıçap taraması
        # kapalı ucu karşı pervaza birkaç piksel kala bırakabiliyor.
        a, b = np.array(a, float), np.array(b, float)
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        a, b = a - 3 * d, b + 3 * d
        cv2.line(m, tuple(map(round, a)), tuple(map(round, b)), 255, kural.kiris_kalinlik)

    for k in kapilar:
        if k.bosluk is not None:
            ciz(k.bosluk.a, k.bosluk.b)
        ciz(k.mentese, k.kapali_uc)
    for p in pencereler:
        if p.kanat is not None:
            ciz(p.kanat.mentese, p.kanat.kapali_uc)
        ciz(p.bosluk.a, p.bosluk.b)
    for b in bosluklar:
        ciz(b.a, b.b)
    return m


def ayak_izi(duvar_maske: np.ndarray, cekirdek: int) -> np.ndarray:
    """Duvar maskesinin ``cekirdek`` karesiyle kapanmışı, delikleri dolu (255)."""
    # Kenar payı: OpenCV kapamada erozyon resmin dışını dolu sayar; duvar kenara
    # çekirdeğin yarısından yakınsa kapanmış maske kenara yapışır ve dış kalmaz.
    pay = cekirdek + 1
    genis = cv2.copyMakeBorder(duvar_maske, pay, pay, pay, pay, cv2.BORDER_CONSTANT, value=0)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (cekirdek, cekirdek))
    kapali = cv2.morphologyEx(genis, cv2.MORPH_CLOSE, k)
    maske = np.zeros((kapali.shape[0] + 2, kapali.shape[1] + 2), np.uint8)
    cv2.floodFill(kapali, maske, (0, 0), 128)
    return np.where(kapali[pay:-pay, pay:-pay] == 128, 0, 255).astype(np.uint8)


def odalar(
    kapali: np.ndarray, kural: OdaKural = VARSAYILAN_ODA_KURAL, ayak: np.ndarray | None = None
) -> tuple[np.ndarray, list[Oda]]:
    """(etiket görüntüsü, odalar). Etiket: 0 duvar/kiriş, DIS dış, oda
    etiketleri 1'den başlar; küçük bileşenler 0'a katılır. ``ayak`` verilirse
    onun dışı dış, yoksa resmin kenarına değen bileşen dış."""
    bos = kapali == 0
    if ayak is not None:
        bos &= ayak > 0
    n, etiket, istat, merkez = cv2.connectedComponentsWithStats(
        bos.astype(np.uint8), connectivity=4
    )
    h, w = kapali.shape
    x, y, gen, yuk, alan = istat.T
    if ayak is None:
        dis = (x == 0) | (y == 0) | (x + gen == w) | (y + yuk == h)
    else:
        dis = np.zeros(n, bool)
    yeni = np.zeros(n, np.int32)
    liste: list[Oda] = []
    for i in range(1, n):
        if dis[i]:
            yeni[i] = DIS
        elif alan[i] >= kural.min_alan:
            yeni[i] = len(liste) + 1
            liste.append(Oda(len(liste) + 1, int(alan[i]), tuple(map(float, merkez[i]))))
    sonuc = yeni[etiket]
    if ayak is not None:
        sonuc[(ayak == 0) & (kapali == 0)] = DIS
    return sonuc, liste


def _yan_etiketi(etiket: np.ndarray, m: np.ndarray, n: np.ndarray, uzaklik) -> int | None:
    h, w = etiket.shape
    say: Counter = Counter()
    for d in uzaklik:
        x, y = (m + d * n).round().astype(int)
        if 0 <= x < w and 0 <= y < h and etiket[y, x] != 0:
            say[int(etiket[y, x])] += 1
    return say.most_common(1)[0][0] if say else None


def kapi_kenarlari(
    etiket: np.ndarray, kapilar: list[Kapi], kural: OdaKural = VARSAYILAN_ODA_KURAL
) -> list[tuple[int, int, int]]:
    """(oda, oda ya da DIS, kapı sırası). İki yan aynı ya da biri bulunamadıysa
    kenar yok."""
    kenarlar = []
    for i, k in enumerate(kapilar):
        h, j = np.array(k.mentese), np.array(k.kapali_uc)
        n = np.array(k.acik_uc) - h
        n = n / max(np.linalg.norm(n), 1e-9)
        m = (h + j) / 2
        a = _yan_etiketi(etiket, m, n, kural.yan_uzaklik)
        b = _yan_etiketi(etiket, m, -n, kural.yan_uzaklik)
        if a is None or b is None or a == b or (a == DIS and b == DIS):
            continue
        kenarlar.append((a, b, i) if b == DIS or (a != DIS and a < b) else (b, a, i))
    return kenarlar


def oda_grafi(
    duvar_maske: np.ndarray,
    kapilar: list[Kapi],
    pencereler: list[Pencere],
    kural: OdaKural = VARSAYILAN_ODA_KURAL,
    bosluklar: Sequence[Bosluk] = (),
) -> tuple[nx.MultiGraph, np.ndarray]:
    """Düğümler: odalar (``alan``, ``merkez``) ve dış (``DIS``); kenarlar:
    kapılar (``kapi`` = kapı sırası). Etiket görüntüsü de döner.
    ``bosluklar``: kapı/pencere olmayan, yine de kapatılacak boşluk adayları."""
    kapali = kapali_maske(duvar_maske, kapilar, pencereler, kural, bosluklar)
    ayak = ayak_izi(duvar_maske, kural.ayak_izi_cekirdek) if kural.ayak_izi_cekirdek else None
    etiket, liste = odalar(kapali, kural, ayak)
    g = nx.MultiGraph()
    for o in liste:
        g.add_node(o.etiket, alan=o.alan, merkez=o.merkez)
    g.add_node(DIS, dis=True)
    for a, b, i in kapi_kenarlari(etiket, kapilar, kural):
        g.add_edge(a, b, kapi=i)
    return g, etiket
