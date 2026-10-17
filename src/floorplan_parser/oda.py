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
from scipy import ndimage
from skimage.segmentation import watershed

from floorplan_parser.aciklik import Bosluk, Kapi, Pencere
from floorplan_parser.duvar import duvarlari_bul

DIS = -1  # graf düğümü: bina dışı


@dataclass(frozen=True)
class OdaKural:
    min_alan: int = 5000  # px; CVC-FP I-II cevabında oda alanı %1'lik dilim 13 391
    kiris_kalinlik: int = 5
    yan_uzaklik: tuple[int, ...] = (8, 16, 24, 32, 40)
    # Ayak izi kapama çekirdeği (px, 0 = kapalı); geliştirmede 301-601 plato (§10).
    ayak_izi_cekirdek: int = 301
    # Ayak izi içindeki bölgenin çevresinde duvar/kiriş payı bunun altındaysa
    # dış sayılır (büyük çekirdeğin içeri aldığı sundurma, teras). 0 = kapalı.
    min_duvar_payi: float = 0.8
    # Havza bölütleme (``havza_bolutle``): çekirdek eşiği px (0 = bağlı bileşen)
    # ve geniş geçitte birleşme oranı.
    havza_cekirdek: float = 45.0  # geliştirmede 35-55 plato (§11)
    havza_birlesme: float = 0.8


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


def ince_sinirlar(
    ikili: np.ndarray,
    duvar_maske: np.ndarray,
    min_uzunluk: int = 80,
    max_kalinlik: float = 6.0,
    uc_tol: float = 6.0,
) -> np.ndarray:
    """Yalnız oda bölme için sınır çizgileri (255): duvar olmayan mürekkepte,
    ince, uzun, düz ve iki ucu da duvara değen parçalar. İnce çizgiyle
    çizilmiş bölme duvarları yakalamak için; duvar sayılmaz."""
    genis = cv2.dilate(duvar_maske, np.ones((5, 5), np.uint8))
    aday = cv2.bitwise_and(ikili, cv2.bitwise_not(genis))
    uzak = cv2.distanceTransform((duvar_maske == 0).astype(np.uint8), cv2.DIST_L2, 3)
    h, w = duvar_maske.shape
    m = np.zeros_like(duvar_maske)
    for p in duvarlari_bul(aday, max_bosluk=10, min_uzunluk=min_uzunluk):
        if p.kalinlik > max_kalinlik:
            continue
        k = np.array(p.kapsam or (p.x1, p.y1, p.x2, p.y2))
        a, b = k[:2], k[2:]
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        a, b = a - (uc_tol + 2) * d, b + (uc_tol + 2) * d
        degiyor = all(
            uzak[min(max(round(q[1]), 0), h - 1), min(max(round(q[0]), 0), w - 1)] <= 1
            for q in (a, b)
        )
        if degiyor:
            cv2.line(m, tuple(map(round, a)), tuple(map(round, b)), 255, 3)
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


def havza_bolutle(
    bos: np.ndarray, cekirdek: float, birlesme: float
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Boş alanı uzaklık dönüşümü + watershed ile böler.

    Çekirdek: duvara uzaklığı ≥ ``cekirdek`` olan piksellerin bileşenleri (oda
    ortaları); çekirdeği olmayan küçük bileşen kendi işaretçisi. Havzalar
    dar geçitlerde (kapı, kapanmamış açıklık) buluşur. İki komşu havza,
    aralarındaki eyerin yüksekliği küçük tepenin ``birlesme`` katından
    büyükse (geçit geniş: L biçimli oda, girinti) birleşir.

    Dönüş ``cv2.connectedComponentsWithStats`` ile aynı biçimde:
    (n, etiket, istat [x, y, gen, yuk, alan], merkez).
    """
    dt = cv2.distanceTransform(bos.astype(np.uint8), cv2.DIST_L2, 5)
    n_c, isaret = cv2.connectedComponents((dt >= cekirdek).astype(np.uint8), connectivity=8)
    n_b, bilesen = cv2.connectedComponents(bos.astype(np.uint8), connectivity=4)
    cekirdekli = np.zeros(n_b, bool)
    cekirdekli[np.unique(bilesen[dt >= cekirdek])] = True
    cekirdekli[0] = True
    ek = np.zeros(n_b, np.int32)
    eksik = np.flatnonzero(~cekirdekli)
    ek[eksik] = n_c + np.arange(len(eksik))
    isaret = isaret.astype(np.int32)
    ek_harita = ek[bilesen]
    isaret[ek_harita > 0] = ek_harita[ek_harita > 0]
    lab = watershed(-dt, isaret, mask=bos)

    # Komşu havza çiftleri ve aralarındaki eyer (sınırdaki en yüksek uzaklık).
    m = int(lab.max()) + 1
    tepe = np.zeros(m)
    np.maximum.at(tepe, lab.ravel(), dt.ravel())
    ciftler, degerler = [], []
    for a, b, da, db in (
        (lab[:, :-1], lab[:, 1:], dt[:, :-1], dt[:, 1:]),
        (lab[:-1, :], lab[1:, :], dt[:-1, :], dt[1:, :]),
    ):
        s = (a != b) & (a > 0) & (b > 0)
        ciftler.append(np.stack([np.minimum(a[s], b[s]), np.maximum(a[s], b[s])], 1))
        degerler.append(np.minimum(da[s], db[s]))
    ata = np.arange(m)

    def kok(i):
        while ata[i] != i:
            ata[i] = ata[ata[i]]
            i = ata[i]
        return i

    cift = np.vstack(ciftler) if ciftler else np.zeros((0, 2), int)
    deger = np.concatenate(degerler) if degerler else np.zeros(0)
    if len(cift):
        anahtar = cift[:, 0].astype(np.int64) * m + cift[:, 1]
        sira = np.lexsort((deger, anahtar))
        anahtar, deger, cift = anahtar[sira], deger[sira], cift[sira]
        son = np.r_[anahtar[1:] != anahtar[:-1], True]
        for (a, b), eyer in zip(cift[son], deger[son]):
            if eyer >= birlesme * min(tepe[a], tepe[b]):
                ra, rb = kok(a), kok(b)
                if ra != rb:
                    ata[rb] = ra
    kokler = np.array([kok(i) for i in range(m)])
    benzersiz, yeni = np.unique(kokler, return_inverse=True)
    etiket = yeni[lab].astype(np.int32)  # 0 kökü 0'a gider (lab 0 → kök 0)
    n = len(benzersiz)
    alan = np.bincount(etiket.ravel(), minlength=n)
    ys, xs = np.indices(etiket.shape)
    sx = np.bincount(etiket.ravel(), xs.ravel(), minlength=n)
    sy = np.bincount(etiket.ravel(), ys.ravel(), minlength=n)
    merkez = np.stack([sx / np.maximum(alan, 1), sy / np.maximum(alan, 1)], 1)
    istat = np.zeros((n, 5), np.int64)
    istat[:, 4] = alan
    for i, dilim in enumerate(ndimage.find_objects(etiket), start=1):
        if dilim is not None:
            istat[i, :4] = [
                dilim[1].start,
                dilim[0].start,
                dilim[1].stop - dilim[1].start,
                dilim[0].stop - dilim[0].start,
            ]
    return n, etiket, istat, merkez


def odalar(
    kapali: np.ndarray, kural: OdaKural = VARSAYILAN_ODA_KURAL, ayak: np.ndarray | None = None
) -> tuple[np.ndarray, list[Oda]]:
    """(etiket görüntüsü, odalar). Etiket: 0 duvar/kiriş, DIS dış, oda
    etiketleri 1'den başlar; küçük bileşenler 0'a katılır. ``ayak`` verilirse
    onun dışı dış, yoksa resmin kenarına değen bileşen dış."""
    bos = kapali == 0
    if ayak is not None:
        bos &= ayak > 0
    if kural.havza_cekirdek > 0:
        n, etiket, istat, merkez = havza_bolutle(bos, kural.havza_cekirdek, kural.havza_birlesme)
    else:
        n, etiket, istat, merkez = cv2.connectedComponentsWithStats(
            bos.astype(np.uint8), connectivity=4
        )
    h, w = kapali.shape
    x, y, gen, yuk, alan = istat.T
    if ayak is None:
        dis = (x == 0) | (y == 0) | (x + gen == w) | (y + yuk == h)
    else:
        dis = np.zeros(n, bool)
        if kural.min_duvar_payi > 0:
            # Bileşenin dış sınırındaki pikseller: duvar/kirişe mi, ayak izinin
            # dışına mı komşu. Payı düşük bölge içbükey dış alandır.
            k3 = np.ones((3, 3), np.uint8)
            disari = ((ayak == 0) & (kapali == 0)).astype(np.uint8)
            duvara = cv2.dilate((kapali > 0).astype(np.uint8), k3) & bos
            disa = cv2.dilate(disari, k3) & bos
            n_duvar = np.bincount(etiket[duvara > 0], minlength=n)
            n_dis = np.bincount(etiket[disa > 0], minlength=n)
            pay = n_duvar / np.maximum(n_duvar + n_dis, 1)
            dis = (n_dis > 0) & (pay < kural.min_duvar_payi)
            dis[0] = False
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
    ayak_bosluklari: Sequence[Bosluk] | None = None,
    sinir_maske: np.ndarray | None = None,
) -> tuple[nx.MultiGraph, np.ndarray]:
    """Düğümler: odalar (``alan``, ``merkez``) ve dış (``DIS``); kenarlar:
    kapılar (``kapi`` = kapı sırası). Etiket görüntüsü de döner.

    - ``bosluklar``: kapı/pencere olmayan, oda bölmede yine de kapatılacak
      boşluk adayları.
    - ``ayak_bosluklari``: verilirse ayak izi, duvar maskesinden değil bu
      boşluklar ve kapı/pencere kirişleriyle kapanmış maskeden çıkarılır
      (dış duvardaki açıklıklar sınırı delmesin; oda bölmeyi etkilemez).
    - ``sinir_maske``: yalnız oda bölmede sınır sayılan ek çizgiler (ör. ince
      çizgili bölme duvarlar); duvar sayılmaz.
    """
    kapali = kapali_maske(duvar_maske, kapilar, pencereler, kural, bosluklar)
    if sinir_maske is not None:
        kapali = np.maximum(kapali, sinir_maske)
    ayak = None
    if kural.ayak_izi_cekirdek:
        kaynak = duvar_maske
        if ayak_bosluklari is not None:
            kaynak = kapali_maske(duvar_maske, kapilar, pencereler, kural, ayak_bosluklari)
        ayak = ayak_izi(kaynak, kural.ayak_izi_cekirdek)
    etiket, liste = odalar(kapali, kural, ayak)
    g = nx.MultiGraph()
    for o in liste:
        g.add_node(o.etiket, alan=o.alan, merkez=o.merkez)
    g.add_node(DIS, dis=True)
    for a, b, i in kapi_kenarlari(etiket, kapilar, kural):
        g.add_edge(a, b, kapi=i)
    return g, etiket
