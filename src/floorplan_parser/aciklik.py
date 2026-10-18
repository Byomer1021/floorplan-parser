"""Açıklık: duvardaki boşluklar, kapı (yay) ve pencere (paralel ince çizgiler).

Akış:

1. Boşluk adayı: her duvar parçasının iki ucundan dışarı ışın. Işın
   ``min_genislik``-``max_genislik`` arasında başka bir duvara çarparsa arada
   boşluk var: aynı doğrudaki duvarın ucu ya da dik duvarın yüzü. Aynı boşluk
   karşı uçtan da bulunur; ikisi birleşir.
2. Kapı: menteşe boşluğun bir ucunda, kanat bir yana açılır (4 varsayım).
   Her varsayımda çeyrek yayın geçmesi gereken noktalarda ince mürekkep
   (duvarlar çıkarılmış) aranır; yarıçap ve menteşenin duvar yüzüne göre yeri
   küçük bir aralıkta taranır. Bilinen geometrili şablon eşleme.
3. Pencere: boşluğun içinde duvar boyunca uzanan en az iki paralel ince çizgi.

Bu modül duvar raster'ı ve parçaları ister (``duvar.iki_olcekli_duvarlar``).
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from floorplan_parser.duvar import Parca, aci_farki, dortgen_ciz


@dataclass(frozen=True)
class AciklikKural:
    min_genislik: float = 40.0
    max_genislik: float = 200.0
    yay_esik: float = 0.8  # yay örneklerinin mürekkebe yakın olma payı
    # Bu yarıçaptan küçük yay kanatlı pencere (CVC-FP I-II: kapı yarıçapı %5'lik
    # dilim 89 px, kanatlı pencere medyan 73 px; olcumler §8). Ölçeğe bağlı.
    kapi_min_yaricap: float = 75.0
    # Yarı açık çizilmiş kapı: yay + kanat, yalnız kapı boyunda ve bu eşikle.
    # Kapalı (> 1): geliştirmede 0.85-0.95 F1'i düşürdü (olcumler §8).
    kismi_esik: float = 1.01
    kismi_acilar: tuple[float, ...] = (30.0, 45.0, 60.0)
    kismi_pencerede: bool = True  # False: pencere çizgisi taşıyan boşlukta kısmi yok
    kismi_kanat_yan: float = 0.3  # kanadın 7 px yanında mürekkep payı en çok
    yay_tol: float = 3.0  # örnek noktanın mürekkebe en çok uzaklığı (px)
    # Kapı yayı yarıçapı / boşluk genişliği taraması. Adım ~2 × yay_tol'dan
    # büyükse doğru yarıçap atlanabiliyor (olcumler §15).
    yaricap_olcekleri: tuple[float, ...] = (0.85, 0.92, 1.0, 1.06)
    # Kavşaktaki uçtan dik duvar yüzüne boşluklar da yay adayı (T-kavşağı kapısı).
    # Geliştirmede kapı +3 doğru, fazla aynı (olcumler §15).
    kavsak_aday: bool = True
    # Ek ölçekler yalnız yarıçap ≥ kapi_min_yaricap (kapı boyu) iken denenir:
    # kanatlı pencerede ince tarama pencere çizgisini yaya çeviriyor.
    # Geliştirmede kapı F1 0.854 → 0.881, pencere aynı (olcumler §15).
    kapi_ek_olcekleri: tuple[float, ...] = (0.83, 0.87, 0.91, 0.95, 0.99, 1.03, 1.07)
    pencere_kapsama: float = 0.8  # paralel çizginin boşluk boyunca doluluğu
    pencere_min_cizgi: int = 2
    kose_aci_tol: float = 25.0  # raster köşesi açısı 90 ± bu kadar derece
    # Çift kanatlı pencere: boşluğun iki ucunda menteşe, yarım genişlikte iki yay.
    cift_kanat: bool = True
    # Tek kanatlı pencerenin poligonuna boşluk dikdörtgeni de katılır.
    kanat_dikdortgen: bool = True
    # Geniş boşluklarda (genis_bosluklar, 150-450 px) da pencere aranır.
    # Üçü birlikte geliştirmede pencere F1 0.671 → 0.815 (olcumler §13).
    genis_pencere: bool = True
    # Pencere için geniş boşlukta karşı duvar kısa, parçasız bir ayak olabilir.
    # Geliştirmede pencere F1 0.815 → 0.839 (olcumler §13).
    genis_kisa_duvar: bool = True


VARSAYILAN_ACIKLIK_KURAL = AciklikKural()


@dataclass(frozen=True)
class Bosluk:
    """Duvardaki boşluk: ``a`` duvar ucu (orta çizgi), ``b`` karşı duvarın
    yüzündeki çarpma noktası; ``kalinlik`` ``a``'daki duvarın kalınlığı."""

    a: tuple[float, float]
    b: tuple[float, float]
    kalinlik: float

    @property
    def genislik(self) -> float:
        return math.dist(self.a, self.b)


@dataclass(frozen=True)
class Kapi:
    mentese: tuple[float, float]
    kapali_uc: tuple[float, float]  # kapalıyken kanadın ucu (karşı pervaz)
    acik_uc: tuple[float, float]  # 90° açıkken kanadın ucu: açılma yanı
    skor: float
    bosluk: Bosluk | None = None
    aci: float = 90.0  # çizimdeki açılma açısı (yarı açık çizilmiş kapı < 90)

    @property
    def yaricap(self) -> float:
        return math.dist(self.mentese, self.kapali_uc)

    def poligon(self, n: int = 16) -> np.ndarray:
        """Açılma alanı: menteşe + kapalı uçtan ``aci`` kadar yay."""
        h = np.array(self.mentese)
        u = np.array(self.kapali_uc) - h
        v = np.array(self.acik_uc) - h
        t = np.linspace(0, math.radians(self.aci), n)[:, None]
        return np.vstack([h, h + np.cos(t) * u + np.sin(t) * v])


def birlesim_poligonu(poligonlar: list[np.ndarray], olcek: int = 4) -> np.ndarray:
    """Poligonların birleşiminin dış hattı (1/``olcek`` px raster üstünden);
    birleşim parçalıysa en büyük parça."""
    hepsi = np.vstack(poligonlar)
    x0, y0 = np.floor(hepsi.min(0)) - 2
    x1, y1 = np.ceil(hepsi.max(0)) + 2
    m = np.zeros((int((y1 - y0) * olcek) + 1, int((x1 - x0) * olcek) + 1), np.uint8)
    for p in poligonlar:  # tek tek: fillPoly çoklu poligonda tek-çift kuralı uygular
        cv2.fillPoly(m, [np.round((p - [x0, y0]) * olcek).astype(np.int32)], 1)
    hatlar, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h = cv2.approxPolyDP(max(hatlar, key=cv2.contourArea), 1.0, True)[:, 0, :]
    return h / olcek + np.array([x0, y0])


@dataclass(frozen=True)
class Pencere:
    """Boşlukta paralel çizgiler (``cizgi``), kanatlı pencere (``kanat``: küçük
    yarıçaplı yay) ya da çift kanatlı pencere (``kanatlar``: iki uçta yarım
    genişlikte birer yay).

    Poligon: kanatsızda boşluk dikdörtgeni; tek kanatta açılma alanı, ya da
    ``dikdortgen`` ise dikdörtgenle birleşimi; çift kanatta her zaman
    dikdörtgen + iki açılma alanı (cevaptaki gibi, olcumler §13)."""

    bosluk: Bosluk
    cizgi: int = 0
    kanat: Kapi | None = None
    kanatlar: tuple[Kapi, ...] = ()
    dikdortgen: bool = False

    def dikdortgen_poligon(self) -> np.ndarray:
        a, b = np.array(self.bosluk.a), np.array(self.bosluk.b)
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n = np.array([-d[1], d[0]]) * self.bosluk.kalinlik / 2
        return np.array([a + n, b + n, b - n, a - n])

    def poligon(self) -> np.ndarray:
        if self.kanatlar:
            return birlesim_poligonu(
                [k.poligon() for k in self.kanatlar] + [self.dikdortgen_poligon()]
            )
        if self.kanat is not None:
            if self.dikdortgen and self.bosluk.kalinlik > 0:
                return birlesim_poligonu([self.kanat.poligon(), self.dikdortgen_poligon()])
            return self.kanat.poligon()
        return self.dikdortgen_poligon()


def duvar_etiketi(parcalar: list[Parca], duvar_maske: np.ndarray) -> np.ndarray:
    """Işın için: duvar maskesi -1, parça i'nin dikdörtgeni i+1."""
    etiket = np.zeros(duvar_maske.shape, np.int32)
    etiket[duvar_maske > 0] = -1
    for i, p in enumerate(parcalar):
        dortgen_ciz(etiket, p, i + 1)
    return etiket


def bosluk_isini(
    duvar_maske: np.ndarray, uc: np.ndarray, yon: np.ndarray, kural: AciklikKural, ilk_cikis: float
) -> tuple[float, float] | None:
    """Uçtan 2 px içeriden başlayıp ``yon``'de yürür: duvardan çıkış ve
    yeniden duvara giriş uzaklığı (başlangıçtan). Duvardan ``ilk_cikis``
    içinde çıkmazsa ya da boş aralık genişlik sınırları dışındaysa None.
    Kavşaktaki uçta ışın komşu duvarın içinden geçip devam eder."""
    h, w = duvar_maske.shape
    bas = uc - 2 * yon
    t = np.arange(0, ilk_cikis + kural.max_genislik + 2, 1.0)
    q = bas + t[:, None] * yon
    x = np.floor(q[:, 0] + 0.5).astype(np.int64)
    y = np.floor(q[:, 1] + 0.5).astype(np.int64)
    icinde = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    duvar = np.zeros(len(t), bool)
    duvar[icinde] = duvar_maske[y[icinde], x[icinde]] > 0
    bos = np.flatnonzero(~duvar)
    if len(bos) == 0 or t[bos[0]] > ilk_cikis:
        return None
    i0 = bos[0]
    geri = np.flatnonzero(duvar[i0:])
    if len(geri) == 0:
        return None
    i1 = i0 + geri[0]
    genislik = t[i1] - t[i0]
    if not (kural.min_genislik <= genislik <= kural.max_genislik) or not icinde[i1]:
        return None
    return float(t[i0] - 2), float(t[i1] - 2)


def yan_durumu(
    duvar_maske: np.ndarray,
    ga: np.ndarray,
    yon: np.ndarray,
    kalinlik: float,
    s: np.ndarray,
    sinir: float = 220.0,
) -> str:
    """Duvar ucundan yana (``s`` yönünde) bakış: uç yüzünün (``ga``) 2-9 px
    gerisinde, duvar kalınlığının dışından başlayıp ilk duvara uzaklık.
    ``bitisik`` (≤ 3 px), ``yakin`` (3-40), ``aralikli`` (40-``sinir``: kapı
    boşluğu sonra duvar), ``acik``."""
    t = np.arange(0, sinir + 1, 1.0)
    en_yakin = None
    for geri in (2.0, 5.0, 9.0):
        bas = ga - geri * yon + (kalinlik / 2 + 1) * s
        vur = np.flatnonzero(_isabet(duvar_maske, bas + t[:, None] * s))
        if len(vur):
            en_yakin = t[vur[0]] if en_yakin is None else min(en_yakin, t[vur[0]])
    if en_yakin is None:
        return "acik"
    return "bitisik" if en_yakin <= 3 else "yakin" if en_yakin < 40 else "aralikli"


def bosluklar(
    parcalar: list[Parca],
    duvar_maske: np.ndarray,
    kural: AciklikKural = VARSAYILAN_ACIKLIK_KURAL,
    yalniz_hizali: bool = False,
    yan_suzgec: str = "",
    kavsak_serbest: bool = False,
) -> list[Bosluk]:
    """Duvar uçlarından boşluk adayları.

    Serbest uç (kavşağa çekilmemiş): karşıdaki herhangi bir duvar. Kavşaktaki
    uç: ışın komşu duvarı geçer; karşıdaki duvar bu parçayla aynı doğrultuda
    olmalı (kapı köşenin dibinde, duvar kapıdan sonra sürüyor). Böylece
    T-kavşağından odanın karşısına boşluk açılmaz.

    ``yalniz_hizali``: serbest uçta da karşıdaki duvar hizalı (≤ 10°, orta
    çizgisi ≤ yarım kalınlık + 4 px) ya da parçasız olmalı. Oda bölme için:
    dik duvarın yüzüne çarpan ışın çoğunlukla odanın ortasından geçiyor
    (olcumler §14). Kapı adaylarında kullanılmaz (köşedeki kapı dik yüze açılır).

    ``yan_suzgec``: ışının çıktığı uç kesik bir duvar hattının üstündeyse
    (iki yanda da duvar: ``"aa"`` ikisi de aralıklı, ``"duvarli"`` aralıklı ya
    da bitişik) boşluk atlanır; T-kavşağında iki yanı kapılı koridor duvarı
    gibi. Yalnız oda bölme için (olcumler §14).

    ``kavsak_serbest``: kavşaktaki uçta da karşıdaki herhangi bir duvar kabul
    (T-kavşağının dibindeki kapı, iki dik yüz arasında). Yalnız yay testi
    adayı olarak (olcumler §15); oda bölmede ve pencere testinde kullanılmaz.
    """
    etiket = duvar_etiketi(parcalar, duvar_maske)
    h, w = duvar_maske.shape
    en_kalin = max((p.kalinlik for p in parcalar), default=0.0)
    bulunan: list[Bosluk] = []
    for i, p in enumerate(parcalar):
        k = np.array(p.kapsam or (p.x1, p.y1, p.x2, p.y2))
        a, b = k[:2], k[2:]
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        graf = (np.array([p.x1, p.y1]), np.array([p.x2, p.y2]))
        for uc, yon, g_uc in ((a, -d, graf[0]), (b, d, graf[1])):
            kavsak = np.linalg.norm(g_uc - uc) > 1.0
            r = bosluk_isini(duvar_maske, uc, yon, kural, en_kalin + 4 if kavsak else 4)
            if r is None:
                continue
            ga, gb = uc + r[0] * yon, uc + r[1] * yon
            if kavsak or yalniz_hizali:
                xi = min(max(int(np.floor(gb[0] + 0.5 + yon[0])), 0), w - 1)
                yi = min(max(int(np.floor(gb[1] + 0.5 + yon[1])), 0), h - 1)
                e = etiket[yi, xi]
            if kavsak and not kavsak_serbest:
                if e <= 0 or e == i + 1 or aci_farki(parcalar[e - 1].aci, p.aci) > math.radians(10):
                    continue
            elif kavsak and e == i + 1:
                continue
            elif yalniz_hizali and e > 0:
                q = parcalar[e - 1]
                q_orta = (np.array([q.x1, q.y1]) + np.array([q.x2, q.y2])) / 2
                n = np.array([-d[1], d[0]])
                if (
                    e == i + 1
                    or aci_farki(q.aci, p.aci) > math.radians(10)
                    or abs((q_orta - a) @ n) > max(p.kalinlik, q.kalinlik) / 2 + 4
                ):
                    continue
            if yan_suzgec:
                n = np.array([-d[1], d[0]])
                yanlar = {yan_durumu(duvar_maske, ga, yon, p.kalinlik, s) for s in (n, -n)}
                kabul = {"aralikli"} if yan_suzgec == "aa" else {"aralikli", "bitisik"}
                if yanlar <= kabul:
                    continue
            yeni = Bosluk(tuple(map(float, ga)), tuple(map(float, gb)), p.kalinlik)
            # Aynı boşluk karşı uçtan da bulunur: orta noktalar ve genişlik yakın.
            orta = (ga + gb) / 2
            if any(
                np.linalg.norm(orta - (np.array(q.a) + np.array(q.b)) / 2) < 0.25 * q.genislik
                and abs(q.genislik - yeni.genislik) < 0.25 * q.genislik
                for q in bulunan
            ):
                continue
            bulunan.append(yeni)
    return bulunan


def _hizali_kisa_duvar(
    duvar_maske: np.ndarray, nokta: np.ndarray, yon: np.ndarray, kalinlik: float
) -> bool:
    """Işının çarptığı duvar, ışının geldiği duvarla hizalı kısa bir parça mı
    (iki pencere arasındaki ayak gibi): çarpma noktasının 2 px içinde enine
    duvar genişliği ``kalinlik``'a yakın ve ışına ortalı. Dik duvar enine
    boyunca uzandığı için geçemez."""
    n = np.array([-yon[1], yon[0]])
    q = nokta + 2 * yon
    sinir = 2 * kalinlik + 10
    t = np.arange(0, sinir + 1, 1.0)
    uzanim = []
    for s in (n, -n):
        ic = _isabet(duvar_maske, q + t[:, None] * s)
        disari = np.flatnonzero(~ic)
        if len(disari) == 0:
            return False
        uzanim.append(t[disari[0]])
    genislik = uzanim[0] + uzanim[1]
    kayma = abs(uzanim[0] - uzanim[1]) / 2
    return 0.5 * kalinlik <= genislik <= 1.5 * kalinlik + 6 and kayma <= kalinlik / 2 + 4


def genis_bosluklar(
    parcalar: list[Parca],
    duvar_maske: np.ndarray,
    min_genislik: float = 150.0,
    max_genislik: float = 450.0,
    kisa_duvar: bool = False,
) -> list[Bosluk]:
    """Kapı boyundan geniş açıklıklar (garaj kapısı, porte-fenêtre): yalnız
    karşıdaki duvar aynı doğrultuda sürüyorsa. Işın odanın karşı duvarına
    çarpmasın diye karşı duvar bu parçayla ≤ 10° ve orta çizgisi ≤ yarım
    kalınlık + 4 px uzakta olmalı. Oda bölme ve bina sınırı için.

    ``kisa_duvar``: karşıdaki duvar parçaya dönüşmemiş ya da öyle görünmüyorsa
    (kısa ayak, köşe çıkıntısı) ``_hizali_kisa_duvar`` testi de kabul edilir."""
    kural = AciklikKural(min_genislik=min_genislik, max_genislik=max_genislik)
    etiket = duvar_etiketi(parcalar, duvar_maske)
    h, w = duvar_maske.shape
    en_kalin = max((p.kalinlik for p in parcalar), default=0.0)
    bulunan: list[Bosluk] = []
    for i, p in enumerate(parcalar):
        k = np.array(p.kapsam or (p.x1, p.y1, p.x2, p.y2))
        a, b = k[:2], k[2:]
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n = np.array([-d[1], d[0]])
        graf = (np.array([p.x1, p.y1]), np.array([p.x2, p.y2]))
        for uc, yon, g_uc in ((a, -d, graf[0]), (b, d, graf[1])):
            kavsak = np.linalg.norm(g_uc - uc) > 1.0
            r = bosluk_isini(duvar_maske, uc, yon, kural, en_kalin + 4 if kavsak else 4)
            if r is None:
                continue
            ga, gb = uc + r[0] * yon, uc + r[1] * yon
            xi = min(max(int(np.floor(gb[0] + 0.5 + yon[0])), 0), w - 1)
            yi = min(max(int(np.floor(gb[1] + 0.5 + yon[1])), 0), h - 1)
            e = etiket[yi, xi]
            if e == 0 or e == i + 1:
                continue
            hizali = False
            if e > 0:
                q = parcalar[e - 1]
                q_orta = (np.array([q.x1, q.y1]) + np.array([q.x2, q.y2])) / 2
                hizali = (
                    aci_farki(q.aci, p.aci) <= math.radians(10)
                    and abs((q_orta - a) @ n) <= max(p.kalinlik, q.kalinlik) / 2 + 4
                )
            if not hizali and not (
                kisa_duvar and _hizali_kisa_duvar(duvar_maske, gb, yon, p.kalinlik)
            ):
                continue
            yeni = Bosluk(tuple(map(float, ga)), tuple(map(float, gb)), p.kalinlik)
            orta = (ga + gb) / 2
            if any(
                np.linalg.norm(orta - (np.array(s.a) + np.array(s.b)) / 2) < 0.25 * s.genislik
                for s in bulunan
            ):
                continue
            bulunan.append(yeni)
    return bulunan


def ince_murekkep(ikili: np.ndarray, duvar_maske: np.ndarray, tol: float) -> np.ndarray:
    """Duvarlar (2 px genişletilmiş) çıkarılmış mürekkebin ``tol`` kadar
    genişletilmişi: yay örneği bunun içindeyse mürekkebe yakın."""
    duvarsiz = cv2.bitwise_and(
        ikili, cv2.bitwise_not(cv2.dilate(duvar_maske, np.ones((5, 5), np.uint8)))
    )
    r = math.ceil(tol)
    return cv2.dilate(
        duvarsiz, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    )


def _isabet(maske: np.ndarray, noktalar: np.ndarray) -> np.ndarray:
    h, w = maske.shape
    x = np.floor(noktalar[:, 0] + 0.5).astype(np.int64)
    y = np.floor(noktalar[:, 1] + 0.5).astype(np.int64)
    icinde = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    deger = np.zeros(len(noktalar), bool)
    deger[icinde] = maske[y[icinde], x[icinde]] > 0
    return deger


def _pay(maske: np.ndarray, noktalar: np.ndarray) -> float:
    return float(_isabet(maske, noktalar).mean())


# Tam yay: 8-82° arası 24 örnek. Skorlar eşiğe yakın yığıldığı için
# örnekleme değişince sonuç değişiyor (2° adımlı 38 örnekle fazla kapı 12'den
# 116'ya çıktı, olcumler §8); eşikler bu örneklemeyle seçildi.
_TAM_TETA = np.radians(np.linspace(8, 82, 24))
_KISMI_TETA = np.radians(np.arange(6, 60, 2.0))
_KANAT = np.linspace(0.2, 0.9, 12)
KISMI_ACILAR = (30.0, 45.0, 60.0)


def kapi_dene(
    bosluk: Bosluk,
    murekkep: np.ndarray,
    kismi_esik: float = 1.01,
    kismi_min_yaricap: float = 0.0,
    kismi_acilar: tuple[float, ...] = KISMI_ACILAR,
    kismi_kanat_yan: float = 0.3,
    olcekler: tuple[float, ...] = (0.85, 0.92, 1.0, 1.06),
    ek_olcekler: tuple[float, ...] = (),
    ek_min_yaricap: float = 0.0,
    kismi_kayit: list | None = None,
    kayit_esik: float = 1.01,
) -> Kapi | None:
    """Dört varsayımın (menteşe iki uçtan biri, açılma iki yandan biri) en
    iyisi; yarıçap ve menteşenin yüze göre yeri küçük aralıkta taranır.

    Skor: tam çeyrek yayın (8-82°) mürekkebe yakın payı. Yarı açık çizilmiş
    kapı için (``KISMI_ACILAR``) yay kapalı uçtan o açıya kadar ve o açıdaki
    kanat çizgisi birlikte aranır; skor ikisinin küçüğü. Kısmi yay zayıf
    kanıt: kanat olmadan, ``kismi_esik``'in altında ve ``kismi_min_yaricap``'tan
    küçük yarıçapta sayılmaz (varsayılan: hiç sayılmaz). Hiçbir varsayımda
    pay yoksa None.

    ``kismi_kayit`` verilirse, ``kayit_esik``'i geçen en iyi yarı açık
    varsayım, dönen sonuçtan iyiyse oraya eklenir (dönüş değişmez). Oda bölme
    için yarı açık kapıları ikinci bir tarama yapmadan toplamak içindir.
    """
    a, b = np.array(bosluk.a), np.array(bosluk.b)
    g = np.linalg.norm(b - a)
    d = (b - a) / max(g, 1e-9)
    n0 = np.array([-d[1], d[0]])
    en_iyi = None
    en_iyi_kismi = None
    for h0, j0 in ((a, b), (b, a)):
        u = (j0 - h0) / g
        for n in (n0, -n0):
            # Menteşe duvarın açılma yanındaki yüzünde ya da orta çizgide.
            for kaydir in (bosluk.kalinlik / 2, 0.0):
                h = h0 + kaydir * n
                ekler = tuple(e for e in ek_olcekler if e * g >= ek_min_yaricap)
                for olcek in olcekler + ekler:
                    r = olcek * g
                    yay = h + r * (np.cos(_TAM_TETA)[:, None] * u + np.sin(_TAM_TETA)[:, None] * n)
                    secenek = [(_pay(murekkep, yay), 90.0)]
                    kismi_yay = h + r * (
                        np.cos(_KISMI_TETA)[:, None] * u + np.sin(_KISMI_TETA)[:, None] * n
                    )
                    isabet = _isabet(murekkep, kismi_yay) if r >= kismi_min_yaricap else None
                    for phi in kismi_acilar if isabet is not None else ():
                        p = math.radians(phi)
                        kisim = _KISMI_TETA <= p - np.radians(4)
                        yon = math.cos(p) * u + math.sin(p) * n
                        kanat = h + _KANAT[:, None] * r * yon
                        kismi = min(float(isabet[kisim].mean()), _pay(murekkep, kanat))
                        # Kanat tek çizgi olmalı: iki yanında, 7 px ötede mürekkep
                        # yok. Pencere çerçevesinin sık paralel çizgileri bunu geçemez.
                        dik = np.array([-yon[1], yon[0]])
                        yan = max(_pay(murekkep, kanat + 7 * dik), _pay(murekkep, kanat - 7 * dik))
                        if kismi >= kismi_esik and yan <= kismi_kanat_yan:
                            secenek.append((kismi, phi))
                        if (
                            kismi_kayit is not None
                            and kismi >= kayit_esik
                            and yan <= kismi_kanat_yan
                            and (en_iyi_kismi is None or kismi > en_iyi_kismi.skor)
                        ):
                            en_iyi_kismi = Kapi(
                                tuple(map(float, h)),
                                tuple(map(float, h + r * u)),
                                tuple(map(float, h + r * n)),
                                kismi,
                                bosluk,
                                phi,
                            )
                    skor, aci = max(secenek)
                    if en_iyi is None or skor > en_iyi.skor:
                        en_iyi = Kapi(
                            tuple(map(float, h)),
                            tuple(map(float, h + r * u)),
                            tuple(map(float, h + r * n)),
                            skor,
                            bosluk,
                            aci,
                        )
    if en_iyi_kismi is not None and (en_iyi is None or en_iyi_kismi.skor > en_iyi.skor):
        kismi_kayit.append(en_iyi_kismi)
    return en_iyi if en_iyi is not None and en_iyi.skor > 0 else None


def cift_kanat_dene(bosluk: Bosluk, murekkep: np.ndarray) -> tuple[Kapi, Kapi] | None:
    """Çift kanatlı pencere: menteşeler boşluğun iki ucunda, iki kanat aynı
    yana açılır ve ortada buluşur (yarıçap yarım genişlik). Açılma yanı,
    menteşenin yüze göre yeri ve yarıçap ``kapi_dene``'deki gibi taranır.
    Skor iki yayın paylarının küçüğü; hiç pay yoksa None."""
    a, b = np.array(bosluk.a), np.array(bosluk.b)
    g = np.linalg.norm(b - a)
    d = (b - a) / max(g, 1e-9)
    n0 = np.array([-d[1], d[0]])
    en_iyi = None
    for n in (n0, -n0):
        for kaydir in (bosluk.kalinlik / 2, 0.0):
            for olcek in (0.85, 0.92, 1.0, 1.06):
                r = olcek * g / 2
                kanatlar = []
                for h, u in ((a + kaydir * n, d), (b + kaydir * n, -d)):
                    yay = h + r * (np.cos(_TAM_TETA)[:, None] * u + np.sin(_TAM_TETA)[:, None] * n)
                    kanatlar.append((h, u, _pay(murekkep, yay)))
                skor = min(k[2] for k in kanatlar)
                if en_iyi is None or skor > en_iyi[0].skor:
                    en_iyi = tuple(
                        Kapi(
                            tuple(map(float, h)),
                            tuple(map(float, h + r * u)),
                            tuple(map(float, h + r * n)),
                            skor,
                            bosluk,
                        )
                        for h, u, _ in kanatlar
                    )
    return en_iyi if en_iyi is not None and en_iyi[0].skor > 0 else None


def pencere_dene(bosluk: Bosluk, murekkep_ham: np.ndarray, kural: AciklikKural) -> Pencere | None:
    """Boşluğun %10-%90'ı boyunca, duvar kalınlığı içindeki her enine
    ofsette (1 px adım) mürekkep kapsaması; kapsaması yüksek ardışık ofset
    kümeleri ayrı çizgi sayılır."""
    a, b = np.array(bosluk.a), np.array(bosluk.b)
    g = np.linalg.norm(b - a)
    d = (b - a) / max(g, 1e-9)
    n = np.array([-d[1], d[0]])
    s = np.linspace(0.1, 0.9, max(int(g) // 2, 5))[:, None]
    yarim = bosluk.kalinlik / 2 + 2
    dolu = []
    for o in np.arange(-yarim, yarim + 0.5, 1.0):
        dolu.append(_pay(murekkep_ham, a + s * (b - a) + o * n) >= kural.pencere_kapsama)
    cizgi = sum(1 for i, v in enumerate(dolu) if v and (i == 0 or not dolu[i - 1]))
    return Pencere(bosluk, cizgi) if cizgi >= kural.pencere_min_cizgi else None


def dis_koseler(duvar_maske: np.ndarray, aci_tol: float, eps: float = 2.5) -> list[tuple]:
    """Duvar maskesi dış hatlarının ~90° dışbükey köşeleri: (köşe, a, b, la, lb).

    ``a`` ve ``b`` köşeden iki komşu kenar boyunca birim yönler; duvarın
    gövdesi aralarındaki açının içinde. -a ve -b kenarların köşeden öteye
    uzantısı: duvar ucundaki boşluk bu yönlerden birinde. ``la``, ``lb``
    kenar uzunlukları; duvar ucunda biri uç yüzü, yani duvar kalınlığı.
    """
    hatlar, _ = cv2.findContours(duvar_maske, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    koseler = []
    for h in hatlar:
        if len(h) < 8:
            continue
        p = cv2.approxPolyDP(h, eps, True)[:, 0, :].astype(np.float64)
        if len(p) < 3:
            continue
        # Hat dolu bölgeyi bir yönde, delik bölgeyi öbür yönde dolanır;
        # dışbükeylik işareti hattın kendi alan işaretine göre.
        yon = np.sign(cv2.contourArea(p.astype(np.float32), oriented=True)) or 1.0
        for i in range(len(p)):
            o, k, s = p[i - 1], p[i], p[(i + 1) % len(p)]
            a, b = o - k, s - k
            la, lb = np.linalg.norm(a), np.linalg.norm(b)
            if la < 3 or lb < 3:
                continue
            a, b = a / la, b / lb
            aci = math.degrees(math.acos(float(np.clip(a @ b, -1, 1))))
            capraz = a[0] * b[1] - a[1] * b[0]
            if abs(aci - 90) <= aci_tol and np.sign(capraz) == -yon:
                koseler.append((k, a, b, la, lb))
    return koseler


def kose_bosluklari(
    duvar_maske: np.ndarray, kural: AciklikKural = VARSAYILAN_ACIKLIK_KURAL
) -> list[Bosluk]:
    """Raster köşelerinden boşluk adayları: her dışbükey köşeden, köşeyi
    oluşturan iki yüzün uzantısı boyunca ışın. Parçaya dönüşmeyen kısa,
    tıknaz duvar uçlarını ve eğik duvarları da kapsar. Boşluk duvar yüzü
    hattında: menteşe o hatta, kalınlık 0 (orta çizgiye kaydırma yok).

    Boşluğu uç yüzünün yarısı kadar orta çizgiye taşımak denendi, kötüleşti
    (olcumler §8): köşenin öbür kenarı her zaman duvar kalınlığı değil."""
    bulunan: list[Bosluk] = []
    for k, a, b, _, _ in dis_koseler(duvar_maske, kural.kose_aci_tol):
        for yon in (-a, -b):
            r = bosluk_isini(duvar_maske, k, yon, kural, 4)
            if r is None:
                continue
            ga, gb = k + r[0] * yon, k + r[1] * yon
            bulunan.append(Bosluk(tuple(map(float, ga)), tuple(map(float, gb)), 0.0))
    return bulunan


def kapi_bastir(kapilar: list[Kapi], esik: float = 0.3) -> list[Kapi]:
    """Açılma alanları ``esik``'ten çok örtüşen kapılardan skoru yüksek kalır."""
    from floorplan_parser.degerlendirme import poligon_iou

    secilen: list[Kapi] = []
    for x in sorted(kapilar, key=lambda k: -k.skor):
        px = x.poligon()
        if all(
            math.dist(x.mentese, y.mentese) > x.yaricap + y.yaricap
            or poligon_iou(px, y.poligon()) < esik
            for y in secilen
        ):
            secilen.append(x)
    return secilen


def aciklik_bul(
    ikili: np.ndarray,
    parcalar: list[Parca],
    duvar_maske: np.ndarray,
    kural: AciklikKural = VARSAYILAN_ACIKLIK_KURAL,
    kapi_yontemi: str = "bosluk_kose",
    kismi_kayit: list | None = None,
    kismi_kayit_esik: float = 0.95,
) -> tuple[list[Kapi], list[Pencere], list[Bosluk]]:
    """(kapılar, pencereler, sınıflanmamış boşluklar).

    ``kismi_kayit`` verilirse, kapı listesine girmeyen yarı açık kapılar
    (pay ≥ ``kismi_kayit_esik``, pencere çizgisi taşımayan boşlukta, kapı
    boyunda, örtüşen yaylarla bastırılmış) oraya eklenir. Oda bölmede kiriş
    için; ayrı bir yarı açık tarama ile aynı sonucu tek geçişte verir.

    1. Yay: parça uçlarından bulunan boşluklarda (``kapi_yontemi``
       "bosluk_kose" ise raster köşelerinden bulunanlarda da) dört menteşe
       varsayımıyla yay; skoru ``yay_esik``'i geçenler örtüşmeye göre bastırılır.
    2. Yarıçapı ``kapi_min_yaricap``'tan büyük yay kapı, küçük yay kanatlı pencere.
    3. Yaylardan biriyle örtüşmeyen parça boşlukları paralel çizgili pencere
       diye denenir (köşe boşluklarında denemek fazla pencereyi artırdı).

    ``kural.cift_kanat``: parça boşluklarında çift kanat da denenir; skoru
    eşiği ve tek yayın skorunu geçerse çift kanatlı pencere. ``genis_pencere``:
    açıklıklarla örtüşmeyen geniş boşluklarda (150-450 px) çift kanat ve
    paralel çizgi denenir.
    """
    murekkep = ince_murekkep(ikili, duvar_maske, kural.yay_tol)
    parca_bosluk = bosluklar(parcalar, duvar_maske, kural)
    adaylar = list(parca_bosluk)
    if kapi_yontemi == "bosluk_kose":
        adaylar += kose_bosluklari(duvar_maske, kural)
    yalniz_yay: set[Bosluk] = set()
    if kural.kavsak_aday:
        var = set(parca_bosluk)
        yalniz_yay = {
            b for b in bosluklar(parcalar, duvar_maske, kural, kavsak_serbest=True) if b not in var
        }
        adaylar += sorted(yalniz_yay, key=lambda b: (b.a, b.b))
    yaylar = []
    kismi_aday: list[Kapi] = []
    ciftler: list[Pencere] = []
    for bo in adaylar:
        kismi_esik = kural.kismi_esik
        # Pencere çerçevesi kısa kısmi yayla ve kanatla karışıyor (olcumler §11):
        # paralel pencere çizgisi taşıyan boşlukta yalnız tam yay.
        pencereli = None
        if (kismi_esik <= 1.0 and not kural.kismi_pencerede) or kismi_kayit is not None:
            pencereli = pencere_dene(bo, ikili, kural) is not None
        if kismi_esik <= 1.0 and not kural.kismi_pencerede and pencereli:
            kismi_esik = 1.01
        kayit = [] if kismi_kayit is not None and not pencereli else None
        k = kapi_dene(
            bo,
            murekkep,
            kismi_esik,
            kural.kapi_min_yaricap,
            kural.kismi_acilar,
            kural.kismi_kanat_yan,
            kural.yaricap_olcekleri,
            kural.kapi_ek_olcekleri,
            kural.kapi_min_yaricap,
            kayit,
            kismi_kayit_esik,
        )
        c = None
        if kural.cift_kanat and bo.kalinlik > 0 and bo not in yalniz_yay:
            c = cift_kanat_dene(bo, murekkep)
            if c is not None and c[0].skor < kural.yay_esik:
                c = None
        # Yarı açık kayıt: çift kanat ondan iyi değilse (ayrı taramadaki seçim).
        if kayit and (c is None or c[0].skor < kayit[0].skor):
            kismi_aday.extend(kayit)
        if c is not None and (k is None or c[0].skor >= k.skor):
            ciftler.append(Pencere(bo, kanatlar=c))
            continue
        if k is not None and k.skor >= kural.yay_esik:
            yaylar.append(k)
    cift_poligon = [p.poligon().astype(np.float32) for p in ciftler]
    if kismi_kayit is not None:
        # Ayrı taramadaki gibi: tam ve yarı açık yaylar birlikte bastırılır,
        # kalan yarı açıklar (kapı boyunda, çift kanatlının içinde değil) kayda.
        for k in kapi_bastir(yaylar + kismi_aday):
            if (
                k.aci < 90
                and k.yaricap >= kural.kapi_min_yaricap
                and not any(
                    cv2.pointPolygonTest(cp, tuple(map(float, k.poligon().mean(0))), False) >= 0
                    for cp in cift_poligon
                )
            ):
                kismi_kayit.append(k)
    # Çift kanatlı pencerenin içinde kalan tek yay onun bir kanadı.
    yaylar = [
        k
        for k in kapi_bastir(yaylar)
        if not any(
            cv2.pointPolygonTest(cp, tuple(map(float, k.poligon().mean(0))), False) >= 0
            for cp in cift_poligon
        )
    ]
    kapilar = [k for k in yaylar if k.yaricap >= kural.kapi_min_yaricap]
    pencereler = [
        Pencere(k.bosluk, kanat=k, dikdortgen=kural.kanat_dikdortgen)
        for k in yaylar
        if k.yaricap < kural.kapi_min_yaricap
    ]
    pencereler += ciftler
    poligonlar = [k.poligon().astype(np.float32) for k in yaylar] + cift_poligon

    def ortulu(bo: Bosluk) -> bool:
        orta = ((bo.a[0] + bo.b[0]) / 2, (bo.a[1] + bo.b[1]) / 2)
        return any(
            cv2.pointPolygonTest(p, orta, True) >= -(bo.kalinlik / 2 + 2) for p in poligonlar
        )

    kalan = []
    for bo in parca_bosluk:
        if ortulu(bo):
            continue
        p = pencere_dene(bo, ikili, kural)
        if p is not None:
            pencereler.append(p)
        else:
            kalan.append(bo)
    if kural.genis_pencere:
        poligonlar += [p.poligon().astype(np.float32) for p in pencereler]
        for bo in genis_bosluklar(parcalar, duvar_maske, kisa_duvar=kural.genis_kisa_duvar):
            if ortulu(bo):
                continue
            c = cift_kanat_dene(bo, murekkep) if kural.cift_kanat else None
            p = (
                Pencere(bo, kanatlar=c)
                if c is not None and c[0].skor >= kural.yay_esik
                else pencere_dene(bo, ikili, kural)
            )
            if p is not None:
                pencereler.append(p)
                poligonlar.append(p.poligon().astype(np.float32))
    return kapilar, pencereler, kalan
