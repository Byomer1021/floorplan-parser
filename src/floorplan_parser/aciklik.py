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
    yay_tol: float = 3.0  # örnek noktanın mürekkebe en çok uzaklığı (px)
    pencere_kapsama: float = 0.8  # paralel çizginin boşluk boyunca doluluğu
    pencere_min_cizgi: int = 2
    kose_aci_tol: float = 25.0  # raster köşesi açısı 90 ± bu kadar derece


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


@dataclass(frozen=True)
class Pencere:
    """Boşlukta paralel çizgiler (``cizgi``) ya da kanatlı pencere (``kanat``:
    küçük yarıçaplı yay; poligon açılma alanı, cevaptaki gibi)."""

    bosluk: Bosluk
    cizgi: int = 0
    kanat: Kapi | None = None

    def poligon(self) -> np.ndarray:
        if self.kanat is not None:
            return self.kanat.poligon()
        a, b = np.array(self.bosluk.a), np.array(self.bosluk.b)
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n = np.array([-d[1], d[0]]) * self.bosluk.kalinlik / 2
        return np.array([a + n, b + n, b - n, a - n])


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


def bosluklar(
    parcalar: list[Parca], duvar_maske: np.ndarray, kural: AciklikKural = VARSAYILAN_ACIKLIK_KURAL
) -> list[Bosluk]:
    """Duvar uçlarından boşluk adayları.

    Serbest uç (kavşağa çekilmemiş): karşıdaki herhangi bir duvar. Kavşaktaki
    uç: ışın komşu duvarı geçer; karşıdaki duvar bu parçayla aynı doğrultuda
    olmalı (kapı köşenin dibinde, duvar kapıdan sonra sürüyor). Böylece
    T-kavşağından odanın karşısına boşluk açılmaz.
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
            if kavsak:
                xi = min(max(int(np.floor(gb[0] + 0.5 + yon[0])), 0), w - 1)
                yi = min(max(int(np.floor(gb[1] + 0.5 + yon[1])), 0), h - 1)
                e = etiket[yi, xi]
                if e <= 0 or e == i + 1 or aci_farki(parcalar[e - 1].aci, p.aci) > math.radians(10):
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
) -> Kapi | None:
    """Dört varsayımın (menteşe iki uçtan biri, açılma iki yandan biri) en
    iyisi; yarıçap ve menteşenin yüze göre yeri küçük aralıkta taranır.

    Skor: tam çeyrek yayın (8-82°) mürekkebe yakın payı. Yarı açık çizilmiş
    kapı için (``KISMI_ACILAR``) yay kapalı uçtan o açıya kadar ve o açıdaki
    kanat çizgisi birlikte aranır; skor ikisinin küçüğü. Kısmi yay zayıf
    kanıt: kanat olmadan, ``kismi_esik``'in altında ve ``kismi_min_yaricap``'tan
    küçük yarıçapta sayılmaz (varsayılan: hiç sayılmaz). Hiçbir varsayımda
    pay yoksa None.
    """
    a, b = np.array(bosluk.a), np.array(bosluk.b)
    g = np.linalg.norm(b - a)
    d = (b - a) / max(g, 1e-9)
    n0 = np.array([-d[1], d[0]])
    en_iyi = None
    for h0, j0 in ((a, b), (b, a)):
        u = (j0 - h0) / g
        for n in (n0, -n0):
            # Menteşe duvarın açılma yanındaki yüzünde ya da orta çizgide.
            for kaydir in (bosluk.kalinlik / 2, 0.0):
                h = h0 + kaydir * n
                for olcek in (0.85, 0.92, 1.0, 1.06):
                    r = olcek * g
                    yay = h + r * (np.cos(_TAM_TETA)[:, None] * u + np.sin(_TAM_TETA)[:, None] * n)
                    secenek = [(_pay(murekkep, yay), 90.0)]
                    kismi_yay = h + r * (
                        np.cos(_KISMI_TETA)[:, None] * u + np.sin(_KISMI_TETA)[:, None] * n
                    )
                    isabet = _isabet(murekkep, kismi_yay) if r >= kismi_min_yaricap else None
                    for phi in KISMI_ACILAR if isabet is not None else ():
                        p = math.radians(phi)
                        kisim = _KISMI_TETA <= p - np.radians(4)
                        kanat = h + _KANAT[:, None] * r * (math.cos(p) * u + math.sin(p) * n)
                        kismi = min(float(isabet[kisim].mean()), _pay(murekkep, kanat))
                        if kismi >= kismi_esik:
                            secenek.append((kismi, phi))
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
    return en_iyi if en_iyi is not None and en_iyi.skor > 0 else None


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
) -> tuple[list[Kapi], list[Pencere], list[Bosluk]]:
    """(kapılar, pencereler, sınıflanmamış boşluklar).

    1. Yay: parça uçlarından bulunan boşluklarda (``kapi_yontemi``
       "bosluk_kose" ise raster köşelerinden bulunanlarda da) dört menteşe
       varsayımıyla yay; skoru ``yay_esik``'i geçenler örtüşmeye göre bastırılır.
    2. Yarıçapı ``kapi_min_yaricap``'tan büyük yay kapı, küçük yay kanatlı pencere.
    3. Yaylardan biriyle örtüşmeyen parça boşlukları paralel çizgili pencere
       diye denenir (köşe boşluklarında denemek fazla pencereyi artırdı).
    """
    murekkep = ince_murekkep(ikili, duvar_maske, kural.yay_tol)
    parca_bosluk = bosluklar(parcalar, duvar_maske, kural)
    adaylar = list(parca_bosluk)
    if kapi_yontemi == "bosluk_kose":
        adaylar += kose_bosluklari(duvar_maske, kural)
    yaylar = []
    for bo in adaylar:
        k = kapi_dene(bo, murekkep, kural.kismi_esik, kural.kapi_min_yaricap)
        if k is not None and k.skor >= kural.yay_esik:
            yaylar.append(k)
    yaylar = kapi_bastir(yaylar)
    kapilar = [k for k in yaylar if k.yaricap >= kural.kapi_min_yaricap]
    pencereler = [Pencere(k.bosluk, kanat=k) for k in yaylar if k.yaricap < kural.kapi_min_yaricap]
    poligonlar = [k.poligon().astype(np.float32) for k in yaylar]
    kalan = []
    for bo in parca_bosluk:
        orta = ((bo.a[0] + bo.b[0]) / 2, (bo.a[1] + bo.b[1]) / 2)
        if any(cv2.pointPolygonTest(p, orta, True) >= -(bo.kalinlik / 2 + 2) for p in poligonlar):
            continue
        p = pencere_dene(bo, ikili, kural)
        if p is not None:
            pencereler.append(p)
        else:
            kalan.append(bo)
    return kapilar, pencereler, kalan
