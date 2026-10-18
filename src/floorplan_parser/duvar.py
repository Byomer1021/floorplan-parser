"""Duvar: kalın maskeden düz duvar parçaları.

Akış:

1. İskelet (skimage) — kalın duvar tek piksellik orta çizgiye iner.
2. Olasılıksal Hough (``cv2.HoughLinesP``) iskelette çizgi parçaları bulur.
3. Eşdoğrusal parçalar birleşir: açı ve orta çizgi uzaklığı yakın, aradaki
   boşluk ``max_bosluk``'tan küçük.
4. Uçlar orta çizgi boyunca maskenin sınırına uzatılır (iskelet uçta kısalır).
5. Başka bir duvara yakın uç, iki orta çizginin kesişimine çekilir.
6. Kalınlık, uzaklık dönüşümünün orta çizgi boyunca medyanının iki katı.

Kritik ayar ``max_bosluk``: Hough'un ve birleştirmenin köprülediği en büyük
boşluk. Kapı genişliğinden büyükse kapı boşluğu kapanır ve iki duvar tek
duvar olur (onisleme'deki kapama sorununun aynısı).
"""

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np
from skimage.morphology import skeletonize

from floorplan_parser.onisleme import ac, ac_yonsuz

_ALT_PIKSEL = 4


@dataclass(frozen=True)
class Parca:
    """Orta çizgi (x1, y1)-(x2, y2): graf uçları, köşede kesişim noktası.

    ``kapsam``: aynı orta çizgi üzerinde duvarın maskedeki gerçek uçları
    (köşede öbür duvarın dış yüzüne kadar). Piksel maskesi bundan çizilir.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    kalinlik: float = 0.0
    kapsam: tuple[float, float, float, float] | None = None

    @property
    def uzunluk(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def aci(self) -> float:
        """[0, π) radyan."""
        return math.atan2(self.y2 - self.y1, self.x2 - self.x1) % math.pi


def aci_farki(a: float, b: float) -> float:
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def iskelet(kalin: np.ndarray) -> np.ndarray:
    return skeletonize(kalin > 0).astype(np.uint8) * 255


def hough(isk: np.ndarray, esik: int, min_uzunluk: int, max_bosluk: int) -> np.ndarray:
    """(N, 4) x1 y1 x2 y2."""
    cizgiler = cv2.HoughLinesP(
        isk, 1, np.pi / 180, esik, minLineLength=min_uzunluk, maxLineGap=max_bosluk
    )
    if cizgiler is None:
        return np.zeros((0, 4))
    # OpenCV 4 (N, 1, 4), OpenCV 5 (N, 4) döndürüyor.
    return cizgiler.reshape(-1, 4).astype(np.float64)


def birlestir(
    parcalar: np.ndarray, aci_tol: float, dik_tol: float, max_bosluk: float
) -> np.ndarray:
    """Eşdoğrusal parçaları tek parçaya indirir. (N, 4) → (M, 4)."""
    if len(parcalar) == 0:
        return parcalar
    uzunluk = np.hypot(parcalar[:, 2] - parcalar[:, 0], parcalar[:, 3] - parcalar[:, 1])
    kumeler: list[dict] = []
    for i in np.argsort(-uzunluk):
        x1, y1, x2, y2 = parcalar[i]
        aci = math.atan2(y2 - y1, x2 - x1) % math.pi
        orta = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        for k in kumeler:
            if aci_farki(aci, k["aci"]) <= aci_tol and abs(k["n"] @ orta - k["rho"]) <= dik_tol:
                k["uyeler"].append(i)
                break
        else:
            d = np.array([math.cos(aci), math.sin(aci)])
            n = np.array([-d[1], d[0]])
            kumeler.append({"aci": aci, "d": d, "n": n, "rho": n @ orta, "uyeler": [i]})

    cikti = []
    for k in kumeler:
        d, n = k["d"], k["n"]
        uyeler = parcalar[k["uyeler"]]
        agirlik = uzunluk[k["uyeler"]]
        ortalar = (uyeler[:, :2] + uyeler[:, 2:]) / 2
        rho = np.average(ortalar @ n, weights=agirlik)
        t = np.stack([uyeler[:, :2] @ d, uyeler[:, 2:] @ d], axis=1)
        araliklar = sorted(zip(t.min(1), t.max(1)))
        bas, son = araliklar[0]
        for a, b in araliklar[1:]:
            if a <= son + max_bosluk:
                son = max(son, b)
            else:
                cikti.append([*(rho * n + bas * d), *(rho * n + son * d)])
                bas, son = a, b
        cikti.append([*(rho * n + bas * d), *(rho * n + son * d)])
    return np.array(cikti)


def _yuru(maske: np.ndarray, p: np.ndarray, d: np.ndarray, sinir: float, adim: float) -> np.ndarray:
    """``p`` (S, 2) noktalarından ``d`` yönünde ``adim`` adımlarla yürür;
    maske içinde kalınan son uzaklık (S,). Piksel (i, j) sürekli düzlemde
    [i-0.5, i+0.5) aralığı: sınır piksel kenarında."""
    h, w = maske.shape
    t = np.arange(1, int(sinir / adim) + 1) * adim
    q = p[:, None, :] + t[None, :, None] * d
    x = np.floor(q[..., 0] + 0.5).astype(np.int64)
    y = np.floor(q[..., 1] + 0.5).astype(np.int64)
    icinde = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    icinde[icinde] = maske[y[icinde], x[icinde]] > 0
    disari = (~icinde).argmax(axis=1)  # ilk dışarı adım; hiç yoksa 0
    disari[icinde.all(axis=1)] = len(t)
    return disari * adim


def uzat(
    parca: np.ndarray, maske: np.ndarray, sinir: float = 2000, adim: float = 0.5
) -> np.ndarray:
    a, b = parca[:2], parca[2:]
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    ta = _yuru(maske, a[None], -d, sinir, adim)[0]
    tb = _yuru(maske, b[None], d, sinir, adim)[0]
    return np.concatenate([a - ta * d, b + tb * d])


def enine_profil(
    parca: np.ndarray,
    maske: np.ndarray,
    sinir: float = 200,
    adim: float = 0.25,
    aralik: tuple[float, float] = (0.1, 0.9),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orta çizgi boyunca örnek başına (artı yana, eksi yana, merkez içeride mi).

    Örnekler ``aralik`` kesrinde, iki pikselde bir. Merkez maske dışındaysa
    iki yan da 0 çıkar (boşluk).
    """
    a, b = parca[:2], parca[2:]
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    n = np.array([-d[1], d[0]])
    t = np.linspace(*aralik, max(int(np.linalg.norm(b - a)) // 2, 3))[:, None]
    p = a + t * (b - a)
    h, w = maske.shape
    x = np.clip(np.floor(p[:, 0] + 0.5).astype(int), 0, w - 1)
    y = np.clip(np.floor(p[:, 1] + 0.5).astype(int), 0, h - 1)
    return _yuru(maske, p, n, sinir, adim), _yuru(maske, p, -n, sinir, adim), maske[y, x] > 0


def enine_olc(
    parca: np.ndarray, maske: np.ndarray, sinir: float = 200, adim: float = 0.25
) -> tuple[np.ndarray, float]:
    """Orta çizginin %10-%90'ı boyunca duvarı enine ölçer: her noktadan iki
    yana maske bitene kadar yürür. Ortalanmış parça ve kalınlık (medyan).

    İskelet çift kalınlıkta yarım piksel kayık durur; bu onu düzeltir.
    Kavşaktaki örnekler öbür duvara taşar, medyan bunları eler.
    """
    a, b = parca[:2], parca[2:]
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    n = np.array([-d[1], d[0]])
    arti_s, eksi_s, _ = enine_profil(parca, maske, sinir, adim)
    arti, eksi = float(np.median(arti_s)), float(np.median(eksi_s))
    kayma = (arti - eksi) / 2
    # Son içeride nokta ile sınır arası ortalama yarım adım, iki yanda bir adım.
    return parca + np.tile(kayma * n, 2), arti + eksi + adim


def _kesisim(p1, p2, q1, q2) -> np.ndarray | None:
    r, s = p2 - p1, q2 - q1
    payda = r[0] * s[1] - r[1] * s[0]
    if abs(payda) < 1e-9:
        return None
    t = ((q1 - p1)[0] * s[1] - (q1 - p1)[1] * s[0]) / payda
    return p1 + t * r


def kavsaklara_cek(
    parcalar: np.ndarray,
    kalinliklar: np.ndarray,
    min_aci: float,
    oynar: set[int] | None = None,
) -> np.ndarray:
    """Başka duvara yakın ucu iki orta çizginin kesişimine taşır.

    Yakınlık: uç, kesişim noktasına iki kalınlıktan büyüğü kadar yakın;
    kesişim öbür parçanın üzerinde (uçlarından yarım kalınlık taşmaya izinle)
    ve ucun gerisinde en çok öbür duvarın yarım kalınlığı kadar. Daha
    gerideki kesişim parçanın ortasındaki bir T-kavşağıdır: ucu oraya çekmek
    duvarı kısaltır (kapıya bakan serbest uç 20-60 px geri çekiliyordu).
    ``oynar`` verilirse yalnız o parçaların uçları taşınır; hedef hepsi.
    """
    yeni = parcalar.copy()
    for i, p in enumerate(parcalar):
        if oynar is not None and i not in oynar:
            continue
        p_uzunluk = max(np.linalg.norm(p[2:] - p[:2]), 1e-9)
        for uc in (0, 1):
            e = p[2 * uc : 2 * uc + 2]
            disari = (p[2:] - p[:2]) / p_uzunluk * (1 if uc else -1)
            en_iyi, en_iyi_uzaklik = None, math.inf
            for j, q in enumerate(parcalar):
                if i == j:
                    continue
                aci_p = math.atan2(p[3] - p[1], p[2] - p[0])
                aci_q = math.atan2(q[3] - q[1], q[2] - q[0])
                if aci_farki(aci_p, aci_q) < min_aci:
                    continue
                x = _kesisim(p[:2], p[2:], q[:2], q[2:])
                if x is None:
                    continue
                yaricap = max(kalinliklar[i], kalinliklar[j])
                uzaklik = np.linalg.norm(x - e)
                q_uzunluk = np.linalg.norm(q[2:] - q[:2])
                t = (x - q[:2]) @ (q[2:] - q[:2]) / max(q_uzunluk, 1e-9)
                tasma = kalinliklar[j] / 2 + 1
                ileri = (x - e) @ disari  # kesişim ucun ötesinde (+) ya da gerisinde (-)
                if (
                    uzaklik <= yaricap
                    and -tasma <= t <= q_uzunluk + tasma
                    and ileri >= -(kalinliklar[j] / 2 + 2)
                    and uzaklik < en_iyi_uzaklik
                ):
                    en_iyi, en_iyi_uzaklik = x, uzaklik
            if en_iyi is not None:
                yeni[i, 2 * uc : 2 * uc + 2] = en_iyi
    return yeni


def duvarlari_bul(
    kalin: np.ndarray,
    max_bosluk: int = 20,
    hough_esik: int = 10,
    min_uzunluk: int = 15,
    aci_tol_derece: float = 3.0,
    dik_tol: float = 3.0,
) -> list[Parca]:
    isk = iskelet(kalin)
    ham = hough(isk, hough_esik, min_uzunluk, max_bosluk)
    birlesik = birlestir(ham, math.radians(aci_tol_derece), dik_tol, max_bosluk)
    if len(birlesik) == 0:
        return []
    olculen = [enine_olc(p, kalin) for p in birlesik]
    kalinliklar = np.array([k for _, k in olculen])
    kapsam = np.array([uzat(p, kalin) for p, _ in olculen])
    cekili = kavsaklara_cek(kapsam, kalinliklar, math.radians(20))
    return [
        Parca(*map(float, p), kalinlik=float(k), kapsam=tuple(map(float, u)))
        for p, k, u in zip(cekili, kalinliklar, kapsam)
        if np.linalg.norm(u[2:] - u[:2]) >= min_uzunluk
    ]


@dataclass(frozen=True)
class InceKural:
    """İnce aday parçanın duvar sayılma koşulları (K-007, eşikler geliştirme
    bölümünde seçildi, docs/olcumler.md §6)."""

    # 45'ti (§6); 30 kısa bölme ve eğik parçaları da alıyor, duvar IoU aynı
    # (olcumler §16).
    min_uzunluk: float = 30
    max_kalinlik: float = 10
    max_degisim: float = 0.2  # kalınlık IQR / medyan
    max_paralel: int = 1  # merdiven basamakları paralel ve yakın
    paralel_uzaklik: float = 60
    # Açmada kare + disk (``onisleme.ac_yonsuz``): eğik ince duvarlar kare
    # açmada siliniyordu (olcumler §16). "" kapalı, "ince" yalnız ince
    # ölçekte, "hepsi" kalın ölçekte de (kalın yazı harflerini duvar yapıyor),
    # "ek": kare adaylar aynen, diskin getirdiği yeni piksellerden yalnız eğik
    # parçalar eklenir (mevcut parçalar kaymaz).
    yonsuz: str = "ek"
    egik_min_sapma: float = 10.0  # "ek" kipinde eksenlerden en az sapma (derece)


VARSAYILAN_INCE_KURAL = InceKural()


def ince_ozellikler(
    p: Parca, maske: np.ndarray, hepsi: list[Parca], paralel_uzaklik: float
) -> dict:
    """Uzunluk, kalınlık (medyan), kalınlık değişimi, doluluk, paralel komşu sayısı."""
    k = np.array(p.kapsam or (p.x1, p.y1, p.x2, p.y2))
    arti, eksi, merkez = enine_profil(k, maske, aralik=(0.05, 0.95))
    gen = (arti + eksi + 0.25)[merkez]
    a, b = k[:2], k[2:]
    uzunluk = float(np.linalg.norm(b - a))
    d = (b - a) / max(uzunluk, 1e-9)
    n = np.array([-d[1], d[0]])
    paralel = 0
    for q in hepsi:
        if q is p or aci_farki(q.aci, p.aci) > math.radians(3):
            continue
        qk = np.array(q.kapsam or (q.x1, q.y1, q.x2, q.y2))
        qa, qb = qk[:2], qk[2:]
        if abs(((qa + qb) / 2 - a) @ n) > paralel_uzaklik:
            continue
        t0, t1 = sorted([(qa - a) @ d, (qb - a) @ d])
        if min(t1, uzunluk) - max(t0, 0) >= 0.5 * min(uzunluk, t1 - t0):
            paralel += 1
    if len(gen):
        medyan = float(np.median(gen))
        degisim = float(np.subtract(*np.percentile(gen, [75, 25])) / max(medyan, 1e-9))
    else:
        medyan, degisim = 0.0, math.inf
    return {
        "uzunluk": uzunluk,
        "kalinlik": medyan,
        "kalinlik_degisim": degisim,
        "doluluk": float(merkez.mean()),
        "paralel": paralel,
    }


def ince_secer(oz: dict, kural: InceKural) -> bool:
    return (
        oz["uzunluk"] >= kural.min_uzunluk
        and oz["kalinlik"] <= kural.max_kalinlik
        and oz["kalinlik_degisim"] <= kural.max_degisim
        and oz["paralel"] <= kural.max_paralel
    )


def dortgen(p: Parca) -> np.ndarray:
    """Parçanın ``kapsam`` uçları arasında kalınlığı kadar dikdörtgeni, (4, 2).

    Köşeler sınır piksellerinin merkezinde: ``cv2.fillConvexPoly`` kenar
    piksellerini de boyar, yarım genişlik (kalınlık - 1) / 2.
    """
    x1, y1, x2, y2 = p.kapsam or (p.x1, p.y1, p.x2, p.y2)
    a, b = np.array([x1, y1]), np.array([x2, y2])
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    n = np.array([-d[1], d[0]]) * (p.kalinlik - 1) / 2
    return np.array([a + n, b + n, b - n, a - n])


def dortgen_ciz(goruntu: np.ndarray, p: Parca, deger: int) -> None:
    kose = np.round(dortgen(p) * (1 << _ALT_PIKSEL)).astype(np.int32)
    cv2.fillConvexPoly(goruntu, kose, deger, cv2.LINE_8, _ALT_PIKSEL)


def ince_aday_maskesi(
    ikili: np.ndarray, k_kalin: int, k_ince: int, yonsuz: str = ""
) -> tuple[np.ndarray, np.ndarray]:
    """(kalın maske, ince aday maskesi). İnce aday: k_ince açmasından kalan,
    kalın maskenin 1 piksel genişletilmişi dışında kalan kısım. ``yonsuz``:
    "ince" ince açma, "hepsi" iki açma da kare + disk (``onisleme.ac_yonsuz``)."""
    kalin = (ac_yonsuz if yonsuz == "hepsi" else ac)(ikili, k_kalin)
    genis = cv2.dilate(kalin, np.ones((3, 3), np.uint8))
    ince = (ac_yonsuz if yonsuz in ("ince", "hepsi") else ac)(ikili, k_ince)
    return kalin, cv2.bitwise_and(ince, cv2.bitwise_not(genis))


def egik_ince_parcalar(
    ikili: np.ndarray,
    kalin: np.ndarray,
    aday: np.ndarray,
    k_ince: int,
    kural: InceKural,
    max_bosluk: int,
) -> list[Parca]:
    """Kare açmanın sildiği eğik ince duvarlar: disk açmasının kareye göre
    fazlası (kalın maske ve kare adaylar 1 px genişletilip çıkarılmış) ayrı
    bir maske; oradaki parçalardan eksenlere ``egik_min_sapma``'dan açılı
    olup ince kurallarını geçenler."""
    k3 = np.ones((3, 3), np.uint8)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_ince, k_ince))
    fazla = cv2.morphologyEx(ikili, cv2.MORPH_OPEN, disk)
    for m in (kalin, aday):
        fazla = cv2.bitwise_and(fazla, cv2.bitwise_not(cv2.dilate(m, k3)))
    parcalar = duvarlari_bul(fazla, max_bosluk)
    sinir = math.radians(kural.egik_min_sapma)
    return [
        p
        for p in parcalar
        if min(p.aci % (math.pi / 2), math.pi / 2 - p.aci % (math.pi / 2)) >= sinir
        and ince_secer(ince_ozellikler(p, fazla, parcalar, kural.paralel_uzaklik), kural)
    ]


def iki_olcekli_duvarlar(
    ikili: np.ndarray,
    k_kalin: int = 13,
    k_ince: int = 7,
    kural: InceKural = VARSAYILAN_INCE_KURAL,
    max_bosluk: int = 20,
) -> tuple[np.ndarray, list[Parca], list[Parca]]:
    """K-007 (a): kalın duvarlar her zaman, ince adaylar kurala uyarsa.

    ``ikili``: mürekkep 255 (ör. ``onisleme.delik_doldur(ikile(gri), 4)``).
    Dönüş: (kalın maske, kalın parçalar, seçilen ince parçalar).
    """
    kalin, aday = ince_aday_maskesi(
        ikili, k_kalin, k_ince, "" if kural.yonsuz == "ek" else kural.yonsuz
    )
    kalin_parcalar = duvarlari_bul(kalin, max_bosluk)
    adaylar = duvarlari_bul(aday, max_bosluk)
    secilen = [
        p
        for p in adaylar
        if ince_secer(ince_ozellikler(p, aday, adaylar, kural.paralel_uzaklik), kural)
    ]
    if kural.yonsuz == "ek":
        secilen += egik_ince_parcalar(ikili, kalin, aday, k_ince, kural, max_bosluk)
    if not secilen:
        return kalin, kalin_parcalar, secilen
    # İnce parça kalın maskenin 1 piksel genişletilmişinden kesildi: ucu kalın
    # duvarın yüzünde. Graf ucu için kalın duvarın orta çizgisine çekilir.
    hepsi = kalin_parcalar + secilen
    uclar = np.array([[p.x1, p.y1, p.x2, p.y2] for p in hepsi])
    kalinliklar = np.array([p.kalinlik for p in hepsi])
    oynar = set(range(len(kalin_parcalar), len(hepsi)))
    cekili = kavsaklara_cek(uclar, kalinliklar, math.radians(20), oynar)
    # Kapsam da çekilen uca kadar uzar: yoksa ince ile kalın duvar arasında
    # raster'da 1-2 px yarık kalır ve odalar o yarıktan birbirine akar.
    secilen = [
        replace(
            p,
            x1=float(u[0]),
            y1=float(u[1]),
            x2=float(u[2]),
            y2=float(u[3]),
            kapsam=_kapsami_uzat(p.kapsam, u),
        )
        for p, u in zip(secilen, cekili[len(kalin_parcalar) :])
    ]
    return kalin, kalin_parcalar, secilen


def _kapsami_uzat(kapsam: tuple, graf: np.ndarray) -> tuple[float, float, float, float]:
    """Kapsamı, graf uçlarının doğru üzerindeki izdüşümlerini de kapsayacak
    kadar uzatır (kısaltmaz)."""
    k = np.array(kapsam)
    a, b = k[:2], k[2:]
    L = max(np.linalg.norm(b - a), 1e-9)
    d = (b - a) / L
    t = [0.0, L, (graf[:2] - a) @ d, (graf[2:] - a) @ d]
    yeni_a, yeni_b = a + min(t) * d, a + max(t) * d
    return (*map(float, yeni_a), *map(float, yeni_b))
