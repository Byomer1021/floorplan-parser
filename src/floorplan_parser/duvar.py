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
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize


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
    t = np.linspace(0.1, 0.9, max(int(np.linalg.norm(b - a)) // 2, 3))[:, None]
    p = a + t * (b - a)
    arti = float(np.median(_yuru(maske, p, n, sinir, adim)))
    eksi = float(np.median(_yuru(maske, p, -n, sinir, adim)))
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


def kavsaklara_cek(parcalar: np.ndarray, kalinliklar: np.ndarray, min_aci: float) -> np.ndarray:
    """Başka duvara yakın ucu iki orta çizginin kesişimine taşır.

    Yakınlık: uç, kesişim noktasına iki kalınlıktan büyüğü kadar yakın ve
    kesişim öbür parçanın üzerinde (uçlarından yarım kalınlık taşmaya izinle).
    """
    yeni = parcalar.copy()
    for i, p in enumerate(parcalar):
        for uc in (0, 1):
            e = p[2 * uc : 2 * uc + 2]
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
                if (
                    uzaklik <= yaricap
                    and -tasma <= t <= q_uzunluk + tasma
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
