"""Taranmış plandan sentetik telefon fotoğrafı (ölçüm için, kararlar K-010).

Gerçek fotoğrafın cevabı yok. Burada sayfanın fotoğrafta nereye düştüğü
bilindiği için cevap da aynı dönüşümle taşınır ve tarama ile fotoğraf aynı
ölçüyle kıyaslanır.

Benzetilen: kamera eğimi (iğne deliği kamera, perspektif), kâğıdın dönmesi,
masa, eşit olmayan ışık ve gölge, bulanıklık, gürültü, JPEG. İsteğe bağlı:
buruşturulup açılmış kâğıt (kırışık izleri, kırışık boyunca gölge ve küçük
yerel kayma). Kırışığın yerel kayması homografinin dışında kalır. Kahin
köşeleriyle düzeltilmiş sayfada bile kalan bozulma budur.

Sınır: tek renk (gri), ışık yumuşak bir alan. Parlama, odak dışı bölge ve
kâğıdın kıvrılması (silindirik eğilme) yok.
"""

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from floorplan_parser.cvcfp import Cevap, Nesne


@dataclass(frozen=True)
class FotoAyar:
    egim: float = 20.0  # derece, kamera ekseninin kâğıt normaline açısı
    donme: float = 8.0  # derece, kâğıdın görüntüde dönmesi en fazla (±, rastgele)
    odak: float = 1.2  # odak uzaklığı / sayfanın uzun kenarı
    masa_payi: float = 0.08  # kâğıdın çevresinde masa, fotoğraf boyutunun oranı
    isik: float = 0.30  # doğrusal ışık düşüşü, en karanlık uçta oran
    golge: float = 0.25  # yumuşak gölgenin merkezde karartması
    gurultu: float = 4.0  # gri seviye standart sapması
    bulanik: float = 0.8  # Gauss sigması (piksel)
    jpeg: int = 85
    buruk: int = 0  # kırışık sayısı (0: düz kâğıt)
    tohum: int = 0


@dataclass
class Foto:
    goruntu: np.ndarray  # uint8 gri
    hom: np.ndarray  # sayfa → fotoğraf homografisi (kırışık kayması hariç)
    kose: np.ndarray  # kâğıdın fotoğraftaki dört köşesi (gerçek)
    kayma: tuple[np.ndarray, np.ndarray] | None  # kırışık: sayfada (dx, dy) alanı

    def tasi(self, noktalar: np.ndarray) -> np.ndarray:
        """Sayfa noktalarını fotoğrafa taşır (kırışık kayması dahil)."""
        p = np.asarray(noktalar, np.float64).reshape(-1, 2)
        if self.kayma is not None:
            p = _ileri_kayma(p, *self.kayma)
        return cv2.perspectiveTransform(p.reshape(-1, 1, 2), self.hom).reshape(-1, 2)


def _rodrigues(eksen: np.ndarray, aci: float) -> np.ndarray:
    r, _ = cv2.Rodrigues(np.asarray(eksen, np.float64) * aci)
    return r


def _kamera_homografisi(gen: int, yuk: int, ayar: FotoAyar, rng) -> tuple[np.ndarray, tuple]:
    """Sayfa → fotoğraf homografisi ve fotoğraf boyutu. Sayfanın merkezi
    kameranın ekseninde, merkezde ölçek 1 (tarama çözünürlüğü korunur)."""
    f = ayar.odak * max(gen, yuk)
    donme = math.radians(rng.uniform(-ayar.donme, ayar.donme))
    yon = rng.uniform(0, 2 * math.pi)  # eğimin yönü
    r = _rodrigues([math.cos(yon), math.sin(yon), 0], math.radians(ayar.egim)) @ _rodrigues(
        [0, 0, 1], donme
    )
    merkez = np.array([[1, 0, -gen / 2], [0, 1, -yuk / 2], [0, 0, 1]], np.float64)
    rt = np.column_stack([r[:, 0], r[:, 1], [0, 0, f]])
    k = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]], np.float64)
    hom = k @ rt @ merkez
    kose = cv2.perspectiveTransform(
        np.array([[[0, 0]], [[gen, 0]], [[gen, yuk]], [[0, yuk]]], np.float64), hom
    ).reshape(4, 2)
    x0, y0 = kose.min(0)
    x1, y1 = kose.max(0)
    pay = ayar.masa_payi * max(x1 - x0, y1 - y0)
    kaydir = np.array([[1, 0, pay - x0], [0, 1, pay - y0], [0, 0, 1]], np.float64)
    boyut = (math.ceil(x1 - x0 + 2 * pay), math.ceil(y1 - y0 + 2 * pay))
    return kaydir @ hom, boyut


def _masa(boyut: tuple[int, int], rng) -> np.ndarray:
    """Koyu, dokulu masa: düşük frekanslı leke + ince tahta damarı + gürültü."""
    gen, yuk = boyut
    taban = rng.uniform(70, 120)
    leke = cv2.resize(rng.normal(0, 1, (8, 8)), (gen, yuk), interpolation=cv2.INTER_CUBIC)
    damar = cv2.resize(
        rng.normal(0, 1, (yuk // 40 + 1, 1)), (gen, yuk), interpolation=cv2.INTER_LINEAR
    )
    return (taban + 12 * leke + 8 * damar).astype(np.float32)


def _isik_alani(boyut: tuple[int, int], ayar: FotoAyar, rng) -> np.ndarray:
    gen, yuk = boyut
    y, x = np.mgrid[0:yuk, 0:gen].astype(np.float32)
    a = rng.uniform(0, 2 * math.pi)
    t = (x * math.cos(a) + y * math.sin(a)) / (abs(gen * math.cos(a)) + abs(yuk * math.sin(a)))
    t = t - t.min()
    alan = 1 - ayar.isik * t / max(float(t.max()), 1e-6)
    cx, cy = rng.uniform(0.2, 0.8) * gen, rng.uniform(0.2, 0.8) * yuk
    s = 0.25 * max(gen, yuk)
    alan *= 1 - ayar.golge * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
    return alan


def _kirisiklar(gen: int, yuk: int, sayi: int, rng):
    """Kırışık doğruları: (nokta, birim normal) listesi, kâğıdı boydan geçer."""
    out = []
    for _ in range(sayi):
        p = np.array([rng.uniform(0.15, 0.85) * gen, rng.uniform(0.15, 0.85) * yuk])
        a = rng.uniform(0, math.pi)
        out.append((p, np.array([-math.sin(a), math.cos(a)])))
    return out


def _kayma_alani(gen: int, yuk: int, kirisiklar, rng) -> tuple[np.ndarray, np.ndarray]:
    """Sayfada (dx, dy): her kırışık bir yanı birkaç piksel iter (kırık), üstüne
    düşük frekanslı dalga. Açılmış kâğıdın düz olmayan yerel kayması."""
    y, x = np.mgrid[0:yuk, 0:gen].astype(np.float32)
    dx = np.zeros((yuk, gen), np.float32)
    dy = np.zeros((yuk, gen), np.float32)
    for p, nrm in kirisiklar:
        d = (x - p[0]) * nrm[0] + (y - p[1]) * nrm[1]
        genlik = rng.uniform(2, 5)
        adim = genlik * np.tanh(d / 30)
        dx += adim * nrm[0]
        dy += adim * nrm[1]
    for alan in (dx, dy):
        alan += 3 * cv2.resize(rng.normal(0, 1, (5, 5)), (gen, yuk), interpolation=cv2.INTER_CUBIC)
    return dx, dy


def _ileri_kayma(p: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Sayfa noktası q'nun kırışık sonrası yeri: p = q + D(p), sabit nokta
    yinelemesi (kayma küçük ve yumuşak, üç adım yeter)."""
    yuk, gen = dx.shape
    q = p.copy()
    for _ in range(3):
        xi = np.clip(np.round(p[:, 0]).astype(int), 0, gen - 1)
        yi = np.clip(np.round(p[:, 1]).astype(int), 0, yuk - 1)
        p = q + np.column_stack([dx[yi, xi], dy[yi, xi]])
    return p


def foto_uret(gri: np.ndarray, ayar: FotoAyar) -> Foto:
    """Taranmış sayfadan sentetik fotoğraf."""
    rng = np.random.default_rng(ayar.tohum)
    yuk, gen = gri.shape
    # Kâğıt beyazı 240, mürekkep 35: fotoğrafta tam siyah ve tam beyaz yok.
    sayfa = 240 - (255 - gri.astype(np.float32)) * (205 / 255)
    kayma = None
    if ayar.buruk:
        kir = _kirisiklar(gen, yuk, ayar.buruk, rng)
        kayma = _kayma_alani(gen, yuk, kir, rng)
        yy, xx = np.mgrid[0:yuk, 0:gen].astype(np.float32)
        sayfa = cv2.remap(
            sayfa, xx - kayma[0], yy - kayma[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        carpan = np.ones((yuk, gen), np.float32)
        for p, nrm in kir:
            d = (xx - p[0]) * nrm[0] + (yy - p[1]) * nrm[1]
            carpan *= 1 - rng.uniform(0.15, 0.35) * np.exp(-(d**2) / (2 * 1.2**2))  # çizgi
            carpan *= 1 - rng.uniform(0.03, 0.08) * (d > 0) * np.exp(-d / 60)  # bir yanda gölge
        sayfa *= carpan

    hom, boyut = _kamera_homografisi(gen, yuk, ayar, rng)
    kagit = cv2.warpPerspective(sayfa, hom, boyut, flags=cv2.INTER_LINEAR, borderValue=0)
    maske = cv2.warpPerspective(
        np.ones((yuk, gen), np.float32), hom, boyut, flags=cv2.INTER_LINEAR, borderValue=0
    )
    foto = kagit + _masa(boyut, rng) * (1 - maske)
    foto *= _isik_alani(boyut, ayar, rng)
    if ayar.bulanik > 0:
        foto = cv2.GaussianBlur(foto, (0, 0), ayar.bulanik)
    foto += rng.normal(0, ayar.gurultu, foto.shape).astype(np.float32)
    foto = np.clip(foto, 0, 255).astype(np.uint8)
    _, tampon = cv2.imencode(".jpg", foto, [cv2.IMWRITE_JPEG_QUALITY, ayar.jpeg])
    foto = cv2.imdecode(tampon, cv2.IMREAD_GRAYSCALE)
    kose = cv2.perspectiveTransform(
        np.array([[[0, 0]], [[gen, 0]], [[gen, yuk]], [[0, yuk]]], np.float64), hom
    ).reshape(4, 2)
    return Foto(foto, hom, kose, kayma)


def _sik_ornekle(noktalar: np.ndarray, adim: float = 10.0) -> np.ndarray:
    """Kapalı poligonun kenarlarına ``adim`` aralıkla nokta ekler (doğrusal
    olmayan dönüşümde kenar düz kalmaz)."""
    out = []
    n = len(noktalar)
    for i in range(n):
        a, b = noktalar[i], noktalar[(i + 1) % n]
        k = max(1, math.ceil(np.linalg.norm(b - a) / adim))
        out.extend(a + (b - a) * t for t in np.arange(k) / k)
    return np.array(out)


def cevap_tasi(c: Cevap, donustur, boyut: tuple[int, int], sik: bool = False) -> Cevap:
    """Cevabın bütün poligonlarını ``donustur`` ile taşır. ``boyut`` (gen, yuk)
    yeni görüntünün boyutu. ``sik``: doğrusal olmayan dönüşüm için kenarları
    sık örnekle."""
    nesneler = [
        Nesne(n.id, n.sinif, donustur(_sik_ornekle(n.noktalar) if sik else n.noktalar))
        for n in c.nesneler
    ]
    return replace(c, genislik=boyut[0], yukseklik=boyut[1], nesneler=nesneler)
