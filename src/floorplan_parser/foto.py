"""Fotoğraftan sayfaya: kâğıdı bul, perspektifi düzelt, ışığı düzelt.

Masadaki bir kat planının açılı telefon fotoğrafı, akışın beklediği taranmış
sayfaya çevrilir (kararlar K-010). Kâğıt düz bir yüzey olduğu için açılı
fotoğrafı tek bir homografi düzeltir. Bütün adımlar klasik: eşikleme,
morfoloji, kontur, doğru uydurma, homografi. Öğrenilmiş model yok (K-002).

Varsayım: kâğıt masadan açık renkli ve dört köşesi fotoğrafın içinde. Kâğıt
boyutu A4 kabul edilir ve sayfa 300 dpi'a (CVC-FP taramalarının ölçeği)
düzeltilir. Başka boyutta kâğıtta ölçek ayrıca kestirilmeli.
"""

import math

import cv2
import numpy as np

from floorplan_parser.akis import PlanSonucu, plani_coz

A4_300DPI = (2480, 3508)  # kısa, uzun kenar (piksel)
_KUCUK = 1000  # kâğıt aramada görüntünün uzun kenarı


def kose_sirala(kose: np.ndarray) -> np.ndarray:
    """Dört noktayı sol üst, sağ üst, sağ alt, sol alt sırasına koyar
    (kâğıt 45 dereceden az dönmüşse doğru)."""
    kose = np.asarray(kose, np.float64).reshape(4, 2)
    toplam, fark = kose.sum(1), kose[:, 1] - kose[:, 0]
    return kose[[np.argmin(toplam), np.argmin(fark), np.argmax(toplam), np.argmax(fark)]]


def _dogru_kesisimi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(vx, vy, x0, y0) biçimindeki iki doğrunun kesişimi."""
    (ax, ay, ax0, ay0), (bx, by, bx0, by0) = a, b
    m = np.array([[ax, -bx], [ay, -by]])
    t, _ = np.linalg.solve(m, [bx0 - ax0, by0 - ay0])
    return np.array([ax0 + t * ax, ay0 + t * ay])


def _parlak_bolge(kucuk: np.ndarray) -> np.ndarray:
    """Otsu ile açık bölge. Çizim çizgileri kapamayla kapanır."""
    _, acik = cv2.threshold(kucuk, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(acik, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def _kenarla_kapali_bolge(kucuk: np.ndarray) -> np.ndarray:
    """Kenarlarla çevrili bölge: Canny kenarları kalınlaştırılır, görüntünün
    çerçevesinden kenara çarpmadan ulaşılabilen her yer masa sayılır, gerisi
    kâğıt. Masa bir yerde kâğıttan parlaksa Otsu kâğıdı masaya sızdırır. Kâğıdın
    kenarı ise ışıktan bağımsız güçlü bir basamak."""
    kenar = cv2.Canny(kucuk, 20, 60)
    kenar = cv2.dilate(kenar, np.ones((5, 5), np.uint8))
    _, etiket = cv2.connectedComponents((kenar == 0).astype(np.uint8), connectivity=4)
    cerceve = np.unique(np.concatenate([etiket[0], etiket[-1], etiket[:, 0], etiket[:, -1]]))
    bolge = ~np.isin(etiket, cerceve)
    return cv2.morphologyEx(bolge.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def _en_buyuk_ic_bilesen(maske: np.ndarray) -> np.ndarray | None:
    """Çerçeveye değmeyen, görüntünün en az %10'u olan en büyük bileşen.
    Kâğıdın dört köşesi fotoğrafın içinde varsayılıyor. Çerçeveye değen bileşen
    masaya sızmıştır."""
    n, etiket, istat, _ = cv2.connectedComponentsWithStats(maske, connectivity=4)
    h, w = maske.shape
    en_iyi, en_iyi_alan = None, 0.1 * maske.size
    for i in range(1, n):
        x, y, gen, yuk, alan = istat[i]
        if x == 0 or y == 0 or x + gen == w or y + yuk == h:
            continue
        if alan > en_iyi_alan:
            en_iyi, en_iyi_alan = i, alan
    return None if en_iyi is None else (etiket == en_iyi).astype(np.uint8)


def kagit_bul(gri: np.ndarray) -> np.ndarray | None:
    """Kâğıdın dört köşesi (sol üst, sağ üst, sağ alt, sol alt, x y), yoksa None.

    Küçültülmüş ve bulanıklaştırılmış görüntüde kâğıt bölgesi (önce parlaklık,
    olmazsa kenar), en büyük bileşenin dışbükey zarfından dörtgen. Köşeler,
    her kenara yakın kontur noktalarına uydurulan doğruların kesişimiyle, sonra
    tam çözünürlükte iyileştirilir.
    """
    h, w = gri.shape
    k = _KUCUK / max(h, w)
    kucuk = cv2.resize(gri, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    kucuk = cv2.GaussianBlur(kucuk, (0, 0), 2)
    for bolge in (_parlak_bolge, _kenarla_kapali_bolge):
        kose = _dortgen(_en_buyuk_ic_bilesen(bolge(kucuk)))
        if kose is not None:
            return _kose_iyilestir(gri, kose / k)
    return None


def _dortgen(maske: np.ndarray | None) -> np.ndarray | None:
    """Bileşenin dörtgeni: dışbükey zarfa dört köşeli yaklaşım, sonra her
    kenarın kontur noktalarına doğru uydurup kesiştirme (yaklaşımın köşesi
    kontur noktasına takılı, doğru kesişimi değil)."""
    if maske is None:
        return None
    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    kontur = max(konturlar, key=cv2.contourArea)
    zarf = cv2.convexHull(kontur)
    cevre = cv2.arcLength(zarf, True)
    for oran in (0.01, 0.02, 0.03, 0.05, 0.08):
        yaklasik = cv2.approxPolyDP(zarf, oran * cevre, True)
        if len(yaklasik) == 4:
            break
    else:
        return None
    kaba = kose_sirala(yaklasik.reshape(4, 2))

    # Her kenar: iki köşesinin arasında, kenara 3 pikselden yakın kontur
    # noktaları (köşelerin çevresi hariç, köşe yuvarlanmış olabilir).
    noktalar = kontur.reshape(-1, 2).astype(np.float64)
    dogrular = []
    for i in range(4):
        p, q = kaba[i], kaba[(i + 1) % 4]
        yon = (q - p) / np.linalg.norm(q - p)
        normal = np.array([-yon[1], yon[0]])
        t = (noktalar - p) @ yon
        uzunluk = np.linalg.norm(q - p)
        secili = noktalar[
            (np.abs((noktalar - p) @ normal) < 3) & (t > 0.1 * uzunluk) & (t < 0.9 * uzunluk)
        ]
        if len(secili) < 10:
            return kaba
        dogrular.append(
            cv2.fitLine(secili.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        )
    return np.array([_dogru_kesisimi(dogrular[i - 1], dogrular[i]) for i in range(4)])


def _kose_iyilestir(gri: np.ndarray, kose: np.ndarray, yarim: float = 20.0) -> np.ndarray:
    """Köşeleri tam çözünürlükte iyileştirir: her kenar boyunca 60 noktada
    kenara dik parlaklık kesiti alınır, en dik geçiş (kâğıt → masa) kenar
    noktasıdır. Bu noktalara doğru uydurulup komşu doğrular kesiştirilir."""
    g = cv2.GaussianBlur(gri, (0, 0), 1.5).astype(np.float32)
    kayma = np.arange(-yarim, yarim + 0.25, 0.5, dtype=np.float32)
    dogrular = []
    for i in range(4):
        p, q = kose[i], kose[(i + 1) % 4]
        yon = (q - p) / np.linalg.norm(q - p)
        normal = np.array([-yon[1], yon[0]])
        merkez = p + np.linspace(0.1, 0.9, 60)[:, None] * (q - p)
        xs = (merkez[:, 0:1] + kayma[None] * normal[0]).astype(np.float32)
        ys = (merkez[:, 1:2] + kayma[None] * normal[1]).astype(np.float32)
        kesit = cv2.remap(g, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        egim = np.abs(np.diff(kesit, axis=1))
        j = egim.argmax(1)
        ofset = (kayma[j] + kayma[j + 1]) / 2
        noktalar = merkez + ofset[:, None] * normal
        dogrular.append(
            cv2.fitLine(noktalar.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        )
    return np.array([_dogru_kesisimi(dogrular[i - 1], dogrular[i]) for i in range(4)])


def duzelt(
    gri: np.ndarray, kose: np.ndarray, boyut: tuple[int, int] = A4_300DPI
) -> tuple[np.ndarray, np.ndarray]:
    """Kâğıdı tam yukarıdan bakışa çevirir: (sayfa, H). H fotoğraf
    koordinatını sayfa koordinatına taşır. ``boyut`` kısa ve uzun kenar.
    yatay mı dikey mi, fotoğraftaki kenar uzunluklarından seçilir."""
    kose = kose_sirala(kose)
    yatay = (np.linalg.norm(kose[1] - kose[0]) + np.linalg.norm(kose[2] - kose[3])) / 2
    dikey = (np.linalg.norm(kose[3] - kose[0]) + np.linalg.norm(kose[2] - kose[1])) / 2
    kisa, uzun = boyut
    gen, yuk = (uzun, kisa) if yatay > dikey else (kisa, uzun)
    hedef = np.array([[0, 0], [gen, 0], [gen, yuk], [0, yuk]], np.float64)
    hom = cv2.getPerspectiveTransform(kose.astype(np.float32), hedef.astype(np.float32))
    sayfa = cv2.warpPerspective(
        gri, hom, (gen, yuk), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return sayfa, hom


def isik_duzelt(gri: np.ndarray, cekirdek: int = 61) -> np.ndarray:
    """Düzensiz ışığı ve gölgeyi giderir (flat-field düzeltme).

    Kâğıdın parlaklığı, mürekkebi silen bir kapamayla (çekirdek duvar
    kalınlığından büyük) kestirilip yumuşatılır, görüntü buna bölünür. Hız için
    dörtte bir ölçekte."""
    k = 4
    kucuk = cv2.resize(gri, None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA)
    c = max(3, cekirdek // k) | 1
    arka = cv2.morphologyEx(kucuk, cv2.MORPH_CLOSE, np.ones((c, c), np.uint8))
    arka = cv2.GaussianBlur(arka, (0, 0), c / 2)
    arka = cv2.resize(arka, (gri.shape[1], gri.shape[0]), interpolation=cv2.INTER_LINEAR)
    duz = gri.astype(np.float32) / np.maximum(arka.astype(np.float32), 1) * 255
    return np.clip(duz, 0, 255).astype(np.uint8)


def fotograftan_sayfa(
    gri: np.ndarray, kose: np.ndarray | None = None, isik: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Fotoğraf → akışa verilecek sayfa: (sayfa, H). ``kose`` verilmezse
    kâğıt aranır. Bulunamazsa ValueError (arayüz köşeleri kullanıcıya sorar)."""
    if kose is None:
        kose = kagit_bul(gri)
        if kose is None:
            raise ValueError("kâğıdın dört köşesi bulunamadı")
    sayfa, hom = duzelt(gri, kose)
    if isik:
        sayfa = isik_duzelt(sayfa)
    return sayfa, hom


def foto_ikile(sayfa: np.ndarray, fark: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Fotoğraf sayfasını ikiler: (akışa verilecek sayfa, sıkı mürekkep).

    Taramada 1-2 piksellik ince çizgiler (kapı yayı, pencere çizgileri)
    fotoğrafta bulanıklaşıp açık griye döner. Otsu eşiği bunları kesik kesik
    bırakır, pencere ve kapı bulunamaz (olcumler §18). Burada kâğıdın
    parlaklığından ``fark`` gri seviye koyu olan her piksel mürekkep: ince
    çizgi bütün kalır. Bedeli, kalın çizginin kenarındaki bulanıklık payının
    da mürekkep sayılması. Çıktıdaki duvar maskesi bu yüzden Otsu'nun sıkı
    mürekkebiyle kırpılır (``fotograftan_coz``).

    Dönen sayfa 0 (mürekkep) ve 255 (kâğıt). Akışın Otsu'su bunu olduğu gibi
    bırakır. Sayfa önce ``isik_duzelt``'ten geçmiş olmalı (kâğıt düz beyaz)."""
    kagit = float(np.median(sayfa))
    duz = np.where(sayfa < kagit - fark, 0, 255).astype(np.uint8)
    _, siki = cv2.threshold(sayfa, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return duz, siki


def fotograftan_coz(
    gri: np.ndarray, kose: np.ndarray | None = None, fark: int = 50
) -> tuple[PlanSonucu, np.ndarray, np.ndarray]:
    """Fotoğraf → (PlanSonucu, sayfa, H). Sonuç sayfa koordinatında, H
    fotoğraf koordinatını sayfaya taşır."""
    sayfa, hom = fotograftan_sayfa(gri, kose)
    duz, siki = foto_ikile(sayfa, fark)
    s = plani_coz(duz)
    s.duvar_maske = cv2.bitwise_and(s.duvar_maske, siki)
    return s, sayfa, hom


def homografi_uygula(hom: np.ndarray, noktalar: np.ndarray) -> np.ndarray:
    """(n, 2) noktaya homografi."""
    p = np.asarray(noktalar, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, hom).reshape(-1, 2)


def kose_hatasi(kose: np.ndarray, gercek: np.ndarray) -> float:
    """İki köşe dörtlüsü arasındaki en büyük uzaklık (piksel)."""
    return float(math.sqrt(((kose_sirala(kose) - kose_sirala(gercek)) ** 2).sum(1).max()))
