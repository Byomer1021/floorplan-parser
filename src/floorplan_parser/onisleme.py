"""Ön işleme: gri → ikili → morfolojik temizlik → kalın/ince ayrımı.

İkili görüntüde ön plan (255) mürekkep, arka plan (0) kâğıt.

Çekirdek boyutu iki sınır arasında seçilir:

- Açma, çekirdekten ince her şeyi siler. Tuz-biber gürültüsü için 3 yeter;
  duvar kalınlığına yakın bir çekirdek ise yalnız duvarları bırakır
  (``kalin_ayir``). Kapı yayı, pencere çizgisi, yazı ve ölçü çizgisi incedir.
- Kapama, çekirdekten dar boşlukları doldurur. Kapı boşluğu da bir boşluk:
  çekirdek en dar kapıdan genişse kapı kapanır ve duvar kesintisiz görünür.

Sıra önemli (sentetik planda ölçüldü, docs/olcumler.md §2): gürültülü planda
``kalin_ayir`` önce gelirse duvardaki her delik çevresini de götürür; kapama
önce gelirse duvar tam çıkar ama pencerenin paralel ince çizgileri birleşip
duvar olur.
"""

import cv2
import numpy as np


def ikile(gri: np.ndarray, yontem: str = "otsu", blok: int = 51, c: int = 10) -> np.ndarray:
    """Mürekkebi 255, kâğıdı 0 yapar.

    ``otsu`` tek eşik; taranmış, aydınlatması düzensiz planda ``uyarlamali``
    (``blok`` piksellik pencerede yerel ortalamanın ``c`` altı) daha güvenli.
    """
    if gri.ndim != 2:
        raise ValueError("gri tonlu görüntü bekleniyor")
    if yontem == "otsu":
        _, ikili = cv2.threshold(gri, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif yontem == "uyarlamali":
        ikili = cv2.adaptiveThreshold(
            gri, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blok, c
        )
    else:
        raise ValueError(f"bilinmeyen yöntem: {yontem}")
    return ikili


def _kare(boyut: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (boyut, boyut))


def ac(ikili: np.ndarray, boyut: int) -> np.ndarray:
    return cv2.morphologyEx(ikili, cv2.MORPH_OPEN, _kare(boyut))


def ac_yonsuz(ikili: np.ndarray, boyut: int) -> np.ndarray:
    """Kare ve disk açmasının birleşimi. Kare yöne duyarlı: 45° eğik bir bant
    ancak kalınlığı boyut × √2 ise kalır. Disk her yönde aynı; karenin içinde
    olduğundan birleşim karenin tuttuğu her pikseli tutar (köşeler dahil),
    fazladan yalnız karenin sığmadığı eğik bantlar gelir."""
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boyut, boyut))
    return cv2.max(ac(ikili, boyut), cv2.morphologyEx(ikili, cv2.MORPH_OPEN, disk))


def kapa(ikili: np.ndarray, boyut: int) -> np.ndarray:
    return cv2.morphologyEx(ikili, cv2.MORPH_CLOSE, _kare(boyut))


def delik_doldur(ikili: np.ndarray, max_alan: int) -> np.ndarray:
    """Dört yanı mürekkeple çevrili, alanı en fazla ``max_alan`` piksel olan
    kâğıt bölgelerini doldurur (duvar içindeki tarama gürültüsü).

    Yapısal kapalı bölgeler büyük kalmalı: odalar, kapı yayının içi ve
    pencerenin paralel çizgileri arasındaki şerit (sentetik planda 100+).
    """
    kagit = (ikili == 0).astype(np.uint8)
    _, etiket, istat, _ = cv2.connectedComponentsWithStats(kagit, connectivity=4)
    h, w = ikili.shape
    x, y, gen, yuk, alan = istat.T
    kenara_deger = (x == 0) | (y == 0) | (x + gen == w) | (y + yuk == h)
    doldur = (alan <= max_alan) & ~kenara_deger
    doldur[0] = False  # 0 etiketi mürekkep
    sonuc = ikili.copy()
    sonuc[doldur[etiket]] = 255
    return sonuc


def kalin_ayir(ikili: np.ndarray, kalinlik: int) -> tuple[np.ndarray, np.ndarray]:
    """``kalinlik`` piksellik kareyle açma: (kalın, ince).

    Kalın kısım duvar adayı, ince kısım kapı/pencere/yazı adayı. Kalın + ince
    = ikili. İnce çizginin kalın duvara değdiği köşelerde birkaç piksel kalın
    tarafa geçebilir.
    """
    kalin = ac(ikili, kalinlik)
    return kalin, cv2.subtract(ikili, kalin)


def duvar_adayi(gri: np.ndarray, kalinlik: int, max_delik: int = 4) -> np.ndarray:
    """Varsayılan akış: ikile → küçük delikleri doldur → kalın kısım (K-004)."""
    return kalin_ayir(delik_doldur(ikile(gri), max_delik), kalinlik)[0]
