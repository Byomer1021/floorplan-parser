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


def kapa(ikili: np.ndarray, boyut: int) -> np.ndarray:
    return cv2.morphologyEx(ikili, cv2.MORPH_CLOSE, _kare(boyut))


def kalin_ayir(ikili: np.ndarray, kalinlik: int) -> tuple[np.ndarray, np.ndarray]:
    """``kalinlik`` piksellik kareyle açma: (kalın, ince).

    Kalın kısım duvar adayı, ince kısım kapı/pencere/yazı adayı. Kalın + ince
    = ikili. İnce çizginin kalın duvara değdiği köşelerde birkaç piksel kalın
    tarafa geçebilir.
    """
    kalin = ac(ikili, kalinlik)
    return kalin, cv2.subtract(ikili, kalin)
