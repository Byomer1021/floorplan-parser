# Ölçümler

Numaralı ölçüm kaydı. Her bölüm: ne ölçüldü, nasıl, sayı, ne anlama
geliyor. Sayı değişirse eski satır silinmez, yeni bölüm açılır.

---

## §1 Ortam (2026-10-17)

- Python 3.12.10, uv 0.11.0. Kilitte 53 paket (proje dahil).
- opencv-python 5.0.0, numpy 2.5.4, scikit-image 0.26.0, scipy 1.18.1,
  networkx 3.7, matplotlib 3.11.2.
- `uv sync`: 42 s (önbellekte paketlerin bir kısmı vardı). `.venv` 460 MB.
- **OpenCV 5.0.0, Windows, ASCII olmayan yol:** `cv2.imwrite` False,
  `cv2.imread` None döndürüyor, hata vermiyor. `goruntu.oku/yaz` bu yüzden
  bayt okuyup `imdecode/imencode` kullanıyor.

## §2 Ön işleme, sentetik plan (2026-10-17)

Plan: 600x400, 10 piksel duvar, 2 piksel ince çizgi, bir kapı (60 piksel
boşluk, kanat ve çeyrek yay), bir pencere (100 piksel, üç paralel çizgi).
Duvar maskesi 17300 piksel. Gürültü: her piksel %1 olasılıkla ters.

**Temiz plan, `kalin_ayir(ikile(g), 5)`:**

| | değer |
|---|---|
| kalın vs duvar IoU | 0.9991 |
| fazla (duvar dışı kalın) | 16 piksel |
| eksik (kalın olmayan duvar) | 0 |
| ince çizgilerin ince tarafta kalan payı | 1338 / 1354 |

16 fazla piksel ince çizginin duvara değdiği yerlerde.

**Gürültülü plan, 5 tohum:**

| tohum | duvardaki delik | önce açma: eksik | önce açma: fazla | önce 3x3 kapama: eksik | önce 3x3 kapama: fazla |
|---|---|---|---|---|---|
| 0 | 207 | 911 | 16 | 0 | 1237 |
| 1 | 176 | 626 | 16 | 0 | 1219 |
| 2 | 177 | 684 | 17 | 0 | 1229 |
| 3 | 184 | 707 | 16 | 0 | 1219 |
| 4 | 161 | 643 | 16 | 0 | 1220 |

- Önce açma: her delik çevresiyle birlikte gidiyor, eksik delik sayısının
  3.6-4.4 katı.
- Önce kapama: duvar tam, ama pencere bölgesi (100x12 ≈ 1200 piksel) duvar
  olarak çıkıyor. Temiz planda da aynı: pencere bölgesinde kalın oranı
  kapama olmadan 0, 3x3 kapamayla 1.
- 3x3 açma arka plan gürültüsünü (5 tohumda 2129-2225 piksel) siliyor;
  kalan 0-2 piksel ince çizgiye yapışık, çizimden 2 pikselden uzakta 0.

Karar: [K-004](kararlar.md).
