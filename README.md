# floorplan-parser

**Raster kat planlarından duvarları, kapıları ve pencereleri öğrenilmiş
model kullanmadan, yalnız klasik görüntü işlemeyle çıkaran kural tabanlı bir
çözümleyici. Çıktı, oda/duvar/kapı ilişkilerini taşıyan bir graf.**

*A rule-based parser that extracts walls, doors and windows from raster floor
plans with classical image processing only (thresholding, morphology, Hough
transforms, template matching), no learned models, and outputs a
room/wall/door graph.*

---

## Durum

GTÜ, Görüntü İşleme dersi projesi (2026 güz). Ders kodu, takvim ve takım
henüz kayıtlı değil.

**Hafta 0: ortam kuruldu, ön işleme ve cevabı bilinen sentetik test planı
yazıldı. Gerçek bir plan üzerinde henüz hiçbir şey ölçülmedi.** Bu sayfaya
her sayı ölçüldüğü anda girer.

Şu ana kadar ölçülen, yalnız sentetik planda ([olcumler §2](docs/olcumler.md)):

- 5x5 açmayla kalın/ince ayrımı temiz planda duvarı ince çizgilerden
  ayırıyor: duvar IoU 0.9991 (16 fazla piksel, eksik yok); kapı yayı,
  kanadı ve pencere çizgilerinin 1338/1354 pikseli ince tarafta.
- Gürültülü planda (%1 piksel ters) ön işleme sırası bir ödünleşim:
  önce açma yapılırsa duvardaki 161-207 delik 626-911 eksik piksele
  büyüyor; önce kapama yapılırsa eksik kalmıyor ama pencere duvara
  dönüşüyor (~1220 fazla piksel). Karar gerçek veriye bakılarak verilecek
  ([K-004](docs/kararlar.md)).

## Akış

| # | Adım | Yöntem | Modül | Durum |
|---|---|---|---|---|
| 1 | Okuma | bayt + `imdecode` (Windows'ta Türkçe yol) | `goruntu` | yazıldı |
| 2 | İkileme | Otsu / uyarlamalı eşik | `onisleme` | yazıldı |
| 3 | Gürültü | açma / kapama | `onisleme` | yazıldı |
| 4 | Kalın/ince ayrımı | duvar kalınlığına yakın çekirdekle açma | `onisleme` | yazıldı |
| 5 | Duvar | olasılıksal Hough / LSD, eşdoğrusal parçaları birleştirme | | yazılmadı |
| 6 | Kapı | duvardaki kesinti + yay (Hough çember / şablon eşleme) | | yazılmadı |
| 7 | Pencere | duvar boşluğunda paralel ince çizgiler | | yazılmadı |
| 8 | Graf | odalar düğüm, kapılar kenar (networkx) | | yazılmadı |
| 9 | Değerlendirme | veri setinin SVG cevabıyla piksel ve nesne düzeyinde | | yazılmadı |

`sentetik` modülü cevabı bilinen plan çizer; her adımın testi önce orada
geçer.

## Kısıt: öğrenilmiş model yok

YOLO, CNN, segmentasyon ağı, önceden eğitilmiş ağırlık kullanılmıyor
([K-002](docs/kararlar.md)). `tests/test_ortam.py`, `uv.lock`'ta torch,
tensorflow, ultralytics vb. bulursa düşer.

## Kurulum

```
uv sync
uv run pytest
```

Python 3.12, bağımlılıklar `uv.lock`'ta sabit.

## Düzen

```
src/floorplan_parser/   kod, adım başına bir modül
tests/                  sentetik plan üzerinde testler
data/                   veri seti (depoya girmez, bkz. data/README.md)
docs/kararlar.md        tasarım kararları, K-NNN
docs/olcumler.md        numaralı ölçüm kaydı
docs/gunluk/            günlük çalışma kaydı
results/ara/            yeniden üretilebilen ara çıktılar (depoya girmez)
```
