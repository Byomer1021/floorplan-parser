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

**Hafta 0: ortam, ön işleme ve duvar adımının ilk sürümü yazıldı; CVC-FP
indirildi ve geliştirme bölümünde ilk ölçümler alındı.** Bu sayfaya her
sayı ölçüldüğü anda girer. Test bölümüne henüz bakılmadı.

Şu ana kadar ölçülen:

- **Veri:** CVC-FP, 122 plan, dört çizim stili. Duvarı dolu siyah çizilmiş
  olanlar 86 plan (%70); kalanı taralı ya da çift çizgi
  ([olcumler §3](docs/olcumler.md)).
- **Sentetik plan:** duvar adımı 7 duvarın 7'sini buluyor, düz ve 7-60°
  döndürülmüş, uç hatası ≤ 2 px, kalınlık ±0.25 px
  ([olcumler §4](docs/olcumler.md)).
- **CVC-FP, dolu duvarlı 62 geliştirme planı, iki ölçekli duvar:** duvar
  maskesinin cevapla IoU'su medyan 0.926, Hough parçalarından çizilen maske
  0.880; plan başına 1.3 s. Mürekkep seçen bir yöntemin tavanı 0.95; tavana
  oran 0.977 ([olcumler §6](docs/olcumler.md)). Tek çekirdekle 0.84'tü
  ([§5](docs/olcumler.md)): ~7-8 px'lik bölme duvarlar merdiven gibi aynı
  kalınlıktaki çizimlerden ayrılamıyordu ([K-007](docs/kararlar.md)).
- **Kapı ve pencere** (aynı 62 plan): kapı kesinlik 0.972, duyarlılık
  0.760, F1 0.853; pencere F1 0.670 ([olcumler §8](docs/olcumler.md)).
  Kalan kapı hatalarının çoğu yarı açık çizilmiş kapılar.
- Eşikler geliştirme bölümünde seçildi; **test bölümünde henüz
  ölçülmedi.**
- **Bilinen yanlış:** merdivenin taban çizgisi gibi tek duran, duvar
  kalınlığında çizgi duvar sayılıyor; elemek için denenen kurallar daha
  çok gerçek duvar kaybettirdi ([§7](docs/olcumler.md)).
- **Kapsam dışı:** taralı ve çift çizgili stiller (36 plan); yöntem bunlarda
  duvar bulmuyor (IoU ~0).

## Akış

| # | Adım | Yöntem | Modül | Durum |
|---|---|---|---|---|
| 1 | Okuma | bayt + `imdecode` (Windows'ta Türkçe yol) | `goruntu` | yazıldı |
| 2 | İkileme | Otsu / uyarlamalı eşik | `onisleme` | yazıldı |
| 3 | Gürültü | açma / kapama | `onisleme` | yazıldı |
| 4 | Kalın/ince ayrımı | iki ölçek: kalın açma + kurala uyan ince parçalar | `onisleme`, `duvar` | yazıldı |
| 5 | Duvar | iskelet + olasılıksal Hough, eşdoğrusal birleştirme, kavşağa çekme | `duvar` | ilk sürüm |
| 6 | Kapı | duvar ucundan ışınla boşluk + 4 varsayımlı çeyrek yay testi | `aciklik` | yazıldı |
| 7 | Pencere | boşlukta paralel ince çizgiler; küçük yay kanatlı pencere | `aciklik` | yazıldı |
| 8 | Graf | odalar düğüm, kapılar kenar (networkx) | | yazılmadı |
| 9 | Değerlendirme | duvar piksel IoU; kapı/pencere poligon IoU ≥ 0.5 eşleme | `degerlendirme`, `cvcfp` | duvar, kapı, pencere |

`sentetik` modülü cevabı bilinen plan çizer; her adımın testi önce orada
geçer.

## Kısıt: öğrenilmiş model yok

YOLO, CNN, segmentasyon ağı, önceden eğitilmiş ağırlık kullanılmıyor
([K-002](docs/kararlar.md)). `tests/test_ortam.py`, `uv.lock`'ta torch,
tensorflow, ultralytics vb. bulursa düşer.

## Kurulum

```
uv sync
uv run python scripts/cvcfp_indir.py   # veri seti, ~27 MB
uv run pytest
uv run python scripts/duvar_deney.py --grup I-II --kalinlik 13 --yontem iki
uv run python scripts/aciklik_deney.py
```

Python 3.12, bağımlılıklar `uv.lock`'ta sabit.

## Düzen

```
src/floorplan_parser/   kod, adım başına bir modül
scripts/                veri indirme, envanter, deneyler
tests/                  sentetik plan ve CVC-FP okuyucu testleri
data/                   veri seti (depoya girmez, bkz. data/README.md)
docs/kararlar.md        tasarım kararları, K-NNN
docs/olcumler.md        numaralı ölçüm kaydı
docs/gunluk/            günlük çalışma kaydı
results/                deney CSV'leri; results/ara/ görseller (depoya girmez)
```
