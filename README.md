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

**Yöntem donduruldu ve test bölümünde bir kez ölçüldü (2026-10-18, kod
`ec5a90b`).** Bu sayfaya her sayı ölçüldüğü anda girer.

**Test sonucu** (CVC-FP, dolu duvarlı 24 test planı; eşikler geliştirmede
seçildi, testten sonra ayar yok; [olcumler §17](docs/olcumler.md)):

| | kesinlik | duyarlılık | F1 / IoU |
|---|---|---|---|
| duvar (piksel IoU, plan ortalaması) | | | 0.913 |
| kapı | 0.985 | 0.835 | 0.904 |
| pencere | 0.863 | 0.798 | 0.830 |
| oda | 0.825 | 0.722 | 0.770 |
| oda-kapı-oda kenarı | 0.580 | 0.441 | 0.502 |

Bütün stillerde (36 test planı): kapı F1 0.770, pencere 0.625, oda 0.651,
kenar 0.428; taralı ve çift çizgili planlarda duvar bulunmuyor. Kenar
testte geliştirmeden (0.392) belirgin yüksek; örneklem küçük ve kenar,
koridorun bulunmasına çok duyarlı.

Geliştirmede ölçülen (yöntem seçimi bunlarla yapıldı):

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
  kalınlıktaki çizimlerden ayrılamıyordu ([K-007](docs/kararlar.md)). Eğik
  ince duvarlar (kare yapı elemanı siliyordu) eklendikten sonra medyan
  0.928 / ortalama 0.909 ([§16](docs/olcumler.md)).
- **Kapı ve pencere** (aynı 62 plan): kapı kesinlik 0.957, duyarlılık
  0.836, F1 0.892 ([olcumler §8, §15-16](docs/olcumler.md)); pencere
  kesinlik 0.882, duyarlılık 0.798, F1 0.838 ([§13](docs/olcumler.md); çift
  kanatlı ve geniş pencereler eklenmeden önce 0.671). Kalan kapı hatalarının
  çoğu yarı açık çizilmiş kapılar.
- **Oda grafı** (aynı 62 plan): oda F1 0.783 (kesinlik 0.859, duyarlılık
  0.719), kapı kenarı F1 0.392 ([olcumler §9-11, §14-16](docs/olcumler.md)).
  Odalar uzaklık dönüşümü + watershed ile bölünüyor. Kaçan odaların en
  büyük payı duvarsız ayrımı olan komşuyla birleşenler (bu yöntemle
  bulunamaz); ikincisi kapısı bulunamayan komşuyla birleşenler. Kenar için
  iki odanın ikisi de bulunmalı.
- Uçtan uca `akis.plani_coz`; plan başına medyan 6.2 s.
- Eşikler geliştirme bölümünde seçildi; test bölümünde bir kez ölçüldü
  (yukarıda). Duvar, kapı, pencere ve oda geliştirmeyle ±0.015 içinde.
- **Bilinen yanlış:** merdivenin taban çizgisi gibi tek duran, duvar
  kalınlığında çizgi duvar sayılıyor; elemek için denenen kurallar daha
  çok gerçek duvar kaybettirdi ([§7](docs/olcumler.md)).
- **Kapsam dışı:** taralı ve çift çizgili stiller (36 plan); yöntem bunlarda
  duvar bulmuyor (IoU ~0). Bütün stillerde (86 geliştirme planı) duvar IoU
  ortalaması 0.667, kapı F1 0.730, pencere F1 0.652, oda F1 0.647
  ([olcumler §12-16](docs/olcumler.md)).
- **Literatür:** dolu duvarlı planlarda bulunan tek klasik duvar sayısının
  üstünde (0.908 / 0.831). Bütün stillerde öğrenilmiş yöntemlerin
  (0.77-0.89) çok altında. Kapı (kutu IoU) dolu planlarda K 0.963 / D
  0.841, FloorYOLO bütün stillerde 0.810 / 0.786; pencere altında (kutu
  IoU'lu F1 0.675 / ~0.76). Kurulum farklarıyla birlikte:
  [docs/literatur.md](docs/literatur.md).

## Akış

| # | Adım | Yöntem | Modül | Durum |
|---|---|---|---|---|
| 1 | Okuma | bayt + `imdecode` (Windows'ta Türkçe yol) | `goruntu` | yazıldı |
| 2 | İkileme | Otsu / uyarlamalı eşik | `onisleme` | yazıldı |
| 3 | Gürültü | açma / kapama | `onisleme` | yazıldı |
| 4 | Kalın/ince ayrımı | iki ölçek: kalın açma + kurala uyan ince parçalar; eğik ince duvarlar disk açmanın fazlasından | `onisleme`, `duvar` | yazıldı |
| 5 | Duvar | iskelet + olasılıksal Hough, eşdoğrusal birleştirme, kavşağa çekme | `duvar` | ilk sürüm |
| 6 | Kapı | duvar ucundan ışınla boşluk + 4 varsayımlı çeyrek yay testi | `aciklik` | yazıldı |
| 7 | Pencere | boşlukta paralel ince çizgiler; küçük yay kanatlı, iki yarım yay çift kanatlı pencere; 150-450 px geniş boşluklar da | `aciklik` | yazıldı |
| 8 | Graf | boşlukları kirişle kapat (geniş açıklıklar ve yarı açık kapılar dahil; kesik duvar hattındaki uçtan kiriş yok) + ayak izi + uzaklık dönüşümü ve watershed; kapı iki yandaki odayı bağlar | `oda`, `akis` | yazıldı |
| 9 | Değerlendirme | duvar piksel IoU; kapı/pencere/oda IoU ≥ 0.5 eşleme; kenar = kapı + iki oda | `degerlendirme`, `cvcfp` | hepsi |

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
uv run python scripts/oda_deney.py
uv run python scripts/degerlendir.py --bolum gelistirme   # uçtan uca
uv run python scripts/demo.py --cvcfp IIa_AP3301        # demo: katmanlı görüntü + graf JSON
uv run python scripts/demo.py yol/plan.png              # herhangi bir plan görüntüsü
uv run python scripts/sekiller.py                       # rapor şekilleri → rapor/sekiller/
```

Rapor (IEEE, İngilizce): [rapor/](rapor/), derleme notu
[rapor/README.md](rapor/README.md).

Python 3.12, bağımlılıklar `uv.lock`'ta sabit.

## Düzen

```
src/floorplan_parser/   kod, adım başına bir modül
scripts/                veri indirme, envanter, deneyler
tests/                  sentetik plan ve CVC-FP okuyucu testleri
data/                   veri seti (depoya girmez, bkz. data/README.md)
docs/kararlar.md        tasarım kararları, K-NNN
docs/olcumler.md        numaralı ölçüm kaydı
docs/literatur.md       yayımlanmış sonuçlarla kıyas
docs/gunluk/            günlük çalışma kaydı
results/                deney CSV'leri; results/ara/ görseller (depoya girmez)
```
