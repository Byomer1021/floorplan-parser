# Kararlar

Her tasarım kararı bir kayıt: ne seçildi, alternatifler, gerekçe, neyle
değişir. Değişen karar silinmez, üstü çizilir ve yeni kayda bağlanır.

Durum: **geçerli**, **açık** (karar bekliyor), **öneri** (onay bekliyor),
**değişti** (→ K-NNN).

---

## K-001: Proje konumu ve depo, geçerli (2026-10-17)

- Yerel: `D:\floorplan-parser`. C'de 17 GB boş, D'de 580 GB.
- Uzak: `github.com/Byomer1021/floorplan-parser`, **private**. Public'e
  geçiş veri seti lisansına bakıldıktan sonra (K-003).
- Ömer Can'ın makinesinde uv önbelleği D'de: `UV_CACHE_DIR=D:\.cache\uv`.
  Projeye yazılmadı, makineye özel.

## K-002: Öğrenilmiş model yok, geçerli (2026-10-17)

Dersin şartı: YOLO ve benzerleri kullanılmayacak.

**Kapsam dışı:** derin öğrenme (CNN, YOLO, Mask R-CNN, U-Net, SAM vb.),
önceden eğitilmiş her türlü ağırlık, veriden parametre öğrenen
sınıflandırıcılar.

**Kullanılan:** eşikleme, morfoloji, bağlı bileşen, iskelet çıkarma, kenar
bulma, Hough (çizgi ve çember), LSD, şablon eşleme, RANSAC gibi sağlam
kestirim. OpenCV, scikit-image, scipy.

**Gri bölge, hocaya sorulacak:**

- Çizgileri k-means / DBSCAN gibi kümelemeyle gruplamak.
- Eşikleri küçük bir doğrulama kümesinde ızgara aramasıyla seçmek.

**Uygulama:** `tests/test_ortam.py` kilit dosyasında derin öğrenme
kütüphanesi bulursa düşer.

## K-003: Veri seti: CVC-FP, geçerli (2026-10-17)

**2026-10-17: CVC-FP alındı** (Ömer Can onayıyla; gerekirse başka set
sonradan eklenir). İndirme, sha256, lisans ve içerik:
[data/README.md](../data/README.md). Lisans cümlesi sayfada çelişkili
(BY / BY-NC); ticari olmayan kullanım varsayılıyor. Dört stilin ölçümü
[olcumler §3](olcumler.md).

Seçim öncesi değerlendirme:

| | ne | cevap | not |
|---|---|---|---|
| **CVC-FP** | 122 taranmış plan, 4 alt küme (köken ve çizim stili farklı) | duvar, kapı, pencere, oda, oda ayırıcı; SVG | araştırma topluluğuna serbest bırakıldı (de las Heras vd., 2015); sayfada lisans metni görülmedi |
| CubiCasa5K | 5000 plan, çoğu Finlandiya emlak ilanından; 4200/400/400 bölünmüş | 80+ sınıf, poligon, SVG | lisans adı doğrulanmadı |

**Öneri: CVC-FP ile başla.** Küçük, taranmış (gürültü gerçek), klasik
yöntemlerin karşılaştırıldığı set ve cevabı duvar/kapı/pencere düzeyinde.
CubiCasa5K stil çeşitliliği için ikinci set olabilir.

**Doğrulanmadı:** CubiCasa5K'nın bağlantısı ve lisansı.

Kaynaklar: [CVC-FP sayfası](https://dag.cvc.uab.es/dataset/cvc-fp-database-for-structural-floor-plan-analysis/),
[CubiCasa5K makalesi](https://arxiv.org/abs/1904.01920),
[CubiCasa5K Zenodo](https://zenodo.org/records/2613548).

## K-004: Ön işleme sırası, öneri (2026-10-17)

Sentetik planda ölçülen ödünleşim ([olcumler §2](olcumler.md)): gürültülü
planda kalın/ince ayrımı (5x5 açma) önce gelirse duvar delikleri büyüyor;
3x3 kapama önce gelirse pencerenin paralel çizgileri birleşip duvar oluyor.

**Seçenekler:**

- a) **Önce küçük delik doldurma, sonra açma** (`onisleme.duvar_adayi`).
  Dört yanı mürekkep, alanı ≤ 4 piksel kâğıt bölgeleri dolar; pencere
  şeritleri (100+ piksel) ve odalar kalır. Sentetik gürültülü planda duvar
  parçaları 5 tohumun 4'ünde tam, birinde 1 fazla ([olcumler §4](olcumler.md)).
  Eşik büyüyünce (10, 50) kötüleşiyor: gürültü pencere şeridini küçük
  parçalara bölüyor, parçalar doluyor.
- b) Önce açma: 5 tohumda 2-8 fazla parça. Önce kapama: her tohumda 1 fazla
  2 eksik (pencere duvar oluyor).
- c) Medyan filtre, sonra ikileme. Denenmedi.

**Öneri: (a), varsayılan akış.** CVC-FP'de sıralar karşılaştırılmadı;
taramadaki gürültü bağımsız piksel çevirmeye benzemeyebilir.

## K-005: Geliştirme / test bölmesi, geçerli (2026-10-17)

CVC-FP'nin resmî bölmesi yok. Her gruptan %30'u test: grup içinde ada göre
sha1 sırası, ilk `round(0.3 n)` (`cvcfp.ayrilan_test_adlari`). 86
geliştirme, 36 test (I-II 63/27, sayi 7/3, image 13/5, p 3/1).

- Eşik ve yöntem seçimi yalnız geliştirmede. Test bir kez, sonda.
- Envanter (§3) cevap istatistiği olarak 122 planın hepsine baktı; sistem
  çıktısı değil.
- Stili görmek için 6 planın görüntüsüne bakıldı, ikisi testte
  (`1`, `IIa_A01001`). Bu planlara göre eşik seçilmedi.
- I-II'nin içi boş 4 planından 3'ü teste düştü (rastgele).

## K-006: Duvar vektörleştirme: iskelet + olasılıksal Hough, geçerli (2026-10-17)

Kalın maske → iskelet → `HoughLinesP` → eşdoğrusal birleştirme → uçları
maskeye uzatma → kavşağa çekme → enine kalınlık ölçümü (`duvar.py`).

- Hough iskelette: her duvar tek orta çizgi verir, iki yüzü ayrı ayrı
  eşlemek gerekmez.
- `max_bosluk` kapı genişliğinden küçük olmalı: sentetik planda 60 piksellik
  kapı, 70'te köprüleniyor. CVC-FP I-II geliştirmede en küçük kapı 77
  piksel (%1'lik dilim 91), varsayılan 20.
- Kalınlık enine çeyrek piksel adımla ölçülüyor; tam piksel adımı eğik
  duvarda ±1 piksel sapıyordu.

**Karşılaştırılmadı:** Canny + Hough ile iki yüzü eşleme, LSD, iskelet
takibiyle parçalara bölme. Raporda "neden Hough" sorusu için en az biri
gerekli.

## K-007: İnce bölme duvarlar: iki ölçek, geçerli (2026-10-17)

CVC-FP dolu duvarlı planlarda iki duvar ailesi var: ~25 piksel dış/taşıyıcı
ve ~7-8 piksel dolu siyah bölme duvar. Merdiven çizgisi gibi çizimler de bu
kalınlıkta (bir örnekte görüldü). Tek çekirdekle açma ikisini ayıramıyor ([olcumler §5](olcumler.md)):
k=7 bölme duvarı tutuyor ama merdiveni de duvar sayıyor, k≥9 bölme duvarı
kaybediyor.

**Seçenekler:**

- a) **İki ölçek** (`duvar.iki_olcekli_duvarlar`): kalın maske (k=13) her
  zaman duvar; ince aday (k=7 maskesi eksi kalın) Hough parçalarına bölünür,
  parça kurala uyarsa duvar: uzunluk ≥ 45, kalınlık ≤ 10, kalınlık değişimi
  (IQR/medyan) ≤ 0.2, yakın paralel komşu ≤ 1 (merdiven). İnce parçanın ucu
  kalın duvarın orta çizgisine çekilir.
- b) Uçların duvara bağlılığını da isteyen graf kuralı. Gerçek bölme
  duvarların yarısı kalın duvara değil başka bölme duvara bağlanıyor, bu
  yüzden bağlantı tek tek değil graf üzerinden bakılmalı. Denenmedi.
- c) Plan başına kalınlık histogramından iki tepe bulup eşiği plana göre
  seçmek. Denenmedi.

**Seçilen: (a), 2026-10-17.** Geliştirmede maske IoU 0.84 → 0.926, parça
IoU 0.81 → 0.880; tavana oran 0.977 ([olcumler §6](olcumler.md)). Eşikler
aynı geliştirme planlarında seçildi; test bölümünde bir kez ölçülecek.

**Bilinen yanlış:** tek başına duran, uzun, düz, duvar kalınlığındaki çizgi
(merdiven taban çizgisi) duvar sayılıyor. (b) bunu eler; şimdilik alanca
etkisi küçük (adaylar içinde kesinlik 0.998).
