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

## K-003: Veri seti, açık (2026-10-17)

| | ne | cevap | not |
|---|---|---|---|
| **CVC-FP** | 122 taranmış plan, 4 alt küme (köken ve çizim stili farklı) | duvar, kapı, pencere, oda, oda ayırıcı; SVG | araştırma topluluğuna serbest bırakıldı (de las Heras vd., 2015); sayfada lisans metni görülmedi |
| CubiCasa5K | 5000 plan, çoğu Finlandiya emlak ilanından; 4200/400/400 bölünmüş | 80+ sınıf, poligon, SVG | lisans adı doğrulanmadı |

**Öneri: CVC-FP ile başla.** Küçük, taranmış (gürültü gerçek), klasik
yöntemlerin karşılaştırıldığı set ve cevabı duvar/kapı/pencere düzeyinde.
CubiCasa5K stil çeşitliliği için ikinci set olabilir.

**Doğrulanmadı:** iki setin de indirme bağlantısının çalıştığı, lisans
metni, CVC-FP'deki dört stilin hangilerinde duvarın dolu siyah çizildiği.

Kaynaklar: [CVC-FP sayfası](https://dag.cvc.uab.es/dataset/cvc-fp-database-for-structural-floor-plan-analysis/),
[CubiCasa5K makalesi](https://arxiv.org/abs/1904.01920),
[CubiCasa5K Zenodo](https://zenodo.org/records/2613548).

## K-004: Ön işleme sırası, açık (2026-10-17)

Sentetik planda ölçülen ödünleşim ([olcumler §2](olcumler.md)): gürültülü
planda kalın/ince ayrımı (5x5 açma) önce gelirse duvar delikleri büyüyor;
3x3 kapama önce gelirse pencerenin paralel çizgileri birleşip duvar oluyor.

**Seçenekler (denenmedi):**

- a) Önce açma, sonra yalnız duvar maskesinde küçük delik doldurma.
- b) Gürültüyü morfoloji yerine bağlı bileşen alanıyla silmek (alanı
  eşikten küçük lekeler), sonra kalın/ince ayrımı.
- c) Medyan filtre, sonra ikileme.

**Neyle karar verilir:** veri setindeki gerçek gürültü tipine bakılarak.
Sentetik gürültü (bağımsız piksel çevirme) taramadaki gürültüye benzemeyebilir.
