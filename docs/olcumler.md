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

## §3 CVC-FP envanteri (2026-10-17)

`scripts/cvcfp_envanter.py` → `results/cvcfp_envanter.csv`. 122 plan, 20 s.
Cevap istatistiği; sistem çıktısı yok, bu yüzden 122 planın hepsine bakıldı.

| grup | n | genişlik (px) | duvar | kapı | pencere | oda | duvar kalınlığı (px) | duvar dolgusu |
|---|---|---|---|---|---|---|---|---|
| I-II | 90 | 2479-3508 | 40.5 [19-85] | 9 [3-20] | 10 [4-18] | 10 [5-18] | 22.6 [6.5-34.3] | 0.950 [0.254-0.986] |
| sayi | 10 | 1027-2218 | 60.5 [24-71] | 10 [4-13] | 10 [2-14] | 11 [5-13] | 10.1 [5.2-18] | 0.559 [0.520-0.573] |
| image | 18 | 5671 | 79 [66-93] | 10.5 [8-13] | 12.5 [12-13] | 12.5 [10-15] | 24.9 [22.6-27.4] | 0.291 [0.233-0.343] |
| p | 4 | 2550 | 35 [27-43] | 6.5 [4-9] | 5.5 [3-8] | 6.5 [4-9] | 22.5 [20-23.9] | 0.339 [0.325-0.355] |

Plan başına medyan [en az-en çok]. Kalınlık: duvar poligonu başına
2·alan/çevre, plan medyanı. Dolgu: cevaptaki duvar piksellerinin Otsu ile
mürekkep çıkan payı.

- **Dolgu stili ayırıyor.** I-II'de 86/90 plan ≥ 0.8 (dolu siyah duvar);
  4'ü 0.25-0.34 (içi boş: IIa_A01001, IIa_DL0701, IIa_GL0401, IIa_GT0602).
  Diğer üç grup 0.23-0.57: taralı ya da çift çizgi. Kalın/ince ayrımı yalnız
  dolu duvarda anlamlı: veri setinin 86/122'si (%70).
- Dolu planlarda plan medyan kalınlığı %5/50/95: 12.2 / 22.6 / 32.5 px.
- **Yöntemin tavanı:** mürekkep seçen bir yöntemin IoU'su en fazla dolgu
  oranı kadar olabilir (cevabın mürekkep olmayan kısmı bulunamaz). Dolu
  planlarda medyan 0.95.

**Duvar başına kalınlık** (dolu planlar, geliştirme, 2543 duvar).
2·alan/çevre kısa duvarda kalınlığı düşük gösterir (kare parça 25x25 →
12.5), alt aralıklar kısa parçalarla şişkin:

| kalınlık (px) | duvar | alan payı |
|---|---|---|
| < 6 | 5.2% | 1.0% |
| 6-9 | 19.5% | 5.0% |
| 9-12 | 9.6% | 3.6% |
| 12-16 | 4.7% | 1.3% |
| 16-24 | 16.8% | 13.8% |
| 24-40 | 41.6% | 65.0% |
| ≥ 40 | 2.4% | 10.2% |

**Kapı boyu** (geliştirme, kapı poligonunun en küçük döndürülmüş
dikdörtgeninin uzun kenarı, px, %1 / 5 / 50 / 95): I-II 91 / 94 / 102 /
146, en küçük 77; sayi 43 / 44 / 67 / 105; image 192 / 192 / 197 / 246;
p 137 / 142 / 152 / 177. `max_bosluk` için üst sınır.

**Cevap maskesi hatası (düzeltildi):** `cv2.fillPoly` birden çok poligonu
tek-çift kuralıyla dolduruyor; cevapta duvarlar köşede üst üste bindiği için
üst üste binen yerler delik çıkıyordu. 122 planda duvar piksellerinin
%1.0'ı (45 968 878 yerine 45 499 667). Poligonlar artık tek tek çiziliyor;
envanter düzeltmeden sonra yeniden çalıştırıldı, dolgu en çok 0.006 değişti.

## §4 Duvar parçaları, sentetik plan (2026-10-17)

`duvar.duvarlari_bul` (iskelet + olasılıksal Hough, K-006), varsayılanlar:
`max_bosluk` 20, Hough eşiği 10, en kısa 15 px. Eşleme: açı ≤ 5°, iki uç
≤ 2 px. Cevap 7 duvar (en uzun düz parçalar, köşe ucu orta çizgi kesişimi).

| durum | doğru / fazla / eksik | kalınlık | parça maskesi IoU |
|---|---|---|---|
| düz, duvar 10 / 16 / 22 px | 7 / 0 / 0 (üçü de) | tam | 0.9994 |
| 7°, 15°, 30°, 45°, 60° döndürülmüş | 7 / 0 / 0 (hepsi) | ±0.25 | 0.970-0.995 |
| `max_bosluk` 50 | 7 / 0 / 0 | | |
| `max_bosluk` 70 (kapı 60 px) | 5 / 1 / 2 | | |

Döndürülmüşte parça IoU'su kalın maskeninkinden (0.996-0.999) düşük; büyük
ihtimalle en yakın komşu döndürmenin merdivenli kenarı ile düz dikdörtgen
farkı. Doğrulanmadı.

Kalınlık ölçümü önce tam piksel adımlıydı: eğik duvarda ±1 px sapma, parça
IoU 0.92-0.98. Çeyrek piksel adımla ±0.25.

**Gürültülü plan** (%1 piksel ters, 5 tohum):

| ön işleme | doğru / fazla / eksik, tohum 0-4 |
|---|---|
| önce açma (`kalin_ayir`) | 7/3/0, 6/5/1, 7/2/0, 6/5/1, 5/8/2 |
| önce 3x3 kapama | 5/1/2 hepsinde (pencere duvar oluyor) |
| delik doldur (≤ 4 px) + açma | 7/0/0, 7/0/0, 7/0/0, 7/0/0, 7/1/0 |
| delik doldur (≤ 10 px) + açma | 7/0/0, 6/2/1, 7/0/0, 7/0/0, 7/1/0 |
| delik doldur (≤ 50 px) + açma | 7/0/0, 6/2/1, 6/1/1, 7/0/0, 7/1/0 |

600x400 planda süre ~60 ms.

## §5 Duvar, CVC-FP geliştirme bölümü (2026-10-17)

`scripts/duvar_deney.py`, varsayılan akış (`duvar_adayi`: Otsu → delik
doldur ≤ 4 → k×k açma). Yalnız geliştirme (86 plan). IoU cevaptaki
duvar maskesiyle, plan başına; medyan [%10-%90].

**Açma çekirdeği taraması, kalın maske IoU:**

| k | I-II dolu (62) | I-II boş (1) | sayi (7) | image (13) | p (3) |
|---|---|---|---|---|---|
| 5 | 0.832 [0.764-0.860] | 0.174 | 0.278 | 0.001 | 0.307 |
| 7 | **0.844** [0.770-0.870] | 0.175 | 0.153 | 0.000 | 0.004 |
| 9 | 0.774 [0.710-0.831] | 0 | 0.036 | 0 | 0.003 |
| 11 | 0.799 [0.731-0.853] | 0 | 0.000 | 0 | 0 |
| 13 | 0.841 [0.759-0.900] | 0 | 0 | 0 | 0 |
| 15 | **0.844** [0.764-0.903] | 0 | 0 | 0 | 0 |
| 17 | 0.844 [0.764-0.903] | 0 | 0 | 0 | 0 |

- Dolu olmayan stillerde yöntem duvar bulmuyor. Beklenen; artık sayıyla.
- I-II dolu planlarda k=9'da çukur. Hata ayrışımı (ilk 20 dolu plan,
  fazla / kesişim, eksik / kesişim): k=7 0.112 / 0.081, k=9 0.101 / 0.200,
  k=13 0.017 / 0.200.
- Görsel (IIa_BK0701, `results/ara/duvar_hata_kesit.png`, depoda değil):
  ~7-8 px dolu bölme duvarlar k≥9'da tamamen kayboluyor; k=7'de merdiven
  çizgisi duvar olarak kalıyor. İki aile tek çekirdekle ayrılmıyor (K-007).
- Tavan (§3): medyan 0.95.

**Hough parçaları** (`max_bosluk` 20; I-II dolu 62 plan):

| k | kalın maske IoU | parça maskesi IoU | vektör kaybı | parça / cevap duvarı | süre / plan |
|---|---|---|---|---|---|
| 7 | 0.844 [0.770-0.870] | 0.796 [0.738-0.839] | 0.036 | 3.08 | 1.1 s (en çok 3.1) |
| 13 | 0.841 [0.759-0.900] | 0.812 [0.744-0.856] | 0.027 | 0.67 | 0.6 s (en çok 2.4) |

- Parça / cevap oranı yalnız işaret: cevaptaki duvar poligonu ile en uzun
  düz parça aynı birim değil.
- Ic_MT0601'de k=7'nin 104 parçasının 69'u plan alanı dışında (başlık
  yazısı vb.); k=13'te 0. Plan alanını ayırmak ayrı bir adım olabilir.
- Görsel: `results/ara/duvar_parcalar_ornek.png` (depoda değil). Dış
  duvarlar pencerelerde doğru bölünüyor, köşe uçları kesişimde.

## §6 İnce bölme duvarlar: iki ölçek (2026-10-17)

K-007 (a). Yalnız geliştirme, I-II dolu duvarlı 62 plan.

**Aday incelemesi** (`scripts/ince_aday_ozellik.py`): ince aday = k_ince
açmasından kalan, kalın maskenin (k=13) 1 px genişletilmişi dışındaki kısım;
Hough parçalarına bölünür. Etiket: parçanın dikdörtgeninin cevaptaki duvarla
örtüşmesi ≥ 0.5. Örtüşme ikiye ayrık: 4729 parça < 0.1, 851 parça ≥ 0.7,
arası yok.

| k_ince | aday | duvar | duvarın alan payı |
|---|---|---|---|
| 7 | 5580 | 851 (%15.3) | %38.4 |
| 5 | 6944 | 898 (%12.9) | %36.5 |

Özellik dağılımı, k_ince=7, yüzdelik 10 / 50 / 90:

| özellik | duvar | duvar değil |
|---|---|---|
| uzunluk (px) | 75 / 184 / 365 | 29 / 37 / 59 |
| kalınlık (px) | 7 / 7 / 8 | 10.5 / 13 / 36 |
| kalınlık değişimi (IQR/medyan) | 0 / 0 / 0 | 0.03 / 0.84 / 2.0 |
| doluluk | 1 / 1 / 1 | 0.73 / 1 / 1 |
| yakın paralel parça | 0 / 0 / 1 | 0 / 1 / 4 |
| uca en yakın kalın duvar (px) | 1 / 2 / 299 | 27 / 167 / 373 |

- Duvar olmayan kalın adaylar (kalınlık 13-37) başlık yazısının kalın
  harfleri (IIa_BK0701'de "Plan du Rez de Chaussée", görüntüyle bakıldı).
- Gerçek bölme duvarların yarısı bir ucuyla kalın duvara değiyor (≤ 2 px);
  kalanı başka bölme duvarlara bağlanıyor. Bağlantı kuralı bu yüzden
  "kalın duvara değer" olamaz; graf gerekir. Şimdilik gerekmedi (aşağıda).
- Zor yanlış: merdivenin taban çizgisi (IIa_BK0701) uzun, düz, 7 px, tek;
  kurallar bunu geçiriyor.

**Kural taraması** (alanca, adaylar içinde; 4 eşik, kaba ızgara, aynı
geliştirme planlarında seçildi, iyimser olabilir):

| k_ince | kural | seçilen | doğru | kesinlik | duyarlılık |
|---|---|---|---|---|---|
| 7 | uzunluk ≥ 45, kalınlık ≤ 10, değişim ≤ 0.2, paralel ≤ 1 | 829 | 827 | 0.998 | 0.987 |
| 5 | aynı | 893 | 856 | 0.989 | 0.962 |

Seçilen: k_ince=7 ve bu kural (`duvar.InceKural`).

**Sentetik plan** (24 px dış, 8 px bölme duvar, 8 px'lik 6 merdiven
basamağı; `plan_ciz(duvar=24, ic_duvar=8, merdiven=True)`):

| akış | doğru / fazla / eksik |
|---|---|
| tek çekirdek k=7 | 7 / 6 / 0 (basamaklar duvar) |
| tek çekirdek k=13 | 5 / 0 / 2 (bölme duvar yok) |
| iki ölçek | 7 / 0 / 0, parça IoU 0.9996 |

Basamakların paralel komşu sayısı 3-5. Bölme duvarın uçları kalın duvarın
orta çizgisine çekiliyor (49.4; cevap 49.5).

**CVC-FP, iki ölçek** (`duvar_deney.py --yontem iki --kalinlik 13 --k-ince 7`):

| akış | maske IoU | parça IoU | süre / plan |
|---|---|---|---|
| tek k=7 (§5) | 0.844 [0.770-0.870] | 0.796 [0.738-0.839] | 1.1 s |
| tek k=13 (§5) | 0.841 [0.759-0.900] | 0.812 [0.744-0.856] | 0.6 s |
| **iki ölçek** | **0.926** [0.845-0.946] | **0.880** [0.821-0.929] | 1.3 s (en çok 2.8) |

- Plan başına fark (iki - tek): k=7'ye göre maske +0.078 medyan
  [%10 +0.059, %90 +0.104], 61/62 planda daha iyi; k=13'e göre +0.081,
  62/62.
- Tavana oran (maske IoU / dolgu) medyan 0.977.
- k=7'den kötü tek plan IIa_BK0703: 0.9002'ye 0.9008, pratikte eşit. Hata
  haritası: kalan hata duvar kenarlarında ince şeritler (cevap poligonu
  mürekkepten biraz geniş) ve bir kısa duvar parçası.
- Maske ile parça IoU arası 0.046: ince parçalar kalın duvara 1 px kala
  bitiyor; kısa ve dikdörtgen olmayan parçalar kayboluyor. Ayrıştırılmadı.

## §7 Bağlantı kuralları: merdiven çizgisi elenemedi (2026-10-17)

K-007'de kalan iki yanlış (geliştirmede 2751 px, ikisi de merdivenin parçası:
IIa_BK0701'de taban çizgisi, Ic_MT0601'de döner merdivenin dikmesi) için
denenen kurallar. Hepsi yerel kuraldan geçen 829 ince parça üzerinde, alanca.

| kural | elenen yanlış | kaybedilen gerçek duvar | duyarlılık |
|---|---|---|---|
| yalnız yerel kural | — | — | 0.987 |
| kalın duvara graf (temas + aynı doğruda 150 px) | 2 | 38 parça, 43 189 px | 0.952 |
| ışınla graf (uçtan 150 px) | 0 | 106 parça, 107 321 px | 0.900 |
| ışınla iki uç | 2 | 490 parça, 661 903 px | 0.448 |
| en az bir uç kavşakta | 2 | 65 parça, 84 679 px | — |
| temas, iki yönlü (uç gövdeye ya da gövde uca) | 2 | 39 parça, 46 742 px | — |
| düzenli basamak dizisi (≥ 5 kesişme, aralık CV < 0.1) | 1 (dikmede kesişme yok) | 12 parça | — |

- Gerçek bölme duvarların ~%5'i hiçbir duvara değmiyor: iki kapı arasında
  ya da ince çizgiyle çizilmiş dolapların yanında duruyor. Topolojik kural
  bunları merdiven çizgisinden ayıramıyor.
- Merdiveni sınırlayan gerçek duvarlarda da basamaklar düzenli aralıkla
  bitiyor (12 duvar); basamak işareti de ayırmıyor.
- Işın kuralının ilk sürümünde hata vardı: etiket görüntüsünde kesişen
  dikdörtgenler birbirinin üstüne yazılıyor, kavşaktan çıkan ışın komşuyu
  görmüyordu. Sonuç tablodaki gibi kötü; düzeltilmiş hali denenmedi, çünkü
  temas kuralı (aynı bilgiyi geometriyle veren) da kaybettiriyor.

Sonuç: bağlantı kuralı yok. Yanlış alan, ince duvar alanının %0.2'si.
Test: `test_bilinen_sinir_serbest_cizgi_duvar_sayilir`.

## §8 Kapı ve pencere (2026-10-17)

`aciklik.py`, `scripts/aciklik_deney.py`. Geliştirme, I-II dolu duvarlı 62
plan; cevapta 555 kapı, 574 pencere. Eşleme: poligon IoU ≥ 0.5, bire bir.
Kapı poligonu açılma alanı (menteşe + çeyrek yay), pencere poligonu boşluk
dikdörtgeni ya da kanatlı pencerede açılma alanı.

**Sürümler:**

| sürüm | kapı D/F/E | kapı K / D / F1 | pencere D/F/E | pencere F1 |
|---|---|---|---|---|
| 1. parça boşlukları, yay ≥ 0.6 (ışın hatalı) | 391/135/164 | 0.743 / 0.705 / 0.723 | 308/94/266 | 0.631 |
| 1a. ışın yalnız serbest uçlardan | 350/108/205 | 0.764 / 0.631 / 0.691 | 307/86/267 | 0.635 |
| 2. boşluk = ışının boş aralığı, kavşak ucunda aynı doğrultu | 345/91/210 | 0.791 / 0.622 / 0.696 | 280/70/294 | 0.606 |
| köşede yay merkezi (menteşe dışbükey köşede) | 240/514/315 | 0.318 / 0.432 / 0.367 | 243/8/331 | 0.589 |
| 2 + raster köşelerinden boşluk, yay ≥ 0.6 | 443/283/112 | 0.610 / 0.798 / 0.692 | 219/55/355 | 0.517 |
| **3. + yay ≥ 0.8, kapı yarıçapı ≥ 75, küçük yay kanatlı pencere** | **422/12/133** | **0.972 / 0.760 / 0.853** | **335/91/239** | **0.670** |
| 4. köşe boşluğunu orta çizgiye taşı, pencereyi her boşlukta dene | 376/8/179 | 0.979 / 0.677 / 0.801 | 325/204/249 | 0.589 |
| 3 + yarı açık kapı (yay + kanat), eşik 0.95 | 450/73/105 | 0.860 / 0.811 / 0.835 | 296/89/278 | 0.617 |
| 3 + yarı açık kapı, eşik 0.90 | 452/93/103 | 0.829 / 0.814 / 0.822 | 295/89/279 | 0.616 |

Seçilen: 3. Eşikler (0.8, 75 px) bu planlarda seçildi; test bölümünde
bir kez ölçülecek. §9'daki ince duvar kapsam düzeltmesinden sonra 3:
kapı 421/10/134, K 0.977 / D 0.759 / F1 0.854; pencere 335/89/239, F1
0.671 (duvar IoU değişmedi).

- **Menteşe her zaman dışbükey köşede değil:** kapı boşluğu bir duvar ucu
  ile dik duvarın düz yüzü arasında olabilir ve menteşe o yüzde durur.
  Köşeler yay merkezi olarak değil, boşluk aramanın başlangıcı olarak işe
  yarıyor.
- **Kanatlı pencere:** yay çizilmiş küçük pencereler (WC, banyo dış duvarı)
  cevapta Window. Yarıçap, %5 / medyan / %95: kapı 89 / 91 / 102 px,
  kanatlı pencere 60 / 73 / 86 px. Sınır 75 px; cevaptaki en küçük kapı
  77 px (§3). Bu piksel eşiği CVC-FP I-II'nin tarama ölçeğine bağlı.
- Yay skoru eşiğe yakın yığılıyor: tam yay örneklemesi 24 örnekten 2°
  adımlı 38 örneğe çıkınca fazla kapı 12'den 116'ya çıktı. Eşikler mevcut
  örneklemeyle geçerli.
- Kavşağa çekmede düzeltilen hata: kesişim ucun gerisindeyse (parçanın
  ortasındaki T-kavşağı) uç oraya çekilip duvar 20-60 px kısalıyordu.
  Duvar IoU'su değişmedi (§6 yeniden ölçüldü: 0.926 / 0.880).

**Sürüm 3'ün kalan hataları** (`scripts/aciklik_hata.py`, kapsam
düzeltmesinden sonra):

| eksik kapı (134) | | fazla kapı (10) | |
|---|---|---|---|
| yay zayıf (aday var, en iyi pay < 0.8) | 95 | cevapta pencere | 3 |
| aday yok | 35 | geometri (0.1 ≤ IoU < 0.5) | 3 |
| geometri | 3 | boşta | 4 |
| diğer | 1 | | |

- Cevaptaki 555 kapının 520'sinde (%94) parça ya da köşe boşluğu adayı var.
- "Yay zayıf" kapıların (düzeltmeden önceki 96 kapı) en iyi payı %10 / 50 / 90: 0.33 / 0.58 / 0.75.
  Görüntüyle bakılan 8 örneğin çoğu yarı açık çizilmiş kapı (kanat 30-60°,
  yay o açıya kadar); denenen kısmi varsayım net kazanç vermedi (tablo).

## §9 Oda grafı (2026-10-17)

`oda.py`, `scripts/oda_deney.py`, `scripts/oda_hata.py`. Geliştirme, I-II
dolu duvarlı 62 plan; cevapta 602 oda, kapılı `access` ilişkisi (Door,
Room, Room) 434, duvarsız ayrım ilişkisi (Room, Room, Separation) 131.

Akış: duvar maskesi + kapı ve pencere kirişleri; bina ayak izi (duvar
maskesinin büyük kareyle kapanmışı, delikleri dolu) içindeki boş alanın
4-bağlı bileşenleri oda (≥ 5000 px); her kapının iki yanındaki oda kenar.

Eşleme: oda bölgesi ile cevap Room poligonu IoU ≥ 0.5, bire bir. Kenar
doğru: kapısı cevap kapısıyla eşleşmiş ve iki odası o kapının `access`
ilişkisindeki iki odayla eşleşmiş.

| ayak izi çekirdeği | sınıflanmamış boşlukları da kapat | oda D/F/E | oda K / D / F1 | kenar D/F/E | kenar F1 |
|---|---|---|---|---|---|
| yok (dış = kenara değen) | hayır | 184/11/418 | 0.944 / 0.306 / 0.462 | 21/20/413 | 0.088 |
| yok | evet | 282/576/320 | 0.329 / 0.468 / 0.386 | 19/209/415 | 0.057 |
| **151** | **hayır** | **246/74/356** | **0.769 / 0.409 / 0.534** | **74/119/360** | **0.236** |
| 151 | evet | 277/527/325 | 0.345 / 0.460 / 0.394 | 16/196/418 | 0.050 |
| 251 | hayır | 243/123/359 | 0.664 / 0.404 / 0.502 | | |
| 301 | hayır | 263/132/339 | 0.666 / 0.437 / 0.528 | | |
| 401 | hayır | 264/144/338 | 0.647 / 0.439 / 0.523 | | |

Seçilen: çekirdek 151, sınıflanmamış boşluklar açık. Kapı kenarlarının
112'sinin bir yanı bina dışı (cevapta karşılığı yok, sayılmadı).

- **Bütün boşluk adaylarını kapatmak odaları bölüyor** (576 fazla oda):
  adayların bir kısmı açıklık değil, oda ortasından geçen kiriş üretiyor.
- **Dış = resmin kenarına değen bileşen** tanımında kapanmamış tek açıklık
  bütün iç mekânı dışarı akıtıyordu; ayak izi bunu kesiyor.
- Düzeltilen iki hata: ince duvar kalın duvara 1-2 px yarıkla değiyordu
  (ince parça kapsamı kavşağa çekilen uca uzatıldı; duvar IoU değişmedi,
  kapı F1 0.853 → 0.854); ayak izinde kapama resmin kenarına yapışıyordu
  (kenar payı eklendi).

**Kaçan 356 odanın nedeni** (`scripts/oda_hata.py`, çekirdek 151):

| neden | oda |
|---|---|
| dış sayıldı (ayak izinin dışında kaldı) | 204 |
| komşuyla birleşti, aralarında cevapta kapı var (kapı bulunamadı ya da açıklık kapanmadı) | 101 |
| komşuyla birleşti, aralarında duvarsız ayrım var (bu yöntemle bulunamaz) | 30 |
| komşuyla birleşti, cevapta ilişki yok | 20 |
| kısmi (IoU < 0.5) | 1 |

Görüntüyle bakılan 3 planda (IIa_BK0701, IIa_AP3401, Ib_RI0501):

- Dış duvarda 150 px'ten geniş açıklık (porte-fenêtre) ayak izini deliyor;
  daha büyük çekirdek bunu kapatıyor ama sundurma ve teras gibi içbükey dış
  alanları içeri alıyor (tablo: doğru oda artıyor, fazla daha çok artıyor).
- Bazı bölme duvarlar 2-3 px çizgiyle çizilmiş (Ib_RI0501: HALL, CUISINE,
  SEJOUR arası); ince duvar adımı 7 px'ten inceyi atıyor, odalar birleşiyor.

## §10 Oda grafı, ikinci tur (2026-10-17)

Aynı 62 plan, aynı eşleme (§9). `scripts/oda_deney.py --varyant`, tek
geçişte birden çok oda varyantı; `scripts/oda_hata.py`,
`scripts/kenar_hata.py` hata dökümü.

**Kenar kaybı nereden** (§9 ayarı, `kenar_hata.py`): cevaptaki 434 kapı
kenarının 255'inde iki odadan biri bulunamamış, 105'inde kapı bulunamamış,
74'ü doğru; kapı ve iki oda bulunduğunda yanlış bağlanan 0. Kenar, oda
duyarlılığına bağlı: iki odanın ikisi de bulunmalı.

**"Dış" sayılan odalar:** çevrelerinin medyan %68'i duvar/kirişte (bulunan
odalarda %92), ayak izi içinde kalan payları medyan 0. Neden (IIa_BA0710,
görüntüyle): dış duvarda tek bir çok geniş açıklık (garaj kapısı, 250+ px)
151 px kapamayla köprülenmiyor, dışarıdan doldurma içeri girip bütün evi dış
yapıyor. Ayak izi 414 bin px, yalnız duvarlar 264 bin px.

| varyant | oda D/F/E | oda K / D / F1 | kenar D/F/E | kenar F1 |
|---|---|---|---|---|
| §9 seçilen (çekirdek 151) | 246/74/356 | 0.769 / 0.409 / 0.534 | 74/119/360 | 0.236 |
| ayak izi parça boşluklarıyla kapanmış duvardan | 241/75/361 | 0.763 / 0.400 / 0.525 | 72/134/362 | 0.225 |
| ayak izi bütün boşluklar kapanmış duvardan, oda bölmede parça boşlukları | 282/185/320 | 0.604 / 0.468 / 0.528 | 75/161/359 | 0.224 |
| ayak izi bütün boşluklardan + ince sınır çizgileri (iki ucu duvara değen) | 249/314/353 | 0.442 / 0.414 / 0.427 | 67/159/367 | 0.203 |
| çekirdek 301 + duvar payı ≥ 0.6 | 263/69/339 | 0.792 / 0.437 / 0.563 | 61/200/373 | 0.176 |
| çekirdek 401 + duvar payı ≥ 0.7 | 260/58/342 | 0.818 / 0.432 / 0.565 | 47/222/387 | 0.134 |
| geniş açıklıklar (sınır + oda bölme), 151, pay 0.6 | 271/67/331 | 0.802 / 0.450 / 0.577 | 68/176/366 | 0.201 |
| geniş + parça boşlukları, 151, pay 0.7 | 313/127/289 | 0.711 / 0.520 / 0.601 | 85/176/349 | 0.245 |
| geniş + parça + köşe boşlukları, 151, pay 0.6 | 331/583/271 | 0.362 / 0.550 / 0.437 | 27/247/407 | 0.076 |
| geniş + parça, 251, pay 0.7 | 322/132/280 | 0.709 / 0.535 / 0.610 | 90/194/344 | 0.251 |
| **geniş + parça, 301, pay 0.8** | **328/118/274** | **0.735 / 0.545 / 0.626** | **94/196/340** | **0.260** |
| geniş + parça, 401, pay 0.7 | 336/134/266 | 0.715 / 0.558 / 0.627 | 96/212/338 | 0.259 |
| geniş + parça, 501, pay 0.7 | 341/136/261 | 0.715 / 0.566 / 0.632 | 98/218/336 | 0.261 |
| geniş + parça, 601, pay 0.8 | 333/121/269 | 0.733 / 0.553 / 0.631 | 96/206/338 | 0.261 |

- **Geniş açıklık:** parça ucundan 150-450 px ötede, aynı doğrultuda (≤ 10°,
  orta çizgi ≤ yarım kalınlık + 4 px) sürüyorsa kapatılır. Aynı doğrultu
  şartı ışının odanın karşı duvarına çarpmasını önlüyor.
- **Duvar payı:** ayak izi içindeki bölgenin çevre pikselleri duvar/kirişe mi
  ayak izi dışına mı komşu; payı eşiğin altındaki bölge dış (sundurma,
  teras).
- Çekirdek 301'den sonra plato (farklar ≤ 0.006 F1). En yükseği değil,
  platonun başı seçildi: daha küçük çekirdek ölçeğe daha az bağlı ve hızlı.
- Bu tabloda sınır ve oda bölme kaynakları dışında her şey aynı; eşikler
  aynı geliştirme planlarında seçildi.

**Seçilenin kalan hataları:**

| kaçan oda (274) | | bulunamayan kenar (340) | |
|---|---|---|---|
| kapısı kapanmadığı için birleşti | 99 | odalardan biri yok | 215 |
| duvarsız ayrım yüzünden birleşti (bulunamaz) | 91 | kapı yok | 105 |
| dış | 40 | yanlış oda çiftine bağlandı | 20 |
| başka nedenle birleşti | 33 | | |
| kısmi | 11 | | |

- Duvarsız ayrım 91 odayı birleştiriyor: bu yöntemle oda duyarlılığının
  üst sınırı yaklaşık (602 - 91) / 602 = 0.85.
- "Yanlış oda" 20: parça boşluğu kirişi kapının yanından geçince kapının bir
  yanı yanlış bölgede kalıyor (§9'da 0'dı).

**Uçtan uca doğrulama:** `scripts/degerlendir.py` (`akis.plani_coz` + bütün
ölçüler) geliştirmede ayrı deneylerin sayılarını birebir verdi: duvar 0.926
/ 0.880, kapı 421/10/134, pencere 335/89/239, oda 328/118/274, kenar
94/196/340. Plan başına süre medyan 2.1 s.

## §11 Yarı açık kapı (ikinci deneme) ve havza bölütleme (2026-10-18)

Aynı 62 geliştirme planı.

**Yarı açık kapı.** §8'deki kısmi açılma varsayımıyla bulunan 88 kapının
hepsi 30° varsayımından (45° ve 60° hiç seçilmedi); 28 doğru, 60 yanlış.
Yanlışların görüntüsü (12 örnek): dış duvardaki pencere çerçevesi; 30°'de
aranan yay parçası (6-26°) çerçevenin duvara dik kısa kenarıyla çakışıyor.
20°'lik yay 90 px yarıçapta düz çizgiden ~1.4 px sapıyor, yerel olarak
ayırt edilemiyor. Kanat için "tek çizgi" testi (kanadın 7 px iki yanında
mürekkep payı ≤ eşik) eklendi:

| ayar | kapı D/F/E | kapı F1 | pencere F1 |
|---|---|---|---|
| kısmi kapalı (varsayılan) | 421/10/134 | 0.854 | 0.671 |
| kısmi 0.95, 30/45/60°, kanat testi yok, pencere çizgili boşlukta kapalı | 449/66/106 | 0.839 | 0.626 |
| kısmi 0.95, yalnız 45/60° | 421/10/134 | 0.854 | 0.671 |
| kısmi 0.95, kanat yanı ≤ 0.3 | 441/62/114 | 0.834 | 0.620 |
| kısmi 0.95, kanat yanı ≤ 0.15 | 438/47/117 | 0.842 | 0.636 |
| kısmi 0.90, kanat yanı ≤ 0.15, pencere çizgili boşlukta kapalı | 438/40/117 | 0.848 | 0.645 |

Son ayar uçtan uca (`degerlendir.py --kismi 0.9:0.15:0`, bağlı bileşen oda
bölmeyle): oda F1 0.626 → 0.632, kenar F1 0.260 → 0.260. 17 kapı daha
bulunuyor ama oda ve kenar kıpırdamıyor. Kısmi açılma kapalı kaldı.

**Kapısı olan iki oda neden birleşik** (`plani_coz`, bağlı bileşen): 95 oda
çiftinin 59'unda kapı bulunamamış; 36'sında kapı bulunmuş, kirişi doğru
yerde, odalar başka bir yoldan (kapanmamış ikinci açıklık, girinti) sızıyor
(3 örnek görüntüyle). Sızıntı yolları planlara göre değişiyor; tek tek
yamamak yerine bölütleme değişti.

**Havza bölütleme** (`oda.havza_bolutle`): boş alanın uzaklık dönüşümü;
duvara uzaklığı ≥ eşik olan bileşenler çekirdek (oda ortası); watershed
havzaları dar geçitlerde (kapı, kapanmamış açıklık) buluşur. İki komşu
havza, aradaki eyer küçük tepenin `birlesme` katından yüksekse (geniş geçit:
L biçimli oda) birleşir. Çekirdeksiz küçük bileşen kendi işaretçisi.

| oda bölme (geniş + parça boşlukları, ayak izi 301, pay 0.8) | oda D/F/E | oda K / D / F1 | kenar D/F/E | kenar F1 |
|---|---|---|---|---|
| bağlı bileşen (§10) | 328/118/274 | 0.735 / 0.545 / 0.626 | 94/196/340 | 0.260 |
| havza, eşik 35, birleşme 0.8 | 390/120/212 | 0.765 / 0.648 / 0.701 | 121/194/313 | 0.323 |
| **havza, eşik 45, birleşme 0.8** | **393/118/209** | **0.769 / 0.653 / 0.706** | **121/197/313** | **0.322** |
| havza, eşik 55, birleşme 0.8 | 395/115/207 | 0.775 / 0.656 / 0.710 | 119/197/315 | 0.317 |
| havza 45, birleşme 0.6 | 391/115/211 | 0.773 / 0.650 / 0.706 | 121/197/313 | 0.322 |
| havza 45, birleşme 0.9 | 393/119/209 | 0.768 / 0.653 / 0.706 | 121/197/313 | 0.322 |

Eşik 35-55 ve birleşme 0.6-0.9 arasında oda F1 0.701-0.710: parametreye
duyarsız. Eşik 45 (yaklaşık yarım kapı genişliği) seçildi.

Havza ile öteki ayarların etkileşimi (havza 45 / 0.8):

| değişen | oda F1 | kenar F1 |
|---|---|---|
| hiçbiri (varsayılan) | 0.706 | 0.322 |
| oda bölmede parça boşlukları kapanmıyor | 0.687 (K 0.872, D 0.566) | 0.290 |
| ayak izi çekirdeği 151 | 0.685 | 0.321 |
| ayak izi ham duvardan | 0.704 | 0.328 |
| duvar payı filtresi yok | 0.643 (234 fazla oda) | 0.316 |

Varsayılan değişmedi.

**Uçtan uca** (`scripts/degerlendir.py --bolum gelistirme`, ayrı deneylerle
birebir aynı): duvar 0.926 / 0.880; kapı 421/10/134 (F1 0.854); pencere
335/89/239 (F1 0.671); oda 393/118/209 (F1 0.706); kenar 121/197/313 (F1
0.322). Plan başına süre medyan 4.2 s (havza 2.1 s ekliyor).

**Kalan hatalar:**

| kaçan oda (209) | | bulunamayan kenar (313) | |
|---|---|---|---|
| duvarsız ayrım yüzünden birleşti (bulunamaz) | 79 | odalardan biri yok | 184 |
| kapısı kapanmadığı için birleşti | 62 | kapı yok | 105 |
| dış | 40 | yanlış oda çiftine bağlandı | 24 |
| kısmi | 13 | | |
| başka nedenle birleşti | 15 | | |

## §12 Literatürle kıyas için: bütün stiller, kutu IoU (2026-10-18)

`scripts/degerlendir.py --bolum gelistirme --stil {dolu,hepsi}`. Akış ve
eşikler §11 ile aynı; değişen yalnız ölçüm. Eklenenler:

- `--stil hepsi`: geliştirme bölümünün 86 planının hepsi (taralı, çift
  çizgili, içi boş dahil).
- **Kutu IoU:** kapı ve pencere poligonlarının eksen hizalı sınır kutusuyla
  eşleme, IoU ≥ 0.5, bire bir. Nesne tespiti yayınları kutu ölçüyor.
- **Havuzlanmış duvar IoU:** bütün planların kesişim toplamı / birleşim
  toplamı.

Kıyas: [literatur.md](literatur.md).

| | dolu (62) | hepsi (86) |
|---|---|---|
| duvar maske IoU medyan / ortalama | 0.926 / 0.908 | 0.876 / 0.666 |
| duvar IoU havuzlanmış | 0.906 | 0.407 |
| kapı, poligon D/F/E, K / D / F1 | 421/10/134, 0.977 / 0.759 / 0.854 | 425/14/363, 0.968 / 0.539 / 0.693 |
| kapı, kutu D/F/E, K / D / F1 | 423/8/132, 0.981 / 0.762 / 0.858 | 427/12/361, 0.973 / 0.542 / 0.696 |
| pencere, poligon D/F/E, F1 | 335/89/239, 0.671 | 335/137/490, 0.517 |
| pencere, kutu D/F/E, K / D / F1 | 246/178/328, 0.580 / 0.429 / 0.493 | 246/226/579, 0.521 / 0.298 / 0.379 |
| oda D/F/E, F1 | 393/118/209, 0.706 | 410/129/455, 0.584 |
| kenar D/F/E, F1 | 121/197/313, 0.322 | 121/201/509, 0.254 |

"dolu" satırları §11'deki uçtan uca ölçümle aynı; yeniden üretildi.

Grup başına, hepsi (`results/degerlendir_gelistirme_hepsi.csv`):

| grup | plan | duvar IoU ort | kapı D/F/E | pencere D/F/E | oda D/F/E | kenar D/F/E |
|---|---|---|---|---|---|---|
| I-II dolu | 62 | 0.908 | 421/10/134 | 335/89/239 | 393/118/209 | 121/197/313 |
| I-II içi boş | 1 | 0.195 | 4/0/2 | 0/4/5 | 4/1/3 | 0/2/4 |
| sayi (taralı) | 7 | 0.109 | 0/4/71 | 0/29/69 | 13/10/61 | 0/2/55 |
| image (taralı dolgu) | 13 | 0.000 | 0/0/139 | 0/15/163 | 0/0/165 | 0/0/124 |
| p (çift çizgi) | 3 | 0.002 | 0/0/17 | 0/0/14 | 0/0/17 | 0/0/13 |

- Dolu olmayan 24 planda yöntem neredeyse hiçbir şey bulmuyor. Doğru
  kapıların 421/425'i, doğru pencerelerin hepsi, doğru odaların 393/410'u
  dolu planlardan.
- Havuzlanmış IoU'yu `image` grubu belirliyor: 13 plan, her biri
  5671x7383 ve hepsinde IoU 0.
- Pencerede kutu IoU'lu doğru sayısı poligon IoU'lu doğru sayısından az
  (246 ve 335). Nedeni incelenmedi.
- Kapı kenarı duyarlılığı yaklaşık olarak üç olasılığın çarpımı: kapı
  bulunur ve iki oda bulunur, 0.76 × 0.65 × 0.65 ≈ 0.32. Bağımsızlık
  varsayımıyla bu bir kaba hesap. Ölçülen 0.28.
- Oda duyarlılığının tavanı (duvarsız ayrımla birleşen 79 oda bulunamaz):
  (602 - 79) / 602 ≈ 0.87. §10'daki 0.85 o zamanki 91 odayla
  hesaplanmıştı.

## §13 Pencere: çift kanat, geniş boşluk (2026-10-18)

Geliştirme, I-II dolu duvarlı 62 plan; cevapta 574 pencere. Hata
sınıflandırması: `scripts/pencere_hata.py` (sınıf tanımları betikte).

**Başlangıç (§11 akışı), 335/89/239:**

| eksik (239) | | fazla (89) | |
|---|---|---|---|
| aday yok | 113 | geometri (0.1 ≤ IoU < 0.5) | 70 |
| geometri | 53 | cevapta kapı | 10 |
| çizgi testi geçmedi | 39 | boş | 9 |
| yalnız köşe boşluğu | 18 | | |
| bir yay tutmuş | 16 | | |
| duvar sayılmış | 0 | | |

- "Aday yok"ların 109'u 200 px'ten geniş (medyan 283 px); boşluk arama
  200 px'te kesiyordu. 56'sının çevresinde geniş boşluk (150-450) vardı.
- Geometri hataları görüntüyle: çoğu çift kanatlı pencere. Cevap poligonu
  boşluk dikdörtgeni + iki kanadın açılma alanı (medyan ~144 × 117 px),
  bizimki yalnız dikdörtgen (~140 × 43): IoU ~0.40. Tek kanatlı pencerede
  de cevap dikdörtgen + açılma alanı.

**Varyantlar** (`scripts/aciklik_deney.py`, bayraklar `AciklikKural`'da):

| varyant | pencere D/F/E | K / D / F1 |
|---|---|---|
| v0 eski (birebir §8/§11) | 335/89/239 | 0.790 / 0.584 / 0.671 |
| v4 yalnız tek kanatta dikdörtgen | 340/84/234 | 0.802 / 0.592 / 0.681 |
| v1 çift kanat | 386/41/188 | 0.904 / 0.672 / 0.771 |
| v2 çift kanat + dikdörtgen | 391/36/183 | 0.916 / 0.681 / 0.781 |
| v3 v2 + geniş boşlukta pencere | 439/64/135 | 0.873 / 0.765 / 0.815 |
| **v5 v3 + geniş boşlukta kısa ayak** | **465/70/109** | **0.869 / 0.810 / 0.839** |

Kapı beş varyantta da 421/10/134. Seçilen v5.

- Çift kanat: iki uçta menteşe, yarıçap yarım genişlik × (0.85-1.06),
  menteşe yüzde ya da orta çizgide, iki yan; skor iki yayın paylarının
  küçüğü, eşik kapıyla aynı (0.8).
- Kısa ayak: geniş boşluk ışını parçaya dönüşmemiş bir duvara (iki pencere
  arasındaki ayak, köşe çıkıntısı) çarpınca, çarpma noktasında enine duvar
  genişliği kalınlığın 0.5-1.5 katı (+6 px) ve ışına ortalıysa kabul.
  Yalnız pencere aramada; oda grafının geniş boşlukları değişmedi.

**Uçtan uca** (`degerlendir.py --stil dolu`): pencere 465/70/109, F1 0.839;
kutu IoU 369/166/205, K 0.690 / D 0.643 / F1 0.665 (önce 0.493). Duvar ve
kapı aynı. Oda 394/120/208, F1 0.706 (önce 393/118/209); kenar 121/200/313,
F1 0.321 (önce 0.322). Plan başına süre medyan 4.2 s.

Bütün stiller (`--stil hepsi`, 86 plan): pencere 465/122/360, F1 0.659
(önce 0.517); kutu IoU F1 0.523 (önce 0.379). Öteki sayılar §12 ile aynı;
oda 411/131/454. CSV'ler: `results/degerlendir_gelistirme_pencere.csv`
(dolu), `results/degerlendir_gelistirme_hepsi_pencere.csv` (hepsi).

**Kalan hatalar** (v5, 465/70/109):

| eksik (109) | | fazla (70) | |
|---|---|---|---|
| boşluk var, iki test de geçmedi | 46 | boş | 40 |
| aday yok | 27 | geometri | 20 |
| yalnız köşe boşluğu | 17 | cevapta kapı | 10 |
| geometri | 13 | | |
| bir yay tutmuş | 6 | | |

- **Garaj kapısı:** "boş" fazlaların 29'u cevapta Parking poligonunun
  içinde; hepsi geniş boşluktan. Dış duvardaki garaj açıklığı iç yüze
  oturmuş ince bir panel ve ortada enine çentikle çizilmiş; cevapta ne
  Door ne Window. Sürgülü pencereden ayrımı panelin açıklıktan birkaç
  piksel taşması. Kural yazılmadı.
- Geniş boşluktan gelenler: 74 doğru, 34 fazla. Çizgi sayısı ayırmıyor
  (2 çizgi: hem doğruların hem fazlaların çoğu); 350 px üstü 4 fazla, 0
  doğru. Eşik konmadı (dört örnek).
- Görüntüyle bakılan "iki test de geçmedi" örnekleri: yarı açık çizilmiş
  tek kanatlı pencere, sabit + kanatlı karma pencere (kanatlar boşluğun bir
  kısmında, çift kanat varsayımı tutmuyor).
