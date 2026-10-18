# Literatür kıyası

2026-10-18. Soru: aynısı ya da çok benzeri yapılmış mı, yapıldıysa sayıları
bizimkiyle nasıl kıyaslanıyor? Her sayının yanında kaynağı ve onu kimin
okuduğu yazılı. "Okundu" kaynağın metni açılıp tablosu bizzat okundu
demek. "İkinci el" başka bir yayının aktarımı demek. Bizim sayılarımız
geliştirme bölümünden ([olcumler §12](olcumler.md)). Test bölümü henüz
ölçülmedi.

## Kısa sonuç

- **Birebir aynısı yok.** CVC-FP üzerinde duvar + kapı + pencere + oda +
  oda-kapı grafını öğrenilmiş model kullanmadan çıkaran ve hepsini
  ölçen tek bir yayın bulamadık. En yakını de las Heras vd. 2014; tam
  akış, aynı veri. Metnine erişilemedi ([aşağıda](#erişilemeyenler)).
- **Duvar, dolu duvarlı planlar:** bulduğumuz tek klasik sayının üstündeyiz:
  0.908, Swaileh vd. 2021 klasik maske üretimi 0.831. Kurulumlar aynı
  değil.
- **Duvar, bütün stiller:** en basit klasik taban çizgisi düzeyindeyiz.
  Ortalama 0.666, havuzlanmış 0.407. Dodge 2017'nin Otsu + morfoloji taban
  çizgisi 0.619, aynı veride öğrenilmiş yöntemler 0.77-0.89. Taralı ve
  çift çizgili 24 planda yöntem neredeyse hiç duvar bulmuyor.
- **Kapı:** dolu planlarda kesinlik çok yüksek (0.981), duyarlılık
  FloorYOLO'ya yakın (0.762 / 0.786). Bütün stillerde duyarlılık 0.542'ye
  düşüyor.
- **Pencere:** FloorYOLO'nun altındayız. Kutu IoU'lu F1 0.665, onlarınki
  yaklaşık 0.76. Bu belge ilk yazıldığında 0.49'du. Çift kanatlı ve geniş
  pencereler eklendikten sonra yükseldi ([olcumler §13](olcumler.md)).
- **Oda:** Ahmed vd. ve Macé vd. bizim dolu grubumuzla aynı plan ailesinde
  çalışmış, ama ölçüt farklı: kısmi eşleşmeye puan veren DR/RA. Sayılar
  doğrudan kıyaslanamaz (ayrıntı aşağıda).
- **Kapı kenarı (oda-kapı-oda):** CVC-FP üzerinde yayımlanmış bir sayı
  bulamadık.

## Bizim sayılar

`scripts/degerlendir.py --bolum gelistirme --stil {dolu,hepsi}`. Eşleme:
bire bir, IoU ≥ 0.5, kısmi kredi yok. K = kesinlik, D = duyarlılık.

| | dolu duvarlı (62 plan) | bütün stiller (86 plan) |
|---|---|---|
| duvar piksel IoU, plan ortalaması | 0.908 | 0.666 |
| duvar piksel IoU, havuzlanmış | 0.906 | 0.407 |
| kapı, poligon IoU: K / D / F1 | 0.977 / 0.759 / 0.854 | 0.968 / 0.539 / 0.693 |
| kapı, kutu IoU: K / D / F1 | 0.981 / 0.762 / 0.858 | 0.973 / 0.542 / 0.696 |
| pencere, poligon IoU: K / D / F1 | 0.790 / 0.584 / 0.671 | 0.710 / 0.406 / 0.517 |
| pencere, kutu IoU: K / D / F1 | 0.580 / 0.429 / 0.493 | 0.521 / 0.298 / 0.379 |
| oda, bölge IoU: K / D / F1 | 0.769 / 0.653 / 0.706 | 0.761 / 0.474 / 0.584 |
| kapı kenarı: K / D / F1 | 0.381 / 0.279 / 0.322 | 0.376 / 0.192 / 0.254 |

Havuzlanmış IoU, bütün planların kesişim pikselleri toplamının birleşim
pikselleri toplamına oranı. Bütün stillerde ortalamadan çok düşük: en büyük
planlar (`image` grubu, 5671x7383) IoU'su 0 olan taralı planlar.

## Duvar

| çalışma | yöntem | öğrenme | veri | ölçüt | sayı |
|---|---|---|---|---|---|
| Dodge vd. 2017, okundu | Otsu (GT) | yok | CVC-FP 122, 5 katlı çapraz doğrulama | duvar JI | 41.5 |
| 〃 | Otsu + morfolojik işlem (GT+MC) | yok | 〃 | 〃 | 61.9 |
| 〃 | GT+MC + Google Vision ile metin silme | metin silmede var | 〃 | 〃 | 71.2 |
| 〃 | yama + rastgele orman | var | 〃 | 〃 | 79.2 |
| 〃 | FCN-2s | var | 〃 | 〃 | 89.2 |
| de las Heras vd. 2013 (Cabrera Vargas 2018'de ikinci el) | yama tabanlı, stil başına eğitim | var | CVC-FP | JI | 86.1 ve 97.14 |
| Swaileh vd. 2021, okundu | bileşen filtresi + yönlü filtre + Otsu | yok | CVC-FP dolu duvarlı (%70), arka planı Versailles dokusuyla değiştirilmiş | IoU | 83.05 |
| 〃 | U-net | var | CVC-FP 122, 5 katlı | ortalama IoU (iki sınıf) | 95.01 |
| **bu çalışma** | Otsu + açma + iki ölçek + kurallar | yok | CVC-FP dolu duvarlı, geliştirme 62 | duvar IoU, plan ortalaması | **90.8** |
| **bu çalışma** | 〃 | yok | CVC-FP bütün stiller, geliştirme 86 | duvar IoU, ortalama / havuzlanmış | **66.6 / 40.7** |

Kıyas notları:

- **JI = duvar sınıfının IoU'su.** Dodge'un tablosundaki "Mean IoU"
  (FCN-2s 94.4) duvar ve arka plan sınıflarının ortalaması. Arka plan
  kolay olduğu için bu sayı şişer. Kıyasta JI kullanıldı. JI'nin plan
  ortalaması mı, havuzlanmış mı olduğunu Dodge yazmıyor. Bu yüzden iki
  hâlimizi de verdik.
- **Bizim bütün stiller sayımız 122 değil, 86 plan.** 36 test planına
  bakılmadı. Bizim eşiklerimiz bu 86 planda seçildi, onlarınki çapraz
  doğrulamayla. Bu bize avantaj sağlıyor.
- **Swaileh'in dolu duvarlı kümesi** 122 planın %70'i. Bizim 86 dolu
  planımız da %70.5. Muhtemelen aynı alt küme, ama doğrulamadık. Onların
  girdisi arka planı renkli dokuyla değiştirilmiş CVC-FP. Bizimki özgün
  beyaz arka plan. "IoU"nun iki sınıf ortalaması mı, yalnız duvar mı
  olduğu metinden kesin çıkmıyor. İki sınıf ortalamasıysa duvar IoU'ları
  daha düşük.
- **Dodge'un GT+MC'si bizim ön işlemeye en yakın taban çizgisi** (Otsu +
  morfoloji). Bütün stillerde 61.9. Bizim plan ortalamamız 66.6, ama
  havuzlanmış değerimiz 40.7 ve onun düz Otsu'su 41.5. Hangi tanımı
  kullandıklarını bilmeden "üstündeyiz" diyemeyiz.
- **de las Heras'ın 97.14'ü** muhtemelen tek bir stilde. Hangi alt
  kümede olduğunu ikinci el kaynaktan çıkaramadık; makale açılamadı.

## Kapı ve pencere

| çalışma | yöntem | veri | ölçüt | kapı K / D | pencere K / D |
|---|---|---|---|---|---|
| Santhosh vd. 2026 (FloorYOLO), okundu, aktarımla | YOLOv8n + dikkat bloğu | CVC-FP 123, 98 eğitim / 25 doğrulama, 640 px giriş | kutu, doğrulama kümesi | 0.810 / 0.786 | 0.791 / 0.727 |
| Ziran ve Marinai 2018 (FloorYOLO'da ikinci el) | Faster R-CNN, SSD | CVC-FP | mAP | 0.31 (d1) / 0.39 (d2) | |
| **bu çalışma, dolu** | boşluk + çeyrek yay şablonu, paralel çizgi | geliştirme 62 | kutu IoU ≥ 0.5 | **0.981 / 0.762** | **0.580 / 0.429** |
| **bu çalışma, bütün stiller** | 〃 | geliştirme 86 | 〃 | **0.973 / 0.542** | **0.521 / 0.298** |
| **bu çalışma, dolu, §13 sonrası** | + çift kanat, geniş boşluk | geliştirme 62 | 〃 | 0.981 / 0.762 | **0.690 / 0.643** |
| **bu çalışma, bütün stiller, §13 sonrası** | 〃 | geliştirme 86 | 〃 | 0.973 / 0.542 | **0.629 / 0.447** |

Notlar:

- FloorYOLO'nun F1'i makalede sınıf başına verilmiyor. K ve D'den
  hesaplanan değerler: kapı ≈ 0.80, pencere ≈ 0.76. Bunlar bizim
  hesabımız. K/D'nin hangi IoU eşiğinde verildiğini doğrulamadık. YOLO
  araçlarının varsayılanı 0.5.
- FloorYOLO bütün stillerde, 25 planlık doğrulama kümesinde ölçüyor; bizim
  kapı sayımız ise yalnız dolu planlarda ona yakın. Bütün stillerde kapı
  duyarlılığımız 0.542, yani belirgin altında. Taralı planlarda duvar
  bulunamayınca boşluk da bulunamıyor.
- Pencerede kutu IoU, poligon IoU'dan düşük çıkıyor: 246 ve 335 doğru.
  Nedenini incelemedik.
- Rezvanifar 2021 (doktora tezi, okundu) kapı yayını kısmi elips oylamasıyla
  arıyor; bizim çeyrek yay şablonumuza en yakın klasik yöntem bu. Ölçümü
  SESYD sentetik verisinde ve kendi gerçek planlarında; CVC-FP sayısı yok.

## Oda

| çalışma | yöntem | veri | ölçüt | sayı |
|---|---|---|---|---|
| Macé vd. 2010 (Ahmed 2011'de ikinci el) | kalın/ince çizgi ayrımı + oda bulma (ayrıntısı okunmadı) | 80 plan, 2479x3508 | DR / RA | 85 / 69 |
| Ahmed vd. 2011, okundu | kalın/orta/ince çizgi ayrımı, dış duvarların dışbükey zarfı dışını silme, boşluk kapatma | aynı 80 plan | DR / RA | 89 / 79 |
| Ahmed vd. 2012, okundu | 2011 + OCR'lı oda etiketiyle duvarsız bölme | aynı 80 plan | DR / RA | 85 / 82 |
| **bu çalışma** | kirişle boşluk kapatma + ayak izi + havza bölütleme | CVC-FP dolu duvarlı, geliştirme 62 | bire bir IoU ≥ 0.5, D / K | **65.3 / 76.9** |
| **bu çalışma, §14 sonrası** | + kesik hattaki uçtan kiriş yok, yarı açık kapı kirişi | 〃 | 〃 | **65.1 / 84.1** |

Neden doğrudan kıyaslanamaz:

- **Aynı plan ailesi.** Ahmed'in planları 2479x3508 ve aynı kaynaktan
  (Macé 2010). Bizim I-II grubumuz da bu boyutta (90 plan). CVC-FP bu
  ailenin genişletilmiş ve yeniden etiketlenmiş hâli. Planlar büyük
  olasılıkla örtüşüyor, ama cevaplar (oda sınırları) aynı değil.
- **Ölçüt farklı.** DR/RA Phillips-Chhabra protokolü. Bire bir eşleşmenin
  yanında bire çok ve çoka bir eşleşmeye de kısmi puan veriyor. Bizim
  ölçütte iki odayı tek oda bulmak bir eksik + bir fazla sayılır, puan
  sıfır. Bizim kaçan 209 odanın 79'u tam olarak bu durumda (duvarsız
  ayrım, [olcumler §11](olcumler.md)). DR'de bunlar kısmen sayılırdı.
- Ahmed 2012'nin "duvarsız bölme" için kullandığı yol OCR ile oda adını
  okumak. Bizde duvarsız ayrımlar bulunamıyor ve bu yolu kullanmıyoruz.
- Kıyas için protokolü bizim çıktıya uygulamak gerekir. Protokolün
  ağırlıkları ve eşikleri Ahmed'in metninde yok (Phillips-Chhabra 1999'a
  gönderiyor). Uygulamadık.

## Kapı kenarı

CVC-FP cevabında oda-kapı-oda ilişkisi var (`access`, 1374 ilişki). Bu
ilişkiyi ölçen yayımlanmış bir sayı bulamadık. CVC-FP'nin 2015 makalesi bu
ilişkileri tanımlıyor ama açılamadı.

## Erişilemeyenler

Sayılarını doğrulayamadık:

- de las Heras, Ahmed, Liwicki, Valveny, Sánchez. *Statistical segmentation
  and structural recognition for floor plan interpretation.* IJDAR 17,
  2014. Tam akış, CVC-FP; bize en yakın çalışma. refbase.cvc.uab.es ve
  .cat bağlantıyı reddetti, ResearchGate ve DeepDyve'da tam metin yok.
- de las Heras, Terrades, Robles, Sánchez. *CVC-FP and SGT.* IJDAR, 2015.
  Veri setinin makalesi.
- de las Heras vd. 2013 iki ICDAR/GREC makalesi (HSL2013, HFV2013):
  refbase erişilemedi.
- de las Heras doktora tezi (tesisenred.net, DNS çözülmedi).
- Macé vd. 2010 kendi metni (HAL sayfası PDF yerine HTML döndü).
- Ziran ve Marinai 2018 (flore.unifi.it 403).

## Kaynaklar

- Dodge, Xu, Stenger. *Parsing floor plan images.* MVA 2017.
  <https://www.mva-org.jp/Proceedings/2017USB/papers/10-01.pdf>
- Ahmed, Liwicki, Weber, Dengel. *Improved automatic analysis of
  architectural floor plans.* ICDAR 2011.
  <https://www-live.dfki.de/fileadmin/user_upload/import/5675_4520a864.pdf>
- Ahmed, Liwicki, Weber, Dengel. *Automatic room detection and room
  labeling from architectural floor plans.* DAS 2012.
  <https://www-live.dfki.de/fileadmin/user_upload/import/6204_4661a339.pdf>
- Swaileh, Kotzinos, Ghosh, Jordan, Vu, Qian. *Versailles-FP dataset: wall
  detection in ancient floor plans.* ICDAR 2021.
  <https://arxiv.org/pdf/2103.08064>
- Santhosh, Rashmika, Sasithradevi, Nathan. *FloorYOLO: an
  attention-enhanced YOLOv8n for floor plan object recognition.* Frontiers
  in Artificial Intelligence 9, 2026.
  <https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1911589/full>
- Cabrera Vargas. *Wall extraction and room detection for multi-unit
  architectural floor plans.* Yüksek lisans tezi, University of Victoria,
  2018.
  <https://dspace.library.uvic.ca/bitstreams/189b0857-eaf8-49d0-b6b1-11c768e85ce4/download>
- Rezvanifar. *Analyzing symbols in architectural floor plans via
  traditional computer vision and deep learning approaches.* Doktora tezi,
  University of Victoria, 2021.
  <https://dspace.library.uvic.ca/bitstreams/5312c723-30a8-410f-a075-5e7600b3efb4/download>
- Macé, Locteau, Valveny, Tabbone. *A system to detect rooms in
  architectural floor plan images.* DAS 2010.
  <https://hal.inria.fr/inria-00493393/document> (açılmadı)
