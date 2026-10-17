# Veri

Veri seti depoya girmez (`data/raw/`, `data/processed/` .gitignore'da):
lisansı ayrı ve boyutu büyük. Seçim ve gerekçe: [K-003](../docs/kararlar.md).

```
data/raw/<set_adi>/       indirilen arşivin açılmış hali, dokunulmaz
data/processed/           koddan yeniden üretilen her şey
```

## CVC-FP (kullanılıyor)

```
uv run python scripts/cvcfp_indir.py
```

| | |
|---|---|
| kaynak sayfa | https://dag.cvc.uab.es/dataset/cvc-fp-database-for-structural-floor-plan-analysis/ |
| arşiv | http://dag.cvc.uab.es/DATASETS/CVC-FP.zip (yalnız http) |
| indirme | 2026-10-17, 27 399 177 bayt, sunucuda son değişiklik 2025-09-16 |
| sha256 | `3c636ed188fb4877f7070e138ae500c1105add561d913f5a359725d03d4c881a` |
| içerik | `ImagesGT/`: 123 görüntü (119 png, 4 jpg), 122 SVG cevap; `image001a`'nın cevabı yok |
| lisans | sayfada "Creative Commons Attribution 4.0 International License (CC BY-NC 4.0)" yazıyor: ad BY, kısaltma BY-NC, kendi içinde çelişkili. Ticari olmayan kullanım varsayılıyor (ders projesi). |
| atıf | de las Heras, Terrades, Robles, Sánchez. *CVC-FP and SGT: a new database for structural floor plan analysis and its groundtruthing tool.* IJDAR, 2015. |

Sitenin TLS sertifikası doğrulanamadı (zincir eksik); sayfa doğrulamasız
okundu, arşiv http ile indi. Elimizdeki kopyanın bütünlüğü sha256 ile
sabit.

**Cevap biçimi:** her nesne `class`'lı bir `<polygon>`: Wall 5845, Room 1239,
Window 1201, Door 1144, Separation 244, Parking 81, Text 1. İlişkiler
`<relation type objects>`: incident 9863, neighbour 1997, access 1374,
surround 1225, outerP 140. Okuyucu: `floorplan_parser.cvcfp`.

**Gruplar** (dosya adından; makaledeki alt küme adlarıyla eşleşmesi
doğrulanmadı, makale okunmadı):

| grup | dosya adı | plan | boyut (px) | duvar çizimi (gözle + ölçüm, [olcumler §3](../docs/olcumler.md)) |
|---|---|---|---|---|
| I-II | `Ia_…`, `IIa_…` vb. | 90 | 2479-3508 | 86'sında dolu siyah; 4'ünde içi boş |
| sayi | `1`…`10` | 10 | 905-2227 | taralı |
| image | `image0NN` | 18 | 5671x7383 | taralı dolgu, ince kenar |
| p | `p1`…`p4` | 4 | 2550x3300 | çift çizgi, içi boş |

**Bölme** ([K-005](../docs/kararlar.md)): grup başına %30 test, ada göre
sha1 sırasıyla; 86 geliştirme, 36 test. `cvcfp.ornekler(bolum=...)`.

## Aday, alınmadı

- **CubiCasa5K:** 5000 plan, 80+ sınıf poligon cevabı.
  https://zenodo.org/records/2613548 — lisansı ve bağlantısı doğrulanmadı.
  Stil çeşitliliği gerekirse.
