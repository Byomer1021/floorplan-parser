# Rapor (IEEE, İngilizce)

`main.tex` IEEEtran konferans şablonu; kaynakça `refs.bib`, şekiller
`sekiller/`. Teslimatlar İngilizce (ders: Görüntü İşleme).

## Derleme

Yerelde (LaTeX kurulumu yok; tectonic taşınabilir, `D:\tools\tectonic`):

```
cd rapor
set TECTONIC_CACHE_DIR=D:\tools\tectonic\cache
D:\tools\tectonic\tectonic.exe -X compile main.tex
```

İlk derleme gereken paketleri indirir (önbellek D'de). Overleaf'te: bu
klasörü (main.tex, refs.bib, sekiller/) yükleyip pdfLaTeX ile derle.

## Şekiller

```
uv run python scripts/sekiller.py            # dördü
uv run python scripts/sekiller.py --yalniz pipeline,failures
```

Sayılar `docs/olcumler.md`'den elle girildi (§16 geliştirme, §17 test);
ölçüm değişirse tablolar elle güncellenir.

## Sunum ve demo

- Sunum (İngilizce, 14 slayt, konuşmacı notlu) repo dışında tutuluyor.
  Görselleri
  `scripts/sekiller.py` üretir (`slide_input.png`, `slide_output.png` ve
  rapor şekilleri).
- Demo: `uv run python scripts/demo.py --cvcfp IIa_LZ1101` (sunumdaki
  plan). Sunumdan önce bir kez çalıştırılmalı.

## Sürümler

Buradaki rapor tek yazarlı (Ömer Can Atlı). Derse teslim edilen sürüm aynı
metnin ekip yazar bloğuyla derlenmiş hâli; repo dışında tutuluyor.
