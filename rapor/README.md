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

## Açık

- Yazar bloğunda yalnız Ömer Can var; ekip üyeleri eklenecek.
