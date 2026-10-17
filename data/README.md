# Veri

Veri seti depoya girmez (`data/raw/`, `data/processed/` .gitignore'da):
lisansı ayrı ve boyutu büyük. Seçim ve gerekçe: [K-003](../docs/kararlar.md).

Beklenen düzen:

```
data/raw/<set_adi>/       indirilen arşivin açılmış hali, dokunulmaz
data/processed/           koddan yeniden üretilen her şey
```

## Adaylar

- **CVC-FP:** 122 taranmış plan, duvar/kapı/pencere/oda cevabı SVG.
  https://dag.cvc.uab.es/dataset/cvc-fp-database-for-structural-floor-plan-analysis/
- **CubiCasa5K:** 5000 plan, 80+ sınıf poligon cevabı.
  https://zenodo.org/records/2613548

İndirme bağlantıları ve lisanslar henüz doğrulanmadı. İndirildiğinde buraya:
kaynak URL, indirme tarihi, arşivin sha256'sı, lisans.
