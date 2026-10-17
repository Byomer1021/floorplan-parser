"""Raster kat planı çözümleme: klasik görüntü işleme, öğrenilmiş model yok.

Akış (her adım ayrı modül):

1. ``goruntu``  — okuma/yazma
2. ``onisleme`` — ikileme, açma/kapama, kalın/ince çizgi ayrımı
3. duvar        — Hough ile düz duvar parçaları (yazılmadı)
4. açıklık      — kapı yayı (Hough çember / şablon), pencere (yazılmadı)
5. graf         — duvar/oda/kapı ilişkileri, networkx (yazılmadı)

``sentetik`` cevabı bilinen test planları çizer.
"""
