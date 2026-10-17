"""CVC-FP'yi indirir, sha256'yı doğrular, data/raw/cvc-fp/ altına açar.

Kaynak sayfa: https://dag.cvc.uab.es/dataset/cvc-fp-database-for-structural-floor-plan-analysis/
Arşiv yalnız http ile sunuluyor; bütünlük 2026-10-17'de alınan kopyanın
sha256'sıyla denetlenir. Özet tutmazsa açılmaz.

Kullanım: uv run python scripts/cvcfp_indir.py
"""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "http://dag.cvc.uab.es/DATASETS/CVC-FP.zip"
SHA256 = "3c636ed188fb4877f7070e138ae500c1105add561d913f5a359725d03d4c881a"
HEDEF = Path(__file__).resolve().parents[1] / "data" / "raw" / "cvc-fp"


def main() -> None:
    arsiv = HEDEF / "_arsiv" / "CVC-FP.zip"
    if not arsiv.exists():
        arsiv.parent.mkdir(parents=True, exist_ok=True)
        print(f"indiriliyor: {URL}")
        urllib.request.urlretrieve(URL, arsiv)
    ozet = hashlib.sha256(arsiv.read_bytes()).hexdigest()
    if ozet != SHA256:
        sys.exit(
            f"sha256 tutmuyor: {ozet}\nbeklenen:        {SHA256}\n{arsiv} silinip yeniden denenebilir."
        )
    with zipfile.ZipFile(arsiv) as z:
        for ad in z.namelist():
            # Arşiv dışına yazan yol olmasın.
            if Path(ad).is_absolute() or ".." in Path(ad).parts:
                sys.exit(f"beklenmeyen yol arşivde: {ad}")
        z.extractall(HEDEF)
    n = len(list((HEDEF / "ImagesGT").glob("*.svg")))
    print(f"tamam: {HEDEF / 'ImagesGT'} ({n} cevap dosyası)")


if __name__ == "__main__":
    main()
