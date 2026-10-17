import numpy as np
import pytest

from floorplan_parser.goruntu import oku, yaz
from floorplan_parser.sentetik import plan_ciz


def test_ascii_olmayan_yolda_gidip_gelir(tmp_path):
    p = plan_ciz()
    yol = tmp_path / "çizimler" / "kat_planı_ü.png"
    yaz(yol, p.goruntu)
    assert np.array_equal(oku(yol), p.goruntu)


def test_olmayan_dosya_hata_verir(tmp_path):
    with pytest.raises(FileNotFoundError):
        oku(tmp_path / "yok.png")


def test_bozuk_dosya_hata_verir(tmp_path):
    yol = tmp_path / "bozuk.png"
    yol.write_bytes(b"png degil")
    with pytest.raises(ValueError):
        oku(yol)
