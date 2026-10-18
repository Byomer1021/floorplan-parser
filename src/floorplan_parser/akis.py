"""Uçtan uca akış: gri plan görüntüsü → duvarlar, kapılar, pencereler, oda grafı.

Her adım kendi modülünde; burada varsayılan ayarlarla sırayla çağrılır
(kararlar K-004, K-006, K-007, K-008, K-009).
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np

from floorplan_parser.aciklik import (
    VARSAYILAN_ACIKLIK_KURAL,
    AciklikKural,
    Bosluk,
    Kapi,
    Pencere,
    aciklik_bul,
    bosluklar,
    genis_bosluklar,
)
from floorplan_parser.degerlendirme import parca_maskesi
from floorplan_parser.duvar import Parca, iki_olcekli_duvarlar
from floorplan_parser.oda import VARSAYILAN_ODA_KURAL, OdaKural, oda_grafi
from floorplan_parser.onisleme import delik_doldur, ikile


@dataclass
class PlanSonucu:
    duvar_maske: np.ndarray  # 255 duvar
    kalin_parcalar: list[Parca]
    ince_parcalar: list[Parca]
    kapilar: list[Kapi]
    pencereler: list[Pencere]
    graf: nx.MultiGraph  # düğüm: oda (ve DIS), kenar: kapı
    oda_etiket: np.ndarray  # 0 duvar/kiriş, DIS dış, 1.. oda


def plani_coz(
    gri: np.ndarray,
    oda_kural: OdaKural = VARSAYILAN_ODA_KURAL,
    aciklik_kural: AciklikKural = VARSAYILAN_ACIKLIK_KURAL,
) -> PlanSonucu:
    """Varsayılan akış. Oda bölmede kapı ve pencerelerin yanında parça
    uçlarından bulunan bütün boşluklar ve aynı doğrultudaki geniş açıklıklar
    (150-450 px) kapatılır; bina sınırı geniş açıklıklar kapanmış duvardan."""
    ikili = delik_doldur(ikile(gri), 4)
    kalin, kalin_p, ince_p = iki_olcekli_duvarlar(ikili)
    parcalar = kalin_p + ince_p
    duvar = np.maximum(kalin, parca_maskesi(ince_p, gri.shape))
    # Yarı açık kapılar yalnız oda bölmede kiriş (kapı listesine girmez); aynı
    # taramada kaydedilir.
    kismi_kapilar: list[Kapi] | None = [] if oda_kural.kismi_kapi_esik <= 1.0 else None
    kapilar, pencereler, _ = aciklik_bul(
        ikili,
        parcalar,
        duvar,
        aciklik_kural,
        kismi_kayit=kismi_kapilar,
        kismi_kayit_esik=oda_kural.kismi_kapi_esik,
    )
    genis = genis_bosluklar(parcalar, duvar, kisa_duvar=oda_kural.genis_kisa_duvar)
    parca = bosluklar(
        parcalar, duvar, yalniz_hizali=oda_kural.hizali_bosluk, yan_suzgec=oda_kural.yan_suzgec
    )
    kismi = [Bosluk(k.mentese, k.kapali_uc, 0.0) for k in kismi_kapilar or []]
    graf, etiket = oda_grafi(
        duvar,
        kapilar,
        pencereler,
        oda_kural,
        bosluklar=parca + genis + kismi,
        ayak_bosluklari=genis,
    )
    return PlanSonucu(duvar, kalin_p, ince_p, kapilar, pencereler, graf, etiket)
