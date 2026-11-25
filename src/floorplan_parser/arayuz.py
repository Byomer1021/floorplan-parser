"""Arayüz: fotoğraf ya da tarama yükle, kâğıdın köşelerini gör ya da
işaretle, sonucu al (kararlar K-012).

Gradio yalnız ``uygulama()`` içinde yüklenir, çekirdek fonksiyonlar Gradio
olmadan test edilir. Ekrandaki metinler İngilizce (demo ve sunum İngilizce).

Başlatma: ``uv run --group arayuz python scripts/arayuz.py``
"""

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from floorplan_parser.akis import PlanSonucu, plani_coz
from floorplan_parser.foto import fotograftan_coz, kagit_bul, kose_sirala
from floorplan_parser.gorsel import VARSAYILAN_LEJANT, graf_sozluk, kapi_ortasi, katmanlar, lejant
from floorplan_parser.oda import DIS

ONIZLEME = 1600  # köşe ekranının uzun kenarı (piksel)
SONUC_BOYU = 2000  # sonuç görüntülerinin uzun kenarı
OTOMATIK, ELLE, TARAMA = "auto", "manual", "scan"


@dataclass
class Sonuc:
    sayfa: np.ndarray  # akışın çalıştığı gri sayfa (düzeltilmiş ya da tarama)
    plan: PlanSonucu
    kose: np.ndarray | None  # fotoğraftaki köşeler, tarama kipinde None
    kose_kaynagi: str  # OTOMATIK, ELLE ya da TARAMA
    sure: float


def gri_yap(goruntu: np.ndarray) -> np.ndarray:
    """RGB, RGBA ya da gri görüntüden gri uint8."""
    g = np.asarray(goruntu)
    if g.ndim == 3:
        g = cv2.cvtColor(g[..., :3], cv2.COLOR_RGB2GRAY)
    return g.astype(np.uint8)


def _kucult(g: np.ndarray, uzun: int) -> tuple[np.ndarray, float]:
    k = min(1.0, uzun / max(g.shape[:2]))
    if k == 1.0:
        return g.copy(), 1.0
    return cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), k


def onizleme(
    foto: np.ndarray, kose: np.ndarray | None, tiklanan: list[tuple[float, float]]
) -> tuple[np.ndarray, float]:
    """Köşe ekranı (RGB) ve küçültme oranı. Otomatik bulunan kâğıt yeşil
    dörtgen, kullanıcının tıkladığı noktalar numaralı turuncu. Dört nokta
    tamamlanınca turuncu dörtgen."""
    g, k = _kucult(gri_yap(foto), ONIZLEME)
    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    kalin = max(2, round(max(rgb.shape) / 400))
    if len(tiklanan) == 4:
        p = np.round(kose_sirala(np.array(tiklanan)) * k).astype(np.int32)
        cv2.polylines(rgb, [p], True, (255, 140, 0), kalin, cv2.LINE_AA)
    elif kose is not None:
        p = np.round(kose_sirala(kose) * k).astype(np.int32)
        cv2.polylines(rgb, [p], True, (0, 200, 0), kalin, cv2.LINE_AA)
    for i, (x, y) in enumerate(tiklanan, 1):
        merkez = (round(x * k), round(y * k))
        cv2.circle(rgb, merkez, 4 * kalin, (255, 140, 0), -1, cv2.LINE_AA)
        cv2.putText(
            rgb,
            str(i),
            (merkez[0] + 5 * kalin, merkez[1] - 5 * kalin),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * kalin,
            (255, 140, 0),
            kalin,
            cv2.LINE_AA,
        )
    return rgb, k


def coz(foto: np.ndarray, tiklanan: list | None = None, tarama: bool = False) -> Sonuc:
    """Tek girdi için uçtan uca. ``tiklanan`` dört nokta ise köşeler onlar.
    Değilse kâğıt aranır, bulunamazsa ValueError."""
    gri = gri_yap(foto)
    t0 = time.perf_counter()
    if tarama:
        return Sonuc(gri, plani_coz(gri), None, TARAMA, time.perf_counter() - t0)
    if tiklanan and len(tiklanan) == 4:
        kose, kaynak = kose_sirala(np.array(tiklanan, np.float64)), ELLE
    else:
        kose, kaynak = kagit_bul(gri), OTOMATIK
        if kose is None:
            raise ValueError("paper not found: click its four corners on the photo")
    s, sayfa, _ = fotograftan_coz(gri, kose)
    return Sonuc(sayfa, s, kose, kaynak, time.perf_counter() - t0)


def sonuc_goruntusu(r: Sonuc) -> np.ndarray:
    """Katmanlı sonuç (RGB): odalar, duvar, kapı, pencere, oda grafı."""
    g = katmanlar(r.sayfa, r.plan)
    lejant(g, VARSAYILAN_LEJANT, olcek=max(0.6, max(g.shape) / 2200))
    g, _ = _kucult(g, SONUC_BOYU)
    return cv2.cvtColor(g, cv2.COLOR_BGR2RGB)


def graf_goruntusu(r: Sonuc) -> np.ndarray:
    """Oda grafı (RGB): odalar düğüm (oda merkezinde), kapılar kenar. Dışarı
    açılan kapı gri halka. Arkada soluk sayfa."""
    s = r.plan
    h, w = r.sayfa.shape
    fig = Figure(figsize=(10, 10 * h / w), dpi=110)
    ax = fig.subplots()
    ax.imshow(r.sayfa, cmap="gray", alpha=0.25, vmin=0, vmax=255)
    merkez = {u: d["merkez"] for u, d in s.graf.nodes(data=True) if u != DIS}
    for u, v, d in s.graf.edges(data=True):
        kapi = kapi_ortasi(s.kapilar[d["kapi"]])
        uclar = [merkez[x] for x in (u, v) if x in merkez]
        for c in uclar:
            ax.plot([c[0], kapi[0]], [c[1], kapi[1]], color="black", lw=1.5, zorder=2)
        if DIS in (u, v):
            ax.scatter(*kapi, s=160, facecolors="none", edgecolors="gray", lw=2, zorder=3)
        ax.scatter(*kapi, s=30, color="#dc2828", zorder=4)
    for u, c in merkez.items():
        ax.scatter(*c, s=500, color="white", edgecolors="black", lw=1.5, zorder=5)
        ax.text(*c, f"R{u}", ha="center", va="center", fontsize=9, zorder=6)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()
    ax.set_title(
        f"{len(merkez)} rooms, {s.graf.number_of_edges()} door connections "
        "(red: door, gray ring: exit)"
    )
    fig.tight_layout()
    tuval = FigureCanvasAgg(fig)
    tuval.draw()
    return np.asarray(tuval.buffer_rgba())[..., :3].copy()


def ozet(r: Sonuc) -> str:
    """Sonucun kısa İngilizce özeti (Markdown)."""
    s = r.plan
    kaynak = {
        OTOMATIK: "paper found automatically",
        ELLE: "corners marked by hand",
        TARAMA: "scan, no paper detection",
    }[r.kose_kaynagi]
    n_oda = s.graf.number_of_nodes() - 1
    return (
        f"**{len(s.kapilar)} doors, {len(s.pencereler)} windows, {n_oda} rooms, "
        f"{s.graf.number_of_edges()} door connections** in {r.sure:.1f} s ({kaynak})."
    )


def json_sozluk(r: Sonuc) -> dict:
    """Graf JSON'u ve girdinin nasıl işlendiği. Koordinatlar düzeltilmiş
    sayfada (piksel, A4 300 dpi)."""
    d = graf_sozluk(r.plan)
    d["input"] = {
        "mode": r.kose_kaynagi,
        "paper_corners_in_photo": None
        if r.kose is None
        else [[round(float(x), 1), round(float(y), 1)] for x, y in r.kose],
        "page_size_px": [int(r.sayfa.shape[1]), int(r.sayfa.shape[0])],
        "time_s": round(r.sure, 2),
    }
    return d


def foto_oku(yol: str | Path) -> np.ndarray:
    """Fotoğraf dosyası → RGB. Telefon fotoğrafı çoğu zaman yan kaydedilir
    ve doğru yönü EXIF etiketinde durur. Yön burada uygulanır."""
    from PIL import Image, ImageOps

    with Image.open(yol) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def ornek_foto(tohum: int | None = None) -> np.ndarray:
    """Kamerasız deneme için sentetik fotoğraf (RGB): örnek ev planı
    (``sentetik.ev_ciz``), rastgele 15-35 derece eğimle. Plan kodla çizildiği
    için lisans sorunu yok, yerelde ve Space'te aynı örnek."""
    from floorplan_parser.foto_benzetim import FotoAyar, foto_uret
    from floorplan_parser.sentetik import ev_ciz

    rng = np.random.default_rng(tohum)
    ayar = FotoAyar(egim=float(rng.uniform(15, 35)), tohum=int(rng.integers(10**9)))
    f = foto_uret(ev_ciz().goruntu, ayar)
    return cv2.cvtColor(f.goruntu, cv2.COLOR_GRAY2RGB)


def uygulama():
    """Gradio arayüzü (``gr.Blocks``)."""
    import gradio as gr

    def kose_ekrani(foto, tarama):
        if foto is None:
            return None, [], 1.0, ""
        if tarama:
            g, k = onizleme(foto, None, [])
            return g, [], k, "Scan mode: the whole image is used as the page."
        kose = kagit_bul(gri_yap(foto))
        g, k = onizleme(foto, kose, [])
        mesaj = (
            "Paper found (green). Click four corners on the image to override."
            if kose is not None
            else "Paper not found. Click its four corners on the image, then Parse."
        )
        return g, [], k, mesaj

    def yuklendi(yol, tarama):
        foto = foto_oku(yol) if yol else None
        return foto, *kose_ekrani(foto, tarama)

    def ornek_yaz():
        yol = Path(tempfile.mkdtemp()) / "sample_photo.jpg"
        cv2.imwrite(str(yol), cv2.cvtColor(ornek_foto(), cv2.COLOR_RGB2BGR))
        return str(yol)

    def tiklandi(foto, tiklanan, olcek, evt: gr.SelectData):
        if foto is None:
            return None, tiklanan, ""
        x, y = evt.index
        tiklanan = ([] if len(tiklanan) >= 4 else list(tiklanan)) + [(x / olcek, y / olcek)]
        g, _ = onizleme(foto, None, tiklanan)
        kalan = 4 - len(tiklanan)
        mesaj = f"{kalan} more corner(s)." if kalan else "Four corners set (orange). Press Parse."
        return g, tiklanan, mesaj

    # Gradio ilerleme çubuğunu varsayılan değerden tanıyor.
    def calistir(foto, tiklanan, tarama, ilerleme=gr.Progress()):  # noqa: B008
        if foto is None:
            raise gr.Error("Upload a photo or a scan first.")
        ilerleme(0.1, desc="Parsing")
        try:
            r = coz(foto, tiklanan, tarama)
        except ValueError as e:
            raise gr.Error(str(e)) from e
        d = json_sozluk(r)
        yol = Path(tempfile.mkdtemp()) / "floorplan_graph.json"
        yol.write_text(json.dumps(d, indent=1), encoding="utf-8")
        sayfa = cv2.cvtColor(_kucult(r.sayfa, SONUC_BOYU)[0], cv2.COLOR_GRAY2RGB)
        return sonuc_goruntusu(r), graf_goruntusu(r), sayfa, d, str(yol), ozet(r)

    with gr.Blocks(title="Floor plan parser") as demo:
        gr.Markdown(
            "# Floor plan parser\n"
            "Walls, doors, windows, rooms and the room–door graph from a photo of a "
            "printed floor plan. Classical image processing only, no trained models."
        )
        foto = gr.State(None)  # RGB, EXIF yönü uygulanmış
        tiklanan = gr.State([])
        olcek = gr.State(1.0)
        with gr.Row():
            with gr.Column(scale=1):
                giris = gr.Image(
                    label="Photo or scan",
                    type="filepath",
                    sources=["upload", "webcam", "clipboard"],
                )
                tarama = gr.Checkbox(label="Input is a scan (skip paper detection)", value=False)
                with gr.Row():
                    ornek_d = gr.Button("Sample photo")
                    sifirla_d = gr.Button("Reset corners")
                coz_d = gr.Button("Parse", variant="primary")
                mesaj = gr.Markdown()
                ozet_md = gr.Markdown()
            with gr.Column(scale=2), gr.Tabs():
                with gr.Tab("Paper corners"):
                    kose_g = gr.Image(label="Click four corners to override", interactive=False)
                with gr.Tab("Result"):
                    sonuc_g = gr.Image(label="Result", interactive=False)
                with gr.Tab("Room graph"):
                    graf_g = gr.Image(label="Room graph", interactive=False)
                with gr.Tab("Rectified page"):
                    sayfa_g = gr.Image(label="Rectified page", interactive=False)
                with gr.Tab("JSON"):
                    json_g = gr.JSON(label="Graph")
                    dosya = gr.File(label="Download")

        ekran = [kose_g, tiklanan, olcek, mesaj]
        giris.change(yuklendi, [giris, tarama], [foto, *ekran])
        tarama.change(kose_ekrani, [foto, tarama], ekran)
        sifirla_d.click(kose_ekrani, [foto, tarama], ekran)
        kose_g.select(tiklandi, [foto, tiklanan, olcek], [kose_g, tiklanan, mesaj])
        ornek_d.click(ornek_yaz, None, giris)
        coz_d.click(
            calistir,
            [foto, tiklanan, tarama],
            [sonuc_g, graf_g, sayfa_g, json_g, dosya, ozet_md],
        )
    return demo
