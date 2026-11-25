"""Arayüzü Hugging Face Spaces'e yükler (kararlar K-013).

Space deponun bir alt kümesi: ``app.py`` ve ``src/floorplan_parser``. Veri
seti, sonuçlar, rapor ve deney betikleri gitmez. Veri seti olmadığı için
Space'te "Sample photo" sentetik planı kullanır.

Dosyalar depodan doğrudan yüklenir, geçici klasöre kopyalanmaz (bu makinede
antivirüs %TEMP% altına .py yazılmasını engelliyor, otonomarac'ta görüldü).
Space'e özel README.md ve requirements.txt bellekten yüklenir. Hepsi tek
commit.

Önkoşul, bir kez: ``uv run --group arayuz hf auth login`` (Write yetkili
token, https://huggingface.co/settings/tokens).

    uv run --group arayuz python scripts/space_yukle.py --kuru   # yalnız listele
    uv run --group arayuz python scripts/space_yukle.py          # yükle
"""

import argparse
from importlib.metadata import version
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
SPACE = "byomer1021/floorplan-parser"
GITHUB = "https://github.com/Byomer1021/floorplan-parser"

# Space'in pip ile kuracağı paketler, yerelde test edilen sürümlerle. Linux
# konteynerinde ekran yok, OpenCV'nin başsız (headless) paketi.
PAKETLER = {
    "opencv-python-headless": "opencv-python",
    "scikit-image": "scikit-image",
    "scipy": "scipy",
    "numpy": "numpy",
    "networkx": "networkx",
    "matplotlib": "matplotlib",
    "pillow": "pillow",
}


def dosyalar() -> list[Path]:
    kod = sorted((KOK / "src" / "floorplan_parser").glob("*.py"))
    return [KOK / "app.py", *kod]


def readme() -> str:
    return f"""---
title: Floor Plan Parser
emoji: 🏠
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: {version("gradio")}
python_version: "3.12"
app_file: app.py
pinned: false
---

# Floor plan parser

Walls, doors, windows, rooms and the room–door graph from a photo of a
printed floor plan. Classical image processing only (thresholding,
morphology, Hough transform, template matching, watershed). No trained
models.

Upload a photo of a plan lying on a table. The paper is found and
rectified, then the plan is parsed. If the paper is not found, click its
four corners.

Runs on the free CPU tier. One photo takes several seconds.

Source code, decisions and measurements: [{GITHUB}]({GITHUB})
"""


def gereksinimler() -> str:
    satirlar = ["# scripts/space_yukle.py üretti, yerelde test edilen sürümler"]
    satirlar += [f"{ad}=={version(yerel)}" for ad, yerel in PAKETLER.items()]
    return "\n".join(satirlar) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kuru", action="store_true", help="Hub'a dokunmadan listele")
    ap.add_argument("--gizli", action="store_true", help="Space'i gizli aç")
    a = ap.parse_args()

    yuklenecek = dosyalar()
    print(f"Space: {SPACE}")
    for p in yuklenecek:
        print(f"  {p.relative_to(KOK).as_posix():40} {p.stat().st_size:>7} B")
    print("  README.md                                (bellekten)")
    print("  requirements.txt                         (bellekten)\n")
    print(gereksinimler())
    if a.kuru:
        print(readme())
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    try:
        api.whoami()
    except Exception as e:
        raise SystemExit("Hugging Face girişi yok: uv run --group arayuz hf auth login") from e
    api.create_repo(SPACE, repo_type="space", space_sdk="gradio", private=a.gizli, exist_ok=True)
    islemler = [
        CommitOperationAdd(path_in_repo=p.relative_to(KOK).as_posix(), path_or_fileobj=str(p))
        for p in yuklenecek
    ]
    islemler += [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme().encode()),
        CommitOperationAdd(
            path_in_repo="requirements.txt", path_or_fileobj=gereksinimler().encode()
        ),
    ]
    sonuc = api.create_commit(
        SPACE, repo_type="space", operations=islemler, commit_message="Arayüz güncellendi"
    )
    print(sonuc.commit_url)
    print(f"https://huggingface.co/spaces/{SPACE}")


if __name__ == "__main__":
    main()
