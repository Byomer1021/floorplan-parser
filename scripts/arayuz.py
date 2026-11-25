"""Arayüzü başlatır (kararlar K-012).

    uv run --group arayuz python scripts/arayuz.py            # bu bilgisayarda
    uv run --group arayuz python scripts/arayuz.py --ag       # aynı Wi-Fi'daki telefondan

Tarayıcıda http://127.0.0.1:7860 açılır. ``--ag`` ile bilgisayarın yerel ağ
adresinden de erişilir (ör. http://192.168.1.20:7860). Telefonda dosya
yükleme kamerayı da açar. Windows güvenlik duvarı ilk seferde izin
isteyebilir. İnternete açık bir bağlantı (Gradio share) kullanılmıyor.
"""

import argparse

from floorplan_parser.arayuz import uygulama


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag", action="store_true", help="yerel ağdan erişime aç (0.0.0.0)")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    uygulama().launch(
        server_name="0.0.0.0" if a.ag else "127.0.0.1",
        server_port=a.port,
        share=False,
    )


if __name__ == "__main__":
    main()
