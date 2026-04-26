"""
generate_qr.py
--------------
Run this once you have the Streamlit Cloud URL to produce a QR code image.

Usage:
    python generate_qr.py --url https://yourname-fraud-demo.streamlit.app
"""
import argparse
from pathlib import Path

import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer

def make_qr(url: str, out: str = "demo_qr.png"):
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=12,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=RoundedModuleDrawer(),
        back_color="white",
        fill_color="#1a1a2e",
    )
    img.save(out)
    print(f"[QR] Saved to: {out}")
    print(f"[QR] URL: {url}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Streamlit app URL")
    parser.add_argument("--out", default="demo_qr.png", help="Output image path")
    args = parser.parse_args()
    make_qr(args.url, args.out)
