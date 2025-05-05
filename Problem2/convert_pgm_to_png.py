#!/usr/bin/env python3
# bulk_pgm2png.py  —  convert all PGM files in a folder to PNG
# usage: python bulk_pgm2png.py  <input_dir>  <output_dir>

import sys, os
from pathlib import Path
from PIL import Image

def main(indir: Path, outdir: Path):
    if not indir.is_dir():
        sys.exit(f"Input dir not found: {indir}")
    outdir.mkdir(parents=True, exist_ok=True)

    pgms = list(indir.glob("*.pgm"))
    if not pgms:
        print("No .pgm files found.")
        return

    for pgm in pgms:
        png_name = pgm.stem + ".png"
        out_path = outdir / png_name
        with Image.open(pgm) as im:
            im.convert("L").save(out_path, format="PNG")
        print(f"→ {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: bulk_pgm2png.py <input_dir> <output_dir>")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
