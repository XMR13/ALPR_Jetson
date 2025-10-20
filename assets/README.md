Assets directory

Purpose
- Holds local, non-redistributable resources used by tools, especially fonts for synthetic OCR dataset generation.

Usage
- Place one or more TTF/OTF fonts under `assets/fonts/` that resemble Indonesian license plate fonts (sans/condensed works well).
- Recommended free options you can copy from your system:
  - Linux: `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`, `DejaVuSansCondensed.ttf`, or Liberation/Roboto/Noto Sans variants.
  - Windows: `C:\\Windows\\Fonts\\arial.ttf`, `arialbd.ttf`, `arialn.ttf` (condensed/narrow), or install free fonts locally.
  - macOS: `/System/Library/Fonts/Supplemental/Arial Narrow.ttf`, `Helvetica.ttc`, or install Roboto/Noto.

Generating synthetic plates
- Example:
  `python tools/synth_plates.py --outdir data/ocr/synth --count 5000 --fonts-dir assets/fonts --width 160 --height 32 --seed 123`

Outputs
- `data/ocr/synth/crops/*.jpg`
- `data/ocr/synth/labels_train.csv`, `labels_val.csv`
- `data/ocr/synth/charset.txt`

Notes
- This repo does not ship font files for licensing reasons.
- Ensure `opencv-python` and `Pillow` are available in your environment when running the generator on a dev machine.

