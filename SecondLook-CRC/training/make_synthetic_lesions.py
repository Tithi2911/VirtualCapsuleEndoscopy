"""Write a labelled synthetic lesion image set (benign / precancerous / cancerous).

The images are procedural 2D drawings: pale, smooth hyperplastic-like polyps;
redder, lobulated adenoma-like polyps with a tubular surface pattern; and large,
ragged, dark red masses with a fibrin-covered ulcer. They are for testing the
optical-diagnosis pipeline end to end and, at most, for synthetic pre-training
before fine-tuning on real histology-labelled data. A model trained only on
them has learnt these drawings, not pathology, and must never be used on
patients. The realistic synthetic source is VR-Caps (this repository) and
MADSyncro renders with labelled lesion materials; see docs/DATASETS.md.

Each synthetic patient has one lesion photographed --views-per-lesion times
(different framing, distance, rotation, lighting, focus and noise), so the
patient-grouped split in train_classifier.py has something to do. A fraction
of lesions (--atypical-fraction) is drawn part-way towards the neighbouring
category but keeps its true label, as real lesions overlap in appearance;
without that, the classes are perfectly separable and calibration and
abstention cannot be tested.

Output (in --out), in the format training/train_classifier.py reads:

    images/SYN00001_v1.png ...   lesion photographs
    masks/SYN00001_v1.png ...    lesion masks (white = lesion)
    labels.csv                   image,label,patient_id,mask,appearance (typical / atypical)
    dataset.json                 what was generated

Example:

    python training/make_synthetic_lesions.py --out data/synth_lesions --n-per-class 400
    python training/train_classifier.py --data data/synth_lesions --out models/cadx-synth \\
        --backbone tiny --size 96 --epochs 10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from secondlook.synthetic import generate_lesion_dataset


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output directory (created if missing)")
    ap.add_argument("--n-per-class", type=int, default=60, help="synthetic patients (lesions) per category")
    ap.add_argument("--views-per-lesion", type=int, default=3, help="images of each lesion")
    ap.add_argument("--size", type=int, default=256, help="image width and height in pixels")
    ap.add_argument("--atypical-fraction", type=float, default=0.15,
                    help="share of lesions drawn part-way towards a neighbouring category (0-1)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.n_per_class < 1 or args.views_per_lesion < 1:
        ap.error("--n-per-class and --views-per-lesion must be at least 1")
    if args.size < 64:
        ap.error("--size must be at least 64")
    if not 0 <= args.atypical_fraction <= 1:
        ap.error("--atypical-fraction must be between 0 and 1")

    started = time.perf_counter()
    summary = generate_lesion_dataset(args.out, args.n_per_class, args.views_per_lesion,
                                      (args.size, args.size), args.seed, atypical_fraction=args.atypical_fraction)
    print(f"Wrote {summary['images']} images of {summary['patients']} synthetic lesions to {args.out} "
          f"in {time.perf_counter() - started:.1f} s")
    for category, counts in summary["per_class"].items():
        print(f"  {category:<13} {counts['patients']:>5} lesions ({counts['atypical_patients']} atypical) "
              f"{counts['images']:>6} images")
    print(f"Labels: {summary['labels_csv']}")
    print("Synthetic drawings for software testing and pre-training only; not clinical data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
