import argparse
import os
from pathlib import Path
import sys
import urllib.request

# NOTE: SAM2 release assets/URLs may change.
# Visit https://github.com/facebookresearch/sam2/releases for the latest.
CANDIDATES = {
    "sam2.1_hiera_large": [
        # Example placeholder URL; replace with official URL if changed:
        "https://dl.fbaipublicfiles.com/segment_anything_2/sam2.1_hiera_large.pt"
    ]
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=CANDIDATES.keys())
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    urls = CANDIDATES[args.model]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}.pt"

    for url in urls:
        try:
            print(f"Downloading {args.model} from {url}")
            urllib.request.urlretrieve(url, out_path)
            print(f"Saved to {out_path}")
            return
        except Exception as e:
            print(f"Failed from {url}: {e}", file=sys.stderr)

    print("All candidate URLs failed. Please download manually from the official SAM2 releases.", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()