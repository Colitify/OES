"""Download LIBS Benchmark dataset from Figshare."""
import os, requests, sys
from pathlib import Path

URLS = {
    "train.h5": "https://ndownloader.figshare.com/files/20065616",
    "test.h5":  "https://ndownloader.figshare.com/files/20065574",
    "test_labels.csv": "https://ndownloader.figshare.com/files/20065628",
}

def download(name, url, dest_dir):
    dest = Path(dest_dir) / name
    if dest.exists():
        print(f"{name} already exists ({dest.stat().st_size/1e9:.2f} GB), skipping.")
        return
    tmp = dest.with_suffix(".tmp")
    headers = {}
    if tmp.exists():
        headers["Range"] = f"bytes={tmp.stat().st_size}-"
        mode = "ab"
        print(f"Resuming {name} from {tmp.stat().st_size/1e6:.1f} MB")
    else:
        mode = "wb"
        print(f"Downloading {name}...")
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, mode) as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {pct:.1f}%  {done/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
    tmp.rename(dest)
    print(f"\nDone: {name}")

if __name__ == "__main__":
    dest = Path(__file__).parent
    files = sys.argv[1:] if len(sys.argv) > 1 else list(URLS.keys())
    for f in files:
        download(f, URLS[f], dest)
