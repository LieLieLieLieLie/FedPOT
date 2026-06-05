"""
Dataset preparation helper for FedPOT.

Examples:
  python prepare_data.py
  python prepare_data.py --dataset office_home
  python prepare_data.py --dataset cwru
"""

import argparse
import os
import sys
import urllib.request
import zipfile

import torch

OFFICE_HOME_URL = (
    "https://huggingface.co/huangyuyang11/officehome/resolve/main/"
    "OfficeHomeDataset_10072016.zip"
)
OFFICE_HOME_DOMAINS = ["Art", "Clipart", "Product", "Real World"]
OFFICE_HOME_CLASSES_10 = [
    "Backpack", "Bike", "Calculator", "Keyboard", "Laptop",
    "Monitor", "Mouse", "Mug", "Printer", "Webcam",
]

CWRU_URLS = {
    "97":  "https://engineering.case.edu/sites/default/files/97.mat",
    "98":  "https://engineering.case.edu/sites/default/files/98.mat",
    "99":  "https://engineering.case.edu/sites/default/files/99.mat",
    "100": "https://engineering.case.edu/sites/default/files/100.mat",
    "105": "https://engineering.case.edu/sites/default/files/105.mat",
    "106": "https://engineering.case.edu/sites/default/files/106.mat",
    "107": "https://engineering.case.edu/sites/default/files/107.mat",
    "108": "https://engineering.case.edu/sites/default/files/108.mat",
    "118": "https://engineering.case.edu/sites/default/files/118.mat",
    "119": "https://engineering.case.edu/sites/default/files/119.mat",
    "120": "https://engineering.case.edu/sites/default/files/120.mat",
    "121": "https://engineering.case.edu/sites/default/files/121.mat",
    "130": "https://engineering.case.edu/sites/default/files/130.mat",
    "131": "https://engineering.case.edu/sites/default/files/131.mat",
    "132": "https://engineering.case.edu/sites/default/files/132.mat",
    "133": "https://engineering.case.edu/sites/default/files/133.mat",
}


def _office_home_root(data_dir):
    nested = os.path.join(data_dir, "OfficeHomeDataset_10072016")
    return nested if os.path.isdir(nested) else data_dir


def check_office_home(data_dir):
    root = _office_home_root(data_dir)
    return all(os.path.isdir(os.path.join(root, d)) for d in OFFICE_HOME_DOMAINS)


def _download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def prepare_office_home(data_dir, device):
    print("\n-- Office-Home ------------------------------------------------")
    print(f"   data_dir : {data_dir}")
    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(data_dir, "OfficeHomeDataset_10072016.zip")
    if not check_office_home(data_dir):
        if not os.path.exists(zip_path):
            print("   downloading Office-Home zip ...")
            _download(OFFICE_HOME_URL, zip_path)
        print("   extracting Office-Home zip ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)

    if not check_office_home(data_dir):
        print("Office-Home layout is still incomplete.")
        print("Expected: data/office_home/OfficeHomeDataset_10072016/{Art,Clipart,Product,Real World}/")
        sys.exit(1)

    print("   layout OK")
    print("   pre-extracting ResNet50 features for Product and Real World (10 classes) ...")

    sys.path.insert(0, os.path.dirname(__file__))
    from config import Config, DataConfig
    from feddata.office_home import OfficeHomeDataModule

    cfg = Config(data=DataConfig(
        dataset="office_home",
        data_dir=data_dir,
        source_domain="product",
        target_domain="real_world",
        n_classes=len(OFFICE_HOME_CLASSES_10),
        office_home_classes=OFFICE_HOME_CLASSES_10,
    ), device=device)
    OfficeHomeDataModule(cfg)
    print("   Office-Home ready")


def prepare_cwru(data_dir):
    print("\n-- CWRU Bearing Fault Dataset --------------------------------")
    print(f"   data_dir : {data_dir}")
    os.makedirs(data_dir, exist_ok=True)

    failed = []
    for stem, url in CWRU_URLS.items():
        dest = os.path.join(data_dir, f"{stem}.mat")
        if os.path.exists(dest):
            print(f"   {stem}.mat exists, skip")
            continue
        print(f"   downloading {stem}.mat ...", end=" ", flush=True)
        try:
            _download(url, dest)
            print("OK")
        except Exception as e:
            print(f"FAILED ({e})")
            failed.append(stem)

    if failed:
        print(f"\n  failed files: {failed}")
        print("  manually download from: https://engineering.case.edu/bearingdatacenter/download-data-file")
    else:
        found = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
        print(f"   CWRU ready: {len(found)} .mat files")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all", choices=["office_home", "cwru", "all"])
    p.add_argument("--office_home_dir", default="./data/office_home")
    p.add_argument("--cwru_dir", default="./data/cwru")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.dataset in ("office_home", "all"):
        prepare_office_home(args.office_home_dir, args.device)
    if args.dataset in ("cwru", "all"):
        prepare_cwru(args.cwru_dir)

    print("\nDone. Example:")
    print("  python main.py --dataset office_home --baselines")
    print("  python main.py --dataset office_home --ablation")


if __name__ == "__main__":
    main()
