"""
prepare_data.py — 数据下载验证 + 特征预提取（运行一次即可）

  python prepare_data.py                    # 验证两个数据集
  python prepare_data.py --dataset cwru     # 只下载 CWRU
  python prepare_data.py --dataset office_caltech  # 只验证 Office-Caltech10
"""

import argparse
import os
import sys
import urllib.request
import torch

OC_DOMAINS = ["amazon", "caltech", "dslr", "webcam"]
OC_CLASSES = [
    "back_pack", "bike", "calculator", "headphones", "keyboard",
    "laptop_computer", "monitor", "mouse", "mug", "projector",
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


# ── Office-Caltech10 ──────────────────────────────────────────────────────────

def check_office_caltech(data_dir):
    ok = True
    for dom in OC_DOMAINS:
        for cls in OC_CLASSES:
            d = os.path.join(data_dir, dom, cls)
            if not os.path.isdir(d):
                print(f"  [MISSING] {d}")
                ok = False
    return ok


def prepare_office_caltech(data_dir, device):
    print("\n── Office-Caltech10 ─────────────────────────────────────")
    print(f"   data_dir : {data_dir}")

    if not check_office_caltech(data_dir):
        print("""
  数据集结构不完整，请手动下载：

  方式 A — Kaggle CLI:
    pip install kaggle
    kaggle datasets download -d tarunbisht11/office-caltech-10-dataset
    unzip office-caltech-10-dataset.zip -d ./data/

  方式 B — 手动下载:
    访问: https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md
    找 "Office+Caltech" 下载（约500MB），解压到:
      ./data/office_caltech_10/
        ├── amazon/  ├── caltech/  ├── dslr/  └── webcam/
""")
        sys.exit(1)

    print("   结构验证 OK ✓")
    print("   正在预提取 ResNet50 特征（有GPU约2分钟，CPU约15分钟）...")

    sys.path.insert(0, os.path.dirname(__file__))
    from config import Config, DataConfig
    from feddata.office_caltech import OfficeCaltechDataModule

    for src in OC_DOMAINS:
        tgt = [d for d in OC_DOMAINS if d != src][0]
        cfg = Config(data=DataConfig(
            dataset="office_caltech", data_dir=data_dir,
            source_domain=src, target_domain=tgt,
        ), device=device)
        print(f"   提取: {src} ...")
        OfficeCaltechDataModule(cfg)

    print("   Office-Caltech10 准备完成 ✓")


# ── CWRU ─────────────────────────────────────────────────────────────────────

def prepare_cwru(data_dir):
    print("\n── CWRU Bearing Fault Dataset ───────────────────────────")
    print(f"   data_dir : {data_dir}")
    os.makedirs(data_dir, exist_ok=True)

    failed = []
    for stem, url in CWRU_URLS.items():
        dest = os.path.join(data_dir, f"{stem}.mat")
        if os.path.exists(dest):
            print(f"   {stem}.mat 已存在，跳过")
            continue
        print(f"   下载 {stem}.mat ...", end=" ", flush=True)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r, \
                 open(dest, "wb") as f:
                f.write(r.read())
            print("✓")
        except Exception as e:
            print(f"失败 ({e})")
            failed.append(stem)

    if failed:
        print(f"\n  下载失败: {failed}")
        print("  请手动下载: https://engineering.case.edu/bearingdatacenter/download-data-file")
        print(f"  将 .mat 文件放到: {data_dir}")
    else:
        found = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
        print(f"   CWRU 准备完成 ✓  共 {len(found)} 个 .mat 文件")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  default="all",
                   choices=["office_caltech", "cwru", "all"])
    p.add_argument("--oc_dir",   default="./data/office_caltech_10")
    p.add_argument("--cwru_dir", default="./data/cwru")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("=" * 55)
    print("  FedPOT — 数据准备")
    print("=" * 55)
    print(f"  device : {args.device}")

    if args.dataset in ("office_caltech", "all"):
        prepare_office_caltech(args.oc_dir, args.device)

    if args.dataset in ("cwru", "all"):
        prepare_cwru(args.cwru_dir)

    print("\n  全部完成！现在可以运行:")
    print("    python main.py                    # 两个数据集，训练+测试")
    print("    python main.py --mode train       # 只训练")
    print("    python main.py --mode test        # 只测试")
    print("    python main.py --sweep            # 全域名 sweep")
    print("    python main.py --ablation         # 消融实验")


if __name__ == "__main__":
    main()
