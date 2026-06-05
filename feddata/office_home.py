"""
feddata/office_home.py - Office-Home loader.

FTL simulation:
  t-side (unlabeled) : target domain -> features[:split_dim]
  d-side (labeled)   : source domain -> features[split_dim:]
"""

import hashlib
import os
import numpy as np
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import Config

DOMAIN_ALIASES = {
    "art": "Art",
    "clipart": "Clipart",
    "product": "Product",
    "real_world": "Real World",
    "real world": "Real World",
}

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _root(data_dir: str) -> str:
    nested = os.path.join(data_dir, "OfficeHomeDataset_10072016")
    return nested if os.path.isdir(nested) else data_dir


def _domain_dir(data_dir: str, domain: str) -> str:
    dom = DOMAIN_ALIASES.get(str(domain).lower(), str(domain))
    path = os.path.join(_root(data_dir), dom)
    if not os.path.isdir(path):
        valid = ", ".join(sorted(DOMAIN_ALIASES))
        raise FileNotFoundError(
            f"Office-Home domain not found: {path}. Valid domains: {valid}"
        )
    return path


def _classes(data_dir: str):
    root = _root(data_dir)
    domains = [os.path.join(root, d) for d in sorted(set(DOMAIN_ALIASES.values()))]
    class_sets = []
    for dom in domains:
        if os.path.isdir(dom):
            class_sets.append({
                name for name in os.listdir(dom)
                if os.path.isdir(os.path.join(dom, name))
            })
    if not class_sets:
        raise FileNotFoundError(
            f"Office-Home data not found under {data_dir}. Expected "
            "OfficeHomeDataset_10072016/{Art,Clipart,Product,Real World}/..."
        )
    return sorted(set.intersection(*class_sets))


def _select_classes(data_dir: str, selected):
    available = _classes(data_dir)
    if not selected:
        return available

    lookup = {c.lower(): c for c in available}
    classes = []
    missing = []
    for name in selected:
        key = str(name).lower()
        if key in lookup:
            classes.append(lookup[key])
        else:
            missing.append(str(name))
    if missing:
        raise FileNotFoundError(
            "Office-Home selected classes not found: "
            f"{missing}. Available examples: {available[:12]}"
        )
    return classes


def _cache_suffix(classes):
    sig = hashlib.sha1("|".join(classes).encode("utf-8")).hexdigest()[:8]
    return f"{len(classes)}c_{sig}"


class _ImageFolder(Dataset):
    def __init__(self, domain_dir, classes, transform=_TRANSFORM):
        self.samples, self.transform = [], transform
        for label, cls in enumerate(classes):
            d = os.path.join(domain_dir, cls)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(d, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), label


def _extract(domain_dir, classes, device, cache) -> Tuple[np.ndarray, np.ndarray]:
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if "paths" in d and int(d.get("n_classes", -1)) == len(classes):
            return d["features"], d["labels"], d["paths"].astype(str)

    ds = _ImageFolder(domain_dir, classes)
    if len(ds) == 0:
        raise FileNotFoundError(f"No Office-Home images found in {domain_dir}")
    loader = DataLoader(ds, batch_size=64, num_workers=0, pin_memory=False)

    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone = backbone.to(device).eval()

    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=f"  {os.path.basename(domain_dir)}", ncols=70):
            feats.append(backbone(imgs.to(device)).cpu().numpy())
            labels.append(lbls.numpy())

    feats = np.concatenate(feats, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0).astype(np.int64)
    paths = np.array([p for p, _ in ds.samples], dtype=object)

    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, features=feats, labels=labels, paths=paths,
                 n_classes=np.array(len(classes), dtype=np.int64))
    return feats, labels, paths


class OfficeHomeDataModule:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.classes = _select_classes(
            cfg.data.data_dir,
            getattr(cfg.data, "office_home_classes", None),
        )
        cfg.data.n_classes = len(self.classes)
        cfg.prototype.n_clusters = len(self.classes)
        cache_dir = os.path.join(cfg.data.data_dir, ".cache")

        src_domain_dir = _domain_dir(cfg.data.data_dir, cfg.data.source_domain)
        tgt_domain_dir = _domain_dir(cfg.data.data_dir, cfg.data.target_domain)
        src_cache_name = str(cfg.data.source_domain).replace(" ", "_")
        tgt_cache_name = str(cfg.data.target_domain).replace(" ", "_")
        cache_suffix = _cache_suffix(self.classes)

        src_f, src_y, src_paths = _extract(
            src_domain_dir, self.classes, cfg.device,
            os.path.join(cache_dir, f"{src_cache_name}_resnet50_{cache_suffix}.npz"),
        )
        tgt_f, tgt_y, tgt_paths = _extract(
            tgt_domain_dir, self.classes, cfg.device,
            os.path.join(cache_dir, f"{tgt_cache_name}_resnet50_{cache_suffix}.npz"),
        )

        s = cfg.data.split_dim
        t_x = tgt_f[:, :s].astype(np.float32)
        d_x = src_f[:, s:].astype(np.float32)

        self.t_train_x, self.t_test_x, self.t_train_y, self.t_test_y, \
            self.t_train_paths, self.t_test_paths = \
            train_test_split(t_x, tgt_y, tgt_paths, test_size=cfg.data.test_ratio,
                             random_state=cfg.seed, stratify=tgt_y)
        self.d_train_x, self.d_test_x, self.d_train_y, self.d_test_y, \
            self.d_train_paths, self.d_test_paths = \
            train_test_split(d_x, src_y, src_paths, test_size=cfg.data.test_ratio,
                             random_state=cfg.seed, stratify=src_y)

    def summary(self) -> dict:
        return {
            "dataset": "Office-Home",
            "source (d-side)": self.cfg.data.source_domain,
            "target (t-side)": self.cfg.data.target_domain,
            "classes": len(self.classes),
            "class_names": self.classes,
            "t_train": len(self.t_train_x),
            "t_test": len(self.t_test_x),
            "d_train": len(self.d_train_x),
            "d_test": len(self.d_test_x),
            "t_feat_dim": self.t_train_x.shape[1],
            "d_feat_dim": self.d_train_x.shape[1],
        }
