"""
feddata/office_caltech.py — Office-Caltech10 loader.

FTL simulation:
  t-side (unlabeled) : target domain → features[:split_dim]
  d-side (labeled)   : source domain → features[split_dim:]
"""

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

CLASSES = [
    "back_pack", "bike", "calculator", "headphones", "keyboard",
    "laptop_computer", "monitor", "mouse", "mug", "projector",
]

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


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


def _extract(domain_dir, device, cache) -> Tuple[np.ndarray, np.ndarray]:
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if "paths" in d:
            return d["features"], d["labels"], d["paths"].astype(str)

    ds     = _ImageFolder(domain_dir, CLASSES)
    loader = DataLoader(ds, batch_size=64, num_workers=0, pin_memory=False)

    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone = backbone.to(device).eval()

    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=f"  {os.path.basename(domain_dir)}", ncols=70):
            feats.append(backbone(imgs.to(device)).cpu().numpy())
            labels.append(lbls.numpy())

    feats  = np.concatenate(feats,  axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0).astype(np.int64)
    paths = np.array([p for p, _ in ds.samples], dtype=object)

    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, features=feats, labels=labels, paths=paths)
    return feats, labels, paths


class OfficeCaltechDataModule:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cache_dir = os.path.join(cfg.data.data_dir, ".cache")

        src_f, src_y, src_paths = _extract(
            os.path.join(cfg.data.data_dir, cfg.data.source_domain),
            cfg.device,
            os.path.join(cache_dir, f"{cfg.data.source_domain}_paths_v2.npz"),
        )
        tgt_f, tgt_y, tgt_paths = _extract(
            os.path.join(cfg.data.data_dir, cfg.data.target_domain),
            cfg.device,
            os.path.join(cache_dir, f"{cfg.data.target_domain}_paths_v2.npz"),
        )

        s   = cfg.data.split_dim
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
            "dataset":         "Office-Caltech10",
            "source (d-side)": self.cfg.data.source_domain,
            "target (t-side)": self.cfg.data.target_domain,
            "t_train":         len(self.t_train_x),
            "t_test":          len(self.t_test_x),
            "d_train":         len(self.d_train_x),
            "d_test":          len(self.d_test_x),
            "t_feat_dim":      self.t_train_x.shape[1],
            "d_feat_dim":      self.d_train_x.shape[1],
        }
