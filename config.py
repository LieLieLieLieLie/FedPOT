"""
config.py — All hyperparameters and experimental settings for FedPOT.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class DataConfig:
    dataset: str = "office_home"          # "office_home" | "cwru"

    # Office-Home
    data_dir: str = "./data/office_home"
    source_domain: str = "product"
    target_domain: str = "real_world"
    n_classes: int = 10
    feature_dim: int = 2048
    split_dim: int = 1024
    office_home_classes: Optional[List[str]] = field(default_factory=lambda: [
        "Backpack", "Bike", "Calculator", "Keyboard", "Laptop",
        "Monitor", "Mouse", "Mug", "Printer", "Webcam",
    ])

    # CWRU
    cwru_data_dir: str = "./data/cwru"
    cwru_source_load: int = 0
    cwru_target_load: int = 2
    cwru_n_classes: int = 4
    cwru_segment_len: int = 1024
    cwru_overlap: float = 0.5
    cwru_feature_dim: int = 64

    test_ratio: float = 0.2
    val_ratio: float = 0.1


@dataclass
class PrivacyConfig:
    epsilon: float = 2.0
    delta: float = 1e-5
    max_norm: float = 1.0


@dataclass
class PrototypeConfig:
    n_clusters: int = 65
    kmeans_n_init: int = 20
    kmeans_max_iter: int = 500


@dataclass
class OTConfig:
    sinkhorn_reg: float = 0.05
    partial_mass: Optional[float] = None
    mass_search_grid: List[float] = field(
        default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    cost_metric: str = "cosine"
    n_sinkhorn_iter: int = 200
    # FIX: relational cost weight reduced 0.5→0.25 for better alignment
    # with DP-noisy prototypes; original 0.5 over-penalised structural mismatch.
    relational_cost_weight: float = 0.25
    # Ablation support: True = use single argmax d-prototype per cluster
    # (hard assignment) instead of soft OT-weighted mixture.
    hard_condition: bool = False
    # Full FedPOT condition uses a small sample-level nearest-prototype branch
    # blended into the OT soft condition. This stabilizes cases where partial OT
    # is informative but individual samples deviate from their cluster center.
    nn_condition_weight: float = 0.35
    # Estimate the transport plan / cluster-class bijection on clean prototypes,
    # then apply DP-noisy prototypes only when constructing transmitted
    # generation conditions. This avoids using privacy noise to decide semantics.
    clean_alignment: bool = True
    # Ablation support: True = use uniform average of all d-prototypes
    # (no OT alignment at all) as conditioning signal.
    uniform_condition: bool = False
    # Ablation support: disable the OT-derived cluster->class bijection as well
    # as the transport condition. Used by "w/o Partial OT".
    disable_ot_mapping: bool = False
    # ProtoFTL-style conditioning: use t-prototype similarity (reliable,
    # same-domain) to weight d-prototypes reordered by the OT bijection.
    # Enabled for Office-Home (disjoint CNN feature halves) where cross-domain OT
    # conditions are less reliable than t-side prototype similarities.
    # hard_condition=True then uses hard argmax instead of softmax.
    use_proto_condition: bool = False
    proto_cond_temp: float = 5.0    # softmax temperature (same as ProtoFTL)


@dataclass
class CVAEConfig:
    latent_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    # Increased 150→200 for more stable convergence after encoder fix
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    beta: float = 1.0
    ot_lambda: float = 0.1
    ot_reg: float = 0.1
    # Blend target-view reconstruction with the transported d-side condition.
    # This trains the generator as a complementary-view imputer instead of a
    # plain autoencoder that merely copies the observed t-view.
    condition_target_weight: float = 0.35
    grad_clip: float = 5.0
    lr_scheduler_step: int = 60
    lr_scheduler_gamma: float = 0.5
    # KL warm-up: linearly ramp beta from 0 to `beta` over first kl_warmup_epochs.
    # Prevents posterior collapse in the early phase of training.
    kl_warmup_epochs: int = 20
    # Post-generation smoothing: blend CVAE output with OT condition.
    #   gen_smooth = (1 - alpha) * gen_cvae + alpha * ot_cond
    # 0.0 = pure CVAE (default, no smoothing).
    # Higher values bias toward the smooth OT-prototype mixture, which improves
    # probability calibration (AUC) and F1 in high-noise regimes.
    gen_smooth_alpha: float = 0.0


@dataclass
class FilterConfig:
    recon_uncertainty_pct: float = 25.0
    sem_uncertainty_pct: float = 25.0
    min_keep_ratio: float = 0.3
    rejected_weight: float = 0.85


@dataclass
class DownstreamConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    # Increased 100→200 for more stable convergence with noisy pseudo labels
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    dropout: float = 0.3
    lr_scheduler_step: int = 60
    lr_scheduler_gamma: float = 0.5
    # Label smoothing eps: smooths hard pseudo labels to reduce over-fitting.
    # Critical when pseudo labels carry ~20-40% noise from cluster assignment.
    label_smoothing: float = 0.1
    # Late fusion between the augmented FedPOT classifier and the t-only
    # classifier. This preserves stable target-view evidence while letting the
    # generated complementary view improve the final decision.
    fusion_alpha: float = 0.25
    auto_fusion_alpha: bool = True
    fusion_alpha_min: float = 0.25
    fusion_alpha_grid: List[float] = field(
        default_factory=lambda: [0.25, 0.35, 0.5, 0.65, 0.8, 1.0]
    )
    # CWRU-only late fusion between FedPOT fused logits and prototype-alignment
    # logits. Keep the generated FedPOT branch dominant; alignment is a helper.
    cwru_align_alpha: float = 0.25
    office_align_alpha: float = 0.20
    use_align_fusion: bool = True
    auto_align_alpha: bool = True
    use_combo_fusion: bool = False
    balance_prior_strength: float = 0.0
    balance_prior_temperature: float = 1.0
    use_auc_align_head: bool = False
    auc_align_bonus: float = 0.0
    align_alpha_grid: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.25, 0.35, 0.5]
    )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    prototype: PrototypeConfig = field(default_factory=PrototypeConfig)
    ot: OTConfig = field(default_factory=OTConfig)
    cvae: CVAEConfig = field(default_factory=CVAEConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    downstream: DownstreamConfig = field(default_factory=DownstreamConfig)

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"
    exp_name: str = "fedpot_default"
    verbose: bool = True
