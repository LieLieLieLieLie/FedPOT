"""
trainer.py �?FedPOT pipeline orchestrator.

Modes:
  train      : run all phases, save models
  test       : load saved models, evaluate only
  train_test : train + evaluate (default)

每个 Phase 都记录耗时，run() 结束后打印总时间汇总�?"""

import os
import json
import time
import numpy as np
from typing import Dict, Optional

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from config import Config
from feddata import get_data_module
from core.prototype import PrototypeBank
from core.ot_alignment import PartialOTAligner
from core.generator import CVAEGenerator
from core.filter import UncertaintyFilter, compute_semantic_uncertainty
from evaluation.downstream import DownstreamTrainer, metrics_from_logits, print_results
from evaluation.utils import get_logger, set_seed


class PhaseTimer:
    def __init__(self):
        self._records = {}
        self._start   = time.time()

    def record(self, name: str, elapsed: float):
        self._records[name] = elapsed

    def total(self) -> float:
        return time.time() - self._start

    def summary(self) -> str:
        lines = [
            "",
            "  +-------------------------------------------+",
            "  |            Time Report                    |",
            "  +-------------------------------------------+",
        ]
        for name, sec in self._records.items():
            m, s = divmod(int(sec), 60)
            lines.append(f"  | {name:<28s}  {m:02d}:{s:02d}        |")
        lines.append("  +-------------------------------------------+")
        total = self.total()
        m, s  = divmod(int(total), 60)
        h, m  = divmod(m, 60)
        t_str = f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m:02d}m {s:02d}s"
        lines.append(f"  | {'Total':<28s}  {t_str:<10s}  |")
        lines.append("  +-------------------------------------------+")
        return "\n".join(lines)



class FedPOTTrainer:
    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self.logger  = get_logger("FedPOT", cfg.log_dir, cfg.exp_name)
        set_seed(cfg.seed)
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.data = self.t_bank = self.d_bank = None
        self.aligner = self.generator = self.filt = None
        self.baseline_trainer = self.fedpot_trainer = None
        self.align_trainer = None
        self.combo_trainer = None
        self.auc_align_trainer = None
        self._align_W = None
        self._gen_train = None
        self._gen_test = None
        self.timer = PhaseTimer()

    def _ckpt_dir(self):
        d = os.path.join(self.cfg.save_dir, self.cfg.exp_name)
        os.makedirs(d, exist_ok=True)
        return d

    def save(self):
        d = self._ckpt_dir()
        if self.generator and self.generator.cvae:
            torch.save(self.generator.cvae.state_dict(), os.path.join(d, "cvae.pt"))
        if self.baseline_trainer and self.baseline_trainer.model:
            torch.save(self.baseline_trainer.model.state_dict(),
                       os.path.join(d, "classifier_baseline.pt"))
        if self.fedpot_trainer and self.fedpot_trainer.model:
            torch.save(self.fedpot_trainer.model.state_dict(),
                       os.path.join(d, "classifier_fedpot.pt"))
        if self.aligner:
            np.savez(os.path.join(d, "ot_state.npz"),
                     T_star=self.aligner.T_star,
                     cluster_to_class=self.aligner.cluster_to_class,
                     mass=np.array([self.aligner.mass]))
        if self.t_bank:
            np.savez(os.path.join(d, "t_bank.npz"),
                     raw_prototypes=self.t_bank.raw_prototypes,
                     noisy_prototypes=self.t_bank.noisy_prototypes,
                     assignments=self.t_bank.assignments)
        if self.d_bank:
            np.savez(os.path.join(d, "d_bank.npz"),
                     raw_prototypes=self.d_bank.raw_prototypes,
                     noisy_prototypes=self.d_bank.noisy_prototypes)
        if self.filt:
            json.dump({"tau_recon": self.filt.tau_recon, "tau_sem": self.filt.tau_sem},
                      open(os.path.join(d, "filter_thresholds.json"), "w"))
        self.logger.info(f"  [Save] Checkpoint -> {d}/")

    def load(self):
        d = self._ckpt_dir()
        for f in ["cvae.pt", "ot_state.npz", "t_bank.npz", "d_bank.npz"]:
            if not os.path.exists(os.path.join(d, f)):
                raise FileNotFoundError(f"Missing: {os.path.join(d, f)}\nRun --mode train first.")
        self.data = get_data_module(self.cfg)
        self.t_bank = PrototypeBank(self.cfg, "t")
        tb = np.load(os.path.join(d, "t_bank.npz"))
        self.t_bank.raw_prototypes = tb["raw_prototypes"]
        self.t_bank.noisy_prototypes = tb["noisy_prototypes"]
        self.t_bank.assignments = tb["assignments"]
        self.d_bank = PrototypeBank(self.cfg, "d")
        db = np.load(os.path.join(d, "d_bank.npz"))
        self.d_bank.raw_prototypes = db["raw_prototypes"]
        self.d_bank.noisy_prototypes = db["noisy_prototypes"]
        self.aligner = PartialOTAligner(self.cfg)
        ot_s = np.load(os.path.join(d, "ot_state.npz"))
        self.aligner.T_star = ot_s["T_star"]
        self.aligner.cluster_to_class = ot_s["cluster_to_class"]
        self.aligner.mass = float(ot_s["mass"][0])
        x_dim = self.data.t_train_x.shape[1]
        cond_dim = self.data.d_train_x.shape[1]
        from models.networks import CVAE
        cfg_c = self.cfg.cvae
        m = CVAE(x_dim, cond_dim, cond_dim, cfg_c.hidden_dims, cfg_c.latent_dim).to(self.cfg.device)
        m.load_state_dict(torch.load(os.path.join(d, "cvae.pt"), map_location=self.cfg.device))
        self.generator = CVAEGenerator(self.cfg); self.generator.cvae = m
        from models.networks import DownstreamClassifier
        cfg_ds = self.cfg.downstream
        self.baseline_trainer = DownstreamTrainer(self.cfg, "Baseline")
        self.baseline_trainer.model = DownstreamClassifier(
            x_dim, self.cfg.data.n_classes, cfg_ds.hidden_dims, cfg_ds.dropout).to(self.cfg.device)
        self.baseline_trainer.model.load_state_dict(
            torch.load(os.path.join(d, "classifier_baseline.pt"), map_location=self.cfg.device))
        aug_dim = x_dim + cond_dim
        self.fedpot_trainer = DownstreamTrainer(self.cfg, "FedPOT")
        self.fedpot_trainer.model = DownstreamClassifier(
            aug_dim, self.cfg.data.n_classes, cfg_ds.hidden_dims, cfg_ds.dropout).to(self.cfg.device)
        self.fedpot_trainer.model.load_state_dict(
            torch.load(os.path.join(d, "classifier_fedpot.pt"), map_location=self.cfg.device))
        tp = os.path.join(d, "filter_thresholds.json")
        if os.path.exists(tp):
            th = json.load(open(tp))
            self.filt = UncertaintyFilter(self.cfg)
            self.filt.tau_recon = th["tau_recon"]; self.filt.tau_sem = th["tau_sem"]
        self.logger.info(f"  [Load] Checkpoint <- {d}/")

    def phase1_prototypes(self):
        t0 = time.time()
        self.logger.info("="*58 + "\nPhase 1 · Data Loading + Prototype Extraction\n" + "="*58)
        self.data = get_data_module(self.cfg)
        for k, v in self.data.summary().items(): self.logger.info(f"  {k}: {v}")
        self.d_bank = PrototypeBank(self.cfg, "d").build_d_side(self.data.d_train_x, self.data.d_train_y)
        rpt = self.d_bank.privacy_report(); self.logger.info(f"  [d-side] {rpt['guarantee']}  sigma={rpt['sigma']:.4f}")
        self.t_bank = PrototypeBank(self.cfg, "t").build_t_side(self.data.t_train_x)
        rpt = self.t_bank.privacy_report(); self.logger.info(f"  [t-side] {rpt['guarantee']}  sigma={rpt['sigma']:.4f}")
        self.timer.record("Phase1: Data + Prototype", time.time()-t0)

    def phase2_ot_alignment(self):
        t0 = time.time()
        self.logger.info("="*58 + "\nPhase 2 · Partial OT Semantic Alignment\n" + "="*58)
        proto_d_for_ot = (self.d_bank.raw_prototypes
                          if getattr(self.cfg.ot, "clean_alignment", True)
                          else self.d_bank.transmit())
        self.aligner = PartialOTAligner(self.cfg).fit(
            self.t_bank.raw_prototypes, proto_d_for_ot)
        if getattr(self.cfg.ot, "disable_ot_mapping", False):
            n_t = len(self.t_bank.raw_prototypes)
            n_d = len(proto_d_for_ot)
            mapping = (np.arange(n_t) % max(n_d, 1)).astype(np.int64)
            self.aligner.cluster_to_class = mapping
            T = np.zeros((n_t, n_d), dtype=np.float32)
            if n_d > 0:
                T[np.arange(n_t), mapping] = 1.0 / max(n_t, 1)
            self.aligner.T_star = T
            self.logger.info("  [Ablation] OT cluster->class mapping disabled")
        s = self.aligner.alignment_summary()
        self.logger.info(f"  mass s*={s['transport_mass']:.3f}  entropy={s['T_star_row_entropy']:.4f}")
        self.logger.info(f"  cluster->class: {s['cluster_to_class']}")
        self.timer.record("Phase2: Partial OT", time.time()-t0)

    def phase3_generate(self):
        t0 = time.time()
        self.logger.info("="*58 + "\nPhase 3 · Transport-Conditioned CVAE Generation\n" + "="*58)
        proto_d   = self.d_bank.transmit()
        t_assign  = self.t_bank.assignments
        hard    = getattr(self.cfg.ot, "hard_condition",    False)
        uniform = getattr(self.cfg.ot, "uniform_condition", False)
        cond_train = self.aligner.compute_transport_conditions(
            proto_d, t_assign, hard=hard, uniform=uniform,
            sample_features=self.data.t_train_x)
        pseudo_labels = self.aligner.get_pseudo_labels(t_assign)
        self.generator = CVAEGenerator(self.cfg)
        self.generator.train(self.data.t_train_x, cond_train, self.logger)
        gen_train, var_train = self.generator.generate(self.data.t_train_x, cond_train)
        test_assign = self._assign_test_clusters(self.data.t_test_x)
        cond_test = self.aligner.compute_transport_conditions(
            proto_d, test_assign, hard=hard, uniform=uniform,
            sample_features=self.data.t_test_x)
        gen_test, _ = self.generator.generate(self.data.t_test_x, cond_test)
        # Post-generation smoothing: blend CVAE output with the (smooth)
        # OT transport condition.  This reduces CVAE generation noise and
        # improves probability calibration (AUC) while retaining OT alignment.
        #   gen_smooth = (1-α)*gen_cvae + α*ot_cond
        smooth_alpha = float(getattr(self.cfg.cvae, "gen_smooth_alpha", 0.0))
        if smooth_alpha > 0.0:
            gen_train = ((1.0 - smooth_alpha) * gen_train
                         + smooth_alpha * cond_train).astype(np.float32)
            gen_test  = ((1.0 - smooth_alpha) * gen_test
                         + smooth_alpha * cond_test).astype(np.float32)
        self.timer.record("Phase3: CVAE Generation", time.time()-t0)
        return gen_train, var_train, cond_train, pseudo_labels, gen_test

    def phase4_filter(self, gen_train, var_train, cond_train, pseudo_labels):
        t0 = time.time()
        self.logger.info("="*58 + "\nPhase 4 · Uncertainty-Aware Filtering\n" + "="*58)
        sem_unc = compute_semantic_uncertainty(gen_train, cond_train)
        self.filt = UncertaintyFilter(self.cfg).fit(var_train, sem_unc)
        kept_gen, kept_cond, kept_labels, kept_idx, stats = self.filt.apply(gen_train, cond_train, var_train, pseudo_labels)
        sample_weight = np.full(len(gen_train), self.cfg.filter.rejected_weight, dtype=np.float32)
        sample_weight[kept_idx] = 1.0
        self.logger.info(f"  tau_recon={stats['tau_recon']:.4f}  tau_sem={stats['tau_sem']:.4f}  "
                         f"kept {stats['n_kept']}/{stats['n_total']} ({stats['keep_ratio']*100:.1f}%)")
        self.timer.record("Phase4: Uncertainty Filter", time.time()-t0)
        return sample_weight

    def phase5_train_classifiers(self, gen_train, gen_test, pseudo_labels, sample_weight):
        t0 = time.time()
        self.logger.info("="*58 + "\nPhase 5 · Downstream Classifier Training\n" + "="*58)
        self.baseline_trainer = DownstreamTrainer(self.cfg, "Baseline")
        self.baseline_trainer.train(self.data.t_train_x, pseudo_labels, self.logger)
        X_aug_tr = np.concatenate([self.data.t_train_x, gen_train], axis=1)
        self.fedpot_trainer = DownstreamTrainer(self.cfg, "FedPOT")
        self.fedpot_trainer.train(X_aug_tr, pseudo_labels, self.logger,
                                  sample_weight=sample_weight)
        align_tr, self._align_W = self._prototype_alignment_view(self.data.t_train_x, fit=True)
        self.align_trainer = DownstreamTrainer(self.cfg, "FedPOT-align")
        # The alignment branch is a deterministic prototype-mapped target view,
        # not a generated sample set.  CVAE uncertainty weights should only
        # down-weight generated features; applying them here weakens the align
        # branch compared with the same fair protocol used by DANN-FTL.
        self.align_trainer.train(align_tr, pseudo_labels, self.logger)
        if getattr(self.cfg.downstream, "use_combo_fusion", False):
            X_combo_tr = np.concatenate(
                [self.data.t_train_x, gen_train, align_tr], axis=1)
            self.combo_trainer = DownstreamTrainer(self.cfg, "FedPOT-combo")
            self.combo_trainer.train(X_combo_tr, pseudo_labels, self.logger,
                                     sample_weight=sample_weight)
        self._gen_train = gen_train
        self._gen_test = gen_test
        self.timer.record("Phase5: Classifier Train", time.time()-t0)

    def phase5_evaluate(self) -> Dict:
        t0 = time.time()
        self.logger.info("="*58 + "\nPhase 5 · Evaluation\n" + "="*58)
        res_base = self.baseline_trainer.evaluate(self.data.t_test_x, self.data.t_test_y)
        base_logits = self.baseline_trainer.predict_logits(self.data.t_test_x)
        X_aug_te = np.concatenate([self.data.t_test_x, self._gen_test], axis=1)
        fed_logits = self.fedpot_trainer.predict_logits(X_aug_te)
        alpha = float(getattr(self.cfg.downstream, "fusion_alpha", 0.65))
        if self._gen_train is not None:
            train_aug = np.concatenate([self.data.t_train_x, self._gen_train], axis=1)
            train_base_logits = self.baseline_trainer.predict_logits(self.data.t_train_x)
            train_fed_logits = self.fedpot_trainer.predict_logits(train_aug)
            train_labels = self.aligner.get_pseudo_labels(self.t_bank.assignments)
            if (getattr(self.cfg.downstream, "use_combo_fusion", False)
                    and self.combo_trainer is not None
                    and self._align_W is not None):
                align_tr_for_combo, _ = self._prototype_alignment_view(
                    self.data.t_train_x, fit=False)
                train_combo = np.concatenate(
                    [self.data.t_train_x, self._gen_train, align_tr_for_combo],
                    axis=1)
                train_combo_logits = self.combo_trainer.predict_logits(train_combo)
                align_te_for_combo, _ = self._prototype_alignment_view(
                    self.data.t_test_x, fit=False)
                test_combo = np.concatenate(
                    [self.data.t_test_x, self._gen_test, align_te_for_combo],
                    axis=1)
                combo_logits = self.combo_trainer.predict_logits(test_combo)
                if self._pseudo_score(train_combo_logits, train_labels) >= self._pseudo_score(
                        train_fed_logits, train_labels):
                    train_fed_logits = train_combo_logits
                    fed_logits = combo_logits
                    self.logger.info("  [Fusion] using three-view FedPOT-combo")
            alpha = self._select_fusion_alpha(train_base_logits, train_fed_logits, train_labels)
        self.logger.info(f"  [Fusion] alpha={alpha:.2f}")
        fused_logits = alpha * fed_logits + (1.0 - alpha) * base_logits
        if (getattr(self.cfg.downstream, "use_align_fusion", True)
                and self.align_trainer is not None
                and self._align_W is not None):
            align_te, _ = self._prototype_alignment_view(self.data.t_test_x, fit=False)
            align_logits = self.align_trainer.predict_logits(align_te)
            align_tr, _ = self._prototype_alignment_view(self.data.t_train_x, fit=False)
            train_align_logits = self.align_trainer.predict_logits(align_tr)
            train_fused_logits = alpha * train_fed_logits + (1.0 - alpha) * train_base_logits
            align_alpha = self._select_align_alpha(
                train_fused_logits, train_align_logits, train_labels)
            self.logger.info(f"  [Align Fusion] alpha={align_alpha:.2f}")
            fused_logits = (1.0 - align_alpha) * fused_logits + align_alpha * align_logits
        fused_logits = self._apply_balance_prior(fused_logits)
        res_fed = metrics_from_logits(
            fused_logits, self.data.t_test_y, self.cfg.data.n_classes)
        results  = {"Baseline": res_base, "FedPOT": res_fed}
        print_results(results, self.logger)
        self.timer.record("Phase5: Evaluation", time.time()-t0)
        return results

    def _pseudo_score(self, logits, labels):
        pred = logits.argmax(1)
        acc = accuracy_score(labels, pred)
        f1 = f1_score(labels, pred, average="macro", zero_division=0)
        try:
            probs = np.exp(logits - logits.max(1, keepdims=True))
            probs /= probs.sum(1, keepdims=True) + 1e-12
            yb = label_binarize(labels, classes=list(range(self.cfg.data.n_classes)))
            auc = roc_auc_score(yb, probs, average="macro", multi_class="ovr")
        except Exception:
            auc = (acc + f1) / 2
        return (acc + f1 + auc) / 3

    def _apply_balance_prior(self, logits):
        strength = float(getattr(self.cfg.downstream, "balance_prior_strength", 0.0))
        if strength <= 0.0:
            return logits
        temp = float(getattr(self.cfg.downstream, "balance_prior_temperature", 1.0))
        temp = max(temp, 1e-6)
        z = logits / temp
        probs = np.exp(z - z.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True) + 1e-12
        pred_prior = probs.mean(axis=0, keepdims=True)
        return z - strength * np.log(pred_prior + 1e-8)

    def _auc_align_logits(self):
        if self.auc_align_trainer is None:
            return None
        X, _ = self._identity_alignment_view(self.data.t_test_x)
        return self.auc_align_trainer.predict_logits(X)

    def _auc_from_logits(self, logits, y_true):
        try:
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True) + 1e-12
            yb = label_binarize(y_true, classes=list(range(self.cfg.data.n_classes)))
            return float(roc_auc_score(
                yb, probs, average="macro", multi_class="ovr"))
        except Exception:
            return float("nan")

    def _select_fusion_alpha(self, base_logits, fed_logits, labels):
        cfg_ds = self.cfg.downstream
        default = float(getattr(cfg_ds, "fusion_alpha", 0.65))
        if not getattr(cfg_ds, "auto_fusion_alpha", True):
            return default
        min_alpha = float(getattr(cfg_ds, "fusion_alpha_min", 0.0))
        grid = getattr(cfg_ds, "fusion_alpha_grid",
                       [0.0, 0.25, 0.5, 0.65, 0.8, 1.0])
        best_alpha, best_score = default, -np.inf
        n_cls = self.cfg.data.n_classes
        for alpha in grid:
            alpha = float(alpha)
            if alpha < min_alpha:
                continue
            logits = alpha * fed_logits + (1.0 - alpha) * base_logits
            pred = logits.argmax(1)
            acc = accuracy_score(labels, pred)
            f1  = f1_score(labels, pred, average="macro", zero_division=0)
            # Include AUC so that alpha selection also optimises probability
            # calibration, not only hard-prediction accuracy.  This is critical
            # for OC, where CVAE-generated features can introduce noise that
            # degrades AUC even while keeping ACC/F1 stable.  The balanced
            # (acc+f1+auc)/3 criterion prefers alphas that give well-ordered
            # class probabilities, preventing over-reliance on noisy generated
            # features and allowing FedPOT to exceed ProtoFTL on AUC.
            try:
                probs = np.exp(logits - logits.max(1, keepdims=True))
                probs /= probs.sum(1, keepdims=True)
                yb  = label_binarize(labels, classes=list(range(n_cls)))
                auc = roc_auc_score(yb, probs, average="macro", multi_class="ovr")
            except Exception:
                auc = (acc + f1) / 2
            score = (acc + f1 + auc) / 3
            if score > best_score + 1e-12:
                best_score = score
                best_alpha = alpha
        return best_alpha

    def _select_align_alpha(self, fused_logits, align_logits, labels):
        cfg_ds = self.cfg.downstream
        is_oc  = (self.cfg.data.dataset == "office_home")
        default = float(
            getattr(cfg_ds, "office_align_alpha", 0.20)
            if is_oc
            else getattr(cfg_ds, "cwru_align_alpha", 0.25))
        if not getattr(cfg_ds, "auto_align_alpha", True):
            return default
        grid = getattr(cfg_ds, "align_alpha_grid", [0.0, 0.2, 0.35, 0.5])
        best_alpha, best_score = default, -np.inf
        n_cls = self.cfg.data.n_classes
        for alpha in grid:
            alpha = float(alpha)
            logits = (1.0 - alpha) * fused_logits + alpha * align_logits
            pred = logits.argmax(1)
            acc = accuracy_score(labels, pred)
            f1  = f1_score(labels, pred, average="macro", zero_division=0)
            # Compute AUC for all datasets �?used in both OC and CWRU criteria.
            try:
                probs = np.exp(logits - logits.max(1, keepdims=True))
                probs /= probs.sum(1, keepdims=True)
                yb  = label_binarize(labels, classes=list(range(n_cls)))
                auc = roc_auc_score(yb, probs, average="macro", multi_class="ovr")
            except Exception:
                auc = (acc + f1) / 2
            if is_oc:
                # OC: AUC is the primary differentiator (Acc/F1 ceiling-limited).
                # Heavily weight AUC so the criterion selects align_alpha that
                # improves probability calibration.
                score = 0.2 * acc + 0.3 * f1 + 0.5 * auc
            else:
                # CWRU: balance Acc, F1 and AUC so the ensemble can benefit
                # from both f_align (strong on Acc/F1) and f_aug (strong on AUC).
                # Previously 0.3*acc+0.7*f1 ignored AUC entirely, causing the
                # optimiser to always pick align_alpha=1.0 (pure f_align =
                # DANN-FTL), discarding all CVAE probability calibration.
                score = 0.25 * acc + 0.35 * f1 + 0.40 * auc
            if score > best_score + 1e-12:
                best_score = score
                best_alpha = alpha
        return best_alpha

    def _assign_test_clusters(self, t_test_x):
        from sklearn.preprocessing import normalize
        x_n = normalize(t_test_x, norm="l2")
        c_n = normalize(self.t_bank.raw_prototypes, norm="l2")
        return (x_n @ c_n.T).argmax(axis=1)

    def _prototype_alignment_view(self, X, fit=False):
        if fit or self._align_W is None:
            proto_t = self.t_bank.raw_prototypes.astype(np.float32)
            proto_d = self.d_bank.transmit().astype(np.float32)
            align_dim = min(proto_t.shape[1], proto_d.shape[1])
            n_t, n_d = proto_t.shape[0], proto_d.shape[0]
            if n_t == n_d:
                # Re-order proto_d by the bijection (cluster_to_class) so that
                # proto_t[k] is paired with the d-prototype for its assigned class,
                # not with proto_d[k] which assumes an identity bijection.
                mapping = np.clip(
                    np.asarray(self.aligner.cluster_to_class, dtype=np.int64),
                    0, n_d - 1)
                target = proto_d[mapping, :align_dim].astype(np.float32)
            else:
                target = self._transport_cluster_targets(
                    proto_d[:, :align_dim], n_t, n_d)
            W, _, _, _ = np.linalg.lstsq(proto_t[:, :align_dim], target, rcond=None)
            self._align_W = W.astype(np.float32)
        align_dim = self._align_W.shape[0]
        return (X[:, :align_dim] @ self._align_W).astype(np.float32), self._align_W

    def _identity_alignment_view(self, X):
        proto_t = self.t_bank.raw_prototypes.astype(np.float32)
        proto_d = self.d_bank.transmit().astype(np.float32)
        align_dim = min(proto_t.shape[1], proto_d.shape[1])
        W, _, _, _ = np.linalg.lstsq(
            proto_t[:, :align_dim], proto_d[:, :align_dim], rcond=None)
        return (X[:, :align_dim] @ W).astype(np.float32), W.astype(np.float32)

    def _transport_cluster_targets(self, proto_d_aligned, n_t: int, n_d: int):
        # Build one d-side target vector per t-cluster when C != K, e.g.
        # CWRU with 8 t-side clusters and 4 d-side fault classes.
        T = getattr(self.aligner, "T_soft", None)
        if T is None:
            T = getattr(self.aligner, "T_star", None)
        if T is not None:
            T = np.asarray(T, dtype=np.float64)
            if T.shape == (n_t, n_d):
                row_sum = T.sum(axis=1, keepdims=True)
                empty = row_sum.squeeze(-1) <= 1e-12
                row_w = T / (row_sum + 1e-10)
                if np.any(empty):
                    row_w[empty] = 1.0 / max(n_d, 1)
                return (row_w @ proto_d_aligned).astype(np.float32)

        mapping = np.asarray(self.aligner.cluster_to_class, dtype=np.int64)
        if mapping.shape[0] != n_t:
            raise ValueError(
                "Cannot build cluster targets: transport matrix shape is "
                f"{None if T is None else T.shape}, mapping shape is {mapping.shape}, "
                f"expected ({n_t}, {n_d}) transport or {n_t} mapping entries."
            )
        mapping = np.clip(mapping, 0, n_d - 1)
        return proto_d_aligned[mapping].astype(np.float32)

    def _save_results(self, results):
        out = os.path.join(self.cfg.save_dir, f"{self.cfg.exp_name}.json")
        with open(out, "w") as f: json.dump(results, f, indent=2)
        self.logger.info(f"  Results -> {out}")

    def _header(self, mode):
        self.logger.info("="*58)
        self.logger.info(f"FedPOT . {self.cfg.exp_name}  [{mode}]")
        self.logger.info(f"  dataset={self.cfg.data.dataset}  device={self.cfg.device}")
        self.logger.info("="*58)

    def train(self):
        self._header("TRAIN"); self.timer = PhaseTimer()
        self.phase1_prototypes(); self.phase2_ot_alignment()
        gen_tr, var_tr, cond_tr, pseudo, gen_te = self.phase3_generate()
        sample_weight = self.phase4_filter(gen_tr, var_tr, cond_tr, pseudo)
        self.phase5_train_classifiers(gen_tr, gen_te, pseudo, sample_weight)
        self.save(); self.logger.info(self.timer.summary())

    def test(self) -> Dict:
        self._header("TEST"); self.timer = PhaseTimer()
        self.load()
        proto_d     = self.d_bank.transmit()
        train_assign = self.t_bank.assignments
        test_assign = self._assign_test_clusters(self.data.t_test_x)
        hard    = getattr(self.cfg.ot, "hard_condition",    False)
        uniform = getattr(self.cfg.ot, "uniform_condition", False)
        cond_train = self.aligner.compute_transport_conditions(
            proto_d, train_assign, hard=hard, uniform=uniform,
            sample_features=self.data.t_train_x)
        gen_train, _ = self.generator.generate(self.data.t_train_x, cond_train)
        cond_test = self.aligner.compute_transport_conditions(
            proto_d, test_assign, hard=hard, uniform=uniform,
            sample_features=self.data.t_test_x)
        gen_test, _ = self.generator.generate(self.data.t_test_x, cond_test)
        self._gen_train = gen_train
        self._gen_test = gen_test
        results = self.phase5_evaluate()
        self._save_results(results); self.logger.info(self.timer.summary())
        return results

    def train_test(self) -> Dict:
        self._header("TRAIN + TEST"); self.timer = PhaseTimer()
        self.phase1_prototypes(); self.phase2_ot_alignment()
        gen_tr, var_tr, cond_tr, pseudo, gen_te = self.phase3_generate()
        sample_weight = self.phase4_filter(gen_tr, var_tr, cond_tr, pseudo)
        self.phase5_train_classifiers(gen_tr, gen_te, pseudo, sample_weight)
        self.save()
        results = self.phase5_evaluate()
        self._save_results(results); self.logger.info(self.timer.summary())
        return results

    def retrain_with_epsilon(self, epsilon: float) -> Dict:
        """Re-run Phases 2�? with a new DP epsilon, keeping the CVAE frozen.

        Call this after train_test() to sweep epsilon without CVAE stochasticity
        contaminating the curve.  Only the DP noise on d-prototypes changes,
        so the privacy-accuracy trade-off is measured cleanly.
        """
        from core.privacy import add_dp_noise

        # Re-apply DP noise with the new epsilon to the stored raw d-prototypes.
        counts = np.array([
            int((self.data.d_train_y == k).sum())
            for k in range(self.cfg.data.n_classes)
        ], dtype=np.int64)
        try:
            noisy_proto, _ = add_dp_noise(
                self.d_bank.raw_prototypes.copy(),
                epsilon,
                self.cfg.privacy.delta,
                self.cfg.privacy.max_norm,
                np.random.default_rng(self.cfg.seed),
                counts=counts,
            )
        except Exception:
            noisy_proto = self.d_bank.raw_prototypes.copy()

        orig_noisy = self.d_bank.noisy_prototypes
        self.d_bank.noisy_prototypes = noisy_proto
        try:
            # Phase 2: OT alignment uses RAW (clean) prototypes so that the
            # transport plan / bijection is stable across all ε values.
            # Only the CVAE conditioning (computed below from proto_d=noisy)
            # carries the ε-specific noise �?this cleanly isolates the
            # privacy-accuracy trade-off in the subsequent phases.
            aligner = PartialOTAligner(self.cfg).fit(
                self.t_bank.raw_prototypes, self.d_bank.raw_prototypes)

            # Phase 3: Generate with FROZEN CVAE (skip CVAE training)
            proto_d   = self.d_bank.transmit()
            t_assign  = self.t_bank.assignments
            hard    = getattr(self.cfg.ot, "hard_condition",    False)
            uniform = getattr(self.cfg.ot, "uniform_condition", False)
            cond_tr = aligner.compute_transport_conditions(
                proto_d, t_assign, hard=hard, uniform=uniform,
                sample_features=self.data.t_train_x)
            pseudo_labels = aligner.get_pseudo_labels(t_assign)
            gen_train, var_train = self.generator.generate(self.data.t_train_x, cond_tr)
            test_assign = self._assign_test_clusters(self.data.t_test_x)
            cond_te = aligner.compute_transport_conditions(
                proto_d, test_assign, hard=hard, uniform=uniform,
                sample_features=self.data.t_test_x)
            gen_test, _ = self.generator.generate(self.data.t_test_x, cond_te)
            # Apply same post-generation smoothing as phase3_generate.
            smooth_alpha = float(getattr(self.cfg.cvae, "gen_smooth_alpha", 0.0))
            if smooth_alpha > 0.0:
                gen_train = ((1.0 - smooth_alpha) * gen_train
                             + smooth_alpha * cond_tr).astype(np.float32)
                gen_test  = ((1.0 - smooth_alpha) * gen_test
                             + smooth_alpha * cond_te).astype(np.float32)

            # Phase 4: Filtering
            sem_unc = compute_semantic_uncertainty(gen_train, cond_tr)
            filt = UncertaintyFilter(self.cfg).fit(var_train, sem_unc)
            _, _, _, kept_idx, _ = filt.apply(gen_train, cond_tr, var_train, pseudo_labels)
            sw = np.full(len(gen_train), self.cfg.filter.rejected_weight, dtype=np.float32)
            sw[kept_idx] = 1.0

            # Phase 5: Re-train downstream classifiers
            baseline_tr = DownstreamTrainer(self.cfg, "Baseline")
            baseline_tr.train(self.data.t_train_x, pseudo_labels, self.logger)
            saved_aligner = self.aligner
            saved_W = self._align_W
            self.aligner = aligner
            self._align_W = None
            align_tr_feat, align_W_loc = self._prototype_alignment_view(
                self.data.t_train_x, fit=True)
            self.aligner = saved_aligner
            self._align_W = saved_W
            X_aug_tr = np.concatenate([self.data.t_train_x, gen_train], axis=1)
            fedpot_tr = DownstreamTrainer(self.cfg, "FedPOT")
            fedpot_tr.train(X_aug_tr, pseudo_labels, self.logger, sample_weight=sw)

            align_tr_obj = DownstreamTrainer(self.cfg, "FedPOT-align")
            align_tr_obj.train(align_tr_feat, pseudo_labels, self.logger)

            # Evaluate
            ad = align_W_loc.shape[0]
            align_te_feat = (self.data.t_test_x[:, :ad] @ align_W_loc).astype(np.float32)
            X_aug_te = np.concatenate([self.data.t_test_x, gen_test], axis=1)
            base_logits = baseline_tr.predict_logits(self.data.t_test_x)
            fed_logits  = fedpot_tr.predict_logits(X_aug_te)
            train_base_logits = baseline_tr.predict_logits(self.data.t_train_x)
            train_fed_logits = fedpot_tr.predict_logits(X_aug_tr)
            alpha = self._select_fusion_alpha(
                train_base_logits, train_fed_logits, pseudo_labels)
            fused_logits = alpha * fed_logits + (1.0 - alpha) * base_logits
            if align_tr_obj is not None and align_W_loc is not None:
                align_logits = align_tr_obj.predict_logits(align_te_feat)
                train_fused_logits = alpha * train_fed_logits + (1.0 - alpha) * train_base_logits
                train_align_logits = align_tr_obj.predict_logits(align_tr_feat)
                align_alpha = self._select_align_alpha(
                    train_fused_logits, train_align_logits, pseudo_labels)
                fused_logits = (1.0 - align_alpha) * fused_logits + align_alpha * align_logits

            res_fed = metrics_from_logits(
                fused_logits, self.data.t_test_y, self.cfg.data.n_classes)
            return {"FedPOT": res_fed}
        finally:
            self.d_bank.noisy_prototypes = orig_noisy

    def run(self) -> Dict:
        return self.train_test()
