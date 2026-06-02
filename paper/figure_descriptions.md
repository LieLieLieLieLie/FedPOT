# FedPOT figure descriptions

This file describes the non-experimental figures currently reserved in
`cas-sc-template.tex`. The placeholders are intentionally blank boxes; replace
them later with final artwork when the visual style is ready.

## Graphical abstract

Purpose: one-page visual summary for Elsevier submission.

Recommended content: left-to-right flow with four blocks: source party with
labelled data, DP Gaussian prototype release, target-side relational partial OT
alignment, transport-conditioned CVAE generation, and target prediction. Use a
privacy boundary between source and target. Emphasise that only sanitised
prototypes cross the boundary.

## Figure 1: Motivation and contribution map

Purpose: explain why FedPOT is needed before technical details.

Recommended content: show three isolated challenges in separate lanes:
privacy, semantic alignment, and feature augmentation. Then show FedPOT as a
unified solution that connects the lanes. A compact Venn/flow hybrid works
well: DP prototypes protect transmission, relational OT resolves semantic
matching, and CVAE generation creates complementary features.

## Figure 2: Problem setting

Purpose: define unsupervised vertical FTL visually.

Recommended content: two parties. Source side has labelled source features
`(x_s, y_s)` and class prototypes. Target side has unlabelled target features
`x_t`, clustering, pseudo-label inference, and classifier training. Add a
strict "no raw data sharing" boundary. This figure should make clear why direct
source-target feature comparison is not available.

## Figure 3: Relational partial OT alignment

Purpose: make the key OT novelty intuitive.

Recommended content: show target cluster centres and source prototypes in
separate spaces. Instead of drawing a direct distance between them, draw
within-domain distance matrices, row-sorted structural signatures, a relational
cost matrix, a partial OT plan, and finally a cluster-to-class bijection. This
figure should visually justify why the method is robust when feature spaces are
disjoint or DP-noisy.

## Figure 4: Transport-conditioned CVAE architecture

Purpose: explain how alignment becomes generative augmentation.

Recommended content: encoder receives only target feature `x_t` and outputs
latent parameters. Decoder receives latent code plus OT condition and produces
source-style complementary feature. Show three losses around the decoder:
Gaussian reconstruction NLL, KL warm-up, and Sinkhorn OT regularisation. Add a
small post-generation smoothing block: generated feature plus OT condition to
final complementary feature.

## Figure 5: Hard-label ceiling vs probability ranking

Purpose: support the Office-Caltech discussion.

Recommended content: two panels. Left: hard-label predictions have the same
accuracy ceiling because all methods depend on the same unsupervised
cluster-to-class pseudo-label quality. Right: probability score rankings differ,
so AUC separates methods. FedPOT should have better ordered positive/negative
score distributions or a higher ROC curve.
