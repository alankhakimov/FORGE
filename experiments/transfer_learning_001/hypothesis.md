H1 — Performance (Primary) A pretrained vision model, fine-tuned on our task, will achieve higher performance than the same architecture trained from scratch, on a small/limited labeled dataset — and the size of that gain will depend on how well the pretraining domain matches the target domain. Specifically:

- H1a: Any pretraining (ImageNet or RadImageNet) will outperform training from scratch, because pretraining provides transferable low- and mid-level feature detectors that a from-scratch model must otherwise learn from far less data.
- H1b: RadImageNet-pretrained will outperform ImageNet-pretrained, because RadImageNet's pretraining domain (radiological images, including ultrasound) is a closer match to the target task than ImageNet's natural photographs — the transferred features should require less adaptation.

Three arms, same architecture, only initialization differs:

1. From scratch (random init)
2. ImageNet-pretrained (domain-mismatched: natural photos → ultrasound)
3. RadImageNet-pretrained (domain-matched: radiology, including ultrasound → ultrasound)

Falsified if: performance is equal across arms, or from-scratch matches/beats either pretrained arm (falsifies H1a); or ImageNet-pretrained matches/beats RadImageNet-pretrained (falsifies H1b specifically, while leaving H1a potentially intact).

H2 — Representation Geometry (Secondary) The two pretrained models will show clearer class separation at earlier layers than the from-scratch model's, and their attention/activation maps will localize more consistently to task-relevant image regions — i.e., the H1 performance gains are explained by better-organized internal structure, not just "better weights." Further, RadImageNet's representations are expected to organize around task-relevant structure (tissue texture, lesion boundaries) more directly than ImageNet's, which organizes around natural-image concepts (edges, textures, object parts) that only partially transfer to ultrasound.

Why H2 matters: H1 alone only shows which model(s) win. H2 shows why — and with three arms instead of two, it can also show whether a performance edge from domain-matched pretraining (H1b) actually corresponds to a qualitatively different internal representation, or just a quantitatively better version of the same one. That distinction is the difference between an architecture-learning exercise and a single benchmark number.

Architecture Decision: Use an existing, well-documented architecture — ResNet-50 — across all three arms, not a custom-designed CNN.

Rationale: Designing a custom CNN and comparing it to pretrained models introduces a confound: a loss could mean "pretraining helps" (the actual claim) or simply "this custom design is mediocre" (an unrelated claim). This would invalidate the isolation the experiment needs. Using the same architecture in every arm — random init vs. ImageNet init vs. RadImageNet init — makes pretraining source the only variable, which is what H1 and H2 actually claim to be testing. ResNet-50 (\~25M params) is the specific size the change is built around: RadImageNet's public pretrained checkpoints are released as ResNet50, DenseNet121, and InceptionV3 — not ResNet18 — so ResNet-50 is the architecture that keeps all three arms genuinely comparable, at the cost of a somewhat heavier model than the ResNet-18 originally planned. It remains tractable to train from scratch on a small dataset with the frozen-backbone and regularization strategy already planned, and is a standard baseline in medical imaging literature, so results stay externally comparable.

Deferred to a separate experiment: custom architecture design (e.g., tailored to ultrasound-specific characteristics like speckle noise or radial geometry) belongs in its own decoupled hypothesis — proposed as Experiment 002 — once a working pipeline and baseline numbers exist to beat.
