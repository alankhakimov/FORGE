H1 — Performance (Primary)
A pretrained vision model, fine-tuned on our task, will achieve higher performance than the same architecture trained from scratch, on a small/limited labeled dataset — because pretraining provides transferable low- and mid-level feature detectors that a from-scratch model must learn from far less data.

H2 — Representation Geometry (Secondary)
The pretrained model's internal representations will show clearer class separation at earlier layers than the from-scratch model's, and its attention/activation maps will localize more consistently to task-relevant image regions — i.e., the H1 performance gain is explained by better-organized internal structure, not just "better weights."

Why H2 matters: H1 alone only shows that one model wins. H2 shows why, turning this into an architecture-learning exercise and a mechanistically defensible result, rather than a single benchmark number.

Architecture Decision:
Use an existing, well-documented architecture (ResNet-18) in both arms of H1 — not a custom-designed CNN.

Rationale:
Designing a custom CNN and comparing it to a pretrained model introduces a confound: a loss could mean "pretraining helps" (the actual claim) or simply "this custom design is mediocre" (an unrelated claim). This would invalidate the isolation the experiment needs.
Using the same architecture in both arms — random init vs. pretrained init — makes training strategy the only variable, which is what H1 and H2 actually claim to be testing.
ResNet-18 (~11M params) is tractable to train from scratch on a small dataset, well documented, and is a standard baseline in medical imaging literature, so results are externally comparable.

Deferred to a separate experiment: custom architecture design (e.g., tailored to ultrasound-specific characteristics like speckle noise or radial geometry) belongs in its own decoupled hypothesis — proposed as Experiment 002 — once a working pipeline and baseline numbers exist to beat.
