# TechRxiv Submission Notes

Copy/paste-ready metadata for the TechRxiv upload form.

---

## Title

To Plot a Stone with Six Birds: A Geometry is A Theory

## Abstract

We present a reproducible computational pipeline that derives metric structure from finite-state Markov dynamics without assuming coordinates: substrate → lens ladder → closed macro kernel → path-cost metric. The pipeline includes a suite of coherence audits (idempotence defect, prototype stability, route mismatch, inter-scale distortion, and connectivity) together with a curvature diagnostic based on loop holonomy, and all experiments are driven from committed configuration files with mechanized Lean anchors for the core metric constructions. Building on Six Birds Theory (SBT), we treat geometry not as a background container but as a closure artifact: points are equivalence classes under a packaging map (lens) and distance is the minimal cost of composed transitions (protocols) in the induced macro dynamics. Across controlled substrates, we obtain three auditable results: (i) on an isotropic grid, the induced metric is connected and remains coherent across refinement (bounded closure and distortion audits), while constraints (anisotropic gating) induce a systematic deformation of the emergent metric; (ii) under a fixed deterministic holonomy protocol, loop residue is near-zero on the plane-like grid (median 0.0479) but shifts strongly upward on a sphere-like substrate (median 0.5980), separating flat from curved regimes; (iii) defining cost as negative log transition probability under staged isotropic diffusion yields an approximately quadratic, separable cost law in which the median Pythagorean residual drops from 33.19 at τ=4 to 0.0586 at τ=128, while a Manhattan (L1) control produces diamond contours and does not exhibit this collapse. These results provide a falsification-first, reproducible workflow for comparing emergent geometries (flat versus curved versus constrained versus fractal) without assuming coordinates, and for attaching each claim to audits and explicit controls.

## Keywords

- emergent geometry
- computational metric construction
- closure audits
- graph distances
- reproducible pipelines
- emergence calculus
- path-cost metrics
- manifold learning
- representation auditing
- Lean mechanized anchors

## Author

- **Ioannis Tsiokos**
  - Affiliation: Automorph Inc., Wilmington, DE, USA
  - Email: ioannis@automorph.io
  - ORCID: 0009-0009-7659-5964

## Suggested TechRxiv Categories / Subject Tags

**Primary:**
- Computer Science — Algorithms and Theory

**Secondary (choose 1–2 as applicable):**
- Computer Science — Computational Geometry
- Computer Science — Software Engineering
- Mathematics — Applied Mathematics (if cross-listing allowed)

**Justification:** The paper presents an algorithmic pipeline for constructing metric spaces from Markov dynamics, implements a suite of computational diagnostics (idempotence, distortion, holonomy), provides reproducible experiment configurations, and includes mechanized Lean proofs. The core contributions are computational methods, not pure mathematics or physics.

## Links

- **Zenodo DOI (paper):** https://doi.org/10.5281/zenodo.18494975
- **GitHub repository:** https://github.com/ioannist/six-birds-space
- **SBT Foundations reference:** https://doi.org/10.5281/zenodo.18365949

## License Recommendation

**Recommended: CC BY 4.0**

**Rationale:** CC BY 4.0 (Creative Commons Attribution) is the most widely adopted open-access license for preprints. It maximizes redistribution and reuse while requiring citation, which is standard academic practice. TechRxiv supports this license. If the author later publishes in a journal that requires exclusive rights transfer, note that a CC BY preprint version remains permanently available under that license (which is the standard expectation for preprints). If the author prefers to retain maximum flexibility for future publisher negotiations, "No license" is an alternative — but this limits reuse and may reduce citation and discoverability.

The paper already carries CC-BY 4.0 in its footer, so this is consistent.
