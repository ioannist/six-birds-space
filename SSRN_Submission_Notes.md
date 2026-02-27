# SSRN Submission Notes

Copy/paste-ready metadata and upload checklist for SSRN.

## Build Command

```bash
python3 scripts/build_space_paper.py
```

This generates the SSRN upload set in `paper/build/`.

## Upload Files

- PDF (required): `paper/build/2026_Tsiokos_To_Plot_a_Stone_with_Six_Birds_v1.pdf`
- Form metadata helper: `paper/build/ssrn_metadata.json`

## Title

To Plot a Stone with Six Birds: Constructing and Auditing Emergent Geometry from Markov Dynamics

## Abstract

We present a reproducible computational pipeline that derives metric structure from finite-state Markov dynamics without assuming coordinates: substrate -> lens ladder -> closed macro kernel -> path-cost metric. The pipeline includes a suite of coherence audits (idempotence defect, prototype stability, route mismatch, inter-scale distortion, and connectivity) together with a curvature diagnostic based on loop holonomy, and all experiments are driven from committed configuration files with mechanized Lean anchors for the core metric constructions. Building on Six Birds Theory (SBT), we treat geometry not as a background container but as a closure artifact: points are equivalence classes under a packaging map (lens) and distance is the minimal cost of composed transitions (protocols) in the induced macro dynamics. Across controlled substrates, we obtain three auditable results: (i) on an isotropic grid, the induced metric is connected and remains coherent across refinement (bounded closure and distortion audits), while constraints (anisotropic gating) induce a systematic deformation of the emergent metric; (ii) under a fixed deterministic holonomy protocol, loop residue is near-zero on the plane-like grid (median 0.0479) but shifts strongly upward on a sphere-like substrate (median 0.5980), separating flat from curved regimes; (iii) defining cost as negative log transition probability under staged isotropic diffusion yields an approximately quadratic, separable cost law in which the median Pythagorean residual drops from 33.19 at tau=4 to 0.0586 at tau=128, while a Manhattan (L1) control produces diamond contours and does not exhibit this collapse. These results provide a falsification-first, reproducible workflow for comparing emergent geometries (flat versus curved versus constrained versus fractal) without assuming coordinates, and for attaching each claim to audits and explicit controls. Throughout, "theory" is used in the SBT technical sense - a closure (lens + completion + audit) - not as a speculative hypothesis or manuscript-type category. We emphasize that stronger claims (e.g., continuum-limit convergence to manifolds or exact curvature tensors) are not established here; our conclusions rely on diagnostic proxies under controlled substrates with documented failure modes. This work extends the SBT foundation and complements our prior treatment of emergent mathematics.

## Keywords

- emergent geometry
- computational metric construction
- closure audits
- graph distances
- reproducible pipelines
- emergence calculus

## Author

- Ioannis Tsiokos
  - Affiliation: Automorph Inc., Wilmington, DE, USA
  - Email: ioannis@automorph.io
  - ORCID: 0009-0009-7659-5964

## Suggested SSRN Form Choices

- Paper type: Research Paper
- Language: English
- Classifications: select one primary classification and up to six secondary classifications
- Include coauthor metadata only if coauthors are listed on the title page

## Final Checklist Before Upload

- PDF opens and renders correctly
- PDF file size is <= 100MB
- PDF is not encrypted
- Fonts are embedded and no Type 3 fonts are present
- Title, abstract, keywords, and author metadata in the form match the manuscript
- Upload uses the versioned PDF from `paper/build/`
- Upload is the preprint/manuscript version intended for SSRN (not a publisher-final version)
