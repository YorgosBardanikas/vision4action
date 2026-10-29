# vision4action

[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2026.03.03.709252-blue.svg)](https://doi.org/10.64898/2026.03.03.709252)

`vision4action` is the analysis code accompanying the study **"Reciprocal Fronto-Parietal Interactions Support Distinct Motor Strategies during Sequential Reaching"** (Bardanikas et al., 2026). The repository contains the Python pipeline used to process behavioral (hand/eye tracking) and electrophysiological (multi-unit activity) recordings from macaque monkeys performing a visually-guided sequential reaching task, and to quantify fronto-parietal interactions during motor anticipation.

- [Overview](#overview)
- [Paper / Citation](#paper--citation)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Repository Structure](#repository-structure)
- [Running the Analyses](#running-the-analyses)

# Overview

This repository holds the analysis scripts used to:

- Quantify **behavioral anticipation** from hand and eye trajectories as target predictability increases (cosine similarity, angular deviation, eye velocity).
- Compute **single-unit activity (SUA)** and **multi-unit activity (MUA)** envelopes from raw extracellular recordings in PMd/M1 and area 7A.
- Decode movement direction from population activity using **Linear Discriminant Analysis (LDA)** to track when preparatory neural representations emerge relative to target onset.
- Quantify directed, feature-specific **inter-areal information transfer** between PMd/M1 and 7A using **Feature-specific Information Transfer (FIT)**.
- Assess shared representational geometry between areas with **Canonical Correlation Analysis (CCA)**.
- Run non-parametric **permutation-based statistics** for group-level comparisons across conditions.

# Paper / Citation

Bardanikas, G., Jana, S., Meirhaeghe, N., Barthélemy, F., Riehle, A., Grün, S., Brovelli, A., & Brochier, T. (2026). *Reciprocal Fronto-Parietal Interactions Support Distinct Motor Strategies during Sequential Reaching*. bioRxiv. https://doi.org/10.64898/2026.03.03.709252

# System Requirements

## Hardware requirements

The core analysis scripts run on a standard workstation with enough RAM to hold session-level electrophysiological recordings in memory. Large-scale batch processing across sessions (e.g. FIT computation over thousands of channel pairs) is intended to be run on a computing cluster; see [`submit.sh`](./submit.sh) for a template job-submission script.

## Software requirements

### OS Requirements

This package is intended for *Linux* environments typical of neuroscience computing clusters.

### Python Dependencies

`vision4action` depends on the Python scientific stack, pinned in [`requirements.txt`](./requirements.txt). Python 3.10+ is recommended.

# Installation Guide

### Install from GitHub

```bash
git clone https://github.com/YorgosBardanikas/vision4action
cd vision4action
pip install -r requirements.txt
python setup.py install
```

- `sudo`, if required for a system-wide install
- Consider installing inside a virtual environment (e.g. `conda`) to keep dependencies isolated

# Repository Structure

```
vision4action/
├── analysis_scripts    # Core analysis pipeline
   ├── CCA/
   ├── FIT/
   ├── LDA/
   ├── mua_plots/
   ├── preprocessing/
   ├── units_analysis/
   ├── behavioral_analysis/
├── README.md
├── requirements.txt     # Python dependencies
├── setup.py             # Package installation script
└── submit.sh            # Template script for submitting batch analysis jobs to a compute cluster
```

# Running the Analyses

The `analysis_scripts/` directory contains the scripts used to reproduce the figures and results reported in the paper.

To submit a batch analysis job to a cluster (e.g. SLURM), adapt and run:

```bash
bash submit.sh
```
