# MultiScaleToxic

> A research implementation for peptide property prediction via **atom–residue dual-scale integration**.

## Overview

**MultiScaleToxic** is a research codebase for peptide property prediction under a dual-scale modeling paradigm.  
The associated manuscript presents the framework as **UniPept**, a universal peptide representation model that jointly integrates:

- **atom-level local chemical and geometric information**
- **residue-level global semantic and physicochemical information**

instead of relying on only a single scale.

The main motivation is straightforward: peptide function and toxicity are rarely determined by only one view.  
Atom-level interactions capture local chemistry such as bond environments, distances, and micro-structural effects, while residue-level representations capture sequence semantics, physicochemical profiles, and higher-order functional context. UniPept combines both views in a unified architecture to improve robustness, interpretability, and generalization across peptide-related tasks.

---

## Why this project matters

Single-scale peptide models usually suffer from one of two limitations:

- **Residue-only models** capture sequence semantics and global context, but miss fine-grained atomic interactions.
- **Atom/structure-only models** capture local geometry, but often underuse sequence evolution and physicochemical information.

This project is built around the idea that **peptide behavior emerges from cross-scale interaction**.  
For example:

- in **toxicity prediction**, local chemical interactions and global conformational properties jointly affect membrane disruption and safety;
- in **bioactivity prediction**, atom-level geometry helps model local functional constraints while residue-level features capture global physicochemical behavior;
- in **interaction tasks**, local spatial contacts and global domain compatibility both matter.

---

## Method summary

The overall framework contains four major stages.

### 1. Atomic feature extraction

At the atomic scale, peptide sequences are converted into atom-level representations using molecular construction and geometric encoding.

This branch integrates:

- **RDKit-derived atomic descriptors**
- **2D molecular topology**
- **atom–atom shortest-path relations**
- **3D atomic coordinates from predicted peptide structures**
- **distance-aware geometric encoding**

These features are passed into a geometry-aware encoder to model local spatial interactions.

### 2. Residue feature extraction

At the residue scale, each peptide is represented using:

- **ESM2 residue embeddings**
- **AAindex physicochemical descriptors**
- **residue–residue geometric relationships**
- **position-aware residue modeling**

This branch captures sequence semantics, residue chemistry, and global conformational context.

### 3. Atom–residue fusion

The core contribution of the framework is the fusion block.

Instead of simply concatenating atom and residue features, the model uses:

- **cross-attention** for cross-scale alignment
- **dynamic gated attention** for adaptive information injection
- **residual adaptation and normalization** for stable fusion

Conceptually:

- atom-level features act as **queries**
- residue-level features act as **keys/values**
- the gate dynamically decides how much residue information should be injected into the final peptide representation

This design allows the model to preserve local structural precision while incorporating global biological semantics.

### 4. Downstream prediction

The fused peptide representation is used for multiple downstream tasks, including:

- peptide toxicity prediction
- anti-inflammatory peptide prediction
- antibacterial peptide prediction
- antifungal peptide prediction
- antiviral peptide prediction
- anticancer peptide prediction
- exploratory downstream transfer to DTI/PPI-style settings

---

## Architecture at a glance

```text
Peptide sequence
    │
    ├── Atom branch
    │   ├── RDKit descriptors
    │   ├── 2D topology / shortest paths
    │   ├── 3D coordinates
    │   └── Geometry-aware encoder
    │
    ├── Residue branch
    │   ├── ESM2 embeddings
    │   ├── AAindex features
    │   ├── Residue geometry
    │   └── Position-aware attention
    │
    └── Atom–Residue Fusion
        ├── Cross-attention
        ├── Dynamic gated fusion
        ├── Residual adaptation
        └── Final peptide representation
                │
                └── Property prediction head
