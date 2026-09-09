# Product-Image Classifier (CNN)

A convolutional neural network that classifies e-commerce product photos into **Apparel**, **Electronics**, and **Home** — built as a portfolio project simulating a real catalogue-automation task.

## Problem

An e-commerce catalogue team wants to auto-sort incoming product images into three top-level categories instead of tagging them by hand. The model needs to be accurate enough to trust on the easy cases, and clear about *where it struggles* so ambiguous images can be routed to manual review.

## Goals

- Train a small CNN from scratch (no huge pretrained backbone) on Colab's free GPU
- Push validation accuracy past 85%
- Produce a confusion matrix that names exactly which category pairs get confused, and explain why

## Dataset

*(fill in once finalized)* — Images sourced from public e-commerce datasets on Kaggle, organized into `data/raw/Apparel`, `data/raw/Electronics`, `data/raw/Home`. See `data/README.md` for sources and download steps.

## Approach

- **Framework:** TBD
- **Architecture:** small custom CNN (conv → pool blocks + dense head)
- **Preprocessing:** resize, normalize, light augmentation (flip / rotate / zoom)
- **Split:** 80/20 train/validation

## Results

*(fill in after training)*

- Validation accuracy: —
- Confusion matrix: see `reports/`
- Key failure modes: —

## Run it yourself

1. Open the notebook in `notebooks/` in Google Colab
2. Set runtime to GPU (Runtime → Change runtime type → GPU)
3. Follow `data/README.md` to pull the dataset
4. Run all cells

## Tech

Computer Vision · CNNs · TensorFlow/PyTorch · Colab GPU · scikit-learn (confusion matrix)

---
*Built as a self-directed portfolio project.*
