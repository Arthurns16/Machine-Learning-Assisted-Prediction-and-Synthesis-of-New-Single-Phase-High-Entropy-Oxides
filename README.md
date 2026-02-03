# Machine Learning Assisted Prediction and Synthesis of New High-Entropy Fluorite Oxide

This repository contains the code and data needed to reproduce (and extend) the results from the manuscript:

**Machine Learning Assisted Prediction and Synthesis of New High-Entropy Fluorite Oxide**

## Abstract (paper)
High-entropy oxides are a novel class of materials with promising applications in energy conversion and storage; however, their rational design remains challenging due to the immense compositional space. In this work, we propose a machine-learning-based methodology to design stable, single-phase HEOs. After engineering composition-derived descriptors, we trained predictive models and used them to identify candidate compositions under predefined design constraints. The ensemble achieved reasonable performance in a six-class classification task, as evaluated through a nested stratified cross-validation protocol and an external validation protocol (F1-scores of 77% and 70%, respectively). We further applied SHAP analysis to assess the physical relevance of the predictors. Experimentally, we synthesized Ce0.2La0.2Nd0.2Mg0.2Al0.2O2−δ, a composition predicted to form a fluorite phase. X-ray diffraction with Rietveld refinement confirmed this prediction, and Transmission Electron Microscopy (TEM) combined with Selected Area Electron Diffraction (SAED) was also used to validate the phase assignment. Overall, these results demonstrate that machine learning is a powerful approach to navigate the complex HEO compositional landscape and accelerate the discovery of materials with targeted properties.

## Requirements
- **Python:** 3.11.x (recommended: 3.11.6)
- Main dependencies are listed in `requirements.txt`.

### Install (recommended)
Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows (CMD)
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt```

# References for detailed information about each model

Random forests and Gradient boosting: https://scikit-learn.org/1.5/modules/ensemble.html;

Multi-layer Perceptron: https://scikit-learn.org/1.5/modules/neural_networks_supervised.html#multi-layer-perceptron;

Extreme Learning Machine: https://github.com/5663015/elm . 




