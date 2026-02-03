# Machine Learning Assisted Prediction and Synthesis of New High-Entropy Fluorite Oxide

This repository contains the code and data needed to reproduce (and extend) the results from the manuscript:

**Machine Learning Assisted Prediction and Synthesis of New High-Entropy Fluorite Oxide**

## Abstract (paper)
High-entropy oxides are a novel class of materials with promising applications in energy conversion and storage; however, their rational design remains challenging due to the immense compositional space. Here, we propose a machine-learning-based methodology to design stable, single-phase HEOs. We trained predictive models to identify candidate fluorite-structured compositions. The ensemble achieved reasonable performance in a six-class classification task, as evaluated using nested stratified cross-validation and external validation (weighted-average F1-scores of 77% and 70%, respectively). We further applied SHAP analysis to assess the physical relevance of the predictors. Experimentally, we synthesized Ce0.2La0.2Nd0.2Mg0.2Al0.2O2−δ. X-ray diffraction confirmed this prediction, and transmission electron microscopy (TEM) combined with selected-area electron diffraction (SAED) validated the phase assignment. Overall, these results demonstrate that machine learning is a powerful approach to navigate the complex HEO compositional landscape and accelerate the discovery of materials with targeted properties.

## Requirements
- **Python:** 3.11.x (recommended: 3.11.6)
- Main dependencies are listed in `requirements.txt`.

## Install (recommended)
Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows (CMD)
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt```


## Repository structure (quick guide)

Below is a brief description of the main folders in this repository:

- `Dataset/`  
  Input datasets and auxiliary data used throughout training, evaluation, and validation.

- `Feature_calculation/`  
  Feature engineering pipelines used to transform compositions into descriptors.
  - `Feature_calculation/Atomic_features_package/`  
    Atomic/composition-derived descriptor generation utilities.
  - `Feature_calculation/Mendeleev_features/`  
    Descriptor generation based on Mendeleev/periodic-table properties.
  - `Feature_calculation/Thermo_features/`  
    Thermodynamic descriptor workflows (e.g., energetics/thermo-derived features).

- `Evaluation_framework/`  
  Model evaluation pipelines and artifacts (e.g., nested CV runs, selection workflows).
  - `Evaluation_framework/Pearson/`  
    Pearson-correlation based feature filtering/evaluation workflow and artifacts.
  - `Evaluation_framework/PFI/`  
    Permutation Feature Importance (PFI) workflow and artifacts.

- `External validation/`  
  Scripts and artifacts for the external validation protocol (hold-out external dataset evaluation).

- `Figures_codes/`  
  Code used to reproduce manuscript figures.
  - `Figures_codes/Confusion_Matrix/`  
    Confusion matrix plotting scripts.
  - `Figures_codes/Pareto_Chart/`  
    Pareto chart scripts.
  - `Figures_codes/Pearson_Corr_Matrix/`  
    Pearson correlation matrix plotting scripts.
  - `Figures_codes/ROC/`  
    ROC curve generation scripts and fold-level artifacts.
  - `Figures_codes/SHAP/`  
    SHAP analysis scripts and artifacts.

- `Prediction_framework/`  
  Inference utilities to predict the most likely phase for new compositions using the trained ensemble.


## References for detailed information about each model

Random forests and Gradient boosting: https://scikit-learn.org/1.5/modules/ensemble.html;

Multi-layer Perceptron: https://scikit-learn.org/1.5/modules/neural_networks_supervised.html#multi-layer-perceptron;

Extreme Learning Machine: https://github.com/5663015/elm . 




