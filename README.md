# Machine Learning–Assisted Prediction and Synthesis of a New High-Entropy Fluorite Oxide

This repository provides the code, data, and scripts required to reproduce (and extend) the results reported in the manuscript:

**Machine Learning–Assisted Prediction and Synthesis of a New High-Entropy Fluorite Oxide**

---

## Abstract (paper)

High-entropy oxides are a novel class of materials with promising applications in energy conversion and storage; however, their rational design remains challenging due to the immense compositional space. Here, we propose a machine-learning-based methodology to design stable, single-phase HEOs. We trained predictive models to identify candidate fluorite-structured compositions. The ensemble achieved reasonable performance in a six-class classification task, as evaluated using nested stratified cross-validation and external validation (weighted-average F1-scores of 77% and 70%, respectively). We further applied SHAP analysis to assess the physical relevance of the predictors. Experimentally, we synthesized Ce<sub>0.2</sub>La<sub>0.2</sub>Nd<sub>0.2</sub>Mg<sub>0.2</sub>Al<sub>0.2</sub>O<sub>2−δ</sub>. X-ray diffraction confirmed this prediction, and transmission electron microscopy (TEM) combined with selected-area electron diffraction (SAED) validated the phase assignment. Overall, these results demonstrate that machine learning is a powerful approach to navigate the complex HEO compositional landscape and accelerate the discovery of materials with targeted properties.

---

## Requirements

- **Python:** 3.11.x (recommended: 3.11.6)
- Core dependencies are listed in `requirements.txt`.

---

## Installation (recommended)

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv

# Windows (CMD)
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Repository structure

Below is a brief overview of the main folders in this repository. Each directory contains a `Usage.txt` file with minimal execution examples for the scripts.

- `Dataset/`  
  Feature-engineered input dataset. Raw data source: http://dx.doi.org/10.3390/jcs5120311 (Leong et al.)

- `Feature_calculation/`  
  Feature engineering pipelines used to transform compositions into descriptors.
  - `Feature_calculation/Atomic_features_package/`  
    Atomic/composition-derived descriptor generation utilities. Raw data source: https://nomad-lab.eu/nomad-lab/ (NOMAD: NOvel MAterials Discovery Laboratory)
  - `Feature_calculation/Mendeleev_features/`  
    Descriptor generation based on periodic-table properties. Raw data source: https://mendeleev.readthedocs.io/en/stable/ (Mendeleev)
  - `Feature_calculation/Thermo_features/`  
    Thermodynamic descriptor workflows. Raw data source: https://factsage.com/ (FactSage)

- `Evaluation_framework/`  
  Cluster-friendly model evaluation pipelines and artifacts for **nested stratified cross-validation**.  
  If you want to recompute workflow performance metrics, run `nestedcv_finalize.py` and then submit `finalize.sbatch`.
  - `Evaluation_framework/Pearson/`  
    Pearson-correlation-based feature filtering/evaluation workflow and artifacts.
  - `Evaluation_framework/PFI/`  
    Permutation Feature Importance (PFI) workflow and artifacts.

- `External validation/`  
  Scripts and artifacts for the external validation protocol. External dataset source: http://dx.doi.org/10.1016/j.mser.2021.100644 (Akrami et al.)

- `Figures_codes/`  
  Code used to reproduce manuscript figures.
  - `Figures_codes/Confusion_Matrix/` — confusion matrix plotting scripts  
  - `Figures_codes/Pareto_Chart/` — Pareto chart scripts  
  - `Figures_codes/Pearson_Corr_Matrix/` — Pearson correlation matrix plotting scripts  
  - `Figures_codes/ROC/` — ROC curve generation scripts and fold-level artifacts  
  - `Figures_codes/SHAP/` — SHAP analysis scripts and artifacts  

- `Prediction_framework/`  
  Inference utilities to predict the most likely phase for new compositions using the trained ensemble.

  **Step-by-step inference example (fluorite prediction):**
  1. Open a terminal and go to `Prediction_framework/`.
  2. Note that this folder contains the trained models and all required files.
  3. Note that `Inference_dataset.xlsx` contains the computed features for  
     Ce<sub>0.2</sub>La<sub>0.2</sub>Nd<sub>0.2</sub>Mg<sub>0.2</sub>Al<sub>0.2</sub>O<sub>2−δ</sub>.
  4. Following `Usage.txt`, run:
     ```bash
     python run_inference_improved_v2.py --bundle_dir . --input Inference_dataset.xlsx
     ```
     This will reproduce the inference result for the example composition. You can adapt the same workflow to evaluate any other composition, provided the input feature file follows the expected format.

---

## References (model details)

- Random Forests and Gradient Boosting (scikit-learn): https://scikit-learn.org/1.5/modules/ensemble.html  
- Multi-layer Perceptron (scikit-learn): https://scikit-learn.org/1.5/modules/neural_networks_supervised.html#multi-layer-perceptron  
- Extreme Learning Machine (ELM): https://github.com/5663015/elm
