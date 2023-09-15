# Pretraining

## Data Location:

**Sleep EDF Expanded Dataset (We used Sleep Cassette Data):** 
https://www.physionet.org/content/sleep-edfx/1.0.0/

**MDD Dataset:**
https://figshare.com/articles/dataset/EEG_Data_New/4244171

## Data Preprocessing Code:

**sleep_preprocessing.m** - MATLAB script used to preprocess sleep data
**mdd_preprocessing.py** - Python used to preprocessing MDD data

## Model Training Code:

**sleep_train.py** - Python script for training the Model S sleep models that were later used in mdd_pretrained.py

**mdd_pretrain.py** - Python script for training MDD Models A through D

## Performance Analysis Code:

**mdd_pretrain_performance_analysis.ipynb** - Jupyter Notebook for performing statistical analysis comparing MDD models performance

## Explainability Code:

**mdd_pretrain_explainability.py** - Python script used to generate spatial and spectral explanations for MDD Models A through D

**mdd_pretrain_visualize_explainability.ipynb** - Jupyter Notebook used to visualize spatial and spectral explainability results
