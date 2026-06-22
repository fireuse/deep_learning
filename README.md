# Forecasting Electricity Consumption and Classifying Weekends

**Authors:** Adam Topolski (12600468), Adrianna Bartoszek (12601333), Wojciech Jurewicz (12600946), Katarzyna Kordala (12600809)

This repository contains the implementation of a deep learning approach (PatchTST) and a classical baseline (Prophet) to forecast electricity consumption for 370 clients.

## Tasks
1. **Multivariate Forecasting**: Given 256 past 15-minute intervals (64 hours), forecast the next 96 intervals (24 hours).
2. **Weekend Classification**: Classify whether the consumption data falls on a weekend using a shared encoder approach.

## Project Structure
- `eda_deep.ipynb` - Deep exploratory data analysis, stationarity tests, and feature discovery.
- `patchtst_final_training.ipynb` - Final architecture, training loop, and evaluation for the PatchTST model.
- `patchtst_patch_size_experiments.ipynb` - Hyperparameter sweep and patch-size experiments.
- `prophet_all_normalized_256_96.ipynb` - Prophet forecasting baseline (matched to the 256 lookback/96 horizon).
- `xgboost.ipynb` - XGBoost classification baseline.

## Getting Started

1. **Dataset Setup**: Download the _Electricity Load Diagrams 2011-2014_ dataset from the UCI Machine Learning Repository and place the extracted `LD2011_2014.txt` file directly into the `dataset/` folder.
2. **Dependencies**: This project uses `uv`. Install the required packages via `pyproject.toml` and `uv.lock`.
3. **Execution**: You can reproduce the results by running the notebooks, starting with the EDA and proceeding to the model trainings.