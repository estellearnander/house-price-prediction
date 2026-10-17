# House Price Prediction

## Objective
Predict house sale prices for the Kaggle House Prices: Advanced Regression
Techniques competition, using a reproducible, packaged Python project.

## Author
Estelle Arnander (Kaggle username: estellearnander03)

## Setup

1. Create the conda environment:
```bash
   conda env create -f environment.yml
```

2. Activate the environment:
```bash
   conda activate house-price-prediction
```

3. Install the package in editable mode:
```bash
   python -m pip install -e .
```

## Running the analysis

1. Open `notebooks/house_price_analysis.ipynb` in Jupyter or PyCharm.
2. Make sure the notebook's kernel is set to the `house-price-prediction`
   environment.
3. Run all cells from top to bottom.
4. The trained model's predictions are saved to `results/submission.csv`,
   ready for submission to Kaggle.

## Running the tests

```bash
pytest -q
```

## Dataset

`data/train.csv` and `data/test.csv` were downloaded from the Kaggle
House Prices: Advanced Regression Techniques competition page.

## Results summary

An XGBoost regression model was tuned using cross-validated random search,
achieving a validation RMSE of 0.128 on log-transformed sale prices. See
`report.md` for full details and `results/feature_importance.png` for the
most influential predictors.

## Limitations

See `report.md` for a full discussion of limitations, including
simplified imputation and the absence of model ensembling.
