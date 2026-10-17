# House Price Prediction — Kaggle: House Prices (Advanced Regression Techniques)

**Kaggle username:** estellearnander03
**Account number:** 36879656

## Objective
Predict house sale prices for the Kaggle House Prices: Advanced Regression
Techniques competition, using a reproducible, packaged Python project.

## Approach

### Data and preprocessing
The dataset (1460 training houses, 1459 test houses, 80 features) contains
a mix of numeric and categorical variables, with substantial missing data
in several columns. Missing values were handled in two ways:

- Columns where a missing value means a feature does not exist (e.g. no
  pool, no garage, no basement) were filled with `"None"`.
- Columns with genuinely missing data (e.g. `LotFrontage`, `Electrical`)
  were imputed using median (numeric) or mode (categorical) values.

Categorical columns were one-hot encoded, and the train and test sets were
combined during preprocessing to keep encoded columns consistent, then
split back apart afterward.

The target variable, `SalePrice`, is right-skewed, so it was
log-transformed (`log1p`) before training, matching the competition's own
RMSE-on-log-price evaluation metric. Predictions were converted back to
dollar values (`expm1`) before submission.

### Model
An XGBoost regressor was used, given its strong performance on structured,
tabular data with mixed feature types. An initial baseline model was
trained, then hyperparameters (`max_depth`, `learning_rate`, `n_estimators`,
`subsample`, `colsample_bytree`) were tuned using `RandomizedSearchCV` with
5-fold cross-validation, evaluated on RMSE.

## Results

| Model             | RMSE (log scale) | MAE (log scale) | R²    |
|-------------------|-------------------|-------------------|-------|
| Baseline XGBoost  | 0.133             | 0.089             | 0.905 |
| Tuned XGBoost      | 0.128             | 0.085             | 0.912 |

Tuning improved all three metrics. The final model was retrained on the
full labeled dataset before generating predictions on the Kaggle test set.

## Feature importance

The most influential features were `OverallQual` (overall material and
finish quality), `GarageCars` (garage capacity), and `GrLivArea`
(above-ground living area), followed by several quality-related
categorical features (e.g. kitchen quality, central air). This aligns with
real-world intuition: house price is driven primarily by overall quality
and size rather than any single structural detail.

## Limitations

This model does not account for external factors such as local market
conditions or interest rates at time of sale. Some imputation choices
(e.g. filling missing `LotFrontage` with the overall median) are
simplifications; more targeted imputation (e.g. grouped by neighborhood)
could improve accuracy further. No ensemble of multiple model types was
attempted.
