from xgboost import XGBRegressor


def build_xgb_model(**params) -> XGBRegressor:
    default_params = {
        "objective": "reg:squarederror",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": 1,
    }
    default_params.update(params)
    return XGBRegressor(**default_params)
