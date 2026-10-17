import pandas as pd
from house_price_prediction.data import load_house_data
from house_price_prediction.preprocessing import preprocess_data


def test_preprocess_data_has_no_missing_values():
    train_df = load_house_data("data/train.csv").head(50)
    test_df = load_house_data("data/test.csv").head(50)

    X_processed, test_processed, y = preprocess_data(train_df, test_df)

    assert X_processed.isna().sum().sum() == 0
    assert test_processed.isna().sum().sum() == 0
