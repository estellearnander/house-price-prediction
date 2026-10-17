import numpy as np
import pandas as pd


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X = train_df.drop(columns=["saleprice"])
    y = np.log1p(train_df["saleprice"])

    n_train = X.shape[0]
    combined = pd.concat([X, test_df], axis=0, ignore_index=True)

    none_cols = [
        "poolqc", "miscfeature", "alley", "fence", "fireplacequ",
        "garagetype", "garagefinish", "garagequal", "garagecond",
        "bsmtqual", "bsmtcond", "bsmtexposure", "bsmtfintype1", "bsmtfintype2",
        "masvnrtype"
    ]
    combined[none_cols] = combined[none_cols].fillna("None")

    combined["lotfrontage"] = combined["lotfrontage"].fillna(combined["lotfrontage"].median())
    combined["electrical"] = combined["electrical"].fillna(combined["electrical"].mode()[0])
    combined["masvnrarea"] = combined["masvnrarea"].fillna(0)
    combined["garageyrblt"] = combined["garageyrblt"].fillna(0)

    cat_mode_cols = ["mszoning", "utilities", "functional", "exterior1st", "exterior2nd", "kitchenqual", "saletype"]
    for col in cat_mode_cols:
        combined[col] = combined[col].fillna(combined[col].mode()[0])

    num_zero_cols = ["bsmtfullbath", "bsmthalfbath", "bsmtfinsf1", "bsmtfinsf2", "bsmtunfsf", "totalbsmtsf", "garagecars", "garagearea"]
    for col in num_zero_cols:
        combined[col] = combined[col].fillna(0)

    combined_encoded = pd.get_dummies(combined, drop_first=True)

    X_processed = combined_encoded.iloc[:n_train, :]
    test_processed = combined_encoded.iloc[n_train:, :]

    return X_processed, test_processed, y
