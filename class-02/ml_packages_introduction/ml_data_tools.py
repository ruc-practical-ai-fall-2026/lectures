import numpy as np
import pandas as pd


def make_train_test_split_dataframes(
    input_df, features_list, target_column, test_fraction=0.2
):
    """Makes train and test dataframes and records the split in the original."""
    number_of_rows = len(input_df)
    number_of_ones = int(number_of_rows * test_fraction)
    test_rows = np.zeros(number_of_rows)
    indices_to_set_to_1 = np.random.choice(
        number_of_rows, number_of_ones, replace=False
    )
    test_rows[indices_to_set_to_1] = 1
    input_df["test flag"] = test_rows.tolist()

    train_df = input_df[input_df["test flag"] == 0]
    test_df = input_df[input_df["test flag"] == 1]

    x_train = train_df[features_list].to_numpy()
    y_train = train_df[target_column].to_numpy().reshape(-1, 1).ravel()

    x_test = test_df[features_list].to_numpy()
    y_test = test_df[target_column].to_numpy().reshape(-1, 1).ravel()

    return x_train, x_test, y_train, y_test
