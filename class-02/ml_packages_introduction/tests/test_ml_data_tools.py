"""Unit tests for ml_data_tools."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
import ml_data_tools


def test_make_train_test_split_records_selected_rows_and_returns_arrays(
    monkeypatch,
) -> None:
    input_df = pd.DataFrame(
        {
            "feature one": [10, 20, 30, 40, 50],
            "feature two": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        }
    )
    monkeypatch.setattr(
        ml_data_tools.np.random,
        "choice",
        lambda number_of_rows, number_of_ones, replace: np.array([1, 3]),
    )

    x_train, x_test, y_train, y_test = (
        ml_data_tools.make_train_test_split_dataframes(
            input_df,
            features_list=["feature one", "feature two"],
            target_column="target",
            test_fraction=0.4,
        )
    )

    np.testing.assert_array_equal(
        x_train, np.array([[10, 1], [30, 3], [50, 5]])
    )
    np.testing.assert_array_equal(x_test, np.array([[20, 2], [40, 4]]))
    np.testing.assert_array_equal(y_train, np.array([0, 0, 0]))
    np.testing.assert_array_equal(y_test, np.array([1, 1]))
    np.testing.assert_array_equal(
        input_df["test flag"], np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    )


def test_make_train_test_split_with_no_test_rows_returns_empty_test_arrays() -> (
    None
):
    input_df = pd.DataFrame({"feature": [1, 2], "target": [3, 4]})

    x_train, x_test, y_train, y_test = (
        ml_data_tools.make_train_test_split_dataframes(
            input_df,
            features_list=["feature"],
            target_column="target",
            test_fraction=0.0,
        )
    )

    np.testing.assert_array_equal(x_train, np.array([[1], [2]]))
    assert x_test.shape == (0, 1)
    np.testing.assert_array_equal(y_train, np.array([3, 4]))
    assert y_test.shape == (0,)
    np.testing.assert_array_equal(input_df["test flag"], np.zeros(2))
