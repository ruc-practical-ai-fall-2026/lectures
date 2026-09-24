"""Unit tests for data_generators."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import data_generators


def test_generate_random_columns_uses_column_parameters() -> None:
    columns = data_generators._generate_random_columns(
        np.array([1.0, 2.0]), np.zeros(2), n_samples=3
    )

    np.testing.assert_array_equal(columns, np.array([[1.0, 2.0]] * 3))


def test_generate_list_of_distributions_returns_one_matrix_per_row() -> None:
    distributions = data_generators._generate_list_of_distributions(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.zeros((2, 2)),
        n_samples_per_distribution=2,
    )

    assert len(distributions) == 2
    np.testing.assert_array_equal(distributions[0], np.array([[1.0, 2.0]] * 2))
    np.testing.assert_array_equal(distributions[1], np.array([[3.0, 4.0]] * 2))


def test_merge_distributions_selects_rows_by_choice(monkeypatch) -> None:
    distributions = [
        np.array([[10.0], [11.0], [12.0]]),
        np.array([[20.0], [21.0], [22.0]]),
    ]

    monkeypatch.setattr(
        data_generators.np.random,
        "choice",
        lambda choices, size, p: np.array([1, 0, 1]),
    )

    merged = data_generators._merge_distributions(
        np.array([0.5, 0.5]), distributions
    )

    np.testing.assert_array_equal(merged, np.array([[11.0], [20.0], [22.0]]))


def test_generate_class_distribution_has_requested_shape_and_values() -> None:
    distribution = data_generators._generate_class_distribution(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.zeros((2, 2)),
        np.array([1.0, 0.0]),
        n_samples=4,
    )

    assert distribution.shape == (4, 2)
    np.testing.assert_array_equal(distribution, np.array([[1.0, 2.0]] * 4))


def test_generate_multiclass_dataset_returns_features_and_numeric_labels() -> None:
    features, labels = data_generators.generate_multiclass_multidistribution_dataset(
        2,
        [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])],
        [np.zeros((1, 2)), np.zeros((1, 2))],
        [np.array([1.0]), np.array([1.0])],
        ["first", "second"],
    )

    np.testing.assert_array_equal(
        features, np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]])
    )
    np.testing.assert_array_equal(labels, np.array([0.0, 0.0, 1.0, 1.0]))


def test_generate_xor_dataset_has_expected_labels_and_points() -> None:
    features, labels, semantic_labels = data_generators.generate_xor_dataset(3)

    assert features.shape == (6, 2)
    np.testing.assert_array_equal(labels, np.array([0.0] * 3 + [1.0] * 3))
    assert semantic_labels == ["X1 = X2", "X1 != X2"]
    assert {tuple(point) for point in features[:3]} <= {(0.0, 0.0), (1.0, 1.0)}
    assert {tuple(point) for point in features[3:]} <= {(0.0, 1.0), (1.0, 0.0)}


def test_generate_half_moon_dataset_has_expected_shapes_and_labels() -> None:
    features, labels, semantic_labels = data_generators.generate_half_moon_dataset(
        n_samples_per_class=4,
        n_clusters_per_class=2,
        radius_r=1.0,
        separation_offset_d=0.0,
        sigma=0.0,
    )

    assert features.shape == (8, 2)
    np.testing.assert_array_equal(labels, np.array([0.0] * 4 + [1.0] * 4))
    assert semantic_labels == ["A", "B"]
