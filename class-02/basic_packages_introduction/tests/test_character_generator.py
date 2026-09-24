"""Unit tests for character_generator."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
import character_generator


def test_add_white_gaussian_noise_is_reproducible() -> None:
    signal = np.ones((2, 2))

    noisy_signal_one = character_generator.add_white_gaussian_noise(
        signal, 10, np.random.default_rng(7)
    )
    noisy_signal_two = character_generator.add_white_gaussian_noise(
        signal, 10, np.random.default_rng(7)
    )

    np.testing.assert_array_equal(noisy_signal_one, noisy_signal_two)
    assert noisy_signal_one.shape == signal.shape


def test_generate_capital_letters() -> None:
    expected_a = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
        ]
    )
    expected_b = np.array(
        [
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    )

    np.testing.assert_array_equal(character_generator.generate_capital_a(), expected_a)
    np.testing.assert_array_equal(character_generator.generate_capital_b(), expected_b)


def test_labels_and_letter_lookup() -> None:
    assert character_generator.get_numerical_label_for_character("A") == 65
    np.testing.assert_array_equal(
        character_generator.generate_letter_array("A"),
        character_generator.generate_capital_a(),
    )
    np.testing.assert_array_equal(
        character_generator.generate_letter_array("B"),
        character_generator.generate_capital_b(),
    )


def test_convert_lists_to_dataframe() -> None:
    pixels = [np.zeros((5, 5))]
    dataframe = character_generator.convert_letters_lists_to_dataframe(
        pixels, pixels, [10.0], [65], ["A"]
    )

    expected = pd.DataFrame(
        {
            "Letter Pixels": pixels,
            "Clean Letter Pixels": pixels,
            "SNR Values (dB)": [10.0],
            "Labels": [65],
            "Strings": ["A"],
        }
    )
    pd.testing.assert_frame_equal(dataframe, expected)


def test_generate_letters_shape_contents_and_reproducibility() -> None:
    first = character_generator.generate_letters(["A", "B"], [0.0, 10.0], 2, 3)
    second = character_generator.generate_letters(["A", "B"], [0.0, 10.0], 2, 3)

    assert first.shape == (8, 5)
    assert first["Strings"].tolist() == ["A", "A", "B", "B"] * 2
    assert first["Labels"].tolist() == [65, 65, 66, 66] * 2
    assert first["SNR Values (dB)"].tolist() == [0.0, 10.0, 0.0, 10.0] * 2
    for clean_pixels, letter in zip(first["Clean Letter Pixels"], first["Strings"]):
        np.testing.assert_array_equal(
            clean_pixels, character_generator.generate_letter_array(letter)
        )
    pd.testing.assert_frame_equal(first, second)
