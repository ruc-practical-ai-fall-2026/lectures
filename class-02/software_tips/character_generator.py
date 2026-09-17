"""Generate a datasets of letters for machine learning experiments."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def add_white_gaussian_noise(
    signal_array: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Add white Gaussian noise to the signal array.

    This function is similar to MATLAB's add white Gaussian noise function.

    Args:
        signal_array: The signal for which noise is to be added.
        snr_db: The desired signal to noise ratio in 10*log10 dB.
        rng: The random number generator to use.

    Returns:
        The noisy signal as an array of the same shape as the original.
    """
    snr = 10 ** (snr_db / 10)
    signal_power = np.average(np.abs(signal_array) ** 2)
    noise_power = signal_power / snr
    mu_noise_mean = 0
    sigma_noise_std_dev = np.sqrt(noise_power)
    noise = rng.normal(mu_noise_mean, sigma_noise_std_dev, signal_array.shape)
    signal_with_noise = signal_array + noise
    return signal_with_noise


def generate_capital_a() -> np.ndarray:
    """Generate a capital letter A as a numpy array.

    Returns:
        A capital letter A as a 5x5 numpy array.
    """
    capital_letter_a = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
        ]
    )
    return capital_letter_a


def generate_capital_b() -> np.ndarray:
    """Generate a capital letter B as a numpy array.

    Returns:
        A capital letter B as a 5x5 numpy array.
    """
    capital_letter_b = np.array(
        [
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    return capital_letter_b


letter_generator_dictionary = {
    "A": generate_capital_a,
    "B": generate_capital_b,
}


def get_numerical_label_for_character(letter: str) -> int:
    """Get a numerical label corresponding to a character.

    Args:
        letter: The letter to generate as a string.

    Returns:
        A number corresponding to a character's ASCII value to use as a label
        in machine learning problems.
    """
    integer_label = ord(letter)
    return integer_label


def generate_letter_array(letter: str) -> np.ndarray:
    """Generate the numpy array for the given letter.

    Args:
        letter: The letter to generate as a string.

    Returns:
        The array corresponding to the letter.
    """
    letter_generator = letter_generator_dictionary[letter]
    letter_array = letter_generator()
    return letter_array


def convert_letters_lists_to_dataframe(
    letters: List[np.ndarray],
    clean_letters: List[np.ndarray],
    snr_values: List[float],
    labels: List[int],
    strings: List[str],
) -> pd.DataFrame:
    """Convert list of information about letters into a DataFrame.

    Args:
        letters: The list of letters as arrays of pixels.
        clean_letters: The list of letters as arrays of pixels without noise.
        snr_values: The list of SNR values for added noise.
        labels: The list of labels as integers.
        strings: The list of human readable strings for each letter.

    Returns:
        DataFrame of letter pixels, labels, and human readable strings.
    """
    df_dict = {
        "Letter Pixels": letters,
        "Clean Letter Pixels": clean_letters,
        "SNR Values (dB)": snr_values,
        "Labels": labels,
        "Strings": strings,
    }
    letters_df = pd.DataFrame(df_dict)
    return letters_df


def generate_letters(
    letters_list: List[str],
    snr_db_list: List[float],
    number_of_instances_per_letter: int = 50,
    rng_seed=1,
) -> pd.DataFrame:
    """Generate a DataFrame of letters for machine learning experiments.

    The number of rows in the DataFrame returned will be len(letters_list) x
    len(snr_list) x number_of_instances_per_letter.

    Args:
        letters_list: List of letters to generate.
        snr_list: List of signal to noise ratios (SNRs) in 10*log10 dB.
        number_of_instances_per_letter: Number of repeats for each letter for
            each SNR value provided.
        rng_seed: Repeatability seed for the random number generator.
    """
    letters = []
    clean_letters = []
    snr_values = []
    labels = []
    strings = []

    rng = np.random.default_rng(rng_seed)

    for _ in range(number_of_instances_per_letter):
        for letter in letters_list:
            for snr in snr_db_list:
                letter_array = generate_letter_array(letter)
                letter_label = get_numerical_label_for_character(letter)
                noisy_letter_array = add_white_gaussian_noise(
                    letter_array, snr, rng
                )

                letters.append(noisy_letter_array)
                clean_letters.append(letter_array)
                snr_values.append(snr)
                labels.append(letter_label)
                strings.append(letter)

    letters_dataframe = convert_letters_lists_to_dataframe(
        letters, clean_letters, snr_values, labels, strings
    )

    return letters_dataframe


def visualize_letters_dataframe(
    letters_df: pd.DataFrame,
    snr_values: List[float],
    letters: List[str],
    snr_atol: float = 1e-3,
) -> None:
    """Visualize a DataFrame of letters.

    Args:
        letters_df: The DataFrame to be visualized.
        snr_values: The values of SNR of interest for visualization.
        letters: The letters of interest for visualization.
        snr_atol: The absolute tolerance for locating images with the selected
            SNRs in 10*log10 dB.

    Returns:
        None
    """
    fig, ax = plt.subplots(len(snr_values), len(letters), figsize=(4, 10))
    for snr_index, snr in enumerate(snr_values):
        snr_mask = np.isclose(
            letters_df["SNR Values (dB)"], snr, atol=snr_atol, rtol=0.0
        )
        for letter_index, letter in enumerate(letters):
            letter_mask = letters_df["Strings"] == letter
            mask = letter_mask & snr_mask
            qualifying_images_series = letters_df[mask]["Letter Pixels"]
            first_qualifying_image = qualifying_images_series.iloc[0]
            ax[snr_index][letter_index].imshow(
                first_qualifying_image, cmap="Grays"
            )
            ax[snr_index][letter_index].axis("off")
            ax[snr_index][letter_index].set_title(f"{letter}\nSNR = {snr} dB")
    fig.tight_layout()
