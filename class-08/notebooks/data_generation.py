import numpy as np


def generate_class_distribution(means, standard_deviations, n_samples):
    features = []
    for mean, standard_deviation in zip(means, standard_deviations):
        feature = np.random.normal(mean, standard_deviation, n_samples)
        features.append(feature)
    features_x = np.vstack(features).T
    return features_x


def generate_dataset(
    class1_means,
    class1_standard_deviations,
    class2_means,
    class2_standard_deviations,
    n_samples_per_class,
):
    x1 = generate_class_distribution(
        class1_means, class1_standard_deviations, n_samples_per_class
    )
    x2 = generate_class_distribution(
        class2_means, class2_standard_deviations, n_samples_per_class
    )
    x_features = np.vstack((x1, x2))

    ya = np.zeros(n_samples_per_class)
    yb = np.ones(n_samples_per_class)
    y_targets = np.hstack((ya, yb))

    return x_features, y_targets


def generate_challenge_dataset(n_useless_features=1, n_challenge_rows=400):
    n_useful_features = 15

    base_mu_useless = 0
    base_mu_useful = 0

    delta_mu_useless = 0
    delta_mu_useful = 0.8

    sigma_useless = 1
    sigma_useful = 1

    # Build a list of the means of each feature of the form such that the first
    # entries are the means of each useless feature and the second entries are
    # the means of each useful feature.
    mu1 = [
        base_mu_useless,
    ] * n_useless_features + [
        base_mu_useful,
    ] * n_useful_features
    mu2 = [
        base_mu_useless + delta_mu_useless,
    ] * n_useless_features + [
        base_mu_useful + delta_mu_useful,
    ] * n_useful_features

    # Build a list of the standard deviations of each feature of the form such
    # that the first entries are the means of each useless feature and the
    # second entries are the means of each useful feature.
    sigma1 = [
        sigma_useless,
    ] * n_useless_features + [
        sigma_useful,
    ] * n_useful_features
    sigma2 = [
        sigma_useless,
    ] * n_useless_features + [
        sigma_useful,
    ] * n_useful_features

    x, y = generate_dataset(
        class1_means=mu1,
        class1_standard_deviations=sigma1,
        class2_means=mu2,
        class2_standard_deviations=sigma2,
        n_samples_per_class=1000,
    )

    # Select n rows to make useful features inseparable
    challenge_indices = np.random.choice(
        x.shape[0], size=n_challenge_rows, replace=False
    )
    x[challenge_indices, n_useless_features:] = 2
    return x, y
