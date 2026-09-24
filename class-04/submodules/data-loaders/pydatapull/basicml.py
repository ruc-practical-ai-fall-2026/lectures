"""Basic data generators for creating simple multidimensional datasets."""

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch import float32 as TorchFloat32
from torch import tensor as TorchTensor


def _generate_random_columns(
    mu_vector: np.ndarray,
    sigma_vector: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Generate a random matrix.

    The elements of each column are sampled from a different normal
    distribution. The parameters of the different normal distributions are
    specified by the mu_vector (means) and the sigma_vector (standard
    deviations).

    Each row of N rows is a sample and each column is a dimension of an
    M-dimensional multidimensional distribution.

    mu_vector and sigma_vector must be the same length.

    Args:
        mu_vector: A 1xM vector of means of a multidimensional normal
            distribution.
        sigma_vector: A 1xM vector of standard deviations of a
            multidimensional normal distribution.
        n_samples: The number, N, of samples (rows) in the matrix.

    Returns:
        The random matrix.
    """
    distribution_list: List[np.ndarray] = []
    for mu, sigma in zip(mu_vector, sigma_vector):
        distribution: np.ndarray = np.random.normal(mu, sigma, n_samples)
        distribution_list.append(distribution)
    random_columns: np.ndarray = np.column_stack(distribution_list)
    return random_columns


def _generate_list_of_distributions(
    mu_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    n_samples_per_distribution: int,
) -> List[np.ndarray]:
    """Generate a list of different random matrices.

    Each random matrix is of shape NxM where each row is one of N samples and
    the M columns have elements sampled from M different normal distributions.

    The parameters of the different normal distributions are specified by the
    mu_matrix (means) and the sigma_matrix (standard deviations). Both
    mu_matrix and sigma_matrix are of shape LxM where L is the number of
    distributions and M is the dimensionality of each distribution, i.e., the
    rows of the mu_matrix and sigma_matrix correspond to different
    distributions. Each column is a dimension of an M-dimensional distribution.

    mu_matrix and sigma_matrix must be the same shape.

    Args:
        mu_matrix: an LxM matrix of means where L is the number of
            distributions the matrix represents and M is the number of
            dimensions of the distributions. For example, a 10x3 matrix
            represents the means of 10 3-dimensional distributions.
        sigma_matrix: an LxM matrix of standard deviations where L is the
            number of distributions the matrix represents and M is the number
            of dimensions of the distributions. For example, a 10x3 matrix
            represents the standard deviations of 10 3-dimensional
            distributions.
        n_samples_per_distribution: specifies the number of samples, N, to
            generate per each distribution.

    Returns:
        A list of NxM matrices where the individual column vectors which
            comprise each matrix are each drawn from one of the distributions
            specified by mu_matrix and sigma_matrix, and there is one matrix
            (list element) for each row (distribution) defined in mu_matrix
            and sigma_matrix.
    """
    distributions_list: List[np.ndarray] = []
    for mu_vector, sigma_vector in zip(mu_matrix, sigma_matrix):
        distribution: np.ndarray = _generate_random_columns(
            mu_vector, sigma_vector, n_samples=n_samples_per_distribution
        )
        distributions_list.append(distribution)
    return distributions_list


def _merge_distributions(
    choice_probabilities: np.ndarray,
    distributions_list: List[np.ndarray],
) -> np.ndarray:
    """Probabilistically merge a list of distributions.

    Given a list of distributions (same-size random matrices where the columns
    correspond to dimensions in an M-dimensional space) and a vector of
    probabilities, select which rows of each random matrix in the list of
    distributions to keep based on the probabilities, in order to merge them
    into a final random matrix of the same size as the original random
    matrices.

    The length of choice_probabilities must equal the length of
    distributions_list.

    Each array in distributions_list must be of the same shape.

    Args:
        choice_probabilities: L element vector where L is the number of total
            probability distributions to draw from (here, L must be equal to
            the number of elements in distribution_list). Each element in the
            vector represents the probability of drawing from the distribution
            whose index in distributions_list corresponds to the index of the
            element in choice_probabilities. For example, choice_probabilities
            = np.array([0.4, 0.6]) will draw from the distribution in
            distributions_list[0] 40% of the time and draw from the
            distribution in distributions_list[1] 60% of the time.
        distributions_list: L element list of NxM matrices, each representing
            N samples drawn from an M-dimensional normal distribution.

    Returns:
        The merged array of size NxM.
    """
    n_samples_per_distribution: int = distributions_list[0].shape[0]
    n_choices: int = len(distributions_list)
    choices: NDArray[np.int32] = np.arange(0, n_choices)
    choice_vector: np.ndarray = np.random.choice(
        choices, size=n_samples_per_distribution, p=choice_probabilities
    )
    selected_distributions: List[np.ndarray] = []
    for distribution_index in range(n_choices):
        distribution: np.ndarray = distributions_list[distribution_index]
        mask: np.ndarray = choice_vector == distribution_index
        selected_distribution: np.ndarray = distribution[mask]
        selected_distributions.append(selected_distribution)
    final_distribution: np.ndarray = np.vstack(selected_distributions)
    return final_distribution


def _generate_class_distribution(
    mu_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    choice_probabilities: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Generate the distribution of a single class.

    The distribution for a class is a weighted union of the multi-dimensional
    distributions defined by mu_matrix and sigma_matrix, where each is an LxM
    matrix, L is the number of distributions, and M is the number of dimensions
    in each distribution. All distributions are normal. The probabilities in
    choice_probabilities, a vector of length L will determine how frequently
    each distribution is drawn from to create the overall union.

    An NxM matrix is returned with N samples drawn from the M-dimensional
    distributions with the draw probabilities specified by
    choice_probabilities.

    mu_matrix and sigma_matrix must have the same shape.

    choice_probabilities must have length equal to the number of rows in
    sigma_matrix and mu_matrix.

    Args:
        mu_matrix: LxM matrix defining the means of L M-dimensional normal
            distributions.
        sigma_matrix: LxM matrix defining the standard deviations of L
            M-dimensional normal distributions.
        choice_probabilities: Vector of length L defining the probability of
            sampling from each distribution.
        n_samples: The number of samples to draw.

    Returns:
        An NxM matrix with N samples drawn from the M-dimensional distributions
            with the specified draw probabilities.
    """
    distributions_list: List[np.ndarray] = _generate_list_of_distributions(
        mu_matrix, sigma_matrix, n_samples
    )
    final_distribution: np.ndarray = _merge_distributions(
        choice_probabilities, distributions_list
    )
    return final_distribution


def generate_multiclass_multidistribution_dataset(
    n_samples_per_class: int,
    mu_matrix_list: List[np.ndarray],
    sigma_matrix_list: List[np.ndarray],
    choice_probabilities_list: List[np.ndarray],
    labels_list: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a multi-class, multi-distribution, stochastic dataset.

    This function generates a multi-class, multi-distribution stochastic
    dataset to use for benchmarking machine learning algorithms on basic,
    controlled datasets.

    The distribution for a class is a weighted union of the multi-dimensional
    distributions defined by mu_matrix and sigma_matrix, where each is an LxM
    matrix, L is the number of distributions, and M is the number of dimensions
    in each distribution. All distributions are normal. The probabilities in
    choice_probabilities, a vector of length L will determine how frequently
    each distribution is drawn from to create the overall union.

    An NxM matrix is returned with N samples per class drawn from the
    M-dimensional distributions with the draw probabilities specified by
    choice_probabilities. For example, if 3 classes are specified with a 4
    dimensional feature vector, and 100 samples per class, then the feature
    matrix returned will be of shape 300 x 4. The label vector returned will
    be of length 300.

    **NOTE**: No bounds checking is performed for the user! Look before you
    leap if unsure if input lists and arrays are of compatible size. It is the
    responsibility of the user code to ensure that the following constraints
    are met.

        * Corresponding elements of mu_matrix_list and sigma_matrix_list must
         have the same shape.

        * Elements of choice_probabilities_list must have length equal to the
          number of rows in sigma_matrix and mu_matrix.

        * mu_matrix_list, sigma_matrix_list, choice_probabilities_list, and
          labels must all have the same number of elements, corresponding to
          the number of classes, K.

        * All distributions defined must have the same number of dimensions, M
          to ultimately be merged to an M-dimensional feature vector.

    Args:
        n_samples_per_class: The number of samples to generate for each class.

        mu_matrix_list: K-element list of L_k x M matrices defining the means
            of L_k M_k-dimensional normal distributions, where k is in [1, K]
            such that each element defines a distribution which is the union of
            L_k distributions.

        sigma_matrix_list: K-element list of L_k x M matrices defining the
            standard deviations of L_k M_k-dimensional normal distributions,
            where k is in [1, K] such that each element defines a distribution
            which is the union of L_k distributions.

        choice_probabilities_list: K-element list of vectors, each of length
            L_k defining the probability of sampling from each distribution,
            where k is in [1, K] such that each element of the list defines the
            weights of a weighted draw from each of the L_k distributions.

        label_list: K-element list of labels where each label is a string.

    Returns:
        An NxM matrix with N samples drawn from the M-dimensional distributions
            with the specified draw probabilities and an N-element vector of
            class labels.
    """
    numerical_labels = np.arange(0, len(labels_list))
    class_distributions = []
    label_vectors = []
    for (
        mu_matrix,
        sigma_matrix,
        choice_probabilities_vector,
        numerical_label,
    ) in zip(
        mu_matrix_list,
        sigma_matrix_list,
        choice_probabilities_list,
        numerical_labels,
    ):
        class_distribution = _generate_class_distribution(
            mu_matrix,
            sigma_matrix,
            choice_probabilities_vector,
            n_samples_per_class,
        )
        label_vector = np.ones(n_samples_per_class) * numerical_label
        class_distributions.append(class_distribution)
        label_vectors.append(label_vector)

    x_features = np.vstack(class_distributions)
    y_labels = np.hstack(label_vectors)
    return x_features, y_labels


def generate_multiclass_multidistribution_dataset_torch(
    n_samples_per_class: int,
    mu_matrix_list: List[np.ndarray],
    sigma_matrix_list: List[np.ndarray],
    choice_probabilities_list: List[np.ndarray],
    labels_list: List[str],
) -> Tuple[Tensor, Tensor]:
    """Generate a multi-class, multi-distribution, stochastic dataset.

    This function generates a multi-class, multi-distribution stochastic
    dataset to use for benchmarking machine learning algorithms on basic,
    controlled datasets.

    The distribution for a class is a weighted union of the multi-dimensional
    distributions defined by mu_matrix and sigma_matrix, where each is an LxM
    matrix, L is the number of distributions, and M is the number of dimensions
    in each distribution. All distributions are normal. The probabilities in
    choice_probabilities, a vector of length L will determine how frequently
    each distribution is drawn from to create the overall union.

    An NxM matrix is returned with N samples per class drawn from the
    M-dimensional distributions with the draw probabilities specified by
    choice_probabilities. For example, if 3 classes are specified with a 4
    dimensional feature vector, and 100 samples per class, then the feature
    matrix returned will be of shape 300 x 4. The label vector returned will
    be of length 300.

    **NOTE**: No bounds checking is performed for the user! Look before you
    leap if unsure if input lists and arrays are of compatible size. It is the
    responsibility of the user code to ensure that the following constraints
    are met.

        * Corresponding elements of mu_matrix_list and sigma_matrix_list must
         have the same shape.

        * Elements of choice_probabilities_list must have length equal to the
          number of rows in sigma_matrix and mu_matrix.

        * mu_matrix_list, sigma_matrix_list, choice_probabilities_list, and
          labels must all have the same number of elements, corresponding to
          the number of classes, K.

        * All distributions defined must have the same number of dimensions, M
          to ultimately be merged to an M-dimensional feature vector.

    Args:
        n_samples_per_class: The number of samples to generate for each class.

        mu_matrix_list: K-element list of L_k x M matrices defining the means
            of L_k M_k-dimensional normal distributions, where k is in [1, K]
            such that each element defines a distribution which is the union of
            L_k distributions.

        sigma_matrix_list: K-element list of L_k x M matrices defining the
            standard deviations of L_k M_k-dimensional normal distributions,
            where k is in [1, K] such that each element defines a distribution
            which is the union of L_k distributions.

        choice_probabilities_list: K-element list of vectors, each of length
            L_k defining the probability of sampling from each distribution,
            where k is in [1, K] such that each element of the list defines the
            weights of a weighted draw from each of the L_k distributions.

        label_list: K-element list of labels where each label is a string.

    Returns:
        An NxM matrix with N samples drawn from the M-dimensional distributions
            with the specified draw probabilities and an N-element vector of
            class labels.
    """
    x_features, y_labels = generate_multiclass_multidistribution_dataset(
        n_samples_per_class,
        mu_matrix_list,
        sigma_matrix_list,
        choice_probabilities_list,
        labels_list,
    )
    # Convert data to a torch tensor; reshape the labels into an Nx1 element
    # matrix since this is what pytorch expects.
    x_features_tensor = TorchTensor(x_features, dtype=TorchFloat32)
    y_labels_tensor = TorchTensor(y_labels.reshape(-1, 1), dtype=TorchFloat32)
    return x_features_tensor, y_labels_tensor


def generate_xor_dataset(
    n_samples_per_class: int, sigma: float = 0
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate a noisy XOR dataset.

    Args:
        n_samples_per_class: the number of samples to generate per class (the
            total number of samples will be twice the number specified).

        sigma: the standard deviation of the noise applied to each dimension.

    Returns:
        Tuple with the features array as an NxM array where N is the total
        number of samples and M is the number of features, the label vector
        as an N-element vector, and a list of sematic labels as strings.

    Example:
        >>> x_features, y_labels, labels = generate_xor_dataset(100, sigma=0.1)
    """
    mu_a: np.ndarray = np.array([[0, 0], [1, 1]])
    mu_b: np.ndarray = np.array([[0, 1], [1, 0]])
    mu_matrix_list: List[np.ndarray] = [mu_a, mu_b]
    sigma_matrix_list: List[np.ndarray] = [np.full((2, 2), sigma)] * 2
    choice_probabilities_list: List[np.ndarray] = [np.array([0.5, 0.5])] * 2
    labels = ["X1 = X2", "X1 != X2"]
    x_features, y_labels = generate_multiclass_multidistribution_dataset(
        n_samples_per_class,
        mu_matrix_list,
        sigma_matrix_list,
        choice_probabilities_list,
        labels,
    )
    return x_features, y_labels, labels


def generate_xor_dataset_torch(
    n_samples_per_class: int, sigma: float = 0
) -> Tuple[Tensor, Tensor, List[str]]:
    """Generate a noisy XOR dataset, ready to use with torch.

    Args:
        n_samples_per_class: the number of samples to generate per class (the
            total number of samples will be twice the number specified).

        sigma: the standard deviation of the noise applied to each dimension.

    Returns:
        Tuple with the features array as an NxM torch tensor where N is the
        total number of samples and M is the number of features, the label
        vector as an Nx1 torch tensor, and a list of sematic labels as strings.

    Example:
        >>> x_features, y_labels, labels = generate_xor_dataset(100, sigma=0.1)
    """
    x_features, y_labels, labels = generate_xor_dataset(
        n_samples_per_class, sigma
    )
    # Convert data to a torch tensor; reshape the labels into an Nx1 element
    # matrix since this is what pytorch expects.
    x_features_tensor = TorchTensor(x_features, dtype=TorchFloat32)
    y_labels_tensor = TorchTensor(y_labels.reshape(-1, 1), dtype=TorchFloat32)
    return x_features_tensor, y_labels_tensor, labels


def generate_half_moon_dataset(
    n_samples_per_class: int,
    n_clusters_per_class: int,
    radius_r: float,
    separation_offset_d: float,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate a noisy half-moon dataset.

    The half-moon classes are defined by the following equations:

    The first class is defined by

        f_xa(t) = -r / 2 + r * cos(t_a),
        f_ya(t) = d + r * sin(t_a),

    where

        0 <= t_a <= pi.

    The second class is defined by

        f_xb(t) = r / 2 + r * cos(t_b)
        f_yb(t) = -d + r * np.sin(t_b)

    where

        pi <= t_b <= 2*pi.

    Args:
        n_samples_per_class: the number of samples to generate per class (the
            total number of samples will be twice the number specified).

        n_clusters_per_class: the number of clusters to generate per class. Set
            to a large number to make the classes appear as lines in feature
            space.

        radius_r: the radius of the half-moons. Set to larger values to
            generate larger arcs.

        separation_offset_d: defines how separated the half-moons are in the
            y-dimension. Positive values increase separation and negative
            values decrease separation.

        sigma: the standard deviation of the noise applied to each dimension.

    Returns:
        Tuple with the features array as an NxM array where N is the total
        number of samples and M is the number of features, the label vector
        as an N-element vector, and a list of sematic labels as strings.

    Examples:

    The following code generates a smooth and tight half-moon example.

        >>> x_features, y_labels, _ = generate_half_moon_dataset(
        >>>     n_samples_per_class=1000,
        >>>     n_clusters_per_class=100,
        >>>     radius_r=1,
        >>>     separation_offset_d=-0.2,
        >>>     sigma=0.1,
        >>> )

    The following code generates a half-moon example where the clusters
    which comprise each half-moon are apparent.

        >>> x_features, y_labels, _ = generate_half_moon_dataset(
        >>>     n_samples_per_class=1000,
        >>>     n_clusters_per_class=10,
        >>>     radius_r=10,
        >>>     separation_offset_d=-2,
        >>>     sigma=0.2,
        >>> )
    """
    t_param_class_a: np.ndarray = np.linspace(0, np.pi, n_clusters_per_class)
    t_param_class_b: np.ndarray = np.linspace(
        np.pi, 2 * np.pi, n_clusters_per_class
    )

    mu_x_class_a: np.ndarray = -radius_r / 2 + radius_r * np.cos(
        t_param_class_a
    )
    mu_y_class_a: np.ndarray = separation_offset_d + radius_r * np.sin(
        t_param_class_a
    )

    mu_x_class_b: np.ndarray = radius_r / 2 + radius_r * np.cos(
        t_param_class_b
    )
    mu_y_class_b: np.ndarray = -separation_offset_d + radius_r * np.sin(
        t_param_class_b
    )

    mu_class_a: np.ndarray = np.column_stack((mu_x_class_a, mu_y_class_a))
    mu_class_b: np.ndarray = np.column_stack((mu_x_class_b, mu_y_class_b))

    sigma_matrix: np.ndarray = np.full(mu_class_a.shape, sigma)

    mu_matrix_list: List[np.ndarray] = [mu_class_a, mu_class_b]
    sigma_matrix_list: List[np.ndarray] = [sigma_matrix] * 2
    choice_probabilities_list: List[np.ndarray] = [
        np.ones(n_clusters_per_class) / n_clusters_per_class
    ] * 2
    labels: List[str] = ["A", "B"]
    x_features, y_labels = generate_multiclass_multidistribution_dataset(
        n_samples_per_class,
        mu_matrix_list,
        sigma_matrix_list,
        choice_probabilities_list,
        labels,
    )
    return x_features, y_labels, labels


def generate_half_moon_dataset_torch(
    n_samples_per_class: int,
    n_clusters_per_class: int,
    radius_r: float,
    separation_offset_d: float,
    sigma: float,
) -> Tuple[Tensor, Tensor, List[str]]:
    """Generate a noisy half-moon dataset.

    The half-moon classes are defined by the following equations:

    The first class is defined by

        f_xa(t) = -r / 2 + r * cos(t_a),
        f_ya(t) = d + r * sin(t_a),

    where

        0 <= t_a <= pi.

    The second class is defined by

        f_xb(t) = r / 2 + r * cos(t_b)
        f_yb(t) = -d + r * np.sin(t_b)

    where

        pi <= t_b <= 2*pi.

    Args:
        n_samples_per_class: the number of samples to generate per class (the
            total number of samples will be twice the number specified).

        n_clusters_per_class: the number of clusters to generate per class. Set
            to a large number to make the classes appear as lines in feature
            space.

        radius_r: the radius of the half-moons. Set to larger values to
            generate larger arcs.

        separation_offset_d: defines how separated the half-moons are in the
            y-dimension. Positive values increase separation and negative
            values decrease separation.

        sigma: the standard deviation of the noise applied to each dimension.

    Returns:
        Tuple with the features array as an NxM array where N is the total
        number of samples and M is the number of features, the label vector
        as an N-element vector, and a list of sematic labels as strings.

    Examples:

    The following code generates a smooth and tight half-moon example.

        >>> x_features, y_labels, _ = generate_half_moon_dataset(
        >>>     n_samples_per_class=1000,
        >>>     n_clusters_per_class=100,
        >>>     radius_r=1,
        >>>     separation_offset_d=-0.2,
        >>>     sigma=0.1,
        >>> )

    The following code generates a half-moon example where the clusters
    which comprise each half-moon are apparent.

        >>> x_features, y_labels, _ = generate_half_moon_dataset(
        >>>     n_samples_per_class=1000,
        >>>     n_clusters_per_class=10,
        >>>     radius_r=10,
        >>>     separation_offset_d=-2,
        >>>     sigma=0.2,
        >>> )
    """
    x_features, y_labels, labels = generate_half_moon_dataset(
        n_samples_per_class,
        n_clusters_per_class,
        radius_r,
        separation_offset_d,
        sigma,
    )
    # Convert data to a torch tensor; reshape the labels into an Nx1 element
    # matrix since this is what pytorch expects.
    x_features_tensor = TorchTensor(x_features, dtype=TorchFloat32)
    y_labels_tensor = TorchTensor(y_labels.reshape(-1, 1), dtype=TorchFloat32)
    return x_features_tensor, y_labels_tensor, labels
