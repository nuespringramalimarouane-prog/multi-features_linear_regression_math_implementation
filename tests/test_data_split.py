import numpy as np

from main import normalize_features, split_data


def test_split_is_reproducible_and_uses_disjoint_indices():
    X = np.arange(50, dtype=float).reshape(25, 2)
    y = np.arange(25, dtype=float)

    first = split_data(X, y, test_size=0.2, random_state=17)
    second = split_data(X, y, test_size=0.2, random_state=17)

    X_train, X_test, y_train, y_test = first
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert len(X_test) == 5
    assert len(X_train) == 20
    assert set(y_train).isdisjoint(set(y_test))


def test_normalization_uses_training_statistics_for_test_data():
    X_train = np.array([[1.0, 10.0], [3.0, 30.0]])
    X_test = np.array([[5.0, 50.0]])

    train_norm, test_norm = normalize_features(X_train, X_test)

    assert np.allclose(train_norm.mean(axis=0), [0.0, 0.0])
    assert np.allclose(test_norm, [[1.0, 1.0]])
