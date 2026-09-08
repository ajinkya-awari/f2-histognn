import numpy as np
import pytest

from data.graph import build_knn_graph
from data.sampling import farthest_point_sampling


def test_knn_graph_has_deterministic_lowest_index_tie_breaking():
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    edge_index, edge_attr = build_knn_graph(coordinates, k=1)

    np.testing.assert_array_equal(
        edge_index,
        np.array([[0, 1, 2], [1, 0, 1]], dtype=np.int64),
    )
    assert edge_attr.shape == (3, 1)


def test_knn_graph_is_repeatable_and_clamps_k_to_n_minus_one():
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )

    first = build_knn_graph(coordinates, k=99)
    second = build_knn_graph(coordinates, k=99)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[0].shape == (2, 12)
    assert np.all(first[0][0] != first[0][1])


def test_knn_edge_features_are_one_column_and_graph_locally_normalized():
    coordinates = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])

    edge_index, edge_attr = build_knn_graph(coordinates, k=2)

    expected_distances = np.linalg.norm(
        coordinates[edge_index[0]] - coordinates[edge_index[1]],
        axis=1,
    )
    np.testing.assert_allclose(edge_attr[:, 0], expected_distances / 5.0)
    assert edge_attr.shape == (6, 1)
    assert np.all(edge_attr >= 0.0)
    assert np.all(edge_attr <= 1.0)
    assert np.isclose(edge_attr.max(), 1.0)


def test_single_node_graph_has_empty_edge_feature_matrix():
    edge_index, edge_attr = build_knn_graph(np.array([[2.0, 3.0]]), k=4)

    assert edge_index.shape == (2, 0)
    assert edge_attr.shape == (0, 1)


def test_knn_rejects_empty_nonfinite_and_zero_distance_graphs():
    with pytest.raises(ValueError, match="non-empty"):
        build_knn_graph(np.empty((0, 2)), k=1)
    with pytest.raises(ValueError, match="finite"):
        build_knn_graph(np.array([[0.0, np.nan], [1.0, 0.0]]), k=1)


def test_knn_rejects_duplicate_coordinates():
    with pytest.raises(ValueError, match="duplicate"):
        build_knn_graph(np.array([[0.0, 0.0], [0.0, 0.0]]), k=1)


def test_fps_preserves_input_order_when_under_cap():
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
    )

    selected = farthest_point_sampling(coordinates, cap=5, random_start=False)

    np.testing.assert_array_equal(selected, np.array([0, 1, 2], dtype=np.int64))


def test_fps_selects_deterministic_farthest_subset_over_cap():
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
    )

    first = farthest_point_sampling(coordinates, cap=3, random_start=False)
    second = farthest_point_sampling(coordinates, cap=3, random_start=False)

    np.testing.assert_array_equal(first, np.array([0, 4, 2], dtype=np.int64))
    np.testing.assert_array_equal(first, second)


def test_fps_requires_false_random_start_and_rejects_invalid_inputs():
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="random_start=False"):
        farthest_point_sampling(coordinates, cap=1, random_start=True)
    with pytest.raises(ValueError, match="non-empty"):
        farthest_point_sampling(np.empty((0, 2)), cap=1, random_start=False)
    with pytest.raises(ValueError, match="finite"):
        farthest_point_sampling(
            np.array([[0.0, 0.0], [np.inf, 1.0]]),
            cap=1,
            random_start=False,
        )
    with pytest.raises(ValueError, match="duplicate"):
        farthest_point_sampling(
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            cap=1,
            random_start=False,
        )
