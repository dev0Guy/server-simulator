import random

import numpy as np
import hypothesis.strategies as st
import pytest
from hypothesis import given, assume, HealthCheck, settings
import numpy.typing as npt
import typing as tp
from src.utils import array_operations


kernel_strategy = st.tuples(
    st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5)
)

array_strategy = st.builds(
    lambda m_x, m_y, n_res, n_ticks: np.random.rand(m_x, m_y, n_res, n_ticks).astype(
        np.float64
    ),
    m_x=st.integers(min_value=1, max_value=50),
    m_y=st.integers(min_value=1, max_value=50),
    n_res=st.integers(min_value=1, max_value=10),
    n_ticks=st.integers(min_value=1, max_value=10),
)
reduction_operation_strategy = st.sampled_from([np.mean, np.max, np.min, np.average])


def block_operation(array, kernel, operation):
    k_x, k_y = kernel
    m_x, m_y = array.shape[:2]
    n_x, n_y = m_x // k_x, m_y // k_y
    reshaped = array[: n_x * k_x, : n_y * k_y].reshape(
        n_x, k_x, n_y, k_y, *array.shape[2:]
    )
    return operation(reshaped, axis=(1, 3))


@given(
    arr=array_strategy, kernel=kernel_strategy, operation=reduction_operation_strategy
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_pool_independent(
    arr: npt.NDArray[tp.Any], kernel: tp.Tuple[int, int], operation: tp.Callable
):
    k_x, k_y = kernel
    m_x, m_y = arr.shape[:2]

    assume(m_x % k_x == 0 and m_y % k_y == 0)

    out = array_operations.pool_2d_first_two_dimensions(arr, kernel, operation)

    assert out.shape == (m_x // k_x, m_y // k_y, arr.shape[2], arr.shape[3])

    block_mean = operation(arr[:k_x, :k_y, 0, 0])
    assert np.isclose(out[0, 0, 0, 0], block_mean)


@given(
    arr=array_strategy, kernel=kernel_strategy, operation=reduction_operation_strategy
)
def test_value_error_when_array_is_not_divisible_by_kernel(
    arr: npt.NDArray[tp.Any], kernel: tp.Tuple[int, int], operation: tp.Callable
):
    k_x, k_y = kernel
    m_x, m_y, n_res, n_ticks = arr.shape

    assume(m_x % k_x != 0 or m_y % k_y != 0)

    arr = np.random.rand(m_x, m_y, n_res, n_ticks).astype(np.float64)
    kernel = (k_x, k_y)

    with pytest.raises(ValueError):
        array_operations.pool_2d_first_two_dimensions(arr, kernel, operation=operation)


@given(arr=array_strategy, kernel=kernel_strategy, fill_value=st.floats(-10, 10))
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_pad_for_hierarchy(
    arr: npt.NDArray[tp.Any], kernel: tp.Tuple[int, int], fill_value: float
):
    m_x, m_y = arr.shape[:2]
    k_x, k_y = kernel
    assume(m_x > k_x and m_y > k_y)

    padded = array_operations.pad_for_hierarchy(arr, kernel, fill_value=fill_value)
    new_m_x, new_m_y = padded.shape[:2]

    assert new_m_x % k_x == 0 and new_m_y % k_y == 0

    assert np.allclose(padded[:m_x, :m_y, ...], arr)

    if new_m_x > m_x:
        assert np.all(padded[m_x:, :, ...] == fill_value)
    if new_m_y > m_y:
        assert np.all(padded[:, m_y:, ...] == fill_value)


@given(array=array_strategy, kernel=kernel_strategy)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_compute_max_levels(array, kernel):
    m_x, m_y = array.shape[:2]
    k_x, k_y = kernel

    assume(k_x > 1 and k_y > 1)
    assume(m_x % k_x == 0 and m_y % k_y == 0)

    max_level_x, max_level_y = array_operations.compute_levels(array.shape, kernel)
    max_levels = min(max_level_x, max_level_y)

    assert max_levels >= 1, "Value of max level can't be negative or zero"

    new_m_x = m_x
    for _ in range(max_level_x):
        new_m_x = new_m_x // k_x

    new_m_y = m_y
    for _ in range(max_level_y):
        new_m_y = new_m_y // k_y

    assert new_m_x >= 1, "Value of new x_axis level can't be negative or zero"
    assert new_m_y >= 1, "Value of new y_axis level can't be negative or zero"

    new_m_x_next = new_m_x // k_x
    new_m_y_next = new_m_y // k_y
    assert new_m_x_next < k_x or new_m_y_next < k_y, (
        f"compute_max_levels returned an incorrect value: after {max_levels} levels, "
        f"the next pooling step would not reduce at least one dimension below the kernel size. "
        f"Next dimensions would be (new_m_x_next={new_m_x_next}, new_m_y_next={new_m_y_next}), "
        f"kernel={kernel}, original dimensions={array.shape[:2]}"
    )


@given(
    array=array_strategy, kernel=kernel_strategy, operation=reduction_operation_strategy
)
@settings(
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    deadline=None,
)
def test_hierarchical_pooling(
    array: npt.NDArray[tp.Any], kernel: tp.Tuple[int, int], operation: tp.Callable
):
    k_x, k_y = kernel
    m_x, m_y = array.shape[:2]

    assume(k_x > 1 and k_y > 1)
    assume(m_x > k_x and m_y > k_y)
    assume(m_x % k_x == 0 and m_y % k_y == 0)

    original, *outputs = array_operations.hierarchical_pooling(
        array, kernel, operation=operation
    )

    assert isinstance(outputs, list), (
        f"hierarchical_pooling should return a list, got {type(outputs)} instead"
    )
    assert len(outputs) > 0, (
        "hierarchical_pooling returned an empty list; at least one level of pooling is expected"
    )

    for i, level in enumerate(outputs):
        assert level.shape[0] % k_x == 0 or level.shape[0] < k_x, (
            f"Level {i} has invalid x-dimension {level.shape[0]} for kernel {k_x}; "
            "should be divisible by kernel or smaller than kernel"
        )
        assert level.shape[1] % k_y == 0 or level.shape[1] < k_y, (
            f"Level {i} has invalid y-dimension {level.shape[1]} for kernel {k_y}; "
            "should be divisible by kernel or smaller than kernel"
        )

    padded = array_operations.pad_for_hierarchy(array, kernel)
    assert np.allclose(original, padded)

    prev_shape = padded.shape[:2]
    for i, level in enumerate(outputs):
        new_shape = level.shape[:2]
        assert new_shape[0] <= prev_shape[0], (
            f"Level {i} x-dimension {new_shape[0]} is larger than previous level {prev_shape[0]}"
        )
        assert new_shape[1] <= prev_shape[1], (
            f"Level {i} y-dimension {new_shape[1]} is larger than previous level {prev_shape[1]}"
        )
        prev_shape = new_shape

    expected_first_level = block_operation(padded, kernel, operation=operation)
    assert np.allclose(outputs[0], expected_first_level), (
        "First level of hierarchical pooling does not match expected block-wise mean "
        f"for kernel {kernel}. Check padding and pooling computation."
    )


@given(
    array=array_strategy,
    kernel=kernel_strategy,
    fill_value=st.floats(-10, 10),
    operation=reduction_operation_strategy,
)
@settings(suppress_health_check=[HealthCheck.filter_too_much], deadline=None)
def test_get_window_from_cell(
    array: npt.NDArray[tp.Any],
    kernel: tp.Tuple[int, int],
    fill_value: float,
    operation: tp.Callable,
):
    k_x, k_y = kernel
    m_x, m_y = array.shape[:2]

    assume(k_x > 1 and k_y > 1)
    assume(m_x > k_x and m_y > k_y)
    assume(m_x % k_x == 0 and m_y % k_y == 0)

    outputs = array_operations.hierarchical_pooling(
        array, kernel, operation=operation, fill_value=fill_value
    )
    max_level = len(outputs) - 1

    if max_level == 0:
        return

    level = random.randint(1, max_level)

    level_array = outputs[level]
    n_cells_x, n_cells_y = level_array.shape[:2]

    cx = random.randint(0, max(0, n_cells_x - 1))
    cy = random.randint(0, max(0, n_cells_y - 1))
    cell = (cx, cy)

    window = array_operations.get_window_from_cell(outputs, level, cell, kernel)

    assert window.shape[0] == kernel[0], (
        f"Window x-dimension {window.shape[0]} != kernel {kernel[0]}"
    )
    assert window.shape[1] == kernel[1], (
        f"Window y-dimension {window.shape[1]} != kernel {kernel[1]}"
    )

    prev_level = outputs[level - 1]
    expected_window = prev_level[
        cx * k_x : (cx + 1) * k_x, cy * k_y : (cy + 1) * k_y, ...
    ]
    assert np.allclose(window, expected_window), (
        "Values in window do not match expected block in previous level"
    )


@given(
    prev_cell=st.tuples(st.integers(0, 10), st.integers(0, 10)),
    current_index=st.tuples(st.integers(0, 5), st.integers(0, 5)),
    kernel=kernel_strategy,
)
def test_global_cell_from_local(
    prev_cell: tp.Tuple[int, int],
    current_index: tp.Tuple[int, int],
    kernel: tp.Tuple[int, int],
):
    px, py = prev_cell
    cx, cy = current_index
    kx, ky = kernel

    assume(cx < kx and cy < ky)

    global_cell = array_operations.global_cell_from_local(
        prev_cell, current_index, kernel
    )

    assert isinstance(global_cell, tuple)
    assert len(global_cell) == 2

    expected_x = px * kx + cx
    expected_y = py * ky + cy
    assert global_cell == (expected_x, expected_y), (
        f"Expected {(expected_x, expected_y)}, got {global_cell}. "
        f"prev_cell={prev_cell}, current_index={current_index}, kernel={kernel}"
    )

    assert global_cell[0] >= 0
    assert global_cell[1] >= 0
