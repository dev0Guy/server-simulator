from src.cluster.core.dilation import DilationOperationReturnType, DilationOperation
from src.cluster.implementation.metric_based.dilation import MetricBasedDilator
from src.utils import array_operations
import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck, reproduce_failure

from tests.test_cluster.test_utils.test_array_operations import reduction_operation_strategy

array_strategy = st.builds(
    lambda m_x, m_y, n_resources, n_ticks: np.random.rand(m_x, m_y, n_resources, n_ticks).astype(np.float64),
    m_x=st.integers(2, 8),
    m_y=st.integers(2, 8),
    n_resources=st.integers(1,3),
    n_ticks=st.integers(1,4)
)

def points_strategy(x, y, *, min_size=1, max_size=100):
    return st.lists(
        st.tuples(
            st.integers(min_value=0,max_value=x),
            st.integers(min_value=0, max_value=y),
        ),
        min_size=min_size,
        max_size=max_size,
    )

def action_strategy(kernel):
    kx, ky = kernel
    max_action = kx * ky
    return st.integers(min_value=-1, max_value=max_action)

kernel_strategy = st.tuples(st.integers(2, 10), st.integers(2, 10))

@st.composite
def kernel_and_points(draw):
    kernel = draw(kernel_strategy)
    kx, ky = kernel

    points = draw(points_strategy(kx-1, ky-1))
    return kernel, points

@given(
    array=array_strategy,
    kernel=kernel_strategy,
    operation=reduction_operation_strategy
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_dilator_init(array, kernel, operation):
    k_x, k_y = kernel
    m_x, m_y = array.shape[:2]

    assume(k_x > 1 and k_y > 1)
    assume(m_x > k_x and m_y > k_y)

    x_levels, y_levels = array_operations.compute_levels(array.shape, kernel)
    assume(x_levels > 1 and y_levels > 1)

    print(f"{kernel=}, {array.shape=}")
    dilator = MetricBasedDilator(kernel=kernel, state=array, operation=operation)
    assert dilator.kernel_shape() == kernel, "Kernel shape should match input kernel"
    assert dilator._n_levels == len(dilator._dilation_levels), "Number of levels mismatch"
    assert dilator._current_dilation_level_ptr == len(dilator._dilation_levels) - 1, "Pointer should start at top level"
    assert isinstance(dilator._prev_selected_cell, list)
    assert all(x is None for x in dilator._prev_selected_cell)

@given(
    array=array_strategy,
    kernel_with_points=kernel_and_points(),
    operation=reduction_operation_strategy
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_zoom_in_and_zoom_out(array, kernel_with_points, operation):
    kernel, points = kernel_with_points

    k_x, k_y = kernel
    m_x, m_y = array.shape[:2]

    assume(k_x > 1 and k_y > 1)
    assume(m_x > k_x and m_y > k_y)

    x_levels, y_levels = array_operations.compute_levels(array.shape, kernel)
    assume(x_levels > 1 and y_levels > 1)
    dilator = MetricBasedDilator(kernel=kernel, state=array, operation=operation)
    assume(dilator._n_levels >= 2)
    original_state = dilator.update(array)

    selected_cell = points[0]
    zi = dilator.zoom_in(selected_cell)

    assert isinstance(zi, DilationOperationReturnType)
    assert zi.state.shape[:2] == kernel

    zo = dilator.zoom_out()
    assert isinstance(zo, DilationOperationReturnType)
    assert zo.state.shape[:2] == kernel
    assert np.allclose(original_state, zo.state)

@given(
    array=array_strategy,
    kernel_with_points=kernel_and_points(),
    operation=reduction_operation_strategy,
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_zoom_in_until_select_real_machine(array, kernel_with_points, operation):
    kernel, points = kernel_with_points
    k_x, k_y = kernel
    m_x, m_y, n_resource, n_ticks = array.shape

    assume(k_x > 1 and k_y > 1)
    assume(m_x > k_x and m_y > k_y)

    x_levels, y_levels = array_operations.compute_levels(array.shape, kernel)
    assume(x_levels > 1 and y_levels > 1)

    dilator = MetricBasedDilator(kernel=kernel, state=array, operation=operation)
    assume(len(points) >= dilator._n_levels+1)

    points_iter = iter(points)

    while dilator._current_dilation_level_ptr != 0:
        dilation_action = next(points_iter)
        assert dilator.zoom_in(dilation_action).state.shape == (*kernel, n_resource, n_ticks)

    selection_action = next(points_iter)
    result = dilator.select_real_machine(selection_action)
    assert result.operation == DilationOperation.Execute
    assert result.state.shape == (n_resource, n_ticks)

@given(
    array=array_strategy,
    kernel=kernel_strategy,
    operation=reduction_operation_strategy,
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_zoom_out_on_initialize_state_return_error(array, kernel, operation):
    k_x, k_y = kernel
    m_x, m_y, n_resource, n_ticks = array.shape

    assume(k_x > 1 and k_y > 1)
    assume(m_x > k_x and m_y > k_y)

    dilator = MetricBasedDilator(kernel=kernel, state=array, operation=operation)

    dilator._prev_selected_cell = [None for _ in range(dilator._n_levels)]
    operation = dilator.zoom_out()
    assert operation.operation == DilationOperation.Error


@given(
    array=array_strategy,
    kernel_with_points=kernel_and_points(),
    operation=reduction_operation_strategy,
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_execute_action(array, kernel_with_points, operation):
    kernel, points = kernel_with_points
    kx, ky = kernel
    mx, my, nr, nt = array.shape

    assume(mx > kx and my > ky)

    dilator = MetricBasedDilator(kernel=kernel, state=array, operation=operation)
    assume(dilator._n_levels > 1)

    action = points[0][0] * ky + points[0][1]

    zoom_in_result = dilator.execute_action(action)

    if zoom_in_result.operation == DilationOperation.ZoomIn:
        assert zoom_in_result.state.shape == (*kernel, nr, nt)

        result = dilator.execute_action(-1)
        assert result.state.shape == (*kernel, nr, nt)

        result = dilator.execute_action(action)
        assert result.state.shape == (*kernel, nr, nt)
        assert result == zoom_in_result

    if zoom_in_result.operation == DilationOperation.Execute:
        assert zoom_in_result.state.shape == (nr, nt)
