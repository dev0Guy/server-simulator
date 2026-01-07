from src.cluster.core.dilation import  DilationSteps
from src.cluster.implementation.metric_based.dilation import MetricBasedDilator
from src.utils import array_operations
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


def assume_valid_dilation_case(array, kernel):
    k_x, k_y = kernel
    m_x, m_y = array.shape[:2]

    assume(k_x > 1 and k_y > 1)
    assume(m_x > k_x and m_y > k_y)

    x_levels, y_levels = array_operations.compute_levels(array.shape, kernel)
    assume(x_levels > 1 and y_levels > 1)

@given(
    array=array_strategy,
    kernel=kernel_strategy,
    operation=reduction_operation_strategy
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_dilator_init(array, kernel, operation):
    assume_valid_dilation_case(array, kernel)

    dilator = MetricBasedDilator(kernel=kernel, state=array, operation=operation)

    assert dilator._n_levels == len(dilator._dilation_levels), "Number of levels mismatch"

    match dilator.state:
        case DilationSteps.Initial(value, level):
            assert level == dilator._n_levels - 1 ,"Pointer should start at top level"
            assert isinstance(value, np.ndarray)
            assert value.shape[:2] == kernel, "Kernel shape should match input kernel"
        case _:
            raise AssertionError(f"Dilator initialize step value is not `DilationSteps.Initial`")


@given(
    array=array_strategy,
    kernel_with_points=kernel_and_points(),
    operation=reduction_operation_strategy
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_expand_and_contract(array, kernel_with_points, operation):
    kernel, points = kernel_with_points

    assume_valid_dilation_case(array, kernel)

    dilator = MetricBasedDilator(kernel=kernel, state=array, operation=operation)

    assume(dilator._n_levels >= 2)
    original_state = dilator.generate_dilation_expansion(array)
    expanded_state = dilator.expand(points[0])

    match expanded_state:
        case DilationSteps.Expanded(prev, value, level) | DilationSteps.FullyExpanded(prev, value, level):
            assert value.shape[:2] == original_state.value.shape[:2] == kernel
            assert level == dilator._n_levels - 2
        case _:
            raise AssertionError

    contracted_state = dilator.contract()

    match contracted_state:
        case DilationSteps.Initial(value, level) | DilationSteps.Expanded(_, value, level):
            assert value.shape[:2] == original_state.value.shape[:2] == kernel
            assert level == dilator._n_levels - 1
            assert np.allclose(original_state.value, value)
        case _:
            raise AssertionError

@given(
    array=array_strategy,
    kernel_with_points=kernel_and_points(),
    operation=reduction_operation_strategy,
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_zoom_in_until_fully_expanded(array, kernel_with_points, operation):
    kernel, points = kernel_with_points

    assume_valid_dilation_case(array, kernel)

    dilator = MetricBasedDilator(kernel=kernel, state=array, operation=operation)

    assume(dilator._n_levels > 1)
    assume(len(points) >= dilator._n_levels)

    dilator.generate_dilation_expansion(array)

    expand_cells = points[:dilator._n_levels]

    for cell in expand_cells:
        match dilator.expand(cell):
            case DilationSteps.Expanded(prev, value, level):
                assert value.shape == (*kernel, *array.shape[2:])
            case DilationSteps.FullyExpanded(prev, value, level):
                assert value.shape == (*kernel, *array.shape[2:])
                break
            case _:
                raise AssertionError

@given(
    array=array_strategy,
    kernel=kernel_strategy,
    operation=reduction_operation_strategy,
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_zoom_out_on_initialization_does_nothing(array, kernel, operation):
    assume_valid_dilation_case(array, kernel)

    dilator = MetricBasedDilator(kernel=kernel, state=array, operation=operation)
    dilator.generate_dilation_expansion(array)
    match dilator.contract():
        case DilationSteps.Initial(_, _): ...
        case _:
            raise AssertionError

@given(
    array=array_strategy,
    kernel_with_points=kernel_and_points(),
    operation=reduction_operation_strategy,
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_zoom_in_with_zoom_out_until_fully_expanded(array, kernel_with_points, operation):
    kernel, points = kernel_with_points

    assume_valid_dilation_case(array, kernel)

    dilator = MetricBasedDilator(kernel=kernel, state=array, operation=operation)

    assume(dilator._n_levels > 1)
    assume(len(points) >= dilator._n_levels*2)

    dilator.generate_dilation_expansion(array)

    expand_cells = points[:dilator._n_levels*2]

    for idx, cell in enumerate(expand_cells):
        match dilator.expand(cell):
            case DilationSteps.Expanded(prev, value, _):
                assert value.shape == (*kernel, *array.shape[2:])
                if cell % 2 == 0:
                    prev_state = dilator.contract()
                    assert prev_state == prev
            case DilationSteps.FullyExpanded(_, value, level):
                assert value.shape == (*kernel, *array.shape[2:])
                assert level == 0
                break
            case _:
                raise AssertionError