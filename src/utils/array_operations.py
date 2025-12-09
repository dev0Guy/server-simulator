import numpy as np
import numpy.typing as npt
import typing as tp
import math

def _smallest_n(shape: npt.ArrayLike, kernel: tp.Tuple[int, int]):
    m_x, m_y = shape[:2]
    k_x, k_y = kernel

    n_x = math.ceil(m_x / k_x)
    n_y = math.ceil(m_y / k_y)

    return n_x, n_y


def compute_levels(shape: npt.ArrayLike, kernel: tp.Tuple[int, int]) -> tp.Tuple[int, int]:
    """
    Computes the  number of recursive pooling levels possible for first to dimensions
    for the first two dimensions of the array.
    """
    m_x, m_y = shape[:2]
    k_x, k_y = kernel

    # Handle kernel values of 1 safely
    if k_x == 1 and k_y == 1:
        raise ValueError("Kernel dimensions must be > 1")
    if m_x < k_x or m_y < k_y:
        raise ValueError("Some Problem")

    levels_x = int(math.floor(math.log(m_x, k_x)))
    levels_y = int(math.floor(math.log(m_y, k_y)))

    return levels_x, levels_y


def pool_2d_first_two_dimensions(arr: npt.NDArray[tp.Any], kernel: tp.Tuple[int, int]) -> npt.NDArray[tp.Any]:
    """
    arr shape: (WindowX, WindowY, OtherDim, OtherDim)
    kernel: (k_h, k_w)
    applies 2D pooling on (WindowX, WindowY) only
    stride = kernel
    """

    if len(arr.shape) != 4:
        raise ValueError(f"Array shape should have 4 dimension of (WindowX, WindowY, OtherDim, OtherDim) and not {arr.shape}.")
    if len(kernel) != 2:
        raise ValueError(f"Kernel should be of size 2, not {len(kernel)}")

    m_x, m_y, n_resources, n_ticks = arr.shape
    k_x, k_y = kernel

    if m_x % k_x != 0 or m_y % k_y != 0:
        raise ValueError(
            f"The array dimensions {arr.shape[:2]} are not divisible by the kernel size {kernel}. "
            "Each dimension of the array must be an exact multiple of the corresponding kernel dimension "
            "to perform block operations."
        )

    adjusted_array = arr.reshape(m_x// k_x, k_x, m_y // k_y, k_y, n_resources, n_ticks)

    return adjusted_array.mean(axis=(1, 3))


# def pad_for_hierarchy(array: npt.NDArray[tp.Any], kernel: tp.Tuple[int, int], *, fill_value: float = 0) -> np.ndarray:
#     """
#     Pads the first two dimensions of arr so they are multiples of kernel.
#     Pads with zeros at the end of each axis.
#     """
#     m_x, m_y = array.shape[:2]
#     k_x, k_y = kernel
#
#     n_x, n_y = _smallest_n((m_x, m_y), kernel)
#     n_xy = max(n_x, n_y)
#
#     pad_x = n_xy * k_x - m_x
#     pad_y = n_xy * k_y - m_y
#
#     pad_width = ((0, pad_x), (0, pad_y)) + ((0, 0),) * (array.ndim - 2)
#
#     return np.pad(array, pad_width=pad_width, mode='constant', constant_values=fill_value)

def pad_for_hierarchy(array: npt.NDArray[tp.Any], kernel: tp.Tuple[int, int], *, fill_value: float = 0) -> np.ndarray:
    """
    Pads the first two dimensions of `array` so that recursive pooling
    with `kernel` can be applied for all levels.

    Padding is applied at the end of each axis with `fill_value`.

    This ensures that after repeated pooling, the resulting array dimensions
    remain divisible by the kernel at every level.
    """
    m_x, m_y = array.shape[:2]
    k_x, k_y = kernel

    if k_x < 1 or k_y < 1:
        raise ValueError("Kernel dimensions must be >= 1")

    # Compute how many recursive levels we can pool (ignoring rounding)
    def max_recursive_levels(dim, k):
        if k == 1:
            return 0  # No reduction possible
        levels = 0
        while dim > 1:
            dim = math.ceil(dim / k)
            levels += 1
        return levels

    levels_x = max_recursive_levels(m_x, k_x)
    levels_y = max_recursive_levels(m_y, k_y)
    max_levels = max(levels_x, levels_y)

    # Compute total required size to sustain all levels
    target_m_x = k_x ** max_levels
    target_m_y = k_y ** max_levels

    # Ensure we at least match original size
    target_m_x = max(target_m_x, m_x)
    target_m_y = max(target_m_y, m_y)

    pad_x = target_m_x - m_x
    pad_y = target_m_y - m_y

    pad_width = ((0, pad_x), (0, pad_y)) + ((0, 0),) * (array.ndim - 2)
    return np.pad(array, pad_width=pad_width, mode='constant', constant_values=fill_value)


def hierarchical_pooling(array: npt.NDArray[tp.Any], kernel: tp.Tuple[int, int], fill_value: float = 0) -> tp.List[npt.NDArray[tp.Any]]:
    """
    Recursively applies pool_2d_first_two_dimensions on arr.
    Pads once using minimal padding so that all levels are divisible by kernel.
    Returns a list of outputs at each level.
    """
    padded = pad_for_hierarchy(array, kernel, fill_value=fill_value)
    outputs = []
    max_levels = compute_levels(padded.shape, kernel)
    current = padded

    for _ in range(max(max_levels)):
        current = pool_2d_first_two_dimensions(current, kernel)
        outputs.append(current)

        if current.shape[0] <= kernel[0] or current.shape[1] <= kernel[1]:
            break

    return outputs


def get_window_from_cell(outputs: tp.List[npt.NDArray[tp.Any]],level: int, cell: tp.Tuple[int, int], kernel: tp.Tuple[int, int]) -> npt.NDArray[tp.Any]:
    """
    Given hierarchical outputs, a level, and a cell (x,y) at that level,
    return the corresponding window of size `kernel` in the previous level.
    """
    k_x, k_y = kernel
    cx, cy = cell

    if level == 0:
        raise ValueError("Level 0 is the first pooled level; no previous window exists")

    prev_level = outputs[level - 1]

    start_x = cx * k_x
    start_y = cy * k_y
    end_x = start_x + k_x
    end_y = start_y + k_y

    return prev_level[start_x:end_x, start_y:end_y, ...]