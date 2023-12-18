import ast
from abc import abstractmethod
from typing import Callable, Dict

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from scipy.stats import linregress

from ...utils.log import get_logger
from . import Backend


def sliding_window_view_with_padding(
    arr, window_shape, axis=0, padding_value=0
):
    """Generate a sliding window view of an array with padding.

    Parameters:
        arr (numpy.ndarray): Input array.
        window_shape (int): Size of the sliding window.
        axis (int, optional): The axis along which the sliding window is applied. Default is 0.
        padding_value (int, float, optional): The value to use for padding. Default is 0.

    Returns:
        numpy.ndarray: A sliding window view with padding.

    """

    # Calculate padding shape
    padding_shape = list(arr.shape)
    padding_shape[axis] = window_shape - 1

    # Create a padding array
    padding_arr = np.full(padding_shape, padding_value, dtype=arr.dtype)

    # Concatenate the padding array and the input array along the specified axis
    padded_arr = np.concatenate((padding_arr, arr), axis=axis)

    # Generate the sliding window view on the padded array
    sliding_view = sliding_window_view(
        padded_arr, window_shape=window_shape, axis=axis
    )

    return sliding_view


class ComputeBackend(Backend):
    """The base compute backend that implements the generic compute fn. Compute
    backends should be stateless. They are just a collection of operator
    functions.

    Examples
    --------
    data = {
        "a": pd.DataFrame([1, 2, 3]),
        "b": pd.DataFrame([4, 5, 6]),
        "c": pd.DataFrame([7, 8, 9]),
    }

    backend = ComputeBackend()
    result = backend.compute(
        data, "data['a'] + data['b'] * data['c'] - data['a'] * data['c']"
    )
    print(result)

    """

    backend_type: type = "compute"
    operator_map = {
        "Add": "add",
        "Sub": "sub",
        "Mult": "mul",
        "Div": "div",
        "Mod": "mod",
        "Pow": "pow",
        "LShift": "lshift",
        "RShift": "rshift",
        "BitOr": "bit_or",
        "BitAnd": "bit_and",
        "BitXor": "bit_xor",
        "FloorDiv": "floor_div",
        "Invert": "invert",
        "Not": "not_",
        "UAdd": "uadd",
        "USub": "usub",
        "Eq": "eq",
        "NotEq": "ne",
        "Lt": "lt",
        "LtE": "le",
        "Gt": "gt",
        "GtE": "ge",
        "Is": "is_",
        "IsNot": "is_not",
        "In": "in_",
        "NotIn": "not_in",
        "And": "and_",
        "Or": "or_",
    }

    @classmethod
    def _compute(cls, data: Dict[str, pd.DataFrame], expr: str) -> pd.DataFrame:
        parsed_expr = ast.parse(expr, mode="eval")
        result = cls._eval_expr(parsed_expr.body, data)
        return result

    @classmethod
    @abstractmethod
    def compute(cls, data: Dict[str, pd.DataFrame], expr: str) -> pd.DataFrame:
        raise NotImplementedError()

    @classmethod
    def _eval_expr(
        cls, node: ast.AST, data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        if isinstance(node, ast.Name):
            return data[node.id]
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            args = [cls._eval_expr(arg, data) for arg in node.args]

            if (
                "ts" in func_name
            ):  # Trunc floating point number arguments in ts functions
                for i, arg in enumerate(args):
                    if isinstance(arg, float):
                        args[i] = int(np.floor(arg))

            func = cls.__getattr__(func_name)
            cls.logger.info(f"Calling {func_name} with args:\n{args}")
            result = func(*args)
            cls.logger.info(f"Result: {result}")
            return result
        elif isinstance(node, ast.BinOp):
            operator_func = cls._get_operator_func(node.op)
            left = cls._eval_expr(node.left, data)
            right = cls._eval_expr(node.right, data)

            return operator_func(left, right)
        elif isinstance(node, ast.UnaryOp):
            operator_func = cls._get_operator_func(node.op)
            operand = cls._eval_expr(node.operand, data)

            return operator_func(operand)

        elif isinstance(node, ast.Compare):
            left = cls._eval_expr(node.left, data)
            ops = node.ops
            comparators = [
                cls._eval_expr(comp, data) for comp in node.comparators
            ]

            for op, comp in zip(ops, comparators):
                operator_func = cls._get_operator_func(op)
                left = operator_func(left, comp)

            return left

        elif isinstance(node, ast.BoolOp):
            operator_func = cls._get_operator_func(node.op)
            values = [cls._eval_expr(value, data) for value in node.values]

            result = values[0]
            for value in values[1:]:
                result = operator_func(result, value)

            return result

        elif isinstance(node, ast.Constant):
            return node.value
        else:
            raise ValueError(f"Unsupported node type: {type(node).__name__}")

    @classmethod
    def _get_operator_func(
        cls, op: ast.operator
    ) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        return cls.__getattr__(cls.operator_map[type(op).__name__])

    @staticmethod
    def usub(a):
        return -a


class NumpyComputeBackend(ComputeBackend):
    logger = get_logger("NumpyComputeBackend")

    @classmethod
    def __getattr__(cls, name):
        if hasattr(cls, name):  # Override existing methods in hxdf
            cls.logger.info(f"Using {name} from PandasComputeBackend")
            return getattr(cls, name)
        elif hasattr(np.ndarray, name):  # Inherit from hxdf
            cls.logger.info(f"Using {name} from np.ndarray")
            return getattr(np.ndarray, name)
        elif hasattr(np.ndarray, f"__{name}__"):  # try magic methods
            cls.logger.info(f"Using magic method __{name}__ from np.ndarray")
            return getattr(np.ndarray, f"__{name}__")
        else:
            raise AttributeError(f"Unsupported operator type: {name}")

    @classmethod
    def compute(
        cls, data: Dict[str, pd.DataFrame], expr: str, index: int = None
    ) -> pd.DataFrame:
        new_data = {k: v.values for k, v in data.items()}
        result_np: np.ndarray = cls._compute(new_data, expr)
        example_df = list(data.values())[0]
        df_index = example_df.index
        df_col = example_df.columns
        result = pd.DataFrame(result_np, index=df_index, columns=df_col)
        return result

    # Polymorphic operators
    @staticmethod
    def max(a: np.ndarray, b):
        if isinstance(b, np.ndarray):
            return np.where(a > b, a, b)
        else:
            time_window = int(np.floor(b))
            return sliding_window_view_with_padding(
                a, window_shape=time_window
            ).max(axis=-1)

    @staticmethod
    def min(a: pd.DataFrame, b):
        if isinstance(b, pd.DataFrame):
            return np.where(a < b, a, b)
        else:
            time_window = int(np.floor(b))
            return sliding_window_view_with_padding(
                a, window_shape=time_window
            ).min(axis=-1)

    # operator overloading
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def sub(a, b):
        return a - b

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def div(a, b):
        return a / b

    @staticmethod
    def pow(a: pd.DataFrame, b):
        if isinstance(a, float) or isinstance(a, int):
            return a**b
        if isinstance(b, pd.DataFrame):
            new_values = np.power(a.values, b.values)
            return new_values
        else:
            return a**b

    @classmethod
    def delay(cls, arr: np.ndarray, n: int) -> np.ndarray:
        """Time-series delay (lag)."""
        return np.roll(arr, shift=n, axis=0)

    @classmethod
    def gt(cls, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Greater than operation."""
        return arr1 > arr2

    @classmethod
    def lt(cls, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Less than operation."""
        return arr1 < arr2

    @classmethod
    def rank(cls, arr: np.ndarray) -> np.ndarray:
        """Ranking of elements."""
        return stats.rankdata(arr)

    @classmethod
    def residual(cls, arr: np.ndarray, window: int) -> np.ndarray:
        """Calculate the residual from linear regression within a rolling
        window."""

        def calc_residual(a):
            y = a
            x = np.arange(len(a))
            slope, intercept, _, _, _ = stats.linregress(x, y)
            return y - (slope * x + intercept)

        sliding_windows = sliding_window_view_with_padding(
            arr, window_shape=window, axis=0
        )
        return np.apply_along_axis(calc_residual, axis=-1, arr=sliding_windows)

    @classmethod
    def slope(cls, arr: np.ndarray, window: int) -> np.ndarray:
        """Calculate the slope of linear regression within a rolling window."""

        def calc_slope(a):
            y = a
            x = np.arange(len(a))
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope

        sliding_windows = sliding_window_view_with_padding(
            arr, window_shape=window, axis=0
        )
        return np.apply_along_axis(calc_slope, axis=-1, arr=sliding_windows)

    @classmethod
    def ts_argmax(cls, arr: np.ndarray, window: int) -> np.ndarray:
        """Index of maximum value within a rolling window."""
        sliding_windows = sliding_window_view_with_padding(
            arr, window_shape=window, axis=0
        )
        return np.argmax(sliding_windows, axis=-1)

    @classmethod
    def ts_argmin(cls, arr: np.ndarray, window: int) -> np.ndarray:
        """Index of minimum value within a rolling window."""
        sliding_windows = sliding_window_view_with_padding(
            arr, window_shape=window, axis=0
        )
        return np.argmin(sliding_windows, axis=-1)

    @classmethod
    def ts_corr(
        cls, arr1: np.ndarray, arr2: np.ndarray, window: int
    ) -> np.ndarray:
        """Correlation between two series within a rolling window."""
        sliding_windows1 = sliding_window_view_with_padding(
            arr1, window_shape=window, axis=0
        )
        sliding_windows2 = sliding_window_view_with_padding(
            arr2, window_shape=window, axis=0
        )
        return np.corrcoef(sliding_windows1, sliding_windows2, rowvar=False)

    @classmethod
    def ts_max(cls, arr: np.ndarray, window: int) -> np.ndarray:
        """Maximum value within a rolling window."""
        sliding_windows = sliding_window_view_with_padding(
            arr, window_shape=window, axis=0
        )
        return np.max(sliding_windows, axis=-1)

    @classmethod
    def ts_mean(cls, arr: np.ndarray, window: int) -> np.ndarray:
        """Mean value within a rolling window."""
        sliding_windows = sliding_window_view_with_padding(
            arr, window_shape=window, axis=0
        )
        return np.mean(sliding_windows, axis=-1)

    @classmethod
    def ts_min(cls, arr: np.ndarray, window: int) -> np.ndarray:
        """Minimum value within a rolling window."""
        sliding_windows = sliding_window_view_with_padding(
            arr, window_shape=window, axis=0
        )
        return np.min(sliding_windows, axis=-1)

    @classmethod
    def ts_quantile(
        cls, arr: np.ndarray, window: int, quantile: float
    ) -> np.ndarray:
        """Quantile value within a rolling window."""
        sliding_windows = sliding_window_view_with_padding(
            arr, window_shape=window, axis=0
        )
        return np.quantile(sliding_windows, q=quantile, axis=-1)

    @classmethod
    def ts_rsquare(cls, arr: np.ndarray, window: int) -> np.ndarray:
        """R-squared value of linear regression within a rolling window."""

        def calc_rsquare(a):
            y = a
            x = np.arange(len(a))
            _, _, r_value, _, _ = stats.linregress(x, y)
            return r_value**2

        sliding_windows = sliding_window_view_with_padding(
            arr, window_shape=window, axis=0
        )
        return np.apply_along_axis(calc_rsquare, axis=-1, arr=sliding_windows)

    @classmethod
    def ts_std(cls, arr: np.ndarray, window: int) -> np.ndarray:
        """Standard deviation within a rolling window."""
        sliding_windows = sliding_window_view_with_padding(
            arr, window_shape=window, axis=0
        )
        return np.std(sliding_windows, axis=-1)

    @classmethod
    def ts_sum(cls, arr: np.ndarray, window: int) -> np.ndarray:
        """Sum of values within a rolling window."""
        sliding_windows = sliding_window_view_with_padding(
            arr, window_shape=window, axis=0
        )
        return np.sum(sliding_windows, axis=-1)

    @classmethod
    def condition(
        cls, arr1: np.ndarray, arr2: np.ndarray, arr3: np.ndarray
    ) -> np.ndarray:
        """Conditional operation."""
        return np.where(arr1, arr2, arr3)

    @classmethod
    def eq(cls, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Equality check."""
        return arr1 == arr2

    @classmethod
    def grouped_demean(cls, arr: np.ndarray, group: np.ndarray) -> np.ndarray:
        """Group-wise demeaning."""
        # Note: Group-wise demeaning might involve pandas for ease. Direct NumPy implementation can be complex.
        unique_groups = np.unique(group)
        demeaned = np.zeros_like(arr)
        for g in unique_groups:
            indices = np.where(group == g)
            demeaned[indices] = arr[indices] - np.mean(arr[indices])
        return demeaned

    @classmethod
    def or_(cls, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Logical OR operation."""
        return np.logical_or(arr1, arr2)

    @classmethod
    def sign(cls, arr: np.ndarray) -> np.ndarray:
        """Sign function."""
        return np.sign(arr)

    @classmethod
    def ts_cov(
        cls, arr1: np.ndarray, arr2: np.ndarray, window: int
    ) -> np.ndarray:
        """Covariance within a rolling window."""
        sliding_windows1 = sliding_window_view_with_padding(
            arr1, window_shape=window, axis=0
        )
        sliding_windows2 = sliding_window_view_with_padding(
            arr2, window_shape=window, axis=0
        )
        return np.cov(sliding_windows1, sliding_windows2, rowvar=False)

    @classmethod
    def ts_decayed_linear(
        cls, arr: np.ndarray, window: int, decay: float
    ) -> np.ndarray:
        """Decayed linear combination within a rolling window."""

        def decayed_linear(a):
            weights = np.power(decay, np.arange(len(a)))
            return np.dot(a, weights) / np.sum(weights)

        sliding_windows = sliding_window_view_with_padding(
            arr, window_shape=window, axis=0
        )
        return np.apply_along_axis(decayed_linear, axis=-1, arr=sliding_windows)

    @classmethod
    def ts_delta(cls, arr: np.ndarray, periods: int) -> np.ndarray:
        """Delta (difference) within a rolling window."""
        return np.diff(arr, n=periods, axis=0)

    @classmethod
    def ts_product(cls, arr: np.ndarray, window: int) -> np.ndarray:
        """Product within a rolling window."""
        sliding_windows = sliding_window_view_with_padding(
            arr, window_shape=window, axis=0
        )
        return np.prod(sliding_windows, axis=-1)

    @classmethod
    def ts_rank(cls, arr: np.ndarray, window: int) -> np.ndarray:
        """Rank within a rolling window."""

        def calc_rank(a):
            return stats.rankdata(a)[-1]

        sliding_windows = sliding_window_view_with_padding(
            arr, window_shape=window, axis=0
        )
        return np.apply_along_axis(calc_rank, axis=-1, arr=sliding_windows)

    @classmethod
    def twise_a_scale(cls, arr: np.ndarray, a: float) -> np.ndarray:
        """Two-wise scaling along the time(axis=0), scaled to 'a'."""
        scaling_factor = np.sum(arr, axis=0)
        return a * (arr / scaling_factor)


class PandasComputeBackend(ComputeBackend):
    logger = get_logger("PandasComputeBackend")

    @classmethod
    def __getattr__(cls, name):
        if hasattr(cls, name):  # Override existing methods in hxdf
            cls.logger.info(f"Using {name} from PandasComputeBackend")
            return getattr(cls, name)
        elif hasattr(pd.DataFrame, name):  # Inherit from hxdf
            cls.logger.info(f"Using {name} from pd.DataFrame")
            return getattr(pd.DataFrame, name)
        elif hasattr(pd.DataFrame, f"__{name}__"):  # try magic methods
            cls.logger.info(f"Using magic method __{name}__ from pd.DataFrame")
            return getattr(pd.DataFrame, f"__{name}__")
        else:
            raise AttributeError(f"Unsupported operator type: {name}")

    @classmethod
    def attr_getter(clse, name):
        def getter(self):
            return self[name]

        return getter

    @classmethod
    def compute(
        cls, data: Dict[str, pd.DataFrame], expr: str, index: int = None
    ) -> pd.DataFrame:
        result: pd.DataFrame = cls._compute(data, expr)
        return result

    # Polymorphic operators
    @staticmethod
    def max(a: pd.DataFrame, b):
        if isinstance(b, pd.DataFrame):
            return pd.DataFrame.where(a, a > b, b)
        else:
            time_window = int(np.floor(b))
            return a.rolling(time_window).max()

    @staticmethod
    def min(a: pd.DataFrame, b):
        if isinstance(b, pd.DataFrame):
            return pd.DataFrame.where(a, a < b, b)
        else:
            time_window = int(np.floor(b))
            return a.rolling(time_window).min()

    # operator overloading
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def sub(a, b):
        return a - b

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def div(a, b):
        return a / b

    @staticmethod
    def pow(a: pd.DataFrame, b):
        if isinstance(a, float) or isinstance(a, int):
            return a**b
        if isinstance(b, pd.DataFrame):
            new_values = np.power(a.values, b.values)
            return pd.DataFrame(new_values, index=a.index, columns=b.columns)
        else:
            return a**b

    @classmethod
    def delay(cls, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Implements time-series delay (lag)."""
        return df.shift(n)

    @classmethod
    def gt(cls, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Implements greater-than operation."""
        return df1 > df2

    @classmethod
    def lt(cls, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Implements less-than operation."""
        return df1 < df2

    @classmethod
    def rank(cls, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Implements ranking within a rolling window."""
        return df.rolling(window=window).apply(lambda x: x.rank().iloc[-1])

    # Additional specialized operators can be added here
    @classmethod
    def residual(cls, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate the residual from linear regression within a rolling
        window."""

        def calc_residual(arr: np.ndarray) -> float:
            y = arr
            x = np.arange(len(arr))
            slope, intercept, _, _, _ = linregress(x, y)
            return y[-1] - (slope * x[-1] + intercept)

        return df.rolling(window=window).apply(calc_residual)

    @classmethod
    def slope(cls, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate the slope of linear regression within a rolling window."""

        def calc_slope(arr: np.ndarray) -> float:
            y = arr
            x = np.arange(len(arr))
            slope, _, _, _, _ = linregress(x, y)
            return slope

        return df.rolling(window=window).apply(calc_slope)

    @classmethod
    def ts_argmax(cls, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Index of maximum value within a rolling window."""
        return df.rolling(window=window).apply(np.argmax)

    @classmethod
    def ts_argmin(cls, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Index of minimum value within a rolling window."""
        return df.rolling(window=window).apply(np.argmin)

    @classmethod
    def ts_corr(
        cls, df1: pd.DataFrame, df2: pd.DataFrame, window: int
    ) -> pd.DataFrame:
        """Correlation between two series within a rolling window."""
        return df1.rolling(window=window).corr(df2)

    @classmethod
    def ts_max(cls, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Maximum value within a rolling window."""
        return df.rolling(window=window).max()

    @classmethod
    def ts_mean(cls, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Mean value within a rolling window."""
        return df.rolling(window=window).mean()

    @classmethod
    def ts_min(cls, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Minimum value within a rolling window."""
        return df.rolling(window=window).min()

    @classmethod
    def ts_quantile(
        cls, df: pd.DataFrame, window: int, quantile: float
    ) -> pd.DataFrame:
        """Quantile value within a rolling window."""
        return df.rolling(window=window).quantile(quantile)

    @classmethod
    def ts_rsquare(cls, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """R-squared value of linear regression within a rolling window."""

        def calc_rsquare(arr: np.ndarray) -> float:
            y = arr
            x = np.arange(len(arr))
            _, _, r_value, _, _ = linregress(x, y)
            return r_value**2

        return df.rolling(window=window).apply(calc_rsquare)

    @classmethod
    def ts_std(cls, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Standard deviation within a rolling window."""
        return df.rolling(window=window).std()

    @classmethod
    def ts_sum(cls, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Sum of values within a rolling window."""
        return df.rolling(window=window).sum()

    @classmethod
    def condition(
        cls, df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame
    ) -> pd.DataFrame:
        """Conditional operation."""
        return df2.where(df1, df3)

    @classmethod
    def eq(cls, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Equality check."""
        return df1 == df2

    @classmethod
    def grouped_demean(cls, df: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
        """Group-wise demeaning."""
        return df - df.groupby(group).transform("mean")

    @classmethod
    def or_(cls, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Logical OR operation."""
        return df1 | df2

    @classmethod
    def sign(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Sign function."""
        return np.sign(df)

    @classmethod
    def ts_cov(
        cls, df1: pd.DataFrame, df2: pd.DataFrame, window: int
    ) -> pd.DataFrame:
        """Covariance within a rolling window."""
        return df1.rolling(window=window).cov(df2)

    @classmethod
    def ts_decayed_linear(
        cls, df: pd.DataFrame, window: int, decay: float
    ) -> pd.DataFrame:
        """Decayed linear combination within a rolling window."""

        def decayed_linear(arr: np.ndarray) -> float:
            weights = np.power(decay, np.arange(len(arr)))
            return np.dot(arr, weights) / np.sum(weights)

        return df.rolling(window=window).apply(decayed_linear)

    @classmethod
    def ts_delta(cls, df: pd.DataFrame, periods: int) -> pd.DataFrame:
        """Delta (difference) within a rolling window."""
        return df.diff(periods=periods)

    @classmethod
    def ts_product(cls, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Product within a rolling window."""
        return df.rolling(window=window).apply(np.product)

    @classmethod
    def ts_rank(cls, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Rank within a rolling window."""
        return df.rolling(window=window).apply(lambda x: x.rank().iloc[-1])

    @classmethod
    def twise_a_scale(cls, df: pd.DataFrame, a: float) -> pd.DataFrame:
        """Two-wise scaling along the time(index) axis, scaled to 'a'."""
        scaling_factor = df.sum(axis=0)
        return a * (df / scaling_factor)
