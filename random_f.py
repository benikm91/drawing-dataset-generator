from typing import Optional, Callable, List, Any

import numpy as np
import scipy


class TruncatedNormalSampler:
    def __init__(self, loc: float, scale: float, min_value: Optional[float] = None, max_value: Optional[float] = None, integer: bool=False):
        self.loc = loc
        self.scale = scale
        self.min_value = min_value
        self.max_value = max_value
        self.integer = integer

    def __call__(self) -> float:
        lower_bound = -np.inf if self.min_value is None else (self.min_value - self.loc) / self.scale
        upper_bound = np.inf if self.max_value is None else (self.max_value - self.loc) / self.scale
        value = scipy.stats.truncnorm(a=lower_bound, b=upper_bound, loc=self.loc, scale=self.scale)
        result = value.rvs()
        if self.integer:
            return int(round(result))
        return result


class UniformSampler:
    def __init__(self, values: List[Any]):
        self.values = values

    def __call__(self) -> Any:
        return np.random.choice(self.values)
