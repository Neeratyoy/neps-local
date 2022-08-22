from __future__ import annotations


from abc import abstractmethod
import random
from typing import Any, List, Dict

import numpy as np
import metahyper

from metahyper.api import ConfigResult

from ...search_spaces.search_space import SearchSpace
from ...search_spaces.numerical.integer import IntegerParameter


class SamplingPolicy:
    """Base class for implementing a sampling straregy for Successive halving and its subclasses"""

    def __init__(
            self,
            pipeline_space: SearchSpace
    ):
        self.pipeline_space = pipeline_space

    @abstractmethod
    def sample(self, num_configs: int = 1) -> list[SearchSpace]:
        pass


class UniformRandomPolicy(SamplingPolicy):
    """A random policy for sampling configuration, i.e. the default for SH / hyperband

    Args:
        SamplingPolicy ([type]): [description]
    """

    def __init__(
        self,
        pipeline_space: SearchSpace,

    ):
        super().__init__(pipeline_space=pipeline_space)

    def sample(self, num_configs):
        pass
