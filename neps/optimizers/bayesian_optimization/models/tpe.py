import logging
from copy import deepcopy
from typing import Iterable, Union

import numpy as np
import scipy.stats as sps
import torch
from statsmodels.nonparametric._kernel_base import _adjust_shape, gpke, kernel_func
from statsmodels.nonparametric.kernel_density import KDEMultivariate

import neps
from neps.search_spaces import (
    CategoricalParameter,
    FloatParameter,
    IntegerParameter,
    NumericalParameter,
)
from neps.search_spaces.search_space import SearchSpace


class MultiFidelityPriorWeightedTreeParzenEstimator:

    def __init__(self, search_space: SearchSpace, max_fidelity: int, prior_num_evals: float = 1.0, good_fraction: float = 0.333):

        pass
        # TODO normalize the training data (plus half on either side like usual in SMAC)
        # TODO log the training data

    def _get_types(self):
    """extracts the needed types from the configspace for faster retrival later

    type = 0 - numerical (continuous or integer) parameter
    type >=1 - categorical parameter

    TODO: figure out a way to properly handle ordinal parameters

    """
    types = []
    num_values = []
    logs = []
    for param_name, hp in self.search_space.items():
        if isinstance(hp, CategoricalParameter):
            # u as in unordered - used to play nice with the statsmodels KDE implementation
            types.append("u")
            logs.append(False)
            num_values.append(len(hp.choices))
       elif isinstance(hp, FloatParameter):
            # c as in continous
            types.append("c")
            logs.append(hp.log)
            num_values.append(np.inf)
        elif isinstance(hp, (NumericalParameter, IntegerParameter)):
            # o as in ordered
            types.append("o")
            logs.append(False)
            num_values.append(len(hp.sequence))
        else:
            raise ValueError("Unsupported Parametertype %s" % type(hp))

    return types, num_values, logs

    def _convert_config_to_data(self, configs):
        """Converts incoming configurations to a numpy array format

        Args:
            configs ([type]): [description]
        """
        pass

    def _convert_data_to_config(self, data):
        """Converts outgoing numpy arrays to configuration format

        Args:
            configs ([type]): [description]
        """
        pass

    def _normalize_data(self):
        pass

    # TODO allow this for integers as well - now only supports floats
    def _convert_to_logscale(self):
        pass

    def fit_kdes(self, previous_results):
        pass

    def register_pending(self, pending_evaluations, register_as_bad: bool = True)
        pass

    def load_results(
            self,
            previous_results: dict[str, ConfigResult],
            pending_evaluations: dict[str, ConfigResult],
        ) -> None:
        pass

    def get_config_and_ids(  # pylint: disable=no-self-use
            self,
        ) -> tuple[SearchSpace, str, str | None]:
        pass

if __name__ == "__main__":

    some_placeholder_space = dict(
        learning_rate=neps.FloatParameter(
            lower=1e-4, upper=1e0, log=True, default=1e-1, default_confidence="high"
        ),
        categorical=neps.CategoricalParameter(choices=["catA", "catB"]),
        integer=neps.IntegerParameter(lower=2, upper=5, log=True),
        epoch=neps.IntegerParameter(lower=1, upper=100, is_fidelity=True),
    )
    tpe_test = MultiFidelityPriorWeightedTreeParzenEstimator(SearchSpace(**some_placeholder_space))
