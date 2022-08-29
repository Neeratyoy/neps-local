from __future__ import annotations

import logging
from copy import deepcopy
from typing import Iterable, Union

import numpy as np
import scipy.stats as sps
import torch
from statsmodels.nonparametric._kernel_base import _adjust_shape, gpke, kernel_func
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from metahyper.api import ConfigResult, instance_from_map

import neps
from neps.search_spaces import (
    CategoricalParameter,
    FloatParameter,
    IntegerParameter,
    NumericalParameter,
)
from neps.optimizers import BaseOptimizer
from neps.search_spaces.search_space import SearchSpace


class MultiFidelityPriorWeightedTreeParzenEstimator(BaseOptimizer):

    def __init__(self,
        pipeline_space: SearchSpace,
        max_fidelity: int,
        prior_num_evals: float = 1.0,
        good_fraction: float = 0.333,
        random_interleave_prob: float = 0.333,
        initial_design_size: int = 0,
        prior_as_samples: bool = False,
        patience: int = 50,
        logger=None,
        budget: None | int | float = None,
        loss_value_on_error: None | float = None,
        cost_value_on_error: None | float = None
    ):
        """[summary]

        Args:
            pipeline_space: Space in which to search
            max_fidelity (int): [description]
            prior_num_evals (float, optional): [description]. Defaults to 1.0.
            good_fraction (float, optional): [description]. Defaults to 0.333.
            random_interleave_prob: Frequency at which random configurations are sampled
                instead of configurations from the acquisition strategy.
            initial_design_size: Number of 'x' samples that are to be evaluated before
                selecting a sample using a strategy instead of randomly. If there is a
                user prior, we can rely on the model from the very first iteration.
            prior_as_samples: Whether to sample from the KDE and incorporate that way, or
            just have the distribution be an linear combination of the KDE and the prior. 
            Should be True if the prior happens to be unnormalized.
            patience: How many times we try something that fails before giving up.
            budget: Maximum budget
            loss_value_on_error: Setting this and cost_value_on_error to any float will
                supress any error during bayesian optimization and will use given loss
                value instead. default: None
            cost_value_on_error: Setting this and loss_value_on_error to any float will
                supress any error during bayesian optimization and will use given cost
                value instead. default: None
            logger: logger object, or None to use the neps logger
        """
        super().__init__(
            pipeline_space=pipeline_space,
            patience=patience,
            logger=logger,
            budget=budget,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error
        )
        self.pipeline_space = pipeline_space
        self.max_fidelity = max_fidelity
        self.prior_num_evals = prior_num_evals
        self.good_fraction = good_fraction
        self.random_interleave_prob = random_interleave_prob
        self.initial_design_size = initial_design_size
        
        param_types, num_values, logged_params = self._get_types() 
        

        if self.pipeline_space.has_prior:
            pass

        # TODO normalize the training data (plus half on either side like usual in SMAC)
        # TODO log the training data

    def eval(self):
        pass

    def _get_types(self):
        """extracts the needed types from the configspace for faster retrival later

        type = 0 - numerical (continuous or integer) parameter
        type >=1 - categorical parameter

        TODO: figure out a way to properly handle ordinal parameters

        """
        types = []
        num_values = []
        logs = []
        for param_name, hp in self.pipeline_space.items():
            print(hp)
            if isinstance(hp, CategoricalParameter):
                # u as in unordered - used to play nice with the statsmodels KDE implementation
                types.append("u")
                logs.append(False)
                num_values.append(len(hp.choices))
            elif isinstance(hp, IntegerParameter):
                # o as in ordered
                types.append("o")
                logs.append(False)
                num_values.append(hp.upper - hp.lower + 1)
            elif isinstance(hp, FloatParameter):
                # c as in continous
                types.append("c")
                logs.append(hp.log)
                num_values.append(np.inf)

            else:
                raise ValueError("Unsupported Parametertype %s" % type(hp))

        return types, num_values, logs

    def _convert_config_to_data(self, configs):
        """Converts incoming configurations to a numpy array format

        Args:
            configs ([type]): [description]
        """
        train_X, train_y = configs
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

    def register_pending(self, pending_evaluations, register_as_bad: bool = True):
        pass

    def load_results(
            self,
            previous_results: dict[str, ConfigResult],
            pending_evaluations: dict[str, ConfigResult],
        ) -> None:
        train_x = [el.config for el in previous_results.values()]
        train_y = [self.get_loss(el.result) for el in previous_results.values()]
        self._num_train_x = len(train_x)
        self._pending_evaluations = [el for el in pending_evaluations.values()]


    def get_config_and_ids(  # pylint: disable=no-self-use
            self,
        ) -> tuple[SearchSpace, str, str | None]:
        pass

if __name__ == "__main__":

    some_placeholder_space = dict(
        learning_rate=FloatParameter(
            lower=1e-4, upper=1e0, log=True, default=1e-1, default_confidence="high"
        ),
        categorical=CategoricalParameter(choices=["catA", "catB"]),
        integer=IntegerParameter(lower=2, upper=5, log=True),
        epoch=IntegerParameter(lower=1, upper=100, is_fidelity=True),
    )
    tpe_test = MultiFidelityPriorWeightedTreeParzenEstimator(
        pipeline_space=SearchSpace(**some_placeholder_space), 
        max_fidelity=100
    )
