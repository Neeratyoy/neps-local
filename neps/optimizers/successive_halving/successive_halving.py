from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from metahyper.api import ConfigResult
from typing_extensions import Literal

from ...search_spaces.numerical.integer import IntegerParameter
from ...search_spaces.search_space import SearchSpace
from ..base_optimizer import BaseOptimizer
from .promotion_policy import PromotionPolicy
from .sampling_policy import RandomUniformPolicy, SamplingPolicy


class SuccessiveHalving(BaseOptimizer):
    """Implements a SuccessiveHalving procedure with a sampling and promotion policy."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 4,
        early_stopping_rate: int = 0,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: SamplingPolicy = RandomUniformPolicy,
        promotion_policy: PromotionPolicy = None,
    ):
        super().__init__(pipeline_space=pipeline_space, budget=budget)
        self.min_budget = pipeline_space.fidelity.lower
        self.max_budget = pipeline_space.fidelity.upper
        self.eta = eta
        # SH implicitly sets early_stopping_rate to 0
        # the parameter is exposed to allow HB to call SH with different stopping rates
        self.early_stopping_rate = early_stopping_rate
        self.sampling_policy = sampling_policy
        self.promotion_policy = promotion_policy

        # `max_budget_init` checks for the number of configurations that have been
        # evaluated at the target budget
        self.initial_design_type = initial_design_type

        # check to ensure no rung ID is negative
        # equivalent to s_max in https://arxiv.org/pdf/1603.06560.pdf
        self.stopping_rate_limit = np.floor(
            np.log(self.max_budget / self.min_budget) / np.log(self.eta)
        ).astype(int)
        assert self.early_stopping_rate <= self.stopping_rate_limit

        # maps rungs to a fidelity value for an SH bracket with `early_stopping_rate`
        self.rung_map = self._get_rung_map(self.early_stopping_rate)
        self.config_map = self._get_config_map(self.early_stopping_rate)

        self.max_rung = len(self.rung_map) - 1
        self.fidelities = list(self.rung_map.values())
        # stores the observations made and the corresponding fidelity explored
        # crucial data structure used for determining promotion candidates
        self.observed_configs = pd.DataFrame([], columns=("config", "rung", "perf"))
        # stores which configs occupy each rung at any time
        self.rung_members: dict = {}  # stores config IDs per rung
        self.rung_members_performance: dict = dict()  # performances recorded per rung
        self.rung_promotions: dict = dict()  # records a promotable config per rung
        self.total_fevals = 0

    def _get_rung_map(self, s: int = 0) -> dict:
        """Maps rungs (0,1,...,k) to a fidelity value based on fidelity bounds, eta, s."""
        assert s <= self.stopping_rate_limit
        new_min_budget = self.min_budget * (self.eta**s)
        nrungs = (
            np.floor(np.log(self.max_budget / new_min_budget) / np.log(self.eta)).astype(
                int
            )
            + 1
        )
        _max_budget = self.max_budget
        rung_map = dict()
        for i in reversed(range(nrungs)):
            rung_map[i] = (
                int(_max_budget)
                if isinstance(self.pipeline_space.fidelity, IntegerParameter)
                else _max_budget
            )
            _max_budget /= self.eta
        return rung_map

    def _get_config_map(self, s: int = 0) -> dict:
        """Maps rungs (0,1,...,k) to the number of configs for each fidelity"""
        assert s <= self.stopping_rate_limit
        new_min_budget = self.min_budget * (self.eta**s)
        nrungs = (
            np.floor(np.log(self.max_budget / new_min_budget) / np.log(self.eta)).astype(
                int
            )
            + 1
        )
        s_max = self.stopping_rate_limit + 1
        _s = self.stopping_rate_limit - s
        # L2 from Alg 1 in https://arxiv.org/pdf/1603.06560.pdf
        _n_config = np.floor(s_max / (_s + 1)) * self.eta**_s
        config_map = dict()
        for i in reversed(range(nrungs)):
            config_map[i] = int(_n_config)
            _n_config //= self.eta
        return config_map

    # TODO: check pending
    def _load_previous_observations(
        self, previous_results: dict[str, ConfigResult]
    ) -> None:
        for config_id, config_val in previous_results.items():
            _config, _rung = config_id.split("_")
            if int(_config) in self.observed_configs.index:
                # config already recorded in dataframe
                rung_recorded = self.observed_configs.at[int(_config), "rung"]
                if rung_recorded < int(_rung):
                    # config recorded for a lower rung but higher rung eval available
                    self.observed_configs.at[int(_config), "rung"] = int(_rung)
                    self.observed_configs.at[int(_config), "perf"] = self.get_loss(
                        config_val.result
                    )
            else:
                _df = pd.DataFrame(
                    [[config_val.config, int(_rung), self.get_loss(config_val.result)]],
                    columns=self.observed_configs.columns,
                    index=pd.Series(int(_config)),  # key for config_id
                )
                self.observed_configs = pd.concat(
                    (self.observed_configs, _df)
                ).sort_index()
        return

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        """This is basically the fit method.

        Args:
            previous_results (dict[str, ConfigResult]): [description]
            pending_evaluations (dict[str, ConfigResult]): [description]
        """
        # TODO: Read in rungs using the config id (alternatively, use get/load state)
        self.observed_configs = pd.DataFrame([], columns=("config", "rung", "perf"))

        # previous optimization run exists and needs to be loaded
        self._load_previous_observations(previous_results)
        self.total_fevals = len(previous_results)
        # iterates over all previous results and updates the list of observed
        # configs with the highest fidelity it was evaluated on and its performance
        for config_id, config_val in previous_results.items():
            _config, _rung = config_id.split("_")
            perf = self.get_loss(config_val.result)
            if int(_config) not in self.observed_configs.index:
                # this condition and check is important to handle async scenarios as
                # the `previous_results` can provide configs that have not been
                # encountered by this instantiation of the optimizer object
                _df = pd.DataFrame(
                    [[config_val.config, int(_rung), perf]],
                    columns=self.observed_configs.columns,
                    index=pd.Series(int(_config)),  # key for config_id
                )
                self.observed_configs = pd.concat(
                    (self.observed_configs, _df)
                ).sort_index()
            else:
                if int(_rung) >= self.observed_configs.at[int(_config), "rung"]:
                    self.observed_configs.at[int(_config), "rung"] = int(_rung)
                    self.observed_configs.at[int(_config), "perf"] = perf
        # iterates over all pending evaluations and updates the list of observed
        # configs with the rung and performance as None
        for config_id, _ in pending_evaluations.items():
            _config, _rung = config_id.split("_")
            if int(_config) not in self.observed_configs.index:
                _df = pd.DataFrame(
                    [[None, int(_rung), None]],
                    columns=self.observed_configs.columns,
                    index=pd.Series(int(_config)),  # key for config_id
                )
                self.observed_configs = pd.concat(
                    (self.observed_configs, _df)
                ).sort_index()
            else:
                self.observed_configs.at[int(_config), "rung"] = int(_rung)
                self.observed_configs.at[int(_config), "perf"] = None

        # to account for incomplete evaluations from being promoted
        _observed_configs = self.observed_configs.copy().dropna(inplace=False)
        # iterates over the list of explored configs and buckets them to respective
        # rungs depending on the highest fidelity it was evaluated at
        self.rung_members = {k: [] for k in range(self.max_rung)}
        self.rung_members_performance = {k: [] for k in range(self.max_rung)}
        for _rung in _observed_configs.rung.unique():
            self.rung_members[_rung] = _observed_configs.index[
                _observed_configs.rung == _rung
            ].values
            self.rung_members_performance[_rung] = _observed_configs.perf[
                _observed_configs.rung == _rung
            ].values
        # identifying promotion list per rung
        self.rung_promotions = dict()
        for _rung in self.rung_map.keys():
            if _rung == self.max_rung:
                # cease promotions for the highest rung (configs at max budget)
                continue
            top_k = len(self.rung_members_performance[_rung]) // self.eta
            self.rung_promotions[_rung] = []

            # TODO make this dependent on the promotion policy
            if top_k > 0:
                if self.promotion_policy is None:
                    self.rung_promotions[_rung] = np.array(self.rung_members[_rung])[
                        np.argsort(self.rung_members_performance[_rung])[:top_k]
                    ].tolist()
                else:
                    promotion_kwargs = {}
                    self.rung_promotions[_rung] = self.promotion_policy.promote(
                        self.rung_members[_rung],
                        self.rung_members_performance[_rung],
                        **promotion_kwargs,
                    ).tolist()

        return

    def is_promotable(self) -> int | None:
        rung_to_promote = None
        for _rung in reversed(range(self.max_rung)):
            if len(self.rung_promotions[_rung]) > 0:
                rung_to_promote = _rung
                break
        return rung_to_promote

    def is_init_phase(self) -> bool:
        """Decides if optimization is still under the warmstart phase/model-based search.

        If `initial_design_type` is `max_budget`, the number of evaluations made at the
            target budget is compared to `initial_design_size`.
        If `initial_design_type` is `unique_configs`, the total number of unique
            configurations is compared to the `initial_design_size`.
        """
        _observed_configs = self.observed_configs.copy().dropna()
        if self.initial_design_type == "max_budget":
            val = (
                np.sum(_observed_configs.rung == self.max_rung)
                < self._initial_design_size
            )
        else:
            val = len(_observed_configs) <= self._initial_design_size
        return val

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:
        """...and this is the method that decides which point to query.

        Returns:
            [type]: [description]
        """
        rung_to_promote = self.is_promotable()
        if rung_to_promote is not None:
            # promote existing config
            # TODO implement the promotion rules?
            row = self.observed_configs.iloc[self.rung_promotions[rung_to_promote][0]]
            config = deepcopy(row["config"])
            rung = rung_to_promote + 1
            # assigning the fidelity to evaluate the config at
            config.fidelity.value = self.rung_map[rung]
            # updating config IDs
            previous_config_id = f"{row.name}_{rung_to_promote}"
            config_id = f"{row.name}_{rung}"
        else:
            if self.sampling_policy is None:
                config = self.pipeline_space.sample(
                    patience=self.patience, user_priors=True, ignore_fidelity=True
                )
                fidelity_value = self.rung_map[0]  # base rung is always 0
                config.fidelity.value = fidelity_value
                print("Sampled the config from pipeline space.")
            else:
                # TODO acquire arguments for the sampling policy
                sampling_args = {}
                config = self.sampling_policy.sample(**sampling_args)
                fidelity_value = self.rung_map[0]  # base rung is always 0
                config.fidelity.value = fidelity_value

            previous_config_id = None
            config_id = f"{len(self.observed_configs)}_0"

        print("Config:", config.hp_values())
        return config.hp_values(), config_id, previous_config_id  # type: ignore


if __name__ == "__main__":
    pass
