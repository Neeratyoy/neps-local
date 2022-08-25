from __future__ import annotations

from abc import abstractmethod

import numpy as np


class PromotionPolicy:
    """Base class for implementing a sampling straregy for SH and its subclasses"""

    def __init__(self, eta: int):
        self.rung_members: dict = {}
        self.rung_members_performance: dict = {}
        self.rung_promotions: dict = {}
        self.eta = eta

    def set_state(self, members: dict, performances: dict, **kwargs) -> None:
        self.rung_members = members
        self.rung_members_performance = performances

    @abstractmethod
    def retrieve_promotions(self) -> dict:
        raise NotImplementedError


class SyncPromotionPolicy(PromotionPolicy):
    """Implements a synchronous promotion from lower to higher fidelity.

    Promotes only when all predefined number of config slots are full.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config_map: dict = None

    def set_state(
        self, members: dict, performances: dict, config_map: dict, **kwargs
    ) -> None:
        super().set_state(members, performances)
        self.config_map = config_map

    def retrieve_promotions(self) -> dict:
        """Returns the top 1/eta configurations per rung if enough configurations seen"""
        assert self.config_map is not None
        max_rung = int(max(list(self.config_map.keys())))
        rung_promotions: dict = {}
        for rung in self.config_map.keys():
            if rung == max_rung:
                # cease promotions for the highest rung (configs at max budget)
                continue
            top_k = len(self.rung_members_performance[rung]) // self.eta
            if len(self.rung_members[rung]) >= self.config_map[rung]:
                _ordered_idx = np.argsort(self.rung_members_performance[rung])
                # stores the index of the top 1/eta configurations in the rung
                self.rung_promotions[rung] = np.array(self.rung_members[rung])[
                    _ordered_idx
                ][:top_k].tolist()
            else:
                # synchronous SH waits if each rung has not seen the budgeted configs
                rung_promotions[rung] = []
        return self.rung_promotions


class AsyncPromotionPolicy(PromotionPolicy):
    """Implements an asynchronous promotion from lower to higher fidelity.

    Promotes whenever a higher fidelity has at least eta configurations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def retrieve_promotions(self) -> dict:
        """Returns the top 1/eta configurations per rung if enough configurations seen"""
        max_rung = int(max(list(self.rung_map.keys())))
        rung_promotions: dict = {}
        for rung in self.config_map.keys():
            if rung == max_rung:
                # cease promotions for the highest rung (configs at max budget)
                continue
            # if less than eta configurations seen, no promotions occur as top_k=0
            top_k = len(self.rung_members_performance[rung]) // self.eta
            _ordered_idx = np.argsort(self.rung_members_performance[rung])
            self.rung_promotions[rung] = np.array(self.rung_members[rung])[_ordered_idx][
                :top_k
            ].tolist()
        return self.rung_promotions
