from __future__ import annotations

import logging
from copy import deepcopy
from typing import Iterable, List, Union, Any

import numpy as np
import numpy.typing as npt
from statsmodels.nonparametric._kernel_base import (
    _adjust_shape,
    gpke,
    kernel_func,
    GenericKDE,
    EstimatorSettings
)
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from metahyper.api import ConfigResult


def weighted_generalized_kernel_prod(
    bw: npt.ArrayLike,
    data: npt.ArrayLike,
    data_predict: npt.ArrayLike,
    var_type: List[str],
    data_weights: npt.ArrayLike = None,
    cont_kerneltype: str = 'gaussian',
    ordered_kerneltype: str = 'wangryzin',
    unordered_kerneltype: str = 'aitchisonaitken',
    to_sum: bool = True,
) -> np.ndarray:
    """
    Ripped straight from Statsmodels, with generalization to datapoint weights.
    Returns the non-normalized Generalized Product Kernel Estimator
    Parameters
    ----------
    bw : 1-D ndarray
        The user-specified bandwidth parameters.
    data : 1D or 2-D ndarray
        The training data.
    data_predict : 1-D ndarray
        The evaluation points at which the kernel estimation is performed.
    var_type : str, optional
        The variable type (continuous, ordered, unordered).
    cont_kerneltype : str, optional
        The kernel used for the continuous variables.
    ordered_kerneltype : str, optional
        The kernel used for the ordered discrete variables.
    unordered_kerneltype : str, optional
        The kernel used for the unordered discrete variables.
    to_sum : bool, optional
        Whether or not to sum the calculated array of densities.  Default is
        True.
    Returns
    -------
    dens : array_like
        The generalized product kernel density estimator.
    Notes
    -----
    The formula for the multivariate kernel estimator for the pdf is:
    .. math:: f(x)=\frac{1}{nh_{1}...h_{q}}\sum_{i=1}^
                        {n}K\left(\frac{X_{i}-x}{h}\right)
    where
    .. math:: K\left(\frac{X_{i}-x}{h}\right) =
                k\left( \frac{X_{i1}-x_{1}}{h_{1}}\right)\times
                k\left( \frac{X_{i2}-x_{2}}{h_{2}}\right)\times...\times
                k\left(\frac{X_{iq}-x_{q}}{h_{q}}\right)
    """
    kernel_types = dict(c=cont_kerneltype,
                        o=ordered_kerneltype, u=unordered_kerneltype)

    K_val = np.empty(data.shape)
    for dim, vtype in enumerate(var_type):
        func = kernel_func[kernel_types[vtype]]
        K_val[:, dim] = func(
            bw[dim], data[:, dim], data_predict[dim]) * data_weights

    iscontinuous = np.array([c == 'c' for c in var_type])

    # pseudo-normalization of the density, seems to work so-so
    dens = K_val.prod(axis=1) / np.prod(bw[iscontinuous])
    if to_sum:
        return dens.sum(axis=0)
    else:
        return dens


class MultiFidelityPriorWeightedKDE(GenericKDE):

    def __init__(self,
                 param_types,
                 num_values,
                 is_fidelity: list | None = None,
                 fixed_bw: list | None = None,
                 prior=None,
                 prior_weight: float = 0.0,
                 prior_as_samples: bool = False,
                 min_density: float = 1e-12,
                 **estimator_kwargs: dict
                 ):
        """
        Implementation of a Kernel Density Estimator with multivariate weighting functionality and
        possible prior incorporation.

        Args:
            var_type ([type]): [description]
            bw ([type], optional): [description]. Defaults to None.
            defaults ([type], optional): [description]. Defaults to None.
            data_weights ([type], optional): [description]. Defaults to None.
            prior ([type], optional): [description]. Defaults to None.
            prior_as_samples (bool, optional): [description]. Defaults to False.
        """
        # filter away the fitelity
        self.fixed_bw = fixed_bw

        if any(is_fidelity):
            self.fid_array = np.array(is_fidelity)
            self.param_types = np.array(param_types)[~self.fid_array]
            self.num_values = np.array(num_values)[~self.fid_array]
            if fixed_bw is not None:
                self.fixed_bw = np.array(fixed_bw)[~self.fid_array]

        self.prior = prior
        self.prior_weight = prior_weight
        self.prior_as_samples = prior_as_samples
        self.min_density = min_density
        self.estimator_kwargs = estimator_kwargs or {}

    def fit(self, configs: List[ConfigResult], config_weights: List[float] = None):

        if config_weights is None:
            data_weights = np.ones(len(configs))
        else:
            data_weights = np.asarray(config_weights)

        if self.prior_as_samples:
            # Now, we resample from the prior each time we fit - not sure if it's a good idea or not
            # May want to have these samples static
            configs, data_weights = self._prior_enhance_data(
                configs, data_weights)

        self.data = self._convert_configs_to_numpy(configs)
        self.data_weights = data_weights / np.mean(data_weights)

        # These are attributes to fit within the statsmodels KDE framework
        # https://github.com/statsmodels/statsmodels/blob/b79d71862dd9ca30ed173c9ad9b96a18e48d8dbb/statsmodels/nonparametric/_kernel_base.py#L99
        # TODO build own KDE if we see the need

        # the defaults have to be set with each fit call, since we get new data for each one
        defaults = EstimatorSettings(**self.estimator_kwargs)
        self._set_defaults(defaults)

        self.nobs, self.k_vars = np.shape(self.data)
        if not self.efficient:
            self.bw = self._compute_bw(self.fixed_bw)
        else:
            self.bw = self._compute_efficient(self.fixed_bw)

    def _convert_configs_to_numpy(self, configs, drop_fidelity=True):
        """Creates a N x D normalized numpy array for N configurations of dimensionality D. Does not
        support graphs.

        Args:
            configs ([type]):

        Returns:
            np.ndarray: N x D normalized numpy array of configs
        """
        configs_np = np.array(
            [[x_.normalized().value for x_ in list(x.values())] for x in configs])
        return configs_np[:, ~self.fid_array]

    def pdf(self, configs):
        """Evaluates the combined probability density function of the KDE and the prior. If the prior
        is not specified as samples, the linear combination of the prior and KDE is queried. Otherwise,
        only the KDE is queried.
        """
        # TODO incorporate querying the prior?
        return self._pdf(self._convert_configs_to_numpy(configs))

    def _pdf(self, X):
        """
        Evaluate the probability density function. Operates on a numpy array
        Parameters
        ----------
        X : array_like, optional
            Points to evaluate at.
        Returns
        -------
        pdf_est : array_like
            Probability density function evaluated at `X`.
        Notes
        -----
        The probability density is given by the generalized product kernel
        estimator:
        .. math:: K_{h}(X_{i},X_{j}) =
            \prod_{s=1}^{q}h_{s}^{-1}k\left(\frac{X_{is}-X_{js}}{h_{s}}\right)
        """
        X = _adjust_shape(X, self.k_vars)

        pdf_est = []
        for i in range(np.shape(X)[0]):
            # TODO add in the number of values for each of the non-continous variables
            pdf_est.append(weighted_generalized_kernel_prod(
                self.bw,
                data=self.data,
                data_predict=X[i, :],
                data_weights=self.data_weights,
                var_type=self.param_types) / self.nobs)

        pdf_est = np.squeeze(pdf_est)
        return np.maximum(pdf_est, self.min_density)

    def _prior_enhance_data(self, configs):
        # TODO implement this properly
        return configs


if __name__ == "__main__":

    some_placeholder_space = dict(
        learning_rate=neps.FloatParameter(
            lower=1e-4, upper=1e0, log=True, default=1e-1, default_confidence="high"
        ),
        categorical=neps.CategoricalParameter(choices=["catA", "catB"]),
        integer=neps.IntegerParameter(lower=2, upper=5, log=True),
        epoch=neps.IntegerParameter(lower=1, upper=100, is_fidelity=True),
    )
    norm_consts = np.array([15, 1, 15, 75])
    train_data = np.array(
        [
            [3, 1, 7, 1],
            [2, 0, 5, 75],
            [2, 1, 3, 75],
            [1, 0, 6, 1],
            [0.5, 1, 7, 11],
            [1, 0, 11, 75],
            [15, 1, 15, 15]
        ]
    ) / norm_consts
    test_data = np.array([[0.5, 1, 7, 75], [8, 0, 15, 1],
                         [15, 1, 15, 15]]) / norm_consts
    kde_test = PriorWeightedKDE(train_data, [
                                'c', 'u', 'o', 'c'], num_values=None, data_weights=[1, 1, 1, 1, 1, 1, 0.1])
