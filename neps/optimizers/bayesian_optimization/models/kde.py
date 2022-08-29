import logging
from copy import deepcopy
from typing import Iterable, List, Union

import numpy as np
import numpy.typing as npt
from statsmodels.nonparametric._kernel_base import _adjust_shape, gpke, kernel_func
from statsmodels.nonparametric.kernel_density import KDEMultivariate


def weighted_generalized_kernel_prod(
    bw: npt.ArrayLike,
    data: npt.ArrayLike,
    data_predict: npt.ArrayLike,
    var_type: List[str],
    data_weights: npt.ArrayLike = None,
    cont_kerneltype: str = 'gaussian',
    ordered_kerneltype: str = 'wangryzin',
    unordered_kerneltype: str = 'aitchisonaitken',
    to_sum: bool = True)
     -> np.ndarray:
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
            bw[dim], data[:, dim], data_predict[dim]) * data_weights / data_weights.mean()

    iscontinuous = np.array([c == 'c' for c in var_type])

    # pseudo-normalization of the density, seems to work so-so
    dens = K_val.prod(axis=1) / np.prod(bw[iscontinuous])
    if to_sum:
        return dens.sum(axis=0)
    else:
        return dens


class PriorWeightedKDE(KDEMultivariate):

    def __init__(self, data, var_type, num_values, bw=None, defaults=None, data_weights=None, prior=None, prior_weight=0, prior_as_samples=False, min_density=-np.inf):
        """
        Implementation of a Kernel Density Estimator with multivariate weighting functionality and
        possible prior incorporation.

        Args:
            data ([type]): [description]
            var_type ([type]): [description]
            bw ([type], optional): [description]. Defaults to None.
            defaults ([type], optional): [description]. Defaults to None.
            data_weights ([type], optional): [description]. Defaults to None.
            prior ([type], optional): [description]. Defaults to None.
            prior_as_samples (bool, optional): [description]. Defaults to False.
        """

        super().__init__(data, var_type, bw=None, defaults=None)

        if data_weights is None:
            self.data_weights = np.ones(data.shape[0])
        else:
            self.data_weights = np.asarray(data_weights)

        self.num_values = num_values
        self.prior = prior
        self.prior_weight = prior_weight
        self.prior_as_samples = prior_as_samples
        self.min_density = min_density
        if self.prior_as_samples:
            self.prior_enhanced_data = self._prior_enhance_data()
        else:
            self.prior_enhanced_data = self.data

    def pdf(self, data_predict=None):
        """
        Evaluate the probability density function.
        Parameters
        ----------
        data_predict : array_like, optional
            Points to evaluate at.  If unspecified, the training data is used.
        Returns
        -------
        pdf_est : array_like
            Probability density function evaluated at `data_predict`.
        Notes
        -----
        The probability density is given by the generalized product kernel
        estimator:
        .. math:: K_{h}(X_{i},X_{j}) =
            \prod_{s=1}^{q}h_{s}^{-1}k\left(\frac{X_{is}-X_{js}}{h_{s}}\right)
        """
        if data_predict is None:
            data_predict = self.data
        else:
            data_predict = _adjust_shape(data_predict, self.k_vars)

        pdf_est = []
        for i in range(np.shape(data_predict)[0]):
            pdf_est.append(weighted_generalized_kernel_prod(
                self.bw,
                data=self.data,
                data_predict=data_predict[i, :],
                data_weights=self.data_weights,
                var_type=self.var_type) / self.nobs)

        pdf_est = np.squeeze(pdf_est)
        return np.maximum(pdf_est, self.min_density)

    def _prior_enhance_data(self):
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
    print(kde_test.bw)
    print(kde_test.pdf(test_data))
