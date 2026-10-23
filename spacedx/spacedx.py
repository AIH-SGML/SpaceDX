"""SpaceDX: Spatial Differential Expression models with attention mechanisms.

This module implements variational inference models for spatial transcriptomics
data analysis. It includes:

- SpaceDX: A multiple instance learning model with learned attention weights
  for identifying spatial differential gene expression patterns.

- UniformAttentionBaseline: A pseudobulk baseline model with uniform attention
  for comparison.

Both models use variational inference with hierarchical priors, optimized via Evidence Lower Bound (ELBO) maximization.
"""

import sys

sys.path.append("/.")

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.distributions.normal import Normal
from spacedx.utils import get_weight_act, BinomialLik, kl_divergence_normal
import tqdm
import time
from torchmin import Minimizer
from functools import cached_property

torch.set_default_dtype(torch.float64)


@dataclass
class HyperpriorConfig:
    """Configuration for hyperprior parameters in the hierarchical model.

    The prior standard deviations of the effect size parameters (beta) and
    attention weights (gamma) are parameterized as softplus(lvar), where lvar
    is an unconstrained variable. The hyperprior places a Normal distribution
    on lvar (i.e. on the pre-softplus prior standard deviation).

    Attributes:
        beta_m: Mean of the hyperprior on lvar_beta, the pre-softplus prior std
            of beta. Larger values imply larger expected effect sizes. Default: 2.0.
        beta_s: Standard deviation of the hyperprior on lvar_beta.
            Controls uncertainty in the prior std of beta. Default: 0.5.
        gamma_m: Mean of the hyperprior on lvar_gamma, the pre-softplus prior
            std of gamma. Larger values allow more variable attention weights.
            Default: 1.0.
        gamma_s: Standard deviation of the hyperprior on lvar_gamma.
            Controls uncertainty in the prior std of gamma. Default: 0.2.
    """

    beta_m: float = 2.0
    beta_s: float = 0.5
    gamma_m: float = 1.0
    gamma_s: float = 0.2


class _BaseModel(nn.Module):
    """Base class for spatial differential expression models.

    This abstract base class provides shared functionality for variational
    inference models, including:
    - Fixed effect parameters (alpha)
    - Random effect parameters for effect sizes (beta) with hierarchical priors
    - Optimization routines (L-BFGS, BFGS, Adam)
    - Monte Carlo sampling for ELBO estimation

    Subclasses must implement `elbo()` and `iw_elbo()` methods.

    Attributes:
        y: Binary phenotype tensor of shape (n_samples,).
        F: Fixed effect design matrix of shape (n_samples, n_fixed_effects).
        xs: List of gene expression vector, one per spatial region/sample.
        R: Optional repeat/aggregation matrix for grouping samples belonging to same biological replicate.
        lik: Likelihood function (e.g., BinomialLik).
        optimizer: Optimization method ('lbfgs', 'bfgs', or 'adam').
        hyperprior: HyperpriorConfig for hierarchical priors.
    """

    def __init__(
        self,
        y,
        F,
        xs,
        R=None,
        n_samples=128,
        seed=0,
        lik=BinomialLik(),
        optimizer="bfgs",
        hyperprior=None,
    ):
        """Initialize the base model.

        Args:
            y: Binary phenotype tensor of shape (n_samples,)
            F: Fixed effect design matrix of shape (n_samples, n_fixed_effects).
            xs: List of gene expression vectors, where xs[k] has shape
                (n_spots_k, 1).
            R: Optional repeat/aggregation matrix for grouping samples belonging to same biological replicate. Default: None.
            n_samples: Number of Monte Carlo samples for ELBO estimation.
                More samples reduce variance but increase computation. Default: 128.
            seed: Random seed for reproducibility. Default: 0.
            lik: Likelihood function object with `log_prob` and `predict` methods.
                Default: BinomialLik().
            optimizer: Optimization algorithm. One of 'lbfgs' (default),
                'bfgs', or 'adam'.
            hyperprior: HyperpriorConfig object specifying prior parameters.
                If None, uses default HyperpriorConfig().
        """
        super().__init__()
        self.hyperprior = hyperprior or HyperpriorConfig()
        self.y = y
        self.F = F
        self.xs = xs
        self.R = R
        self._dtype = y.dtype
        self.lik = lik
        self.optimizer = optimizer

        # Intercept
        self._alpha = nn.Parameter(1e-3 * torch.randn(F.shape[1], dtype=self._dtype))

        # Beta random effects (shared by SpaceDX and PseudoBulk)
        self._beta_m = nn.Parameter(
            1e-3 * torch.randn(xs[0].shape[1], dtype=self._dtype)
        )
        self._log_beta_s = nn.Parameter(
            1e-3 * torch.randn(xs[0].shape[1], dtype=self._dtype)
        )

        # Beta prior std, pre-softplus (variational params, regularized by hyperprior)
        self._lvar_beta_m = nn.Parameter(
            self.hyperprior.beta_m * torch.ones(1, dtype=self._dtype)
        )
        self._log_lvar_beta_s = nn.Parameter(
            np.log(self.hyperprior.beta_s) * torch.ones(1, dtype=self._dtype)
        )

        # Sample noise
        self._sample_eps(n_samples, seed)

    def _sample_eps(self, n_samples=128, seed=42):
        """Sample standard normal noise for reparameterization trick.

        Pre-samples epsilon values used in the reparameterization trick for
        variational inference. This allows gradients to flow through the
        sampling process.

        Args:
            n_samples: Number of Monte Carlo samples to draw. Default: 128.
            seed: Random seed for reproducibility. Default: 42.

        Note:
            Sets `_eps_beta` with shape (n_features, n_samples) and
            `_eps_lvar_beta` with shape (1, n_samples).
        """
        torch.manual_seed(seed)
        self._eps_beta = torch.randn(self.xs[0].shape[1], n_samples)
        self._eps_lvar_beta = torch.randn(1, n_samples)

    def optimize(self, max_iter=500, factr=1e7, verbose=False):
        """Optimize model parameters by maximizing the ELBO.

        Performs variational inference by optimizing the Evidence Lower Bound
        (ELBO) using the specified optimizer. Supports quasi-Newton methods
        (L-BFGS, BFGS) and gradient descent (Adam).

        Args:
            max_iter: Maximum number of optimization iterations. Default: 500.
            factr: Factor for convergence tolerance. The iteration stops when
                (loss_k - loss_{k+1}) < factr * machine_epsilon.
                Default: 1e7 (moderate tolerance).
            verbose: If True, displays progress bar and convergence info.
                Default: False.

        Returns:
            Tuple of (loss, gradient, converged):
                - loss: Final negative ELBO value (torch.Tensor).
                - gradient: Concatenated gradient of all parameters (torch.Tensor).
                - converged: Boolean indicating if convergence criterion was met.

        Raises:
            ValueError: If optimizer is not one of 'lbfgs', 'bfgs', or 'adam'.
        """
        ftol = factr * np.finfo(float).eps

        if self.optimizer == "lbfgs":
            optim = Minimizer(
                self.parameters(),
                method="l-bfgs",
                options={"line_search": "strong-wolfe"},
            )
        elif self.optimizer == "bfgs":
            optim = Minimizer(
                self.parameters(),
                method="bfgs",
                options={"line_search": "strong-wolfe"},
            )
        elif self.optimizer == "adam":
            optim = torch.optim.Adam(self.parameters(), lr=0.1)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        def closure():
            optim.zero_grad()
            loss = -self.elbo()
            loss.backward(retain_graph=True)
            return loss

        def grad():
            optim.zero_grad()
            loss = -self.elbo()
            loss.backward()
            return torch.cat([p.grad.flatten() for p in self.parameters()])

        t0 = time.time()
        conv = False
        previous_loss = -self.elbo()
        iterator = (
            tqdm.tqdm(range(max_iter), desc=f"Optimize {self.__class__.__name__}")
            if verbose
            else range(max_iter)
        )

        for _iter in iterator:
            loss = optim.step(closure)
            if _iter > 2 and (loss.item() - previous_loss.item()) < ftol:
                conv = True
                if verbose:
                    print(f"Converged after {_iter} steps")
                break
            previous_loss = loss

        if verbose:
            print("Elapsed:", time.time() - t0)

        return loss, grad(), conv

    def elbo(self):
        """Compute the Evidence Lower Bound (ELBO).

        The ELBO is the objective function for variational inference,
        consisting of the expected log-likelihood minus KL divergences
        between variational posteriors and priors.

        Returns:
            torch.Tensor: Scalar ELBO value.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _compute_iw_elbo(self, K, S):
        """Compute the Importance Weighted ELBO (IW-ELBO).

        The IW-ELBO provides a tighter lower bound on the log marginal
        likelihood than the standard ELBO, useful for model evaluation.

        Args:
            K: Number of importance samples per data point.
            S: Number of sample sets (total samples = K * S).

        Returns:
            torch.Tensor: IW-ELBO values of shape (S,).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    # Shared properties
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha.data = value

    @property
    def beta_m(self):
        return self._beta_m

    @property
    def beta_posterior_m(self):
        return self._beta_m.item()

    @beta_m.setter
    def beta_m(self, value):
        self._beta_m.data = value

    @property
    def beta_s(self):
        return torch.exp(self._log_beta_s) + 1e-4

    @beta_s.setter
    def beta_s(self, value):
        self._log_beta_s.data = torch.log(value)

    @property
    def lvar_beta_m(self):
        return self._lvar_beta_m

    @property
    def lvar_beta_s(self):
        return torch.exp(self._log_lvar_beta_s) + 1e-4

    @property
    def loglik(self):
        return self._loglik.item()

    @property
    def kld_beta(self):
        return self._kld_beta.item()

    @property
    def kld_lvar_beta(self):
        return self._kld_lvar_beta.item()

    @cached_property
    def iw_elbo(self):
        return self._compute_iw_elbo().mean().item()


class SpaceDX(_BaseModel):
    """Spatial Differential Expression model with learned attention weights.

    SpaceDX is a multiple instance learning (MIL) model for spatial
    transcriptomics that learns cell-level attention weights to identify
    which cells contribute most to sample-level phenotypes. The model uses:

    - Learned attention weights (gamma) to aggregate cell-level features
    - Effect size parameters (beta) linking aggregated features to outcomes
    - Hierarchical priors on both gamma and beta for regularization
    - Variational inference for posterior estimation

    The model is particularly useful for:
    - Identifying spatially varying gene expression patterns
    - Understanding which cell types/states drive phenotypic differences
    - Spatial differential expression analysis in heterogeneous tissues

    Attributes:
        Ws: List of weight feature matrices for computing attention.
        weight_act: Activation function for attention weights (e.g., softmax).
        gamma_m: Variational mean of attention weight parameters.
        gamma_s: Variational std of attention weight parameters.
    """

    def __init__(
        self,
        y,
        F,
        xs,
        Ws=None,
        R=None,
        n_samples=128,
        seed=0,
        lik=BinomialLik(),
        weight_act="softmax",
        optimizer="bfgs",
        hyperprior=None,
    ):
        """Initialize the SpaceDX model.

        Args:
            y: Target variable tensor of shape (n_samples,).
            F: Fixed effect design matrix of shape (n_samples, n_fixed_effects).
            xs: List of K feature matrices, where xs[k] has shape
                (n_cells_k, n_features). Cell-level features to be aggregated.
            Ws: List of K weight feature matrices for attention computation.
                If None, uses xs for both features and attention. Each Ws[k]
                has shape (n_cells_k, n_weight_features). Default: None.
            R: Optional repeat matrix of shape (n_aggregated, K) for grouping
                regions. Default: None.
            n_samples: Number of Monte Carlo samples for ELBO. Default: 128.
            seed: Random seed for reproducibility. Default: 0.
            lik: Likelihood function. Default: BinomialLik().
            weight_act: Attention activation function. One of 'softmax',
                'sigmoid', etc. Default: 'softmax'.
            optimizer: Optimization algorithm ('lbfgs', 'bfgs', 'adam').
                Default: 'lbfgs'.
            hyperprior: HyperpriorConfig for hierarchical priors. Default: None.
        """
        super().__init__(y, F, xs, R, n_samples, seed, lik, optimizer, hyperprior)

        self.Ws = Ws if Ws is not None else xs
        self.weight_act = get_weight_act(weight_act, dim=0)

        self._init_gamma_params()
        self._sample_gamma_eps(n_samples, seed)

    def _init_gamma_params(self):
        """Initialize gamma (attention) parameters and their hierarchical priors.

        Creates variational parameters for:
        - gamma: Attention weight parameters (mean and log-std)
        - lvar_gamma: Pre-softplus prior std of gamma (mean and log-std)
        """
        # Gamma random effects (attention weights)
        self._gamma_m = nn.Parameter(
            1e-3 * torch.randn(self.Ws[0].shape[1], dtype=self._dtype)
        )
        self._log_gamma_s = nn.Parameter(
            1e-3 * torch.randn(self.Ws[0].shape[1], dtype=self._dtype)
        )

        # Gamma prior std, pre-softplus (variational params, regularized by hyperprior)
        self._lvar_gamma_m = nn.Parameter(
            self.hyperprior.gamma_m * torch.ones(1, dtype=self._dtype)
        )
        self._log_lvar_gamma_s = nn.Parameter(
            np.log(self.hyperprior.gamma_s) * torch.ones(1, dtype=self._dtype)
        )

    def _sample_gamma_eps(self, n_samples=128, seed=42):
        """Sample standard normal noise for gamma reparameterization.

        Args:
            n_samples: Number of Monte Carlo samples. Default: 128.
            seed: Random seed (offset by 1 from beta seed). Default: 42.
        """
        torch.manual_seed(seed + 1)  # Different seed for gamma
        self._eps_gamma = torch.randn(self.Ws[0].shape[1], n_samples)
        self._eps_lvar_gamma = torch.randn(1, n_samples)

    def _get_xmil(self, gamma, xs=None, Ws=None, R=None):
        """Compute attention-weighted aggregation of cell features (MIL pooling).

        For each region, computes a weighted sum of cell features where weights
        are determined by the attention mechanism parameterized by gamma.

        Args:
            gamma: Attention parameters of shape (n_weight_features, n_mc_samples).
            xs: List of feature matrices. If None, uses self.xs.
            Ws: List of weight feature matrices. If None, uses self.Ws.
            R: Repeat/aggregation matrix. If None, uses self.R.

        Returns:
            torch.Tensor: Aggregated features of shape (n_samples, n_mc_samples),
                standardized to zero mean and unit variance per MC sample.

        Note:
            The aggregation formula for region k is:
            xmil[k] = sum_i(w_i * x_i) where w_i = softmax(W_i @ gamma)
        """
        xs = xs if xs is not None else self.xs
        Ws = Ws if Ws is not None else self.Ws
        R = R if R is not None else self.R

        xmil = torch.zeros([len(xs), gamma.shape[1]])
        for ix, (x, W) in enumerate(zip(xs, Ws)):
            w = self.weight_act(W.mm(gamma))
            xmil[ix] = torch.einsum("is,ik->s", w, x)

        if R is not None:
            xmil = torch.mm(R, xmil)

        xmil = (xmil - xmil.mean(0)) / (xmil.std(0) + 1e-4)
        return xmil

    # Gamma properties
    @property
    def gamma_m(self):
        return self._gamma_m + 1e-4

    @gamma_m.setter
    def gamma_m(self, value):
        self._gamma_m.data = value

    @property
    def gamma_s(self):
        return torch.exp(self._log_gamma_s) + 1e-4

    @gamma_s.setter
    def gamma_s(self, value):
        self._log_gamma_s.data = torch.log(value)

    @property
    def lvar_gamma_m(self):
        return self._lvar_gamma_m

    @property
    def lvar_gamma_s(self):
        return torch.exp(self._log_lvar_gamma_s) + 1e-4

    @property
    def kld_gamma(self):
        return self._kld_gamma.item()

    @property
    def kld_lvar_gamma(self):
        return self._kld_lvar_gamma.item()

    def get_raw_weights(self, Ws=None):
        """Compute raw (pre-activation) attention weights for all cells.

        Returns the linear attention scores before applying the activation
        function (e.g., before softmax normalization).

        Args:
            Ws: List of weight feature matrices. If None, uses self.Ws.

        Returns:
            Tuple of (mean, std):
                - mean: Mean raw weights across MC samples, shape (n_total_cells,).
                - std: Std of raw weights across MC samples, shape (n_total_cells,).
        """
        gamma = self.gamma_m[:, None] + self.gamma_s[:, None] * self._eps_gamma
        Ws = Ws if Ws is not None else self.Ws

        uwm, uws = [], []
        for W in Ws:
            _uw = self.weight_act.raw_weights(W.mm(gamma))
            uwm.append(_uw.mean(1))
            uws.append(_uw.std(1))
        return torch.cat(uwm, 0), torch.cat(uws, 0)

    def get_attention_weights(self, Ws=None):
        """Compute normalized attention weights for all cells.

        Returns attention weights after applying the activation function
        (e.g., after softmax normalization within each region).

        Args:
            Ws: List of weight feature matrices. If None, uses self.Ws.

        Returns:
            Tuple of (mean, std):
                - mean: Mean attention weights across MC samples, shape (n_total_cells,).
                - std: Std of attention weights across MC samples, shape (n_total_cells,).

        Note:
            For softmax activation, weights sum to 1 within each region.
        """
        gamma = self.gamma_m[:, None] + self.gamma_s[:, None] * self._eps_gamma
        Ws = Ws if Ws is not None else self.Ws

        wm, ws = [], []
        for W in Ws:
            _w = self.weight_act(W.mm(gamma))
            wm.append(_w.mean(1))
            ws.append(_w.std(1))
        return torch.cat(wm, 0).detach().numpy(), torch.cat(ws, 0).detach().numpy()

    def predict(self, xs=None, F=None):
        """Generate predictions for new or existing data.

        Computes posterior predictive distribution by averaging predictions
        across Monte Carlo samples from the variational posterior.

        Args:
            xs: List of feature matrices for prediction. If None, uses self.xs.
            F: Fixed effect design matrix. If None, uses self.F.

        Returns:
            Tuple of (mean, std):
                - mean: Posterior predictive mean, shape (n_samples,).
                - std: Posterior predictive std, shape (n_samples,).
        """
        gamma = self.gamma_m[:, None] + self.gamma_s[:, None] * self._eps_gamma
        xmil = self._get_xmil(gamma, xs=xs)
        F = F if F is not None else self.F
        logits = F.mm(self.alpha[:, None]) + xmil
        ystar = self.lik.predict(logits)
        return ystar.mean(1), ystar.std(1)

    def elbo(self):
        """Compute the Evidence Lower Bound (ELBO) for SpaceDX.

        The ELBO decomposes into:
        - Expected log-likelihood under the variational posterior
        - KL divergence for beta (effect size parameters)
        - KL divergence for gamma (attention parameters)
        - KL divergence for hierarchical prior parameters

        Side effects:
            Updates internal attributes: _loglik, _kld_beta, _kld_gamma,
            _kld_lvar_beta, _kld_lvar_gamma for monitoring.

        Returns:
            torch.Tensor: Scalar ELBO value (higher is better).
        """
        beta = self.beta_m[:, None] + self.beta_s[:, None] * self._eps_beta
        gamma = self.gamma_m[:, None] + self.gamma_s[:, None] * self._eps_gamma

        logits = self.F.mm(self.alpha[:, None]) + self._get_xmil(gamma) * beta
        loglik = self.lik.log_prob(logits, self.y)

        lvar_beta = (
            self.lvar_beta_m[:, None] + self.lvar_beta_s[:, None] * self._eps_lvar_beta
        )
        lvar_gamma = (
            self.lvar_gamma_m[:, None]
            + self.lvar_gamma_s[:, None] * self._eps_lvar_gamma
        )

        kld_beta = (
            kl_divergence_normal(
                self.beta_m[:, None],
                self.beta_s[:, None],
                torch.zeros_like(self.beta_m)[:, None],
                F.softplus(lvar_beta) + 1e-4,
            )
            .mean(1)
            .sum()
        )

        kld_gamma = (
            kl_divergence_normal(
                self.gamma_m[:, None],
                self.gamma_s[:, None],
                torch.zeros_like(self.gamma_m)[:, None],
                F.softplus(lvar_gamma) + 1e-4,
            )
            .mean(1)
            .sum()
        )

        kld_lvar_beta = kl_divergence_normal(
            self.lvar_beta_m,
            self.lvar_beta_s,
            self.hyperprior.beta_m * torch.ones(1),
            self.hyperprior.beta_s * torch.ones(1),
        ).sum()

        kld_lvar_gamma = kl_divergence_normal(
            self.lvar_gamma_m,
            self.lvar_gamma_s,
            self.hyperprior.gamma_m * torch.ones(1),
            self.hyperprior.gamma_s * torch.ones(1),
        ).sum()

        elbo = loglik - kld_beta - kld_gamma - kld_lvar_beta - kld_lvar_gamma

        self._loglik = loglik
        self._kld_beta = kld_beta
        self._kld_gamma = kld_gamma
        self._kld_lvar_beta = kld_lvar_beta
        self._kld_lvar_gamma = kld_lvar_gamma

        return elbo

    def _compute_iw_elbo(self, K=32, S=128):
        """Compute Importance Weighted ELBO for model evaluation.

        The IW-ELBO uses K importance samples to provide a tighter bound
        on the log marginal likelihood. As K -> infinity, IW-ELBO converges
        to the true log marginal likelihood.

        Args:
            K: Number of importance samples per evaluation.
            S: Number of independent IW-ELBO estimates to compute.

        Returns:
            torch.Tensor: S IW-ELBO estimates, shape (S,). Use for model
                comparison or computing confidence intervals.

        Note:
            Draws fresh samples (does not use pre-sampled epsilon).
        """
        n_samples = K * S
        _eps_beta = torch.randn(self.xs[0].shape[1], n_samples)
        _eps_gamma = torch.randn(self.Ws[0].shape[1], n_samples)
        _eps_lvar_beta = torch.randn(1, n_samples)
        _eps_lvar_gamma = torch.randn(1, n_samples)

        beta = self.beta_m[:, None] + self.beta_s[:, None] * _eps_beta
        gamma = self.gamma_m[:, None] + self.gamma_s[:, None] * _eps_gamma

        logits = self.F.mm(self.alpha[:, None]) + self._get_xmil(gamma) * beta
        loglik = self.lik.log_prob(logits, self.y, reduce=False).sum(0)

        lvar_beta = (
            self.lvar_beta_m[:, None] + self.lvar_beta_s[:, None] * _eps_lvar_beta
        )
        lvar_gamma = (
            self.lvar_gamma_m[:, None] + self.lvar_gamma_s[:, None] * _eps_lvar_gamma
        )

        log_p_over_q_beta = (
            Normal(
                torch.zeros_like(self.beta_m)[:, None], F.softplus(lvar_beta)
            ).log_prob(beta)
            - Normal(self.beta_m[:, None], self.beta_s[:, None]).log_prob(beta)
        ).sum(0)

        log_p_over_q_gamma = (
            Normal(
                torch.zeros_like(self.gamma_m)[:, None], F.softplus(lvar_gamma)
            ).log_prob(gamma)
            - Normal(self.gamma_m[:, None], self.gamma_s[:, None]).log_prob(gamma)
        ).sum(0)

        log_p_over_q_lvar = (
            Normal(
                self.hyperprior.beta_m * torch.ones(1, 1),
                self.hyperprior.beta_s * torch.ones(1, 1),
            )
            .log_prob(lvar_beta)
            .sum(0)
            + Normal(
                self.hyperprior.gamma_m * torch.ones(1, 1),
                self.hyperprior.gamma_s * torch.ones(1, 1),
            )
            .log_prob(lvar_gamma)
            .sum(0)
            - Normal(self.lvar_beta_m[:, None], self.lvar_beta_s[:, None])
            .log_prob(lvar_beta)
            .sum(0)
            - Normal(self.lvar_gamma_m[:, None], self.lvar_gamma_s[:, None])
            .log_prob(lvar_gamma)
            .sum(0)
        )

        logw = loglik + log_p_over_q_beta + log_p_over_q_gamma + log_p_over_q_lvar
        return torch.logsumexp(logw.reshape(K, S), dim=0) - np.log(K)


class UniformAttentionBaseline(_BaseModel):
    """Pseudobulk baseline model with uniform (equal) attention weights.

    This model serves as a baseline comparison for SpaceDX. Instead of learning
    cell-specific attention weights, it uses uniform weighting (simple averaging)
    to aggregate cell features into sample-level pseudobulk features.

    The model is equivalent to standard bulk analysis approaches and does not
    capture spatial or cell-type-specific variation in gene expression effects.

    Use this model to:
    - Establish baseline performance for comparison with SpaceDX
    - Test whether learned attention provides benefit over uniform weighting
    - Perform traditional pseudobulk differential expression analysis

    Attributes:
        Inherits all attributes from _BaseModel.
        Does not have gamma (attention) parameters.
    """

    def __init__(
        self,
        y,
        F,
        xs,
        R=None,
        n_samples=128,
        seed=0,
        lik=BinomialLik(),
        optimizer="bfgs",
        hyperprior=None,
    ):
        """Initialize the UniformAttentionBaseline model.

        Args:
            y: Target variable tensor of shape (n_samples,).
            F: Fixed effect design matrix of shape (n_samples, n_fixed_effects).
            xs: List of K feature matrices, where xs[k] has shape
                (n_cells_k, n_features). Will be averaged uniformly.
            R: Optional repeat matrix for grouping regions. Default: None.
            n_samples: Number of Monte Carlo samples for ELBO. Default: 128.
            seed: Random seed for reproducibility. Default: 0.
            lik: Likelihood function. Default: BinomialLik().
            optimizer: Optimization algorithm. Default: 'lbfgs'.
            hyperprior: HyperpriorConfig for hierarchical priors. Default: None.
        """
        super().__init__(y, F, xs, R, n_samples, seed, lik, optimizer, hyperprior)

    def _get_pseudobulk(self, xs=None, R=None):
        """Compute pseudobulk features by uniform averaging.

        Aggregates cell-level features into sample-level features using
        simple mean (uniform attention weights).

        Args:
            xs: List of feature matrices. If None, uses self.xs.
            R: Repeat/aggregation matrix. If None, uses self.R.

        Returns:
            torch.Tensor: Pseudobulk features of shape (n_samples, 1),
                standardized to zero mean and unit variance.

        Note:
            Unlike SpaceDX._get_xmil, this uses equal weights for all cells.
        """
        xs = xs if xs is not None else self.xs
        R = R if R is not None else self.R

        pb = torch.zeros([len(xs)])[:, None]
        for ix, x in enumerate(xs):
            pb[ix] = x.mean(0)
        if R is not None:
            pb = torch.mm(R, pb)
        pb = (pb - pb.mean(0)) / pb.std(0)
        return pb

    def predict(self, xs=None, F=None):
        """Generate predictions (not yet implemented for baseline).

        Args:
            xs: List of feature matrices for prediction.
            F: Fixed effect design matrix.

        Returns:
            None (not implemented).
        """
        pass

    def elbo(self):
        """Compute the Evidence Lower Bound (ELBO) for pseudobulk model.

        Simpler than SpaceDX ELBO as there are no gamma (attention) parameters.
        Only includes KL divergences for beta and its hierarchical prior.

        Side effects:
            Updates internal attributes: _loglik, _kld_beta, _kld_lvar_beta.

        Returns:
            torch.Tensor: Scalar ELBO value (higher is better).
        """
        beta = self.beta_m[:, None] + self.beta_s[:, None] * self._eps_beta

        logits = self.F.mm(self.alpha[:, None]) + self._get_pseudobulk() * beta
        loglik = self.lik.log_prob(logits, self.y)

        lvar_beta = (
            self.lvar_beta_m[:, None] + self.lvar_beta_s[:, None] * self._eps_lvar_beta
        )

        kld_beta = (
            kl_divergence_normal(
                self.beta_m[:, None],
                self.beta_s[:, None],
                torch.zeros_like(self.beta_m)[:, None],
                F.softplus(lvar_beta),
            )
            .mean(1)
            .sum()
        )

        kld_lvar_beta = kl_divergence_normal(
            self.lvar_beta_m,
            self.lvar_beta_s,
            self.hyperprior.beta_m * torch.ones(1),
            self.hyperprior.beta_s * torch.ones(1),
        ).sum()

        elbo = loglik - kld_beta - kld_lvar_beta

        self._loglik = loglik
        self._kld_beta = kld_beta
        self._kld_lvar_beta = kld_lvar_beta

        return elbo

    def _compute_iw_elbo(self, K=32, S=128):
        """Compute Importance Weighted ELBO for pseudobulk model evaluation.

        Simpler than SpaceDX IW-ELBO as there are no gamma parameters.

        Args:
            K: Number of importance samples per evaluation.
            S: Number of independent IW-ELBO estimates to compute.

        Returns:
            torch.Tensor: S IW-ELBO estimates, shape (S,).
        """
        n_samples = K * S
        _eps_beta = torch.randn(self.xs[0].shape[1], n_samples)
        _eps_lvar_beta = torch.randn(1, n_samples)

        beta = self.beta_m[:, None] + self.beta_s[:, None] * _eps_beta

        logits = self.F.mm(self.alpha[:, None]) + self._get_pseudobulk() * beta
        loglik = self.lik.log_prob(logits, self.y, reduce=False).sum(0)

        lvar_beta = (
            self.lvar_beta_m[:, None] + self.lvar_beta_s[:, None] * _eps_lvar_beta
        )

        log_p_over_q_beta = (
            Normal(
                torch.zeros_like(self.beta_m)[:, None], F.softplus(lvar_beta)
            ).log_prob(beta)
            - Normal(self.beta_m[:, None], self.beta_s[:, None]).log_prob(beta)
        ).sum(0)

        log_p_over_q_lvar = Normal(
            self.hyperprior.beta_m * torch.ones(1, 1),
            self.hyperprior.beta_s * torch.ones(1, 1),
        ).log_prob(lvar_beta).sum(0) - Normal(
            self.lvar_beta_m[:, None], self.lvar_beta_s[:, None]
        ).log_prob(
            lvar_beta
        ).sum(
            0
        )

        logw = loglik + log_p_over_q_beta + log_p_over_q_lvar
        return torch.logsumexp(logw.reshape(K, S), dim=0) - np.log(K)
