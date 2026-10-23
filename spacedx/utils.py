import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Binomial
from scipy.spatial import distance
from scipy.sparse import issparse
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pdb


# Preprocessing functions

def spatial_smoothing(xs, A, k_hops):
    xs_spatial = []
    
    for _xs, _A in zip(xs, A):
        for _ in range(k_hops):
            _xs = _A.mm(_xs)
        xs_spatial.append(_xs)

    return xs_spatial

def prepare_general_model_input(idata, latent, k_hops=1):
    '''
    Prepare and return the general (gene-agnostic) input features.

            Parameters:
                    idata (AnnData): Input AnnData object. Must contain:
                        - idata.obs["sample"]: sample-level annotations
                        - idata.obs["group"]: integer group/condition labels
                        - idata.obs["biological_replicate"]: replicate annotations
                        - idata.obsm[latent]: latent embedding matrix (e.g. from PCA or scVI)
                        - idata.obsm["spatial"]: spatial coordinates
                        - idata.uns["spatial"][sample]["scalefactors"]["spot_diameter_fullres"]:
                            spot diameter used for building the spatial neighbor graph
                    latent (str): Key in idata.obsm to use as latent embedding
                        (e.g. "X_pca" or "X_scVI"). The first 10 dimensions are
                        selected and standardized.
                    k_hops (int): Number of hops for spatial smoothing of the
                        latent embeddings via the adjacency matrix. Default: 1.

            Returns:
                    y (torch.Tensor): Binary group labels aggregated to biological
                        replicate level, shape (n_replicates, 1).
                    F (torch.Tensor): Fixed effects matrix (currently only an
                        intercept term), shape (n_replicates, 1).
                    Ls (list of torch.Tensor): Spatially smoothed latent embeddings,
                        one tensor per sample with shape (n_spots, n_features).
                    A (list of torch.Tensor): Row-normalized spatial adjacency
                        matrices, one per sample with shape (n_spots, n_spots).
                    R (torch.Tensor): Section-to-replicate relationship matrix,
                        row-normalized, shape (n_replicates, n_samples).
    '''
    if issparse(idata.X):
        idata.X = np.array(idata.X.todense())
    idata.obsm[latent] = StandardScaler().fit_transform(
        idata.obsm[latent][:, :10]
    ) / np.sqrt(10)

    y = [] # group labels
    Ls = [] # latent embeddings
    A = [] # adjacency matrix

    animal_to_idx = defaultdict(list) # later needed to create relationship matrix between biological replicates

    for idx, sample in enumerate(idata.obs["sample"].unique()):

        _idata = idata[idata.obs["sample"] == sample]

        _y = _idata.obs["group"].astype(int).unique()
        _L = np.array(_idata.obsm[latent]).astype(np.float32)
        y.append(torch.Tensor(_y))
        Ls.append(torch.Tensor(_L))

        animal_to_idx[_idata.obs["biological_replicate"].unique()[0]].append(idx)
        
        # determine adjacency matrix
        n_spots = len(_idata)
        coordinates = _idata.obsm["spatial"]
        distances = distance.cdist(coordinates, coordinates)

        # should work with 100 for most datasets, but Alzheimer's did not have proper scalefactors, therefore some try different scaling, so mean number of edges 6-7 (including self-loops)
        scale = 100
        while True:
            if scale == 0:
                raise ValueError("Something is wrong in the spatial information of the adata.")
            _A = np.zeros((n_spots, n_spots), dtype=int)
            _A[distances <= _idata.uns["spatial"][sample]["scalefactors"]["spot_diameter_fullres"] / 55 * scale] = 1
            scale -= 10
           
            if 6 < np.mean(_A.sum(axis=1)) < 7:
                break

        _A = _A / _A.sum(axis=1)[:,None]
        A.append(torch.Tensor(_A))

    # spatial smoothing of latents   
    Ls = spatial_smoothing(Ls, A, k_hops)

    # construct biological_replicate relationship matrix
    n_biological_replicates = len(animal_to_idx)
    n_sections = len(y)
    R = torch.zeros(n_biological_replicates,n_sections)
    for i, replicate in enumerate(animal_to_idx.keys()):
        for j in animal_to_idx[replicate]:
            R[i,j] = 1

    R = R / torch.sum(R, 1)[:, None]

    # shrink y from sample to replicate level
    y = torch.cat(y)[:, None]
    y = torch.mm(R,y)
    F = torch.ones_like(y) # fixed effect, currently only bias term

    return y, F, Ls, R

# Softmax Variations

class WPlatSoftmax(nn.Module):
    def __init__(self, dim=1):
        super(WPlatSoftmax, self).__init__()
        self.dim = dim
        self.shift = 0.0  # nn.Parameter(torch.zeros(1))

    def forward(self, x):
        softplus_x  = self.raw_weights(x)
        softplus_sum = torch.sum(softplus_x , dim=self.dim, keepdim=True)
        output = softplus_x  / softplus_sum
        return output

    def raw_weights(self, x):
        shifted_x = x + self.shift
        softplus_x  = F.softplus(shifted_x)
        return softplus_x 


class WSoftmax(nn.Module):
    def __init__(self, dim=1):
        super(WSoftmax, self).__init__()
        self.softmax = nn.Softmax(dim=dim)
        self.shift = 0  # nn.Parameter(torch.zeros(1))

    def forward(self, x):
        shifted_x = x + self.shift
        return self.softmax(shifted_x)

    def raw_weights(self, x):
        shifted_x = x + self.shift
        return torch.exp(shifted_x)

class SphericalSoftmax(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.eps = 1e-8
        self.dim = dim
        self.shift = 0.0  # nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_2 = self.raw_weights(x)
        x_2_sum = torch.sum(x_2 , dim=self.dim, keepdim=True)
        output = x_2  / (x_2_sum + self.eps)
        return output

    def raw_weights(self, x):
        shifted_x = x + self.shift
        return shifted_x ** 2 


def get_weight_act(activation, dim=1):
    if activation == "softmax":
        return WSoftmax(dim=dim)
    elif activation == "platsoftmax":
        return WPlatSoftmax(dim=dim)
    elif activation == "sphericalsoftmax":
        return SphericalSoftmax(dim=dim)
    else:
        raise ValueError(f"Unsupported activation: {activation}")

def kl_divergence_normal(q_m, q_s, p_m, p_s):
    return 0.5 * (q_s**2 + (q_m - p_m) ** 2) / p_s**2 + torch.log(p_s / q_s) - 0.5

class BinomialLik(nn.Module):
    def __init__(self, total_count=1):
        super().__init__()
        self.total_count = total_count

    def log_prob(self, logits, targets, reduce=True):
        likelihood = Binomial(total_count=self.total_count, logits=logits)
        out = likelihood.log_prob(targets)
        if reduce:
            return out.sum(0).mean()
        else:
            return out

    def predict(self, logits):
        return self.total_count * torch.sigmoid(logits)

def prepare_gene_input(idata, gene):
    """Prepare spot-level gene expression feature tensors for a single gene.

    Extracts the expression values of a given gene across all
    samples, returning a list of per-sample tensors ready for use as `xs`
    in SpaceDX or UniformAttentionBaseline.

    Args:
        idata (AnnData): Input AnnData object. Must contain:
            - idata.obs["sample"]: sample-level annotations
            - idata.X: expression matrix (dense or sparse)
        gene (str): Gene name to extract (must be in idata.var_names).

    Returns:
        xs (list of torch.Tensor): Per-sample expression tensors, each of
            shape (n_cells_in_sample, 1).
    """
    idata_gene = idata[:, idata.var.index == gene]
    xs = []
    for sample in idata.obs["sample"].unique():
        _idata_gene = idata_gene[idata.obs["sample"] == sample]
        _x = _idata_gene.X.astype(np.float32)
        xs.append(torch.Tensor(_x))
    return xs


def get_results(model):
    _dict = {}

    elbo = model._compute_iw_elbo(1, 128)
    _dict["elbo_m"] = elbo.mean().item()
    _dict["elbo_se"] = elbo.std().item() / np.sqrt(128)

    iw_elbo = model._compute_iw_elbo(32, 128)
    _dict["iw_elbo_m"] = iw_elbo.mean().item()
    _dict["iw_elbo_se"] = iw_elbo.std().item() / np.sqrt(128)

    _dict["loglik"] = model.loglik

    try:
        _dict["beta_m"] = model.beta_m.item()
        _dict["beta_s"] = model.beta_s.item()
        _dict["kld_beta"] = model.kld_beta

        # hyprerior model only
        try:
            _dict["lvar_beta_m"] = model.lvar_beta_m.item()
            _dict["lvar_beta_s"] = model.lvar_beta_s.item()
            _dict["kld_lvar_beta"] = model.kld_lvar_beta

        except:
            pass

    except:
        pass

    try:
        _dict["gamma_m"] = model.gamma_m.detach().cpu().tolist()
        _dict["gamma_s"] = model.gamma_s.detach().cpu().tolist()
        _dict["kld_gamma"] = model.kld_gamma

        # hyprerior model only
        try:
            _dict["lvar_gamma_m"] = model.lvar_gamma_m.item()
            _dict["lvar_gamma_s"] = model.lvar_gamma_s.item()
            _dict["kld_lvar_gamma"] = model.kld_lvar_gamma

        except:
            pass

    except:
        pass

    return _dict