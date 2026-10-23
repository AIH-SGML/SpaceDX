import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_attention_spatial(
    ad,
    group_col="group",
    sample_col="sample",
    value_col="attention_weights",
    spatial_key="spatial",
    groups=(0, 1),
    n_cols=10,
    percentile=99.9,
    cmap="viridis",
    point_size=1,
    random_state=None,
):
    """
    Plot spatial attention maps for randomly selected samples from two groups.

    Parameters
    ----------
    ad : AnnData
        Annotated data matrix.
    group_col : str
        Column in ad.obs indicating group membership.
    sample_col : str
        Column in ad.obs containing sample IDs.
    value_col : str
        Column in ad.obs containing values to plot (e.g., attention weights).
    spatial_key : str
        Key in ad.obsm containing spatial coordinates.
    groups : tuple
        Two group labels to compare (top row, bottom row).
    n_cols : int
        Number of samples per group.
    percentile : float
        Percentile used for clipping values.
    cmap : str
        Matplotlib colormap.
    point_size : float
        Scatter point size.
    random_state : int or None
        Random seed for reproducibility.
    """

    if random_state is not None:
        np.random.seed(random_state)

    figsize = (2 * n_cols + 0.8, 2.25 * 2)

    group_samples = []
    for g in groups:
        samples = np.random.choice(
            ad.obs.loc[ad.obs[group_col] == g, sample_col].unique(),
            size=n_cols,
            replace=False,
        )
        group_samples.append(samples)

    samples_to_plot = list(group_samples[0]) + list(group_samples[1])
    sample_groups = [groups[0]] * n_cols + [groups[1]] * n_cols

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, n_cols, figure=fig, hspace=0.05, wspace=0.05)

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap(cmap)

    for i, (sample, grp) in enumerate(zip(samples_to_plot, sample_groups)):
        row = 0 if grp == groups[0] else 1
        col = i if grp == groups[0] else i - n_cols

        ad_sample = ad[ad.obs[sample_col] == sample].copy()
        ax = fig.add_subplot(gs[row, col])

        vals = ad_sample.obs[value_col].values
        vmax = np.percentile(vals, percentile)
        vals_scaled = np.clip(vals, 0, vmax) / vmax

        ax.scatter(
            x=ad_sample.obsm[spatial_key][:, 0],
            y=ad_sample.obsm[spatial_key][:, 1],
            c=vals_scaled,
            cmap=cmap,
            norm=norm,
            s=point_size,
            rasterized=True,
        )

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        if col == 0:
            ax.set_ylabel(str(grp), fontsize=11, rotation=90, labelpad=8, va="center")

    plt.show()