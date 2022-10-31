"""
get_fred_data
plot_price_series
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

def plot_price_series(df, col_names, save_dir, show_corr=True):
    """
    col_names:  series to be plotted
    df: dataframe containing price series in columns
    show_corr:  boolean to display correlation matrix of series
    """
    
    # Plot price series
    n_cols = len(col_names)
    axes = df.plot( subplots=True,
                    figsize=(14, 6),
                    layout=(n_cols//2 + n_cols % 2, 2),
                    title=col_names,
                    legend=False,
                    rot=0,
                    lw=1, 
                    color='k')
    for ax in axes.flatten():
        ax.set_xlabel('')
    
    plt.suptitle('Original Price Series')
    plt.gcf().tight_layout()
    fig = plt.gcf()    
    sns.despine();
    plt.show()
    
    fname = save_dir / 'ts_plots.png'
    fig.savefig(fname)
    
    # Correlation
    if show_corr:
        df.corr()
        
        sns.clustermap(df.corr(),
                       annot=True,
                       fmt='.2f',
                       cmap=sns.diverging_palette(h_neg=20,
                                                  h_pos=220), center=0);
        fig =plt.gcf()
        plt.show()
        
        fname = save_dir / 'ts_corr.png'
        fig.savefig(fname)


def get_fred_data(hdf, fname, save_dir, plot_series=True, show_corr=True):
    """
    hdf:  HDF5 file   e.g. results_path / 'TimeSeriesGAN.h5'
    fname:  name of file in HDF5 store      e.g. 'data/gdp_crepi'
    """
    df = pd.read_hdf(hdf, fname)
    if plot_series:
        col_names = list(df.columns)
        plot_price_series(df, col_names, save_dir, show_corr=show_corr)
    return df