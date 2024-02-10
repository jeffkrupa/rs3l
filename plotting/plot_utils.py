
import sys
import os
import glob
import h5py
import numpy as np
import awkward as ak
import uproot as uproot
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import awkward as ak
import subprocess
sns.set_context("paper")
import mplhep as hep
import json
matplotlib.use('agg')
import tqdm
plt.style.use(hep.style.CMS)
import argparse
from matplotlib import gridspec

def get_last_part_of_ipath(ipath):
    # Normalize the path (removes any trailing slashes)
    ipath = os.path.normpath(ipath)

    # Extract the last part
    last_part = os.path.basename(ipath)

    return last_part


def plot_binned_data(axes, binedges, data,
               *args, **kwargs):
    #The dataset values are the bin centres
    x = (binedges[1:] + binedges[:-1]) / 2.0
    #The weights are the y-values of the input binned data
    weights = data
    return axes.hist(x, bins=binedges, weights=weights,
               *args, **kwargs)




def fill_between_steps(ax, x, y1, y2=0, step_where='pre', **kwargs):
    ''' fill between a step plot and 

    Parameters
    ----------
    ax : Axes
       The axes to draw to

    x : array-like
        Array/vector of index values.

    y1 : array-like or float
        Array/vector of values to be filled under.
    y2 : array-Like or float, optional
        Array/vector or bottom values for filled area. Default is 0.

    step_where : {'pre', 'post', 'mid'}
        where the step happens, same meanings as for `step`

    **kwargs will be passed to the matplotlib fill_between() function.

    Returns
    -------
    ret : PolyCollection
       The added artist

    '''
    x_tmp=x
    x=x[:-1]
    
    if step_where not in {'pre', 'post', 'mid'}:
        raise ValueError("where must be one of {{'pre', 'post', 'mid'}} "
                         "You passed in {wh}".format(wh=step_where))

    # make sure y values are up-converted to arrays 
    if np.isscalar(y1):
        y1 = np.ones_like(x) * y1

    if np.isscalar(y2):
        y2 = np.ones_like(x) * y2

    # temporary array for up-converting the values to step corners
    # 3 x 2N - 1 array 

    vertices = np.vstack((x, y1, y2))

    # this logic is lifted from lines.py
    # this should probably be centralized someplace
    if step_where == 'pre':
        steps = np.ma.zeros((3, 2 * len(x) - 1), np.float)
        steps[0, 0::2], steps[0, 1::2] = vertices[0, :], vertices[0, :-1]
        steps[1:, 0::2], steps[1:, 1:-1:2] = vertices[1:, :], vertices[1:, 1:]

    elif step_where == 'post':
        steps = np.ma.zeros((3, 2 * len(x) -1), np.float)
       
        steps[0, ::2], steps[0, 1:-1:2] = vertices[0, :], vertices[0, 1:]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :-1]

    elif step_where == 'mid':
        steps = np.ma.zeros((3, 2 * len(x)), np.float)
        steps[0, 1:-1:2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 2::2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 0] = vertices[0, 0]
        steps[0, -1] = vertices[0, -1]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :]
    else:
        raise RuntimeError("should never hit end of if-elif block for validated input")

    # un-pack
    xx, yy1, yy2 = steps
    xx = np.append(xx, xx[len(xx)-1]+(x_tmp[len(x_tmp)-1]-x_tmp[len(x_tmp)-2]))
    yy1 = np.append(yy1,yy1[len(yy1)-1])
    yy2 = np.append(yy2,yy2[len(yy2)-1])
    return ax.fill_between(xx, yy1, y2=yy2,alpha=0.4,linewidth=0.0,**kwargs)

def useEnvelope(up, down):
    for i in range(up.size):
        if(up[i] < 1 and down[i] < 1):
            if (up[i] > down[i]):
                up[i] = 1
            else:
                down[i] = 1   
        
        if(up[i] > 1 and down[i] > 1):
            if (up[i] > down[i]):
                down[i] = 1
            else:
                up[i] = 1   


def make_unit_vector(vec):
    magnitude = np.linalg.norm(vec)
    vec /= magnitude
    return vec
