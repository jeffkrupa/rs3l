import corner 
import numpy as np
import h5py
import sys
import seaborn,sklearn
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib as mpl
from scipy import stats
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
hep.style.use(hep.style.ATLAS)

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', linewidth=0.8)
global whicharray

import argparse
parser = argparse.ArgumentParser(description='Test.')                                 
parser.add_argument('--whichpath', action='store', type=str, help='Path to training.') 


args = parser.parse_args()

processes = [(1,"q"),(2,"c"),(3,"b"),(4,"H"),(5,"g(qq)"),(6,"g(cc)"),(7,"g(bb)"),(8,"g(gg)")]
n_output_nodes=8

color_palette = {
    "q" : "lightcoral",
    "c" : "lightblue",
    "b" : "lightgreen",
    "H" : "magenta",
    "g(qq)" : "red",
    "g(cc)" : "blue",
    "g(bb)" : "green",
    "g(gg)" : "orange",
}


corner_labels = [
"0",
"1",
"2",
"3",
"4",
"5",
"6",
"7",
]

def custom_corner(arr):
    global opath
    global whichsample
    fig, axs = plt.subplots(arr.shape[-1],arr.shape[-1],figsize=(9,9))

    for i in range(arr.shape[-1]):
        #print(i)
        for j in range(arr.shape[-1]):
            axs[i,j].tick_params(direction="in")
            axs[i,j].tick_params(length=3,which='major')
            axs[i,j].tick_params(length=2,which='minor')
            axs[i,j].tick_params(axis='x',which='both',top=False)
            axs[i,j].tick_params(axis='y',which='both',right=False)
            axs[i,j].tick_params(axis='x',which='minor',bottom=False)
            axs[i,j].tick_params(axis='y',which='minor',left=False)
            if i == j:
                axs[i,i].hist(arr[:,i],bins=25,histtype='step',color='red',linewidth=0.8)
                axs[i,j].yaxis.set_ticklabels([])
                axs[i,j].yaxis.set_ticks([])
                axs[i,j].tick_params(axis='x',which='both',top=False)
                axs[i,j].text(0.5,0.5,corner_labels[i],transform=axs[i,j].transAxes,fontsize=8)
            if j > i:                
                axs[i,j].axis('off')
            if j < i:
                axs[i,j].hexbin(arr[:,i],arr[:,j], gridsize=40, cmap='Reds')
                axs[i,j].text(0.7,0.7,"%.2f"%stats.pearsonr(arr[:,i],arr[:,j])[0],transform=axs[i,j].transAxes,fontsize=7)
            if i != arr.shape[-1]-1:
                axs[i,j].xaxis.set_ticklabels([])
            if j != 0:
                axs[i,j].yaxis.set_ticklabels([])
    plt.subplots_adjust(wspace=0.08, hspace=0.08)
    plt.savefig("{}/custom_corner.png".format(args.whichpath),bbox_inches='tight',dpi=300)
    plt.savefig("{}/custom_corner.pdf".format(args.whichpath),bbox_inches='tight',dpi=300)

    return None

if __name__ == "__main__":

    process = "higgs"
    masterarray = h5py.File(f"{args.whichpath}/arrays.h5","r")
    masterarray = np.squeeze(masterarray["qcd_-1"],axis=-1)
    print(masterarray.shape)
    custom_corner(masterarray)
