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
from sklearn.metrics import auc
import subprocess
sns.set_context("paper")
import mplhep as hep
import json
matplotlib.use('agg')
import tqdm
plt.style.use(hep.style.CMS)
import argparse, csv
from matplotlib import gridspec
from plot_utils import *
import matplotlib.cm as cm

#fig_size = plt.rcParams["figure.figsize"]
#fig_size[0] = 9
#fig_size[1] = 9
#plt.rcParams["figure.figsize"] = fig_size

plt.style.use('/afs/cern.ch/user/j/jekrupa/public/rs3l/plotting/rs3l.mplstyle')


which_sig_effs = [0.9,0.8,0.7,0.6,0.5]


parser = argparse.ArgumentParser()
parser.add_argument('--ipath', action='store', type=str, help="Path to h5 files")
parser.add_argument('--draw_feat', action='store', default=None, type=int, help="Feat to plot")
parser.add_argument('--which_qcd', action='store', type=str, default="all", help="All or specific qcd jet type 1..3, 5..8")
args = parser.parse_args()

training = get_last_part_of_ipath(args.ipath)
is_wz = "wz_zz" in args.ipath
is_wqcd = "wz_qcd" in args.ipath
is_hqcd = "h_qcd" in args.ipath

sig_name = "higgs"
bkg_name = "qcd"

which_qcd = args.which_qcd

os.system(f"mkdir /eos/project/c/contrast/public/cl/www/analysis/dec23/{training}")
os.system(f"cp /eos/project/c/contrast/public/cl/www/index.php /eos/project/c/contrast/public/cl/www/analysis/dec23/{training}/index.php")

index_to_feat = {
    0 : "Jet $p_\mathrm{T}\\ \mathrm{(GeV)}$" , # "$\mathrm{Jet\\ p_T\\ (GeV)}$",
    1 : "$\mathrm{Jet\\ \eta}$",
    2 : "$\mathrm{Jet\\ \phi}$",
    3 : "$\mathrm{Jet\\ energy\\ (GeV)}$",
    4 : "$\mathrm{Jet\\ m_{SD}\\ (GeV)}$",
    5 : "$\mathrm{Jet\\ N_2}$",
}

label_dict = {
  "W" : "W",
  "Z" : "Z",
  "higgs" : "H",
  "qcd" : "QCD",
}
var_label_to_name = {
   -1 : "nominal",
   0 : "seed",
   1 : "fsrRenHi",
   2 : "fsrRenLo",
   3 : "herwig",
}
whichfeat = 999

def read_files(process,variation,):
    global training
    arr = None
    counter = 0

    pattern = "%s/*.h5"%(args.ipath)
    #pattern = "%s/*h5"%(args.ipath)
    for i in tqdm.tqdm(glob.glob(pattern)):
        counter += 1
        #print("hello")
        if 1: #try:
            with h5py.File(i,'r') as f:
                feat = f['jet_features'][()][:]
                feat = np.expand_dims(feat,axis=-1)
                
                vartype = f['jet_vartype'][()]
                sel = (vartype==variation) 
                jettype = f['jet_type'][()]
                if "higgs" in process:
                    sel &= (jettype==4) 
                elif "qcd" in process: 
                    sel &= (jettype!=4) 
                sel = sel.flatten()    
                feat = feat[sel]
                print(feat.shape,sel.shape)
                #feat = np.expand_dims(feat,axis=-1)
                vartype = vartype[sel]
                #vartype = np.expand_dims(vartype,axis=-1)
                jettype = jettype[sel]
                #jettype = np.expand_dims(jettype,axis=-1)
                if arr is None:
                    arr = feat 
                    arr_vartype = vartype
                    arr_jettype = jettype
                else:
                    arr = np.concatenate((arr,feat))
                    arr_vartype = np.concatenate((arr_vartype,vartype))
                    arr_jettype = np.concatenate((arr_jettype,jettype))
    if arr is None:
        print("Array is empty!")
        sys.exit()
    return arr, arr_vartype, arr_jettype
masterdict = {}

path=f"/eos/project/c/contrast/public/cl/www/analysis/dec23/{training}/arrays.h5"

if os.path.isfile(path):
    f = h5py.File(path,"a")
else:
    f = h5py.File(path,"w")

def plot(process,variation,):
    try :
        del f[f"{process}_{variation}"]
    except :
        pass 
    f[f"{process}_{variation}"],_,_ = read_files(process,variation,)
    #masterdict[f"{process}_{variation}"] = read_files(process,variation,)

plot(bkg_name,-1)
plot(sig_name,-1)
f.close()
