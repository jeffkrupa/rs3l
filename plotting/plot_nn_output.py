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
from plot_utils import *
#fig_size = plt.rcParams["figure.figsize"]
#fig_size[0] = 9
#fig_size[1] = 9
#plt.rcParams["figure.figsize"] = fig_size

plt.style.use('/afs/cern.ch/user/j/jekrupa/public/rs3l/plotting/rs3l.mplstyle')


parser = argparse.ArgumentParser()
parser.add_argument('--ipath', action='store', type=str, help="Path to h5 files")
parser.add_argument('--is_n2', action='store_true', default=False, help="Plotting N2")
parser.add_argument('--which_qcd', action='store', type=str, default="all", help="All or specific qcd jet type 1..3, 5..8")
args = parser.parse_args()




training = get_last_part_of_ipath(args.ipath)
is_n2 = args.is_n2
which_qcd = args.which_qcd

os.system(f"mkdir /eos/project/c/contrast/public/cl/www/analysis/{training}")
os.system(f"cp /eos/project/c/contrast/public/cl/www/index.php /eos/project/c/contrast/public/cl/www/analysis/{training}/index.php")

var_label_to_name = {
   -1 : "nominal",
   0 : "seed",
   1 : "fsrRenHi",
   2 : "fsrRenLo",
   3 : "herwig",
}
whichfeat = 999
if is_n2:
    whichfeat = 'n2'

def read_files(process,variation,variable):
    global training
    arr = None
    counter = 0

    pattern = "%s/*h5"%(args.ipath)
    print(pattern)
    for i in tqdm.tqdm(glob.glob(pattern)):
        counter += 1

        try:
            with h5py.File(i,'r') as f:
                if variable == 'n2':
                    feat = f['jet_kinematics'][()][:,-1]
                else:
                    feat = f['jet_features'][()]
                
                vartype = f['jet_vartype'][()]
                sel = (vartype==variation) 
                jettype = f['jet_type'][()]
                if "higgs" in process:
                    sel &= (jettype==4) 
                elif "qcd" in process: 
                    sel &= (jettype!=4) 
                feat = feat[sel]
                feat = np.expand_dims(feat,axis=-1)
                if arr is None:
                    arr = feat
                else:
                    arr = np.concatenate((arr,feat))
        except:
            print(f"file {i} doesn't open, skipping...") 
    #print(arr)
    if arr is None:
        print("Array is empty!")
        sys.exit()
    return arr

hist_dict = {}




gs = gridspec.GridSpec(7, 1, height_ratios=[1.8,0.5,0.5,0.5,0.5,0.5,0.5,])

binedges_global = [0,1,30]
if is_n2:
    binedges_global = [0.02,0.48,30]

def plot(axis,process,variation,variable,binedges,color,label,show=True):

    #print("In plotting function")
    arr = read_files(process,variation,variable,)

    #sys.exit(1)
    tmpdict = {'val':np.squeeze(arr)}
    tmpdf = pd.DataFrame.from_dict(tmpdict)
    oname=f"/eos/project/c/contrast/public/cl/www/analysis/{training}/{process}_{var_label_to_name[variation]}_{which_qcd}.csv" 
    tmpdf.to_csv(oname)
        
    bins = np.linspace(binedges_global[0],binedges_global[1],binedges_global[2])
    #print(bins)
    y, x, dummy = axis.hist(arr,bins=bins,linewidth=1.3,density=True,histtype='step',alpha=0)
    
    #print(y,x,dummy)
    if show:
        _ = plot_binned_data(axis, bins, y, histtype="step",stacked=False,color=color,label=label,linewidth=1.3,rwidth=2)
    else:
        _ = plot_binned_data(axis, bins, y, histtype="step",stacked=False,color=color,linewidth=1.3,rwidth=2,alpha=0.)
        
    hist_dict[f'{str(variable)}_{process}_{var_label_to_name[variation]}'] = _


def plot_ratio(axis,process1,variation1,process2,variation2,variable,binedges,color,text=None,doboth=False,other=None):

    global binedges_global
    
    bins = np.linspace(binedges_global[0],binedges_global[1],binedges_global[2])

    y,x,dummy = plot_binned_data(axis, bins, np.nan_to_num(hist_dict[f'{str(variable)}_{process1}_{var_label_to_name[variation2]}'][0]/hist_dict[f'{str(variable)}_{process2}_{var_label_to_name[variation1]}'][0],copy=True,posinf=0), histtype="step",stacked=False,color=color,linewidth=0.0,rwidth=2)
    if doboth:

        y_other,x_other,dummy = plot_binned_data(axis, bins, np.nan_to_num(hist_dict[f'{str(variable)}_{process1}_{var_label_to_name[other]}'][0]/hist_dict[f'{str(variable)}_{process2}_{var_label_to_name[variation1]}'][0],copy=True,posinf=0), histtype="step",stacked=False,color=color,linewidth=0.0,rwidth=2)
        fill_between_steps(axis,bins, y, y_other, step_where="post",color=color,zorder=0)
    else:
        binentries = []
        for i in range(len(hist_dict[f'{str(variable)}_{process1}_{var_label_to_name[variation1]}'][0])):
            #print(hist_dict[f'{str(variable)}_{process1}_{variation1}'][0][i])
            if hist_dict[f'{str(variable)}_{process1}_{var_label_to_name[variation1]}'][0][i] == 0:
                binentries.append(0)
            else:
                binentries.append(1)

        lower = []
        for i in range(len(hist_dict[f'{str(variable)}_{process1}_{var_label_to_name[variation1]}'][0])):
            if y[i] < 1:
                lower.append(y[i]-0.05)
            else:
                lower.append(0.98)
        fill_between_steps(axis,bins, y, binentries , step_where="post",color=color,zorder=0)
        #print(lower)
        #plot_binned_data(axis, bins, lower, histtype="step",stacked=False,color='white',linewidth=1.6,rwidth=2)
    axis.text(0.03,0.7,text,transform=axis.transAxes,fontsize=14)

fig = plt.figure(figsize=(8,6))

ax = plt.subplot(gs[0])
ax_ratio1 = plt.subplot(gs[1])
ax_ratio2 = plt.subplot(gs[2])
ax_ratio3 = plt.subplot(gs[3])
ax_ratio4 = plt.subplot(gs[4])
ax_ratio5 = plt.subplot(gs[5])
ax_ratio6 = plt.subplot(gs[6])
ax.xaxis.set_zorder(99) 
ax.set_yscale('log')

qcd_label = {
  "all" : "QCD",
  "1" : "q",
  "2" : "c",
  "3" : "b",
  "5" : "g(qq)",
  "6" : "g(cc)",
  "7" : "g(bb)",
  "8" : "g(gg)",
}
qcd_legend_label = "QCD"
if which_qcd != 'all':
    qcd_legend_label += " [{qcd_label[which_qcd]}]"
plot(ax,'qcd',-1,whichfeat,[0.05,0.5,30],'steelblue',f'QCD [{qcd_label[which_qcd]}]')
plot(ax,'higgs',-1,whichfeat,[0.05,0.5,30],'magenta','Higgs')
plot(ax,'qcd',0,whichfeat,[0.05,0.5,30],'yellow',0,False)
plot(ax,'higgs',0,whichfeat,[0.05,0.5,30],'yellow',0,False)
plot(ax,'qcd',1,whichfeat,[0.05,0.5,30],'yellow','fsrRenHi',False)
plot(ax,'qcd',2,whichfeat,[0.05,0.5,30],'yellow','fsrRenLo',False)
plot(ax,'higgs',1,whichfeat,[0.05,0.5,30],'yellow','fsrRenHi',False)
plot(ax,'higgs',2,whichfeat,[0.05,0.5,30],'yellow','fsrRenLo',False)
plot(ax,'qcd',3,whichfeat,[0.05,0.5,30],'yellow','herwig',False)
plot(ax,'higgs',3,whichfeat,[0.05,0.5,30],'yellow','herwig',False)

label = "NN output"
if is_n2:
    label = "$N_2$"
ax_ratio6.set_xlabel(label)
ax.set_ylabel("Norm. to unit area",fontsize=21)
ax.legend(loc=(0.4,0.2))

#plot_ratio(ax_ratio1,'qcd','nominal','qcd','fsrRenHi',whichfeat,[0.05,0.5,30],'salmon',"$\mu(FSR)$ [QCD]",doboth=True,other='fsrRenLo')
#plot_ratio(ax_ratio2,'higgs','nominal','higgs','fsrRenHi',whichfeat,[0.05,0.5,30],'steelblue',"$\mu(FSR)$ [Higgs]",doboth=True,other='fsrRenLo')
#plot_ratio(ax_ratio3,'qcd','nominal','qcd','herwig',whichfeat,[0.05,0.5,30],'salmon',"Herwig7 [QCD]")
#plot_ratio(ax_ratio4,'higgs','nominal','higgs','herwig',whichfeat,[0.05,0.5,30],'steelblue',"Herwig7 [Higgs]")
NBINS=30


plot_ratio(ax_ratio1,'qcd',-1,'qcd',0,whichfeat,[0.05,0.5,NBINS],'springgreen',f"seed [{qcd_label[which_qcd]}]")
plot_ratio(ax_ratio2,'higgs',-1,'higgs',0,whichfeat,[0.05,0.5,NBINS],'indigo',f"seed [H]",)
plot_ratio(ax_ratio3,'qcd',-1,'qcd',1,whichfeat,[0.05,0.5,NBINS],'springgreen',f"FSR [{qcd_label[which_qcd]}]",doboth=True,other=2)
plot_ratio(ax_ratio4,'higgs',-1,'higgs',1,whichfeat,[0.05,0.5,NBINS],'indigo',f"FSR [H]",doboth=True,other=2)
plot_ratio(ax_ratio5,'qcd',-1,'qcd',3,whichfeat,[0.05,0.5,NBINS],'springgreen',f"Herwig7 [{qcd_label[which_qcd]}]")
plot_ratio(ax_ratio6,'higgs',-1,'higgs',3,whichfeat,[0.05,0.5,NBINS],'indigo',f"Herwig7 [H]")
axxes = [ax_ratio1,ax_ratio2,ax_ratio3,ax_ratio4,ax_ratio5,ax_ratio6]

#print(hist_dict)

plt.subplots_adjust(top=0.98)
fig.subplots_adjust(hspace=0.00)

ax.set_xlim([binedges_global[0],binedges_global[1]])
ax.xaxis.set_ticklabels([])

#axxes = [ax_ratio1,ax_ratio2,ax_ratio3,ax_ratio4]
for axx in axxes:
    if axx == axxes[-3]:
        axx.set_ylabel("Variation/Nominal",fontsize=13)
    if axx != axxes[-1]:
        axx.xaxis.set_ticklabels([])
    #if axx == axxes[-1] or axx == axxes[-2]:
    #    axx.set_ylim([0.,2.])
    #else:
    #    axx.set_ylim([0.,2.])
    axx.set_ylim([0,2])

    axx.set_yticks([0.5,1.5])
    axx.set_yticklabels(["0.5","1.5"])

    axx.set_xlim([binedges_global[0],binedges_global[1]]) 
    #axx.yaxis.set_ticklabels(['0.5','1.0','1.5'],fontsize=12)
    axx.tick_params(axis='y', which='major', labelsize=12)
    axx.axhline(y=1, linestyle='dashed',color='k')

label = "nn_output"
if is_n2:
    label = "n2_output"
if which_qcd != "all":
    label = label+"_"+which_qcd

plt.savefig(f"/eos/project/c/contrast/public/cl/www/analysis/{training}/{label}.png",dpi=300,bbox_inches='tight')
plt.savefig(f"/eos/project/c/contrast/public/cl/www/analysis/{training}/{label}.pdf",dpi=300,bbox_inches='tight')


