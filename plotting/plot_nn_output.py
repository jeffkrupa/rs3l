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

nn_bins = np.concatenate((np.linspace(-0.001,0.00001,1000),np.linspace(0.00001,0.99999,10000),np.linspace(0.99999,1.0001,1000)))
which_sig_effs = [0.9,0.8,0.7,0.6,0.5]


parser = argparse.ArgumentParser()
parser.add_argument('--ipath', action='store', type=str, help="Path to h5 files")
parser.add_argument('--is_n2', action='store_true', default=False, help="Plotting N2")
parser.add_argument('--which_qcd', action='store', type=str, default="all", help="All or specific qcd jet type 1..3, 5..8")
args = parser.parse_args()


training = get_last_part_of_ipath(args.ipath)
is_wz = "wz_zz" in args.ipath
if is_wz:
   sig_name = "z"
   bkg_name = "w"
else:
   sig_name = "higgs"
   bkg_name = "qcd"

is_n2 = args.is_n2
which_qcd = args.which_qcd

os.system(f"mkdir /eos/project/c/contrast/public/cl/www/analysis/{training}")
os.system(f"cp /eos/project/c/contrast/public/cl/www/index.php /eos/project/c/contrast/public/cl/www/analysis/{training}/index.php")

label_dict = {
  "w" : "W",
  "z" : "Z",
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
if is_n2:
    whichfeat = 'n2'

def read_files(process,variation,variable):
    global training
    arr = None
    counter = 0

    pattern = "%s/*.h5"%(args.ipath)
    #pattern = "%s/*h5"%(args.ipath)
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
                if is_wz:
                    if "w" in process:
                        sel &= (jettype==11)
                    elif "z" in process:
                        sel &= (jettype==10)
                else:
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
    if arr is None:
        print("Array is empty!")
        sys.exit()
    return arr

hist_dict = {}




gs = gridspec.GridSpec(7, 1, height_ratios=[1.8,0.5,0.5,0.5,0.5,0.5,0.5,])

binedges_global = [0,1,30]
if is_n2:
    binedges_global = [0.02,0.48,30]

tprs = {}
fprs = {} 

def get_unc(reverse=False,):

    base=f"/eos/project/c/contrast/public/cl/www/analysis/{training}/"
    fpr_dict,pct_fpr_change = {}, {}
    tpr_dict,pct_tpr_change = {}, {}
    for variation in ["nominal","fsrRenHi","fsrRenLo","herwig"]:
        with open(f"{base}/ROC_{variation}.csv", 'r') as file:
            counter = 0 
            reader = csv.reader(file)
            rows = list(reader)
            for sig_eff in which_sig_effs:
                for index, row in enumerate(rows):
    
                    if index == 0 or index == len(rows)-1: 
                        continue
    
                    if float(rows[index][0]) >= sig_eff and float(rows[index+1][0]) < sig_eff :
                        tpr_dict[f"{sig_eff}_{variation}"] = float(rows[index][0])                  
                        fpr_dict[f"{sig_eff}_{variation}"] = float(rows[index][1])                  
                        break

    for sig_eff in which_sig_effs: 
        for variation in ["fsrRenHi","fsrRenLo","herwig"]:
            pct_tpr_change[f"{sig_eff}_{variation}"] = (tpr_dict[f"{sig_eff}_{variation}"] - tpr_dict[f"{sig_eff}_nominal"] ) / (tpr_dict[f"{sig_eff}_nominal"]) 
            pct_fpr_change[f"{sig_eff}_{variation}"] = (fpr_dict[f"{sig_eff}_{variation}"] - fpr_dict[f"{sig_eff}_nominal"] ) / (fpr_dict[f"{sig_eff}_nominal"])

    efficiencies = set(key.split('_')[0] for key in pct_fpr_change.keys())
    variations = set(key.split('_')[1] for key in pct_fpr_change.keys())
    csv_data = {variation: {eff: pct_fpr_change.get(f"{eff}_{variation}", None) for eff in efficiencies} for variation in variations}
    sorted_variations = sorted(csv_data.keys())    
    # Writing to CSV
    csv_file = f"{base}/change_in_fpr.csv"
    with open(csv_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Variation"] + sorted(efficiencies))
        writer.writeheader()

        for variation in sorted_variations:
            row = {"Variation": variation}
            row.update(csv_data[variation])
            writer.writerow(row) 

def get_tpr_fpr(reverse=False,):

    for var in ["nominal","fsrRenHi","fsrRenLo","herwig"]:
        sig_csv = f"/eos/project/c/contrast/public/cl/www/analysis/{training}/{sig_name}_{var}.csv"
        bkg_csv = f"/eos/project/c/contrast/public/cl/www/analysis/{training}/{bkg_name}_{var}.csv"
    
        df_sig = np.array(pd.read_csv(sig_csv)['val'].values.tolist())
        df_bkg = np.array(pd.read_csv(bkg_csv)['val'].values.tolist())
    
        for i in nn_bins:
            if not reverse:
                tpr = (df_sig >= i).sum()/len(df_sig)
                fpr = (df_bkg >= i).sum()/len(df_bkg)
            else:
                tpr = (df_sig <= i).sum()/len(df_sig)
                fpr = (df_bkg <= i).sum()/len(df_bkg)
            if '%s_%s'%(training,var) not in tprs:
                tprs['%s_%s'%(training,var)] = []
                tprs['%s_%s'%(training,var)].append(tpr)
                fprs['%s_%s'%(training,var)] = []
                fprs['%s_%s'%(training,var)].append(fpr)
            else:
                tprs['%s_%s'%(training,var)].append(tpr)
                fprs['%s_%s'%(training,var)].append(fpr)
    
        plt.clf() 
        fig,ax = plt.subplots()
        ax.plot(tprs['%s_%s'%(training,var)],fprs['%s_%s'%(training,var)],
            label=" AUC=%.4f"%(1.-auc(tprs['%s_%s'%(training,var)],fprs['%s_%s'%(training,var)])),color="black",linewidth=2,alpha=1.)
      
        ax.text(0.2,0.7,var,transform=ax.transAxes) 
        ax.set_xlabel(f"{label_dict[sig_name]} acceptance",fontsize=24) 
        ax.set_ylabel(f"{label_dict[bkg_name]} fake rate",fontsize=24)
        plt.grid(which='both')
        plt.legend(fontsize=16)
        if is_wz: 
            ax.set_ylim([0.001,1.002])
            ax.set_xlim([0.001,1.002])
        else:
            ax.set_ylim([0.003,.08])
            ax.set_xlim([0.25,1.03])
        plt.savefig(f"/eos/project/c/contrast/public/cl/www/analysis/{training}/ROC_{sig_name}_vs_{bkg_name}_{var}.png",dpi=300,bbox_inches='tight')
        plt.savefig(f"/eos/project/c/contrast/public/cl/www/analysis/{training}/ROC_{sig_name}_vs_{bkg_name}_{var}.pdf",dpi=300,bbox_inches='tight')
        ax.set_yscale('log')
        plt.savefig(f"/eos/project/c/contrast/public/cl/www/analysis/{training}/ROC_{sig_name}_vs_{bkg_name}_{var}_log.png",dpi=300,bbox_inches='tight')
        plt.savefig(f"/eos/project/c/contrast/public/cl/www/analysis/{training}/ROC_{sig_name}_vs_{bkg_name}_{var}_log.pdf",dpi=300,bbox_inches='tight')

        roc_csv = sig_csv.replace(sig_name,"ROC")
        with open(roc_csv, 'w', newline='') as file:
            writer = csv.writer(file)

            for a1, a2 in zip(tprs['%s_%s'%(training,var)],fprs['%s_%s'%(training,var)]):
                writer.writerow([a1,a2])

def plot(axis,process,variation,variable,binedges,color,label,show=True):

    #print("In plotting function")
    arr = read_files(process,variation,variable,)

    #sys.exit(1)
    tmpdict = {'val':np.squeeze(arr)}
    tmpdf = pd.DataFrame.from_dict(tmpdict)
    oname=f"/eos/project/c/contrast/public/cl/www/analysis/{training}/{process}_{var_label_to_name[variation]}.csv" 
    tmpdf.to_csv(oname)
        
    bins = np.linspace(binedges_global[0],binedges_global[1],binedges_global[2])
    #print(bins)
    y, x, dummy = axis.hist(arr,bins=bins,linewidth=1.3,density=True,histtype='step',alpha=0)
    
    #print(y,x,dummy)
    if show:
        _ = plot_binned_data(axis, bins, y, histtype="step",stacked=False,color=color,label=label,linewidth=1.3,rwidth=2,linestyle="-")
    else:
        _ = plot_binned_data(axis, bins, y, histtype="step",stacked=False,color=color,linewidth=1.3,rwidth=2,alpha=0.,linestyle="-")
        
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
    axis.text(0.03,0.7,text,transform=axis.transAxes,fontsize=10)

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

label_dict = {
  "w" : "W",
  "z" : "Z",
  "higgs" : "H",
  "qcd" : "QCD",
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
    qcd_legend_label += " {label_dict[which_qcd]}"
NBINS=30

plot(ax,bkg_name,-1,whichfeat,[0.05,0.5,30],'steelblue',label_dict[bkg_name])
plot(ax,sig_name,-1,whichfeat,[0.05,0.5,30],'magenta',label_dict[sig_name])
plot(ax,bkg_name,0,whichfeat,[0.05,0.5,30],'yellow',0,False)
plot(ax,sig_name,0,whichfeat,[0.05,0.5,30],'yellow',0,False)
plot(ax,bkg_name,1,whichfeat,[0.05,0.5,30],'yellow','fsrRenHi',False)
plot(ax,bkg_name,2,whichfeat,[0.05,0.5,30],'yellow','fsrRenLo',False)
plot(ax,sig_name,1,whichfeat,[0.05,0.5,30],'yellow','fsrRenHi',False)
plot(ax,sig_name,2,whichfeat,[0.05,0.5,30],'yellow','fsrRenLo',False)
plot(ax,bkg_name,3,whichfeat,[0.05,0.5,30],'yellow','herwig',False)
plot(ax,sig_name,3,whichfeat,[0.05,0.5,30],'yellow','herwig',False)


plot_ratio(ax_ratio1,bkg_name,-1,bkg_name,0,whichfeat,[0.05,0.5,NBINS],'springgreen',f"seed [{label_dict[bkg_name]}]")
plot_ratio(ax_ratio2,sig_name,-1,sig_name,0,whichfeat,[0.05,0.5,NBINS],'indigo',f"seed [{label_dict[sig_name]}]",)
plot_ratio(ax_ratio3,bkg_name,-1,bkg_name,1,whichfeat,[0.05,0.5,NBINS],'springgreen',f"FSR [{label_dict[bkg_name]}]",doboth=True,other=2)
plot_ratio(ax_ratio4,sig_name,-1,sig_name,1,whichfeat,[0.05,0.5,NBINS],'indigo',f"FSR [{label_dict[sig_name]}]",doboth=True,other=2)
plot_ratio(ax_ratio5,bkg_name,-1,bkg_name,3,whichfeat,[0.05,0.5,NBINS],'springgreen',f"Herwig7 [{label_dict[bkg_name]}]")
plot_ratio(ax_ratio6,sig_name,-1,sig_name,3,whichfeat,[0.05,0.5,NBINS],'indigo',f"Herwig7 [{label_dict[sig_name]}]")

label = "NN output"
if is_n2:
    label = "$N_2$"
ax_ratio6.set_xlabel(label)
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
    if axx == axxes[-1] or axx == axxes[-2]:
        axx.set_ylim([0,2])

        axx.set_yticks([0.5,1.5])
        axx.set_yticklabels(["0.5","1.5"])
    else:
        axx.set_ylim([0.8,1.2])
        axx.set_yticks([0.9,1.1])
        axx.set_yticklabels(["0.9","1.1"])

    axx.set_xlim([binedges_global[0],binedges_global[1]]) 
    #axx.yaxis.set_ticklabels(['0.5','1.0','1.5'],fontsize=12)
    axx.tick_params(axis='y', which='major', labelsize=12)
    axx.axhline(y=1, linestyle='dashed',color='k')

label = "nn_output"
if is_n2:
    label = "n2_output"
if which_qcd != "all":
    label = label+"_"+which_qcd

ax.set_ylabel("Norm. to unit area",fontsize=21,labelpad=20)
ylabel = ax.get_yaxis().get_label()
x, y = ylabel.get_position()
ylabel.set_position((x, y - 0.15))  # You can adjust the 0.1 to whatever works best

ax.legend(loc=(0.41,0.65))

plt.savefig(f"/eos/project/c/contrast/public/cl/www/analysis/{training}/{label}.png",dpi=300,bbox_inches='tight')
plt.savefig(f"/eos/project/c/contrast/public/cl/www/analysis/{training}/{label}.pdf",dpi=300,bbox_inches='tight')

get_tpr_fpr()
get_unc()
#print(tprs,fprs)
