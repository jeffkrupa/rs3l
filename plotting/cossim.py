import h5py
import numpy as np
import sys
from sklearn.preprocessing import normalize
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import glob
import tqdm
import mplhep as hep
import os 
from matplotlib.ticker import FormatStrFormatter
import argparse
from plot_utils import *
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--ipath', action='store', type=str, help="String matching path format.")
args = parser.parse_args()

epochs = np.arange(61,73,2)

list_higgsqcd_ave = [] 
dict_angles_ave = defaultdict(list) 
for epoch in tqdm.tqdm(epochs):
    all_jf = {}
    all_angles = defaultdict(list) 
    path = f"{args.ipath}_epoch{epoch}"
    print(path) 
    training = get_last_part_of_ipath(path)
    for i in glob.glob(path+"/*10*h5"):
        with h5py.File(i, "r") as f:
            jf = f['jet_features'][()]
            jt = f['jet_type'][()].flatten()
            vt = f['jet_vartype'][()].flatten()
            higgs_jf = normalize(f['jet_features'][()][(jt==4) & (vt==-1)],axis=1)
            qcd_jf   = normalize(f['jet_features'][()][(jt!=4) & (vt==-1)],axis=1)
            print(i)
            if f'higgs_epoch' not in all_jf:
                all_jf['higgs'] = higgs_jf
            else:
                all_jf['higgs'] = np.concatenate((all_jf['higgs'],higgs_jf))
            if f'qcd_epoch{epoch}' not in all_jf:
                all_jf['qcd'] = qcd_jf
            else:
                all_jf['qcd'] = np.concatenate((all_jf['qcd'],qcd_jf))
            #calculate dot(anchor, aug) and then take average
            for itype in range(1,9):
                #similarities_tmp = [] 
                for i in range(0,jf.shape[0],2):
                    ianchor = i
                    iaugmentation = i+1 
                    if jt[ianchor] != itype: continue
                    similarity = dot(jf[ianchor], jf[iaugmentation])
                    all_angles[itype].append(similarity)

    for itype in range(1,9):
        dict_angles_ave[itype].append(np.mean(all_angles[itype],axis=0)) 
    ave_higgs = make_unit_vector(np.mean(all_jf['higgs'],axis=0))    
    ave_qcd = make_unit_vector(np.mean(all_jf['qcd'],axis=0))
    higgsqcd_ave = dot(ave_higgs,ave_qcd)
    list_higgsqcd_ave.append(higgsqcd_ave)

np.savez(f"per_epoch/{get_last_part_of_ipath(args.ipath)}.npz",higgsqcd_ave=list_higgsqcd_ave,q_angle_ave=dict_angles_ave[1],c_angle_ave=dict_angles_ave[2],b_angle_ave=dict_angles_ave[3],H_angle_ave=dict_angles_ave[4],gqq_angle_ave=dict_angles_ave[5],gcc_angle_ave=dict_angles_ave[6],gbb_angle_ave=dict_angles_ave[7],ggg_angle_ave=dict_angles_ave[8])
sys.exit()

ave = {}
ave['higgs'] = make_unit_vector(np.mean(all_jf['higgs'],axis=0))    
ave['qcd'] = make_unit_vector(np.mean(all_jf['qcd'],axis=0))

higgsqcd_ave = dot(ave['higgs'],ave['qcd'])
qcdhiggs_ave = dot(ave['qcd'],ave['higgs'])

fig,ax=plt.subplots(figsize=(7,6))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_ylim([0.,6.])
bins = np.linspace(-1,1,25)
for s in to_plot:
    cos_sim = dot(all_jf[s[0]], ave[s[1]])
    plt.hist(cos_sim,density=True,bins=bins,histtype='step',label='$\\vec{v}_\mathrm{%s} \cdot \\vec{v}_{\mathrm{%s}_\mathrm{avg}}$'%(s[0],s[1]))
plt.text(0.075,0.6,f"Epoch {epoch}",transform=ax.transAxes,fontsize=20)
plt.axvline(higgsqcd_ave,linewidth=1.0,label='$\\vec{v}_{\mathrm{higgs}_\mathrm{avg}} \cdot \\vec{v}_{\mathrm{qcd}_\mathrm{avg}}$',color='red')
#plt.axvline(qcdhiggs_ave,label='$\\vec{v}_{\mathrm{qcd}_\mathrm{avg}} \cdot \\vec{v}_{\mathrm{higgs}_\mathrm{avg}}$',color='k')
plt.xlabel("cosine similarity")
plt.ylabel("arb. units")
plt.legend(loc=2,fontsize=16)

if int(epoch) < 10:
   epoch = f"0{epoch}"
plt.savefig(f"/home/tier3/jkrupa/public_html/cl/{sys.argv[2]}/cos_dist_epoch-{epoch}.png",dpi=300,bbox_inches='tight')

