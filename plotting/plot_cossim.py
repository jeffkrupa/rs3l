import numpy as np
import matplotlib.pyplot as plt
from plot_utils import *

plt.style.use('/afs/cern.ch/user/j/jekrupa/public/rs3l/plotting/rs3l.mplstyle')

to_plot = [np.load("per_epoch/mar20_run0_8outputdims.npz"),np.load("per_epoch/mar20_run1.npz"), np.load("per_epoch/mar20_run0.npz"), ]

fig,ax=plt.subplots(figsize=(6,6))
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
#ax.set_ylim([0.,6.])
#bins = np.linspace(-1,1,25)
stack = np.stack((p["higgsqcd_ave"] for p in to_plot),axis=0)
print(stack)
mean = np.mean(stack,axis=0)
std = np.std(stack,axis=0)

ax.fill_between(np.arange(1,11,2), mean-std, mean+std)

#plt.text(0.075,0.6,f"Epoch {epoch}",transform=ax.transAxes,fontsize=20)
#plt.axvline(higgsqcd_ave,linewidth=1.0,label='$\\vec{v}_{\mathrm{higgs}_\mathrm{avg}} \cdot \\vec{v}_{\mathrm{qcd}_\mathrm{avg}}$',color='red')
#plt.axvline(qcdhiggs_ave,label='$\\vec{v}_{\mathrm{qcd}_\mathrm{avg}} \cdot \\vec{v}_{\mathrm{higgs}_\mathrm{avg}}$',color='k')
plt.xlabel("Training step")
plt.ylabel("cosine similarity")
plt.savefig(f"/eos/project/c/contrast/public/cl/www/analysis/dec23/plots/cossim.png", dpi=300,bbox_inches='tight')
plt.savefig(f"/eos/project/c/contrast/public/cl/www/analysis/dec23/plots/cossim.pdf", dpi=300,bbox_inches='tight')

