import os, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import mplhep as hep
import seaborn as sns
import matplotlib.patches as mpatches
from collections import Counter
plt.style.use(hep.style.CMS)
plt.style.use('/afs/cern.ch/user/j/jekrupa/public/rs3l/plotting/rs3l.mplstyle')

 
basepath = "/eos/project/c/contrast/public/cl/www/analysis/"
#common_fpr = np.concatenate((np.linspace(-0.001,0.00001,1000),np.linspace(0.00001,0.99999,10000),np.linspace(0.99999,1.0001,1000)))


plot_A = {
    "Fully-supervised (8M)"      : ["Graph-ntrain=8e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED/"],
    "Fine-tuned (3M, floating)"  : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED/", "Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun1,COMPLETED", "Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun2,COMPLETED"],
    "Fine-tuned (3M, fixed)"     : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED","Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun1,COMPLETED", "Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun2,COMPLETED"] ,
}

plot_B = {
    "Fully-supervised (6M)"      : ["Graph-ntrain=6e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED/", "Graph-ntrain=6e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun1,COMPLETED/", "Graph-ntrain=6e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun2,COMPLETED/"],
    "Fully-supervised (1M)"      : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED", "Graph-ntrain=1e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun1,COMPLETED", "Graph-ntrain=1e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun2,COMPLETED"],
}


plot_C = {
    "Fine-tuned (3M, floating)"  : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED/", "Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun1,COMPLETED", "Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun2,COMPLETED"],
    "Fine-tuned (1M, floating)"  : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED/", "Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun1,COMPLETED", "Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun2,COMPLETED"],

}


plot_D = {
    "Fine-tuned (3M, floating, 5layer)"  : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,fivelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED/","Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,fivelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun2,COMPLETED"],
    "Fine-tuned (1M, floating, 5layer)"  : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,fivelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED/", "Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,fivelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun1,COMPLETED"],

}

plot_E = {
    "Fine-tuned (3M, floating)" :: ["Graph-ntrain=3e6,nval=2e6,augs=012,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0_seed_fsr,COMPLETED", "Graph-ntrain=3e6,nval=2e6,augs=012,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0_seed_fsr,repeatedrun1,COMPLETED","Graph-ntrain=3e6,nval=2e6,augs=012,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0_seed_fsr,repeatedrun2,COMPLETED"]
    
}

color_dict = {
    "Fully-supervised (8M)"     : "black",
    "Fully-supervised (6M)"     : "grey",
    "Fully-supervised (1M)"     : "lightgrey",
    "Fine-tuned (3M, floating)" : "indianred",
    "Fine-tuned (1M, floating)" : "fuchsia",
    "Fine-tuned (3M, fixed)"    : "steelblue",
    "Fine-tuned (3M, floating, 5layer)" : "red",
    "Fine-tuned (1M, floating, 5layer)" : "purple",
    
}

#os.system(f"mkdir {basepath}/plot_A/")

def make_average_roc(plot_dict,title):
    # Initialize lists to store TPR and FPR values
    tpr_list = []
    fpr_list = []
    print(plot_dict.values())
    # Read each CSV file and append the values to the lists
    is_wz = "wz_zz" in plot_dict.values()
    fig,ax = plt.subplots(figsize=(9,8))

    for label, csv_paths in plot_dict.items():
        common_tpr = np.linspace(0, 1, 10000)
        fpr_interpolated = []
        for path in csv_paths:
            data = pd.read_csv(f"{basepath}/{path}/ROC_nominal.csv")
            fpr = data["1.0.1"].values
            tpr = data["1.0"].values

            interp_fpr = interpolate.interp1d(tpr, fpr, kind='linear', bounds_error=False, fill_value="extrapolate")
            fpr_interpolated.append(interp_fpr(common_tpr))

        # Calculate mean and standard deviation of TPR on the common FPR scale
        fpr_mean = np.mean(fpr_interpolated, axis=0)
        fpr_std = np.std(fpr_interpolated, axis=0)
           
        # Plot ROC curve with error bars
        ax.plot(common_tpr, fpr_mean, label=label, linestyle='-', color=color_dict[label], linewidth=2.,alpha=0.5,)
        ax.fill_between(common_tpr, fpr_mean - fpr_std, fpr_mean + fpr_std, alpha=0.)

    if is_wz:
        ax.set_ylim([0.001,1.002])
        ax.set_xlim([0.001,1.002])
    else:
        ax.set_ylim([0.003,.08])
        ax.set_xlim([0.25,1.03])

    ax.set_xlabel("Higgs acceptance",fontsize=24)
    ax.set_yscale("log")
    ax.set_ylabel(f"QCD fake rate",fontsize=24)
    ax.set_yscale('log')
    plt.grid(which='both')
    plt.legend(fontsize=18)

    plt.tight_layout() 
    plt.savefig(f"{basepath}/{title}.png")
    plt.savefig(f"{basepath}/{title}.pdf")

def make_average_unc(plot_dict,title):


  for eff in [0.5,0.6,0.7,0.8,0.9,]:
      fig,ax = plt.subplots(figsize=(9,8))
  
      for label, csv_paths in plot_dict.items():
          print(label)
          dataframes = [pd.read_csv(f"{basepath}/{file}/change_in_fpr.csv") for file in csv_paths]
          combined_df = pd.concat(dataframes)
          melted_df = combined_df.melt(id_vars='Variation', var_name='Efficiency', value_name='Value')
          eff_df = melted_df[melted_df['Efficiency'] == str(eff)]
   
          sns.violinplot(x='Variation', y='Value', data=eff_df,color=color_dict[label],alpha=0.2,label=label)
          #dfs = [pd.read_csv(f"{basepath}/{file}/change_in_fpr.csv").set_index('Variation').abs() for file in csv_paths]
          #average_df = sum(dfs) / len(dfs)
          #for path in csv_paths:
          #    data = pd.read_csv(f"{basepath}/{path}/change_in_fpr.csv")
          #print(average_df)
      patches = [mpatches.Patch(color=color_dict[label], label=label, alpha=0.5) for label, _ in plot_dict.items()] 
      ax.set_xlabel('Variation',fontsize=24)
      ax.set_ylabel(f'change in FPR (TPR={eff})',fontsize=24)
      plt.legend(handles=patches,fontsize=18)
      plt.tight_layout()
      plt.savefig(f"{basepath}/{title}_unc_{eff}.png")
      plt.savefig(f"{basepath}/{title}_unc_{eff}.pdf")
print({**plot_A,**plot_B})

make_average_roc({**plot_A,**plot_B},"roc_A")
make_average_unc(plot_A,"uncs_A")
make_average_roc(plot_C,"roc_C")
make_average_unc(plot_C,"uncs_C")
make_average_roc({**plot_C,**plot_D},"roc_D")
make_average_unc({**plot_C,**plot_D},"D")
make_average_roc({**plot_C,**plot_D},"roc_E")
make_average_unc({**plot_C,**plot_D},"E")
