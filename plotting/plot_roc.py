import os, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import mplhep as hep
import seaborn as sns
import matplotlib.patches as mpatches
from collections import Counter
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance
import re
from collections import OrderedDict
plt.style.use(hep.style.CMS)
plt.style.use('/afs/cern.ch/user/j/jekrupa/public/rs3l/plotting/rs3l.mplstyle')

add_n2 = True
basepath = "/eos/project/c/contrast/public/cl/www/analysis/dec23"
which_sig_effs = [0.3, 0.5, 0.7]
print("TEST")
plot_A = {
    "Fine-tuned (3M, floating)"  : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED","Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun1,COMPLETED", "Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun2,COMPLETED"],
    "Fine-tuned (3M, fixed)"     : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED","Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun1,COMPLETED", "Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun2,COMPLETED"] ,
}

plot_B = {
    #"Fully-supervised (8M, seed)" : ["Graph-ntrain=8e6,nval=2e6,augs=0,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED/",],#"Graph-ntrain=8e6,nval=2e6,augs=0,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun1,COMPLETED/"],
    "Fully-supervised (8M)"       : ["Graph-ntrain=8e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED/"],
    #"Fully-supervised (6M)"      : ["Graph-ntrain=6e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED/", "Graph-ntrain=6e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun1,COMPLETED/", ], #"Graph-ntrain=6e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun2,COMPLETED/"],
    #"Fully-supervised (4M)"      : ["Graph-ntrain=4e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED/", ],#"Graph-ntrain=4e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun1,COMPLETED/", "Graph-ntrain=4e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun2,COMPLETED/"],
    "Fully-supervised (3M)"      : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED/", "Graph-ntrain=3e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun1,COMPLETED/", "Graph-ntrain=3e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun2,COMPLETED/"],
    "Fully-supervised (1M)"      : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED", "Graph-ntrain=1e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun1,COMPLETED", "Graph-ntrain=1e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun2,COMPLETED"],
}


plot_seed = {
    "Fully-supervised (8M, seed)" : ["Graph-ntrain=8e6,nval=2e6,augs=0,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED/",],#"Graph-ntrain=8e6,nval=2e6,augs=0,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun1,COMPLETED/"],
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
    "Fine-tuned (3M, floating)" : ["Graph-ntrain=3e6,nval=2e6,augs=012,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0_seed_fsr,COMPLETED", "Graph-ntrain=3e6,nval=2e6,augs=012,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0_seed_fsr,repeatedrun1,COMPLETED","Graph-ntrain=3e6,nval=2e6,augs=012,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0_seed_fsr,repeatedrun2,COMPLETED"]
    
}

plot_F = {
    "Fine-tuned W vs QCD (1M)" : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,wz_qcd,RS3Lbase=mar20_run0,COMPLETED","Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,wz_qcd,RS3Lbase=mar20_run0,repeatedrun1,COMPLETED","Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,wz_qcd,RS3Lbase=mar20_run0,repeatedrun2,COMPLETED",] ,
    "Fine-tuned W vs QCD (1M on 3M RS3L)" : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,wz_qcd,RS3Lbase=mar20_run0_3M,COMPLETED","Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,wz_qcd,RS3Lbase=mar20_run0_3M,repeatedrun1,COMPLETED",] ,
    "Fine-tuned W vs QCD (3M)" : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,wz_qcd,RS3Lbase=mar20_run0,COMPLETED","Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,wz_qcd,RS3Lbase=mar20_run0,repeatedrun1,COMPLETED","Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,wz_qcd,RS3Lbase=mar20_run0,repeatedrun2,COMPLETED"] ,
    "Fine-tuned W vs QCD (3M on 3M RS3L)" : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,wz_qcd,RS3Lbase=mar20_run0_3M,COMPLETED","Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,wz_qcd,RS3Lbase=mar20_run0_3M,repeatedrun1,COMPLETED",],#"Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,wz_qcd,RS3Lbase=mar20_run0,repeatedrun2,COMPLETED"] ,
}

plot_G = {
    "Fully-supervised W vs QCD (1M)" : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,wz_qcd,COMPLETED","Graph-ntrain=1e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,wz_qcd,repeatedrun1,COMPLETED","Graph-ntrain=1e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,wz_qcd,repeatedrun2,COMPLETED",],
    "Fully-supervised W vs QCD (3M)" : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,wz_qcd,COMPLETED","Graph-ntrain=3e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,wz_qcd,repeatedrun1,COMPLETED"],
    #"Fully-supervised W vs QCD (8M)" : ["Graph-ntrain=8e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,wz_qcd,COMPLETED"],
}

plot_H = {
    "Fine-tuned (1M on 5M RS3L)"  : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED/", "Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun1,COMPLETED"],
    "Fine-tuned (3M on 5M RS3L)"  : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED/","Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun1,COMPLETED/" ],
    
    "Fine-tuned (1M on 3M RS3L)" : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0_3M,COMPLETED","Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0_3M,repeatedrun1,COMPLETED"],
    "Fine-tuned (3M on 3M RS3L)" : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0_3M,COMPLETED","Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0_3M,repeatedrun1,COMPLETED",],
    #"Fully-supervised (6M)" : ["Graph-ntrain=6e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED"], 
    #"Fully-supervised (8M)" : ["Graph-ntrain=8e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED"], 
}


#1 ROC:
#- fully supervised all aug
#- fully supervised, only nominal (+seed)
#- SSL only seed
#- ssl seed + FSR
#- ssl all aug
plot_I = {
    "Fully-supervised (8M, all augs)"     : ["Graph-ntrain=8e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED"],
    "Fully-supervised (8M, seed)"         : ["Graph-ntrain=8e6,nval=2e6,augs=0,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED"],
    "Fine-tuned (3M, floating, seed)"     : ["Graph-ntrain=3e6,nval=2e6,augs=0,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED"],
    "Fine-tuned (3M, floating, seed+FSR)" : ["Graph-ntrain=3e6,nval=2e6,augs=012,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0_seed_fsr,COMPLETED"],
    "Fine-tuned (3M, floating, all augs)"           : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED"],
}

plot_J = {
    "Fine-tuned (1M, fixed, 8 dims)" : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED"],
    "Fine-tuned (1M, fixed, 128 dims)" : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0_128outputdims,COMPLETED"],
    "Fine-tuned (1M, floating, 128 dims)" : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0_128outputdims,COMPLETED"],
    "Fine-tuned (3M, floating, 8 dims)" : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED"],
    "Fine-tuned (3M, floating, 128 dims)" : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0_128outputdims,COMPLETED"],
    "Fully-supervised (8M, 8 dims)" : ["Graph-ntrain=8e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED"],
    "Fully-supervised (8M, 128 dims)" : ["Graph-ntrain=8e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,128outputnodes,COMPLETED"],
}


color_dict = {
    "$\mathrm{N_2}$" : "limegreen",
    "Fully-supervised (8M)"     : "darkblue",
    "Fully-supervised (8M, all augs)"     : "#404040",
    "Fully-supervised (8M, seed)"     : "#808080",
    "Fully-supervised (6M)"     : "#333333",
    "Fully-supervised (4M)"     : "#4C4C4C",
    "Fully-supervised (3M)"     : "dodgerblue",
    "Fully-supervised (1M)"     : "cyan",
    "Fine-tuned (1M, floating)" : "fuchsia",
    "Fine-tuned (3M, floating)" : "red",
    "Fine-tuned (3M, floating, seed)" : "indianred",
    "Fine-tuned (3M, floating, seed+FSR)" : "fuchsia",
    "Fine-tuned (3M, floating, all augs)" : "red",
    "Fine-tuned (1M on 5M RS3L)" : "fuchsia",
    "Fine-tuned (3M on 5M RS3L)" : "indianred",
    "Fine-tuned (3M, fixed)"    : "pink",
    "Fine-tuned (3M, floating, 5layer)" : "red",
    "Fine-tuned (1M, floating, 5layer)" : "purple",
    "Fully-supervised W vs QCD (1M)" : "#808080",
    "Fully-supervised W vs QCD (3M)" : "#BFBFBF",
    "Fully-supervised W vs QCD (8M)" : "#404040",
    "Fine-tuned W vs QCD (1M)" : "fuchsia",
    "Fine-tuned W vs QCD (3M)" : "indianred", 
    "Fine-tuned W vs QCD (1M on 3M RS3L)" : "fuchsia",
    "Fine-tuned W vs QCD (3M on 3M RS3L)" : "indianred", 
    "Fine-tuned (1M on 3M RS3L)" : "steelblue",
    "Fine-tuned (3M on 3M RS3L)" : "darkblue",
}

#os.system(f"mkdir {basepath}/plots/")

def make_average_roc(plot_dict,title):
    # Initialize lists to store TPR and FPR values
    tpr_list = []
    fpr_list = []
    #print(plot_dict.values())
    # Read each CSV file and append the values to the lists
    #print(plot_dict.values()[0][0])
    is_wz = any("wz_zz" in entry[0] for entry in  plot_dict.values())
    is_wqcd = any("wz_qcd" in entry[0] for entry in plot_dict.values())
    fig,ax = plt.subplots(figsize=(9,8))
    colors_tmp = ["green", "blue"]
    for label, csv_paths in plot_dict.items():
        common_tpr = np.linspace(0, 1, 10000)
        fpr_interpolated = []
        for i0, path in enumerate(csv_paths):
            data = pd.read_csv(f"{basepath}/{path}/ROC_nominal.csv")
            fpr = data["1.0.1"].values
            tpr = data["1.0"].values

            interp_fpr = interpolate.interp1d(tpr, fpr, kind='linear', bounds_error=False, fill_value="extrapolate")
            fpr_interpolated.append(interp_fpr(common_tpr))
            #ax.plot(tpr, fpr, linestyle="--", color=colors_tmp[i0], linewidth =0.5, alpha=1)
 
        # Calculate mean and standard deviation of TPR on the common FPR scale
        fpr_mean = np.mean(fpr_interpolated, axis=0)
        fpr_std = np.std(fpr_interpolated, axis=0)
           
        # Plot ROC curve with error bars
        ax.plot(common_tpr, fpr_mean, label=label+f" ({len(csv_paths)}x)", linestyle='-', color=color_dict[label], linewidth=2.,alpha=1.,)
        ax.fill_between(common_tpr, fpr_mean - 1*fpr_std, fpr_mean + 1*fpr_std, color=color_dict[label], alpha=0.6)


    if add_n2:
        if is_wqcd:
           process = "wz_qcd"
        else:
           process = "h_qcd"
        data = pd.read_csv(f"{basepath}/Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,{process},RS3Lbase=mar20_run0,COMPLETED/ROC_nominal_n2.csv")
        print(data.columns)
        data.columns = ["tpr","fpr"]
        fpr = data["fpr"].values
        tpr = data["tpr"].values
        ax.plot(tpr,fpr, label="$\mathrm{N_2}$", linestyle='-', color=color_dict["$\mathrm{N_2}$"], linewidth=2.,alpha=1.)
    if is_wqcd:
        sig_name = "W"
    else:
        sig_name = "Higgs" 
    ax.set_xlabel(f"{sig_name} acceptance",fontsize=24)
    ax.set_ylabel(f"QCD fake rate",fontsize=24)
    plt.grid(which='both')
    # Get handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Sort them by labels
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: x[1])

    # Unzip the sorted tuples
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)

    # Create a new legend
    ax.legend(sorted_handles, sorted_labels,fontsize=18.5, loc =(0.37,0.01))

    #plt.show()
    #plt.legend(fontsize=18)
    plt.tight_layout() 
    plt.savefig(f"{basepath}/plots/{title}.png")
    plt.savefig(f"{basepath}/plots/{title}.pdf")

    if is_wz:
        ax.set_ylim([0.001,1.002])
        ax.set_xlim([0.001,1.002])
    else:
        #ax.set_ylim([0.002,.07])
        ax.set_ylim([0.0003,.07])
        ax.set_xlim([0.15,1.03])
    ax.set_yscale('log')
    plt.savefig(f"{basepath}/plots/{title}_logy.png")
    plt.savefig(f"{basepath}/plots/{title}_logy.pdf")

def make_mistag_table(plot_dict,title):

    is_wqcd = any("wz_qcd" in entry[0] for entry in plot_dict.values())
    table = {}
    table_std = {}
    for label, csv_paths in plot_dict.items():
        for eff in which_sig_effs:
            mistag_rate_tmp = []
            for path in csv_paths:
                #print(path)
                data = pd.read_csv(f"{basepath}/{path}/ROC_nominal.csv")
                fpr = data["1.0.1"].values[::-1]
                tpr = data["1.0"].values[::-1]
                print(label, tpr[:100])
                icut = np.searchsorted(tpr, eff,)
                mistag_rate_tmp.append(1./fpr[icut])
            #break
            table[f"{label}_{eff}"] = np.mean(mistag_rate_tmp)
            table_std[f"{label}_{eff}"] = np.std(mistag_rate_tmp)
    if add_n2:
        if is_wqcd:
            training_label = "wz_qcd"
        else:
            training_label = "h_qcd"

        for eff in which_sig_effs:
            data = pd.read_csv(f"{basepath}/Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,{training_label},RS3Lbase=mar20_run0,COMPLETED/ROC_nominal_n2.csv")
            data.columns =  ["tpr","fpr"]
            tpr = data["tpr"].values 
            fpr = data["fpr"].values
            icut = np.searchsorted(tpr,eff)
            table[f"$\mathrm{{N_2}}_{eff}"] = 1./fpr[icut] 
    labels = []
    efficiencies = []
    values_with_uncertainty = []
    for key, value in table.items():
        label, eff = key.rsplit('_', 1)
        labels.append(label)
        efficiencies.append(eff)
        value_rounded = round(value)
        print(label)
        if "N_2" in key: #treating N2 as no unc 
            uncertainty = 0.
        else:
            uncertainty = round(table_std[key])
        value_uncertainty_str = f"{value_rounded}"
        if uncertainty > 0.:
            value_uncertainty_str += f" \pm {uncertainty}"
        values_with_uncertainty.append(value_uncertainty_str)


        #print(key, eff, round(value,2)) 
    # Creating a DataFrame
    df = pd.DataFrame({'Training setup': labels, 'Higgs efficiency': efficiencies, 'Value': values_with_uncertainty})
    df = df.sort_values(by=df.columns[0], ascending=True)
    # Pivoting the DataFrame
    pivot_df = df.pivot(index='Training setup', columns='Higgs efficiency', values='Value')
    #latex_table = pivot_df.to_latex()
    # Function to bold the maximum value in each column
    ''''
    # Function to bold the maximum value in each column
    def bold_max(s):
        is_max = s == s.max()
        return ['\\textbf{' + str(v) + '}' if max_val else str(v) for v, max_val in zip(s, is_max)]
    
    # Applying the function to each column in the DataFrame
    pivot_df_bold_max = pivot_df.apply(bold_max)
    
    # Generating the LaTeX table with the bolded maximum values
    latex_table_bold_max = pivot_df_bold_max.to_latex(escape=False)
    
    latex_table_bold_max
    '''
    with open(f"{basepath}/plots/{title}_bkgrejection_table.txt","w",encoding='utf-8') as f:
        #f.write(latex_table_bold_max)
        f.write(pivot_df.to_latex(escape=False))
    return 
#    print(table) 


def make_average_unc(plot_dict,title):


    for eff in [0.5,0.6,0.7,0.8,0.9,]:
        fig,ax = plt.subplots(figsize=(9,8))
  
        for label, csv_paths in plot_dict.items():
            dataframes = [pd.read_csv(f"{basepath}/{file}/change_in_fpr.csv") for file in csv_paths]
            combined_df = pd.concat(dataframes)
            melted_df = combined_df.melt(id_vars='Variation', var_name='Efficiency', value_name='Value')
            eff_df = melted_df[melted_df['Efficiency'] == str(eff)]
     
            sns.violinplot(x='Variation', y='Value', data=eff_df,color=color_dict[label],alpha=0.2,label=label)
        patches = [mpatches.Patch(color=color_dict[label], label=label, alpha=0.5) for label, _ in plot_dict.items()] 
        ax.set_xlabel('Variation',fontsize=24)
        ax.set_ylabel(f'change in FPR (TPR={eff})',fontsize=24)
        plt.legend(handles=patches,fontsize=18)
        plt.tight_layout()
        plt.savefig(f"{basepath}/plots/{title}_unc_{eff}.png")
        plt.savefig(f"{basepath}/plots/{title}_unc_{eff}.pdf")

def make_divergence_table(plot_dict, title):
    is_wqcd = any("wz_qcd" in entry[0] for entry in plot_dict.values())
    if is_wqcd:
        results = {"W": {}, "qcd": {}}
        results_std = {"W": {}, "qcd": {}}
    else:
        #results = {"higgs": {}, "qcd": {}, "combined" : {}}
        #results_std = {"higgs": {}, "qcd": {}, "combined" : {}}
        results = { "combined" : {}}
        results_std = { "combined" : {}}
   
    for label, csv_paths in plot_dict.items():
        for iprocess in results.keys():
            results[iprocess][label] = {}
            results_std[iprocess][label] = {}

            for ivariation in ["seed", "fsrRenHi", "fsrRenLo", "herwig"]:
                distances = []

                for csv in csv_paths:
                    if "combined" in iprocess:
                        arr_nom = np.concatenate((pd.read_csv(f"{basepath}/{csv}/higgs_nominal.csv")["val"].to_numpy(), pd.read_csv(f"{basepath}/{csv}/qcd_nominal.csv")["val"].to_numpy()))
                        arr_var = np.concatenate((pd.read_csv(f"{basepath}/{csv}/higgs_{ivariation}.csv")["val"].to_numpy(), pd.read_csv(f"{basepath}/{csv}/qcd_{ivariation}.csv")["val"].to_numpy()))
                    else:
                        arr_nom = pd.read_csv(f"{basepath}/{csv}/{iprocess}_nominal.csv")["val"].to_numpy()
                        arr_var = pd.read_csv(f"{basepath}/{csv}/{iprocess}_{ivariation}.csv")["val"].to_numpy()
                    ws_distance = wasserstein_distance(arr_nom, arr_var)
                    print(iprocess,label,ivariation,ws_distance)
                    distances.append(ws_distance)

                # Average over all runs
                results[iprocess][label][ivariation] = np.mean(distances)
                results_std[iprocess][label][ivariation] = np.std(distances)
    if add_n2:
        if is_wqcd:
            training_label = "wz_qcd"
        else:
            training_label = "h_qcd"
        for iprocess in results.keys():
            results[iprocess][r"$\mathrm{N_{2}}$"] = {}
            for ivariation in ["seed", "fsrRenHi", "fsrRenLo", "herwig"]:
                if "combined" not in iprocess:
                    arr_nom = pd.read_csv(f"{basepath}/Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,{training_label},RS3Lbase=mar20_run0,COMPLETED/{iprocess}_nominal_n2.csv")["val"].to_numpy()        
                    arr_var = pd.read_csv(f"{basepath}/Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,{training_label},RS3Lbase=mar20_run0,COMPLETED/{iprocess}_{ivariation}_n2.csv")["val"].to_numpy()
                else:        
                    arr_nom = np.concatenate((pd.read_csv(f"{basepath}/Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,{training_label},RS3Lbase=mar20_run0,COMPLETED/higgs_nominal_n2.csv")["val"].to_numpy(), pd.read_csv(f"{basepath}/Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,{training_label},RS3Lbase=mar20_run0,COMPLETED/qcd_nominal_n2.csv")["val"].to_numpy()))        
                    arr_var = np.concatenate((pd.read_csv(f"{basepath}/Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,{training_label},RS3Lbase=mar20_run0,COMPLETED/higgs_{ivariation}_n2.csv")["val"].to_numpy() , pd.read_csv(f"{basepath}/Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,{training_label},RS3Lbase=mar20_run0,COMPLETED/qcd_{ivariation}_n2.csv")["val"].to_numpy()))        
                ws_distance = wasserstein_distance(arr_nom, arr_var) 
                results[iprocess][r"$\mathrm{N_{2}}$"][ivariation] = ws_distance
    def to_latex_scientific_notation(num):
        if num == 0:
            return "0"
        else:
            exponent = int(f"{num:e}".split('e')[-1])
            base = num / (10 ** exponent)
            # Format the string for LaTeX
            if exponent == 0:
                return f"{base:.3f}"
            else:
                return f"{base:.3f} \\times 10^{{{exponent}}}"
    latex_tables = {}
    # The rest is just formatting the output table
    for iprocess in results.keys():
        # Debugging print
        #print(f"Processing: {iprocess}")
        #print(f"Results for {iprocess}: {results[iprocess]}")

        data = []
        for label, variations in results[iprocess].items():
            row = [label]
            if "N_{2}" in label: #No unc.
                row += [f"{variations[ivar]}" for ivar in ["seed", "fsrRenHi", "fsrRenLo", "herwig"]]
            else:
                row += [f"{variations[ivar]} \pm {results_std[iprocess][label][ivar]}" for ivar in ["seed", "fsrRenHi", "fsrRenLo", "herwig"]]
            data.append(row)

        def extract_and_process(s):
            # Extract numbers from the string
            numbers = [float(x) for x in re.findall(r"[0-9\.e\-]+", s)]
            # Return the numbers as value and uncertainty (if exists)
            return numbers[0], numbers[1] if len(numbers) > 1 else None

        def to_combined_scientific_notation(value, error):
            # Convert to scientific notation
            value_str = f"{value:.2e}"
            base, exponent = value_str.split('e')
            exponent = int(exponent)
            if error is not None:
                if error > 0. and error/value > 0.005:
                    error_scaled = error / (10 ** exponent)
                    return f"({float(base):.2f} \\pm {error_scaled:.2f}) \\times 10^{{{exponent}}}"
                else: 
                    return f"{float(base):.2f} \\times 10^{{{exponent}}}" 
            else:
                return f"{float(base):.2f} \\times 10^{{{exponent}}}" 
      
        # Process the data to extract values and uncertainties
        processed_data = []
        for row in data:
            processed_row = [row[0]]
            for entry in row[1:]:
                value, uncertainty = extract_and_process(entry)
                processed_row.append((value, uncertainty))
            processed_data.append(processed_row)

        # Find the smallest non-zero value
        # Divide and round
        for row in processed_data:
            for i, (value, uncertainty) in enumerate(row[1:], start=1):
                row[i] = to_combined_scientific_notation(value, uncertainty)
        df = pd.DataFrame(processed_data, columns=["Training setup", "Seed", "fsrRenHi", "fsrRenLo", "Herwig"])
        df = df.sort_values(by=df.columns[0], ascending=True)
        latex_tables[iprocess] = df.to_latex(index=False,escape=False)

    # Creating DataFrames and LaTeX tables
    for iprocess in results.keys():
        caption = f"Wasserstein distances for {iprocess}"
        label = f"table:{iprocess}"
        latex_table_with_caption = (
            '\\begin{table}[ht]\n' 
            '\\centering\n' 
            f'{latex_tables[iprocess]}\n'
            f'\\caption{{{caption}}}\n'
            f'\\label{{{label}}}\n'
            '\\end{table}'
        )

        file_name = f"{basepath}/plots/{title}_table_{iprocess}.txt"
        with open(file_name, 'w') as file:
            file.write(latex_table_with_caption)

        latex_tables[iprocess] = latex_table_with_caption
    
        print(latex_table_with_caption)
    print(latex_tables)
    return latex_tables

make_divergence_table(OrderedDict({**plot_A,**plot_B}),"A")
make_average_roc(OrderedDict({**plot_A,**plot_B}),"A")
make_mistag_table(OrderedDict({**plot_A,**plot_B}),"A")

make_divergence_table(OrderedDict({**plot_A,**plot_B,**plot_seed}),"Awithseed")
make_average_roc(OrderedDict({**plot_A,**plot_B,**plot_seed}),"Awithseed")
make_mistag_table(OrderedDict({**plot_A,**plot_B,**plot_seed}),"Awithseed")
#make_mistag_table(plot_J,"J")
#make_divergence_table(plot_J,"J")
#make_average_roc(plot_J,"J")

#make_average_roc({**plot_F,**plot_G},"WvsQCD")
#make_mistag_table({**plot_F,**plot_G},"WvsQCD")
#make_divergence_table({**plot_F,**plot_G},"WvsQCD")

#make_divergence_table(plot_H, "H")
#make_average_roc(plot_H,"H")
#make_mistag_table(plot_H,"H")

#make_divergence_table(plot_I, "I")
#make_average_roc(plot_I,"I")
#make_mistag_table(plot_I,"I")
#make_average_roc(plot_C,"B")
#make_average_roc({**plot_C,**plot_D},"C")
#make_average_roc({**plot_C,**plot_D},"D")

#make_average_unc(plot_A,"A")
#make_average_unc(plot_C,"C")
#make_average_unc({**plot_C,**plot_D},"D")
#make_average_unc({**plot_C,**plot_D},"E")
