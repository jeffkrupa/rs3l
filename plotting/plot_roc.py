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
plt.style.use(hep.style.CMS)
plt.style.use('/afs/cern.ch/user/j/jekrupa/public/rs3l/plotting/rs3l.mplstyle')

 
basepath = "/eos/project/c/contrast/public/cl/www/analysis/dec23"
#common_fpr = np.concatenate((np.linspace(-0.001,0.00001,1000),np.linspace(0.00001,0.99999,10000),np.linspace(0.99999,1.0001,1000)))


plot_A = {
    "Fine-tuned (3M, floating)"  : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED/", ], #"Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun1,COMPLETED", "Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun2,COMPLETED"],
    "Fine-tuned (3M, fixed)"     : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED",], #"Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun1,COMPLETED", "Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,fixed_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,repeatedrun2,COMPLETED"] ,
}

plot_B = {
    "Fully-supervised (8M)"      : ["Graph-ntrain=8e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED/"],
    "Fully-supervised (6M)"      : ["Graph-ntrain=6e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED/", ], #"Graph-ntrain=6e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun1,COMPLETED/", "Graph-ntrain=6e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun2,COMPLETED/"],
    #"Fully-supervised (4M)"      : ["Graph-ntrain=4e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED/", "Graph-ntrain=4e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun1,COMPLETED/", "Graph-ntrain=4e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun2,COMPLETED/"],
    "Fully-supervised (1M)"      : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED", ], #"Graph-ntrain=1e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun1,COMPLETED", "Graph-ntrain=1e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,repeatedrun2,COMPLETED"],
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
    "Fine-tuned W vs QCD (1M)" : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,wz_qcd,RS3Lbase=mar20_run0,COMPLETED"] ,
    "Fine-tuned W vs QCD (3M)" : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,wz_qcd,RS3Lbase=mar20_run0,COMPLETED"] ,
}

plot_G = {
    "Fully-supervised W vs QCD (1M)" : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,wz_qcd,COMPLETED"],
    "Fully-supervised W vs QCD (3M)" : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,wz_qcd,COMPLETED"],
    "Fully-supervised W vs QCD (8M)" : ["Graph-ntrain=8e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,wz_qcd,COMPLETED"],
}

plot_H = {
    "Fine-tuned (1M, floating) on 5M RS3L space"  : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED/", ],
    "Fine-tuned (3M, floating) on 5M RS3L space"  : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0,COMPLETED/", ],
    
    "Fine-tuned (1M, floating) on 3M RS3L space" : ["Graph-ntrain=1e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0_3M,COMPLETED"],
    "Fine-tuned (3M, floating) on 3M RS3L space" : ["Graph-ntrain=3e6,nval=2e6,augs=0123,fine-tuned,floating_weights,onelayerMLP,h_qcd,RS3Lbase=mar20_run0_3M,COMPLETED"],
    "Fully-supervised (6M)" : ["Graph-ntrain=6e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED"], 
    "Fully-supervised (8M)" : ["Graph-ntrain=8e6,nval=2e6,augs=0123,fully-supervised,floating_weights,onelayerMLP,h_qcd,COMPLETED"], 
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
color_dict = {
    "Fully-supervised (8M)"     : "#404040",
    "Fully-supervised (8M, all augs)"     : "#404040",
    "Fully-supervised (8M, seed)"     : "#808080",
    "Fully-supervised (6M)"     : "#808080",
    "Fully-supervised (4M)"     : "#BFBFBF",
    "Fully-supervised (1M)"     : "#F0F0F0",
    "Fine-tuned (1M, floating)" : "fuchsia",
    "Fine-tuned (3M, floating)" : "indianred",
    "Fine-tuned (3M, floating, seed)" : "indianred",
    "Fine-tuned (3M, floating, seed+FSR)" : "fuchsia",
    "Fine-tuned (3M, floating, all augs)" : "red",
    "Fine-tuned (1M, floating) on 5M RS3L space" : "fuchsia",
    "Fine-tuned (3M, floating) on 5M RS3L space" : "indianred",
    "Fine-tuned (3M, fixed)"    : "steelblue",
    "Fine-tuned (3M, floating, 5layer)" : "red",
    "Fine-tuned (1M, floating, 5layer)" : "purple",
    "Fully-supervised W vs QCD (1M)" : "#808080",
    "Fully-supervised W vs QCD (3M)" : "#BFBFBF",
    "Fully-supervised W vs QCD (8M)" : "#404040",
    "Fine-tuned W vs QCD (1M)" : "fuchsia",
    "Fine-tuned W vs QCD (3M)" : "indianred", 
    "Fine-tuned (1M, floating) on 3M RS3L space" : "steelblue",
    "Fine-tuned (3M, floating) on 3M RS3L space" : "darkblue",
}

os.system(f"mkdir {basepath}/plots/")

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
        ax.plot(common_tpr, fpr_mean, label=label, linestyle='-', color=color_dict[label], linewidth=2.,alpha=1.,)
        ax.fill_between(common_tpr, fpr_mean - fpr_std, fpr_mean + fpr_std, alpha=0.)

    if is_wqcd:
        sig_name = "W"
    else:
        sig_name = "Higgs" 
    ax.set_xlabel(f"{sig_name} acceptance",fontsize=24)
    ax.set_ylabel(f"QCD fake rate",fontsize=24)
    plt.grid(which='both')
    plt.legend(fontsize=18)
    plt.tight_layout() 
    plt.savefig(f"{basepath}/plots/{title}.png")
    plt.savefig(f"{basepath}/plots/{title}.pdf")

    if is_wz:
        ax.set_ylim([0.001,1.002])
        ax.set_xlim([0.001,1.002])
    else:
        ax.set_ylim([0.002,.07])
        ax.set_xlim([0.25,1.03])
    ax.set_yscale('log')
    plt.savefig(f"{basepath}/plots/{title}_logy.png")
    plt.savefig(f"{basepath}/plots/{title}_logy.pdf")

def make_mistag_table(plot_dict,title):

    table = {}
    for label, csv_paths in plot_dict.items():
        for path in csv_paths:
            #print(path)
            data = pd.read_csv(f"{basepath}/{path}/ROC_nominal.csv")
            fpr = data["1.0.1"].values[::-1]
            tpr = data["1.0"].values[::-1]
            #print(tpr[:10000])
            for eff in [0.3,0.5,0.7]:
                icut = np.searchsorted(tpr, eff,)
                table[f"{label}_{eff}"] = 1./fpr[icut]
            break
    labels = []
    efficiencies = []
    values = []
    for key, value in table.items():
        label, eff = key.rsplit('_', 1)
        labels.append(label)
        efficiencies.append(eff)
        values.append(round(value,2))
        #print(key, eff, round(value,2)) 
    # Creating a DataFrame
    df = pd.DataFrame({'Training setup': labels, 'Higgs efficiency': efficiencies, 'Value': values})

    # Pivoting the DataFrame
    pivot_df = df.pivot(index='Training setup', columns='Higgs efficiency', values='Value')
    #latex_table = pivot_df.to_latex()
    # Function to bold the maximum value in each column

    # Function to bold the maximum value in each column
    def bold_max(s):
        is_max = s == s.max()
        return ['\\textbf{' + str(v) + '}' if max_val else str(v) for v, max_val in zip(s, is_max)]
    
    # Applying the function to each column in the DataFrame
    pivot_df_bold_max = pivot_df.apply(bold_max)
    
    # Generating the LaTeX table with the bolded maximum values
    latex_table_bold_max = pivot_df_bold_max.to_latex(escape=False)
    
    latex_table_bold_max
     
    with open(f"{basepath}/plots/{title}_mistagrate_table.txt","w") as f:
        f.write(latex_table_bold_max)
    return 
#    print(table) 


def make_average_unc(plot_dict,title):


    for eff in [0.5,0.6,0.7,0.8,0.9,]:
        fig,ax = plt.subplots(figsize=(9,8))
  
        for label, csv_paths in plot_dict.items():
            #print(label)
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
        plt.savefig(f"{basepath}/plots/{title}_unc_{eff}.png")
        plt.savefig(f"{basepath}/plots/{title}_unc_{eff}.pdf")

def make_divergence_table(plot_dict, title):
    is_wqcd = any("wz_qcd" in entry[0] for entry in plot_dict.values())
    if is_wqcd:
        results = {"W": {}, "qcd": {}}
    else:
        results = {"higgs": {}, "qcd": {}}
   
    for label, csv_paths in plot_dict.items():
        for iprocess in results.keys():
            results[iprocess][label] = {}
            for ivariation in ["seed", "fsrRenHi", "fsrRenLo", "herwig"]:
                distances = []

                for csv in csv_paths:
                    arr_nom = pd.read_csv(f"{basepath}/{csv}/{iprocess}_nominal.csv")["val"].to_numpy()
                    his_nom, _ = np.histogram(arr_nom, bins=np.linspace(0, 1, 100), density=True)
                    arr_var = pd.read_csv(f"{basepath}/{csv}/{iprocess}_{ivariation}.csv")["val"].to_numpy()
                    his_var, _ = np.histogram(arr_var, bins=np.linspace(0, 1, 100), density=True)
                    max_len = len(arr_var)
                    ws_distance = wasserstein_distance(arr_nom[:max_len], arr_var[:max_len])
                    distances.append(ws_distance)

                # Average over all runs
                results[iprocess][label][ivariation] = np.mean(distances)
    latex_tables = {}
    for iprocess in results.keys():
        # Debugging print
        print(f"Processing: {iprocess}")
        print(f"Results for {iprocess}: {results[iprocess]}")

        data = []
        for label, variations in results[iprocess].items():
            row = [label]
            row += [variations[ivar] for ivar in ["seed", "fsrRenHi", "fsrRenLo", "herwig"]]
            data.append(row)

        df = pd.DataFrame(data, columns=["Label", "Seed", "fsrRenHi", "fsrRenLo", "Herwig"])
        latex_tables[iprocess] = df.to_latex(index=False)

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

        file_name = f"{basepath}/plots/{title}_table_{iprocess}.tex"
        with open(file_name, 'w') as file:
            file.write(latex_table_with_caption)

        latex_tables[iprocess] = latex_table_with_caption
    
        print(latex_table_with_caption)
    print(latex_tables)
    return latex_tables

make_divergence_table({**plot_A,**plot_B},"A")
make_average_roc({**plot_A,**plot_B},"A")
make_mistag_table({**plot_A,**plot_B},"A")

#make_average_roc({**plot_F,**plot_G},"WvsQCD")
#make_mistag_table({**plot_F,**plot_G},"WvsQCD")
#make_divergence_table({**plot_F,**plot_G},"WvsQCD")

make_divergence_table(plot_H, "H")
make_average_roc(plot_H,"H")
make_mistag_table(plot_H,"H")

make_divergence_table(plot_I, "I")
make_average_roc(plot_I,"I")
make_mistag_table(plot_I,"I")
#make_average_roc(plot_C,"B")
#make_average_roc({**plot_C,**plot_D},"C")
#make_average_roc({**plot_C,**plot_D},"D")

#make_average_unc(plot_A,"A")
#make_average_unc(plot_C,"C")
#make_average_unc({**plot_C,**plot_D},"D")
#make_average_unc({**plot_C,**plot_D},"E")
