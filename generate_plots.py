import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so

from pathlib import Path
from collections import namedtuple

Arguments = namedtuple('Arguments', ['title', 'legend_title', 'xlabel', 'ylabel', 'idxcolumns', 'xcolumn', 'ycolumn', 'output_file', 'input_list', 'legend'])

def generate_plot(args, legend_pos=None, legend_bbox_align=None):
    of = Path('./figures') / Path(args.output_file)
    df_list = []
    for input_file_path in args.input_list:
        try:
            ip = (Path('.') / Path(input_file_path)).resolve()
            input_file_name = ip.name
            infile = ip.as_posix()
            df = pd.read_csv(infile, header=0, index_col=args.idxcolumns)
            existing_cols = df.columns.copy()
            if args.ycolumn == 'Regret': # Use log10 scale
                df[args.legend[input_file_name]] = np.log10(df[args.ycolumn].values)
#                df[args.legend[input_file_name]] = df[args.ycolumn].values
            elif args.ycolumn == 'ProcessTime': # Scale nanonseconds to minutes
                df[args.legend[input_file_name]] = (df[args.ycolumn].values)/60e9
            elif args.ycolumn == 'SpaceUsage': # Scale bytes to MiB
                df[args.legend[input_file_name]] = (df[args.ycolumn].values)/1e6
            else: # Use raw values
                df[args.legend[input_file_name]] = df[args.ycolumn].values
            df = df.drop(existing_cols, axis=1)
            df_list.append(df)
        except IOError as e:
            print(e)
            exit()
    # End for
    
    df_full = pd.concat(df_list, axis=1).sort_index()
    df_full = df_full.reset_index()
    if "I" in df.columns.values:
        df_full = df_full.drop(["I"], axis=1)
    
    sns.set(rc={"figure.figsize":(20, 16)})
    sns.set(font_scale=3.5)
    sns.set_style("whitegrid", {'axes.grid' : False})
    fig, ax = plt.subplots(figsize=(20, 16))
    
    markers = ['o', '^', 'v', '<', '>', 's', 'd', 'p', 'h', 'X', '*', 'P']
    colors = sns.color_palette("dark")
    for i, ifp in enumerate(args.input_list):
        ip = (Path('.') / Path(ifp)).resolve()
        ycol = args.legend[ip.name]
        sns.lineplot(x=args.xcolumn, y=ycol, estimator='mean', errorbar='ci', ax=ax, data=df_full, marker=None, color=colors[i], label=ycol)
    
    #ax.set_title(args.title, {"fontsize":24})
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    if legend_pos is not None:
        plt.legend(title=args.legend_title, loc=legend_pos, bbox_to_anchor=legend_bbox_align)
    else:
        plt.legend(title=args.legend_title, loc='lower right')
    # End if
    plt.setp(ax.get_legend().get_texts(), fontsize='30') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title
#    if legend_pos is not None:
#        sns.move_legend(ax, legend_pos, bbox_to_anchor=legend_bbox_align)
    plt.tight_layout()
#    sns.despine()
    plt.savefig(of.resolve().as_posix(), format='pdf')
    print('Wrote:', of.resolve().as_posix())
    #plt.show()
# End fn generate_plot

def plot_syn_regrets_long():
    args = Arguments(title='Regret Comparison',
                     legend_title='Regret Curves',
                     xlabel='K (Episode #)',
                     ylabel='Regret (log-scale)',
                     idxcolumns=['K', 'I'],
                     xcolumn='K',
                     ycolumn='Regret',
                     output_file='long-syn-regret.pdf',
                     input_list=['output/syn-full/syn_fixed_reset_full.csv', 'output/syn-full/syn_adaptive_reset_full.csv', 'output/syn-full/syn_baseline_full.csv'],
                     legend={'syn_baseline_full.csv': 'LSVI-UCB (Baseline)',
                                'syn_fixed_reset_full.csv': 'LSVI-UCB-Fixed',
                                'syn_adaptive_reset_full.csv': 'LSVI-UCB-Adaptive'
                     }
    )
    generate_plot(args)
# End fn

def plot_syn_proctime_long():
    args = Arguments(title='Process Time Comparison',
                     legend_title='Process Time Curves',
                     xlabel='K (Episode #)',
                     ylabel='Process Time (mins)',
                     idxcolumns=['K', 'I'],
                     xcolumn='K',
                     ycolumn='ProcessTime',
                     output_file='long-syn-proctime.pdf',
                     input_list=['output/syn-full/syn_fixed_reset_full.csv', 'output/syn-full/syn_adaptive_reset_full.csv', 'output/syn-full/syn_baseline_full.csv'],
                     legend={'syn_baseline_full.csv': 'LSVI-UCB (Baseline)',
                                'syn_fixed_reset_full.csv': 'LSVI-UCB-Fixed',
                                'syn_adaptive_reset_full.csv': 'LSVI-UCB-Adaptive'
                     }
    )
    generate_plot(args, legend_pos='upper right')
# End fn

def plot_syn_spcusage_long():
    args = Arguments(title='Space Usage Comparison',
                     legend_title='Space Usage Curves',
                     xlabel='K (Episode #)',
                     ylabel='Space Usage (MB)',
                     idxcolumns=['K', 'I'],
                     xcolumn='K',
                     ycolumn='SpaceUsage',
                     output_file='long-syn-spcusage.pdf',
                     input_list=['output/syn-full/syn_fixed_reset_full.csv', 'output/syn-full/syn_adaptive_reset_full.csv', 'output/syn-full/syn_baseline_full.csv'],
                     legend={'syn_baseline_full.csv': 'LSVI-UCB (Baseline)',
                                'syn_fixed_reset_full.csv': 'LSVI-UCB-Fixed',
                                'syn_adaptive_reset_full.csv': 'LSVI-UCB-Adaptive'
                     }
    )
    generate_plot(args, legend_pos='upper right')
# End fn

def plot_syn_fixed_paramvars_regret():
    legend_syn_fixed = {
       "output-syn-base.csv": "LSVI-UCB (Baseline)",
       "output-fixed-1.csv":  "Fixed $K^{\\rho}=K^{0.5}$",
       "output-fixed-2.csv":  "Fixed $K^{\\rho}=K^{0.55}$",
       "output-fixed-3.csv":  "Fixed $K^{\\rho}=K^{0.6}$",
       "output-fixed-4.csv":  "Fixed $K^{\\rho}=K^{0.65}$",
       "output-fixed-5.csv":  "Fixed $K^{\\rho}=K^{0.7}$",
       "output-fixed-6.csv":  "Fixed $K^{\\rho}=K^{0.75}$",
    }

    args = Arguments(title='Regret: LSVI-UCB-Fixed',
                     legend_title='Regret Curves',
                     xlabel='K (Episode #)',
                     ylabel='Log-scale Regret',
                     idxcolumns=['K'],
                     xcolumn='K',
                     ycolumn='Regret',
                     output_file='syn-fixed-paramvars-regret.pdf',
                     input_list=(['output/syn-paramvars/base/output-syn-base.csv'] + ['output/syn-paramvars/fixed/output-fixed-%d.csv' % (i) for i in range(1, 7)]),
                     legend=legend_syn_fixed
    )
    generate_plot(args)
# End fn

def plot_syn_fixed_paramvars_proctime():
    legend_syn_fixed = {
       "output-syn-base.csv": "LSVI-UCB (Baseline)",
       "output-fixed-1.csv":  "Fixed $K^{\\rho}=K^{0.5}$",
       "output-fixed-2.csv":  "Fixed $K^{\\rho}=K^{0.55}$",
       "output-fixed-3.csv":  "Fixed $K^{\\rho}=K^{0.6}$",
       "output-fixed-4.csv":  "Fixed $K^{\\rho}=K^{0.65}$",
       "output-fixed-5.csv":  "Fixed $K^{\\rho}=K^{0.7}$",
       "output-fixed-6.csv":  "Fixed $K^{\\rho}=K^{0.75}$",
    }

    args = Arguments(title='Process Time: LSVI-UCB-Fixed',
                     legend_title='Process Time Curves',
                     xlabel='K (Episode #)',
                     ylabel='Process Time (mins)',
                     idxcolumns=['K'],
                     xcolumn='K',
                     ycolumn='ProcessTime',
                     output_file='syn-fixed-paramvars-proctime.pdf',
                     input_list=(['output/syn-paramvars/base/output-syn-base.csv'] + ['output/syn-paramvars/fixed/output-fixed-%d.csv' % (i) for i in range(1, 7)]),
                     legend=legend_syn_fixed
    )
    generate_plot(args, legend_pos='upper right')
# End fn

def plot_syn_fixed_paramvars_spcusage():
    legend_syn_fixed = {
       "output-syn-base.csv": "LSVI-UCB (Baseline)",
       "output-fixed-1.csv":  "Fixed $K^{\\rho}=K^{0.5}$",
       "output-fixed-2.csv":  "Fixed $K^{\\rho}=K^{0.55}$",
       "output-fixed-3.csv":  "Fixed $K^{\\rho}=K^{0.6}$",
       "output-fixed-4.csv":  "Fixed $K^{\\rho}=K^{0.65}$",
       "output-fixed-5.csv":  "Fixed $K^{\\rho}=K^{0.7}$",
       "output-fixed-6.csv":  "Fixed $K^{\\rho}=K^{0.75}$",
    }

    args = Arguments(title='Space Usage: LSVI-UCB-Fixed',
                     legend_title='Space Usage Curves',
                     xlabel='K (Episode #)',
                     ylabel='Space Usage (MB)',
                     idxcolumns=['K'],
                     xcolumn='K',
                     ycolumn='SpaceUsage',
                     output_file='syn-fixed-paramvars-spcusage.pdf',
                     input_list=(['output/syn-paramvars/base/output-syn-base.csv'] + ['output/syn-paramvars/fixed/output-fixed-%d.csv' % (i) for i in range(1, 7)]),
                     legend=legend_syn_fixed
    )
    generate_plot(args, legend_pos='upper right')
# End fn

def plot_syn_adaptive_paramvars_regret():
    legend_syn_adaptive = {
       "output-syn-base.csv":    "LSVI-UCB (Baseline)",
       "output-adaptive-1.csv":  "Adaptive $m=10$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.5}$",
       "output-adaptive-2.csv":  "Adaptive $m=50$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.5}$",
       "output-adaptive-3.csv":  "Adaptive $m=10$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.6}$",
       "output-adaptive-4.csv":  "Adaptive $m=50$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.6}$",
       "output-adaptive-5.csv":  "Adaptive $m=10$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.75}$",
       "output-adaptive-6.csv":  "Adaptive $m=50$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.75}$",
       "output-adaptive-7.csv":  "Adaptive $m=10$, $budget=K^{0.6}$, $K^{\\rho}=K^{0.6}$",
       "output-adaptive-8.csv":  "Adaptive $m=50$, $budget=K^{0.6}$, $K^{\\rho}=K^{0.6}$",
       "output-adaptive-9.csv":  "Adaptive $m=10$, $budget=K^{0.6}$, $K^{\\rho}=K^{0.75}$",
       "output-adaptive-10.csv": "Adaptive $m=50$, $budget=K^{0.6}$, $K^{\\rho}=K^{0.75}$",
       "output-adaptive-11.csv": "Adaptive $m=10$, $budget=K^{0.75}$, $K^{\\rho}=K^{0.75}$",
       "output-adaptive-12.csv": "Adaptive $m=50$, $budget=K^{0.75}$, $K^{\\rho}=K^{0.75}$",
    }

    args = Arguments(title='Regret: LSVI-UCB-Adaptive',
                     legend_title='Regret Curves',
                     xlabel='K (Episode #)',
                     ylabel='Log-scale Regret',
                     idxcolumns=['K'],
                     xcolumn='K',
                     ycolumn='Regret',
                     output_file='syn-adaptive-paramvars-regret.pdf',
                     input_list=(['output/syn-paramvars/base/output-syn-base.csv'] + ['output/syn-paramvars/adaptive/output-adaptive-%d.csv' % i for i in [1,6,7,8,11]]),
                     legend=legend_syn_adaptive
    )
    generate_plot(args)
# End fn

def plot_syn_adaptive_paramvars_proctime():
    legend_syn_adaptive = {
       "output-syn-base.csv":    "LSVI-UCB (Baseline)",
       "output-adaptive-1.csv":  "Adaptive $m=10$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.5}$",
       "output-adaptive-2.csv":  "Adaptive $m=50$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.5}$",
       "output-adaptive-3.csv":  "Adaptive $m=10$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.6}$",
       "output-adaptive-4.csv":  "Adaptive $m=50$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.6}$",
       "output-adaptive-5.csv":  "Adaptive $m=10$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.75}$",
       "output-adaptive-6.csv":  "Adaptive $m=50$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.75}$",
       "output-adaptive-7.csv":  "Adaptive $m=10$, $budget=K^{0.6}$, $K^{\\rho}=K^{0.6}$",
       "output-adaptive-8.csv":  "Adaptive $m=50$, $budget=K^{0.6}$, $K^{\\rho}=K^{0.6}$",
       "output-adaptive-9.csv":  "Adaptive $m=10$, $budget=K^{0.6}$, $K^{\\rho}=K^{0.75}$",
       "output-adaptive-10.csv": "Adaptive $m=50$, $budget=K^{0.6}$, $K^{\\rho}=K^{0.75}$",
       "output-adaptive-11.csv": "Adaptive $m=10$, $budget=K^{0.75}$, $K^{\\rho}=K^{0.75}$",
       "output-adaptive-12.csv": "Adaptive $m=50$, $budget=K^{0.75}$, $K^{\\rho}=K^{0.75}$",
    }

    args = Arguments(title='Proess Time: LSVI-UCB-Adaptive',
                     legend_title='Process Time Curves',
                     xlabel='K (Episode #)',
                     ylabel='Process Time (mins)',
                     idxcolumns=['K'],
                     xcolumn='K',
                     ycolumn='ProcessTime',
                     output_file='syn-adaptive-paramvars-proctime.pdf',
                     input_list=(['output/syn-paramvars/base/output-syn-base.csv'] + ['output/syn-paramvars/adaptive/output-adaptive-%d.csv' % i for i in [1,6,7,8,11]]),
                     legend=legend_syn_adaptive
    )
    generate_plot(args, legend_pos='upper right')
# End fn

def plot_syn_adaptive_paramvars_spcusage():
    legend_syn_adaptive = {
       "output-syn-base.csv":    "LSVI-UCB (Baseline)",
       "output-adaptive-1.csv":  "Adaptive $m=10$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.5}$",
       "output-adaptive-2.csv":  "Adaptive $m=50$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.5}$",
       "output-adaptive-3.csv":  "Adaptive $m=10$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.6}$",
       "output-adaptive-4.csv":  "Adaptive $m=50$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.6}$",
       "output-adaptive-5.csv":  "Adaptive $m=10$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.75}$",
       "output-adaptive-6.csv":  "Adaptive $m=50$, $budget=K^{0.5}$, $K^{\\rho}=K^{0.75}$",
       "output-adaptive-7.csv":  "Adaptive $m=10$, $budget=K^{0.6}$, $K^{\\rho}=K^{0.6}$",
       "output-adaptive-8.csv":  "Adaptive $m=50$, $budget=K^{0.6}$, $K^{\\rho}=K^{0.6}$",
       "output-adaptive-9.csv":  "Adaptive $m=10$, $budget=K^{0.6}$, $K^{\\rho}=K^{0.75}$",
       "output-adaptive-10.csv": "Adaptive $m=50$, $budget=K^{0.6}$, $K^{\\rho}=K^{0.75}$",
       "output-adaptive-11.csv": "Adaptive $m=10$, $budget=K^{0.75}$, $K^{\\rho}=K^{0.75}$",
       "output-adaptive-12.csv": "Adaptive $m=50$, $budget=K^{0.75}$, $K^{\\rho}=K^{0.75}$",
    }

    args = Arguments(title='Space Usage: LSVI-UCB-Adaptive',
                     legend_title='Space Usage Curves',
                     xlabel='K (Episode #)',
                     ylabel='Space Usage (MB)',
                     idxcolumns=['K'],
                     xcolumn='K',
                     ycolumn='SpaceUsage',
                     output_file='syn-adaptive-paramvars-spcusage.pdf',
                     input_list=(['output/syn-paramvars/base/output-syn-base.csv'] + ['output/syn-paramvars/adaptive/output-adaptive-%d.csv' % i for i in [1,6,7,8,11]]),
                     legend=legend_syn_adaptive
    )
    generate_plot(args, legend_pos='upper right')
# End fn

if __name__ == '__main__':
    plot_syn_regrets_long()
    plot_syn_proctime_long()
    plot_syn_spcusage_long()
    plot_syn_fixed_paramvars_regret()
    plot_syn_fixed_paramvars_proctime()
    plot_syn_fixed_paramvars_spcusage()
    plot_syn_adaptive_paramvars_regret()
    plot_syn_adaptive_paramvars_proctime()
    plot_syn_adaptive_paramvars_spcusage()
# End main
