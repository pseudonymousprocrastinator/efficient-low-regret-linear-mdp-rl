import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--legend", default="legend.txt", help="The legend file (with lines of the form file-name:curve-label).")
parser.add_argument("-t", "--title", default="Plot title", help="The title of the plot.")
parser.add_argument("-lt", "--legend-title", default="Plots", help="The caption of the plot legend.")
parser.add_argument("-xl", "--xlabel", default=None, help="The label of the X axis.")
parser.add_argument("-yl", "--ylabel", default=None, help="The label of the Y axis.")
parser.add_argument("-xc", "--xcolumn", default="X", help="The name of the column (in the input CSV files) with X-axis data.")
parser.add_argument("-yc", "--ycolumn", default="Y", help="The name of the column (in the input CSV files) with Y-axis data.")
parser.add_argument("-xr", "--xrange", default="*",  help="The range of X-values to include in the plot (if given, should be of the form start:end:step).")
parser.add_argument("-o", "--output-file", default="plot.pdf", help="The output file name (pdf format).")
parser.add_argument("input_list", nargs="+", help="The list of input CSV files.")

args = parser.parse_args()

if args.xlabel is not None:
	xlabel = args.xlabel
else:
	xlabel = args.xcolumn
if args.ylabel is not None:
	ylabel = args.ylabel
else:
	ylabel = args.ycolumn

if args.xrange == '*':
    xslice = slice(None)
else:
    range_vals = args.xrange.split(':')
    xslice = slice(int(range_vals[0]), int(range_vals[1]), int(range_vals[2]))

plot_df_dict = {}
legend = pd.read_csv(args.legend, sep=':', names=['file_name','label'], index_col='file_name')
plot_df_dict[xlabel] = pd.read_csv(args.input_list[0], header=0, index_col=args.xcolumn).loc[xslice,:].index.values
for input_file_name in args.input_list:
    df = pd.read_csv(input_file_name, header=0, index_col=args.xcolumn)
    plot_df_dict[legend.loc[input_file_name]['label']] = df.loc[xslice,:][args.ycolumn].values
# End for
plot_df = pd.DataFrame(plot_df_dict)

sns.set(rc={"figure.figsize":(12, 9)})
sns.set_style("whitegrid", {'axes.grid' : False})
ax = sns.lineplot(x=xlabel, y=ylabel, hue=args.legend_title, style=args.legend_title, markers=True, 
             data=pd.melt(plot_df, [xlabel], var_name=args.legend_title, value_name=ylabel))
ax.set_title(args.title, {"fontsize":18})
plt.setp(ax.get_legend().get_texts(), fontsize='14') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='16') # for legend title
plt.tight_layout()
plt.savefig(args.output_file, format='pdf')
print('Wrote:', args.output_file)
