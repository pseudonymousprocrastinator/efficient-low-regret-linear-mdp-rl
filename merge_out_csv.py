import numpy as np
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path

if __name__ == '__main__':
    parser = ArgumentParser(prog="merge_out_csv")
    
    parser.add_argument("-o", "--output-file", default="out.csv", help="The name of the output CSV file.")
    parser.add_argument("--imin", type=int, default=100, help="The smallest value to keep in the index column.")
    parser.add_argument("--imax", type=int, default=5000, help="The largest value to keep (inclusive) in the index column.")
    parser.add_argument("--istep", type=int, default=20, help="The step size to use for the index column.")
    parser.add_argument("input_list", nargs="+", help="The list of input CSV files.")
    
    args = parser.parse_args()
    print('Arguments:', args)
    
    df_list = []
    for ifile in args.input_list:
        df_i = pd.read_csv(ifile, header=0, index_col="K")
        if("I" in df_i.columns):
            df_i = df_i.drop(["I"], axis=1)
        df_list.append(df_i)
    # End for
    
    ixrange_arr = np.arange(args.imin, args.imax+1, args.istep)
    
    df = pd.concat(df_list)
    df['I'] = df.groupby("K").cumcount()
    df = df.set_index([df.index, "I"])
    df.groupby("K", group_keys=True)[["Regret", "ProcessTime", "SpaceUsage"]].apply(lambda x: x)
    #df = df[~df.index.duplicated(keep='last')]
    #df = df.sort_index().reindex(ixrange_arr)
    df = df.sort_index()
    df.to_csv(args.output_file, header=True)
    print('Wrote: ', args.output_file)
# End main
