from pandas import read_csv, to_numeric

df = read_csv('../data/best_vals.csv')

df.loc[:, 'cm'] = df.cm.str.replace(r'[\[array\]\(\)\s]', '')

df[['tn', 'fn', 'fp', 'tp']] = df['cm'].str.split(',', expand=True)

df.drop(columns='cm', inplace=True)

conv_to_ints = ['tn', 'fn', 'fp', 'tp']

for col in conv_to_ints:
    df.loc[:, col] = to_numeric(df[col], errors='ignore')

df.sort_values(by=['cols', 'df_len'], ascending=False, inplace=True)
df.reset_index(inplace=True)
df.drop(columns='index', inplace=True)

df.loc[:, 'win_perc'] = round((df.tp + df.fn) * 100 / (
    df.tp + df.fn + df.tn + df.fp), 3)

gdf = df.groupby(
    by=['start_row', 'cols', 'df_len', 'model']).mean().reset_index().sort_values(
        by=['accu'], ascending=False)
gdf.to_csv('../data/gdf.csv', index=False)

