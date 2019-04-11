from time import time
import datetime

from numpy import arange
from pandas import DataFrame as DF
from pandas import concat, read_csv, to_numeric


def df_to_analyze(classification):
    df = read_csv('../data/results/' + classification + '_results.csv')

    df.loc[:, 'cm'] = df.cm.str.replace(r'[\[array\]\(\)\s]', '')

    df[['tn', 'fn', 'fp', 'tp']] = df['cm'].str.split(',', expand=True)

    df.drop(columns='cm', inplace=True)

    conv_to_ints = ['tn', 'fn', 'fp', 'tp']

    for col in conv_to_ints:
        df.loc[:, col] = to_numeric(df[col], errors='ignore')

    df.sort_values(by=['cols', 'df_len'], ascending=False, inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns='index', inplace=True)

    df.loc[:, 'win_perc'] = (df.tp + df.fn) / (df.tp + df.fn + df.tn + df.fp)

    mean_df = df.groupby(by=['cols', 'df_len']
                         ).mean().reset_index().sort_values(
                                 by=['accu'], ascending=False)
    median_df = df.groupby(by=['cols', 'df_len']
                           ).median().reset_index().sort_values(
                                   by=['accu'], ascending=False)
    std_df = df.groupby(by=['cols', 'df_len']
                        ).std().reset_index().sort_values(by=['accu'])
    min_df = df.groupby(by=['cols', 'df_len']
                        ).min().reset_index().sort_values(
                                by=['accu'], ascending=False)
    max_df = df.groupby(by=['cols', 'df_len']
                        ).max().reset_index().sort_values(
                                by=['accu'], ascending=False)

    return max_df, df, mean_df, median_df, std_df, min_df


def get_vals(r_dct, clmn, clmx, clavg, clstd, clmdn, i, tpe):
    r_dct['min_' + tpe] = clmn.loc[i, tpe]
    r_dct['max_' + tpe] = clmx.loc[i, tpe]
    r_dct['mean_' + tpe] = clavg.loc[i, tpe]
    r_dct['std_' + tpe] = clstd.loc[i, tpe]
    r_dct['mdn_' + tpe] = clmdn.loc[i, tpe]
    return r_dct


def gen_best_result_df(classifier, hp, min_mean=.50):

    cl_max, _, cl_mean, cl_median, cl_std, cl_min = df_to_analyze(
            classifier + str(hp))
    results_df = DF()
    tpes = ['accu', 'auc_sc', 'bal_acc', 'f_acc', 'brier_lss', 'log_lss']

    for i in range(len(cl_mean)):
        if cl_mean.loc[i, 'accu'] >= min_mean:
            results_dict = {}
            for tpe in tpes:
                try:
                    results_dict = get_vals(results_dict, cl_min, cl_max, cl_mean,
                                            cl_std, cl_median, i, tpe)
                except KeyError:
                    pass

            results_dict['win_perc'] = round(cl_mean.loc[i, 'win_perc'], 3)
            results_dict['df_len'] = round(cl_mean.loc[i, 'df_len'], 3)
            results_dict['model'] = classifier
            results_dict['hp'] = hp
            results_dict['cols'] = cl_mean.loc[i, 'cols']
            results_df = results_df.append([results_dict], ignore_index=True)
    return results_df


def comb_results_df(classifier, lor=0, ini_hyp=0, fin_hyp=0, incr=0):
    st = time()
    if ini_hyp != 0:
        StP = fin_hyp + incr
        lor = arange(ini_hyp, StP, incr)
    elif lor == 0 and ini_hyp == 0 and fin_hyp == 0:
        results = gen_best_result_df(classifier, 0)
        print(f'{classifier}0 done in {time()-st} Seconds')
        return results

    results = DF()

    for hp in lor:
        try:
            results_ = gen_best_result_df(classifier, hp)
            results = results.append(results_, ignore_index=True)
            print(f'{classifier}{hp} done in {time()-st} Seconds')
        except FileNotFoundError:
            try:
                results_ = gen_best_result_df(classifier, int(hp))
                results = results.append(results_, ignore_index=True)
                print(f'{classifier}{int(hp)} done in {time()-st} Seconds')
            except FileNotFoundError:
                print(f"{classifier}{hp} doesn't exist")
    return results


def get_combined_results_df(ND=''):
    lda_ = comb_results_df('lda', lor=0, ini_hyp=0, fin_hyp=0, incr=0)

    log_bal = comb_results_df('logistic_bal', lor=0, ini_hyp=-2,
                              fin_hyp=3.0, incr=0.25)

    log_unbal = comb_results_df('logistic_unbal', lor=0, ini_hyp=-2,
                                fin_hyp=3.0, incr=0.25)

    qda_list = [0.1, 0.5, 1.0]
    qda_ = comb_results_df('qda', lor=qda_list, ini_hyp=0, fin_hyp=0, incr=0)

    ridge_bal = comb_results_df('ridge_bal', lor=0, ini_hyp=-1,
                                fin_hyp=4.0, incr=0.5)

    ridge_unbal = comb_results_df('ridge_unbal', lor=0, ini_hyp=-1,
                                  fin_hyp=4.0, incr=0.5)

    knn_ = comb_results_df('knn', lor=0, ini_hyp=5, fin_hyp=39, incr=1)

    rf_bal = comb_results_df('random_forest_bal', lor=0, ini_hyp=1,
                             fin_hyp=11, incr=1)

    rf_unbal = comb_results_df('random_forest_unbal', lor=0, ini_hyp=1,
                               fin_hyp=11, incr=1)

    results_df = concat([lda_, log_bal, log_unbal, qda_, rf_bal, rf_unbal,
                         ridge_bal, ridge_unbal, knn_], axis=0, sort=False)

    print(len(results_df))
    dt = datetime.datetime.now().date()
    results_df.to_csv(f'../data/results/botbrnlys-last-all-{ND}-{dt}.csv',
                      index=False)


get_combined_results_df()
