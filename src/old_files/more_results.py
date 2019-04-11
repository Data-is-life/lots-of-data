from time import time
import pandas as pd
from numpy import arange

results_df = pd.read_csv('../data/botbrnlys-rand.csv')


def extract_best_vals_index(results_df, df, classifier, hp):
    final_df = pd.DataFrame()
    temp_df = results_df[results_df.model == classifier]
    temp_df_f = temp_df[temp_df.hp.round(3) == hp]
    if len(temp_df_f) < 1:
        return final_df
    for i in temp_df_f.index:
        new_df = df[df.cols == results_df.cols[i]]
        final_df = final_df.append(
                new_df[new_df.df_len == results_df.df_len[i]])
    final_df.reset_index(inplace=True)
    final_df.drop(columns='index', inplace=True)
    print(len(final_df))
    return final_df


def ein_best_vals(results_df, classifier, lor=0, ini_hyp=0, fin_hyp=0, incr=0):
    final_df = pd.DataFrame()
    st = time()
    if ini_hyp != 0:
        StP = fin_hyp + incr
        lor = arange(ini_hyp, StP, incr)
    elif lor == 0 and ini_hyp == 0 and fin_hyp == 0:
        cl_df = pd.read_csv('../data/' + classifier + '0_results-nb.csv')
        n_df = extract_best_vals_index(results_df, cl_df, classifier, 0)
        print(f'{classifier}0 done in {time()-st} Seconds')
        return n_df
    for hp in lor:
        try:
            cl_df = pd.read_csv(
                    '../data/' + classifier + str(hp) + '_results-nb.csv')
            nw_df = extract_best_vals_index(results_df, cl_df, classifier, hp)
            final_df = final_df.append(nw_df)
            final_df.reset_index(inplace=True)
            final_df.drop(columns='index', inplace=True)
            print(f'{classifier}{hp} done in {time()-st} Seconds')
        except FileNotFoundError:
            try:
                hp_n = str(int(hp))
                cl_df = pd.read_csv(
                        '../data/' + classifier + hp_n + '_results-nb.csv')
                nw_df = extract_best_vals_index(results_df, cl_df, classifier,
                                                hp)
                final_df = final_df.append(nw_df)
                final_df.reset_index(inplace=True)
                final_df.drop(columns='index', inplace=True)
                print(f'{classifier}{int(hp)} done in {time()-st} Seconds')
            except FileNotFoundError:
                print(f"{classifier}{hp} doesn't exist")
    return final_df


def combine_best_val_df(results_df):
    lda_rslt = ein_best_vals(results_df, 'lda', lor=0, ini_hyp=0,
                             fin_hyp=0, incr=0)

    log_rslt = ein_best_vals(results_df, 'logistic', lor=0, ini_hyp=-2,
                             fin_hyp=3.0, incr=0.25)

    qda_list = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    qda_rslt = ein_best_vals(results_df, 'qda', lor=qda_list, ini_hyp=0,
                             fin_hyp=0, incr=0)

    rdg_rslt = ein_best_vals(results_df, 'ridge', lor=0, ini_hyp=-1,
                             fin_hyp=4.0, incr=0.5)

    gauz_rslt = ein_best_vals(results_df, 'gaussian', lor=0, ini_hyp=1,
                              fin_hyp=39, incr=1)

    knn_rslt = ein_best_vals(results_df, 'knn', lor=0, ini_hyp=2,
                             fin_hyp=39, incr=1)

    svc_rslt = ein_best_vals(results_df, 'svc', lor=0, ini_hyp=-2,
                             fin_hyp=2, incr=0.5)

    rf_rslt = ein_best_vals(results_df, 'random_forest', lor=0, ini_hyp=1,
                            fin_hyp=14, incr=1)

    abc_list = [0.25, 0.251, 0.252, 0.253, 0.85, 0.851, 0.852, 0.853,
                1.45, 1.451, 1.452, 1.453, 2.05, 2.051, 2.052, 2.053,
                2.65, 2.651, 2.652, 2.653, 3.25, 3.251, 3.252, 3.253,
                3.85, 3.851, 3.852, 3.853, 4.45, 4.451, 4.452, 4.453]
    abc_rslt = ein_best_vals(results_df, 'ada_boost', lor=abc_list, ini_hyp=0,
                             fin_hyp=0, incr=0)

    xgb_list = [10.5, 11.0, 11.5, 12.0, 13.0, 20.5, 21.0, 21.5, 22.0, 23.0,
                30.5, 31.0, 31.5, 32.0, 33.0, 40.5, 41.0, 41.5, 42.0, 43.0,
                50.5, 51.0, 51.5, 52.0, 53.0]
    xgb_rslt = ein_best_vals(results_df, 'xgboost', lor=xgb_list, ini_hyp=0,
                             fin_hyp=0, incr=0)

    results_df = pd.concat([lda_rslt, log_rslt, qda_rslt, rdg_rslt, gauz_rslt,
                            knn_rslt, svc_rslt, abc_rslt, rf_rslt, xgb_rslt
                            ], axis=0)

    print(len(results_df))
    results_df.to_csv('../data/best_vals-rand.csv', index=False)


combine_best_val_df(results_df)
