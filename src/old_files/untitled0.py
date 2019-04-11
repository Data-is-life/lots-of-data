#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 23:33:58 2019

@author: ocelot
"""
import warnings
import gc
gc.enable()
warnings.filterwarnings('ignore')
#import pandas as pd
import numpy as np
#from sklearn.metrics import mean_squared_error as MSE
#from sklearn.metrics import r2_score
from big_data_clean import final_clean_run
from run_model_and_plot import mega_model_run
#from run_model_and_plot import plot_all
from run_model_and_plot import X_y, get_df
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


ndf, pl_df, ux_df, xl_df, sl_df, bl_df, bs_df = final_clean_run()

pdf = get_df(pl_df, 'pool')
X, y, X_train, X_test, y_train, y_test = X_y(pdf, 200, 800, 'pool')

model = RandomForestRegressor(random_state=5, verbose=1, n_jobs=8,
                              max_features='sqrt', min_samples_leaf=1,
                              min_samples_split=2, n_estimators=200)

#cv_model = GridSearchCV(model, params, cv=3, pre_dispatch=10)
#cv_model.fit(X, y)
#cv_model.best_params_

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

abs(y_pred-y_test).sum()/600

#for tri, tsi in cv.split(X):
#    model.fit(X[tri], y[tri])
#    print(model.score(X[tri], y[tri]))
#    print(model.score(X[tsi], y[tsi]))
    

def get_end_start_row(lnd):
    hp = int(lnd*0.2)
    lp = int(lnd*0.1)
    srl = np.random.randint(0, int(lnd-hp), 10)
    erl = srl + np.random.randint(lp, hp, 10)
    return srl, erl


def get_results(srl, erl, df, rd_tp, mm):
    for i, sp in enumerate(srl):
        print(f'Testing rows: {sp}:{erl[i]-1}')
        print(f'Training rows: 0:{sp-1}, {erl[i]}:{len(pl_df)}')
        mega_model_run(df, rd_tp, sp, erl[i], 'rf', mm)


def run_model_for_all(pl_df, ux_df, xl_df, sl_df, bl_df, bs_df):
    lnp, lnx, lnxl, lns, lnb, lnv = len(pl_df), len(ux_df), len(xl_df), len(sl_df), len(bl_df), len(bs_df)
    psrl, perl = get_end_start_row(lnp)
    xsrl, xerl = get_end_start_row(lnx)
    xlsrl, xlerl = get_end_start_row(lnxl)
    ssrl, serl = get_end_start_row(lns)
    bsrl, berl = get_end_start_row(lnb)
    vsrl, verl = get_end_start_row(lnv)
    print('\n------------Running Pool------------\n')
    get_results(psrl, perl, pl_df, 'pool', 'mid')
    print('\n------------Running UberX------------\n')
    get_results(xsrl, xerl, ux_df, 'uberx', 'mid')
    print('\n------------Running XL------------\n')
    get_results(xlsrl, xlerl, xl_df, 'uber_xl', 'mid')
    print('\n------------Running Select------------\n')
    get_results(ssrl, serl, sl_df, 'select', 'mid')
    print('\n------------Running Black------------\n')
    get_results(bsrl, berl, bl_df, 'black', 'mid')
    print('\n------------Running Black SUV------------\n')
    get_results(vsrl, verl, bs_df, 'black_suv', 'mid')


run_model_for_all(pl_df, ux_df, xl_df, sl_df, bl_df, bs_df)