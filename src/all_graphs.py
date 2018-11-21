import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('seaborn-whitegrid')


def custom_round_int(x, base=20):
    '''Helps to round digits'''

    return int(base * round(float(x) / base))


def custom_round_flt(x, base=.01):
    '''Helps to round digits'''

    return base * round(float(x) / base)


def get_graph_df(file_name):
    '''
    Creates the dataframe from the file
    Converts the time used from seconds to percentage of the time allowed 
    Bins:
     - difference in elo
     - elo
     - opp's elo
     - Number of moves
     - Time used
     - Time used by opp
    creates dummies
    returns:
    X_train, X_test, y_train, y_test'''

    df = pd.read_csv(file_name)

    df.loc[:, 'time_used'] = round((df['time_used'] / df['game_time']) * 100)
    df.loc[:, 'opp_time_used'] = round(
        (df['opp_time_used'] / df['game_time']) * 100)

    df = df[df['time_used'] <= 100].copy()
    df = df[df['opp_time_used'] <= 100].copy()

    bn_op = list(range(1200, 2000, 100))
    bn_op.extend(list(range(0, 601, 100)))
    bn_op.extend(list(range(650, 1101, 10)))
    bn_op = sorted(bn_op)

    bn_di = list(range(-1000, 0, 100))
    bn_di.extend(list(range(100, 1100, 100)))
    bn_di.extend(list(range(-95, 96, 5)))
    bn_di = sorted(bn_di)

    bn_nm = list(range(0, 151, 5))
    bn_el = list(range(600, 1101, 10))
    bn_tu = list(range(0, 101, 5))
    bn_ot = list(range(0, 101, 5))

    df.loc[:, 'bin_elo'] = pd.cut(x=df.elo, bins=bn_el,
                                  labels=bn_el[:-1]).astype(int)

    df.loc[:, 'bin_opp_elo'] = pd.cut(x=df.opp_elo, bins=bn_op,
                                      labels=bn_op[:-1]).astype(float)

    df.loc[:, 'bin_diff'] = pd.cut(x=df['diff'], bins=bn_di,
                                   labels=bn_di[:-1]).astype(float)

    df.loc[:, 'bin_num_moves'] = pd.cut(x=df.num_moves, bins=bn_nm,
                                        labels=bn_nm[:-1]).astype(float)

    df.loc[:, 'bin_time_used'] = pd.cut(x=df.time_used, bins=bn_tu,
                                        labels=bn_tu[:-1]).astype(float)

    df.loc[:, 'bin_opp_time_used'] = pd.cut(x=df.opp_time_used, bins=bn_ot,
                                            labels=bn_ot[:-1]).astype(float)

    return df


def mn_ct_df(df, col_name):

    mndf = df.groupby(col_name).mean()
    mndf.reset_index(inplace=True)
    ctdf = df.groupby(col_name).count()
    ctdf.reset_index(inplace=True)

    return mndf, ctdf


def graph_lim(mndf, ctdf, col_name):

    for num in range(len(mndf)):
        if ctdf.loc[num, 'result'] < 5:
            mndf.drop([num], inplace=True)

    for num in range(len(ctdf)):
        if ctdf.loc[num, 'result'] < 5:
            ctdf.drop([num], inplace=True)

    mndf.sort_values(by='result', inplace=True)
    mndf.reset_index(inplace=True)
    mnm = custom_round_flt(mndf.loc[0, 'result'])
    mxm = custom_round_flt(mndf.loc[mndf.index.max(), 'result'])
    rnm = round(((mxm - mnm) / 10), 2)
    if rnm == 0:
        rnm = 0.005
        mxm += 0.005

    ctdf.sort_values(by='result', inplace=True)
    ctdf.reset_index(inplace=True)
    mnc = custom_round_int(ctdf.loc[0, 'result'], 10)
    mxc = custom_round_int((ctdf.loc[ctdf.index.max(), 'result']), 10)
    mxc = custom_round_int(mxc, 20)
    mnc = custom_round_int(mnc, 20)
    inc = custom_round_int(((mxc - mnc) / 10), 10)

    mnm -= rnm
    mxm += rnm
    mnc -= inc * 2
    mxc += inc * 2

    if mnc < -2:
        mnc = 0

    return mnm, mxm, rnm, mnc, mxc, inc


def x_axis_lims(ctdf, col_name):

    xmn = custom_round_int(ctdf[col_name].min(), 10)
    xmx = custom_round_int(ctdf[col_name].max(), 10)
    xin = custom_round_int((xmx - xmn) / 10, 10)

    if xin == 0:
        xin = custom_round_int((xmx - xmn) / 4, 4)

    xmn -= int(xmn / xin)
    xmx += int(xmx / xin)

    xmn = custom_round_int((xmn - int(xmn / xin)), 10)
    xmx = custom_round_int((xmx - int(xmx / xin)), 10)

    tlgst = ctdf['result'].nlargest(2)
    smyx = tlgst.min()

    return xmn, xmx, xin, smyx


def graph_start_time(df):
    mndf, ctdf = mn_ct_df(df, 'start_time')
    mnm, mxm, rnm, mnc, mxc, inc = graph_lim(mndf, ctdf, 'start_time')

    mndf.plot.scatter(x='start_time', y='result', legend=False)
    plt.title('Result by Time of Day')
    plt.xlabel('Starting Time')
    plt.xlim(-.25, 24)
    plt.xticks(ticks=np.arange(0, 25, step=4))
    plt.ylabel('Winning Ratio')
    plt.ylim(mnm, mxm)
    plt.yticks(ticks=np.arange(mnm, mxm + rnm / 2, step=rnm))

    ctdf.plot.scatter(x='start_time', y='result', legend=False)
    plt.title('# of Games by Time of Day')
    plt.xlabel('Starting Time')
    plt.xlim(-.25, 24)
    plt.xticks(ticks=np.arange(0, 25, step=4))
    plt.ylabel('# of Games')
    plt.ylim(mnc, (mxc - inc * 2))
    plt.yticks(ticks=np.arange(mnc, (mxc - inc * 2) + 1, step=inc))
    plt.show()


def graph_day_of_month(df):
    mndf, ctdf = mn_ct_df(df, 'day')
    mnm, mxm, rnm, mnc, mxc, inc = graph_lim(mndf, ctdf, 'day')

    mndf.plot.scatter(x='day', y='result', legend=False)
    plt.title('Result by Day of the Month')
    plt.xlabel('Day of the Month')
    plt.xlim(0, 32)
    plt.xticks(ticks=np.arange(0, 32, step=5))
    plt.ylabel('Winning Ratio')
    plt.ylim(mnm, mxm)
    plt.yticks(ticks=np.arange(mnm, mxm + rnm / 2, step=rnm))

    ctdf.plot.scatter(x='day', y='result', legend=False)
    plt.title('# of Games by Day of the Month')
    plt.xlabel('Day of the Month')
    plt.xlim(0, 32)
    plt.xticks(ticks=np.arange(0, 32, step=5))
    plt.ylabel('# of Games')
    plt.ylim((mnc + inc), (mxc - inc))
    plt.yticks(ticks=np.arange((mnc + inc), (mxc - inc) + 1, step=inc))
    plt.show()


def day_of_week(df):
    mndf, ctdf = mn_ct_df(df, 'weekday')
    mnm, mxm, rnm, mnc, mxc, inc = graph_lim(mndf, ctdf, 'weekday')

    mndf.plot.scatter(x='weekday', y='result', legend=False)
    plt.title('Result by Day of the Week')
    plt.xlabel('Weekday')
    plt.xticks(ticks=np.arange(7), labels=['Mon', 'Tue', 'Wed', 'Thu',
                                           'Fri', 'Sat', 'Sun'])
    plt.ylabel('Winning Ratio')
    plt.ylim(mnm, mxm)
    plt.yticks(ticks=np.arange(mnm, mxm + rnm / 2, step=rnm))

    ctdf.plot.scatter(x='weekday', y='result', legend=False)
    plt.title('# of Games by Day of the Week')
    plt.xlabel('Weekday')
    plt.xticks(ticks=np.arange(7), labels=['Mon', 'Tue', 'Wed',
                                           'Thu', 'Fri', 'Sat', 'Sun'])
    plt.ylabel('# of Games')
    plt.ylim((mnc + inc), (mxc - inc))
    plt.yticks(ticks=np.arange((mnc + inc), (mxc - inc) + 1, step=inc))
    plt.show()


def castled_or_not(df):
    mndf, ctdf = mn_ct_df(df, 'castled')
    mnm, mxm, rnm, mnc, mxc, inc = graph_lim(mndf, ctdf, 'castled')

    mndf.plot.scatter(x='castled', y='result', legend=False)
    plt.title('Result by Castling')
    plt.xlabel('Castled')
    plt.xlim(-1.02, 1.02)
    plt.xticks(ticks=[-1, 0, 1], labels=['DNC', 'Queen-side', 'King-side'])
    plt.ylabel('Winning Ratio')
    plt.ylim(mnm, mxm)
    plt.yticks(ticks=np.arange(mnm, mxm + rnm / 2, step=rnm))

    ctdf.plot.scatter(x='castled', y='result', legend=False)
    plt.title('# of Games by Castling')
    plt.xlabel('Castled')
    plt.xlim(-1.02, 1.02)
    plt.xticks(ticks=[-1, 0, 1], labels=['DNC', 'Queen-side', 'King-side'])
    plt.ylabel('# of Games')
    plt.ylim((mnc + inc), (mxc - inc * 2))
    plt.yticks(ticks=np.arange((mnc + inc), (mxc - inc), step=inc))

    plt.show()


def opp_castled_or_not(df):
    mndf, ctdf = mn_ct_df(df, 'opp_castled')
    mnm, mxm, rnm, mnc, mxc, inc = graph_lim(mndf, ctdf, 'opp_castled')

    mndf.plot.scatter(x='opp_castled', y='result', legend=False)
    plt.title('Result by Opposition Castling')
    plt.xlabel('Opposition Castled')
    plt.xlim(-1.02, 1.02)
    plt.xticks(ticks=[-1, 0, 1], labels=['DNC', 'Queen-side', 'King-side'])
    plt.ylabel('Winning Ratio')
    plt.ylim(mnm, mxm)
    plt.yticks(ticks=np.arange(mnm, mxm + rnm / 2, step=rnm))

    ctdf.plot.scatter(x='opp_castled', y='result', legend=False)
    plt.title('# of Games for Opposition Castling')
    plt.xlabel('Opposition Castled')
    plt.xlim(-1.02, 1.02)
    plt.xticks(ticks=[-1, 0, 1], labels=['DNC', 'Queen-side', 'King-side'])
    plt.ylabel('# of Games')
    plt.ylim((mnc + inc), (mxc - inc * 2))
    plt.yticks(ticks=np.arange((mnc + inc), (mxc - inc), step=inc))
    plt.show()


def game_time(df):
    mndf, ctdf = mn_ct_df(df, 'game_time')
    mnm, mxm, rnm, mnc, mxc, inc = graph_lim(mndf, ctdf, 'game_time')

    mndf.plot.scatter(x='game_time', y='result', legend=False)
    plt.title('Result by Game Time')
    plt.xlabel('Game Time (Seconds)')
    plt.xlim(170, 610)
    plt.xticks(ticks=[180, 300, 600], labels=[180, 300, 600])
    plt.ylabel('Winning Ratio')
    plt.ylim(mnm, mxm)
    plt.yticks(ticks=np.arange(mnm, mxm + rnm / 2, step=rnm))

    ctdf.plot.scatter(x='game_time', y='result', legend=False)
    plt.title('# of Games for Each Game Time')
    plt.xlabel('Game Time (Seconds)')
    plt.xlim(170, 610)
    plt.xticks(ticks=[180, 300, 600], labels=[180, 300, 600])
    plt.ylabel('# of Games')
    plt.ylim(mnc, (mxc - inc))
    plt.yticks(ticks=np.arange(mnc, (mxc - inc) + 1, step=inc))
    plt.show()


def result_by_color(df):
    mndf, ctdf = mn_ct_df(df, 'color')
    mnm, mxm, rnm, mnc, mxc, inc = graph_lim(mndf, ctdf, 'color')

    mndf.plot.scatter(x='color', y='result', legend=False)
    plt.title('Result by Color')
    plt.xlabel('Color')
    plt.xlim(-.03, 1.01)
    plt.xticks(ticks=[0, 1], labels=['Black', 'White'])
    plt.ylabel('Winning Ratio')
    plt.ylim(mnm, mxm)
    plt.yticks(ticks=np.arange(mnm, mxm + rnm / 2, step=rnm))
    plt.show()


def result_elo(df):
    mndf, ctdf = mn_ct_df(df, 'bin_elo')
    mnm, mxm, rnm, mnc, mxc, inc = graph_lim(mndf, ctdf, 'bin_elo')
    xmn, xmx, xin, smyx = x_axis_lims(ctdf, 'bin_elo')

    mndf.plot.scatter(x='bin_elo', y='result', legend=False)
    plt.title('Result by ELO')
    plt.xlabel('ELO')
    plt.xlim(xmn, xmx)
    plt.xticks(ticks=np.arange(xmn, xmx + xin, step=xin))
    plt.ylabel('Winning Ratio')
    plt.ylim(mnm, mxm)
    plt.yticks(ticks=np.arange(mnm, mxm + rnm / 2, step=rnm))

    ctdf.plot.scatter(x='bin_elo', y='result', legend=False)
    plt.title('# of Games by ELO')
    plt.xlabel('ELO')
    plt.xlim(xmn, xmx)
    plt.xticks(ticks=np.arange(xmn, xmx + xin, step=xin))
    plt.ylabel('# of Games')
    plt.ylim(mnc, (mxc - inc))
    plt.yticks(ticks=np.arange(mnc, (mxc - inc) + 1, step=inc))
    plt.show()


def opp_result_elo(df):
    mndf, ctdf = mn_ct_df(df, 'bin_opp_elo')
    mnm, mxm, rnm, mnc, mxc, inc = graph_lim(mndf, ctdf, 'bin_opp_elo')
    xmn, xmx, xin, smyx = x_axis_lims(ctdf, 'bin_opp_elo')

    mndf.plot.scatter(x='bin_opp_elo', y='result', legend=False)
    plt.title('Result by Opposition ELO')
    plt.xlabel('Opposition ELO')
    plt.xlim(xmn, xmx)
    plt.xticks(ticks=np.arange(xmn, xmx + xin, step=xin))
    plt.ylabel('Winning Ratio')
    plt.ylim(mnm, mxm)
    plt.yticks(ticks=np.arange(mnm, mxm + rnm / 2, step=rnm))

    ctdf.plot.scatter(x='bin_opp_elo', y='result', legend=False)
    plt.title('# of Games by Opposition ELO')
    plt.xlabel('Opposition ELO')
    plt.xlim(xmn, xmx)
    plt.xticks(ticks=np.arange(xmn, xmx + xin, step=xin))
    plt.ylabel('# of Games')
    plt.ylim(mnc, (mxc - inc * 2))
    plt.yticks(ticks=np.arange(mnc, (mxc - inc * 2) + 1, step=inc))
    plt.show()


def result_by_elo_diff(df):
    mndf, ctdf = mn_ct_df(df, 'bin_diff')
    mnm, mxm, rnm, mnc, mxc, inc = graph_lim(mndf, ctdf, 'bin_diff')
    xmn, xmx, xin, smyx = x_axis_lims(ctdf, 'bin_diff')

    mndf.plot.scatter(x='bin_diff', y='result', legend=False)
    plt.title('Result by Difference in ELO')
    plt.xlabel('Difference in ELO')
    plt.xlim(xmn, xmx)
    plt.xticks(ticks=np.arange(xmn, xmx + xin, step=xin))
    plt.ylabel('Winning Ratio')
    plt.ylim(mnm, mxm)
    plt.yticks(ticks=np.arange(mnm, mxm + rnm / 2, step=rnm))

    ctdf.plot.scatter(x='bin_diff', y='result', legend=False)
    plt.title('# of Games by Difference in ELO')
    plt.xlabel('Difference in ELO')
    plt.xlim(xmn, xmx)
    plt.xticks(ticks=np.arange(xmn, xmx + xin, step=xin))
    plt.ylabel('# of Games')
    plt.ylim(mnc, (mxc - inc * 2))
    plt.yticks(ticks=np.arange(mnc, (mxc - inc * 2) + 1, step=inc))
    plt.show()


def won_via(df):
    mndf, ctdf = mn_ct_df(df, 'won_by')
    mnm, mxm, rnm, mnc, mxc, inc = graph_lim(mndf, ctdf, 'won_by')

    mndf.plot.scatter(x='won_by', y='result', legend=False)
    plt.title('Result by the Result Type')
    plt.xlabel('Result Type')
    plt.xticks(ticks=np.arange(9), labels=['Rule', 'Stlmte', 'Abndn', 'Rptn',
                                           'Agr', 'Matrl', 'Time', 'Rsgn',
                                           'Chmte'])
    plt.ylabel('Winning Ratio')
    plt.ylim(mnm, mxm)
    plt.yticks(ticks=np.arange(mnm, mxm + rnm / 2, step=rnm))

    ctdf.plot.scatter(x='won_by', y='result', legend=False)
    plt.title('# of Games by the Result Type')
    plt.xlabel('Result Type')
    plt.xticks(ticks=np.arange(9), labels=['Rule', 'Stlmte', 'Abndn', 'Rptn',
                                           'Agr', 'Matrl', 'Time', 'Rsgn',
                                           'Chmte'])
    plt.ylabel('# of Games')
    plt.ylim(mnc, (mxc - inc))
    plt.yticks(ticks=np.arange(mnc, (mxc - inc) + 1, step=inc))
    plt.show()


def number_of_moves(df):
    mndf, ctdf = mn_ct_df(df, 'bin_num_moves')
    mnm, mxm, rnm, mnc, mxc, inc = graph_lim(mndf, ctdf, 'bin_num_moves')
    xmn, xmx, xin, smyx = x_axis_lims(ctdf, 'bin_num_moves')

    mndf.plot.scatter(x='bin_num_moves', y='result', legend=False)
    plt.title('Result by Number of Moves')
    plt.xlabel('Number of Moves')
    plt.xlim(xmn, xmx + 5)
    plt.xticks(ticks=np.arange(xmn, xmx + xin + 5, step=xin))
    plt.ylabel('Winning Ratio')
    plt.ylim(mnm, mxm)
    plt.yticks(ticks=np.arange(mnm, mxm + rnm / 2, step=rnm))

    ctdf.plot.scatter(x='bin_num_moves', y='result', legend=False)
    plt.title('# of Games by Number of Moves')
    plt.xlabel('Number of Moves')
    plt.xlim(xmn, xmx + 5)
    plt.xticks(ticks=np.arange(xmn, xmx + xin + 5, step=xin))
    plt.ylabel('# of Games')
    plt.ylim(mnc, (mxc - inc))
    plt.yticks(ticks=np.arange(mnc, (mxc - inc) + 1, step=inc))
    plt.show()


def move_num_castled(df):
    mndf, ctdf = mn_ct_df(df, 'castled_on')
    mnm, mxm, rnm, mnc, mxc, inc = graph_lim(mndf, ctdf, 'castled_on')
    xmn, xmx, xin, smyx = x_axis_lims(ctdf, 'castled_on')

    mndf.plot.scatter(x='castled_on', y='result', legend=False)
    plt.title('Result by the Move # of Castling (0 for not castling)')
    plt.xlabel('Move # of Castling')
    plt.xlim(xmn, xmx + xin)
    plt.xticks(ticks=np.arange(xmn, xmx + xin + 1, step=xin))
    plt.ylabel('Winning Ratio')
    plt.ylim(mnm, mxm)
    plt.yticks(ticks=np.arange(mnm, mxm + rnm / 2, step=rnm))

    ctdf.plot.scatter(x='castled_on', y='result', legend=False)
    plt.title('# of Games by the Move # of Castling (0 for not castling)')
    plt.xlabel('Move # of Castling')
    plt.xlim(xmn, xmx + xin)
    plt.xticks(ticks=np.arange(xmn, xmx + xin + 1, step=xin))
    plt.ylabel('# of Games')
    inc = custom_round_int((smyx - mnc) / 10)
    plt.ylim(mnc, smyx + inc)
    plt.yticks(ticks=np.arange(mnc, (smyx + inc * 2), step=inc))
    plt.show()


def opp_move_num_castled(df):
    mndf, ctdf = mn_ct_df(df, 'opp_castled_on')
    mnm, mxm, rnm, mnc, mxc, inc = graph_lim(mndf, ctdf, 'opp_castled_on')
    xmn, xmx, xin, smyx = x_axis_lims(ctdf, 'opp_castled_on')

    mndf.plot.scatter(x='opp_castled_on', y='result', legend=False)
    plt.title('Result by the Move # of Opposition Castling')
    plt.xlabel('Move # of Opp Castling')
    plt.xlim(xmn, xmx + (xin * 2))
    plt.xticks(ticks=np.arange(xmn, (xmx + (xin * 2)) + 1, step=xin))
    plt.ylabel('Winning Ratio')
    plt.ylim(mnm, mxm)
    plt.yticks(ticks=np.arange(mnm, mxm + rnm / 2, step=rnm))

    ctdf.plot.scatter(x='opp_castled_on', y='result', legend=False)
    plt.title('# of Games by the Move # of Opposition Castling')
    plt.xlabel('Move # of Opp Castling')
    plt.xlim(xmn, xmx + (xin * 2))
    plt.xticks(ticks=np.arange(xmn, (xmx + (xin * 2)) + 1, step=xin))
    plt.ylabel('# of Games')
    inc = custom_round_int((smyx - mnc) / 10)
    plt.ylim(mnc, smyx)
    plt.yticks(ticks=np.arange(mnc, (smyx + inc), step=inc))
    plt.show()


def result_by_time_used(df):
    mndf, ctdf = mn_ct_df(df, 'time_used')
    mnm, mxm, rnm, mnc, mxc, inc = graph_lim(mndf, ctdf, 'time_used')
    xmn, xmx, xin, smyx = x_axis_lims(ctdf, 'time_used')

    mndf.plot.scatter(x='time_used', y='result', legend=False)
    plt.title('Result by the Amount of Time Used (in %)')
    plt.xlabel('Amount of Time Used(%)')
    plt.xlim(xmn, xmx + 5)
    plt.xticks(ticks=np.arange(xmn, xmx + 6, step=xin))
    plt.ylabel('Winning Ratio')
    plt.ylim(mnm, mxm)
    plt.yticks(ticks=np.arange(mnm, mxm + rnm / 2, step=rnm))

    ctdf.plot.scatter(x='time_used', y='result', legend=False)
    plt.title('# of Games by the Amount of Time Used (in %)')
    plt.xlabel('Amount of Time Used(%)')
    plt.xlim(xmn, xmx + 5)
    plt.xticks(ticks=np.arange(xmn, xmx + 6, step=xin))
    plt.ylabel('# of Games')
    inc = custom_round_int(((smyx - mnc) / 5), 10)
    plt.ylim(mnc, smyx)
    plt.yticks(ticks=np.arange(mnc, (smyx + inc), step=inc))
    plt.show()


def result_by_opp_time_used(df):
    mndf, ctdf = mn_ct_df(df, 'opp_time_used')
    mnm, mxm, rnm, mnc, mxc, inc = graph_lim(mndf, ctdf, 'opp_time_used')
    xmn, xmx, xin, smyx = x_axis_lims(ctdf, 'opp_time_used')

    mndf.plot.scatter(x='opp_time_used', y='result', legend=False)
    plt.title('Result by the Amount of Time Used by Opposition(in %)')
    plt.xlabel('Amount of Time Used by Opposition(%)')
    plt.xlim(xmn, xmx + 5)
    plt.xticks(ticks=np.arange(xmn, xmx + 6, step=xin))
    plt.ylabel('Winning Ratio')
    plt.ylim(mnm, mxm)
    plt.yticks(ticks=np.arange(mnm, mxm + rnm / 2, step=rnm))

    ctdf.plot.scatter(x='opp_time_used', y='result', legend=False)
    plt.title('# of Games by the Amount of Time Used by Opposition(in %)')
    plt.xlabel('Amount of Time Used by Opposition(%)')
    plt.xlim(xmn, xmx + 5)
    plt.xticks(ticks=np.arange(xmn, xmx + 6, step=xin))
    plt.ylabel('# of Games')
    inc = custom_round_int(((smyx - mnc) / 5), 10)
    plt.ylim(mnc, smyx)
    plt.yticks(ticks=np.arange(mnc, (smyx + inc), step=inc))
    plt.show()
