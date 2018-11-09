# Author: Mohit Gangwani
# Date: 11/08/2018
# Git-Hub: Data-is-Life

import pandas as pd
import numpy as np
import re
from ast import literal_eval
from datetime import datetime
from pandas import *

'''All functions I used to clean up game log from Chess.com'''

def custom_round(x, base=20):
	'''Helps to round digits'''

    return int(base * round(float(x)/base))


def initial_chess_data(filename):
	'''First function:

	Input = Game log from Chess.com

	Cleans the file for any unnessacery lines from the game log file

	Output = All game information as a text'''

    with open(filename, 'r+') as file:
        icd_l = file.readlines()

    icd_t = " ".join(num for num in icd_l if len(num) > 4)

    return icd_t


def chess_data_cleanup(chess_text):
	'''Second function:
	Input = All game information as a text

	Creates a df where one row is for game information, the following row 
	is moves in that game.

	Output = df with game information and moves'''

    chess_text = chess_text.replace('[', "").replace(']', "")
    chess_text = chess_text.replace('\n', ' ')
    chess_text = chess_text.replace('   ', ' ').replace('  ', ' ')
    chess_text = chess_text.replace('... ', 'b":"').replace('. ', 'w":"')
    chess_text = chess_text.replace('", ', '", "').replace(' {%clk ', '^')
    chess_text = chess_text.replace(' {%clk', '^')
    chess_text = chess_text.replace('}', '",').replace('", ', '", "')
    chess_text = chess_text.replace(' Site "Chess.com" D', ', D')
    chess_text = chess_text.replace('Event ', '}~{"Event":')
    chess_text = chess_text.replace('", Date ', '", "Date": ')
    chess_text = chess_text.replace('" Result ', '", "Result": ')
    chess_text = chess_text.replace('" Round ', '", "Round": ')
    chess_text = chess_text.replace('" White ', '", "White": ')
    chess_text = chess_text.replace('" Black ', '", "Black": ')
    chess_text = chess_text.replace('" WhiteElo ', '", "WhiteElo": ')
    chess_text = chess_text.replace('" TimeControl ', '", "TimeControl": ')
    chess_text = chess_text.replace('" EndTime ', '", "EndTime": ')
    chess_text = chess_text.replace('" BlackElo ', '", "BlackElo": ').replace(
        '" Termination ', '", "Termination": ')
    chess_text = chess_text.replace(
        '"Event":"10|0 Blitz", "Date": "2017.02.16", "Round": "5", "White": "TrueMoeG", "Black": "naggvk", "Result": "0-1", "WhiteElo": "784", "BlackElo": "1210", "TimeControl": "600", "EndTime": "12:27:07 PST", "Termination": "naggvk won - game abandoned" }~{', '')
    chess_text = chess_text.replace(
        '"Event":"Live Chess", "Date": "2017.02.20", "Round": "-", "White": "30mate", "Black": "TrueMoeG", "Result": "0-1", "WhiteElo": "820", "BlackElo": "878", "TimeControl": "600", "EndTime": "11:55:45 PST", "Termination": "TrueMoeG won - game abandoned" }~{', '')
    chess_text = chess_text.replace(
        '"Event":"Live Chess", "Date": "2018.07.14", "Round": "-", "White": "Bran17", "Black": "TrueMoeG", "Result": "0-1", "WhiteElo": "1205", "BlackElo": "961", "TimeControl": "300", "EndTime": "18:02:56 PDT", "Termination": "TrueMoeG won - game abandoned" }~{', '')
    chess_text = chess_text.replace(
        '"Event":"Live Chess", "Date": "2017.04.26", "Round": "-", "White": "nerbenator", "Black": "TrueMoeG", "Result": "0-1", "WhiteElo": "829", "BlackElo": "842", "TimeControl": "600", "EndTime": "16:49:05 PDT", "Termination": "TrueMoeG won - game abandoned" }~{', '')
    chess_text = chess_text.replace(
        '"Event":"10|0 Blitz", "Date": "2017.02.16", "Round": "4", "White": "TrueMoeG", "Black": "Shuzakhan", "Result": "0-1", "WhiteElo": "845", "BlackElo": "1183", "TimeControl": "600", "EndTime": "11:58:21 PST", "Termination": "Shuzakhan won by resignation" }~{', '')
    chess_text = chess_text.replace(' PST', '').replace(' PDT', '')
    chess_text = chess_text.replace('   ', ' ').replace('  ', ' ')
    chess_text = chess_text.replace('" 1w":[', '"}~{"1w":[')
    chess_text = chess_text.replace('" 1w":"', '"}~{"1w":"')
    chess_text = chess_text.replace(', "1/2-1/2 }~{', '}~{')
    chess_text = chess_text.replace(', "1-0 }~{', '}~{')
    chess_text = chess_text.replace(', "0-1 }~{', '}~{')
    chess_text = chess_text.replace(', "1-0 ', '}').replace(', "}', '}')

    cl = ''.join([num for num in chess_text]).split("~") # I used '~' as a separator

    df = DataFrame(cl, columns=['a'])

    df = df[df['a'].str.len() > 3]

    return df


def data_cleaning_1(df):
	'''Third function:
	Input:
	df = Moves df
	Creates a new df that has time information.

	Output:
	t_df = Moves time df - all the times for moves
	m_df = Moves df - Fixed column names
	Takes the df from the chess_data_cleanup function and creates two dfs. 
	First df is all games information
	Second df is for all the moves in those games
	c_df = main df with all the rows
	m_df = moves df with all the moves
	d_df = information df with all the game information'''

    c_df = DataFrame(data=list(df['a'].apply(literal_eval)))

    c_df['Date'] = c_df['Date'].fillna(method='ffill')
    c_df['EndTime'] = c_df['EndTime'].fillna(method='ffill')

    '''Convert all the dates and time to dates and times'''
    c_df['date_time'] = to_datetime(c_df['Date'] + ' ' + c_df['EndTime'])
    c_df['Date'] = to_datetime(c_df['Date'])
    c_df['EndTime'] = to_timedelta(c_df['EndTime'])

    '''split moves to a new df drop columns not needed'''
    m_df = c_df[c_df.index % 2 == 1]
    m_df = m_df.sort_values('date_time').reset_index().drop(
        columns=['index', 'Date', 'White', 'Black', 'Result', 'WhiteElo',
                 'BlackElo', 'TimeControl', 'EndTime', 'Termination', 'date_time',
                 'Round', 'Event'])

    '''split game information to a new df'''
    d_df = c_df[c_df.index % 2 == 0]

    d_df = d_df[['Date', 'White', 'Black', 'Result', 'WhiteElo', 'BlackElo',
                 'TimeControl', 'EndTime', 'Termination', 'date_time'
                 ]].sort_values('date_time').reset_index().drop(columns=[
                     'index', 'date_time'])

    '''Rename all columns to lower case and insert "_" to split words'''
    d_df = d_df.rename(columns={
        'Date': 'date', 'White': 'white', 'Black': 'black',
        'Result': 'result', 'WhiteElo': 'white_elo', 'BlackElo': 'black_elo',
        'TimeControl': 'game_time', 'EndTime': 'end_time', 'Termination': 'termination'
        })

    d_df['num_moves'] = m_df.count(axis=1)
    d_df['white_elo'] = to_numeric(d_df['white_elo'])
    d_df['black_elo'] = to_numeric(d_df['black_elo'])
    d_df['color'] = np.where(d_df['white'] == 'TrueMoeG', 1, 0)

    '''drop duplicate rows'''
    d_df.drop_duplicates(inplace=True)
    m_df.drop_duplicates(inplace=True)

    return m_df, d_df


def data_cleaning_2(m_df):
	'''Fourth function:
	Input:
	m_df = Moves df
	Creates a new df that has time information.

	Output:
	t_df = Moves time df - all the times for moves
	m_df = Moves df - Fixed column names'''

    moves_column_names = ['00' + num if len(num) == 2 else
                          num for num in m_df.columns]
    moves_column_names = ['0' + num if len(num) == 3 else
                          num for num in moves_column_names]
    moves_column_names = [num.replace('w', 'a') for num
                          in moves_column_names]

    m_df.columns = [num for num in moves_column_names]

    moves_column_names = sorted(moves_column_names)
    m_df = m_df[[num for num in moves_column_names]]

    t_df = m_df.copy()

    for col_name in m_df.columns:
        m_df[col_name] = m_df[col_name].str.extract(r'(^\w+-?\w+?-?\w?\+?)')
    for col_name in m_df.columns:
        t_df[col_name] = t_df[col_name].str.extract(r'(\d\:\d+:\d+\.?\d)')

    return m_df, t_df


def data_cleaning_3(t_df, d_df):
	'''Fifth function:
	Input:
	t_df = Move times df
	d_df = Game information df
	Cleans the times df to fix the games that give extra time after each move'''

    t_df = t_df.apply(to_timedelta, errors='coerce')
    t_df = t_df.apply(to_numeric, errors='coerce')
    t_df = t_df.div(1000000000)

    t_df['game_time'] = d_df['game_time']
    t_df['extra_time'] = t_df['game_time'].replace([
        '300', '600', '180', '180+2', '300+5'], ['0', '0', '0', '2', '5'])
    t_df['game_time'] = d_df['game_time'].replace(
        ['180+2', '300+5'], [180, 300])

    t_df['extra_time'] = to_numeric(t_df['extra_time'])

    return t_df


def data_cleaning_4(m_df, t_df, d_df):
	'''Sixth function:
	Input:
	m_df = Moves df
	t_df = Move times df
	d_df = Game information df
	Cleans the times df (t_df) to fix the games that give extra time after each
	move'''

    t_df['game_time'] = to_numeric(t_df['game_time'])
    d_df['game_time'] = t_df['game_time']

    wh_m_df = m_df[m_df.columns[::2]].copy()
    bl_m_df = m_df[m_df.columns[1::2]].copy()
    wh_t_df = t_df[t_df.columns[::2]].copy()
    bl_t_df = t_df[t_df.columns[1::2]].copy()

    wh_t_df = wh_t_df.drop(columns=[wh_t_df.columns[-1]])
    bl_t_df = bl_t_df.drop(columns=[bl_t_df.columns[-1]])

    d_df['white_num_moves'] = wh_m_df.count(axis=1)
    d_df['black_num_moves'] = bl_m_df.count(axis=1)

    for num in wh_t_df.columns:
        wh_t_df[num] = t_df['game_time'] - wh_t_df[num]
    for num in bl_t_df.columns:
        bl_t_df[num] = t_df['game_time'] - bl_t_df[num]

    two_list = []
    five_list = []
    i = 0
    while i < len(t_df):
        for num in t_df['extra_time']:
            if num == 2:
                two_list.append(i)
                i += 1
            elif num == 5:
                five_list.append(i)
                i += 1
            else:
                i += 1

    for num in two_list:
        i = 0
        j = 0

        while i < (len(wh_t_df.columns)):
            wh_t_df.iloc[num, i] = wh_t_df.iloc[num, i] + ((i+1) * 2)
            i += 1

        while j < (len(bl_t_df.columns)):
            bl_t_df.iloc[num, j] = wh_t_df.iloc[num, j] + ((j+1) * 2)
            j += 1

    for num in five_list:
        i = 0
        j = 0

        while i < (len(wh_t_df.columns)):
            wh_t_df.iloc[num, i] = wh_t_df.iloc[num, i] + ((i+1) * 5)
            i += 1
        while j < (len(bl_t_df.columns)):
            bl_t_df.iloc[num, j] = wh_t_df.iloc[num, j] + ((j+1) * 5)
            j += 1

    for num in wh_t_df.columns:
        wh_t_df[num] = np.where(wh_t_df[num] > 5000, 0, wh_t_df[num])
    for num in bl_t_df.columns:
        bl_t_df[num] = np.where(bl_t_df[num] > 5000, 0, bl_t_df[num])

    return wh_m_df, wh_t_df, bl_m_df, bl_t_df, d_df, t_df


def data_cleaning_5(c_t_df, t_df, d_df, col):
    tm_df = c_t_df.shift(periods=1, axis=1)
    tm_df = tm_df - c_t_df
    tm_df = -tm_df

    tm_df[col] = c_t_df[col]

    for num in tm_df.columns:
        tm_df[num] = np.where(tm_df[num] <= 0, 0, tm_df[num])

    return tm_df


def help_func1(m_df):
    cast_list = []
    cast_w_list = []
    i = 0
    while i < (len(m_df)):
        a = list(m_df.iloc[i])
        if "O-O" in a:
            cast_list.append(a.index("O-O")+1)
            cast_w_list.append(1)
            i += 1
        elif "O-O-O" in a:
            cast_list.append(a.index("O-O-O")+1)
            cast_w_list.append(0)
            i += 1
        else:
            cast_list.append(0)
            cast_w_list.append(-1)
            i += 1
    return cast_list, cast_w_list


def data_cleaning_6(d_df, m_df, bl_m_df, wh_m_df, wh_t_df, bl_t_df):

    d_df['white_time_used'] = wh_t_df.max(axis=1).apply(
        lambda x: custom_round(x, base=10))
    d_df['black_time_used'] = bl_t_df.max(axis=1).apply(
        lambda x: custom_round(x, base=10))

    d_df['winner'] = d_df['termination'].str.extract(
        '(^[a-zA-Z0-9]+)', expand=False)
    d_df['won_by'] = d_df['termination'].str.extract(
        '([a-zA-Z0-9]+$)', expand=False)

    cstl_l_bl, cstl_loc_l_bl = help_func1(bl_m_df)
    cstl_l_wh, cstl_loc_l_wh = help_func1(wh_m_df)

    d_df['weekday'] = d_df.date.apply(lambda x: x.dayofweek)
    d_df['day'] = d_df.date.apply(lambda x: x.day)

    d_df['result'] = np.where(d_df['winner'] == 'TrueMoeG',
                              1.0, (np.where(d_df['winner'] == 'Game', 0.5, 0.0)))

    d_df['white_castled_on'] = pd.Series(cstl_l_wh)
    d_df['black_castled_on'] = pd.Series(cstl_l_bl)
    d_df['white_castled_where'] = pd.Series(cstl_loc_l_wh)
    d_df['black_castled_where'] = pd.Series(cstl_loc_l_bl)

    d_df['castled_on'] = np.where(d_df['color'] == 1, d_df[
        'white_castled_on'], d_df['black_castled_on'])
    d_df['opp_castled_on'] = np.where(d_df['color'] == 0, d_df[
        'white_castled_on'], d_df['black_castled_on'])
    d_df['castled'] = np.where(d_df['color'] == 1, d_df[
        'white_castled_where'], d_df['black_castled_where'])
    d_df['opp_castled'] = np.where(d_df['color'] == 0, d_df[
        'white_castled_where'], d_df['black_castled_where'])

    d_df['time_used'] = np.where(d_df['color'] == 1, d_df[
        'white_time_used'], d_df['black_time_used'])
    d_df['opp_time_used'] = np.where(d_df['color'] == 0, d_df[
        'white_time_used'], d_df['black_time_used'])

    d_df['time_used'] = np.where(d_df['result'] == 1.0, d_df[
        'time_used'], np.where(d_df['won_by'] == 'time', d_df[
            'game_time'], d_df['time_used']))
    d_df['opp_time_used'] = np.where(d_df['result'] == 0.0, d_df[
        'opp_time_used'], np.where(d_df['won_by'] == 'time', d_df[
            'game_time'], d_df['opp_time_used']))
    d_df['end_time'] = to_numeric(d_df['end_time'])/3600000000
    d_df['start_time'] = d_df['end_time'] - \
        (d_df['time_used']+d_df['opp_time_used'])/3.6

    d_df['num_moves'] = np.where(d_df['color'] == 1, d_df[
        'white_num_moves'], d_df['black_num_moves'])
    d_df['opp_num_moves'] = np.where(d_df['color'] == 0, d_df[
        'white_num_moves'], d_df['black_num_moves'])

    d_df['avg_time'] = d_df['time_used']/d_df['num_moves']
    d_df['opp_avg_time'] = d_df['opp_time_used']/d_df['opp_num_moves']

    d_df['start_time'] = d_df['start_time']//1000
    d_df['end_time'] = d_df['end_time']//1000

    return d_df.drop(columns=[
        'white', 'black', 'termination', 'white_num_moves', 'black_num_moves',
        'white_time_used', 'black_time_used', 'winner', 'white_castled_on',
        'black_castled_on', 'white_castled_where', 'black_castled_where'])


def data_cleaning_7(d_df, wh_tm_df, bl_tm_df):
    d_df['white_max_move'] = wh_tm_df.max(axis=1)
    d_df['black_max_move'] = bl_tm_df.max(axis=1)

    d_df['max_move'] = np.where(d_df['color'] == 1, d_df[
        'white_max_move'], d_df['black_max_move'])
    d_df['opp_max_move'] = np.where(d_df['color'] == 0, d_df[
        'white_max_move'], d_df['black_max_move'])

    d_df['post_elo'] = np.where(
        d_df['color'] == 1, d_df['white_elo'], d_df['black_elo'])
    d_df['opp_post_elo'] = np.where(
        d_df['color'] == 0, d_df['white_elo'], d_df['black_elo'])

    d_df['elo_delta'] = d_df['post_elo'] - d_df['post_elo'].shift(1)

    d_df['elo'] = d_df['post_elo'].subtract(d_df['elo_delta'])
    d_df['elo'].iloc[0] = 1000

    d_df['elo_delta'].iloc[0] = d_df['post_elo'].iloc[0] - d_df['elo'].iloc[0]

    d_df['opp_elo'] = d_df['opp_post_elo'].subtract(d_df['elo_delta'])

    d_df['diff'] = d_df['post_elo'].subtract(d_df['opp_post_elo'])

    d_df = d_df.reset_index().drop(columns=['index'])

    d_df_len = len(d_df)

    d_df = d_df.drop([d_df_len-1])

    d_df['won_by'] = d_df['won_by'].replace([
        'checkmate', 'resignation', 'time', 'material', 'agreement',
        'repetition', 'abandoned', 'stalemate', 'rule'], list(reversed(range(9))))

    return d_df


def main_cleanup(file_name):
    icd_text = initial_chess_data(file_name)

    cdf = chess_data_cleanup(icd_text)

    adf = cdf[cdf.index % 2 == 0]

    adf.to_csv('../data/moves_initial.csv')

    mdf1, ddf1 = data_cleaning_1(cdf)

    mdf2, tdf1 = data_cleaning_2(mdf1, ddf1)

    mdf2.to_csv('../data/moves.csv')

    tdf2 = data_cleaning_3(tdf1, ddf1)

    wh_mdf1, wh_tdf1, bl_mdf1, bl_tdf1, ddf2, tdf3 = data_cleaning_4(
        mdf2, tdf2, ddf1)

    wh_tmdf1 = data_cleaning_5(wh_tdf1, tdf3, ddf2, '001a')
    bl_tmdf1 = data_cleaning_5(bl_tdf1, tdf3, ddf2, '001b')

    ddf3 = data_cleaning_6(ddf2, mdf2, bl_mdf1, wh_mdf1, wh_tdf1, bl_tdf1)

    ddf_final = data_cleaning_7(ddf3, wh_tmdf1, bl_tmdf1)

    analysis_labels = ['date', 'day', 'weekday', 'start_time', 'game_time',
                       'color', 'elo', 'opp_elo', 'diff', 'result', 'won_by',
                       'num_moves', 'opp_num_moves', 'avg_time', 'opp_avg_time',
                       'castled_on', 'opp_castled_on', 'castled', 'opp_castled',
                       'time_used', 'opp_time_used', 'max_move', 'opp_max_move']

    predictions_labels = ['result', 'diff', 'opp_elo', 'elo', 'game_time',
                          'color', 'start_time', 'day', 'weekday']

    ddf_model = ddf_final[predictions_labels]
    ddf_analysis = ddf_final[analysis_labels]

    return ddf_model, ddf_analysis, ddf_final