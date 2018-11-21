# Author: Mohit Gangwani
# Date: 11/08/2018
# Git-Hub: Data-is-Life

import numpy as np
from ast import literal_eval
from pandas import DataFrame, to_datetime, to_timedelta, to_numeric

'''All functions I used to clean up game log from Chess.com'''


def custom_round(x, base=20):
    '''Helps to round digits'''

    return int(base * round(float(x) / base))


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

    cl = ''.join([num for num in chess_text]).split(
        "~")  # I used '~' as a separator

    # Named the only column "a", so it is easier in the next function
    df = DataFrame(cl, columns=['a'])

    # If length of any string is less than 3, it is not needed
    df = df[df['a'].str.len() > 3]

    return df


def data_cleaning_1(df):
    '''Third function:
    Input:
    df = df with all information
    creates two dfs. First df is all games information. Second df is for all
    the moves in those games.
    Output:
    m_df = moves df with all the moves
    d_df = information df with all the game information'''

    c_df = DataFrame(data=list(df['a'].apply(literal_eval)))

    c_df['Date'].fillna(method='ffill', inplace=True)
    c_df['EndTime'].fillna(method='ffill', inplace=True)

    # Convert all the dates and time to dates and times
    c_df.loc[:, 'date_time'] = to_datetime(
        c_df['Date'] + ' ' + c_df['EndTime'])
    c_df.loc[:, 'Date'] = to_datetime(c_df['Date'])
    c_df.loc[:, 'EndTime'] = to_timedelta(c_df['EndTime'])

    # Split moves to a new df drop columns not needed
    m_df = c_df[c_df.index % 2 == 1].copy()
    m_df.sort_values('date_time', inplace=True)
    m_df.reset_index(inplace=True)
    m_df.drop(columns=[
        'index', 'Date', 'White', 'Black', 'Result', 'WhiteElo',
        'BlackElo', 'TimeControl', 'EndTime', 'Termination', 'date_time',
        'Round', 'Event'], inplace=True)

    # Split game information to a new df
    d_df = c_df[c_df.index % 2 == 0].copy()

    d_df = d_df[['Date', 'White', 'Black', 'Result', 'WhiteElo', 'BlackElo',
                 'TimeControl', 'EndTime', 'Termination', 'date_time']]
    d_df.sort_values('date_time', inplace=True)
    d_df.reset_index(inplace=True)
    d_df.drop(columns=['index', 'date_time'], inplace=True)

    # Rename all columns to lower case and insert "_" to split words
    d_df.rename(columns={
        'Date': 'date', 'White': 'white', 'Black': 'black',
        'Result': 'result', 'WhiteElo': 'white_elo', 'BlackElo': 'black_elo',
        'TimeControl': 'game_time', 'EndTime': 'end_time',
        'Termination': 'termination'}, inplace=True)

    d_df.loc[:, 'num_moves'] = m_df.count(axis=1)
    d_df.loc[:, 'white_elo'] = to_numeric(d_df['white_elo'])
    d_df.loc[:, 'black_elo'] = to_numeric(d_df['black_elo'])
    d_df.loc[:, 'color'] = np.where(d_df['white'] == 'TrueMoeG', 1, 0)

    # Drop duplicate rows
    d_df.drop_duplicates(inplace=True)
    m_df.drop_duplicates(inplace=True)

    return m_df, d_df


def data_cleaning_2(m_df):
    '''Fourth function:
    Input:
    m_df = Moves df
    Creates a new df that has time information and cleans moves df column names
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
    Cleans the times df to fix the games that give extra time after each move
    Output:
    t_df = Moves time df - Fixed times'''

    t_df = t_df.apply(to_timedelta, errors='coerce')
    t_df = t_df.apply(to_numeric, errors='coerce')
    t_df = t_df.div(1000000000)

    t_df.loc[:, 'game_time'] = d_df['game_time']
    t_df.loc[:, 'extra_time'] = t_df['game_time'].replace([
        '300', '600', '180', '180+2', '300+5'], ['0', '0', '0', '2', '5'])
    t_df.loc[:, 'game_time'] = d_df['game_time'].replace(
        ['180+2', '300+5'], [180, 300])

    t_df.loc[:, 'extra_time'] = to_numeric(t_df['extra_time'])

    return t_df


def data_cleaning_4(m_df, t_df, d_df):
    '''Sixth function:
    Input:
    m_df = Moves df
    t_df = Move times df
    d_df = Game information df
    Creates four new df moves and move times for white and black pieces.
    Output:
    wh_m_df = All moves by player with white pieces
    wh_t_df = All moves time by player with white pieces
    bl_m_df = All moves by player with black pieces
    bl_t_df = All moves time by player with black pieces
    d_df = Game information df - Corrected game times + number of moves
    t_df = Move times df'''

    # game_time to numeric and copy those to moves time df
    t_df.loc[:, 'game_time'] = to_numeric(t_df['game_time'])
    d_df.loc[:, 'game_time'] = t_df['game_time']

    # Create four new df moves and move times for white and black pieces
    wh_m_df = m_df[m_df.columns[::2]].copy()
    bl_m_df = m_df[m_df.columns[1::2]].copy()
    wh_t_df = t_df[t_df.columns[::2]].copy()
    bl_t_df = t_df[t_df.columns[1::2]].copy()

    wh_t_df.drop(columns=[wh_t_df.columns[-1]], inplace=True)
    bl_t_df.drop(columns=[bl_t_df.columns[-1]], inplace=True)

    # Get number of moves per game
    d_df.loc[:, 'white_num_moves'] = wh_m_df.count(axis=1)
    d_df.loc[:, 'black_num_moves'] = bl_m_df.count(axis=1)

    '''Go through all the columns in the move times df for white and black
    pieces and subtract the time left with the total allowed time. This gives
    time per move'''
    for num in wh_t_df.columns:
        wh_t_df.loc[:, num] = t_df['game_time'] - wh_t_df[num]
    for num in bl_t_df.columns:
        bl_t_df.loc[:, num] = t_df['game_time'] - bl_t_df[num]

    # Create two lists: index for games that give +2 sec/move and +5 sec/move
    two_list = t_df[t_df['extra_time'] == 2].index.tolist()
    five_list = t_df[t_df['extra_time'] == 5].index.tolist()

    # Add 2 seconds to all the moves for games that give +2 sec/move
    for num in two_list:
        for i in range(len(wh_t_df.columns) - 1):
            wh_t_df.iloc[num, i] = wh_t_df.iloc[num, i] + ((i + 1) * 2)
        for j in range(len(wh_t_df.columns) - 1):
            bl_t_df.iloc[num, j] = wh_t_df.iloc[num, j] + ((j + 1) * 2)

    # Add 5 seconds to all the moves for games that give +5 sec/move
    for num in five_list:
        for i in range(len(wh_t_df.columns) - 1):
            wh_t_df.iloc[num, i] = wh_t_df.iloc[num, i] + ((i + 1) * 5)
        for j in range(len(wh_t_df.columns) - 1):
            bl_t_df.iloc[num, j] = wh_t_df.iloc[num, j] + ((j + 1) * 5)

    # Change time values where time is really high
    for num in wh_t_df.columns:
        wh_t_df.loc[:, num] = np.where(wh_t_df[num] > 5000, 0, wh_t_df[num])
    for num in bl_t_df.columns:
        bl_t_df.loc[:, num] = np.where(bl_t_df[num] > 5000, 0, bl_t_df[num])

    return wh_m_df, wh_t_df, bl_m_df, bl_t_df, d_df, t_df


def data_cleaning_5(c_t_df, t_df, d_df, col):
    '''Seventh function:
    Input:
    c_t_df = Black or white pieces moves time df
    t_df = Move times df
    d_df = Game information df
    col = Column name
    Cleans the moves time df.
    Output:
    tm_df = All moves by player with white pieces'''

    tm_df = c_t_df.shift(periods=1, axis=1).copy()
    tm_df = tm_df - c_t_df
    tm_df = -tm_df

    tm_df.loc[:, col] = c_t_df[col]

    for num in tm_df.columns:
        tm_df.loc[:, num] = np.where(tm_df[num] <= 0, 0, tm_df[num])

    return tm_df


def help_func1(m_df, d_df):
    '''Helper function:
    Input:
    m_df = Black or white pieces moves
    Gets castling information.
    Output:
    cast_list = for every game it assigns a value if the player castled.
                1 if castled King side
                0 if castled Queen side
                -1 if didn't castle
    cast_w_list = if the player castled it gets the move number the player
                  castled
                  0 if the player didn't castle'''

    cast_list = []
    cast_w_list = []
    for i in range(len(d_df)):
        a = list(m_df.iloc[i])
        if "O-O" in a:
            cast_list.append(a.index("O-O") + 1)
            cast_w_list.append(1)
        elif "O-O-O" in a:
            cast_list.append(a.index("O-O-O") + 1)
            cast_w_list.append(0)
        else:
            cast_list.append(0)
            cast_w_list.append(-1)
    return cast_list, cast_w_list


def data_cleaning_6(d_df, m_df, bl_m_df, wh_m_df, wh_t_df, bl_t_df):
    '''Eighth function:
    Input:
    d_df = Game information df
    m_df = Move times df
    bl_m_df = All moves by player with black pieces
    wh_m_df = All moves by player with white pieces
    wh_t_df = All moves time by player with white pieces
    bl_t_df = All moves time by player with black pieces
    Adds bunch of information to game information df
    Output:
    d_df = Game information df - bunch of new columns'''

    # Round all times to an integer
    d_df.loc[:, 'white_time_used'] = wh_t_df.max(axis=1).apply(
        lambda x: custom_round(x, base=10))
    d_df.loc[:, 'black_time_used'] = bl_t_df.max(axis=1).apply(
        lambda x: custom_round(x, base=10))

    # Get the winner and how they won
    d_df.loc[:, 'winner'] = d_df['termination'].str.extract(
        '(^[a-zA-Z0-9]+)', expand=False)
    d_df.loc[:, 'won_by'] = d_df['termination'].str.extract(
        '([a-zA-Z0-9]+$)', expand=False)

    # Using helper function to get castling information
    cstl_l_bl, cstl_loc_l_bl = help_func1(bl_m_df, d_df)
    cstl_l_wh, cstl_loc_l_wh = help_func1(wh_m_df, d_df)

    # Get day of the week and day of the month
    d_df.loc[:, 'weekday'] = d_df.date.apply(lambda x: x.dayofweek)
    d_df.loc[:, 'day'] = d_df.date.apply(lambda x: x.day)

    # result is if the player won or lost. 1.0 = Win, 0.5 = Draw, 0.0 = Loss
    d_df.loc[:, 'result'] = np.where(d_df['winner'] == 'TrueMoeG',
                                     1.0, (np.where(d_df['winner'] == 'Game',
                                                    0.5, 0.0)))

    d_df.loc[:, 'white_castled_on'] = cstl_l_wh
    d_df.loc[:, 'black_castled_on'] = cstl_l_bl
    d_df.loc[:, 'white_castled_where'] = cstl_loc_l_wh
    d_df.loc[:, 'black_castled_where'] = cstl_loc_l_bl

    d_df.loc[:, 'castled_on'] = np.where(d_df['color'] == 1, d_df[
        'white_castled_on'], d_df['black_castled_on'])
    d_df.loc[:, 'opp_castled_on'] = np.where(d_df['color'] != 1, d_df[
        'white_castled_on'], d_df['black_castled_on'])
    d_df.loc[:, 'castled'] = np.where(d_df['color'] == 1, d_df[
        'white_castled_where'], d_df['black_castled_where'])
    d_df.loc[:, 'opp_castled'] = np.where(d_df['color'] != 1, d_df[
        'white_castled_where'], d_df['black_castled_where'])

    # Get total time used by each player and input it in the information df
    d_df.loc[:, 'time_used'] = np.where(d_df['color'] == 1, d_df[
        'white_time_used'], d_df['black_time_used'])
    d_df.loc[:, 'opp_time_used'] = np.where(d_df['color'] == 0, d_df[
        'white_time_used'], d_df['black_time_used'])
    d_df.loc[:, 'time_used'] = np.where(d_df['result'] == 1.0, d_df[
        'time_used'], np.where(d_df['won_by'] == 'time', d_df[
            'game_time'], d_df['time_used']))
    d_df.loc[:, 'opp_time_used'] = np.where(d_df['result'] == 0.0, d_df[
        'opp_time_used'], np.where(d_df['won_by'] == 'time', d_df[
            'game_time'], d_df['opp_time_used']))

    # Converting time to numeric for easier calculations
    d_df.loc[:, 'end_time'] = to_numeric(d_df['end_time']) / 3600000000
    d_df.loc[:, 'start_time'] = d_df['end_time'] - \
        (d_df['time_used'] + d_df['opp_time_used']) / 3.6

    d_df.loc[:, 'start_time'] = [num if num >=
                                 0 else num + 24000 for num in d_df.start_time]

    d_df.loc[:, 'num_moves'] = np.where(d_df['color'] == 1, d_df[
        'white_num_moves'], d_df['black_num_moves'])
    d_df.loc[:, 'opp_num_moves'] = np.where(d_df['color'] == 0, d_df[
        'white_num_moves'], d_df['black_num_moves'])

    # Average time per move
    d_df.loc[:, 'avg_time'] = d_df['time_used'] / d_df['num_moves']
    d_df.loc[:, 'opp_avg_time'] = d_df['opp_time_used'] / d_df['opp_num_moves']

    # Rounding the time to start of the hour
    d_df.loc[:, 'start_time'] = d_df['start_time'] // 1000
    d_df.loc[:, 'end_time'] = d_df['end_time'] // 1000

    return d_df.drop(columns=[
        'white', 'black', 'termination', 'white_num_moves', 'black_num_moves',
        'white_time_used', 'black_time_used', 'winner', 'white_castled_on',
        'black_castled_on', 'white_castled_where', 'black_castled_where'])


def data_cleaning_7(d_df, wh_tm_df, bl_tm_df):
    '''Ninth function:
    Input:
    d_df = Game information df
    bl_m_df = All moves by player with black pieces
    wh_m_df = All moves by player with white pieces
    Adds bunch of information to game information df
    Output:
    d_df = Game information df - bunch of new columns'''

    # Max time each player to make a move
    d_df.loc[:, 'white_max_move'] = wh_tm_df.max(axis=1)
    d_df.loc[:, 'black_max_move'] = bl_tm_df.max(axis=1)

    d_df.loc[:, 'max_move'] = np.where(d_df['color'] == 1, d_df[
        'white_max_move'], d_df['black_max_move'])
    d_df.loc[:, 'opp_max_move'] = np.where(d_df['color'] == 0, d_df[
        'white_max_move'], d_df['black_max_move'])

    # Assign elo to each player
    d_df.loc[:, 'post_elo'] = np.where(
        d_df['color'] == 1, d_df['white_elo'], d_df['black_elo'])
    d_df.loc[:, 'opp_post_elo'] = np.where(
        d_df['color'] == 0, d_df['white_elo'], d_df['black_elo'])

    # Amount of elo changed in the last game
    d_df.loc[:, 'elo_delta'] = d_df['post_elo'] - d_df['post_elo'].shift(1)
    d_df.loc[:, 'elo'] = d_df['post_elo'].subtract(d_df['elo_delta'])

    # Chess assigns elo of 1000 to a new member
    d_df.loc[0, 'elo'] = 1000
    d_df.loc[0, 'elo_delta'] = d_df['post_elo'].iloc[0] - d_df['elo'].iloc[0]
    d_df.loc[:, 'opp_elo'] = d_df['opp_post_elo'].subtract(d_df['elo_delta'])

    # diff is the difference in elo between players
    d_df.loc[:, 'diff'] = d_df['post_elo'].subtract(d_df['opp_post_elo'])

    d_df.reset_index(inplace=True)
    d_df.drop(columns=['index'], inplace=True)

    d_df_len = len(d_df)

    d_df.drop([d_df_len - 1], inplace=True)

    # Changed stings of how the player won to integers
    d_df['won_by'].replace(['checkmate', 'resignation', 'time', 'material', 'agreement',
                            'repetition', 'abandoned', 'stalemate', 'rule'],
                           list(reversed(range(9))), inplace=True)

    d_df.drop(columns=['white_elo', 'black_elo', 'white_max_move',
                     'black_max_move'], inplace=True)

    return d_df


def main_cleanup(file_name):
    '''Tenth function:
    Input:
    file_name = Game log from Chess.com
    This puts all the functions in one function
    Output:
    df_model = Use for building prediction models
    df_analysis = Use for analysis
    df_final = All the game information'''

    # First function
    icd_text = initial_chess_data(file_name)

    # Second function
    cdf = chess_data_cleanup(icd_text)

    # Just to create a file with all the moves and times
    adf = cdf[cdf.index % 2 == 0].copy()
    adf.to_csv('../data/moves_initial.csv')
    # Third function
    mdf1, ddf1 = data_cleaning_1(cdf)

    # Fourth function
    mdf2, tdf1 = data_cleaning_2(mdf1)

    # No more need for moves. Saving it to a file
    mdf2.to_csv('../data/moves.csv')

    # Fifth function
    tdf2 = data_cleaning_3(tdf1, ddf1)

    # Sixth function
    wh_mdf1, wh_tdf1, bl_mdf1, bl_tdf1, ddf2, tdf3 = data_cleaning_4(
        mdf2, tdf2, ddf1)

    # Seventh function
    wh_tmdf1 = data_cleaning_5(wh_tdf1, tdf3, ddf2, '001a')
    bl_tmdf1 = data_cleaning_5(bl_tdf1, tdf3, ddf2, '001b')

    # Eighth function
    ddf3 = data_cleaning_6(ddf2, mdf2, bl_mdf1, wh_mdf1, wh_tdf1, bl_tdf1)

    # Ninth function
    df_final = data_cleaning_7(ddf3, wh_tmdf1, bl_tmdf1)

    # Using the following columns to run analysis
    analysis_labels = ['date', 'day', 'weekday', 'start_time', 'game_time',
                       'color', 'elo', 'opp_elo', 'diff', 'result', 'won_by',
                       'num_moves', 'castled', 'opp_castled', 'castled_on',
                       'opp_castled_on', 'time_used', 'opp_time_used']

    # Using the following columns for running prediction models
    predictions_labels = ['result', 'diff', 'opp_elo', 'elo', 'game_time',
                          'color', 'start_time', 'day', 'weekday']

    df_model = df_final[predictions_labels].copy()
    df_analysis = df_final[analysis_labels].copy()

    # Save the files
    df_final.to_csv('../data/main_with_all_info.csv', index=False)
    df_model.to_csv('../data/use_for_predictions.csv', index=False)
    df_analysis.to_csv('../data/use_for_analysis.csv', index=False)

    return df_model, df_analysis, df_final
