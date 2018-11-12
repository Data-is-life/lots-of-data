# coding: utf-8

# # Analysis:

from itertools import islice
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')


def clean_analysis_df(file_name):
    df = pd.read_csv(file_name)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df['time_used'] = round((df['time_used'] / df['game_time']) * 100)
    df['opp_time_used'] = round((df['opp_time_used'] / df['game_time']) * 100)
    df = df[df['time_used'] <= 100]
    df = df[df['opp_time_used'] <= 100]
    return df


def graph_for_start_time(df):
    df_mean_by_start_time = df.groupby('start_time').mean()
    df_mean_by_start_time.reset_index(inplace=True)
    df_mean_by_start_time.plot.scatter(x='start_time', y='result',
                                       title='Result by Time of Day', legend=False)
    plt.ylim(0.4, 0.625)
    plt.ylabel('Winning Ratio')
    plt.xlabel('Starting Time')
    df_count_by_start_time = df.groupby('start_time').count()
    df_count_by_start_time.reset_index(inplace=True)
    df_count_by_start_time.plot.scatter(x='start_time', y='result',
                                        title='# of Games by Time of Day',
                                        legend=False)
    plt.ylabel('# of Games')
    plt.xlabel('Starting Time')
    plt.show()


def graph_for_day_of_month(df):
    df_mean_by_day = df.groupby('day').mean()
    df_mean_by_day.reset_index(inplace=True)
    df_mean_by_day.plot.scatter(x='day', y='result', legend=False,
                                title='Result by Day of the Month')
    plt.ylim(0.4, 0.6)
    plt.ylabel('Winning Ratio')
    plt.xlabel('Day of the Month')
    df_count_by_day = df.groupby('day').count()
    df_count_by_day.reset_index(inplace=True)
    df_count_by_day.plot.scatter(x='day', y='result', legend=False,
                                 title='# of Games by Day of the Month')
    plt.ylabel('# of Games')
    plt.xlabel('Day of the Month')
    plt.show()


def graph_for_day_of_week(df):
    df_mean_by_weekday = df.groupby('weekday').mean()
    df_mean_by_weekday.reset_index(inplace=True)
    df_mean_by_weekday.plot.scatter(x='weekday', y='result', marker='o',
                                    title='Result by Day of the Week',
                                    legend=False)
    plt.ylim(0.44, 0.525)
    plt.yticks(ticks=np.arange(0.445, 0.525, step=0.01))
    plt.xticks(ticks=np.arange(7), labels=[
               'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.ylabel('Winning Ratio')
    plt.xlabel('Weekday')
    df_count_by_weekday = df.groupby('weekday').count()
    df_count_by_weekday.reset_index(inplace=True)
    df_count_by_weekday.plot.scatter(x='weekday', y='result', marker='o',
                                     title='# of Games by Day of the Week',
                                     legend=False)
    plt.xticks(ticks=np.arange(7), labels=[
               'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.ylabel('# of Games')
    plt.xlabel('Weekday')
    plt.show()


def graph_for_where_castled(df):
    df_mean_by_castled = df.groupby('castled').mean()
    df_mean_by_castled.reset_index(inplace=True)
    df_mean_by_castled.plot.scatter(x='castled', y='result', marker='o',
                                    title='Result by Where Castled', legend=False)
    plt.ylabel('Winning Ratio')
    plt.xlabel('Castled')
    df_count_by_castled = df.groupby('castled').count()
    df_count_by_castled.reset_index(inplace=True)
    df_count_by_castled.plot.scatter(x='castled', y='result', marker='o',
                                     title='# of Games for Where Castled', legend=False)
    plt.ylabel('# of Games')
    plt.xlabel('Castled')
    plt.show()


def graph_for_where_opp_castled(df):
    df_mean_by_opp_castled = df.groupby('opp_castled').mean()
    df_mean_by_opp_castled.reset_index(inplace=True)
    df_mean_by_opp_castled.plot.scatter(x='opp_castled', y='result', marker='o',
                                        title='Result by Where Opp Castled', legend=False)
    plt.ylim(0.44, 0.58)
    plt.ylabel('Winning Ratio')
    plt.xlabel('Opposition Castled')
    df_count_by_opp_castled = df.groupby('opp_castled').count()
    df_count_by_opp_castled.reset_index(inplace=True)
    df_count_by_opp_castled.plot.scatter(x='opp_castled', y='result', marker='o',
                                         title='# of Games for Where Opp Castled', legend=False)
    plt.ylabel('# of Games')
    plt.xlabel('Opposition Castled')
    plt.show()


def graph_for_timed_game(df):
    df_mean_by_game_time = df.groupby('game_time').mean()
    df_mean_by_game_time.reset_index(inplace=True)
    df_mean_by_game_time.plot.scatter(x='game_time', y='result', marker='o',
                                      title='Result by Game Time', legend=False)
    plt.ylim(0.4, 0.54)
    plt.ylabel('Winning Ratio')
    plt.xlabel('Game Time')
    df_count_by_game_time = df.groupby('game_time').count()
    df_count_by_game_time.reset_index(inplace=True)
    df_count_by_game_time.plot.scatter(x='game_time', y='result', marker='o',
                                       title='# of Games for Each Game Time', legend=False)
    plt.ylabel('# of Games')
    plt.xlabel('Game Time')
    plt.show()

    df_mean_by_color = df.groupby('color').mean()
    df_mean_by_color.reset_index(inplace=True)
    df_mean_by_color.plot.scatter(x='color', y='result', marker='o',
                                  title='Result by Color', legend=False)
    plt.ylim(0.46, 0.53)
    plt.ylabel('Winning Ratio')
    plt.xlabel('Color')
    plt.show()


def bin_columns(df):
    bin_opp_elo = [650, 1050, 1150, 1250, 1350, 1450, 1650, 1750, 1850]
    bin_opp_elo.extend(list(range(660, 1041, 10)))
    bin_opp_elo = sorted(bin_opp_elo)

    bin_diff = [-500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600]
    bin_diff.extend(list(range(-90, 91, 5)))
    bin_diff = sorted(bin_diff)

    df['bin_elo'] = pd.cut(x=df.elo, bins=list(
        range(650, 1051, 10)), labels=list(range(650, 1041, 10))).astype(int)
    df['bin_opp_elo'] = pd.cut(
        x=df.opp_elo, bins=bin_opp_elo, labels=bin_opp_elo[:-1]).astype(float)
    df['bin_diff'] = pd.cut(x=df['diff'], bins=bin_diff,
                            labels=bin_diff[:-1]).astype(float)
    df['bin_num_moves'] = pd.cut(x=df.num_moves, bins=list(
        range(0, 151, 5)), labels=list(range(0, 146, 5))).astype(float)
    df['bin_time_used'] = pd.cut(x=df.time_used, bins=list(
        range(0, 101, 5)), labels=list(range(0, 96, 5))).astype(float)
    df['bin_opp_time_used'] = pd.cut(x=df.opp_time_used, bins=list(
        range(0, 101, 5)), labels=list(range(0, 96, 5))).astype(float)
    return df


def graph_for_elo(df):
    df_mean_by_elo = df.groupby('bin_elo').mean()
    df_mean_by_elo.reset_index(inplace=True)
    df_mean_by_elo.plot.scatter(x='bin_elo', y='result',
                                title='Result by ELO', legend=False)
    plt.ylabel('Winning Ratio')
    plt.xlabel('ELO')
    df_count_by_elo = df.groupby('bin_elo').count()
    df_count_by_elo.reset_index(inplace=True)
    df_count_by_elo.plot.scatter(x='bin_elo', y='result',
                                 title='# of Games by ELO',
                                 legend=False)
    plt.ylabel('# of Games')
    plt.xlabel('ELO')
    plt.show()


def graph_for_opp_elo(df):
    df_mean_by_opp_elo = df.groupby('bin_opp_elo').mean()
    df_mean_by_opp_elo.reset_index(inplace=True)
    df_mean_by_opp_elo.plot.scatter(x='bin_opp_elo', y='result',
                                    title='Result by Opp ELO',
                                    legend=False)
    plt.ylabel('Winning Ratio')
    plt.xlabel('Opposition ELO')
    plt.xlim((600, 1100))
    df_count_by_opp_elo = df.groupby('bin_opp_elo').count()
    df_count_by_opp_elo.reset_index(inplace=True)
    df_count_by_opp_elo.plot.scatter(x='bin_opp_elo', y='result', legend=False,
                                     title='# of Games by Opp ELO')
    plt.ylabel('# of Games')
    plt.xlabel('Opposition ELO')
    plt.xlim((600, 1100))
    plt.show()


def graph_for_elo_diff(df):
    df_mean_by_diff = df.groupby('bin_diff').mean()
    df_mean_by_diff.reset_index(inplace=True)
    df_mean_by_diff.plot.scatter(x='bin_diff', y='result', legend=False,
                                 title='Result by Difference in ELO')
    plt.xlim((-125, 125))
    plt.ylim((0, 1))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylabel('Winning Ratio')
    plt.xlabel('Difference in ELO')
    df_count_by_diff = df.groupby('bin_diff').count()
    df_count_by_diff.reset_index(inplace=True)
    df_count_by_diff.plot.scatter(x='bin_diff', y='result', legend=False,
                                  title='# of Games by Difference in ELO')
    plt.ylabel('# of Games')
    plt.xlabel('Difference in ELO')
    plt.xlim((-125, 125))
    plt.show()


def graph_for_result_type(df):
    df_mean_by_won_by = df.groupby('won_by').mean()
    df_mean_by_won_by.reset_index(inplace=True)
    df_mean_by_won_by.plot.scatter(x='won_by', y='result',
                                   title='Result by the Result Type')
    plt.ylabel('Winning Ratio')
    plt.xlabel('Result Type')
    df_count_by_won_by = df.groupby('won_by').count()
    df_count_by_won_by.reset_index(inplace=True)
    df_count_by_won_by.plot.scatter(x='won_by', y='result', legend=False,
                                    title='# of Games by the Result Type')
    plt.ylabel('# of Games')
    plt.xlabel('Result Type')
    plt.show()


def graph_for_num_moves(df):
    df_mean_by_num_moves = df.groupby('bin_num_moves').mean()
    df_mean_by_num_moves.reset_index(inplace=True)
    df_mean_by_num_moves.plot.scatter(x='bin_num_moves', y='result',
                                      title='Result by Number of Moves')
    plt.ylim((.35, .75))
    plt.ylabel('Winning Ratio')
    plt.xlabel('Number of Moves')
    df_count_by_num_moves = df.groupby('bin_num_moves').count()
    df_count_by_num_moves.reset_index(inplace=True)
    df_count_by_num_moves.plot.scatter(x='bin_num_moves', y='result', legend=False,
                                       title='# of Games by Number of Moves')
    plt.ylabel('# of Games')
    plt.xlabel('Number of Moves')
    plt.show()


def graph_for_castle_on_move_num(df):
    df_mean_by_castled_on = df.groupby('castled_on').mean()
    df_mean_by_castled_on.reset_index(inplace=True)
    df_mean_by_castled_on.plot.scatter(x='castled_on', y='result', legend=False,
                                       title='Result by the Move # of Castling (0 = not castling)')
    plt.ylim((.35, 1.05))
    plt.ylabel('Winning Ratio')
    plt.xlabel('Move # of Castling')
    df_count_by_castled_on = df.groupby('castled_on').count()
    df_count_by_castled_on.reset_index(inplace=True)
    df_count_by_castled_on.plot.scatter(x='castled_on', y='result', legend=False,
                                        title='# of Games by the Move # of Castling (0 = not castling)')
    plt.ylim((-5, 205))
    plt.ylabel('# of Games')
    plt.xlabel('Move # of Castling')
    plt.show()


def graph_for_opp_castle_on_move_num(df):
    df_mean_by_opp_castled_on = df.groupby('opp_castled_on').mean()
    df_mean_by_opp_castled_on.reset_index(inplace=True)
    df_mean_by_opp_castled_on.plot.scatter(x='opp_castled_on', y='result', ylim=(0.25, 0.75),
                                           title='Result by the Move # of Opp Castling')
    plt.ylabel('Winning Ratio')
    plt.xlabel('Move # of Opposition Castling')
    df_count_by_opp_castled_on = df.groupby('opp_castled_on').count()
    df_count_by_opp_castled_on.reset_index(inplace=True)
    df_count_by_opp_castled_on.plot.scatter(x='opp_castled_on', y='result', legend=False, ylim=(-5, 140),
                                            title='# of Games by the Move # of Opp Castling')
    plt.ylabel('# of Games')
    plt.xlabel('Move # of Opposition Castling')
    plt.show()


def graph_for_time_used(df):
    df_mean_by_time_used = df.groupby('time_used').mean()
    df_mean_by_time_used.reset_index(inplace=True)
    df_mean_by_time_used.plot.scatter(x='time_used', y='result', marker='o',
                                      title='Result by the Amount of Time Used (in %)')
    plt.ylabel('Winning Ratio')
    plt.xlabel('Amount of Time Used(%)')
    df_count_by_time_used = df.groupby('time_used').count()
    df_count_by_time_used.reset_index(inplace=True)
    df_count_by_time_used.plot.scatter(x='time_used', y='result', legend=False,
                                       title='# of Games by the Amount of Time Used (in %)')
    plt.ylim((-5, 70))
    plt.ylabel('# of Games')
    plt.xlabel('Amount of Time Used(%)')
    plt.show()


def graph_for_opp_time_used(df):
    df_mean_by_opp_time_used = df.groupby('opp_time_used').mean()
    df_mean_by_opp_time_used.reset_index(inplace=True)
    df_mean_by_opp_time_used.plot.scatter(x='opp_time_used', y='result',
                                          title='Result by the Amount of Time Used by Opp')
    plt.ylabel('Winning Ratio')
    plt.xlabel('Amount of Time Used by Opposition(%)')
    df_count_by_opp_time_used = df.groupby('opp_time_used').count()
    df_count_by_opp_time_used.reset_index(inplace=True)
    df_count_by_opp_time_used.plot.scatter(x='opp_time_used', y='result', legend=False,
                                           title='# of Games by the Amount of Time Used by Opp')
    plt.ylim((0, 70))
    plt.ylabel('# of Games')
    plt.xlabel('Amount of Time Used by Opposition(%)')
    plt.show()


def analyze_by_graph(file_name, type_of_graph):
    df = clean_analysis_df(file_name)
    df = bin_columns(df)

    if type_of_graph == 'start_time':
        graph_for_start_time(df)
    elif type_of_graph == 'start_time':
        graph_for_day_of_month(df)
    elif type_of_graph == 'start_time':
        graph_for_day_of_week(df)
    elif type_of_graph == 'start_time':
        graph_for_where_castled(df)
    elif type_of_graph == 'start_time':
        graph_for_where_opp_castled(df)
    elif type_of_graph == 'start_time':
        graph_for_timed_game(df)
    elif type_of_graph == 'start_time':
        graph_for_elo(df)
    elif type_of_graph == 'start_time':
        graph_for_opp_elo(df)
    elif type_of_graph == 'start_time':
        graph_for_elo_diff(df)
    elif type_of_graph == 'start_time':
        graph_for_result_type(df)
    elif type_of_graph == 'start_time':
        graph_for_num_moves(df)
    elif type_of_graph == 'start_time':
        graph_for_castle_on_move_num(df)
    elif type_of_graph == 'start_time':
        graph_for_opp_castle_on_move_num(df)
    elif type_of_graph == 'start_time':
        graph_for_time_used(df)
    elif type_of_graph == 'start_time':
        graph_for_opp_time_used(df)
