from itertools import islice
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
plt.style.use('seaborn-whitegrid')


def graph_start_time(df):
    df_mean_by_start_time = df.groupby('start_time').mean()
    df_mean_by_start_time.reset_index(inplace=True)
    df_count_by_start_time = df.groupby('start_time').count()
    df_count_by_start_time.reset_index(inplace=True)
    df_mean_by_start_time.plot.scatter(x='start_time', y='result',
                                       legend=False)
    plt.title('Result by Time of Day')
    plt.xlabel('Starting Time')
    plt.xlim(-.25, 24)
    plt.xticks(ticks=np.arange(0, 25, step=4))
    plt.ylabel('Winning Ratio')
    plt.ylim(0.4, 0.600)
    plt.yticks(ticks=np.arange(0.4, 0.601, step=0.02))

    df_count_by_start_time.plot.scatter(x='start_time', y='result',
                                        legend=False)
    plt.title('Result by Time of Day')
    plt.xlabel('Starting Time')
    plt.xlim(-.25, 24)
    plt.xticks(ticks=np.arange(0, 25, step=4))
    plt.ylabel('# of Games')
    plt.ylim(0, 200)
    plt.yticks(ticks=np.arange(0, 201, step=20))
    plt.show()


def graph_day_of_month(df):
    df_mean_by_day = df.groupby('day').mean()
    df_mean_by_day.reset_index(inplace=True)
    df_count_by_day = df.groupby('day').count()
    df_count_by_day.reset_index(inplace=True)

    df_mean_by_day.plot.scatter(x='day', y='result', legend=False)
    plt.title('Result by Day of the Month')
    plt.xlabel('Day of the Month')
    plt.xlim(0, 32)
    plt.xticks(ticks=np.arange(0, 32, step=5))
    plt.ylabel('Winning Ratio')
    plt.ylim(0.4, 0.6)
    plt.yticks(ticks=np.arange(0.4, 0.61, step=0.02))

    df_count_by_day.plot.scatter(x='day', y='result', legend=False)
    plt.title('# of Games by Day of the Month')
    plt.xlabel('Day of the Month')
    plt.xlim(0, 32)
    plt.xticks(ticks=np.arange(0, 32, step=5))
    plt.ylabel('# of Games')
    plt.ylim(30, 100)
    plt.yticks(ticks=np.arange(30, 101, step=5))
    plt.show()


def day_of_week(df):
    df_mean_by_weekday = df.groupby('weekday').mean()
    df_mean_by_weekday.reset_index(inplace=True)
    df_count_by_weekday = df.groupby('weekday').count()
    df_count_by_weekday.reset_index(inplace=True)

    df_mean_by_weekday.plot.scatter(x='weekday', y='result', legend=False)
    plt.title('Result by Day of the Week')
    plt.xlabel('Weekday')
    plt.xticks(ticks=np.arange(7), labels=['Mon', 'Tue', 'Wed', 'Thu',
                                           'Fri', 'Sat', 'Sun'])
    plt.ylabel('Winning Ratio')
    plt.ylim(0.44, 0.53)
    plt.yticks(ticks=np.arange(0.44, 0.531, step=0.01))

    df_count_by_weekday.plot.scatter(x='weekday', y='result', legend=False)
    plt.title('# of Games by Day of the Week')
    plt.xlabel('Weekday')
    plt.xticks(ticks=np.arange(7), labels=['Mon', 'Tue', 'Wed',
                                           'Thu', 'Fri', 'Sat', 'Sun'])
    plt.ylabel('# of Games')
    plt.ylim(260, 380)
    plt.yticks(ticks=np.arange(260, 400, step=20))
    plt.show()


def castled_or_not(df):
    df_mean_by_castled = df.groupby('castled').mean()
    df_mean_by_castled.reset_index(inplace=True)
    df_count_by_castled = df.groupby('castled').count()
    df_count_by_castled.reset_index(inplace=True)

    df_mean_by_castled.plot.scatter(x='castled', y='result', legend=False)
    plt.title('Result by Castling')
    plt.xlabel('Castled')
    plt.xlim(-1.02, 1.02)
    plt.xticks(ticks=[-1, 0, 1], labels=['DNC', 'Queen-side', 'King-side'])
    plt.ylabel('Winning Ratio')
    plt.ylim(0.46, 0.515)
    plt.yticks(ticks=np.arange(0.46, 0.516, step=0.005))

    df_count_by_castled.plot.scatter(x='castled', y='result', legend=False)
    plt.title('# of Games for Castling')
    plt.xlabel('Castled')
    plt.xlim(-1.02, 1.02)
    plt.xticks(ticks=[-1, 0, 1], labels=['DNC', 'Queen-side', 'King-side'])
    plt.ylabel('# of Games')
    plt.ylim(200, 1600)
    plt.yticks(ticks=np.arange(200, 1700, step=200))

    plt.show()


def opp_castled_or_not(df):
    df_mean_by_opp_castled = df.groupby('opp_castled').mean()
    df_mean_by_opp_castled.reset_index(inplace=True)
    df_count_by_opp_castled = df.groupby('opp_castled').count()
    df_count_by_opp_castled.reset_index(inplace=True)

    df_mean_by_opp_castled.plot.scatter(x='opp_castled', y='result',
                                        legend=False)
    plt.title('Result by Opposition Castling')
    plt.xlabel('Opposition Castled')
    plt.xlim(-1.02, 1.02)
    plt.xticks(ticks=[-1, 0, 1], labels=['DNC', 'Queen-side', 'King-side'])
    plt.ylabel('Winning Ratio')
    plt.ylim(0.44, 0.58)
    plt.yticks(ticks=np.arange(0.44, 0.581, step=0.01))

    df_count_by_opp_castled.plot.scatter(x='opp_castled', y='result',
                                         legend=False)
    plt.title('# of Games for Opposition Castling')
    plt.xlabel('Opposition Castled')
    plt.xlim(-1.02, 1.02)
    plt.xticks(ticks=[-1, 0, 1], labels=['DNC', 'Queen-side', 'King-side'])
    plt.ylabel('# of Games')
    plt.ylim(300, 1200)
    plt.yticks(ticks=np.arange(300, 1300, step=100))
    plt.show()


def game_time(df):
    df_mean_by_game_time = df.groupby('game_time').mean()
    df_mean_by_game_time.reset_index(inplace=True)
    df_count_by_game_time = df.groupby('game_time').count()
    df_count_by_game_time.reset_index(inplace=True)

    df_mean_by_game_time.plot.scatter(x='game_time', y='result', legend=False)
    plt.title('Result by Game Time')
    plt.ylabel('Winning Ratio')
    plt.ylim(0.41, 0.53)
    plt.yticks(ticks=np.arange(0.41, 0.531, step=0.01))
    plt.xlabel('Game Time (Seconds)')
    plt.xlim(170, 610)
    plt.xticks(ticks=[180, 300, 600], labels=[180, 300, 600])

    df_count_by_game_time.plot.scatter(x='game_time', y='result', legend=False)

    plt.title('# of Games for Each Game Time')
    plt.xlabel('Game Time (Seconds)')
    plt.xlim(170, 610)
    plt.xticks(ticks=[180, 300, 600], labels=[180, 300, 600])
    plt.ylabel('# of Games')
    plt.ylim(0, 1700)
    plt.yticks(ticks=np.arange(0, 1801, step=200))
    plt.show()


def result_by_color(df):
    df_mean_by_color = df.groupby('color').mean()
    df_mean_by_color.reset_index(inplace=True)
    df_mean_by_color.plot.scatter(x='color', y='result', legend=False)
    plt.title('Result by Color')
    plt.xlabel('Color')
    plt.xlim(-.03, 1.01)
    plt.xticks(ticks=[0, 1], labels=['Black', 'White'])
    plt.ylabel('Winning Ratio')
    plt.ylim(0.465, 0.53)
    plt.yticks(ticks=np.arange(0.465, 0.531, step=0.005))
    plt.show()


def result_elo(df):
    df_mean_by_elo = df.groupby('bin_elo').mean()
    df_mean_by_elo.reset_index(inplace=True)
    df_count_by_elo = df.groupby('bin_elo').count()
    df_count_by_elo.reset_index(inplace=True)

    df_mean_by_elo.plot.scatter(x='bin_elo', y='result', legend=False)
    plt.title('Result by ELO')
    plt.xlabel('ELO')
    plt.xlim(660, 1050)
    plt.xticks(ticks=np.arange(660, 1051, step=30))
    plt.ylabel('Winning Ratio')
    plt.ylim(0.25, 0.7)
    plt.yticks(ticks=np.arange(0.25, 0.71, step=0.05))

    df_count_by_elo.plot.scatter(x='bin_elo', y='result', legend=False)
    plt.title('# of Games by ELO')
    plt.xlabel('ELO')
    plt.xlim(660, 1050)
    plt.xticks(ticks=np.arange(660, 1051, step=30))
    plt.ylabel('# of Games')
    plt.ylim(-1, 150)
    plt.yticks(ticks=np.arange(0, 151, step=15))
    plt.show()


def opp_result_elo(df):
    df_mean_by_opp_elo = df.groupby('bin_opp_elo').mean()
    df_mean_by_opp_elo.reset_index(inplace=True)
    df_count_by_opp_elo = df.groupby('bin_opp_elo').count()
    df_count_by_opp_elo.reset_index(inplace=True)

    df_mean_by_opp_elo.plot.scatter(x='bin_opp_elo', y='result', legend=False)
    plt.title('Result by Opposition ELO')
    plt.xlabel('Opposition ELO')
    plt.xlim(610, 1160)
    plt.xticks(ticks=np.arange(610, 1161, step=50))
    plt.ylabel('Winning Ratio')
    plt.ylim(0.1, 1.01)
    plt.yticks(ticks=np.arange(0.1, 1.02, step=0.1))

    df_count_by_opp_elo.plot.scatter(x='bin_opp_elo', y='result', legend=False)
    plt.title('# of Games by Opposition ELO')
    plt.xlabel('Opposition ELO')
    plt.xlim(610, 1160)
    plt.xticks(ticks=np.arange(610, 1161, step=50))
    plt.ylabel('# of Games')
    plt.ylim(-1, 120)
    plt.yticks(ticks=np.arange(0, 121, step=20))
    plt.show()


def result_by_elo_diff(df):
    df_mean_by_diff = df.groupby('bin_diff').mean()
    df_mean_by_diff.reset_index(inplace=True)
    df_count_by_diff = df.groupby('bin_diff').count()
    df_count_by_diff.reset_index(inplace=True)

    df_mean_by_diff.plot.scatter(x='bin_diff', y='result', legend=False)
    plt.title('Result by Difference in ELO')
    plt.xlabel('Difference in ELO')
    plt.xlim((-105, 105))
    plt.xticks(np.arange(-105, 106, step=15))
    plt.ylabel('Winning Ratio')
    plt.ylim((0, 1))
    plt.yticks(np.arange(0, 1.01, step=0.1))

    df_count_by_diff.plot.scatter(x='bin_diff', y='result', legend=False)
    plt.title('# of Games by Difference in ELO')
    plt.xlabel('Difference in ELO')
    plt.xlim((-105, 105))
    plt.xticks(np.arange(-105, 106, step=15))
    plt.ylabel('# of Games')
    plt.ylim((10, 100))
    plt.yticks(np.arange(10, 101, step=10))
    plt.show()


def won_via(df):
    df_mean_by_won_by = df.groupby('won_by').mean()
    df_mean_by_won_by.reset_index(inplace=True)
    df_count_by_won_by = df.groupby('won_by').count()
    df_count_by_won_by.reset_index(inplace=True)

    df_mean_by_won_by.plot.scatter(x='won_by', y='result', legend=False)
    plt.title('Result by the Result Type')
    plt.xlabel('Result Type')
    plt.xticks(ticks=np.arange(9), labels=['Rule', 'Stlmte', 'Abndn', 'Rptn',
                                           'Agr', 'Matrl', 'Time', 'Rsgn',
                                           'Chmte'])
    plt.ylabel('Winning Ratio')
    plt.ylim(0.25, 0.81)
    plt.yticks(ticks=np.arange(0.25, 0.86, step=0.05))

    df_count_by_won_by.plot.scatter(x='won_by', y='result', legend=False)
    plt.title('# of Games by the Result Type')
    plt.xlabel('Result Type')
    plt.xticks(ticks=np.arange(9), labels=['Rule', 'Stlmte', 'Abndn', 'Rptn',
                                           'Agr', 'Matrl', 'Time', 'Rsgn',
                                           'Chmte'])
    plt.ylabel('# of Games')
    plt.ylim(-20, 1000)
    plt.yticks(ticks=np.arange(0, 1001, step=200))
    plt.show()


def number_of_moves(df):
    df_mean_by_num_moves = df.groupby('bin_num_moves').mean()
    df_mean_by_num_moves.reset_index(inplace=True)
    df_count_by_num_moves = df.groupby('bin_num_moves').count()
    df_count_by_num_moves.reset_index(inplace=True)

    df_mean_by_num_moves.plot.scatter(
        x='bin_num_moves', y='result', legend=False)
    plt.title('Result by Number of Moves')
    plt.xlabel('Number of Moves')
    plt.xlim(-1, 110)
    plt.xticks(ticks=np.arange(0, 111, step=10))
    plt.ylabel('Winning Ratio')
    plt.ylim((.35, .75))
    plt.yticks(ticks=np.arange(0.35, 0.76, step=0.05))

    df_count_by_num_moves.plot.scatter(
        x='bin_num_moves', y='result', legend=False)
    plt.title('# of Games by Number of Moves')
    plt.xlabel('Number of Moves')
    plt.xlim(-1, 110)
    plt.xticks(ticks=np.arange(0, 111, step=10))
    plt.ylabel('# of Games')
    plt.ylim(-5, 260)
    plt.yticks(ticks=np.arange(0, 261, step=20))
    plt.show()


def move_num_castled(df):
    df_mean_by_castled_on = df.groupby('castled_on').mean()
    df_mean_by_castled_on.reset_index(inplace=True)
    df_count_by_castled_on = df.groupby('castled_on').count()
    df_count_by_castled_on.reset_index(inplace=True)

    df_mean_by_castled_on.plot.scatter(x='castled_on', y='result',
                                       legend=False)
    plt.title('Result by the Move # of Castling (0 for not castling)')
    plt.xlabel('Move # of Castling')
    plt.xlim(-.5, 32)
    plt.xticks(ticks=np.arange(0, 33, step=4))
    plt.ylabel('Winning Ratio')
    plt.ylim((.35, 1.01))
    plt.yticks(ticks=np.arange(0.35, 1.01, step=0.05))

    df_count_by_castled_on.plot.scatter(x='castled_on', y='result',
                                        legend=False)
    plt.title('# of Games by the Move # of Castling (0 for not castling)')
    plt.xlabel('Move # of Castling')
    plt.xlim(-.5, 32)
    plt.xticks(ticks=np.arange(0, 33, step=4))
    plt.ylabel('# of Games')
    plt.ylim(-5, 205)
    plt.yticks(ticks=np.arange(0, 206, step=20))
    plt.show()


def opp_move_num_castled(df):
    df_mean_by_opp_castled_on = df.groupby('opp_castled_on').mean()
    df_mean_by_opp_castled_on.reset_index(inplace=True)
    df_count_by_opp_castled_on = df.groupby('opp_castled_on').count()
    df_count_by_opp_castled_on.reset_index(inplace=True)

    df_mean_by_opp_castled_on.plot.scatter(x='opp_castled_on', y='result',
                                           legend=False)
    plt.title('Result by the Move # of Opposition Castling')
    plt.xlabel('Move # of Opp Castling')
    plt.xlim(-.5, 32)
    plt.xticks(ticks=np.arange(0, 33, step=4))
    plt.ylabel('Winning Ratio')
    plt.ylim((.25, .75))
    plt.yticks(ticks=np.arange(0.25, 0.76, step=0.05))

    df_count_by_opp_castled_on.plot.scatter(x='opp_castled_on', y='result',
                                            legend=False)
    plt.title('# of Games by the Move # of Opposition Castling')
    plt.xlabel('Move # of Opp Castling')
    plt.xlim(0, 32)
    plt.xticks(ticks=np.arange(0, 33, step=4))
    plt.ylabel('# of Games')
    plt.ylim(-1, 135)
    plt.yticks(ticks=np.arange(0, 136, step=15))
    plt.show()

def 