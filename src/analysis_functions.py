def get_game_count_df(val_counts, tot_val_counts):

    main_dict = {}
    main_dict[val_counts.loc[i]] = tot_val_counts.loc[i]

    if len(main_dict) > 32:
        main_dict = dict(islice(main_dict.items(), 31))

    df = pd.DataFrame(data=main_dict, index=([1]))

    return df

def get_df(val_counts, tot_val_counts):
    result_dict = {}
    total_dict = {}

    for i in val_counts.index:
        result_dict[round(i, 2)] = round(
            ((val_counts.loc[i])*100/val_counts.sum()), 1)
        total_dict[round(i, 2)] = round(
            ((val_counts.loc[i])*100/tot_val_counts.loc[i]), 1)

    if len(result_dict) > 32:
        result_dict = dict(islice(result_dict.items(), 31))
        total_dict = dict(islice(total_dict.items(), 31))

    df_result = pd.DataFrame(data=result_dict, index=([1]))
    df_total = pd.DataFrame(data=total_dict, index=([2]))

    df = pd.concat([df_result, df_total])

    return df

def get_stats(df, wld_df, column):
    val_counts = wld_df[column].value_counts()
    tot_val_counts = df[column].value_counts()
    result_dict = {}
    combined_df = get_df(val_counts, tot_val_counts)
    return combined_df