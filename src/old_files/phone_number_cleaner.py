import pandas as pd
import numpy as np
import re
from random import randint


def change_col(df, col_name, intornot):
    x = df[col_name].tolist()
    for i, adr in enumerate(x):
        if str(adr) != 'nan':
            if type(adr) != int or type(adr) != float:
                y = adr.split(' ')
                if len(y) > 1:
                    df.loc[i, col_name] = y[1]
                else:
                    df.loc[i, col_name] = y[0]
            else:
                df.loc[i, col_name] = adr
    if intornot == 'y':
        df.loc[:, col_name] = df[col_name].apply(
            lambda x: ''.join([num for num in re.findall(r'[\.0-9]', str(x))]))
    return df


def change_column_names(df):
    cols_chng = ['Land Use - CoreLogic (TAX|MLS)', 'Last Mkt Rec Date']
    int_chng = ['Bldg Sq Ft (TAX|MLS)', 'Bedrooms (TAX|MLS)',
                'Last Mkt Sale Price', 'Last Mkt 2nd Mtg Amt/Type',
                'Full Baths (TAX|MLS)', 'Last Mkt 1st Mtg Amt/Type',
                'Half Baths (TAX|MLS)']

    for col_name in cols_chng:
        df = change_col(df, col_name, 'n')
    for col_name in int_chng:
        df = change_col(df, col_name, 'y')
    for col in df.columns:
        df.rename(columns={col: col.replace(' (TAX|MLS)', '')}, inplace=True)
        if 'Unnamed' in col or 'unnamed' in col:
            df.drop(columns=[col], inplace=True)

    df.rename(columns={'APN': 'Tax ID', 'Land Use - CoreLogic': 'Class',
                       'Property Address': 'Address',
                       'Property Unit #': 'Unit',
                       'Property City': 'City',
                       'Property Zip': 'Zip',
                       'Property State': 'State',
                       'Bldg Sq Ft': 'Square Feet',
                       'Bedrooms': 'Bedrooms',
                       'Lot Sq Ft': 'Lot SqFt',
                       'RealAVM': 'List Price',
                       'Effective Yr Built': 'Year Built'}, inplace=True)
    return df


def create_new_drop_old(df):
    for i in df.index:
        if str(df['Half Baths'][i]) != '':
            df.loc[i, 'Bathrooms'] = float(df['Full Baths'][i]) + float(
                df['Half Baths'][i]) / 2
        else:
            df.loc[i, 'Bathrooms'] = float(df['Full Baths'][i])

        if type(df['Tax Billing Address'][i]) != float:
            df.loc[i, 'Tax Billing Address'] = df[
                'Tax Billing Address'][i].title()

    df.loc[:, 'Occupied By'] = np.where(
        df['Tax Billing Address'] == df['Address'], 'Owner', 'Tenant')

    df.drop(columns=['Full Baths', 'Half Baths'], inplace=True)

    return df


def owner_name_fix(df):
    for i in df.index:
        if 'Trust' in df['Owner Name'][i]:
            df.loc[i, 'Owners Names'] = df['Owner Name'][i]
        else:
            y = []
            z = df['Owner Name'][i].split(' ')
            if len(z) > 2:
                z = [num for num in z if len(num) > 1 and num != '(Te)']

            if type(df['Owner Name 2'][i]) != float:
                y = df['Owner Name 2'][i].split(' ')
                if len(y) > 2:
                    y = [num for num in y if len(num) > 1 and num != '(Te)']
            else:
                ownof = ' '.join(z[1:])
                df.loc[i, 'Owners Names'] = f'{ownof} {z[0]}'

            if len(y) > 1:
                ownof = ' '.join(z[1:])
                owntf = ' '.join(y[1:])
                if y[0] == z[0]:
                    fn = f'{ownof} & {owntf} {y[0]}'
                    df.loc[i, 'Owners Names'] = fn
                else:
                    fn = f'{ownof} {z[0]} & {owntf} {y[0]}'
                    df.loc[i, 'Owners Names'] = fn
    df.drop(columns=['Owner Name', 'Owner Name 2'], inplace=True)
    return df


def fix_val_types(df):
    for i in df.index:
        df.loc[:, 'Square Feet'] = df['Square Feet'].astype(float)
        df.loc[:, 'Bedrooms'] = df['Bedrooms'].astype(float)
        if type(df['List Price'][i]) != float:
            lp = ''.join(df['List Price'][i].strip('$').split(','))
            df.loc[i, 'List Price'] = float(lp)
        else:
            z = randint(700, 1000)
            df.loc[i, 'List Price'] = df['Square Feet'][i] * z
    return df


def all_realist(df_realist):
    df_realist = change_column_names(df_realist)
    df_realist = create_new_drop_old(df_realist)
    df_realist = owner_name_fix(df_realist)
    df_realist = fix_val_types(df_realist)
    return df_realist


def create_redx_df(df_realist):
    adf = pd.read_csv('./agent_names_numbers.csv')
    df = pd.DataFrame()
    imlsid = 40767000
    lndf = len(df_realist)
    er = imlsid + lndf
    df.loc[:, 'Tax ID'] = df_realist['Tax ID']
    df.loc[:, 'MLS Listing ID'] = list(range(imlsid, er))
    df.loc[:, 'Days on Market'] = 1
    df.loc[:, 'Status'] = 'Expired'
    df.loc[:, 'Class'] = df_realist['Class']
    df.loc[:, 'Address'] = df_realist['Address']
    df.loc[:, 'Unit'] = df_realist['Unit']
    df.loc[:, 'City'] = df_realist['City']
    df.loc[:, 'Area'] = df['City'].apply(lambda x: x.upper())
    df.loc[:, 'Zip'] = df_realist['Zip']
    df.loc[:, 'County'] = df_realist['County']
    df.loc[:, 'State'] = df_realist['State']
    df.loc[:, 'List Price'] = df_realist['List Price']
    df.loc[:, 'List $/SqFt'] = (df_realist['List Price'] /
                                df_realist['Square Feet']).round(2)
    df.loc[:, 'Sold Price'] = ''
    df.loc[:, 'Square Feet'] = df_realist['Square Feet']
    df.loc[:, 'Bedrooms'] = df_realist['Bedrooms']
    df.loc[:, 'Bathrooms'] = df_realist['Bathrooms']
    df.loc[:, 'Lot SqFt'] = df_realist['Lot SqFt']
    df.loc[:, 'Year Built'] = df_realist['Year Built']
    df.loc[:, 'Listing Agent Name'] = adf.iloc[:lndf, 0]
    df.loc[:, 'Listing Agent Phone Number'] = adf.iloc[:lndf, 1]
    df.loc[:, 'Occupied By'] = df_realist['Occupied By']
    df.loc[:, 'Occupant Name'] = np.where(
        df_realist['Occupied By'] == 'Owner', df_realist['Owners Names'], '')
    df.loc[:, 'Occupant Phone'] = ''
    df.loc[:, 'List Date'] = '12/31/2016'
    df.loc[:, 'Contract Date'] = ''
    df.loc[:, 'General Date'] = '01/01/2017'
    df.loc[:, 'Status Date'] = '01/01/2017'
    df.loc[:, 'Listing Type'] = 'Excl Agency'
    df.loc[:, 'Update Date'] = '01/01/2017'
    df.loc[:, 'Off Market Date'] = '01/01/2017'
    df.loc[:, 'Closing Date'] = ''
    df = df[df['Class'] == 'SFR']
    df.fillna('', inplace=True)
    return df
