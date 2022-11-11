#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:45:19 2022

@author: annika

Read in and clean ooni data

"""

from matplotlib import pyplot as plt
import plotly.express as px
import pandas as pd
import os
import geojson
import json


#%% Some variables

path_to_data_folder = '../../Daten/Ooni/'  # where the data is stored

# filenames without the country name
file_name_end_domain = '_by_domain.csv'  

all_file_names = os.listdir(path_to_data_folder)  # in case anyone is interested

path_iso_countries_three_digits = '../../Daten/all_iso_countries.xlsx'
iso_countries_three_digits = pd.read_excel(path_iso_countries_three_digits)  # country-codes to country names
iso_countries_three_digits = iso_countries_three_digits[['ISO', 'ISO3', 'Country', 'Continent']]
iso_countries_three_digits.rename({'ISO':'probe_cc'}, inplace=True, axis=1)  # names as in the other dataframes
iso_countries_three_digits.set_index(keys='probe_cc', inplace=True)  # set index as iso country code

# countries
given_countries = ['Iran', 'Russia', 'Ukraine', 'Germany', 'Belarus', 'China', 'USA', 'Hungary']

# interesting domains:
given_domains = ['Telegram', 'WhatsApp', 'Signal']  #search for these names, not case sensitive

#%% Create large dataframe containing all detailed information about the chosen countries
def ooni_for_countries(given_countries=given_countries):
    dict_country_df = {}  # keys are the country names, values are the dfs of the country's info
    # read in the data for the different countries
    large_df = pd.DataFrame()
    
    for country in given_countries:
    
        current_df = pd.read_csv(path_to_data_folder + country + file_name_end_domain)
        current_df['Country'] = country
        
        large_df = pd.concat([large_df, current_df])
        
        dict_country_df[country] = current_df    

    # %% domain strings check
    all_domains = list(large_df['domain'])  # 9095 domains unique

    # remove top level domians:
    all_domains_no_top_level = [str(d).rsplit('.',1)[0] for d in all_domains]  # 8733

    # remove www:
    all_domains_no_top_level_no_www = []
    for d in all_domains_no_top_level:
        if 'www.' == d[:4]:
            all_domains_no_top_level_no_www.append(d[4:])
        else:
            all_domains_no_top_level_no_www.append(d)
    large_df['domain'] = all_domains_no_top_level_no_www
    large_df.columns
    large_df = large_df.groupby(['domain','Country','measurement_start_day']).sum()
    large_df.reset_index(inplace=True)
    
    # create df containin if a website is blocked or not. Thus not the counts per day, just but 0 and 1
    large_df_original = large_df.copy()
    
    large_df['anomaly_count'] = large_df['anomaly_count']>large_df['ok_count']
    large_df['confirmed_count'] = large_df['confirmed_count']>large_df['ok_count'].astype(int)
    large_df['anomaly_count'] = large_df['anomaly_count'].astype(int)
    large_df['confirmed_count'] = large_df['confirmed_count'].astype(int)
    
    large_df['confANDano'] = large_df['anomaly_count'] + large_df['confirmed_count']
    
    return large_df_original, large_df


#%% read in messenger data
# data for different messengers. From all countries, but with iso name from countries
def ooni_messengers(messenger_names = ['Facebook',  'Whatsapp', 'Signal', 'Telegram']):
   

    path_iso_countries = '../../Daten/all_countries_iso_code.csv'
    iso_countries = pd.read_csv(path_iso_countries)  # country-codes to country names

    messenger_df = pd.DataFrame()

    #  create one large df for the messenger data:
    for messenger in messenger_names:
        current_df = pd.read_csv(path_to_data_folder + 'all_countries_' + messenger + '.csv')
        current_df['messenger'] = messenger 
        messenger_df = pd.concat([messenger_df, current_df])

    # messenger_df.head()
    messenger_df.set_index(keys='probe_cc', inplace=True)  # set index as iso country code

    # merge country names to the messenger dataset:
    iso_countries.rename({'Code':'probe_cc', 'Name':'Country'}, inplace=True, axis=1)  # names as in the other dataframes
    iso_countries.set_index(keys='probe_cc', inplace=True)  # set index as iso country code
    iso_countries.replace({'Iran, Islamic Republic of':'Iran', 'Russian Federation':'Russia', 'United States':'USA'}, inplace=True)  # we want Iran, otherwise wont match the other data

    messenger_df = messenger_df.merge(iso_countries, how="left", on="probe_cc")
    messenger_df.reset_index(inplace=True)
    
    
    return messenger_df
    
def ooni_blcoked_websites():

    # read in data
    df_blocked_websites = pd.read_csv('../../Daten/Ooni/all_countries_webpages.csv')
    
    df_blocked_websites_total_counts = df_blocked_websites.groupby(['probe_cc']).sum().reset_index()
    df_blocked_websites_total_counts = df_blocked_websites_total_counts.merge(iso_countries_three_digits, how="left", on="probe_cc")
    df_blocked_websites_total_counts.reset_index(inplace=True)
    
    df_blocked_websites = df_blocked_websites.merge(iso_countries_three_digits, how="left", on="probe_cc")
    df_blocked_websites_total_counts.rename({'ISO3':'ISO_A3'}, inplace=True,axis='columns')
    
    return df_blocked_websites, df_blocked_websites_total_counts
    
    
    