#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:06:51 2022

@author: annika

we do some quaktiy checks

- if all counts sum up
- if there are nans
- outliers
- ...

"""

import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats
import networkx as nx
from datetime import date, datetime
from matplotlib import animation
import numpy as np
from datetime import date, datetime

import read_in_Oonidata

def prep_xaxis():
    # keep only part of the labels
    n = 20  # Keeps every nth label
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    
    # rotate ticks, better readible
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    
    # remove ticks (annoying, looks ugly)
    ax.tick_params(axis=u'both', which=u'both',length=0)
    
    ax.xaxis.label.set_size(fontsize=14)  # increase fontsize
    

# !!! snowflake and Tor

#%% Variables which could or should be changed by users

path_to_data_folder = '../../Daten/Ooni/'  # where the data is stored

# filenames without the country name
file_name_end_domain = '_by_domain.csv'  

all_file_names = os.listdir(path_to_data_folder)  # in case anyone is interested

# countries
given_countries = ['Iran', 'Russia', 'Ukraine', 'Germany', 'Belarus', 'China', 'USA', 'Hungary']

# interesting domains:
given_domains = ['Telegram', 'WhatsApp', 'Signal']  #search for these names, not case sensitive

# all counts
all_counts = ['anomaly_count', 'confirmed_count', 'measurement_count', 'failure_count', 'ok_count']

#%% Create output folders

folder_lineplots = 'output/Lineplots/'
if not os.path.exists(folder_lineplots):
    os.makedirs(folder_lineplots)

folder_boxplots = 'output/boxplots/'
if not os.path.exists(folder_boxplots):
    os.makedirs(folder_boxplots)
    
folder_histplots = 'output/histplots/'
if not os.path.exists(folder_histplots):
    os.makedirs(folder_histplots)    
    
folder_networks = 'output/networks/'
if not os.path.exists(folder_networks):
    os.makedirs(folder_networks)    
    
folder_gifs = 'output/gifs/'
if not os.path.exists(folder_gifs):
    os.makedirs(folder_gifs)  
    
folder_tables = 'output/tables/'
if not os.path.exists(folder_tables):
    os.makedirs(folder_tables)

different_counts = ['anomaly_count', 'confirmed_count', 'measurement_count', 'failure_count', 'ok_count']
# %% analyse data for given countries
large_df_original, large_df_binary = read_in_Oonidata.ooni_for_countries(given_countries)  # binary: Number of count for blocked or not always 1 not higher. 

# Ukraine war: February 20, 2014
date_format = '%Y-%m-%d'
date_start_ukraine_war = datetime.strptime('2022-02-20', date_format)
date_start_dataset = datetime.strptime(min(large_df_binary['measurement_start_day']), date_format)

delta = date_start_ukraine_war - date_start_dataset
index_date_start_ukraine_war = delta.days  # for our xaxis line

# Iranian protests: 16 September 2022
date_start_iranian_protests = datetime.strptime('2022-09-16', date_format)
date_start_dataset = datetime.strptime(min(large_df_binary['measurement_start_day']), date_format)

delta = date_start_iranian_protests - date_start_dataset
index_date_start_iranian_protests = delta.days  # for our xaxis line

def ukraine_war_line(y):
    # line for when war started
    plt.axvline(x=index_date_start_ukraine_war, color='black', linewidth=3, linestyle='-.')
    plt.text(x=index_date_start_ukraine_war+ 3, y=y, s='Ukraine war', fontsize=14, color='black')
    
def iranian_protest_line(y):
    # line for when Iranian protests started
    plt.axvline(x=index_date_start_iranian_protests, color='black', linewidth=3, linestyle='-.')
    plt.text(x=index_date_start_iranian_protests+ 3, y=y, s='Iranian protests', fontsize=14, color='black')

# check if there are any nan entries: 
number_nans = large_df_binary.isna().sum()  # 6 nan domains
number_nans = large_df_original.isna().sum()  # 6 nan domains

# check if there are negative or non-integers:
dict_min_number = {}
dict_type = {}
for count in different_counts:
    dict_min_number[count] = min(large_df_original[count])  # everywhere 0, except for measurement count
    dict_type[count] = np.array_equal(large_df_original[count], large_df_original[count].astype(int))  # every where integers

# check if measurement counts is always sum of other counts
# min(large_df_original['diff_measurement_vs_rest'])  # all zero
# max(large_df_original['diff_measurement_vs_rest'])  # all zero
# large_df_original.columns


# check differences between measurement counts and others
large_df_original_grouped_dates = large_df_original.groupby(by=['measurement_start_day', 'Country'], axis=0).sum()
large_df_original_grouped_dates.reset_index(inplace=True)

large_df_original_grouped_dates_binary = large_df_binary.groupby(by=['measurement_start_day', 'Country'], axis=0).sum()
large_df_original_grouped_dates_binary.reset_index(inplace=True)

for count in different_counts:
    if not count == 'measurement_count':
        i,j=0,0
        PLOTS_PER_ROW = 4
        large_df_original_grouped_dates[count+'_diff'] = large_df_original_grouped_dates['measurement_count'] - large_df_original_grouped_dates[count]
        # as a figure
        fig, axs = plt.subplots(2,4, figsize=(16,6))
        for country in list(set(large_df_original_grouped_dates['Country'])):
            df_to_use = large_df_original_grouped_dates[large_df_original_grouped_dates['Country']==country]
            ymax = max(df_to_use['measurement_count'])
            x = df_to_use['measurement_start_day']
            y1 = df_to_use['measurement_count']
            y2 = df_to_use[count]
            axs[i][j].plot(x, y1, 'g:')
            axs[i][j].plot(x, y2, 'b:')
            axs[i][j].fill_between(x, y1, y2, color='grey', alpha=0.3)
            axs[i][j].set_xticks([])
            axs[i][j].set_title(country)
            
            ukraine_war_line(ymax)
            iranian_protest_line(ymax)
    
            j+=1
            if j%PLOTS_PER_ROW==0:
                i+=1
                j=0
        fig.tight_layout()
        plt.savefig(folder_lineplots + count + '_diff_measurements.png')  # save figure
        plt.close()
        
        

# %% stackplot
columns_to_use = ['ok_count','failure_count','anomaly_count','confirmed_count']  # order of the counts in the plot
color_map = ["green", "blue", "orange", "red"]  # color for the different counts
# generate for each country one figure
for country in list(set(large_df_original_grouped_dates['Country'])):
    # absolute numbers
    df_to_use = large_df_original_grouped_dates[large_df_original_grouped_dates['Country']==country]  # df for the country
    fig, ax = plt.subplots(figsize=(16,6))  # figure size
    plt.stackplot(df_to_use['measurement_start_day'].values,df_to_use[columns_to_use].T, labels=columns_to_use, colors = color_map)  # plot
    ymax=np.max(df_to_use[columns_to_use].values)
    prep_xaxis()  # remove part of the xlabels
    ukraine_war_line(ymax)  # draw line when war startetd
    iranian_protest_line(ymax)  # draw line when protest started
    plt.title(country)  # title
    plt.legend(loc='upper left')  # where the legend should be at
    fig.tight_layout()  # make figure such that all lables etc can be seen
    plt.savefig(folder_histplots + country + '_diff_measurements_stacked.png')  # save figure
    plt.close()
    
    # in percentage:
    df_to_use.loc[:,columns_to_use] = df_to_use.loc[:,columns_to_use].div(df_to_use[columns_to_use].sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(16,6))
    plt.stackplot(df_to_use['measurement_start_day'].values,df_to_use[columns_to_use].T, labels=columns_to_use, colors = color_map)
    ymax =1
    prep_xaxis()
    ukraine_war_line(ymax)
    iranian_protest_line(ymax)
    plt.title(country)
    plt.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(folder_histplots + country + '_diff_measurements_stacked_percentage.png')  # save figure
    plt.close()
    

# one figure, sames as above but all countries in one figure, each country one frame
i,j=0,0
PLOTS_PER_ROW = 4    
fig, axs = plt.subplots(2,4, figsize=(16,6))    
for country in list(set(large_df_original_grouped_dates['Country'])):
    df_to_use = large_df_original_grouped_dates[large_df_original_grouped_dates['Country']==country]  
    axs[i][j].stackplot(df_to_use['measurement_start_day'].values,df_to_use[columns_to_use].T, labels=columns_to_use, colors = color_map)
    axs[i][j].set_title(country)
    if i+j==0:
        axs[i][j].legend(loc='upper left')
    axs[i][j].set_xticks([])
    j+=1
    if j%PLOTS_PER_ROW==0:
        i+=1
        j=0
fig.tight_layout()
plt.savefig(folder_histplots + 'AllCountries_diff_measurements_stacked.png')  # save figure
plt.close()

# percentage instead of absolute counts
i,j=0,0
PLOTS_PER_ROW = 4    
fig, axs = plt.subplots(2,4, figsize=(16,6))    
for country in list(set(large_df_original_grouped_dates['Country'])):
    # in percentage:
    df_to_use = large_df_original_grouped_dates[large_df_original_grouped_dates['Country']==country]  
    df_to_use.loc[:,columns_to_use] = df_to_use.loc[:,columns_to_use].div(df_to_use[columns_to_use].sum(axis=1), axis=0)
    
    axs[i][j].stackplot(df_to_use['measurement_start_day'].values,df_to_use[columns_to_use].T, labels=columns_to_use, colors = color_map)
    axs[i][j].set_title(country)
    if i+j==0:
        axs[i][j].legend(loc='lower left')
    axs[i][j].set_xticks([])
    j+=1
    if j%PLOTS_PER_ROW==0:
        i+=1
        j=0
fig.tight_layout()
plt.savefig(folder_histplots + 'AllCountries_diff_measurements_stacked_percentage.png')  # save figure
plt.close()

# one figure, binary counts (thus not all counts, only one per day per website)
i,j=0,0
PLOTS_PER_ROW = 4    
fig, axs = plt.subplots(2,4, figsize=(16,6))    
for country in list(set(large_df_original_grouped_dates_binary['Country'])):
    df_to_use = large_df_original_grouped_dates_binary[large_df_original_grouped_dates_binary['Country']==country]  
    axs[i][j].stackplot(df_to_use['measurement_start_day'].values,df_to_use[columns_to_use].T, labels=columns_to_use, colors = color_map)
    axs[i][j].set_title(country)
    if i+j==0:
        axs[i][j].legend(loc='upper left')
    axs[i][j].set_xticks([])
    j+=1
    if j%PLOTS_PER_ROW==0:
        i+=1
        j=0
fig.tight_layout()
plt.savefig(folder_histplots + 'AllCountries_diff_measurements_stacked_binary.png')  # save figure
plt.close()

i,j=0,0
PLOTS_PER_ROW = 4    
fig, axs = plt.subplots(2,4, figsize=(16,6))    
for country in list(set(large_df_original_grouped_dates_binary['Country'])):
    # in percentage:
    df_to_use = large_df_original_grouped_dates_binary[large_df_original_grouped_dates_binary['Country']==country]  
    df_to_use.loc[:,columns_to_use] = df_to_use.loc[:,columns_to_use].div(df_to_use[columns_to_use].sum(axis=1), axis=0)
    
    axs[i][j].stackplot(df_to_use['measurement_start_day'].values,df_to_use[columns_to_use].T, labels=columns_to_use, colors = color_map)
    axs[i][j].set_title(country)
    if i+j==0:
        axs[i][j].legend(loc='lower left')
    axs[i][j].set_xticks([])
    j+=1
    if j%PLOTS_PER_ROW==0:
        i+=1
        j=0
fig.tight_layout()
plt.savefig(folder_histplots + 'AllCountries_diff_measurements_stacked_percentage_binary.png')  # save figure
plt.close()    



# %% messengers:
messenger_df = read_in_Oonidata.ooni_messengers()
# messenger_df.head()

# check differences between measurement counts and others
messenger_df_grouped_dates = messenger_df.groupby(by=['measurement_start_day', 'Country'], axis=0).sum()
messenger_df_grouped_dates.reset_index(inplace=True)

for country in given_countries:
    # absolute numbers
    df_to_use = messenger_df_grouped_dates[messenger_df_grouped_dates['Country']==country]
    fig, ax = plt.subplots(figsize=(16,6))
    plt.stackplot(df_to_use['measurement_start_day'].values,df_to_use[columns_to_use].T, labels=columns_to_use, colors = color_map)
    ymax=np.max(df_to_use[columns_to_use].values)
    prep_xaxis()
    ukraine_war_line(ymax)
    iranian_protest_line(ymax)
    plt.title(country)
    plt.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(folder_histplots + country + '_messengers_diff_measurements_stacked.png')  # save figure
    plt.close()
    
    # in percentage:
    df_to_use.loc[:,columns_to_use] = df_to_use.loc[:,columns_to_use].div(df_to_use[columns_to_use].sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(16,6))
    plt.stackplot(df_to_use['measurement_start_day'].values,df_to_use[columns_to_use].T, labels=columns_to_use, colors = color_map)
    ymax =1
    prep_xaxis()
    ukraine_war_line(ymax)
    iranian_protest_line(ymax)
    plt.title(country)
    plt.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(folder_histplots + country + '_messengers_diff_measurements_stacked_percentage.png')  # save figure
    plt.close()
    

# one figure
i,j=0,0
PLOTS_PER_ROW = 4    
fig, axs = plt.subplots(2,4, figsize=(16,6))    
for country in given_countries:
    df_to_use = messenger_df_grouped_dates[messenger_df_grouped_dates['Country']==country]  
    axs[i][j].stackplot(df_to_use['measurement_start_day'].values,df_to_use[columns_to_use].T, labels=columns_to_use, colors = color_map)
    axs[i][j].set_title(country)
    if i+j==0:
        axs[i][j].legend(loc='upper left')
    axs[i][j].set_xticks([])
    j+=1
    if j%PLOTS_PER_ROW==0:
        i+=1
        j=0
fig.tight_layout()
plt.savefig(folder_histplots + 'AllCountries_messengers_diff_measurements_stacked.png')  # save figure
plt.close()

i,j=0,0
PLOTS_PER_ROW = 4    
fig, axs = plt.subplots(2,4, figsize=(16,6))    
for country in given_countries:
    # in percentage:
    df_to_use = messenger_df_grouped_dates[messenger_df_grouped_dates['Country']==country]  
    df_to_use.loc[:,columns_to_use] = df_to_use.loc[:,columns_to_use].div(df_to_use[columns_to_use].sum(axis=1), axis=0)
    
    axs[i][j].stackplot(df_to_use['measurement_start_day'].values,df_to_use[columns_to_use].T, labels=columns_to_use, colors = color_map)
    axs[i][j].set_title(country)
    if i+j==0:
        axs[i][j].legend(loc='upper left')
    axs[i][j].set_xticks([])
    j+=1
    if j%PLOTS_PER_ROW==0:
        i+=1
        j=0
fig.tight_layout()
plt.savefig(folder_histplots + 'AllCountries_messengers_diff_measurements_stacked_percentage.png')  # save figure
plt.close()

# check how many countries there are and if there are any double entries:
df_blocked_websites, df_blocked_websites_total_counts = read_in_Oonidata.ooni_blcoked_websites()

# %% domain strings check
all_domains = list(set(large_df_binary['domain']))  # 15591 domains

# remove top level domians:
all_domains_no_top_level = list(set([str(d).rsplit('.',1)[0] for d in all_domains]))  # 14990

# remove www:
all_domains_no_top_level_no_www = []
for d in all_domains_no_top_level:
    if 'www.' == d[:4]:
        all_domains_no_top_level_no_www.append(d[4:])
    else:
        all_domains_no_top_level_no_www.append(d)
        
all_domains_no_top_level_no_www = list(set(all_domains_no_top_level_no_www)) #  14525

large_df_binary_grouped =  large_df_binary.groupby(['domain', 'Country']).sum().reset_index()   #  ['b'].sum()



# %% check if there are outliers:

# boxplots for number of counts: 
i,j=0,0
PLOTS_PER_ROW = 2
fig, axs = plt.subplots(3,2, figsize=(16, 6))
for count in different_counts:
    sns.boxplot(data=large_df_original, x=count, y='Country', ax=axs[i][j])
    axs[i][j].set(xscale="log")
    j+=1
    if j%PLOTS_PER_ROW==0:
        i+=1
        j=0
fig.tight_layout()
plt.savefig(folder_boxplots + '_distbribution_log.png')  # save figure
plt.close()


fig, ax = plt.subplots(figsize=(16, 6))  # set the figure height and width
for count in ['anomaly_count', 'confirmed_count', 'measurement_count', 'failure_count', 'ok_count']:
    sns.boxplot(data=large_df_original, x=count, y='Country')
    plt.title('Distribution of ' + count, fontsize=25)
    plt.tight_layout()  # make the figure, such that you can see everything
    plt.savefig(folder_boxplots + count + '_distbribution.png')  # save figure
    plt.close()


# check worldwide how many websites are blocked per day, and which countries and which days block more:
blocked_websites_mean_days = df_blocked_websites.groupby(by=['measurement_start_day'], axis=0).mean()
blocked_websites_mean_days.reset_index(inplace=True)

blocked_websites_mean_countries = df_blocked_websites.groupby(by=['Country'], axis=0).mean()
blocked_websites_mean_countries.reset_index(inplace=True)

dict_above_mean_countries = {}
count = 'confirmed_count'
for count in all_counts:
    current_mean = blocked_websites_mean_countries[count].mean()
    dict_above_mean_countries[count] = list(blocked_websites_mean_countries[blocked_websites_mean_countries[count]>current_mean]['Country'])
    
for_table = blocked_websites_mean_countries[blocked_websites_mean_countries['Country'].isin(dict_above_mean_countries['confirmed_count'])]
for_table[['Country', 'confirmed_count']].to_csv(folder_tables + 'above_mean_blocked.csv')

# not so interesting
dict_above_mean_days = {}
count = 'confirmed_count'
for count in all_counts:
    current_mean = blocked_websites_mean_days[count].mean()
    dict_above_mean_days[count] = list(blocked_websites_mean_days[blocked_websites_mean_days[count]>current_mean]['measurement_start_day'])
    


