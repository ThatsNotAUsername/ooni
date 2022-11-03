#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:55:30 2022

@author: annika

Playaround with Ooni data

Ooni data counts each day which websites or messengers are blocked or show anomalies. 

I use data from 01-01-2022 until 25-10-2022 for some countries I decided could be interesting. 
I then checked how many websites are blocked, which, and from which day on. 

I used different techniques to visualize the data. 

---------------- About OONI --------------------------

users download the app. Using this, ooni collects if websites are blocked or show anomalies. 
Thus: we have several counts per day, depending on the number of users in the country. 

- anomaly_count: An “anomalous” measurement is a testing result which is flagged 
    -- because it presents signs of potential network interference (not necessarily blocked though)
- confirmed_count: confirmed that website is blocked
- failure_count: ?
- ok_count: Normal, not blocked
- measurement_count: Total counts done that day

Note:
I decided that a website is blocked on a specific day, if the confirmed count is
higher than the ok_count. 

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

# !!! snowflake and Tor

#%% Variables which could or should be changed by users

path_to_data_folder = '../../Daten/Ooni/'  # where the data is stored

# filenames without the country name
file_name_end_domain = '_by_domain.csv'  

all_file_names = os.listdir(path_to_data_folder)  # in case anyone is interested

# countries
given_countries = ['Iran', 'Russia', 'Germany', 'Belarus', 'China', 'USA', 'Hungary', 'Ukraine']

# interesting domains:
given_domains = ['Telegram', 'WhatsApp', 'Signal']  #search for these names, not case sensitive


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

#%% Create large dataframe containing all necessary information

# read in the data for the different countries
large_df = pd.DataFrame()

for country in given_countries:

    current_df = pd.read_csv(path_to_data_folder + country + file_name_end_domain)
    current_df['Country'] = country
    
    large_df = pd.concat([large_df, current_df])

# create df containin if a website is blocked or not. Thus not the counts but 0 and 1
large_df_original = large_df.copy()

large_df['anomaly_count'] = large_df['anomaly_count']>large_df['ok_count']
large_df['confirmed_count'] = large_df['confirmed_count']>large_df['ok_count'].astype(int)
large_df['anomaly_count'] = large_df['anomaly_count'].astype(int)
large_df['confirmed_count'] = large_df['confirmed_count'].astype(int)

large_df['confANDano'] = large_df['anomaly_count'] + large_df['confirmed_count']



# %% create some grouped and sorted df

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#      highest number of blocked websites vs day
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

number_highest = 20

# count all, not based on days
large_df_total_counts_websites = large_df.groupby(['Country', 'domain']).sum().reset_index()  

# dictionary for each type of count    
dict_most_often_website_country = {}

# create dictionary containin for each type of count a dataframe conting highest number of blocked websites
for y in ['failure_count', 'measurement_count', 'confirmed_count', 'anomaly_count', 'ok_count', 'confANDano']:
    dict_most_often_website_country[y] = pd.DataFrame()
    for country in given_countries:
        sorted_df = large_df_total_counts_websites[large_df_total_counts_websites['Country']==country].sort_values(y, ascending=False)
        dict_most_often_website_country[y] = pd.concat([dict_most_often_website_country[y], sorted_df.iloc[0:number_highest]])


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#      number of blocked websites vs day vs country
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# count all, not based on domains, per day
large_df_total_counts = large_df.groupby(['measurement_start_day', 'Country']).sum().reset_index()   #  ['b'].sum()

# count all, not based on domains
large_df_counts_country = large_df.groupby(['domain', 'Country']).sum().reset_index()   #  ['b'].sum()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#      dates of important events
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Ukraine war: February 20, 2014
date_format = '%Y-%m-%d'
date_start_ukraine_war = datetime.strptime('2022-02-20', date_format)
date_start_dataset = datetime.strptime(min(large_df_total_counts['measurement_start_day']), date_format)

delta = date_start_ukraine_war - date_start_dataset
index_date_start_ukraine_war = delta.days  # for our xaxis line

# Iranian protests: 16 September 2022
date_start_iranian_protests = datetime.strptime('2022-09-16', date_format)
date_start_dataset = datetime.strptime(min(large_df_total_counts['measurement_start_day']), date_format)

delta = date_start_iranian_protests - date_start_dataset
index_date_start_iranian_protests = delta.days  # for our xaxis line

# !!! %% some statistics

# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #       which websites are blocked higher than everage in Germany
# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# avg_Germany = {}
# # country='Russia'
# # y='confirmed_count'
# for y in ['failure_count', 'measurement_count', 'confirmed_count', 'anomaly_count', 'ok_count']:
#     avg_Germany[y] = large_df_counts_country[(large_df_counts_country['Country']=="Germany")][[y, 'domain']].mean()
#     for country in given_countries:
#         df_country = large_df_counts_country[(large_df_counts_country['Country']==country)][[y, 'domain']]
   
        

# %% Define some functions we need almost always

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
    
def ukraine_war_line():
    # line for when war started
    plt.axvline(x=index_date_start_ukraine_war, color='red')
    plt.text(x=index_date_start_ukraine_war+ 3, y=max(large_df_total_counts[y]), s='Ukraine war', fontsize=12, color='red')
    
def iranian_protest_line():
    # line for when Iranian protests started
    plt.axvline(x=index_date_start_iranian_protests, color='green')
    plt.text(x=index_date_start_iranian_protests+ 3, y=max(large_df_total_counts[y]), s='Iranian protests', fontsize=12, color='green')
    
# %% make some line plots:
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#       number of blocked websites vs day
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

y='confirmed_count'
for y in ['failure_count', 'measurement_count', 'confirmed_count', 'anomaly_count', 'ok_count']:
    
    # set the figure height and width
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # plot with seaborn
    sns.lineplot(data=large_df_total_counts, x='measurement_start_day', y=y, hue='Country',marker='o', ax=ax, markersize=4, alpha=.7, palette='hls')
    
    prep_xaxis() # format the x-axis

    ukraine_war_line() # line for when war started
    iranian_protest_line() # line for when Iranian protests started
    
    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")  # move the legend
    
    plt.tight_layout()  # make the figure, such that you can see everything

    plt.savefig(folder_lineplots + y +'_websites.png')  # save figure
    plt.close()
    

# %% make some hist plots:
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#      highest number of blocked websites vs day
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        
# y = 'confirmed_count'
for y in ['failure_count', 'measurement_count', 'confirmed_count', 'anomaly_count', 'ok_count']:
    
    for country in given_countries:
        
        fig, ax = plt.subplots(figsize=(16, 6))  # set the figure height and width
        
        df_to_use = dict_most_often_website_country[y][dict_most_often_website_country[y]['Country']==country]  # use df given by country and count type
        
        max_count = df_to_use[y].max()
        
        sns.barplot(data=df_to_use, x=y,y='domain', hue='Country', alpha=.5)  # plot with seaborn
        
        # put labels into the bars
        labels = [item.get_text() for item in ax.get_yticklabels()]
        
        for label_x, label_text in enumerate(labels):
            ax.text(max_count/20, label_x,  label_text, va='center', fontsize=16)  # , rotation=90, ha='center',
        
        plt.yticks([])  # remove xticks (since they are already on the bars)
                
        ax.xaxis.label.set_size(fontsize=16)  # increase fontsize
        ax.yaxis.label.set_size(fontsize=16)  # increase fontsize

        plt.title('Most blocked websites in ' + country, fontsize=25)
        
        plt.tight_layout()  # make the figure, such that you can see everything
        
        ax.get_legend().remove()

        plt.savefig(folder_histplots + country + '_' +  y + '_MostBlcokedWebsites.png')  # save figure
        plt.close()


# confirmed and anomaly   : We plot the anomalies and the blocked together. 

for country in given_countries:

    fig, ax = plt.subplots(figsize=(16, 6))  # set the figure height and width
    
    max_count = df_to_use['confANDano'].max()
    
    # first plot the blocked websites
    df_to_use = dict_most_often_website_country['confirmed_count'][dict_most_often_website_country['confirmed_count']['Country']==country]   # use df given by country and count type
    sns.set_color_codes("muted")
    sns.barplot(data=df_to_use, x='confirmed_count',y='domain',alpha=.7, label='blocked', palette='crest')  # plot with seaborn
    
    # put labels into the bars
    labels = [item.get_text() for item in ax.get_yticklabels()]
    
    for label_x, label_text in enumerate(labels):
        ax.text(max_count/20, label_x,  label_text, va='center', fontsize=14)  # , rotation=90, ha='center',
    
    # plot anomalies st it looks as if they are next to the blocked. 
    # For this we plot the number blcoked AND anomalies together in pastel, behnid the already plotted blocked websites.

    df_to_use = dict_most_often_website_country['confANDano'][dict_most_often_website_country['confANDano']['Country']==country]  # use df given by country and count type
    
    sns.set_color_codes("pastel")
    sns.barplot(data=df_to_use, x='confANDano',y='domain', alpha=.5, label='anomaly', palette='crest')  # plot with seaborn
    
    # remove yticks (since they are already on the bars)
    plt.yticks([])
    
    ax.set_xlabel('count')
            
    ax.xaxis.label.set_size(fontsize=20)  # increase fontsize
    ax.yaxis.label.set_size(fontsize=20)
   
    plt.title('Most blocked websites in ' + country, fontsize=25)
    
    # make the figure, such that you can see everything
    plt.tight_layout()
    ax.legend(ncol=2, loc="lower right", frameon=True)
    
    plt.savefig(folder_histplots + country + '_confANDano_MostBlcokedWebsites.png')  # save figure
    plt.close()


# %% networks

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     Country and websites nodes, connected if blocked in country
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

G=nx.from_pandas_edgelist(large_df_counts_country, 'Country', 'domain', ['confirmed_count'])  # generate nw from dataframe
# len(G.nodes)

# remove edges which have no counts, thus not referring to blocked websites
no_edges = [(a,b) for a, b, attrs in G.edges(data=True) if attrs["confirmed_count"] < 1]
G.remove_edges_from(no_edges)

G.remove_nodes_from(list(nx.isolates(G)))  # remove isolates


degrees = [val for (node, val) in G.degree()]  # degrees of the nodes as list

# plot degree distribution
fig, ax = plt.subplots(figsize=(16, 6))

sns.countplot(degrees)
plt.xticks(rotation=45)
plt.title('Blocked in how many countries', fontsize=25)
plt.savefig(folder_networks + 'degreeDistribution_blockedWebsites_Countries.png')
plt.close()

# plot degree distribution, logscale
fig, ax = plt.subplots(figsize=(16, 6))

sns.countplot(degrees, palette='flare')
plt.xticks(rotation=45)
plt.yscale('log')
plt.title('Blocked in how many countries', fontsize=25)
plt.savefig(folder_networks + 'degreeDistribution_blockedWebsites_Countries_logscale.png')
plt.close()

# degrees as dictionary
degrees = {node:val for (node, val) in G.degree()}

# order by degree, highest first
degrees = dict(sorted(degrees.items(), key=lambda item: item[1], reverse=True))

nx.write_edgelist(G, folder_networks + 'network_blockedWebsites_country.txt')


# %% Messengers
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#      read in data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# data for different messengers. From all countries, but with iso name from countries
messenger_names = ['Facebook',  'Whatsapp', 'Signal', 'Telegram']

path_iso_countries = '../Daten/all_countries_iso_code.csv'
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

messenger_considered_countries = messenger_df[messenger_df['Country'].isin(given_countries)]  # keep those which we analyzed before


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#      plot messengers
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

y = 'anomaly_count'
messenger ='Signal'
for messenger in messenger_names:
    for y in ['failure_count', 'measurement_count', 'confirmed_count', 'anomaly_count', 'ok_count']:

        fig, ax = plt.subplots(figsize=(16, 6))  # set the figure height and width
        
        df_to_use = messenger_considered_countries[messenger_considered_countries['messenger']==messenger]  # use only data for given messenger
        
        sns.lineplot(data=df_to_use, x='measurement_start_day', y=y, hue='Country',marker='o', ax=ax, markersize=4, alpha=.7, palette='hls')  # plot with seaborn
        
        # format the x-axis
        prep_xaxis()
        
        # line for when war started
        ukraine_war_line()
        
        # line for when Iranian protests started
        iranian_protest_line()

        ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")  # move the legend
        
        plt.title('Blocked messengers', fontsize=25)  # title
        
        plt.tight_layout()  # make the figure, such that you can see everything
        
        # save figure
        plt.savefig(folder_lineplots + y + '_' + messenger + '.png')
        plt.close()
        
y = 'anomaly_count'
country='Iran'
for country in given_countries:
    for y in ['anomaly_count']:#['failure_count', 'measurement_count', 'confirmed_count', 'anomaly_count', 'ok_count']:
        
        fig, ax = plt.subplots(figsize=(16, 6))          # set the figure height and width
        
        df_to_use = messenger_considered_countries[messenger_considered_countries['Country']==country]  # use only data for given country
        
        g = sns.lineplot(data=df_to_use, x='measurement_start_day', y=y, hue='messenger',
                         marker='o', ax=ax, markersize=4, alpha=.7, palette='hls', linewidth = 4)  # plot with seaborn
        
        # format the x-axis
        prep_xaxis()
        ax.yaxis.label.set_size(fontsize=14)  # increase fontsize
        
        # line for when war started
        ukraine_war_line()
        
        # line for when Iranian protests started
        iranian_protest_line()
# bbox_to_anchor=(0.5, -0.5) , fancybox=True, shadow=True, ncol=5
        lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.1, 0.9), fontsize=15)  # move the legend
        
        plt.title('Blocked messengers in ' + country, fontsize=25)  # title

        plt.tight_layout()  # make the figure, such that you can see everything
        
        # save figure
        plt.savefig(folder_lineplots + y + '_' + country + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
        
        
        #--------- as gif: --------
        # !!!  Attention! Takes forever
        
        df_to_use = messenger_considered_countries[messenger_considered_countries['Country']==country]
        
        # fig, ax = plt.subplots(figsize=(16, 6))
        all_dates = list(set((df_to_use['measurement_start_day'])))
        number_dates = len(all_dates)
        
        all_dates.sort() # dates are not ordered anymore, thus we first have to order them: 
        
        if (len(all_dates) - number_dates)>0:
            all_dates = all_dates[:number_dates]
        df_to_use = df_to_use[df_to_use['measurement_start_day'].isin(all_dates)]
        
        fig, ax = plt.subplots(figsize=(16, 6))
        plt.xlim(0, len(all_dates))
        plt.xticks(range(len(all_dates)))
        ax.set_xticklabels(all_dates[:len(all_dates)])
        
        # line for when war started
        ukraine_war_line()
        
        # iranian protests started
        iranian_protest_line()
        
        n = 20  # Keeps every nth label
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        plt.ylim(0, np.max(df_to_use[y]))
        
        # rotate ticks, better readible
        plt.xticks(rotation=45, fontsize=12)
        
        ax.yaxis.label.set_size(fontsize=14)  # increase fontsize
        
        plt.title('Blocked messengers in ' + country, fontsize=25)  # title
        
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), fancybox=True, shadow=True, ncol=5)

        def animate(i):
            data = df_to_use[df_to_use['measurement_start_day'].isin(all_dates[:int(i+1)])]
            p = sns.lineplot(data=data, x='measurement_start_day', y=y, hue='messenger',
                             marker='o', ax=ax, markersize=4, alpha=.7, palette='hls', linewidth = 4)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            # plt.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, ncol=5)
            plt.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.1, 0.9), fontsize=15)  # move the legend
            plt.tight_layout()  # make the figure, such that you can see everything
            return p 
        anim = animation.FuncAnimation(fig, animate, frames=len(df_to_use[y]), interval=2)
        writergif = animation.PillowWriter(fps=15)  
        anim.save(folder_gifs + y + '_' + country + '.gif',writer=writergif)
        

  
