#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:29:48 2022

@author: annika


We use data from https://ooni.org to count how many and which websites or 
messengers are blocked.
We visualize this data using a world map 

"""

do_html =False

from matplotlib import pyplot as plt
import plotly.express as px
import pandas as pd
import os
import geojson
import json


#%% Some variables

path_to_data_folder = '../../Daten/Ooni/'  # where the data is stored

folder_maps = 'output/maps/'
if not os.path.exists(folder_maps):
    os.makedirs(folder_maps)  

#%% read in messenger data
# data for different messengers. From all countries, but with iso name from countries
messenger_names = ['Facebook',  'Whatsapp', 'Signal', 'Telegram']

path_iso_countries_three_digits = '../../Daten/all_iso_countries.xlsx'
iso_countries_three_digits = pd.read_excel(path_iso_countries_three_digits)  # country-codes to country names
# cols = list(iso_countries_three_digits.columns)
iso_countries_three_digits = iso_countries_three_digits[['ISO', 'ISO3', 'Country', 'Continent']]

#  create one large df for the messenger data:
messenger_df = pd.DataFrame()

for messenger in messenger_names:
    current_df = pd.read_csv(path_to_data_folder + 'all_countries_' + messenger + '.csv')
    current_df['messenger'] = messenger 
    messenger_df = pd.concat([messenger_df, current_df])

# messenger_df.head()
messenger_df.set_index(keys='probe_cc', inplace=True)  # set index as iso country code

# merge country names to the messenger dataset:
iso_countries_three_digits.rename({'ISO':'probe_cc'}, inplace=True, axis=1)  # names as in the other dataframes
iso_countries_three_digits.set_index(keys='probe_cc', inplace=True)  # set index as iso country code

messenger_df = messenger_df.merge(iso_countries_three_digits, how="left", on="probe_cc")
messenger_df.reset_index(inplace=True)

# need to sum up over all the days for the static figure
messenger_df_all_counts = messenger_df.groupby(['Country', 'probe_cc', 'ISO3', 'Continent']).sum().reset_index() 

# %% plot some maps

# bubble map
fig = px.scatter_geo(messenger_df_all_counts, locations="ISO3",
                     hover_name="Country", size="anomaly_count",
                     projection="natural earth",  color="Continent")

fig.write_image(folder_maps + "bubble_messengers_anomalies.png")

# choro map

# read in geo data for the countries, since we need the boundaries of the countries
with open('../../Daten/countries.geojson') as f:
    gj = geojson.load(f)

messenger_df_all_counts.rename({'ISO3':'ISO_A3'}, inplace=True,axis='columns')

# plot figure
fig = px.choropleth(messenger_df_all_counts, locations='ISO_A3', color='anomaly_count',
                           color_continuous_scale="OrRd", locationmode='ISO-3',
                           geojson=gj,
                           range_color=(0, max(messenger_df_all_counts['anomaly_count']))
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.write_image(folder_maps + "heatmap_messengers_anomalies.png")


# %% blocked websites

# read in data
df_blocked_websites = pd.read_csv('../../Daten/Ooni/all_countries_webpages.csv')
df_blocked_websites_total_counts = df_blocked_websites.groupby(['probe_cc']).sum().reset_index()
df_blocked_websites_total_counts = df_blocked_websites_total_counts.merge(iso_countries_three_digits, how="left", on="probe_cc")
df_blocked_websites_total_counts.reset_index(inplace=True)

df_blocked_websites = df_blocked_websites.merge(iso_countries_three_digits, how="left", on="probe_cc")

df_blocked_websites_total_counts.rename({'ISO3':'ISO_A3'}, inplace=True,axis='columns')

# some static figures:
for y in ['anomaly_count', 'confirmed_count']:
    fig = px.choropleth(df_blocked_websites_total_counts, locations='ISO_A3', color=y,
                               color_continuous_scale="Jet", locationmode='ISO-3',
                               geojson=gj,
                               range_color=(0, max(messenger_df_all_counts['anomaly_count']))
                              )
    # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    # make the colorbar where it is supposed to be and a nice size 
    fig.update_layout(margin={"r":0,"t":0,"l":5,"b":0},
        coloraxis_colorbar=dict(
        title="",
        thicknessmode="pixels", thickness=20,
        lenmode="pixels", len=280
    ), 
        title_text=y , title_x=0.5, title_y=.85
        )
    
    fig.write_image(folder_maps + 'heatmap_websites_' + y + '.png')
    
    fig = px.scatter_geo(df_blocked_websites_total_counts, locations="ISO_A3",
                         hover_name="Country", size="anomaly_count",
                         projection="natural earth")

    fig.write_image(folder_maps + "bubble" + y + ".png")

# %% as animation:
# list(df_blocked_websites.columns)
if do_html:
    all_dates = list(df_blocked_websites['measurement_start_day'])
    all_dates_third = list(set(all_dates))[::20]  # every tenth, otherwise too many and my computer faints
    
    df_to_use = df_blocked_websites[df_blocked_websites['measurement_start_day'].isin(all_dates_third)]
    
    fig = px.choropleth(df_to_use, locations='ISO3', color='confirmed_count',
                               color_continuous_scale="viridis", locationmode='ISO-3',
                               geojson=gj, animation_frame="measurement_start_day",
                               range_color=(min(df_to_use['confirmed_count']), max(df_to_use['confirmed_count']))
                              )    
        
    fig.write_html(folder_maps + "heatmap_blocked_websites_smaller.html")
    fig.write_json(folder_maps + "heatmap_blocked_websites_smaller.json")
