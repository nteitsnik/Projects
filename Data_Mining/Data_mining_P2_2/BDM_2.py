# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:54:25 2025

@author: n.nteits
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.neighbors import LocalOutlierFactor
import re
import plotly.figure_factory as ff
import plotly.express as px
import os

import kagglehub
import pandas as pd 
from sklearn.cluster import KMeans
import dash
from dash import dcc, html, Input, Output 
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.neighbors import LocalOutlierFactor
import re

os.environ["OMP_NUM_THREADS"] = "1"
path = kagglehub.dataset_download("alejopaullier/usa-counties-coordinates")

print("Path to dataset files:", path)


coordinates = pd.read_csv(f'{path}\cfips_location.csv')
coordinates['cfips'] = coordinates['cfips'].astype(str).apply(lambda x: x.zfill(5))
coordinates=coordinates.rename(columns={'cfips':'FIPS'})



cwd = os.getcwd()
df = pd.read_excel(r'1952.xls')



pd.set_option('display.max_columns', None)
#Get to kow your Data


df.head(5)
df.tail(5)



df.shape
df.dtypes


df.describe()

#Drop Counties with 0 members
df=df[df['TOTMEMB']!=0]
#Create FIPS Code
df['STCODE'] = df['STCODE'].astype(str).str.zfill(2)
df['CCODE'] = df['CCODE'].astype(str).str.zfill(3)
df['FIPS']= df['STCODE']+df['CCODE']
df.reset_index(level=None, drop=True, inplace=True)

#Percentage of Orthodox
df['ORT_PERC']=(df['ARAPO_M']+df['GRKAD_M']+df['ACROC_M']+df['BEOC_M'])/df['TOTPOP']
df[['CNAME','ORT_PERC']].sort_values(by='ORT_PERC', ascending=False)

#Drop Churches with no ppl
df = df.drop(columns=[col for col in df.columns if (df[col] == 0).all()])

#Keep only Church population
columns_with_m = df.columns[df.columns.str.endswith('_M')]

columns_with_m_list = columns_with_m.tolist()

# Append a new item to the list
columns_with_m_list+=['TOTMEMB']

#Create a dataframe with members
df_members=df[columns_with_m_list]
column_sums = df_members.sum(axis=0)

#Create a Df with percentages
df_members_perc = df_members.div(df['TOTMEMB'], axis=0)
#kirtosis and skewness values
kurtosis_values = pd.DataFrame(data=kurtosis(df_members_perc, axis=1))  # Axis=1 computes row-wise statistics
skew_values = pd.DataFrame(data=skew(df_members_perc, axis=1))  # Axis=1 computes row-wise statistics

kurtosis_values.describe()
skew_values.describe()

df_members_perc = df_members_perc.drop(columns='TOTMEMB')

#LOF on the original data

lof = LocalOutlierFactor(n_neighbors=50, contamination=3/3069) 
outlier_scores = lof.fit_predict(df_members_perc)  # -1 indicates outliers, 1 indicates inliers

indices = np.where(outlier_scores == -1)

rows = df_members_perc.iloc[indices]
print(df['CNAME'].iloc[indices])
#Find the non zero Values
df_members_perc['non_zero_count'] = (df_members_perc != 0).sum(axis=1)

#Append Coordinates

listofcolumns=[]
listofcolumns.extend(df.columns[0:6])
listofcolumns.extend(['FIPS'])
listofcolumns.extend(columns_with_m)


result = pd.merge(df[listofcolumns], coordinates, on='FIPS', how='inner')


#Create 

#initialise K-means model and results
kmeans = KMeans(n_clusters=1, n_init=10, random_state=0, max_iter=1000)
coords=result[['lng','lat']]


# Create a Dataframe Containing Church Name and Population and centroid coordinates

#Select columns
selected_columns = result.columns[7:-3]

#Select sum per column
population_data = result[selected_columns].sum().values  # Calculate the sum for each column

#Initialize Dataframe 
centroids = pd.DataFrame({
    'Church': selected_columns,  # Take the first row from the transposed data
    'Population': population_data,
    'Long' : ' ',
    'Lat' : ' '
})

print(centroids.head(10))
#Fill Centroid Coordinates 
for col in result.columns[7:-3]:
    wt=result[col]    
    wt_kmeansclus = kmeans.fit(coords, sample_weight = wt)
    centroids.loc[centroids['Church']==col,'Long']=wt_kmeansclus.cluster_centers_[0,0]
    centroids.loc[centroids['Church']==col,'Lat']=wt_kmeansclus.cluster_centers_[0,1]

#The final centroid

final_centroid = pd.DataFrame(columns=['Church','Long','Lat'])



kmeans = KMeans(n_clusters=1, n_init=10, random_state=0, max_iter=1000)
final_kmeans = kmeans.fit(centroids[['Long','Lat']], sample_weight = centroids['Population'])
final_centroid.loc[0,'Church']='Cross-Religion Center'
final_centroid.loc[0,'Long']=final_kmeans.cluster_centers_[0,0]
final_centroid.loc[0,'Lat']=final_kmeans.cluster_centers_[0,1]

print(final_centroid)
#Dash


# Load the data
data = result  # Ensure your data contains the columns to choose from

# Load GeoJSON for counties
geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

app = dash.Dash(__name__)
fig_scatter = px.scatter_geo(
    centroids,
    lon='Long',
    lat='Lat',
    hover_name='Church',
    title="Static Points Overlay",
    color='Population',  # Color based on population
   color_continuous_scale='Viridis',
   size='Population'
)


fig_scatter.update_layout(
    paper_bgcolor='#f0f8ff',#Colouring
    plot_bgcolor='#f0f8ff',
    height=1000,  # Height of the plot
    width=1400,  # Width of the plot
    geo=dict(
        bgcolor='#f0f8ff',
        lakecolor='rgb(255, 255, 255)',  # Background color for the map
        projection_type='albers usa' , # Projection type (optional)
        
    )
)

fig_scatter.add_scattergeo(
    lon=final_centroid['Long'],
    lat=final_centroid['Lat'],
    mode='markers',
    marker=dict(
        symbol='star',
        size=20,
        color='red',
    ),
    name="Cross-Religion Centre"  # Legend name for the star
)

# Layout with dynamic dropdown only
app.layout = html.Div(style={'backgroundColor': '#f0f8ff'},children=[
    
    html.H1("Dynamic Choropleth Map with FIPS Codes"),

    # Dropdown for selecting column dynamically
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in data.columns[7:-3] if col not in ['CNAME', 'FIPS']],
        value='AOG_M',  # Default column
        style={'width': '50%'}
    ),

    # Graph to show the choropleth map
    dcc.Graph(id='choropleth-map'),
    dcc.Graph(id='scatter-map', figure=fig_scatter)
])

# Callback to update the choropleth map based on dropdown value
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('column-dropdown', 'value')]
)
def update_map(selected_column):
    # Update the choropleth map based on selected column
    fig = px.choropleth(data,
                        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",  # GeoJSON for counties
                        locations='FIPS',  # FIPS codes
                        color=selected_column,  # Values to color the map
                        scope="usa",  # Focus on the USA
                        hover_name='CNAME',  # Show FIPS codes on hover
                        title=f"County-Level Map with {selected_column} Values",
                        color_continuous_scale='Hot',  # Diverging color scale focusing on 0
                        range_color=[data[selected_column].min(), data[selected_column].max()])  # Adjust color range if needed)
    fig.update_layout(
        paper_bgcolor='#f0f8ff', #Colouring
        plot_bgcolor='#f0f8ff',
        height=1000,  # Height of the plot
        width=1400,  # Width of the plot
        geo=dict(
            bgcolor='#f0f8ff',
            lakecolor='rgb(255, 255, 255)',  # Background color for the map
            projection_type='albers usa'  # Projection type (optional)
        )
    )
    filtered_centroids = centroids[centroids['Church'] == selected_column]
    fig.add_scattergeo(
        lon=filtered_centroids['Long'].values, 
        lat=filtered_centroids['Lat'].values,
        mode='markers',
        marker=dict(size=20, color='purple'),
        name="Point Marker"
    )
    fig.write_html('static_map.html')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    
    

