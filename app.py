import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objects as go

import utm
import json
import plotly.express as px
import pandas as pd
import numpy as np


from flask import Flask
from textwrap import dedent as d
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from utils import calc_plane

from plotly.subplots import make_subplots


data_lata_long = pd.read_csv('data/MannvilleWells_LatLong.csv')
data_tops = pd.read_csv('data/data_tops.csv')
data_core = pd.read_csv('data/INTELLOG.TXT',sep='\t',names =['SitID','Depth','LithID','W_Tar','SW','VSH','PHI','RW'])

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}
server = Flask(__name__)



mapbox_access_token = 'pk.eyJ1IjoiYnJ1bmVkdiIsImEiOiJjazRuNnBzamQxd3dnM2xudG1kM3F2NnYyIn0.SNApMPKF6uvVNn_qcJTiLg'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,server=server, external_stylesheets=external_stylesheets)

px.set_mapbox_access_token(mapbox_access_token)
fig_map = px.scatter_mapbox(data_lata_long, lat="lat", lon="lng", size_max=15
                        , zoom=4)
app.layout = html.Div(children=[
    html.H1(children='Wellbore Data McMurray Field'),
    dcc.Graph( id='basic-interactions',figure=fig_map),
    dcc.Graph( id='cross_section',style={'width': 1500, 'overflowX': 'scroll'}),

    dash_table.DataTable(
        id='well-selected-table',
        columns=[
            {'name': 'UWI', 'id': 'UWI'},
             {'name': 'lat', 'id': 'lat'},
              {'name': 'lng', 'id': 'lng'},
        ],
        data=[],
        editable=True,
    ),

    
])

@app.callback(
    Output('well-selected-table', 'data'),
    [Input('basic-interactions', 'selectedData')])
def display_selected_wells(selectedData):
    if selectedData!=None:
        index_selected = pd.DataFrame(selectedData["points"])
        data_sub = data_lata_long.loc[index_selected.pointNumber.values,['UWI','lat','lng']]
        x,y,_,_=utm.from_latlon(data_sub.lat.values,data_sub.lng.values)
        x = (x-np.mean(x))
        y = (y-np.mean(y))
        if np.abs(pearsonr(x,y)[0])>0.8:
            lr =LinearRegression()
            lr.fit(y.reshape(-1,1),x.reshape(-1,1))
            data_sub['x']=lr.predict(y.reshape(-1,1))

            data_sub.sort_values('x',ascending=False,inplace=True)
        else:
            if (x.max()-x.min())>(y.max()-y.min()):
                data_sub.sort_values('lng',ascending=False,inplace=True)
            else:
                data_sub.sort_values('lat',ascending=False,inplace=True)
        return data_sub.to_dict("rows")
    else:
        return []
@app.callback(
    Output('cross_section', 'figure'),
    [Input('basic-interactions', 'selectedData')])
def display_crossection(selectedData):
    if selectedData!=None:
        index_selected = pd.DataFrame(selectedData["points"])
        data_sub = data_lata_long.loc[index_selected.pointNumber.values,['UWI','SitID','lat','lng']]
        x,y,_,_=utm.from_latlon(data_sub.lat.values,data_sub.lng.values)
        x = (x-np.mean(x))
        y = (y-np.mean(y))
        if np.abs(pearsonr(x,y)[0])>0.8:
            lr =LinearRegression()
            lr.fit(y.reshape(-1,1),x.reshape(-1,1))
            data_sub['x']=lr.predict(y.reshape(-1,1))

            data_sub.sort_values('x',ascending=False,inplace=True)
        else:
            if (x.max()-x.min())>(y.max()-y.min()):
                data_sub.sort_values('lng',ascending=False,inplace=True)
            else:
                data_sub.sort_values('lat',ascending=False,inplace=True)
        list_well_sub = data_sub.SitID.values.tolist()
        fig = make_subplots(rows=1, cols=len(list_well_sub), shared_yaxes=True)
        for j in range(len(list_well_sub)):
            data_core_sub = data_core[data_core.SitID==list_well_sub[j]]
            fig.add_trace(go.Scatter(x=data_core_sub.VSH, y=data_core_sub.Depth),row=1, col=j+1)
        fig.update_layout(title_text="Cross-section",width=5000,yaxis={'range':[data_core.Depth.max(),data_core.Depth.min()]},xaxis={'range':[0,1]})
        return fig
    else:
        return []

if __name__ == '__main__':
    server.run(debug=True)