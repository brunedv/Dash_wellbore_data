""" Dash web app
to visualize cross section
"""
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import utm
import pandas as pd
import numpy as np
import colorlover as cl
import pyarrow.parquet as pq


from flask import Flask
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# Loading data

## Lat/Long 

data_lata_long = pd.read_csv('data/MannvilleWells_LatLong.csv')

## Tops

data_tops = pd.read_csv('data/data_tops.csv')
data_tops.Pick = data_tops.Pick.replace('        ', None)
data_tops.Pick = data_tops.Pick.astype('float')
## Core data
data_core = pd.read_csv('data/INTELLOG.TXT', sep='\t', names=['SitID', 'Depth', 'LithID', 'W_Tar', 'SW', 'VSH', 'PHI', 'RW'])

## Well-logs
table_wellbore = pq.read_table('data/data_wellbore.parquet')
data_wellbore = table_wellbore.to_pandas()

list_tops = data_tops.Tops.unique()
list_color = cl.scales['10']['qual']['Paired']

list_logs = ['DPHI', 'GR', 'ILD', 'NPHI']
dict_range = {'DPHI': [0, 0.4], 'GR': [0, 150], 'ILD': [0.1, 1000], 'NPHI': [0, 0.6]}
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}
server = Flask(__name__)


## mapbox accesss token
mapbox_access_token = 'pk.eyJ1IjoiYnJ1bmVkdiIsImEiOiJjazRuNnBzamQxd3dnM2xudG1kM3F2NnYyIn0.SNApMPKF6uvVNn_qcJTiLg'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)

px.set_mapbox_access_token(mapbox_access_token)
## Map of the field
fig_map = px.scatter_mapbox(data_lata_long, lat="lat", lon="lng", size_max=15, zoom=4, hover_name="SitID")

        
app.layout = html.Div(children=[
    html.H1(children='Wellbore Data McMurray Field'),
    dcc.Graph(id='basic-interactions', figure=fig_map),
    ## selection for the cross-section
    html.Div([
       dcc.Dropdown(id='selected-tops', options=[
           {'label': tops_c, 'value': tops_c} for tops_c in list_tops], value=[list_tops[0]], multi=True, className="six columns"),
       dcc.Dropdown(id='selected-logs', options=[{'label': tops_c, 'value': tops_c} for tops_c in list_logs], clearable=False, 
       value=list_logs[0], className="six columns")], className="row"),
    ## Cross-section
    dcc.Graph( id='cross_section', style={'width': 1500, 'overflowX': 'scroll'}),
    ## Table of the selected wells
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
## Update the dash table, display the seleted wells
@app.callback(
    Output('well-selected-table', 'data'),
    [Input('basic-interactions', 'selectedData')])
def display_selected_wells(selectedData):
    if selectedData != None:
        index_selected = pd.DataFrame(selectedData["points"])
        data_sub = data_lata_long.loc[index_selected.pointNumber.values, ['UWI', 'lat', 'lng']]
        x, y, _, _ = utm.from_latlon(data_sub.lat.values, data_sub.lng.values)
        x = (x-np.mean(x))
        y = (y-np.mean(y))
        if np.abs(pearsonr(x, y)[0]) > 0.8:
            lr = LinearRegression()
            lr.fit(y.reshape(-1, 1), x.reshape(-1, 1))
            data_sub['x'] = lr.predict(y.reshape(-1, 1))

            data_sub.sort_values('x', ascending=False, inplace=True)
        else:
            if (x.max()-x.min()) > (y.max()-y.min()):
                data_sub.sort_values('lng', ascending=False, inplace=True)
            else:
                data_sub.sort_values('lat', ascending=False, inplace=True)
        return data_sub.to_dict("rows")
    else:
        return []
## Update the cross-section with the tops

@app.callback(
    Output('cross_section', 'figure'),
    [Input('basic-interactions', 'selectedData'), Input('selected-tops', 'value'), Input('selected-logs', 'value')])
def display_crossection(selectedData, list_tops, select_logs):
    if selectedData != None:
        index_selected = pd.DataFrame(selectedData["points"])
        data_sub = data_lata_long.loc[index_selected.pointNumber.values, ['UWI', 'SitID', 'lat', 'lng']]
        x, y, _, _ = utm.from_latlon(data_sub.lat.values, data_sub.lng.values)
        x = (x-np.mean(x))
        y = (y-np.mean(y))
        if np.abs(pearsonr(x,y)[0]) > 0.8:
            lr = LinearRegression()
            lr.fit(y.reshape(-1, 1), x.reshape(-1, 1))
            data_sub['x'] = lr.predict(y.reshape(-1, 1))

            data_sub.sort_values('x', ascending=False, inplace=True)
        else:
            if (x.max()-x.min()) > (y.max()-y.min()):
                data_sub.sort_values('lng', ascending=False, inplace=True)
            else:
                data_sub.sort_values('lat', ascending=False, inplace=True)
        list_well_sub = data_sub.SitID.values.tolist()
        current_tops = data_tops[data_tops.SitID.isin(list_well_sub)]
        current_tops_first = current_tops[current_tops.Tops.isin(list_tops)]
        ### Cross-section  creation
        fig = make_subplots(rows=1, cols=len(list_well_sub), shared_yaxes=True, shared_xaxes=True, subplot_titles=list_well_sub)
        ## well log
        for j in range(len(list_well_sub)):
            data_wellbore_sub = data_wellbore[data_wellbore.SitID == list_well_sub[j]]
            fig.add_trace(go.Scattergl(x=data_wellbore_sub[select_logs], y=data_wellbore_sub.DEPT, marker={'color':'blue'}), row=1, col=j+1)
        ## Adding tops
        for j in range(len(list_well_sub)):
            for h in range(len(list_tops)):
                h_tops = list_tops[h]
                if not(current_tops_first.loc[(current_tops_first.SitID == list_well_sub[j]) & (current_tops_first.Tops == h_tops), 'Pick'].empty):
                    depth_tops = float(current_tops_first.loc[(current_tops_first.SitID == list_well_sub[j]) & (current_tops_first.Tops == h_tops), 'Pick'].values[0])
                    if select_logs == 'ILD':
                        fig.add_trace(go.Scattergl(x=[10], y=[depth_tops], mode='text', text=h_tops, textposition="top center", textfont={
                            "color": "Black",
                            "size": 12,
                        }), row=1, col=j+1)
                    else:
                        fig.add_trace(go.Scattergl(x=[sum(dict_range[select_logs])/2], y=[depth_tops], mode='text', text=h_tops, textposition="top center", textfont={
                            "color": "Black",
                            "size": 12,
                        }), row=1, col=j+1)

                    fig.add_shape(
                            # Line Horizontal
                            go.layout.Shape(
                                name=h_tops,
                                type="line",
                                x0=dict_range[select_logs][0],
                                y0=depth_tops,
                                x1=dict_range[select_logs][1],
                                y1=depth_tops,
                                xref='x'+str(j+1),
                                yref='y'+str(j+1),
                                line=dict(
                                    color=list_color[h%10],
                                    width=2,
                                    
                                )
                        )
                    )
        fig.update_layout(title_text="Cross-section", autosize=False, width=150*(len(list_well_sub)+1)
                    , yaxis={'range':[data_wellbore[data_wellbore.SitID.isin(list_well_sub)].DEPT.max()+20, data_wellbore[data_wellbore.SitID.isin(list_well_sub)].DEPT.min()-20]},
                    showlegend=False)
        if select_logs == 'ILD':
            for j in range(len(list_well_sub)):
                fig['layout']['xaxis'+str(j+1)].update(type="log")
                fig['layout']['xaxis'+str(j+1)].update(range=[-1, 3])

        else:
            for j in range(len(list_well_sub)):
                fig['layout']['xaxis'+str(j+1)].update(range=dict_range[select_logs])
    
        return fig
    else:
        return []

if __name__ == '__main__':
    server.run(debug=True, host='0.0.0.0')