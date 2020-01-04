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
import colorlover as cl


from flask import Flask
from textwrap import dedent as d
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from utils import calc_plane

from plotly.subplots import make_subplots


data_lata_long = pd.read_csv('data/MannvilleWells_LatLong.csv')
data_tops = pd.read_csv('data/data_tops.csv')
data_tops.Pick = data_tops.Pick.replace('        ',None)
data_tops.Pick = data_tops.Pick.astype('float')
data_core = pd.read_csv('data/INTELLOG.TXT',sep='\t',names =['SitID','Depth','LithID','W_Tar','SW','VSH','PHI','RW'])

list_tops = data_tops.Tops.unique()
list_color = cl.scales['10']['qual']['Paired']

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
    html.Div([
       dcc.Dropdown(id='selected-tops', options=[
        {'label': tops_c, 'value': tops_c} for tops_c in list_tops],value=[list_tops[0]], multi=True,className="six columns"),

        html.Div(id='legend-tops',className="six columns"),
    ], className="row"),
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
    [Input('basic-interactions', 'selectedData'),Input('selected-tops', 'value')])
def display_crossection(selectedData,list_tops):
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
        current_tops = data_tops[data_tops.SitID.isin(list_well_sub)]
        current_tops_first = current_tops[current_tops.Tops.isin(list_tops)]
        fig = make_subplots(rows=1, cols=len(list_well_sub), shared_yaxes=True,subplot_titles=list_well_sub)
        for j in range(len(list_well_sub)):
            data_core_sub = data_core[data_core.SitID==list_well_sub[j]]
            fig.add_trace(go.Scattergl(x=data_core_sub.VSH, y=data_core_sub.Depth,marker={'color':'blue'}),row=1, col=j+1)
        for j in range(len(list_well_sub)):
            for h in range(len(list_tops)):
                h_tops = list_tops[h]
                if not(current_tops_first.loc[(current_tops_first.SitID==list_well_sub[j]) &(current_tops_first.Tops==h_tops),'Pick'].empty):
                    depth_tops = float(current_tops_first.loc[(current_tops_first.SitID==list_well_sub[j]) &(current_tops_first.Tops==h_tops),'Pick'].values[0])
                    fig.add_trace(go.Scattergl(x=[0.5], y= [depth_tops],mode='text',text=h_tops,textposition="top center",textfont={
        "color": "Black",
        "size": 12,
    },),row=1, col=j+1)

                    fig.add_shape(
                            # Line Horizontal
                            go.layout.Shape(
                                name=h_tops,
                                type="line",
                                x0=0,
                                y0=depth_tops,
                                x1=1,
                                y1=depth_tops,
                                xref= 'x'+str(j+1),
                                yref= 'y'+str(j+1),
                                line=dict(
                                    color=list_color[h%10],
                                    width=2,
                                    
                                ),
                        )
                    )
        fig.update_layout(title_text="Cross-section",autosize=False,width=150*len(list_well_sub)
                    ,yaxis={'range':[data_core.Depth.max(),data_core.Depth.min()]},xaxis={'range':[0,1]},
                    showlegend=False)
        """
        fig.update_layout(title_text="Cross-section",autosize=True,
                    yaxis={'range':[data_core.Depth.max(),data_core.Depth.min()]},xaxis={'range':[0,1]},
                    showlegend=False)
        """
        return fig
    else:
        return []

if __name__ == '__main__':
    server.run(debug=True)