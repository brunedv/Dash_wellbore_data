import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px
import pandas as pd

data_lata_long = pd.read_csv('data/MannvilleWells_LatLong.csv')

mapbox_access_token = 'pk.eyJ1IjoiYnJ1bmVkdiIsImEiOiJjazRuNnBzamQxd3dnM2xudG1kM3F2NnYyIn0.SNApMPKF6uvVNn_qcJTiLg'
px.set_mapbox_access_token(mapbox_access_token)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=px.scatter_mapbox(data_lata_long, lat="lat", lon="lng")
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)