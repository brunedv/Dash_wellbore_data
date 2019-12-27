import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px
import pandas as pd
from flask import Flask

server = Flask(__name__)

data_lata_long = pd.read_csv('data/MannvilleWells_LatLong.csv')

mapbox_access_token = 'pk.eyJ1IjoiYnJ1bmVkdiIsImEiOiJjazRuNnBzamQxd3dnM2xudG1kM3F2NnYyIn0.SNApMPKF6uvVNn_qcJTiLg'


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,server=server, external_stylesheets=external_stylesheets)

px.set_mapbox_access_token(mapbox_access_token)
fig_map = px.scatter_mapbox(data_lata_long, lat="lat", lon="lng", size_max=15
                        , zoom=4)
app.layout = html.Div(children=[
    html.H1(children='Wellbore Data McMurray Field'),

    dcc.Graph( figure=fig_map
    )
])

if __name__ == '__main__':
    server.run(debug=True)