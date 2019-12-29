import csv
import pandas as pd
import numpy as np
from itertools import chain
import plotly.express as px
import geopy.distance
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html

baseCols = [
'dbn'
,'school_name'
,'borocode'
,'overview_paragraph'
,'borough'
,'latitude'
,'longitude'
,'bin'
,'bbl'
,'nta'
,'total_students'
,'neighborhood'
,'specialized'
]

MBToken =  'pk.eyJ1IjoiYnJ1bmVkdiIsImEiOiJjazRuNnBzamQxd3dnM2xudG1kM3F2NnYyIn0.SNApMPKF6uvVNn_qcJTiLg'

px.set_mapbox_access_token(MBToken)

df = pd.read_csv('https://data.cityofnewyork.us/resource/23z9-6uk9.csv')

specDF = df[baseCols]

specSchools = specDF[specDF['specialized'].notna() == True][['school_name'
                                                             ,'overview_paragraph'
                                                             ,'borough'
                                                             ,'latitude'
                                                             ,'longitude'
                                                             ,'total_students'
                                                            ,'neighborhood']]

specSchools['total_students'] = pd.to_numeric(specSchools['total_students'])

specFig = px.scatter_mapbox(specSchools.dropna()
                        , lat="latitude"
                        , lon="longitude"
                        , color="borough"
                        , size="total_students"
                        , text="school_name"
                        , hover_name="school_name"
                        , hover_data=["neighborhood"]
                        , size_max=15
                        , zoom=9
                       )

#specFig.show()

app = dash.Dash()

app.layout = html.Div([
    html.Div([html.H1('NYC Specialized Schools')], style={'textAlign': 'center'}),
    dcc.Graph(figure=specFig),
])

if __name__ == '__main__':
    app.run_server(debug=False)