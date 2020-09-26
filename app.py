""" Dash web app
to visualize cross section
"""
import colorlover as cl
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq
import utm
from dash.dependencies import Input, Output
from flask import Flask
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Loading data

# Lat/Long

DATA_LAT_LONG = pd.read_csv('data/MannvilleWells_LatLong.csv')

# Tops

DATA_TOPS = pd.read_csv('data/data_tops.csv')
DATA_TOPS.Pick = DATA_TOPS.Pick.replace('        ', None)
DATA_TOPS.Pick = DATA_TOPS.Pick.astype('float')
# Core data
DATA_CORE = pd.read_csv('data/INTELLOG.TXT', sep='\t',
                        names=['SitID', 'Depth', 'LithID', 'W_Tar', 'SW', 'VSH', 'PHI', 'RW'])

# Well-logs
TABLE_WELLBORE = pq.read_table('data/data_wellbore.parquet')
DATA_WELLBORE = TABLE_WELLBORE.to_pandas()

LIST_TOPS = DATA_TOPS.Tops.unique()
LIST_COLOR = cl.scales['10']['qual']['Paired']

LIST_LOGS = ['DPHI', 'GR', 'ILD', 'NPHI']
DICT_RANGE = {'DPHI': [0, 0.4], 'GR': [0, 150],
              'ILD': [0.1, 1000], 'NPHI': [0, 0.6]}

server = Flask(__name__)


# mapbox accesss token
MAP_BOX_ACCESS_TOKEN = 'pk.eyJ1IjoiYnJ1bmVkdiIsImEiOiJjazRuNnBzamQxd3dnM2xudG1kM3F2NnYyIn0.SNApMPKF6uvVNn_qcJTiLg'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, server=server,
                external_stylesheets=external_stylesheets)

px.set_mapbox_access_token(MAP_BOX_ACCESS_TOKEN)
# Map of the field
FIG_MAP = px.scatter_mapbox(DATA_LAT_LONG,
                            lat="lat", lon="lng", size_max=15, zoom=4, hover_name="SitID")


app.layout = html.Div(children=[
    html.H1(children='Wellbore Data McMurray Field'),
    dcc.Graph(id='basic-interactions', figure=FIG_MAP),
    # selection for the cross-section
    html.Div([
        dcc.Dropdown(
            id='selected-tops',
            options=[{'label': tops_c, 'value': tops_c} for tops_c in LIST_TOPS],\
            value=[LIST_TOPS[0]],
            multi=True, className="six columns"),
        dcc.Dropdown(
            id='selected-logs',
            options=[{'label': tops_c, 'value': tops_c}
                     for tops_c in LIST_LOGS],
            clearable=False, value=LIST_LOGS[0], className="six columns")], className="row"),
    # Cross-section
    dcc.Graph(id='cross_section', style={
              'width': 1500, 'overflowX': 'scroll'}),
    # Table of the selected wells
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
def display_selected_wells(selected_data):
    """
    Update the dash table, display the seleted wells
    """
    if selected_data != None:
        index_selected = pd.DataFrame(selected_data["points"])
        data_sub = DATA_LAT_LONG.loc[index_selected.pointNumber.values, [
            'UWI', 'lat', 'lng']]
        x_coord, y_coord, _, _ = utm.from_latlon(
            data_sub.lat.values, data_sub.lng.values)
        x_coord = (x_coord-np.mean(x_coord))
        y_coord = (y_coord-np.mean(y_coord))
        if np.abs(pearsonr(x_coord, y_coord)[0]) > 0.8:
            linear_reg = LinearRegression()
            linear_reg.fit(y_coord.reshape(-1, 1), x_coord.reshape(-1, 1))
            data_sub['x'] = linear_reg.predict(y_coord.reshape(-1, 1))

            data_sub.sort_values('x', ascending=False, inplace=True)
        else:
            if (x_coord.max()-x_coord.min()) > (y_coord.max()-y_coord.min()):
                data_sub.sort_values('lng', ascending=False, inplace=True)
            else:
                data_sub.sort_values('lat', ascending=False, inplace=True)
        return data_sub.to_dict("rows")
    else:
        return []


@app.callback(
    Output('cross_section', 'figure'),
    [Input('basic-interactions', 'selectedData'), Input('selected-tops', 'value'), Input('selected-logs', 'value')])
def display_crossection(selected_data, list_tops, select_logs):
    """
    Display the  crossection with the tops
    Return: figure
    """
    if selected_data != None:
        index_selected = pd.DataFrame(selected_data["points"])
        data_sub = DATA_LAT_LONG.loc[index_selected.pointNumber.values, [
            'UWI', 'SitID', 'lat', 'lng']]
        x_coord, y_coord, _, _ = utm.from_latlon(
            data_sub.lat.values, data_sub.lng.values)
        x_coord = (x_coord-np.mean(x_coord))
        y_coord = (y_coord-np.mean(y_coord))
        if np.abs(pearsonr(x_coord, y_coord)[0]) > 0.8:
            linear_reg = LinearRegression()
            linear_reg.fit(y_coord.reshape(-1, 1), x_coord.reshape(-1, 1))
            data_sub['x'] = linear_reg.predict(y_coord.reshape(-1, 1))

            data_sub.sort_values('x', ascending=False, inplace=True)
        else:
            if (x_coord.max()-x_coord.min()) > (y_coord.max()-y_coord.min()):
                data_sub.sort_values('lng', ascending=False, inplace=True)
            else:
                data_sub.sort_values('lat', ascending=False, inplace=True)
        list_well_sub = data_sub.SitID.values.tolist()
        current_tops = DATA_TOPS[DATA_TOPS.SitID.isin(list_well_sub)]
        current_tops_first = current_tops[current_tops.Tops.isin(list_tops)]
        len_list_well_sub = len(list_well_sub)
        # Cross-section  creation
        fig = make_subplots(rows=1, cols=len_list_well_sub, shared_yaxes=True,
                            shared_xaxes=True, subplot_titles=list_well_sub)
        # well log
        for j in range(len_list_well_sub):
            data_wellbore_sub = DATA_WELLBORE[DATA_WELLBORE.SitID ==
                                              list_well_sub[j]]
            fig.add_trace(go.Scattergl(x=data_wellbore_sub[select_logs], y=data_wellbore_sub.DEPT, marker={
                          'color': 'blue'}), row=1, col=j+1)
        # Adding tops
        for j in range(len_list_well_sub):
            for ix_tops in range(len(list_tops)):
                h_tops = list_tops[ix_tops]
                cond_id = (current_tops_first.SitID == list_well_sub[j])
                cond_tops = (current_tops_first.Tops == h_tops)
                if not(current_tops_first.loc[cond_id & cond_tops, 'Pick'].empty):
                    depth_tops = float(
                        current_tops_first.loc[cond_id & cond_tops, 'Pick'].values[0])
                    if select_logs == 'ILD':
                        fig.add_trace(
                            go.Scattergl(
                                x=[10],
                                y=[depth_tops],
                                mode='text',
                                text=h_tops,
                                textposition="top center",
                                textfont={
                                    "color": "Black",
                                    "size": 12,
                                }),
                            row=1, col=j+1)
                    else:
                        fig.add_trace(
                            go.Scattergl(
                                x=[sum(DICT_RANGE[select_logs])/2],
                                y=[depth_tops],
                                mode='text',
                                text=h_tops,
                                textposition="top center",
                                textfont={
                                    "color": "Black",
                                    "size": 12,
                                }),
                            row=1, col=j+1)
                    fig.add_shape(
                        go.layout.Shape(
                            name=h_tops,
                            type="line",
                            x0=DICT_RANGE[select_logs][0],
                            y0=depth_tops,
                            x1=DICT_RANGE[select_logs][1],
                            y1=depth_tops,
                            xref='x'+str(j+1),
                            yref='y'+str(j+1),
                            line=dict(
                                color=LIST_COLOR[ix_tops % 10],
                                width=2,
                            )
                        )
                    )
        fig.update_layout(
            title_text="Cross-section",
            autosize=False,
            width=150*(len_list_well_sub+1),
            yaxis={'range': [DATA_WELLBORE[DATA_WELLBORE.SitID.isin(list_well_sub)].DEPT.max()+20,
                             DATA_WELLBORE[DATA_WELLBORE.SitID.isin(list_well_sub)].DEPT.min()-20]},
            showlegend=False)
        if select_logs == 'ILD':
            for j in range(len_list_well_sub):
                fig['layout']['xaxis'+str(j+1)].update(type="log")
                fig['layout']['xaxis'+str(j+1)].update(range=[-1, 3])

        else:
            for j in range(len_list_well_sub):
                fig['layout']['xaxis' +
                              str(j+1)].update(range=DICT_RANGE[select_logs])

        return fig
    else:
        return []


if __name__ == '__main__':
    server.run(debug=True, host='0.0.0.0')
