# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:55:57 2024

@author: Shubham_Patidar
"""


pricing_dict =  {"gpt-4o":{'input': 5,'output':15},
                 'gpt-4o-2024-08-06':{'input': 5,'output':15},
                 'gpt-4o-2024-05-13':{'input': 5,'output':15},
                 'gpt-4o-mini':{'input': 0.15,'output':0.6},
                 'gpt-4o-mini-2024-07-18':{'input': 0.15,'output':0.6},
                 'text-embedding-3-small': {'input': 0.02,'output':0},
                 'text-embedding-3-large': {'input': 0.13,'output':0},
                 'ada v2':{'input': 0.1,'output':0},
                 "chatgpt-4o-latest": {'input': 5.00, 'output': 15.00},
                "gpt-4-turbo": {'input': 10.00, 'output': 30.00},
                "gpt-4-turbo-2024-04-09": {'input': 10.00, 'output': 30.00},
                "gpt-4": {'input': 30.00, 'output': 60.00},
                "gpt-4-32k": {'input': 60.00, 'output': 120.00},
                "gpt-4-0125-preview": {'input': 10.00, 'output': 30.00},
                "gpt-4-1106-preview": {'input': 10.00, 'output': 30.00},
                "gpt-4-vision-preview": {'input': 10.00, 'output': 30.00},
                "gpt-3.5-turbo-0125": {'input': 0.50, 'output': 1.50},
                "gpt-3.5-turbo-instruct": {'input': 1.50, 'output': 2.00},
                "gpt-3.5-turbo-1106": {'input': 1.00, 'output': 2.00},
                "gpt-3.5-turbo-0613": {'input': 1.50, 'output': 2.00},
                "gpt-3.5-turbo-16k-0613": {'input': 3.00, 'output': 4.00},
                "gpt-3.5-turbo-0301": {'input': 1.50, 'output': 2.00},
                "davinci-002": {'input': 2.00, 'output': 2.00},
                "babbage-002": {'input': 0.40, 'output': 0.40},
                "gpt-4-0613":  {'input': 30, 'output': 60},
                 
}

import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from datetime import datetime
import io

# Function to calculate costs
# Function to calculate costs
def calculate_costs(row, missing_models):
    model = row['model']
    n_context = row['n_context_tokens_total']
    n_generate = row['n_generated_tokens_total']

    if model in pricing_dict:
        input_cost = (n_context / 1e6) * pricing_dict[model]['input']
        output_cost = (n_generate / 1e6) * pricing_dict[model]['output']
        return input_cost + output_cost
    else:
        missing_models.add(model)
        return 0

# Load the CSV data by passing the file path directly
def load_data(file_path):
    df = pd.read_csv(file_path)

    # Check for the 'timestamp' column
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    elif 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)
    else:
        raise KeyError("The 'timestamp' column is missing from the file.")

    # Track missing models
    missing_models = set()

    # Calculate total cost
    df['total_cost'] = df.apply(lambda row: calculate_costs(row, missing_models), axis=1)

    # Filter out null values in the API key column
    df = df[df['api_key_name'].notnull()]

    return df, missing_models

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("API Usage Cost Dashboard"),
    dcc.Input(
        id='file-path-input',
        type='text',
        placeholder='Enter the CSV file path',
        style={'width': '100%'}
    ),
    html.Button('Load Data', id='load-data-button'),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=None,
        end_date=None,
        display_format='Y-MM-DD',
    ),
    dcc.Dropdown(
        id='api-key-dropdown',
        multi=True,
        placeholder="Select API Key(s)"
    ),
    dcc.Graph(id='cost-graph'),
    html.H2(id='total-cost', style={'textAlign': 'center'}),
    html.H3(id='missing-models-alert', style={'color': 'red', 'textAlign': 'center'}),
])

@app.callback(
    [Output('date-picker-range', 'start_date'),
     Output('date-picker-range', 'end_date'),
     Output('api-key-dropdown', 'options'),
     Output('date-picker-range', 'min_date_allowed'),
     Output('date-picker-range', 'max_date_allowed'),
     Output('date-picker-range', 'initial_visible_month'),
     Output('api-key-dropdown', 'value'),
     Output('missing-models-alert', 'children')],
    Input('load-data-button', 'n_clicks'),
    State('file-path-input', 'value')
)
def update_dropdown(n_clicks, file_path):
    if n_clicks is None or file_path is None:
        return None, None, [], None, None, None, None, ""

    try:
        df, missing_models = load_data(file_path)

        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        options = [{'label': key, 'value': key} for key in df['api_key_name'].unique()]

        if missing_models:
            missing_models_text = f"Pricing not available for models: {', '.join(missing_models)}"
        else:
            missing_models_text = ""

        return min_date, max_date, options, min_date, max_date, min_date, [], missing_models_text

    except Exception as e:
        return None, None, [], None, None, None, None, str(e)

@app.callback(
    Output('cost-graph', 'figure'),
    Output('total-cost', 'children'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('api-key-dropdown', 'value'),
     State('file-path-input', 'value')]
)
def update_graph(start_date, end_date, api_keys, file_path):
    if file_path is None:
        return {}, ""

    df, _ = load_data(file_path)

    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    
    if api_keys:
        filtered_df = filtered_df[filtered_df['api_key_name'].isin(api_keys)]

    total_cost = filtered_df['total_cost'].sum()
    total_cost_text = f"Total Cost: ${total_cost:,.2f}"

    fig = px.bar(filtered_df, x='timestamp', y='total_cost', color='api_key_name', barmode='group',
                 title="API Usage Cost Over Time")

    return fig, total_cost_text

if __name__ == '__main__':
    app.run_server(debug=True)