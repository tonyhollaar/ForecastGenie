# -*- coding: utf-8 -*-
"""
Streamlit App: ForecastGenie_TM 
Forecast y based on timeseries data
@author: tholl
"""
#########################################################################
# Import required packages
#########################################################################
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import itertools
import time
import math
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from prophet.diagnostics import cross_validation, performance_metrics    
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
import pywt
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import(
                                    AbstractHolidayCalendar, Holiday, DateOffset, \
                                    SU, MO, TU, WE, TH, FR, SA, \
                                    next_monday, nearest_workday, sunday_to_monday,
                                    EasterMonday, GoodFriday, Easter
                                  )

#########################################################################
# set/modify standard page configuration
#########################################################################
st.set_page_config(page_title="ForecastGenie", 
                   layout="centered", # "centered" or "wide"
                   page_icon="🌀", 
                   initial_sidebar_state="expanded") # "auto" or "expanded" or "collapsed"

###############################################################################
# SET VARIABLES
###############################################################################
# create an empty dictionary to store the results of the models
# that I call after I train the models to display on sidebar under hedaer "Evaluate Models"
metrics_dict = {}

# define calendar
cal = calendar()

# define an empty dataframe
df_raw = pd.DataFrame()

# set the title of page
st.title(":blue[]")
st.markdown(f'<h1 style="color:#45B8AC;"> <center> ForecastGenie™️ </center> </h1>', unsafe_allow_html=True)
# add vertical spacing
st.write("")

# define tabs of data pipeline for user to browse through
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["Load🚀", "Explore🕵️‍♂️", "Clean🧹", "Engineer🧰", "Prepare🧪", "Select🍏", "Train🔢", "Evaluate🎯", "Tune⚙️", "Forecast🔮"])

# Create a global pandas DataFrame to hold model_name and mape values
#results_df = pd.DataFrame(columns=['model_name', 'mape', 'rmse', 'r2', 'features', 'model settings'])
# Initialize results_df in global scope
results_df = pd.DataFrame(columns=['model_name', 'mape', 'rmse', 'r2', 'features', 'model settings'])

if 'results_df' not in st.session_state:
    st.session_state['results_df'] = pd.DataFrame(columns=['model_name', 'mape', 'rmse', 'r2', 'features', 'model settings'])

# Log
print('ForecastGenie Print: Loaded Global Variables')
###############################################################################
# FUNCTIONS
###############################################################################
def display_summary_statistics(df):
    summary = pd.DataFrame()
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            summary[col] = [df[col].min(), df[col].max(), '-', '-', '-', 'datetime']
        elif df[col].dtype != 'object':
            summary[col] = [df[col].min().round(2), df[col].max().round(2), 
                            df[col].mean().round(2), df[col].median().round(2), 
                            df[col].std().round(2), df[col].dtype]
        else:
            summary[col] = [np.nan, np.nan, np.nan, np.nan, np.nan, df[col].dtype]
    summary = summary.transpose()
    summary.columns = ['Min', 'Max', 'Mean', 'Median', 'Std', 'dtype']
    return summary

def display_dataframe_graph(df, key=0):
    fig = px.line(df,
                  x=df.index,
                  y=df.columns,
                  #labels=dict(x="Date", y="y"),
                  title='')
    # Set Plotly configuration options
    fig.update_layout(width=800, height=400, xaxis=dict(title='Date'), yaxis=dict(title='', rangemode='tozero'), legend=dict(x=0.9, y=0.9))
    # set line color and width
    fig.update_traces(line=dict(color='#45B8AC', width=2))


    # Add the range slider to the layout
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label='1w', step='day', stepmode='backward'),
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all')
                ]),
                x=0.3,
                y=1.2,
                yanchor='top',
                font=dict(size=10),
            ),
            rangeslider=dict(
                visible=True,
                range=[df.index.min(), df.index.max()]  # Set range of slider based on data
            ),
            type='date'
        )
    )
    # Display Plotly Express figure in Streamlit
    st.plotly_chart(fig, use_container_width=True, key=key)
    
# define function to generate demo time-series data
def generate_demo_data(seed=42):
    np.random.seed(seed)
    date_range = pd.date_range(start='1/1/2022', end='12/31/2022', freq='D')
    # generate seasonal pattern
    t = np.linspace(0, 1, len(date_range))
    seasonal = 10 * np.sin(2 * np.pi * (t + 0.25)) + 5 * np.sin(2 * np.pi * (2 * t + 0.1))
    
    # generate random variation
    noise = np.random.normal(loc=0, scale=2, size=len(date_range))
    
    # generate time series data
    values = seasonal + noise
    demo_data = pd.DataFrame({'date': date_range, 'demo_data': values})
    return demo_data

# add wavelet features if applicable e.g. user selected it to be included
def forecast_wavelet_features(X, features_df_wavelet, future_dates, df_future_dates):
    ########### START WAVELET CODE ###########  
    # if user selected checkbox for Discrete Wavelet Features run code run prediction for wavelet features
    if select_dwt_features: 
        try:
            # only extract wavelet features
            X_wavelet_features = X.loc[:, features_df_wavelet.columns.intersection(X.columns)]
            X_future_wavelet = pd.DataFrame(index=future_dates, columns=X_wavelet_features.columns)
            # iterate over all wavelet feature column names
            for col in X_wavelet_features.columns:
                # define the length of the dataframe
                n = len(X_wavelet_features)
                # just use the date range as integers as independent feature (1 through n)
                X_wavelet = np.arange(1, n+1).reshape(-1, 1)
                # define target column e.g. each wavelet feature independently is forecasted into future to user defined date
                y_wavelet = X_wavelet_features[col]
                # define forecasting model user can chose for predicting the wavelet features for forecast date range
                if model_type_wavelet == 'Support Vector Regression':
                    model_wavelet = SVR()
                else:
                    model_wavelet = LinearRegression()  
                # fit the model on the historical data
                model_wavelet.fit(X_wavelet, y_wavelet)
                # predict future
                prediction_wavelet = model_wavelet.predict(np.arange(n+1, n+len(future_dates)+1).reshape(-1, 1))
                X_future_wavelet[col] = prediction_wavelet
            # reset the index and rename the datetimestamp column to 'date' - which is now a column that can be used to merge dataframes
            X_future_wavelet = X_future_wavelet.reset_index().rename(columns={'index': 'date'})

            # combine the independent features with the forecast dataframe
            df_future_dates = pd.merge(df_future_dates, X_future_wavelet, on='date', how='left' )
            return df_future_dates
        except:
            st.warning('Error: Discrete Wavelet Features are not created correctly, please remove from selection criteria')
            
def create_rfe_plot(df_ranking):
    """
    Create a scatter plot of feature rankings and selected features.

    Parameters:
        df_ranking (pandas.DataFrame): A DataFrame with feature rankings and selected features.

    Returns:
        None
    """
    fig = px.scatter(df_ranking, x='Features', y='Ranking', color='Selected', hover_data=['Ranking'])
    fig.update_layout(
        title={
            'text': 'Recursive Feature Elimination with Cross-Validation (RFECV)',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Features',
        yaxis_title='Ranking',
        legend_title='Selected',
        xaxis_tickangle=-45 # set the tickangle to x degrees
    )
    return fig

def rfe_cv(X_train, y_train, est_rfe, num_steps_rfe, num_features, timeseriessplit_value_rfe):
    """
    Perform Recursive Feature Elimination with Cross-Validation and display the results using a scatter plot.

    Parameters:
        X_train (pandas.DataFrame): Training data features.
        y_train (pandas.Series): Training data target.
        est_rfe (estimator object): A supervised learning estimator with a `fit` method.
        num_steps_rfe (int): Number of features to remove at each iteration of RFE.
        num_features (int): Number of features to select, defaults to None.
        timeseriessplit_value_rfe (int): Number of splits in time series cross-validation.

    Returns:
        None
    """
    #############################################################
    # Recursive Feature Elemination
    #############################################################
    # define the time series splits set by user in sidebar slider      
    tscv = TimeSeriesSplit(n_splits=timeseriessplit_value_rfe)
    # Set up the recursive feature elimination with cross validation
    rfecv = RFECV(estimator=est_rfe, step=num_steps_rfe, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    # Fit the feature selection model
    rfecv.fit(X_train, y_train)
    # Define the selected features
    if num_features is not None:
        selected_features = X_train.columns[rfecv.ranking_ <= num_features]
    else:
        selected_features = X_train.columns[rfecv.support_]
    # Get the feature ranking
    feature_rankings = pd.Series(rfecv.ranking_, index=X_train.columns).rename('Ranking')
    # Sort the feature rankings in descending order
    sorted_rankings = feature_rankings.sort_values(ascending=True)
    # Create a dataframe with feature rankings and selected features
    df_ranking = pd.DataFrame({'Features': sorted_rankings.index, 'Ranking': sorted_rankings})
    # Sort the dataframe by ranking
    df_ranking = df_ranking.sort_values('Ranking', ascending=True)
    # Highlight selected features
    df_ranking['Selected'] = np.where(df_ranking['Features'].isin(selected_features), 'Yes', 'No')
    #############################################################
    # Scatterplot the results
    #############################################################
    # Create the rfe plot
    fig = create_rfe_plot(df_ranking)
    # Show the plot
    st.plotly_chart(fig, use_container_width=True)    
                    
    # show the ranking and selected features dataframes side by side
    col1, col2, col3, col4 = st.columns([1,2,2,1])
    with col2:
        st.write(':blue[**Selected features:**]', selected_features)
    # Print the feature rankings
    with col3: 
        feature_rankings = pd.Series(rfecv.ranking_, index=X.columns).rename('Ranking')
        st.write(':blue[**Feature rankings:**]')
        st.write(feature_rankings.sort_values())
    #############################################################
    # Show in streamlit the ranking        
    #############################################################              
    # show user selected columns
    selected_cols_rfe = list(selected_features)
    st.info(f'Top {len(selected_cols_rfe)} features selected with RFECV: {selected_cols_rfe}')
    
    show_rfe_info_btn = st.button(f'About RFE plot', use_container_width=True, type='secondary')
    # if user clicks "Submit" button for recursive feature elimination run below
    if show_rfe_info_btn:
        st.write('')
        # show user info about how to interpret the graph
        st.markdown('''**Recursive Feature Elimination** involves recursively removing features and building a model on the remaining features. It then **ranks the features** based on their importance and **eliminates** the **least important feature**.
                    ''')
    return selected_cols_rfe

def plot_scaling_before_after(X_unscaled_train, X_train, numerical_features):
    """
    Plots a figure showing the unscaled and scaled data for each numerical feature.

    Parameters:
    -----------
    X_unscaled_train : pandas.DataFrame
        The unscaled training data.
    X_train : pandas.DataFrame
        The scaled training data.
    numerical_features : list
        The list of numerical feature names.

    Returns:
    --------
    None.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    # create trace for unscaled data for each feature
    for feature in numerical_features:
        trace_unscaled = go.Scatter(
            x=X_unscaled_train.index,
            y=X_unscaled_train[feature],
            mode='lines',
            name=feature
        )
        fig.add_trace(trace_unscaled, row=1, col=1)

    # create trace for scaled data for each feature
    for feature in numerical_features:
        trace_scaled = go.Scatter(
            x=X_train.index,
            y=X_train[feature],
            mode='lines',
            name=f'{feature} (scaled)'
        )
        fig.add_trace(trace_scaled, row=2, col=1)

    # add selection event handler
    updatemenu = []
    buttons = []
    buttons.append(dict(label='All',
                        method='update',
                        args=[{'visible': [True] * len(fig.data)}]))
    
    for feature in numerical_features:
        button = dict(
                        label=feature,
                        method="update",
                        args=[{"visible": [False] * len(fig.data)},
                              {"title": f"Feature: {feature}"}],
                    )
        button['args'][0]['visible'][numerical_features.index(feature)] = True
        button['args'][0]['visible'][numerical_features.index(feature) + len(numerical_features)] = True
        buttons.append(button)

    updatemenu.append(dict(buttons=buttons, 
                           direction="down", 
                           showactive=True, 
                           pad={"r": 5, "t": 5},
                           x=0.5, 
                           xanchor="center", 
                           y=1.2, 
                           yanchor="top"
                           )
                      )
    fig.update_layout(updatemenus=updatemenu, 
                      title="Feature: " + numerical_features[0])
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Unscaled", row=1, col=1, fixedrange=True)
    fig.update_yaxes(title_text="Scaled", row=2, col=1, fixedrange=True)
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.1,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12, family="Arial")
        ),
        updatemenus=updatemenu,
        title={
            'text': '',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    # show the figure in streamlit app
    st.plotly_chart(fig, use_container_width=True)

# Define chart titles with subtitle and font properties
def chart_title(title, subtitle):
    return {
            "text": title,
            "subtitle": subtitle,
            "fontSize": title_font_size,
            "font": title_font,
            "anchor": "middle"
            }

def plot_train_test_split(local_df, split_index):
    """
    Plot the train-test split of the given dataframe.
    
    Parameters:
    -----------
    local_df: pandas.DataFrame
        A pandas dataframe containing the time series data.
    split_index: int
        The index position that represents the split between the training and test set.
    
    Returns:
    --------
    fig: plotly.graph_objs._figure.Figure
        A plotly Figure object containing the train-test split plot.
    """
    # Get the absolute maximum value of the data
    max_value = abs(local_df.iloc[:, 0]).max()
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=local_df.index[:split_index],
                   y=local_df.iloc[:split_index, 0],
                   mode='lines',
                   name='Train',
                   line=dict(color='#217CD0')))
    fig.add_trace(
        go.Scatter(x=local_df.index[split_index:],
                   y=local_df.iloc[split_index:, 0],
                   mode='lines',
                   name='Test',
                   line=dict(color='#FFA500')))
    fig.update_layout(title='',
                      yaxis=dict(range=[-max_value*1.1, max_value*1.1]), # Set y-axis range to include positive and negative values
                      shapes=[dict(type='line',
                                    x0=local_df.index[split_index],
                                    y0=-max_value*1.1, # Set y0 to -max_value*1.1
                                    x1=local_df.index[split_index],
                                    y1=max_value*1.1, # Set y1 to max_value*1.1
                                    line=dict(color='grey',
                                              dash='dash'))],
                      annotations=[dict(x=local_df.index[split_index],
                                        y=max_value*1.05,
                                        xref='x',
                                        yref='y',
                                        text='Train/Test<br>Split',
                                        showarrow=True,
                                        font=dict(color="grey", size=15),
                                        arrowhead=1,
                                        ax=0,
                                        ay=-40)])
    split_date = local_df.index[split_index-1]
    fig.add_annotation(x=split_date,
                       y=0.95*max_value,
                       text=str(split_date.date()),
                       showarrow=False,
                       font=dict(color="grey", size=15))
    return fig

def perform_train_test_split(df, my_insample_forecast_steps, scaler_choice=None, numerical_features=[]):
    """
    Splits a given dataset into training and testing sets based on a user-specified test-set size.

    Args:
        df (pandas.DataFrame): A Pandas DataFrame containing the dataset to be split.
        my_insample_forecast_steps (int): An integer representing the number of rows to allocate for the test-set.
        scaler_choice (str): A string representing the type of scaler to be used. The default is None, which means no
        scaling is performed.
        numerical_features (list): A list of column names representing the numerical features to be scaled.

    Returns:
        tuple: A tuple containing the training and testing sets for both the features (X) and target variable (y),
            as well as the index of the split.

    Raises:
        ValueError: If the specified test-set size is greater than or equal to the total number of rows in the dataset.
    """
    # Check if the specified test-set size is valid
    if my_insample_forecast_steps >= len(df):
        raise ValueError("Test-set size must be less than the total number of rows in the dataset.")

    # Split the data into training and testing sets
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0:1]

    X_train = X.iloc[:-my_insample_forecast_steps, :]
    X_test = X.iloc[-my_insample_forecast_steps:, :]

    y_train = y.iloc[:-my_insample_forecast_steps, :]
    y_test = y.iloc[-my_insample_forecast_steps:, :]
    
    # initialize variable
    scaler = ""
    # Scale the data if user selected a scaler choice in the normalization / standardization in streamlit sidebar
    if scaler_choice != "None":
        # Check if there are numerical features in the dataframe
        if numerical_features:
            # Select only the numerical features to be scaled
            X_train_numeric = X_train[numerical_features]
            X_test_numeric = X_test[numerical_features]
            X_numeric = X[numerical_features]

            if scaler_choice == "MinMaxScaler":
                scaler = MinMaxScaler()
            elif scaler_choice == "RobustScaler":
                scaler = RobustScaler()
            elif scaler_choice == "MaxAbsScaler":
                scaler = MaxAbsScaler()
            elif scaler_choice == "PowerTransformer":
                scaler = PowerTransformer()
            elif scaler_choice == "QuantileTransformer":
                scaler = QuantileTransformer(n_quantiles=100, output_distribution="normal")
            else:
                raise ValueError("Invalid scaler choice. Please choose from: MinMaxScaler, RobustScaler, MaxAbsScaler, "
                                 "PowerTransformer, QuantileTransformer")

            # Fit the scaler on the training set and transform both the training and test sets
            X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
            X_train_numeric_scaled = pd.DataFrame(X_train_numeric_scaled, columns=X_train_numeric.columns, index=X_train_numeric.index)
            X_test_numeric_scaled = scaler.transform(X_test_numeric)
            X_test_numeric_scaled = pd.DataFrame(X_test_numeric_scaled, columns=X_test_numeric.columns, index=X_test_numeric.index)
            # refit the scaler on the entire exogenous features e.g. X which is used for forecasting beyond train/test sets
            X_numeric_scaled = scaler.fit_transform(X_numeric)
            # Convert the scaled array back to a DataFrame and set the column names
            X_numeric_scaled = pd.DataFrame(X_numeric_scaled, columns=X_numeric.columns, index=X_numeric.index)

    # Replace the original
    if scaler_choice != "None":
        X_train[numerical_features] = X_train_numeric_scaled[numerical_features]
        X_test[numerical_features] = X_test_numeric_scaled[numerical_features]
        X[numerical_features] = X_numeric_scaled[numerical_features]
                
    # Return the training and testing sets as well as the scaler used (if any)
    return X, y, X_train, X_test, y_train, y_test, scaler


def perform_train_test_split_standardization(X, y, X_train, X_test, y_train, y_test, my_insample_forecast_steps, scaler_choice=None, numerical_features=[]):
    """
    Splits a given dataset into training and testing sets based on a user-specified test-set size.

    Args:
        df (pandas.DataFrame): A Pandas DataFrame containing the dataset to be split.
        my_insample_forecast_steps (int): An integer representing the number of rows to allocate for the test-set.
        scaler_choice (str): A string representing the type of scaler to be used. The default is None, which means no
        scaling is performed.
        numerical_features (list): A list of column names representing the numerical features to be scaled.

    Returns:
        tuple: A tuple containing the training and testing sets for both the features (X) and target variable (y),
            as well as the index of the split.

    Raises:
        ValueError: If the specified test-set size is greater than or equal to the total number of rows in the dataset.
    """
    # Check if the specified test-set size is valid
    if my_insample_forecast_steps >= len(df):
        raise ValueError("Test-set size must be less than the total number of rows in the dataset.")
    if scaler_choice != "None":
        # Check if there are numerical features in the dataframe
        if numerical_features:
            # Select only the numerical features to be scaled
            X_train_numeric = X_train[numerical_features]
            X_test_numeric = X_test[numerical_features]
            X_numeric = X[numerical_features]
    
            if scaler_choice == "StandardScaler":
                scaler=StandardScaler()
            else:
                raise ValueError("Invalid scaler choice. Please choose from: StandardScaler")
    
            # Fit the scaler on the training set and transform both the training and test sets
            X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
            X_train_numeric_scaled = pd.DataFrame(X_train_numeric_scaled, columns=X_train_numeric.columns, index=X_train_numeric.index)
            X_test_numeric_scaled = scaler.transform(X_test_numeric)
            X_test_numeric_scaled = pd.DataFrame(X_test_numeric_scaled, columns=X_test_numeric.columns, index=X_test_numeric.index)
            # refit the scaler on the entire exogenous features e.g. X which is used for forecasting beyond train/test sets
            X_numeric_scaled = scaler.fit_transform(X_numeric)
            # Convert the scaled array back to a DataFrame and set the column names
            X_numeric_scaled = pd.DataFrame(X_numeric_scaled, columns=X_numeric.columns, index=X_numeric.index)
    
    # Replace the original
    if scaler_choice != "None":
        X_train[numerical_features] = X_train_numeric_scaled[numerical_features]
        X_test[numerical_features] = X_test_numeric_scaled[numerical_features]
        X[numerical_features] = X_numeric_scaled[numerical_features]
                
    # Return the training and testing sets as well as the scaler used (if any)
    return X, y, X_train, X_test, y_train, y_test

def train_test_split_slider():
    """
   Creates a slider for the user to choose the number of days or percentage for train/test split.

   Returns:
       tuple: A tuple containing the number of days and percentage for the in-sample forecast.
   """
    with st.sidebar:
        with st.form('train test split'):
            my_subheader('✂️ Train Test Split', my_style="#FF9F00", my_size=3)
            col1, col2 = st.columns(2)
            with col1:
                split_type = st.radio("*Select split type:*", ("Days", "Percentage"), index=1)
                if split_type == "Days":
                    with col2:
                        in_sample_forecast_steps = st.slider('*Size of the test-set in days:*', min_value=1, max_value=len(df)-1, value=int(len(df)*0.2), step=1, key='days')
                        in_sample_forecast_perc = round((in_sample_forecast_steps / len(df)) * 100, 2)
                else:
                    with col2:
                        in_sample_forecast_perc = st.slider('*Size of the test-set as percentage*', min_value=1, max_value=100, value=20, step=1, key='percentage')
                        in_sample_forecast_steps = round((in_sample_forecast_perc / 100) * len(df))
            col1, col2, col3 = st.columns([4,4,4])
            with col2:       
                train_test_split_btn = st.form_submit_button("Submit", type="secondary")        
    return in_sample_forecast_steps, in_sample_forecast_perc

def correlation_heatmap(X, correlation_threshold=0.8):
    """
    Description:
        This function generates a correlation heatmap for a given pandas DataFrame X, 
        showing only features that have a correlation higher or equal to the given correlation_threshold.
    Parameters:
        X : pandas DataFrame object
            The input dataset for which the correlation heatmap is to be generated.
        correlation_threshold : float (default=0.8)
            The correlation threshold value that will determine which features are shown in the heatmap. 
            Only features with a correlation higher or equal to the given threshold will be displayed in the heatmap.
    Returns:
        None
    """
    # create a new dataframe with only features that have a correlation higher or equal to the threshold
    corr_matrix = X.corr()
    features_to_keep = corr_matrix[abs(corr_matrix) >= corr_threshold].stack().reset_index().iloc[:, [0, 1]]
    features_to_keep.columns = ['feature1', 'feature2']
    features_to_keep = features_to_keep[features_to_keep['feature1'] != features_to_keep['feature2']]
    X_corr = X[features_to_keep['feature1'].unique()]
    # compute correlation matrix
    corr_matrix = X_corr.corr()
    # create heatmap using plotly express
    fig = px.imshow(corr_matrix.values,
                    color_continuous_scale="Blues",
                    zmin=-1,
                    zmax=1,
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    origin='lower')
    # add text annotations to heatmap cells
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                fig.add_annotation(x=i, y=j, text="{:.2f}".format(corr_matrix.iloc[i, j]), font=dict(color='white'))
    # add colorbar title
    fig.update_coloraxes(colorbar_title="Correlation")
    # set x and y axis labels to diagonal orientation
    fig.update_xaxes(tickangle=-45, showticklabels=True)
    fig.update_yaxes(tickangle=0, showticklabels=True)
    # adjust heatmap size and margins
    fig.update_layout(width=800,
                      height=800,
                      margin=dict(l=200, r=200, t=100, b=100))
    # show plotly figure in streamlit
    st.plotly_chart(fig, use_container_width=True)
    
def compute_importance_scores(X, y, estimator):
    """
    Compute feature importance scores using permutation importance.

    Parameters:
    -----------
    X : pandas.DataFrame
        The feature matrix.
    y : pandas.Series
        The target variable.
    estimator : object
        The model used to fit the data.

    Returns:
    --------
    importance_scores : dict
        A dictionary containing the importance scores for each feature.
    """
    # Compute feature importance scores using permutation importance
    # LinearRegression
    result = permutation_importance(estimator, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    # Create dictionary of importance scores
    importance_scores = {}
    for i in range(len(X.columns)):
        importance_scores[X.columns[i]] = result.importances_mean[i]
    return importance_scores

def get_feature_list(X):
    """
    Returns a comma-separated string of column names in a Pandas DataFrame.
    
    Parameters:
    X (Pandas DataFrame): the input DataFrame to get the column names from.
    
    Returns:
    A comma-separated string of column names in the input DataFrame.
    
    Example:
    >>> X = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    >>> get_feature_list(X)
    'col1, col2'
    """
    return ', '.join(X.columns)

def plot_forecast(df_actual, df_forecast, title=''):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_actual.index, y=df_actual.iloc[:,0], name='Actual', mode='lines'))
    fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast['forecast'], name='Forecast', mode='lines', line=dict(dash='dot', color='#87CEEB'))) # dash styles: ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'] 
    
    # add buttons for showing actual data, forecast data, and both actual and forecast data
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                xanchor="center",
                yanchor="middle",
                x=0.5,
                y=1.15,
                buttons=list([
                    dict(label="Actual",
                         method="update",
                         args=[{"visible": [True, False]},
                               {}]),
                    dict(label="Forecast",
                         method="update",
                         args=[{"visible": [False, True]},
                               {}]),
                    dict(label="Actual + Forecast",
                         method="update",
                         args=[{"visible": [True, True]},
                               {}])
                ]),
            )
        ],
        title=title, 
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

def create_date_features(df, year_dummies=True, month_dummies=True, day_dummies=True):
    '''
    This function creates dummy variables for year, month, and day of week from a date column in a pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): Input DataFrame containing a date column.
    year_dummies (bool): Flag to indicate if year dummy variables are needed. Default is True.
    month_dummies (bool): Flag to indicate if month dummy variables are needed. Default is True.
    day_dummies (bool): Flag to indicate if day of week dummy variables are needed. Default is True.

    Returns:
    pandas.DataFrame: A new DataFrame with added dummy variables.
    '''
    if year_dummies:
        df['year'] = df['date'].dt.year
        dum_year = pd.get_dummies(df['year'], columns=['year'], drop_first=True, prefix='year', prefix_sep='_')
        df = pd.concat([df, dum_year], axis=1)

    if month_dummies:
        month_dict = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
        df['month'] = df['date'].dt.month.apply(lambda x: month_dict.get(x))
        dum_month = pd.get_dummies(df['month'], columns=['month'], drop_first=True, prefix='', prefix_sep='')
        df = pd.concat([df, dum_month], axis=1)

    if day_dummies:
        week_dict = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
        df['day'] = df['date'].dt.weekday.apply(lambda x: week_dict.get(x))
        dum_day = pd.get_dummies(df['day'], columns=['day'], drop_first=True, prefix='', prefix_sep='')
        df = pd.concat([df, dum_day], axis=1)
    # Drop any duplicate columns based on their column names
    df = df.loc[:,~df.columns.duplicated()]
    return df

def create_calendar_special_days(df, start_date_calendar=None,  end_date_calendar=None, select_all_days=True):
    """
    # source: https://practicaldatascience.co.uk/data-science/how-to-create-an-ecommerce-trading-calendar-using-pandas
    Create a trading calendar for an ecommerce business in the UK.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame containing order data with index
    select_all_days (bool): Whether to select all days or only specific special days
    
    Returns:
    df_exogenous_vars (pd.DataFrame): DataFrame containing trading calendar with holiday and event columns
    """
    ###############################################
    # Define variables
    ###############################################
    if start_date_calendar == None:
        start_date_calendar = df['date'].min()
    else:
        start_date_calendar = start_date_calendar
    if end_date_calendar == None:
        end_date_calendar = df['date'].max()
    else:
        end_date_calendar = end_date_calendar
    df_exogenous_vars = pd.DataFrame({'date': pd.date_range(start = start_date_calendar, end = end_date_calendar)})
    holidays = cal.holidays(start=start_date_calendar, end=end_date_calendar)
    # holiday = true/ otherwise false
    df_exogenous_vars['holiday'] = (df_exogenous_vars['date'].isin(holidays)).apply(lambda x: 1 if x==True else 0)
    # create column for holiday description
    df_exogenous_vars['holiday_desc'] = df_exogenous_vars['date'].apply(lambda x: my_holiday_name_func(x))
    
    class UKEcommerceTradingCalendar(AbstractHolidayCalendar):
        rules = []
        # Seasonal trading events
        # only add Holiday if user checkmarked checkbox (e.g. equals True) 
        if jan_sales:
            rules.append(Holiday('January sale', month = 1, day = 1))
        if val_day_lod:
            rules.append(Holiday('Valentine\'s Day [last order date]', month = 2, day = 14, offset = BDay(-2)))
        if val_day:
            rules.append(Holiday('Valentine\'s Day', month = 2, day = 14))
        if mother_day_lod:
            rules.append(Holiday('Mother\'s Day [last order date]', month = 5, day = 1, offset = BDay(-2)))
        if mother_day:
            rules.append(Holiday('Mother\'s Day', month = 5, day = 1, offset = pd.DateOffset(weekday = SU(2))))
        if father_day_lod:
            rules.append(Holiday('Father\'s Day [last order date]', month = 6, day = 1, offset = BDay(-2)))
        if father_day:
            rules.append(Holiday('Father\'s Day', month = 6, day = 1, offset = pd.DateOffset(weekday = SU(3))))
        if black_friday_lod:
            rules.append(Holiday("Black Friday [sale starts]", month = 11, day = 1, offset = [pd.DateOffset(weekday = SA(4)), BDay(-5)]))
        if black_friday:
            rules.append(Holiday('Black Friday', month = 11, day = 1, offset = pd.DateOffset(weekday = FR(4))))
        if cyber_monday:
            rules.append(Holiday("Cyber Monday", month = 11, day = 1, offset = [pd.DateOffset(weekday = SA(4)), pd.DateOffset(2)]))
        if christmas_day:
            rules.append(Holiday('Christmas Day [last order date]', month = 12, day = 25, offset = BDay(-2)))
        if boxing_day:
            rules.append(Holiday('Boxing Day sale', month = 12, day = 26))       
    calendar = UKEcommerceTradingCalendar()
    start = df_exogenous_vars.date.min()
    end = df_exogenous_vars.date.max()
    events = calendar.holidays(start = start, end = end, return_name = True)
    events = events.reset_index(name = 'calendar_event_desc').rename(columns = {'index': 'date'})
    df_exogenous_vars = df_exogenous_vars.merge(events, on = 'date', how = 'left').fillna('')
    # source: https://splunktool.com/holiday-calendar-in-pandas-dataframe
    ###############################################
    # Create Pay Days 
    ###############################################
    class UKEcommerceTradingCalendar(AbstractHolidayCalendar):
        rules = []
        # Pay days(based on fourth Friday of the month)
        if pay_days == True:
            rules = [
                    Holiday('January Pay Day', month = 1, day = 31, offset = BDay(-1)),
                    Holiday('February Pay Day', month = 2, day = 28, offset = BDay(-1)),
                    Holiday('March Pay Day', month = 3, day = 31, offset = BDay(-1)),
                    Holiday('April Pay Day', month = 4, day = 30, offset = BDay(-1)),
                    Holiday('May Pay Day', month = 5, day = 31, offset = BDay(-1)),
                    Holiday('June Pay Day', month = 6, day = 30, offset = BDay(-1)),
                    Holiday('July Pay Day', month = 7, day = 31, offset = BDay(-1)),
                    Holiday('August Pay Day', month = 8, day = 31, offset = BDay(-1)),
                    Holiday('September Pay Day', month = 9, day = 30, offset = BDay(-1)),
                    Holiday('October Pay Day', month = 10, day = 31, offset = BDay(-1)),
                    Holiday('November Pay Day', month = 11, day = 30, offset = BDay(-1)),
                    Holiday('December Pay Day', month = 12, day = 31, offset = BDay(-1))
                    ]
    calendar = UKEcommerceTradingCalendar()
    start = df_exogenous_vars.date.min()
    end = df_exogenous_vars.date.max()
    events = calendar.holidays(start = start, end = end, return_name = True)
    events = events.reset_index(name = 'pay_day_desc').rename(columns = {'index': 'date'})
    df_exogenous_vars = df_exogenous_vars.merge(events, on = 'date', how = 'left').fillna('')
    df_exogenous_vars['pay_day'] = df_exogenous_vars['pay_day_desc'].apply(lambda x: 1 if len(x) > 1 else 0)
    # add boolean
    df_exogenous_vars['calendar_event'] = df_exogenous_vars['calendar_event_desc'].apply(lambda x: 1 if len(x)>1 else 0)
    
    ###############################################################################
    # Reorder Columns to logical order e.g. value | description of value
    ###############################################################################
    # ??? improve this dynamically ???
    df_exogenous_vars = df_exogenous_vars[['date', 'holiday', 'holiday_desc', 'calendar_event', 'calendar_event_desc', 'pay_day','pay_day_desc']]

    ###############################################################################
    # combine exogenous vars with df_total | df_clean?
    ###############################################################################
    df_total_incl_exogenous = pd.merge(df, df_exogenous_vars, on='date', how='left' )
    df = df_total_incl_exogenous.copy(deep=True)
    return df 

def my_title(my_string, my_background_color="#45B8AC"):
    st.markdown(f'<h3 style="color:#FFFFFF; background-color:{my_background_color}; padding:5px; border-radius: 5px;"> <center> {my_string} </center> </h3>', unsafe_allow_html=True)

def my_header(my_string, my_style="#217CD0"):
    #st.markdown(f'<h2 style="text-align:center"> {my_string} </h2>', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color:{my_style};"> <center> {my_string} </center> </h2>', unsafe_allow_html=True)

def my_subheader(my_string, my_style="#217CD0", my_size=5):
    st.markdown(f'<h{my_size} style="color:{my_style};"> <center> {my_string} </center> </h{my_size}>', unsafe_allow_html=True)

def my_subheader_metric(string1, color1="#cfd7c2", metric=0, color2="#FF0000", my_style="#000000", my_size=5):
    metric_rounded = "{:.2%}".format(metric)
    metric_formatted = f"<span style='color:{color2}'>{metric_rounded}</span>"
    string1 = string1.replace(f"{metric}", f"<span style='color:{color1}'>{metric_rounded}</span>")
    st.markdown(f'<h{my_size} style="color:{my_style};"> <center> {string1} {metric_formatted} </center> </h{my_size}>', unsafe_allow_html=True)

def wait(seconds):
    start_time = time.time()
    with st.spinner(f"Please wait... {int(time.time() - start_time)} seconds passed"):
        time.sleep(seconds)     

@st.cache_data
def my_holiday_name_func(my_date):
    """
    This function takes a date as input and returns the name of the holiday that falls on that date.

    Parameters:
    -----------
    my_date : str
        The date for which the holiday name is to be returned. The date should be in the format 'YYYY-MM-DD'.

    Returns:
    --------
    str:
        The name of the holiday that falls on the given date. If there is no holiday on that date, an empty string is returned.

    Examples:
    ---------
    >>> my_holiday_name_func('2022-07-04')
    'Independence Day'
    >>> my_holiday_name_func('2022-12-25')
    'Christmas Day'
    >>> my_holiday_name_func('2022-09-05')
    'Labor Day'
    """    
    holiday_name = cal.holidays(start = my_date, end = my_date, return_name = True)
    if len(holiday_name) < 1:
      holiday_name = ""
      return holiday_name
    else: 
     return holiday_name[0]

def my_metrics(my_df, model_name):
   mape = my_df['MAPE'].mean()
   mse = mean_squared_error(my_df['Actual'], my_df['Predicted'])
   rmse = np.sqrt(mse)
   r2 = r2_score(my_df['Actual'], my_df['Predicted'])
   # add the results to the dictionary
   metrics_dict[model_name] = {'mape': mape, 'mse': mse, 'rmse': rmse, 'r2': r2}
   return mape, rmse, r2

def display_my_metrics(my_df, model_name=""):
    """
    Displays the evaluation metrics for a given model using the provided dataframe of predicted and actual values.
    
    Parameters:
    -----------
    my_df : pandas.DataFrame
        Dataframe containing columns 'Actual' and 'Predicted' for the actual and predicted values, respectively.
    model_name : str, default='Linear Regression'
        Name of the model being evaluated. This will be displayed as the title of the expander containing the metrics.
    
    Returns:
    --------
    None
    """
    # put all metrics and graph in expander for linear regression e.g. benchmark model
    st.markdown(f'<h2 style="text-align:center">{model_name}</h2></p>', unsafe_allow_html=True)
    # define vertical spacings
    col0, col1, col2, col3, col4 = st.columns([2, 3, 3, 3, 1])
    # Display the evaluation metrics
    mape, rmse, r2 = my_metrics(my_df, model_name)
    with col1:  
        st.metric(':red[**MAPE:**]', value = "{:.2%}".format(mape))
    with col2:
        st.metric(':red[**RMSE:**]', value = round(rmse,2))
    with col3: 
        st.metric(':green[**R-squared:**]',  value= round(r2, 2))

def evaluate_regression_model(model, X_train, y_train, X_test, y_test, **kwargs):
    """
    Evaluate a regression model on test data.

    Parameters:
        X_train (pd.DataFrame): Training input data.
        y_train (pd.DataFrame): Training output data.
        X_test (pd.DataFrame): Test input data.
        y_test (pd.DataFrame): Test output data.
        model_type (str): Type of regression model to use. Default is "linear".
        lag (str): Lag for seasonal naive model. Options are "day", "week", "month", and "year". Default is None.
        **kwargs: Optional keyword arguments to pass to the regression model.

    Returns:
        df_preds (pd.DataFrame): DataFrame of predicted and actual values on test data.
    """
    # this is for the baseline Naive Model to get a sense of how the model will perform for y_t-1 just having lag of itself 
    # e.g. a day, a week or a month
    st.write(kwargs)
    if 'lag' in kwargs and kwargs['lag'] is not None:
        lag = kwargs['lag']
        if lag == 'NULL':
            lag = None
        elif lag == 'day':
            y_pred = y_test.shift(1) # .fillna(method='bfill') # method{‘backfill’,‘ffill’, None}, default None
        elif lag == 'week':
            y_pred = y_test.shift(7) #.fillna(method='bfill')
        elif lag == 'month':
            y_pred = y_test.shift(30) #.fillna(method='bfill')
        elif lag == 'year':
            y_pred = y_test.shift(365) #.fillna(method='bfill')
        elif lag == 'custom':
            y_pred = y_test.shift(custom_lag_value)
        else:
            raise ValueError('Invalid value for "lag". Must be "day", "week", "month", or "year".')
    else:
        # Train the model using the training sets
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    # Create dataframe for insample predictions versus actual
    df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
    # set the index to just the date portion of the datetime index
    df_preds.index = df_preds.index.date
    # Drop rows with N/A values
    df_preds.dropna(inplace=True)
    # Calculate percentage difference between actual and predicted values and add it as a new column
    df_preds = df_preds.assign(Percentage_Diff = ((df_preds['Predicted'] - df_preds['Actual']) / df_preds['Actual']))
    # Calculate MAPE and add it as a new column
    df_preds = df_preds.assign(MAPE = abs(df_preds['Percentage_Diff']))   
    return df_preds

def evaluate_sarimax_model(order, seasonal_order, exog_train, exog_test, endog_train, endog_test):
    """
    Train and evaluate SARIMAX model on test data.
    
    Parameters:
        order (tuple): Order of the SARIMAX model (p,d,q).
        seasonal_order (tuple): Seasonal order of the SARIMAX model (P,D,Q,s).
        exog_train (pd.DataFrame): Exogenous variables training data.
        exog_test (pd.DataFrame): Exogenous variables test data.
        endog_train (pd.DataFrame): Endogenous variable training data.
        endog_test (pd.DataFrame): Endogenous variable test data.
    
    Returns:
        preds_df (pd.DataFrame): DataFrame of predicted and actual values on test data.
    """
    # =============================================================================
    #    try:
    # =============================================================================
    # Fit the model
    model = sm.tsa.statespace.SARIMAX(endog=endog_train, exog=exog_train, order=order, seasonal_order=seasonal_order)
    print('model')
    results = model.fit()
    print('fit model')
    # Generate predictions
    y_pred = results.predict(start=endog_test.index[0], end=endog_test.index[-1], exog=exog_test)
    print('define y_pred')
    preds_df = pd.DataFrame({'Actual': endog_test.squeeze(), 'Predicted': y_pred.squeeze()}, index=endog_test.index)
    print('preds_df')
    # Calculate percentage difference between actual and predicted values and add it as a new column
    preds_df = preds_df.assign(Percentage_Diff = ((preds_df['Predicted'] - preds_df['Actual']) / preds_df['Actual']))
    print(preds_df)
    # Calculate MAPE and add it as a new column
    preds_df = preds_df.assign(MAPE = abs(preds_df['Percentage_Diff']))   
    return preds_df 
    # =============================================================================
    #    except:
    #        my_warning = st.warning('SARIMAX model did not evaluate test-set correctly within function evaluate_sarimax_model, please contact an administrator!')
    #        return my_warning 
    # =============================================================================

def create_streamlit_model_card(X_train, y_train, X_test, y_test, results_df, model, model_name):
    """
    Creates a Streamlit expander card displaying metrics, a graph, and a dataframe for a given model.
    
    Parameters:
    -----------
    X_train: numpy.ndarray or pandas.DataFrame
        The training set inputs.
    y_train: numpy.ndarray or pandas.Series
        The training set outputs.
    X_test: numpy.ndarray or pandas.DataFrame
        The test set inputs.
    y_test: numpy.ndarray or pandas.Series
        The test set outputs.
    model: estimator object
        The trained machine learning model to evaluate.
    model_name: str, default="your model name"
        The name of the machine learning model being evaluated. This will be displayed in the expander header.
    **kwargs: optional keyword arguments
        Additional keyword arguments to be passed to the evaluate_regression_model() function.
        
    Returns:
    --------
    None
    """
    # Evaluate the insample test-set performance linear regression model
    df_preds = evaluate_regression_model(model, X_train, y_train, X_test, y_test)
    mape, rmse, r2 = my_metrics(df_preds, model_name)
    
    # add the results to sidebar for quick overview for user  
    # set as global variable to be used in code outside function
# =============================================================================
#     results_df = results_df.append({'model_name': model_name, 
#                                     'mape': '{:.2%}'.format(mape),
#                                     'rmse': rmse, 
#                                     'r2':r2, 
#                                     'features':X.columns.tolist(), 
#                                     'model settings': model
#                                     },
#                                     ignore_index=True)
# =============================================================================
# =============================================================================
#     results_df = pd.concat([results_df, pd.DataFrame({'model_name': [model_name],
#                                                        'mape': '{:.2%}'.format(mape),
#                                                        'rmse': rmse, 
#                                                        'r2':r2, 
#                                                        'model settings': model
#                                                      })],
#                                                        ignore_index=True)    
# =============================================================================
    with st.expander('ℹ️ '+ model_name, expanded=True):
        display_my_metrics(my_df=df_preds, model_name=model_name)
        # plot graph with actual versus insample predictions
        plot_actual_vs_predicted(df_preds, my_conf_interval)
        # show the dataframe
        st.dataframe(df_preds.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Percentage_Diff': '{:.2%}', 'MAPE': '{:.2%}'}), use_container_width=True)
        # create download button for forecast results to .csv
        download_csv_button(df_preds, my_file="insample_forecast_linear_regression_results.csv", help_message=f'Download your **{model_name}** model results to .CSV')

def preprocess_data_prophet(y_data):
    """
    Preprocess the given data for Prophet model
    """
    # create a deep copy of dataframe e.g. completely new copy of the DataFrame is created with its own memory space
    # This means that any changes made to the new copy will not affect the original DataFrame.
    y_data_prophet = y_data.copy(deep=True)
    # Check if the index is already a datetime index and revert it to a column if needed
    if isinstance(y_data_prophet.index, pd.DatetimeIndex):
        y_data_prophet.reset_index(inplace=True)
    # REQUIREMENT PROPHET: CREATE DATAFRAME WITH DATE COLUMN 'DS' AND 'Y' column
    y_data_prophet = pd.DataFrame({"ds": y_data_prophet.iloc[:, 0], "y": y_data_prophet.iloc[:, 1]})
    return y_data_prophet

def predict_prophet(y_train, y_test, **kwargs):
    """
    Predict using Prophet model
    """
    y_train_prophet = preprocess_data_prophet(y_train)
    y_test_prophet = preprocess_data_prophet(y_test)
    
    # get the parameters from the settings either preset or adjusted by user and user pressed submit button
    m = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                seasonality_mode=seasonality_mode,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                interval_width=interval_width)
    
    # train the model on the data with set parameters
    m.fit(y_train_prophet)
    # Predict on the test set
    future = m.make_future_dataframe(periods=len(y_test), freq='D')
    forecast = m.predict(future)
    # slice the test-set of the forecast - exclude the forecast on the training set although prophet model does supply it
    # Prophet model provides it to check for overfitting the model, however to prevent user from thinking it trained on whole dataset clearer to provide forecast of test set
    yhat_test = forecast['yhat'][-len(y_test):]
    preds_df_prophet = pd.DataFrame({'Actual': y_test_prophet['y'].values, 'Predicted': yhat_test.values}, index=y_test_prophet['ds'])
    # create column date and set the datetime index to date without the time i.e. 00:00:00
    preds_df_prophet['date'] = preds_df_prophet.index.strftime('%Y-%m-%d')
    # set the date column as index column
    preds_df_prophet = preds_df_prophet.set_index('date')
    # Calculate percentage difference between actual and predicted values and add it as a new column
    preds_df_prophet = preds_df_prophet.assign(Percentage_Diff = ((preds_df_prophet['Predicted'] - preds_df_prophet['Actual']) / preds_df_prophet['Actual']))
    # Calculate Mean Absolute Percentage Error (MAPE) and add it as a new column
    preds_df_prophet = preds_df_prophet.assign(MAPE = abs(preds_df_prophet['Percentage_Diff']))
    return preds_df_prophet

def load_data():
    """
    This function loads data from a CSV file and returns a pandas DataFrame object.
    """
    df_raw = pd.read_csv(uploaded_file, parse_dates=['date'])
    return df_raw

@st.cache_data   
def create_df_pred(y_test, y_pred, show_df_streamlit=True):
    try:
        df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
        # set the index to just the date portion of the datetime index
        df_preds.index = df_preds.index.date
        # Calculate percentage difference between actual and predicted values and add it as a new column
        df_preds = df_preds.assign(Percentage_Diff = ((df_preds['Predicted'] - df_preds['Actual']) / df_preds['Actual']))
        # Calculate MAPE and add it as a new column
        df_preds = df_preds.assign(MAPE = abs(df_preds['Percentage_Diff']))
        # show the predictions versus actual results
        my_df = df_preds.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Percentage_Diff': '{:.2%}'})
        if show_df_streamlit == True:
            return st.write(my_df)
        else:
           return df_preds
    except:
        st.warning('error in create_df_pred function')

def convert_df(my_dataframe):
    return my_dataframe.to_csv(index=False).encode('utf-8')

def convert_df_with_index(my_dataframe):
    return my_dataframe.to_csv(index=True).encode('utf-8')

def download_csv_button(my_df, my_file="forecast_model.csv", help_message = 'Download dataframe to .CSV', set_index=False):
     # create download button for forecast results to .csv
     if set_index:
         csv = convert_df_with_index(my_df)
     else:
         csv = convert_df(my_df)
     col1, col2, col3 = st.columns([5,3,5])
     with col2: 
         st.download_button(":arrow_down: Download", 
                            csv,
                            my_file,
                            "text/csv",
                            #key='', -> streamlit automatically assigns key if not defined
                            help = help_message)

def plot_actual_vs_predicted(df_preds, my_conf_interval):
    # Define the color palette
    colors = ['#5276A7', '#ff7700']
    # set color shading of confidence interval
    my_fillcolor = 'rgba(222,235,247,0.5)'

    # Create the figure with easy on eyes colors
    fig = px.line(df_preds, x=df_preds.index, y=['Actual', 'Predicted'], color_discrete_sequence=colors)
    # Update the layout of the figure
    fig.update_layout(
                        legend_title = 'Legend',
                        font = dict(family='Arial', 
                                    size=12, 
                                    color='#707070'
                                    ),
                        yaxis = dict(
                                    gridcolor='#E1E1E1',
                                    range = [np.minimum(df_preds['Actual'].min(), df_preds['Predicted'].min()) - (df_preds['Predicted'].max() - df_preds['Predicted'].min()), 
                                    np.maximum(df_preds['Actual'].max(), df_preds['Predicted'].max()) + (df_preds['Predicted'].max() - df_preds['Predicted'].min())],
                                    zeroline=False, # remove the x-axis line at y=0
                                    ),
                        xaxis=dict(gridcolor='#E1E1E1'),
                        legend=dict(yanchor="bottom", y=0.0, xanchor="center", x=0.99,  bgcolor= 'rgba(0,0,0,0)' )   
                    )
    # Set the line colors
    for i, color in enumerate(colors):
        fig.data[i].line.color = color
        if fig.data[i].name == 'Predicted':
            fig.data[i].line.dash = 'dot' # dash styles options: ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
    # Compute the level of confidence 
    confidence = float(my_conf_interval/100)
    # level of significance (1 minus confidence interval)
    alpha = 1 - confidence
    n = len(df_preds)
    # Calculates the t-value based on the desired confidence level (alpha) and the sample size (n).
    t_value = stats.t.ppf(1 - alpha / 2, n - 2)
    # Calculates the standard error of the estimate, which measures the amount of variability in the data that is not explained by the model.
    std_error = np.sqrt(np.sum((df_preds['Actual'] - df_preds['Predicted']) ** 2) / (n - 2))
    # Calculates the margin of error for the confidence interval based on the t-value and the standard error.
    margin_error = t_value * std_error
    upper = df_preds['Predicted'] + margin_error
    lower = df_preds['Predicted'] - margin_error
    # Add shaded confidence interval to the plot
    fig.add_trace(go.Scatter(
        x=df_preds.index,
        y=upper,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        fillcolor=my_fillcolor,
        name='Upper Confidence Interval',
        showlegend=False,
        legendgroup='confidence intervals'
    ))
    fig.add_trace(go.Scatter(
        x=df_preds.index,
        y=lower,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        fillcolor=my_fillcolor,
        name=f'{int(my_conf_interval)}% Confidence Interval',
        legendgroup='confidence intervals'
    ))
    
    # Render the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
@st.cache_data   
# remove datatypes object - e.g. descriptive columns not used in modeling
def remove_object_columns(df):
    # Get a list of column names with object datatype
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    # Remove the columns that are not needed
    st.write('the following columns are removed from a copy of the dataframe due to their datatype (object)')
    obj_cols_to_remove = []
    for col in obj_cols:
        obj_cols_to_remove.append(col)
    st.write(obj_cols_to_remove)
    df = df.drop(columns=obj_cols_to_remove)
    return df

def copy_df_date_index(my_df, datetime_to_date=True, date_to_index=True):
    # create a deep copy of dataframe e.g. completely new copy of the DataFrame is created with its own memory space
    # This means that any changes made to the new copy will not affect the original DataFrame.
    my_df_copy = my_df.copy(deep=True)
    if datetime_to_date == True:
        # convert the datetime to date (excl time 00:00:00)
        my_df_copy['date'] = pd.to_datetime(my_df['date']).dt.date
    if date_to_index == True:
        # set the index instead of 0,1,2... to date
        my_df_copy = my_df_copy.set_index('date')
    return my_df_copy

def resample_missing_dates(df, freq_dict, freq):
    """
    Resamples a pandas DataFrame to a specified frequency, fills in missing values with NaNs,
    and inserts missing dates as rows with NaN values. Also displays a message if there are
    missing dates in the data.
    
    Parameters:
    df (pandas DataFrame): The DataFrame to be resampled
    
    Returns:
    pandas DataFrame: The resampled DataFrame with missing dates inserted
    
    """
    # Resample the data to the specified frequency and fill in missing values with NaNs
    resampled_df = df.set_index('date').resample(freq_dict[freq]).asfreq()
    # Find skipped dates and insert them as rows with NaN values
    missing_dates = pd.date_range(start=resampled_df.index.min(), end=resampled_df.index.max(), freq=freq_dict[freq]).difference(resampled_df.index)
    new_df = resampled_df.reindex(resampled_df.index.union(missing_dates)).sort_index()
    # Display a message if there are skipped dates in the data
    if len(missing_dates) > 0:
        st.write("The skipped dates are:")
        st.write(missing_dates)
    # Reset the index and rename the columns
    return new_df.reset_index().rename(columns={'index': 'date'})

def my_fill_method(df, fill_method, custom_fill_value=None):
    # handle missing values based on user input
    if fill_method == 'Backfill':
        df.iloc[:,1] = df.iloc[:, 1].bfill()
    elif fill_method == 'Forwardfill':
        df.iloc[:,1] = df.iloc[:, 1].ffill()
    elif fill_method == 'Mean':
        # rounding occurs to nearest decimal for filling in the average value of y
        df.iloc[:,1] = df.iloc[:, 1].fillna(df.iloc[:,1].mean().round(0))
    elif fill_method == 'Median':
        df.iloc[:,1]  = df.iloc[:, 1].fillna(df.iloc[:,1].median())
    elif fill_method == 'Mode':
        # if True, only apply to numeric columns (numeric_only=False)
        # Don’t consider counts of NaN/NaT (dropna=True)
        df.iloc[:,1] = df.iloc[:, 1].fillna(df.iloc[:,1].mode(dropna=True)[0])
    elif fill_method == 'Custom':
        df.iloc[:,1]  = df.iloc[:,1].fillna(custom_fill_value)
    return df

def plot_overview(df, y):
    """
    Plot an overview of daily, weekly, monthly, quarterly, and yearly patterns
    for a given dataframe and column.
    """
    y_column_index = df.columns.get_loc(y)
    y_colname = df.columns[y_column_index]
    # Create subplots
    fig = make_subplots(rows=6, cols=1,
                        subplot_titles=('Daily Pattern', 
                                        'Weekly Pattern', 
                                        'Monthly Pattern',
                                        'Quarterly Pattern', 
                                        'Yearly Pattern', 
                                        'Histogram'
                                        ))
    # Daily Pattern
    fig.add_trace(px.line(df, x='date', y=y_colname, title='Daily Pattern').data[0], row=1, col=1)
    # Weekly Pattern
    df_weekly = df.resample('W', on='date').mean().reset_index()
    fig.add_trace(px.line(df_weekly, x='date', y=y_colname, title='Weekly Pattern').data[0], row=2, col=1)
    # Monthly Pattern
    df_monthly = df.resample('M', on='date').mean().reset_index()
    fig.add_trace(px.line(df_monthly, x='date', y=y_colname, title='Monthly Pattern').data[0], row=3, col=1)
    # Quarterly Pattern
    df_quarterly = df.resample('Q', on='date').mean().reset_index()
    fig.add_trace(px.line(df_quarterly, x='date', y=y_colname, title='Quarterly Pattern').data[0], row=4, col=1)
    # Yearly Pattern
    df_yearly = df.resample('Y', on='date').mean().reset_index()
    fig.add_trace(px.line(df_yearly, x='date', y=y_colname, title='Yearly Pattern').data[0], row=5, col=1)
    # Histogram
    fig.add_trace(px.histogram(df, x=y_colname, title='Histogram').data[0], row=6, col=1)
    # Update layout
    st.markdown('---')
    my_subheader('Overview of Patterns', my_size=3)
    fig.update_layout(height=1600, title='')
    # Display in Streamlit app
    st.plotly_chart(fig, use_container_width=True)

#################### PACF GRAPH ###########################################
# Define functions to calculate PACF
#################### PACF GRAPH ###########################################
# Create the plot using Plotly Express and difference timeseries 1st order, 2nd order or 3rd order differencing
def df_differencing(df, selection):
    ##### DIFFERENCING #####
    # show graph first, second and third order differencing
    # Calculate the first three differences of the data
    df_diff1 = df.iloc[:, 1].diff()
    df_diff2 = df_diff1.diff()
    df_diff3 = df_diff2.diff()
    # Replace any NaN values with 0
    df_diff1.fillna(0, inplace=True)
    df_diff2.fillna(0, inplace=True)
    df_diff3.fillna(0, inplace=True)
    if selection == 'Original Series':
        fig = px.line(df, x='date', y=df.columns[1], title='Original Series [No Differencing Applied]')
        df_select_diff = df.iloc[:,1]
    elif selection == 'First Order Difference':
        fig = px.line(pd.concat([df.iloc[:, 0], df_diff1], axis=1), 
                                  x='date', 
                                  y=df_diff1.name, 
                                  title='First Order Difference', 
                                  color_discrete_sequence=['#87CEEB'])
        df_select_diff = df_diff1
    elif selection == 'Second Order Difference':
        fig = px.line(pd.concat([df.iloc[:, 0], df_diff2], axis=1), 
                                  x='date', 
                                  y=df_diff2.name, 
                                  title='Second Order Difference', 
                                  color_discrete_sequence=['#1E90FF'])
        df_select_diff = df_diff2
    else:
        fig = px.line(pd.concat([df.iloc[:, 0], df_diff3], axis=1), 
                                  x='date', 
                                  y=df_diff3.name, 
                                  title='Third Order Difference', 
                                  color_discrete_sequence=['#000080'])
        df_select_diff = df_diff3
    return fig, df_select_diff

def calc_pacf(data, nlags, method):
    return pacf(data, nlags=nlags, method=method)

# Define function to plot PACF
def plot_pacf(data, nlags, method):
    '''
    Plots the partial autocorrelation function (PACF) for a given time series data.

    Parameters:
    -----------
    data : pandas.DataFrame
        The time series data to plot the PACF for.
    nlags : int
        The number of lags to include in the PACF plot.
    method : str
        The method to use for calculating the PACF. Can be one of "yw" (default), "ols", "ywunbiased", "ywmle", or "ld".

    Returns:
    --------
    None.

    Notes:
    ------
    This function drops any rows from the input data that contain NaN values before calculating the PACF.
    The PACF plot includes shaded regions representing the 95% and 99% confidence intervals.
    '''
    if data.isna().sum().sum() > 0:
        st.error('''**Warning** ⚠️:              
                 Data contains **NaN** values. **NaN** values were dropped in copy of dataframe to be able to plot below PACF. ''')
    st.markdown('<p style="text-align:center; color: #707070">Partial Autocorrelation (PACF)</p>', unsafe_allow_html=True)
    # Drop NaN values if any
    data = data.dropna(axis=0)
    data = data.to_numpy()
    # Calculate PACF
    pacf_vals = calc_pacf(data, nlags, method)
    # Create trace for PACF plot
    traces = []
    for i in range(nlags + 1):
        trace = go.Scatter(x=[i, i], y=[0, pacf_vals[i]],
                           mode='lines+markers', name='Lag {}'.format(i),
                           line=dict(color='grey', width=1))
        # Color lines based on confidence intervals
        conf95 = 1.96 / np.sqrt(len(data))
        conf99 = 2.58 / np.sqrt(len(data))
        # define the background shape and color for the 95% confidence band
        conf_interval_95_background = go.layout.Shape(
                                        type='rect',
                                        xref='x',
                                        yref='y',
                                        x0=0.5, #lag0 is y with itself so confidence interval starts from lag1 and I want to show a little over lag1 visually so 0.5
                                        y0=-conf95,
                                        x1=nlags+1,
                                        y1=conf95,
                                        fillcolor='rgba(68, 114, 196, 0.3)',
                                        line=dict(width=0),
                                        opacity=0.5
                                    )
        # define the background shape and color for the 99% confidence band
        conf_interval_99_background = go.layout.Shape(
                                        type='rect',
                                        xref='x',
                                        yref='y',
                                        x0=0.5, #lag0 is y with itself so confidence interval starts from lag1 and I want to show a little over lag1 visually so 0.5
                                        y0=-conf99,
                                        x1=nlags+1,
                                        y1=conf99,
                                        fillcolor='rgba(68, 114, 196, 0.3)',
                                        line=dict(width=0),
                                        opacity=0.4
                                    )
        # if absolute value of lag is larger than confidence band 99% then color 'darkred'
        if abs(pacf_vals[i]) > conf99:
            trace.line.color = 'darkred'
            trace.name += ' (>|99%|)'
        # else if absolute value of lag is larger than confidence band 95% then color 'lightcoral'
        elif abs(pacf_vals[i]) > conf95:
            trace.line.color = 'lightcoral'
            trace.name += ' (>|95%|)'
        traces.append(trace)
    # Set layout of PACF plot
    layout = go.Layout(title='',
                       xaxis=dict(title='Lag'),
                       yaxis=dict(title='Partial Autocorrelation'),
                       margin=dict(l=50, r=50, t=0, b=50),
                       shapes=[{'type': 'line', 'x0': -1, 'y0': conf95,
                                'x1': nlags + 1, 'y1': conf95,
                                'line': {'color': 'gray', 'dash': 'dash', 'width': 1},
                                'name': '95% Confidence Interval'},
                               {'type': 'line', 'x0': -1, 'y0': -conf95,
                                'x1': nlags + 1, 'y1': -conf95,
                                'line': {'color': 'gray', 'dash': 'dash', 'width': 1}},
                               {'type': 'line', 'x0': -1, 'y0': conf99,
                                'x1': nlags + 1, 'y1': conf99,
                                'line': {'color': 'gray', 'dash': 'dot', 'width': 1},
                                'name': '99% Confidence Interval'},
                               {'type': 'line', 'x0': -1, 'y0': -conf99,
                                'x1': nlags + 1, 'y1': -conf99,
                                'line': {'color': 'gray', 'dash': 'dot', 'width': 1}}, 
                               conf_interval_95_background, 
                               conf_interval_99_background],
                       showlegend=True,
                       )

    # Create figure with PACF plot
    fig = go.Figure(data=traces, layout=layout)
    # add legend
    fig.update_layout(
        legend=dict(title='Lag (conf. interval)'))
    st.plotly_chart(fig)

def calc_acf(data, nlags):
    '''
    Calculates the autocorrelation function (ACF) for a given time series data.

    Parameters:
    -----------
    data : numpy.ndarray
        The time series data to calculate the ACF for.
    nlags : int
        The number of lags to include in the ACF calculation.

    Returns:
    --------
    acf_vals : numpy.ndarray
        The ACF values for the specified number of lags.
    '''
    # Calculate the mean of the input data
    mean = np.mean(data)
    # Calculate the autocovariance for each lag
    acovf = np.zeros(nlags + 1)
    for k in range(nlags + 1):
        sum = 0.0
        for i in range(k, len(data)):
            sum += (data[i] - mean) * (data[i - k] - mean)
        acovf[k] = sum / (len(data) - k)
    # Calculate the ACF by normalizing the autocovariance with the variance
    acf_vals = np.zeros(nlags + 1)
    var = np.var(data)
    acf_vals[0] = 1.0
    for k in range(1, nlags + 1):
        acf_vals[k] = acovf[k] / var
    return acf_vals

def plot_acf(data, nlags):
    '''
    Plots the autocorrelation function (ACF) for a given time series data.

    Parameters:
    -----------
    data : pandas.DataFrame
        The time series data to plot the ACF for.
    nlags : int
        The number of lags to include in the ACF plot.

    Returns:
    --------
    None.

    Notes:
    ------
    This function drops any rows from the input data that contain NaN values before calculating the ACF.
    The ACF plot includes shaded regions representing the 95% and 99% confidence intervals.
    '''
    if data.isna().sum().sum() > 0:
        st.error('''**Warning** ⚠️:              
                 Data contains **NaN** values. **NaN** values were dropped in copy of dataframe to be able to plot below ACF. ''')
    st.markdown('<p style="text-align:center; color: #707070">Autocorrelation (ACF)</p>', unsafe_allow_html=True)
    
    # Drop NaN values if any
    data = data.dropna(axis=0)
    data = data.to_numpy()
    # Calculate ACF
    acf_vals = calc_acf(data, nlags)
    # Create trace for ACF plot
    traces = []
    for i in range(nlags + 1):
        trace = go.Scatter(x=[i, i], y=[0, acf_vals[i]],
                           mode='lines+markers', name='Lag {}'.format(i),
                           line=dict(color='grey', width=1))
        # Color lines based on confidence intervals
        conf95 = 1.96 / np.sqrt(len(data))
        conf99 = 2.58 / np.sqrt(len(data))
        if abs(acf_vals[i]) > conf99:
            trace.line.color = 'darkred'
            trace.name += ' (>|99%|)'
        elif abs(acf_vals[i]) > conf95:
            trace.line.color = 'lightcoral'
            trace.name += ' (>|95%|)'
        traces.append(trace)
    
    # define the background shape and color for the 95% confidence band
    conf_interval_95_background = go.layout.Shape(
                                                    type='rect',
                                                    xref='x',
                                                    yref='y',
                                                    x0=0.5, #lag0 is y with itself so confidence interval starts from lag1 and I want to show a little over lag1 visually so 0.5
                                                    y0=-conf95,
                                                    x1=nlags+1,
                                                    y1=conf95,
                                                    fillcolor='rgba(68, 114, 196, 0.3)',
                                                    line=dict(width=0),
                                                    opacity=0.5
                                                )
    # define the background shape and color for the 99% confidence band
    conf_interval_99_background = go.layout.Shape(
                                                    type='rect',
                                                    xref='x',
                                                    yref='y',
                                                    x0=0.5, #lag0 is y with itself so confidence interval starts from lag1 and I want to show a little over lag1 visually so 0.5
                                                    y0=-conf99,
                                                    x1=nlags+1,
                                                    y1=conf99,
                                                    fillcolor='rgba(68, 114, 196, 0.3)',
                                                    line=dict(width=0),
                                                    opacity=0.4
                                                )
    # Set layout of ACF plot
    layout = go.Layout(
                       title='',
                       xaxis=dict(title='Lag'),
                       yaxis=dict(title='Autocorrelation'),
                       margin=dict(l=50, r=50, t=0, b=50),
                       shapes=[
                               {'type': 'line', 'x0': -1, 'y0': conf95,
                                'x1': nlags + 1, 'y1': conf95,
                                'line': {'color': 'gray', 'dash': 'dash', 'width': 1},
                                'name': '95% Confidence Interval'},
                               {'type': 'line', 'x0': -1, 'y0': -conf95,
                                'x1': nlags + 1, 'y1': -conf95,
                                'line': {'color': 'gray', 'dash': 'dash', 'width': 1}},
                               {'type': 'line', 'x0': -1, 'y0': conf99,
                                'x1': nlags + 1, 'y1': conf99,
                                'line': {'color': 'gray', 'dash': 'dot', 'width': 1},
                                'name': '99% Confidence Interval'},
                               {'type': 'line', 'x0': -1, 'y0': -conf99,
                                'x1': nlags + 1, 'y1': -conf99,
                                'line': {'color': 'gray', 'dash': 'dot', 'width': 1}},
                               conf_interval_95_background, 
                               conf_interval_99_background,
                             ]
                       )
    # Define Figure
    fig = go.Figure(data=traces, layout=layout)

    # add legend
    fig.update_layout(
        legend=dict(title='Lag (conf. interval)'))
    # Plot ACF with Streamlit Plotly 
    st.plotly_chart(fig)    
 
######### OUTLIER DETECTION FUNCTIONS ##############
def outlier_form():
    with st.form('outlier_form'):
        my_subheader('Handling Outliers 😇😈😇 ', my_size=4, my_style='#440154')
        # form to select outlier handling method
        method = st.selectbox('*Select outlier handling method:*',
                             ('None', 'isolation_forest'))
    
        # sliders for Isolation Forest parameters
        if method == 'isolation_forest':
            contamination = st.slider(
                'Contamination:', min_value=0.01, max_value=0.5, step=0.01, value=0.01)
            # set the random state
            random_state = 10
        else:
            contamination = None
            random_state = None
        col1, col2, col3 = st.columns([4,4,4])
        with col2:
            st.form_submit_button('Submit')
    return method, contamination, random_state    

# define function to handle outliers using Isolation Forest
def handle_outliers(data, method, contamination, random_state):
    if method == 'remove':
        # remove rows with outlier values
        data = data.dropna()
    elif method == 'replace_with_median':
        # replace outlier values with median of column
        medians = data.median()
        for col in data.columns:
            data[col] = np.where(
                data[col] < medians[col],
                data[col],
                medians[col])
    elif method == 'isolation_forest':
        # detect and replace outlier values using Isolation Forest
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state)
        model.fit(data)
        outliers = model.predict(data) == -1
        medians = data.median()
        for col in data.columns:
            data[col][outliers] = medians[col]
    return data 

# Log
print('ForecastGenie Print: Loaded Functions')
###############################################################################
# Create Left-Sidebar Streamlit App with Title + About Information
###############################################################################
# TITLE PAGE + SIDEBAR TITLE
with st.sidebar:   
        st.markdown(f'<h1 style="color:#45B8AC;"> <center> ForecastGenie™️ </center> </h1>', unsafe_allow_html=True)         


# ABOUT SIDEBAR MENU
with st.sidebar.expander('ℹ️ About', expanded=False):
    st.write('''Hi :wave: **Welcome** to the ForecastGenie app created with Streamlit :smiley:
            
                \n**What does it do?**  
                - Analyze, Clean, Select Top Features, Train/Test and Forecast your Time Series Dataset!  
                \n**What do you need?**  
                - Upload your `.CSV` file that contains your **date** (X) column and your target variable of interest (Y)  
            ''')

    st.markdown('---')
    # DISPLAY LOGO
    col1, col2, col3 = st.columns([1,3,2])
    with col2:
        st.image('./images/logo_dark.png', caption="Developed by")  
        # added spaces to align website link with logo in horizontal center
        st.markdown(f'<h6 style="color:#217CD0;"><center><a href="https://www.tonyhollaar.com/">www.tonyhollaar.com</a></center></h5>', unsafe_allow_html=True)
        st.caption(f'<h7><center> ForecastGenie version: 1.1 <br>  Release date: 04-23-2023  </center></h7>', unsafe_allow_html=True)
    st.markdown('---')

###############################################################################
# 1. Create Button to Load Dataset (.CSV format)
###############################################################################
with tab1:
    # create a sidebar with options to load data
    my_title("1. Load Dataset 🚀 ", "#2CB8A1")
    with st.sidebar:
        my_title("1. Load Dataset 🚀 ", "#2CB8A1") # 2CB8A1
        with st.expander('', expanded=True):
            data_option = st.radio("*Choose an option:*", ["Demo Data", "Upload Data"])
        
            if data_option == "Upload Data":
                uploaded_file = st.file_uploader("Upload your .CSV file", type=["csv"])
    
    # check if demo data should be used
    if data_option == "Demo Data":
        df_raw = generate_demo_data()
        df_graph = df_raw.copy(deep=True)
        df_total = df_raw.copy(deep=True)
        # set minimum date
        df_min = df_raw.iloc[:,0].min().date()
        # set maximum date
        df_max = df_raw.iloc[:,0].max().date()
        #st.success('''🗨️ **Great!** your **Demo** Dataset is loaded, you can take a look 👀 by clicking on the **Explore** Tab...''')
        
        with st.expander(' ', expanded=True):
            # create 3 columns for spacing
            col1, col2, col3 = st.columns([1,3,1])
            # display df shape and date range min/max for user
            col2.markdown(f"<center>Your <b>dataframe</b> has <b><font color='#555555'>{df_raw.shape[0]}</b></font> \
                          rows and <b><font color='#555555'>{df_raw.shape[1]}</b></font> columns <br> with date range: \
                          <b><font color='#555555'>{df_min}</b></font> to <b><font color='#555555'>{df_max}</font></b>.</center>", 
                          unsafe_allow_html=True)
            # add a vertical linespace
            st.write("")
            df_graph = copy_df_date_index(my_df=df_graph, datetime_to_date=True, date_to_index=True)
            # set caption
            st.caption('')
          
            # Display Plotly Express figure in Streamlit
            display_dataframe_graph(df=df_graph, key=1)
            # show dataframe below graph        
            st.dataframe(df_graph, use_container_width=True)
            # download csv button
            download_csv_button(df_graph, my_file="raw_data.csv", help_message='Download dataframe to .CSV', set_index=True)
            
    if data_option == "Upload Data" and uploaded_file is None:
        # let user upload a file
        # inform user what template to upload
        with st.expander("", expanded=True):
            my_header("Instructions")
            st.info('''👈 **Please upload a .CSV file with:**  
                     - first column named: **$date$** with format: **$mm/dd/yyyy$**  
                     - second column the target variable: $y$''')
    
    # check if data is uploaded
    if data_option == "Upload Data" and uploaded_file is not None:
        # define dataframe from custom function to read from uploaded read_csv file
        df_raw = load_data()
        df_graph = df_raw.copy(deep=True)
        df_total = df_raw.copy(deep=True)
        # set minimum date
        df_min = df_raw.iloc[:,0].min().date()
        # set maximum date
        df_max = df_raw.iloc[:,0].max().date()
        st.success('''🗨️ **Great!** your data is loaded, you can take a look 👀 by clicking on the **Explore** Tab...''')
        
        with st.expander('', expanded=True):
            # create 3 columns for spacing
            col1, col2, col3 = st.columns([1,3,1])
            # display df shape and date range min/max for user
            col2.markdown(f"<center>Your <b>dataframe</b> has <b><font color='#555555'>{df_raw.shape[0]}</b></font> \
                          rows and <b><font color='#555555'>{df_raw.shape[1]}</b></font> columns <br> with date range: \
                          <b><font color='#555555'>{df_min}</b></font> to <b><font color='#555555'>{df_max}</font></b>.</center>", 
                          unsafe_allow_html=True)
            # add a vertical linespace
            st.write("")
            df_graph = copy_df_date_index(my_df=df_graph, datetime_to_date=True, date_to_index=True)
            # set caption
            st.caption('')
            
            #############################################################################
            ## display/plot graph of dataframe
            display_dataframe_graph(df=df_graph, key=2)
            # show dataframe below graph        
            st.dataframe(df_graph, use_container_width=True)
            # download csv button
            download_csv_button(df_graph, my_file="raw_data.csv", help_message='Download dataframe to .CSV', set_index=True)
        
    if df_raw.empty:
        pass
    # else continue code below
    else:
        with tab2:    
            # set title
            my_title('2. Exploratory Data Analysis 🕵️‍♂️', my_background_color="#217CD0")
            with st.sidebar:
                my_title("2. Exploratory Data Analysis	🕵️‍♂️", "#217CD0")
                
                with st.form('eda'):
                    # Create sliders in sidebar for the parameters of PACF Plot
                    st.write("")
                    my_subheader('Autocorrelation Plot Parameters', my_size=4, my_style='#217CD0')
                    col1, col2, col3 = st.columns([4,1,4])
                    # Set default values for parameters
                    default_lags = 30
                    default_method = "yw"  
                    
                    nlags_acf = st.slider("*Lags ACF*", min_value=1, max_value=(len(df_raw)-1), value=default_lags)
                    col1, col2, col3 = st.columns([4,1,4])
                    with col1:
                        nlags_pacf = st.slider("*Lags PACF*", min_value=1, max_value=int((len(df_raw)-2)/2), value=default_lags)
                    with col3:
                        method_pacf = st.selectbox("*Method PACF*", [ 'ols', 'ols-inefficient', 'ols-adjusted', 'yw', 'ywa', 'ld', 'ywadjusted', 'yw_adjusted', 'ywm', 'ywmle', 'yw_mle', 'lda', 'ldadjusted', 'ld_adjusted', 'ldb', 'ldbiased', 'ld_biased'], index=0)
                    # Define the dropdown menu options
                    options = ['Original Series', 'First Order Difference', 'Second Order Difference', 'Third Order Difference']
                    # Create the sidebar dropdown menu
                    selection = st.selectbox('*Apply Differencing [Optional]:*', options)
                    col1, col2, col3 = st.columns([4,4,4])
                    with col2:
                        # create button in sidebar for the ACF and PACF Plot Parameters
                        st.write("")
                        acf_pacf_btn = st.form_submit_button("Submit", type="secondary")
            # create expandable card with data exploration information
            with st.expander(':arrow_down: EDA', expanded=True):
                #############################################################################
                # Summary Statistics
                #############################################################################
                # create subheader
                my_subheader('Summary Statistics', my_size=3)
                # create linespace
                st.write("")
                # Display summary statistics table
                st.dataframe(display_summary_statistics(df_raw), use_container_width=True)
                
                #############################################################################
                # Call function for plotting Graphs of Seasonal Patterns D/W/M/Q/Y in Plotly Charts
                #############################################################################
                plot_overview(df_raw, y=df_raw.columns[1])
               
            with st.expander('Autocorrelation Plots (ACF & PACF) with optional Differencing applied', expanded=True):         
                # Display the plot based on the user's selection
                fig, df_select_diff = df_differencing(df_raw, selection)
                st.plotly_chart(fig, use_container_width=True)
                
                ############################## ACF & PACF ################################
                # set data equal to the second column e.g. expecting first column 'date' 
                data = df_select_diff
                # Plot ACF        
                plot_acf(data, nlags=nlags_acf)
                # Plot PACF
                plot_pacf(data, nlags=nlags_pacf, method=method_pacf)
                
                # If user clicks button, more explanation on the ACF and PACF plot is displayed
                col1, col2, col3 = st.columns([5,5,5])
                with col1:
                    show_acf_info_btn = st.button(f'About ACF plot', use_container_width=True, type='secondary')
                if show_acf_info_btn == True:
                    st.write('')
                    my_subheader('Autocorrelation Function (ACF)')
                    st.markdown('''
                                The **Autocorrelation Function (ACF)** plot is a statistical tool used to identify patterns of correlation between observations in a time series dataset. 
                                It is commonly used in time series analysis to determine the extent to which a given observation is related to its previous observations.  
                                The **ACF** plot displays the correlation coefficient between the time series and its own lagged values (i.e., the correlation between the series at time $t$ and the series at times $t_{-1}$, $t_{-2}$, $t_{-3}$, etc.).  
                                The horizontal axis of the plot shows the lag or time difference between observations, while the vertical axis represents the correlation coefficient, ranging from -1 to 1.
                                ''')
                    st.write('')
                    my_subheader('How to interpret a ACF plot')
                    st.markdown('''Interpreting the **ACF** plot involves looking for significant peaks or spikes above the horizontal dashed lines (which represent the confidence interval) to determine if there is any correlation between the current observation and the lagged observations. 
                                If there is a significant peak at a particular lag value, it indicates that there is a strong correlation between the observation and its lagged values up to that point.
                                ''')
                    st.write('')                           
                    my_subheader('Key Points:')  
                    st.markdown('''
                                Some key takeaways when looking at an **ACF** plot include:  
                                - If there are no significant peaks, then there is no significant correlation between the observations and their lagged values.
                                - A significant peak at lag $k$ means that the observation at time $t$ is significantly correlated with the observation at time $t_{-k}$.
                                - A gradual decay of the peaks towards zero suggests a stationary time series, while a slowly decaying **ACF** suggests a non-stationary time series.
                                ''')
                with col2:
                    show_pacf_info_btn = st.button(f'About PACF plot', use_container_width=True, type='secondary')
                if show_pacf_info_btn == True:   
                    st.write('')    
                    my_subheader('Partial Autocorrelation Function (PACF)')
                    st.markdown('''
                                The **Partial Autocorrelation Function (PACF)** is a plot of the partial correlation coefficients between a time series and its lags. 
                                The PACF can help us determine the order of an autoregressive (AR) model by identifying the lag beyond which the autocorrelations are effectively zero.
                                
                                The **PACF plot** helps us identify the important lags that are related to a time series. It measures the correlation between a point in the time series and a lagged version of itself while controlling for the effects of all the other lags that come before it.
                                In other words, the PACF plot shows us the strength and direction of the relationship between a point in the time series and a specific lag, independent of the other lags. 
                                A significant partial correlation coefficient at a particular lag suggests that the lag is an important predictor of the time series.
                                
                                If a particular lag has a partial autocorrelation coefficient that falls outside of the **95%** or **99%** confidence interval, it suggests that this lag is a significant predictor of the time series. 
                                The next step would be to consider including that lag in the autoregressive model to improve its predictive accuracy.
                                However, it is important to note that including too many lags in the model can lead to overfitting, which can reduce the model's ability to generalize to new data. 
                                Therefore, it is recommended to use a combination of statistical measures and domain knowledge to select the optimal number of lags to include in the model.
                                
                                On the other hand, if none of the lags have significant partial autocorrelation coefficients, it suggests that the time series is not well explained by an autoregressive model. 
                                In this case, alternative modeling techniques such as **Moving Average (MA)** or **Autoregressive Integrated Moving Average (ARIMA)** may be more appropriate. 
                                Or you could just flip a coin and hope for the best. But I don\'t recommend the latter...
                                ''')
                    st.write('')
                    my_subheader('How to interpret a PACF plot')
                    st.markdown('''
                                The partial autocorrelation plot (PACF) is a tool used to investigate the relationship between an observation in a time series with its lagged values, while controlling for the effects of intermediate lags. Here's a brief explanation of how to interpret a PACF plot:  
                                
                                - The horizontal axis shows the lag values (i.e., how many time steps back we\'re looking).
                                - The vertical axis shows the correlation coefficient, which ranges from **-1** to **1**. 
                                  A value of :green[**1**] indicates a :green[**perfect positive correlation**], while a value of :red[**-1**] indicates a :red[**perfect negative correlation**]. A value of **0** indicates **no correlation**.
                                - Each bar in the plot represents the correlation between the observation and the corresponding lag value. The height of the bar indicates the strength of the correlation. 
                                  If the bar extends beyond the dotted line (which represents the 95% confidence interval), the correlation is statistically significant.  
                                ''')
                    st.write('')                           
                    my_subheader('Key Points:')  
                    st.markdown('''                            
                                - **The first lag (lag 0) is always 1**, since an observation is perfectly correlated with itself.
                                - A significant spike at a particular lag indicates that there may be some **useful information** in that lagged value for predicting the current observation. 
                                  This can be used to guide the selection of lag values in time series forecasting models.
                                - A sharp drop in the PACF plot after a certain lag suggests that the lags beyond that point **are not useful** for prediction, and can be safely ignored.
                                ''')
                    st.write('')
                    my_subheader('An analogy')
                    st.markdown('''
                                Imagine you are watching a magic show where the magician pulls a rabbit out of a hat. Now, imagine that the magician can do this trick with different sized hats. If you were trying to figure out how the magician does this trick, you might start by looking for clues in the size of the hats.
                                Similarly, the PACF plot is like a magic show where we are trying to figure out the "trick" that is causing our time series data to behave the way it does. 
                                The plot shows us how strong the relationship is between each point in the time series and its past values, while controlling for the effects of all the other past values. 
                                It's like looking at different sized hats to see which one the magician used to pull out the rabbit.
                
                                If the **PACF** plot shows a strong relationship between a point in the time series and its past values at a certain lag (or hat size), it suggests that this past value is an important predictor of the time series. 
                                On the other hand, if there is no significant relationship between a point and its past values, it suggests that the time series may not be well explained by past values alone, and we may need to look for other "tricks" to understand it.
                                In summary, the **PACF** plot helps us identify important past values of our time series that can help us understand its behavior and make predictions about its future values.
                                ''')
                with col3:
                    diff_acf_pacf_info_btn = st.button(f'Difference ACF/PACF', use_container_width=True, type='secondary')
                if diff_acf_pacf_info_btn == True: 
                    st.write('')
                    my_subheader('Differences explained between ACF and PACF')
                    st.markdown('''
                                - The **ACF** plot measures the correlation between an observation and its lagged values.
                                - The **PACF** plot measures the correlation between an observation and its lagged values while controlling for the effects of intermediate observations.
                                - The **ACF** plot is useful for identifying the order of a moving average **(MA)** model, while the **PACF** plot is useful for identifying the order of an autoregressive **(AR)** model.
                                ''')
        
            ###############################################################################
            # 3. Data Cleaning
            ############################################################################### 
        with tab3:
            my_title("3. Data Cleaning 🧹", "#440154")
            with st.sidebar:
                my_title("3. Data Cleaning 🧹 ", "#440154")
                # with your form have a button to click and values are updated in streamlit
                with st.form('data_cleaning'):
                    my_subheader('Handling Missing Data 💭', my_size=4, my_style='#440154')
                    # get user input for filling method
                    fill_method = st.selectbox('*Select filling method for missing values:*', ['Backfill', 'Forwardfill', 'Mean', 'Median', 'Mode', 'Custom'])
                    custom_fill_value = None 
                    if fill_method == 'custom':
                        custom_fill_value = int(st.text_input('Enter custom value', value='0'))
                    # Define a dictionary of possible frequencies and their corresponding offsets
                    freq_dict = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
                    
                    # Ask the user to select the frequency of the data
                    freq = st.selectbox('*Select the frequency of the data*', list(freq_dict.keys()))
                    col1, col2, col3 = st.columns([4,4,4])
                    with col2:       
                        data_cleaning_btn = st.form_submit_button("Submit", type="secondary")
            with st.expander('💭 Missing Data', expanded=True):
                #*************************************************
                my_subheader('Handling missing data', my_style="#440154")
                #*************************************************    
                # Apply function to resample missing dates based on user set frequency
                # freq = daily, weekly, monthly, quarterly, yearly
                df_cleaned_dates = resample_missing_dates(df_raw, freq_dict=freq_dict, freq=freq)
        
                # Create matrix of missing values
                missing_matrix = df_cleaned_dates.isnull()
        
                # check if there are no dates skipped for daily data
                missing_dates = pd.date_range(start=df_raw['date'].min(), end=df_raw['date'].max()).difference(df_raw['date'])
                missing_values = df_raw.iloc[:,1].isna().sum()
                
                # Convert the DatetimeIndex to a dataframe with a single column named 'Date'
                df_missing_dates = pd.DataFrame({'Skipped Dates': missing_dates})
                # change datetime to date
                df_missing_dates['Skipped Dates'] = df_missing_dates['Skipped Dates'].dt.date
                
                
                # Create Plotly Express figure
                my_subheader('Missing Values Matrix Plot', my_style="#333333", my_size=6)
                fig = px.imshow(missing_matrix,
                                labels=dict(x="Variables", y="Observations"),
                                x=missing_matrix.columns,
                                y=missing_matrix.index,
                                color_continuous_scale='Viridis',
                                title='')
                # Set Plotly configuration options
                fig.update_layout(width=400, height=400, margin=dict(l=50, r=50, t=0, b=50))
                fig.update_traces(showlegend=False)
                # Display Plotly Express figure in Streamlit
                st.plotly_chart(fig, use_container_width=True)
                
                # check if in continous time-series dataset no dates are missing in between
                if missing_dates.shape[0] == 0:
                    st.success('Pweh 😅, no dates were skipped in your dataframe!')
                else:
                    st.warning(f'💡 **{missing_dates.shape[0]}** dates were skipped in your dataframe, don\'t worry though! I will **fix** this by **imputing** the dates into your cleaned dataframe!')
                if missing_values != 0:
                    st.warning(f'💡 **{missing_values}** missing values are filled with the next available value in the dataset (i.e. backfill method), optionally you can change the *filling method* and press **\"Submit\"**')
                
                #******************************************************************
                # IMPUTE MISSING VALUES WITH FILL METHOD
                #******************************************************************
                df_clean = my_fill_method(df_cleaned_dates, fill_method, custom_fill_value)
        
                # Display original DataFrame with highlighted NaN cells
                # Create a copy of the original DataFrame with NaN cells highlighted in yellow
                col1, col2, col3, col4, col5 = st.columns([2, 0.5, 2, 0.5, 2])
                with col1:
                    # highlight NaN values in yellow in dataframe
                    # got warning: for part of code with `null_color='yellow'`:  `null_color` is deprecated: use `color` instead
                    highlighted_df = df_graph.style.highlight_null(color='yellow').format(precision=0)
                    st.write('**Original DataFrame😐**')
                    # show original dataframe unchanged but with highlighted missing NaN values
                    st.write(highlighted_df)
                with col2:
                    st.write('➡️')
                with col3:
                    my_subheader('Skipped Dates 😳', my_style="#333333", my_size=6)
                    st.write(df_missing_dates)
                    
                    # Display the dates and the number of missing values associated with them
                    my_subheader('Missing Values 😖', my_style="#333333", my_size=6)
                    # Filter the DataFrame to include only rows with missing values
                    missing_df = copy_df_date_index(df_raw.loc[df_raw.iloc[:,1].isna(), df_raw.columns], datetime_to_date=True, date_to_index=True)
                    st.write(missing_df)
                with col4:
                    st.write('➡️')
                # display cleaned dataframe in Streamlit
                with col5:
                    st.write('**Cleaned Dataframe😄**')
                    # fix the datetime to date and set date column as index column
                    df_clean_show = copy_df_date_index(df_clean, datetime_to_date=True, date_to_index=True)
                    # show the cleaned dataframe with if needed dates inserted if skipped to NaN and then the values inserted with impute method user selected backfill/forward fill/mean/median
                    st.write(df_clean_show)
                download_csv_button(df_clean_show, my_file="df_imputed_missing_values.csv", set_index=True, help_message='Download cleaner dataframe to .CSV')
        
            #########################################################
            with st.expander('😇😈😇 Outliers', expanded=True):
                # set page subheader with custum function
                my_subheader('Handling outliers', my_style="#440154")
        
                # define function to generate form and sliders for outlier detection and handling
                ##############################################################################
                with st.sidebar:
                    # display form and sliders for outlier handling method
                    method, contamination, random_state = outlier_form()
                
                # plot data before and after cleaning
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_clean_show.index, 
                                         y=df_clean_show.iloc[:,0], 
                                         mode='markers', 
                                         name='Before'))
                df_cleaned_outliers = handle_outliers(df_clean_show, 
                                               method, 
                                               contamination, 
                                               random_state)
                # add scatterplot
                fig.add_trace(go.Scatter(x=df_cleaned_outliers.index, 
                                         y= df_cleaned_outliers.iloc[:,0], 
                                         mode='markers', 
                                         name='After'))
                # show the outlier plot 
                st.plotly_chart(fig, use_container_width=True)
        
                # create vertical spacings
                col1, col2, col3 = st.columns([4,4,4])
                with col2:
                    # create the button to download dataframe
                    show_df_cleaned_outliers = st.button(f'Show DataFrame', key='df_cleaned_outliers_download_btn', use_container_width=True, type='secondary')
                if show_df_cleaned_outliers == True:
                    # display the cleaned dataframe + optional changes in outliers made by user in streamlit
                    st.dataframe(df_cleaned_outliers, use_container_width=True)
                    # create a download button to download the .csv file of the cleaned dataframe
                    download_csv_button(df_cleaned_outliers, my_file="df_cleaned_outliers.csv", set_index=True)
            
            # reset the index again to have index instead of date column as index for further processing
            df_cleaned_outliers_with_index = df_cleaned_outliers.copy(deep=True)
            df_cleaned_outliers_with_index.reset_index(inplace=True)
            # convert 'date' column to datetime in both DataFrames
            df_cleaned_outliers_with_index['date'] = pd.to_datetime(df_cleaned_outliers_with_index['date'])
            
            ###############################################################################
            # 4. Feature Engineering
            ###############################################################################
        with tab4:
            my_title("4. Feature Engineering 🧰", "#FF6F61")
            with st.sidebar.form('feature engineering sidebar'):
                my_title("4. Feature Engineering 🧰", "#FF6F61")  
                st.info('Select your features to engineer:')
                # show checkbox in middle of sidebar to select all features or none
                col1, col2, col3 = st.columns([0.1,8,3])
                with col3:
                    select_all_days = st.checkbox('Select All Special Days:', value=True, label_visibility='collapsed' )
                    select_all_seasonal = st.checkbox('Select All Sesaonal:', value=True, label_visibility='collapsed' )
                    # create checkbox for Discrete Wavelet Transform features which automatically is checked
                    select_dwt_features = st.checkbox('', value=False, label_visibility='visible', help='In feature engineering, wavelet transform can be used to extract useful information from a time series by decomposing it into different frequency bands. This is done by applying a mathematical function called the wavelet function to the time series data. The resulting wavelet coefficients can then be used as features in machine learning models.')
                with col2:
                    if select_all_days == True:
                       st.write("*🎁 All Special Calendar Days*")
                    else:
                        st.write("*🎁 No Special Calendar Days*") 
                    if select_all_seasonal == True:
                        st.write("*🌓 All Seasonal Periods*")
                    else:
                        st.write("*🌓 No Seasonal Periods*") 
                    if select_dwt_features == True:
                        st.write("*🌊 All Wavelet Features*")
                    else:
                        st.write("*🌊No Wavelet Features*") 
                with st.expander('🔽 Wavelet settings'):
                    wavelet_family = st.selectbox('*Select Wavelet Family*', ['db4', 'sym4', 'coif4'], label_visibility='visible', help=' A wavelet family is a set of wavelet functions that have different properties and characteristics.  \
                                                                                                                                      \n**`db4`** wavelet is commonly used for signals with *smooth variations* and *short-duration* pulses  \
                                                                                                                                       \n**`sym4`** wavelet is suited for signals with *sharp transitions* and *longer-duration* pulses.  \
                                                                                                                                       \n**`coif4`** wavelet, on the other hand, is often used for signals with *non-linear trends* and *abrupt* changes.  \
                                                                                                                                       \nIn general, the **`db4`** wavelet family is a good starting point, as it is a popular choice for a wide range of applications and has good overall performance.')
                    # set standard level of decomposition to 3 
                    wavelet_level_decomposition = st.selectbox('*Select Level of Decomposition*', [1, 2, 3, 4, 5], label_visibility='visible', index=3, help='The level of decomposition refers to the number of times the signal is decomposed recursively into its approximation coefficients and detail coefficients.  \
                                                                                                                                                            \nIn wavelet decomposition, the signal is first decomposed into two components: a approximation component and a detail component.\
                                                                                                                                                            The approximation component represents the coarsest level of detail in the signal, while the detail component represents the finer details.  \
                                                                                                                                                            \nAt each subsequent level of decomposition, the approximation component from the previous level is decomposed again into its own approximation and detail components.\
                                                                                                                                                            This process is repeated until the desired level of decomposition is reached.  \
                                                                                                                                                            \nEach level of decomposition captures different frequency bands and details in the signal, with higher levels of decomposition capturing finer and more subtle details.  \
                                                                                                                                                            However, higher levels of decomposition also require more computation and may introduce more noise or artifacts in the resulting representation of the signal.  \
                                                                                                                                                            \nThe choice of the level of decomposition depends on the specific application and the desired balance between accuracy and computational efficiency.')
                    # add slider or text input to choose window size
                    wavelet_window_size = int(st.slider('*Select Window Size (in days)*', min_value=1, max_value=30, value=7, label_visibility='visible'))
                
                col1, col2, col3 = st.columns([4,4,4])
                with col2:
                    # add submit button to form, when user presses it it updates the selection criteria
                    submitted = st.form_submit_button('Submit')
        
            with st.expander("📌 Calendar Features", expanded=True):
                my_header('Special Calendar Days')
                my_subheader("🎁 Pick your special days to include: ")
                st.write("")
                ###############################################
                # create checkboxes for special days on page
                ###############################################
                col0, col1, col2, col3 = st.columns([6,12,12,1])
                with col1:
                    jan_sales = st.checkbox('January Sale', value=select_all_days)
                    val_day_lod = st.checkbox('Valentine\'s Day [last order date]', value=select_all_days)
                    val_day = st.checkbox('Valentine\'s Day', value=select_all_days)
                    mother_day_lod = st.checkbox('Mother\'s Day [last order date]', value=select_all_days)
                    mother_day = st.checkbox('Mother\'s Day', value=select_all_days)
                    father_day_lod = st.checkbox('Father\'s Day [last order date]', value=select_all_days)
                    pay_days = st.checkbox('Monthly Pay Days (4th Friday of month)', value=select_all_days)
                with col2:
                    father_day = st.checkbox('Father\'s Day', value=select_all_days)
                    black_friday_lod = st.checkbox('Black Friday [sale starts]', value=select_all_days)
                    black_friday = st.checkbox('Black Friday', value=select_all_days)
                    cyber_monday = st.checkbox('Cyber Monday', value=select_all_days)
                    christmas_day = st.checkbox('Christmas Day [last order date]', value=select_all_days)
                    boxing_day = st.checkbox('Boxing Day sale', value=select_all_days)
                # call very extensive function to create all days selected by users as features
                df = create_calendar_special_days(df_cleaned_outliers_with_index)
        
                ##############################
                # Add Day/Month/Year Features
                ##############################
                # create checkboxes for user to checkmark if to include features
                st.markdown('---')
                my_subheader('🌓 Pick your seasonal periods to include: ')
                # vertical space / newline between header and checkboxes
                st.write("")
                # create columns for aligning in middle the checkboxes
                col0, col1, col2, col3, col4 = st.columns([2, 2, 2, 2, 1])
                with col1:
                    year_dummies = st.checkbox('Year', value=select_all_seasonal)
                with col2:
                    month_dummies = st.checkbox('Month', value=select_all_seasonal)
                with col3:
                    day_dummies = st.checkbox('Day', value=select_all_seasonal)
                st.info('ℹ️ **Note**: to prevent perfect multi-collinearity, leave-one-out is applied e.g. one year/month/day')
                # apply function to add year/month and day dummy variables
                df = create_date_features(df, year_dummies=year_dummies, month_dummies=month_dummies, day_dummies=day_dummies)
                
            #######################################
            # Discrete Wavelet Transform (DWT)
            #######################################
            # if user checkmarked checkbox: Discrete Wavelet Transform
            if select_dwt_features:
                with st.expander('🌊 Wavelet Features', expanded=True):
                    my_header('Discrete Wavelet Transform')
                    my_subheader('Feature Extraction')
                    # define wavelet and level of decomposition
                    wavelet = wavelet_family
                    level = wavelet_level_decomposition
                    # define window size (in days)
                    window_size = wavelet_window_size
                    # create empty list to store feature vectors
                    feature_vectors = []
                    # loop over each window in the data
                    for i in range(window_size, len(df)):
                        # extract data for current window
                        data_in_window = df.iloc[i-window_size:i, 1].values
                        # perform DWT on sales data
                        coeffs = pywt.wavedec(data_in_window, wavelet, level=level)
                        # extract features from subbands
                        features = []
                        for j in range(len(coeffs)):
                            subband_features = [coeffs[j].mean(), coeffs[j].std(), coeffs[j].max(), coeffs[j].min()]
                            features.extend(subband_features)
                        # add features to list
                        feature_vectors.append(features)
                    # create new dataframe with features and original date index
                    feature_cols = ['approx_mean', 'approx_std', 'approx_max', 'approx_min'] + \
                                   [f'detail{i+1}_mean' for i in range(level)] + \
                                   [f'detail{i+1}_std' for i in range(level)] + \
                                   [f'detail{i+1}_max' for i in range(level)] + \
                                   [f'detail{i+1}_min' for i in range(level)]
                    # create a dataframe with the created features with discrete wavelet transform on target variable with timewindow set by user
                    features_df_wavelet = pd.DataFrame(feature_vectors, columns=feature_cols, index=df.iloc[:,0].index[window_size:])
                    # merge features dataframe with original data
                    df = pd.merge(df, features_df_wavelet, left_index=True, right_index=True)
                    # create a dataframe again with the index set as the first column
                    # assumption used: the 'date' column is the first column of the dataframe
                    features_df_plot = pd.DataFrame(feature_vectors, columns=feature_cols, index=df.iloc[:,0])
                    fig = px.line(features_df_plot, 
                                  x=features_df_plot.index, 
                                  y=['approx_mean'] + [f'detail{i+1}_mean' for i in range(level)],
                                  title='', 
                                  labels={'value': 'Coefficient Mean', 'variable': 'Subband'})
                    fig.update_layout(xaxis_title='Date')
        
                    st.plotly_chart(fig, use_container_width=True)
                    # show dataframe with features
                    my_subheader('Wavelet Features', my_size=6)
                    st.dataframe(features_df_wavelet, use_container_width=True)
            #################################################################
            # ALL FEATURES COMBINED INTO A DATAFRAME
            #################################################################
            # SHOW DATAFRAME
            with st.expander('🧫Engineered Features', expanded=True):
                my_header('Engineered Features')
                my_subheader('including target variable', my_size=6)
                st.dataframe(df)
                download_csv_button(df, my_file="dataframe_incl_features.csv", help_message="Download your dataset incl. features to .CSV")
        
            ###############################################################################
            # 5. Prepare Data
            ###############################################################################
        with tab5:    
            my_title('5. Prepare Data 🧪', "#FF9F00")
            with st.sidebar:
                my_title('5. Prepare Data 🧪', "#FF9F00")
            
            ##########################################
            # set date as index/ create local_df
            ##########################################
            # create copy of dataframe not altering original
            local_df = df.copy(deep=True)
           
            # set the date as the index of the pandas dataframe
            local_df.index = pd.to_datetime(local_df['date'])
            local_df.drop(columns='date', inplace=True)
            
            # show user which descriptive variables are removed, that just had the purpose to inform user what dummy was from e.g. holiday days such as Martin Luther King Day
            with st.expander('ℹ️ I removed the following descriptive columns automatically from analysis'):
                local_df = remove_object_columns(local_df)
            
            ######################
            # 5.1 TRAIN/TEST SPLIT
            ######################
            with st.expander("✂️ Train/Test Split", expanded=True):
                my_header('Train/Test Split')
                # create a caption on the page for user to read about rule-of-thumb train/test split 80:20 ratio
                st.caption(f'<h6> <center> ℹ️ A commonly used ratio is 80:20 split between train and test set </center> <h6>', unsafe_allow_html=True)
                
                # create sliders for user insample test-size (/train-size automatically as well)
                my_insample_forecast_steps, my_insample_forecast_perc = train_test_split_slider()
                
                # format as new variables insample_forecast steps in days/as percentage e.g. the test set to predict for
                perc_test_set = "{:.2f}%".format((my_insample_forecast_steps/len(df))*100)
                perc_train_set = "{:.2f}%".format(((len(df)-my_insample_forecast_steps)/len(df))*100)
            
                #############################################################
                # Create a figure with a scatter plot of the train/test split
                #############################################################
                # create figure with custom function for train/test split with plotly
                # Set train/test split index
                split_index = len(local_df) - my_insample_forecast_steps
                train_test_fig = plot_train_test_split(local_df, split_index)
                # show the plot inside streamlit app on page
                st.plotly_chart(train_test_fig, use_container_width=True)
                # show user the train test split currently set by user or default e.g. 80:20 train/test split
                st.warning(f"ℹ️ train/test split currently equals :green[**{perc_train_set}**] and :green[**{perc_test_set}**] ")
        
            #******************************************
            # CHANGE DATATYPES: for engineered features
            #******************************************
            columns_to_convert = {'holiday': 'uint8', 'calendar_event': 'uint8', 'pay_day': 'uint8', 'year': 'int32'}
            for column, data_type in columns_to_convert.items():
                if column in local_df:
                    local_df[column] = local_df[column].astype(data_type)
            # Normalize the numerical features only
            # e.g. changed float64 to float to include other floats such as float32 and float16 data types
            numerical_features = list(local_df.iloc[:, 1:].select_dtypes(include=['float', 'int64']).columns)
            
            ##############################
            # 5.2 Normalization
            ##############################
            with st.sidebar:
                with st.form('normalization'):
                    my_subheader('⚖️ Normalization', my_style="#FF9F00", my_size=3)
                    # Add selectbox for normalization choices
                    if numerical_features:
                        normalization_choices = {
                                                    "None": "Do not normalize the data",
                                                    "MinMaxScaler": "Scale features to a given range (default range is [0, 1]).",
                                                    "RobustScaler": "Scale features using statistics that are robust to outliers.",
                                                    "MaxAbsScaler": "Scale each feature by its maximum absolute value.",
                                                    "PowerTransformer": "Apply a power transformation to make the data more Gaussian-like.",
                                                    "QuantileTransformer": "Transform features to have a uniform or Gaussian distribution."
                                                }
                        # create a dropdown menu for user in sidebar to choose from a list of normalization methods
                        normalization_choice = st.selectbox("*Select normalization method:*", list(normalization_choices.keys()), format_func=lambda x: f"{x} - {normalization_choices[x]}"
                                                            , help='**`Normalization`** is a data pre-processing technique to transform the numerical data in a dataset to a standard scale or range.\
                                                                    This process involves transforming the features of the dataset so that they have a common scale, which makes it easier for data scientists to analyze, compare, and draw meaningful insights from the data.')
                    else:
                       # if no numerical features show user a message to inform
                       st.warning("No numerical features to normalize, you can try adding features!")
                       # set normalization_choice to None
                       normalization_choice = "None"
                    # create form button centered on sidebar to submit user choice for normalization method
                    col1, col2, col3 = st.columns([4,4,4])
                    with col2:       
                        normalization_btn = st.form_submit_button("Submit", type="secondary")
                # apply function for normalizing the dataframe if user choice
                # IF user selected a normalization_choice other then "None" the X_train and X_test will be scaled
                X, y, X_train, X_test, y_train, y_test, scaler = perform_train_test_split(local_df, my_insample_forecast_steps, normalization_choice, numerical_features=numerical_features)
                
            # if user did not select normalization (yet) then show user message to select normalization method in sidebar
            if normalization_choice == "None":
                # on page create expander
                with st.expander('⚖️ Normalization ',expanded=True):
                    my_header('Normalization') 
                    my_subheader(f'Method: {normalization_choice}', my_size=6)
                    st.info('👈 Please choose in the sidebar your normalization method for numerical columns. Note: columns with booleans will be excluded')
            
            # else show user the dataframe with the features that were normalized
            else:
                with st.expander('⚖️ Normalization',expanded=True):
                    my_header('Normalized Features') 
                    my_subheader(f'Method: {normalization_choice}', my_size=6)
                    # need original (unnormalized) X_train as well for figure in order to show before/after normalization
                    X_unscaled_train = df.iloc[:, 1:].iloc[:-my_insample_forecast_steps, :]
                    # with custom function create the normalization plot with numerical features i.e. before/after scaling
                    plot_scaling_before_after(X_unscaled_train, X_train, numerical_features)
                    st.success(f'🎉 Good job! **{len(numerical_features)}** numerical features are normalized with **{normalization_choice}**!')
                    st.write(X[numerical_features])
                    # create download button for user, to download the standardized features dataframe with dates as index i.e. first column
                    download_csv_button(X[numerical_features], my_file='standardized_features.csv', help_message='Download standardized features to .CSV', set_index=True)
                    
            ##############################
            # 5.3 Standardization
            ##############################            
            with st.sidebar:
                with st.form('standardization'):
                    my_subheader('🦩 Standardization', my_style="#FF9F00", my_size=3)
                    if numerical_features:
                        standardization_choices = {
                                                    "None": "Do not standardize the data",
                                                    "StandardScaler": "Standardize features by removing the mean and scaling to unit variance.",
                                                                                           }
                        standardization_choice = st.selectbox("*Select standardization method:*", list(standardization_choices.keys()), format_func=lambda x: f"{x} - {standardization_choices[x]}"
                                                              , help='**`Standardization`** is a preprocessing technique used to transform the numerical data to have zero mean and unit variance.\
                                                                      This is achieved by subtracting the mean from each value and then dividing by the standard deviation.\
                                                                      The resulting data will have a mean of zero and a standard deviation of one.\
                                                                      The distribution of the data is changed by centering and scaling the values, which can make the data more interpretable and easier to compare across different features' )
                    else:
                        # if no numerical features show user a message to inform
                       st.warning("No numerical features to standardize, you can try adding features!")
                       # set normalization_choice to None
                       standardization_choice = "None"
                    
                    # create form button centered on sidebar to submit user choice for standardization method   
                    col1, col2, col3 = st.columns([4,4,4])
                    with col2:       
                        standardization_btn = st.form_submit_button("Submit", type="secondary")
        
                # apply function for normalizing the dataframe if user choice
                # IF user selected a normalization_choice other then "None" the X_train and X_test will be scaled
                X, y, X_train, X_test, y_train, y_test = perform_train_test_split_standardization(X,y,X_train, X_test, y_train, y_test, my_insample_forecast_steps, scaler_choice=standardization_choice, numerical_features=numerical_features)
                
            # if user did not select normalization (yet) then show user message to select normalization method in sidebar
            if standardization_choice == "None":
                # on page create expander
                with st.expander('Standardization ',expanded=True):
                    my_header('Standardization') 
                    my_subheader(f'Method: {standardization_choice}', my_size=6)
                    st.info('👈 Please choose in the sidebar your Standardization method for numerical columns. Note: columns with booleans will be excluded.')
            # else show user the dataframe with the features that were normalized
            else:
                with st.expander('Standardization',expanded=True):
                    my_header('Standardized Features') 
                    my_subheader(f'Method: {standardization_choice}', my_size=6)
                    # need original (unnormalized) X_train as well for figure in order to show before/after normalization
                    X_unscaled_train = df.iloc[:, 1:].iloc[:-my_insample_forecast_steps, :]
                    # with custom function create the normalization plot with numerical features i.e. before/after scaling
                    plot_scaling_before_after(X_unscaled_train, X_train, numerical_features)
                    st.info(f'numerical features standardized: {len(numerical_features)}')
                    st.write(X[numerical_features])
                    # create download button for user, to download the standardized features dataframe with dates as index i.e. first column
                    download_csv_button(X[numerical_features], my_file='standardized_features.csv', help_message='Download standardized features to .CSV', set_index=True)
                    
            ###############################################################################
            # 6. Feature Selection
            ###############################################################################
        with tab6:    
            my_title('6. Feature Selection 🍏🍐🍋', "#7B52AB ")
            with st.expander('ℹ️ Selection Methods', expanded=True):
                st.markdown('''Let\'s **review** your **top features** to use in analysis with **three feature selection methods**:  
                            - Recursive Feature Elimination with Cross-Validation  
                            - Principal Component Analysis  
                            - Mutual Information  
                            ''')
                # Display a note to the user about using the training set for feature selection
                st.caption('NOTE: per common practice **only** the training dataset is used for feature selection to prevent **data leakage**.')   
            with st.sidebar:
                # show Title in sidebar 'Feature Selection' with purple background
                my_title('6. Feature Selection 🍏🍐🍋', "#7B52AB ")
                # =============================================================================
                # RFE Feature Selection - SIDEBAR FORM
                # =============================================================================
                with st.form('rfe'):
                     my_subheader('🎨 Recursive Feature Elimination', my_size=4, my_style='#7B52AB')
                     # Add a slider to select the number of features to be selected by the RFECV algorithm
                     num_features = st.slider('*Select number of top features to include:*', min_value=1, max_value=len(X.columns), value=5)
                     # set the options for the rfe (recursive feature elimination)
                     with st.expander('🔽 RFE Settings:', expanded=False):
                         # Add a selectbox for the user to choose the estimator
                         estimator_rfe = st.selectbox('*Set estimator:*', ['Linear Regression', 'Random Forest Regression'], index=0, help = 'the `estimator` parameter is used to specify the machine learning model that will be used to evaluate the importance of each feature. \
                                                                                                                                              The estimator is essentially the algorithm used to fit the data and make predictions.')
                         # Set up the estimator based on the user's selection
                         if estimator_rfe == 'Linear Regression':
                             est_rfe = LinearRegression()
                         elif estimator_rfe == 'Random Forest Regression':
                             est_rfe = RandomForestRegressor()
                         # Add a slider to select the number of n_splits for the RFE method
                         timeseriessplit_value_rfe = st.slider('*Set number of splits for Cross-Validation:*', min_value=2, max_value=5, value=5, help='`Cross-validation` is a statistical method used to evaluate the performance of a model by splitting the dataset into multiple "folds," where each fold is used as a holdout set for testing the model trained on the remaining folds. \
                                                                                                                                                          The cross-validation procedure helps to estimate the performance of the model on unseen data and reduce the risk of overfitting.  \
                                                                                                                                                          In the context of RFE, the cv parameter specifies the number of folds to use for the cross-validation procedure.\
                                                                                                                                                          The RFE algorithm fits the estimator on the training set, evaluates the performance of the estimator on the validation set, and selects the best subset of features. \
                                                                                                                                                          The feature selection process is repeated for each fold of the cross-validation procedure.')
                         # Add a slider in the sidebar for the user to set the number of steps parameter
                         num_steps_rfe = st.slider('*Set Number of Steps*', 1, 10, 1, help='The `step` parameter controls the **number of features** to remove at each iteration of the RFE process.')
                     
                     col1, col2, col3 = st.columns([4,4,4])
                     with col2:       
                         rfe_btn = st.form_submit_button("Submit", type="secondary")
            # =============================================================================
            # RFE Feature Selection - PAGE RESULTS
            # =============================================================================
            try:
                with st.expander('🎨 RFECV', expanded=True):
                    # run function to perform recursive feature elimination with cross-validation and display results using plot
                    selected_cols_rfe = rfe_cv(X_train, y_train, est_rfe, num_steps_rfe, num_features, timeseriessplit_value_rfe)
            except:
                selected_cols_rfe= []
                st.warning(':red[**ERROR**: Recursive Feature Elimination with Cross-Validation could not execute...please adjust your selection criteria]')
                     
            # =============================================================================        
            # PCA Feature Selection
            # =============================================================================
            with st.sidebar:    
                with st.form('pca'):
                    my_subheader('🧮 Principal Component Analysis', my_size=4, my_style='#7B52AB')
                    # Add a slider to select the number of features to be selected by the PCA algorithm
                    num_features_pca = st.slider('*Select number of top features to include:*', min_value=1, max_value=len(X.columns), value=5)
                    col1, col2, col3 = st.columns([4,4,4])
                    with col2:       
                        pca_btn = st.form_submit_button("Submit", type="secondary")
            try:
                with st.expander('🧮 PCA', expanded=True):
                    #my_subheader('Principal Component Analysis', my_size=4, my_style='#7B52AB')
                    pca = PCA(n_components=num_features_pca)
                    pca.fit(X_train)
                    
                    X_pca = pca.transform(X_train)
                    selected_features_pca = ['PC{}'.format(i+1) for i in range(num_features_pca)]
                    feature_names = X_train.columns
                    
                    # Sort features by explained variance ratio
                    sorted_idx = np.argsort(pca.explained_variance_ratio_)[::-1]
                    sorted_features = feature_names[sorted_idx]
                    
                    # Create plot
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=pca.explained_variance_ratio_[sorted_idx], y=sorted_features, 
                                         orientation='h', text=np.round(pca.explained_variance_ratio_[sorted_idx] * 100, 2), textposition='auto'))
                    fig.update_layout(title={
                                            'text': f'Top {len(sorted_features)} <br> Principal Component Analysis Feature Selection',
                                            'x': 0.5,
                                            'y': 0.95,
                                            'xanchor': 'center',
                                            'yanchor': 'top'
                                            },
                                      xaxis_title='Explained Variance Ratio', yaxis_title='Feature Name')
                    # Display plot in Streamlit
                    st.plotly_chart(fig)
                    
                    # show top x features selected
                    selected_cols_pca = sorted_features.tolist() 
                    st.info(f'Top {len(selected_cols_pca)} features selected with PCA: {selected_cols_pca}')
                    
                    show_pca_info_btn = st.button(f'About PCA plot', use_container_width=True, type='secondary')
                    if show_pca_info_btn == True:
                        st.write('')
                        # show user info about how to interpret the graph
                        st.markdown('''When you fit a **PCA** model, it calculates the amount of variance that is captured by each principal component.
                                The variance ratio is the fraction of the total variance in the data that is explained by each principal component.
                                The sum of the variance ratios of all the principal components equals 1.
                                The variance ratio is expressed as a percentage by multiplying it by 100, so it can be easily interpreted.  
                                ''')
                        st.markdown('''
                                    For example, a variance ratio of 0.75 means that 75% of the total variance in the data is captured by the corresponding principal component.
                                    ''')
            except:
                # if list is empty
                selected_cols_pca = []
                st.warning(':red[**ERROR**: Principal Component Analysis could not execute...please adjust your selection criteria]')
            
            # =============================================================================
            # Mutual Information Feature Selection
            # =============================================================================
            try: 
                with st.sidebar:
                    with st.form('mifs'):
                        my_subheader('🎏 Mutual Information', my_size=4, my_style='#7B52AB')
                        # Add slider to select number of top features
                        num_features = st.slider("*Select number of top features to include:*", min_value=1, max_value=len(X.columns), value=5, step=1)
                        col1, col2, col3 = st.columns([4,4,4])
                        with col2:       
                            mifs_btn = st.form_submit_button("Submit", type="secondary")
                with st.expander('🎏 MIFS', expanded=True):
                    # Mutual information feature selection
                    # mutual_info = mutual_info_classif(X, y, random_state=42)
                    mutual_info = mutual_info_regression(X_train, y_train, random_state=42)
                    selected_features_mi = X.columns[np.argsort(mutual_info)[::-1]][:num_features]
                    
                    # Create plot
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=mutual_info[np.argsort(mutual_info)[::-1]][:num_features],
                                         y=selected_features_mi, 
                                         orientation='h',
                                         text=[f'{val:.2f}' for val in mutual_info[np.argsort(mutual_info)[::-1]][:num_features]],
                                         textposition='inside'))
                    fig.update_layout(title={'text': f'Top {num_features} <br> Mutual Information Feature Selection',
                                            'x': 0.5,
                                            'y': 0.95,
                                            'xanchor': 'center',
                                            'yanchor': 'top'})
                    # Display plot in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                    
                    ##############################################################
                    # SELECT YOUR FAVORITE FEATURES TO INCLUDE IN MODELING
                    ##############################################################            
                    # Mutual Information Selection
                    selected_cols_mifs = list(selected_features_mi)
                    st.info(f'Top {num_features} features selected with MIFS: {selected_cols_mifs}')
                    
                    # create button to display information about mutual information feature selection
                    show_mifs_info_btn = st.button(f'About MIFS plot', use_container_width=True, type='secondary')            
                    if show_mifs_info_btn == True:
                        st.write('')
                        # show user info about how to interpret the graph
                        st.markdown('''Mutual Information Feature Selection (MIFS) is a method for selecting the most important features in a dataset for predicting a target variable.  
                                    It measures the mutual information between each feature and the target variable, 
                                    using an entropy-based approach to quantify the amount of information that each feature provides about the target.  
                                    Features with high mutual information values are considered to be more important in predicting the target variable
                                    and features with low mutual information values are considered to be less important.  
                                    MIFS helps improve the accuracy of predictive models by identifying the most informative features to include in the model.
                                ''')
            except: 
                selected_cols_mifs = []
                st.warning(':red[**ERROR**: Mutual Information Feature Selection could not execute...please adjust your selection criteria]')
        
            # =============================================================================
            # Correlation Analysis
            # Remove Highly Correlated Features
            # =============================================================================
            try: 
                with st.sidebar:
                    with st.form('correlation analysis'):
                        # set subheader of form
                        my_subheader('🍻 Correlation Analysis', my_size=4, my_style='#08306B')
                        # set slider threshold for correlation strength
                        corr_threshold = st.slider("*Select Correlation Threshold*", 
                                                   min_value=0.0, 
                                                   max_value=1.0, 
                                                   value=0.8, 
                                                   step=0.05, 
                                                   help='Set `Correlation Threshold` to determine which pair(s) of variables in the dataset are strongly correlated e.g. no correlation = 0, perfect correlation = 1')
            
                        models = {'Linear Regression': LinearRegression(), 'Random Forest Regressor': RandomForestRegressor(n_estimators=100)}
                        selected_corr_model = st.selectbox('*Select **model** for computing **importance scores** for highly correlated feature pairs, to drop the **least important** feature of each pair which is highly correlated*:', list(models.keys()))
                        col1, col2, col3 = st.columns([4,4,4])
                        with col2:       
                            corr_btn = st.form_submit_button("Submit", type="secondary")
                            
                with st.expander('🍻 Correlation Analysis', expanded=True):
                    st.write("")
                    # Display output
                    my_subheader(f'Pairwise Correlation')
                    col1,col2,col3 = st.columns([5,3,5])
                    with col2:
                        st.caption(f'with threshold >={corr_threshold*100:.0f}%')
                    ################################################################
                    # PLOT HEATMAP WITH PAIRWISE CORRELATION OF INDEPENDENT FEATURES.
                    ################################################################
                    # generate correlation heatmap for independent features based on threshold from slider set by user e.g. default to 0.8
                    correlation_heatmap(X, correlation_threshold=corr_threshold)
                    #################################
                    # END HEATMAP CODE 
                    #################################
                    # get the indices of the highly correlated features
                    corr_matrix = X.corr()
                    indices = np.where(abs(corr_matrix) >= corr_threshold)
                
                    # create a dataframe with the pairwise correlation values above the threshold
                    df_pairwise = pd.DataFrame({
                                                    'feature1': corr_matrix.columns[indices[0]],
                                                    'feature2': corr_matrix.columns[indices[1]],
                                                    'correlation': corr_matrix.values[indices]
                                                })
                    
                    ############################
                    # filter out duplicate pairs
                    ############################
                    # Sort feature pairs and drop duplicates
                    df_pairwise = df_pairwise.assign(sorted_features=df_pairwise[['feature1', 'feature2']].apply(sorted, axis=1).apply(tuple))
                    df_pairwise = df_pairwise.loc[df_pairwise['feature1'] != df_pairwise['feature2']].drop_duplicates(subset='sorted_features').drop(columns='sorted_features')
                    
                    # Sort by correlation and format output
                    df_pairwise = df_pairwise.sort_values(by='correlation', ascending=False).reset_index(drop=True)
                    df_pairwise['correlation'] = (df_pairwise['correlation']*100).apply('{:.2f}%'.format)
                    
                    # Display message with pairs in total_features
                    if df_pairwise.empty:
                        st.info(f'There are no **pairwise combinations** in the selected features with **correlation** larger than or equal to the user defined threshold of **{corr_threshold*100:.0f}%**')
                        st.write("")
                    else:
                        st.markdown(f' <center> The following pairwise combinations of features have a correlation >= threshold: </center>', unsafe_allow_html=True)      
                        st.write("")
                        st.dataframe(df_pairwise, use_container_width=True)
                        download_csv_button(df_pairwise, my_file="pairwise_correlation.csv", help_message='Download pairwise correlation to .CSV', set_index=False)
                    st.markdown('---')
                    # Find pairs in total_features
                    total_features = np.unique(selected_cols_rfe + selected_cols_pca + selected_cols_mifs)
                    # convert to list
                    total_features = total_features.tolist()
            
                    pairwise_features = list(df_pairwise[['feature1', 'feature2']].itertuples(index=False, name=None))
                    pairwise_features_in_total_features = [pair for pair in pairwise_features if pair[0] in total_features and pair[1] in total_features]
                    
                    # IMPORTANCE SCORES
                    # create estimator based on user selected model            
                    estimator = models[selected_corr_model]
                    estimator.fit(X, y)
                        
                    # Compute feature importance scores or permutation importance scores
                    importance_scores = compute_importance_scores(X, y, estimator)
                
                    ######################
                    # ALTAIR CHART
                    ######################
                    charts = []
                    # Set title font style and size
                    title_font = "Helvetica"
                    title_font_size = 12
        
                    num_features = len(total_features)
                    num_cols = min(3, num_features)
                    num_rows = math.ceil(num_features / num_cols)
                    
                    for i, (feature1, feature2) in enumerate(pairwise_features_in_total_features):
                        if feature1 in total_features and feature2 in total_features:
                            score1 = importance_scores[feature1]
                            score2 = importance_scores[feature2]
                            data = pd.DataFrame({'Feature': [feature1, feature2], 'Score': [score1, score2]})
                    
                            if score1 > score2:
                                total_features.remove(feature2)
                                chart = alt.Chart(data).mark_bar().encode(
                                    x='Score:Q',
                                    y=alt.Y('Feature:O', sort='-x'),
                                    color=alt.condition(
                                        alt.datum.Feature == feature2,
                                        alt.value('#FFB6C1'),
                                        alt.value('#90EE90')
                                    )
                                ).properties(width=100, height=100, title=chart_title("Removing", feature2))
                            elif score2 > score1:
                                total_features.remove(feature1)
                                chart = alt.Chart(data).mark_bar().encode(
                                    x='Score:Q',
                                    y=alt.Y('Feature:O', sort='-x'),
                                    color=alt.condition(
                                        alt.datum.Feature == feature1,
                                        alt.value('#FFB6C1'),
                                        alt.value('#90EE90')
                                    )
                                ).properties(width=100, height=100, title=chart_title("Removing", feature1))
                            else:
                                total_features.remove(feature1)
                                chart = alt.Chart(data).mark_bar().encode(
                                    x='Score:Q',
                                    y=alt.Y('Feature:O', sort='-x'),
                                    color=alt.condition(
                                        alt.datum.Feature == feature1,
                                        alt.value('#FFB6C1'),
                                        alt.value('#90EE90')
                                    )
                                ).properties(width=100, height=100, title=chart_title("Removing", feature1))
                    
                            charts.append(chart)
                    
                    # Combine all charts into a grid
                    grid_charts = []
                    for i in range(num_rows):
                        row_charts = []
                        for j in range(num_cols):
                            idx = i*num_cols+j
                            if idx < len(charts):
                                row_charts.append(charts[idx])
                        if row_charts:
                            grid_charts.append(alt.hconcat(*row_charts))
                    
                    grid_chart = alt.vconcat(*grid_charts, spacing=10)
                    # title of altair graph of feature importance scores
                    my_subheader("Removing Highly Correlated Features")
                    col1,col2,col3 = st.columns([5.5,4,5])
                    with col2:
                        st.caption(f'pair-wise features >={corr_threshold*100:.0f}%')
                    # show altair chart with pairwise correlation importance scores and in red lowest and green highest
                    st.altair_chart(grid_chart, use_container_width=True)
                    ### END CODE ALTAIR CHART
                    ##############################################################################################################
            except:
                st.warning(':red[**ERROR**: Error with Correlation Analysis...please adjust your selection criteria]')
        
            # =============================================================================
            # Top features
            # =============================================================================
            with st.sidebar:        
                with st.form('top_features'):
                    my_subheader('Selected Features 🟡🟢🟣 ', my_size=4, my_style='#52B57F')
                    # combine list of features selected from feature selection methods and only keep unique features excluding duplicate features
                    #total_features = np.unique(selected_cols_rfe + selected_cols_pca + selected_cols_mifs)
                    
                    # combine 3 feature selection methods and show to user in multi-selectbox to adjust as needed
                    feature_selection_user = st.multiselect("favorite features", list(total_features), list(total_features), label_visibility="collapsed")
                    col1, col2, col3 = st.columns([4,4,4])
                    with col2:       
                        top_features_btn = st.form_submit_button("Submit", type="secondary")
                        
            ######################################################################################################
            # redefine dynamic user picked features for X,y, X_train, X_test, y_train, y_test
            ######################################################################################################
            X = X.loc[:, feature_selection_user]
            y = y
            X_train = X[:(len(df)-my_insample_forecast_steps)]
            X_test = X[(len(df)-my_insample_forecast_steps):]
            # set endogenous variable train/test split
            y_train = y[:(len(df)-my_insample_forecast_steps)]
            y_test = y[(len(df)-my_insample_forecast_steps):]           
        
            with st.expander('🥇 Top Features Selected', expanded=True):
                my_subheader('')
                my_subheader('Your Feature Selection ', my_size=4, my_style='#7B52AB')
                # create dataframe from list of features and specify column header
                df_total_features = pd.DataFrame(total_features, columns = ['Top Features'])
                st.dataframe(df_total_features, use_container_width=True)
                # display the dataframe in streamlit
                st.dataframe(X, use_container_width=True)
                # create download button for forecast results to .csv
                download_csv_button(X, my_file="features_dataframe.csv", help_message="Download your **features** to .CSV")
        
        ###############################################################################
        # 7. Train Models
        ###############################################################################
        with tab7:
            ################################################
            # Create a User Form to Select Model(s) to train
            ################################################
            with st.sidebar:
                my_title("7. Train Models 🔢", "#0072B2")
            with st.sidebar.form('model_train_form'):
                # generic graph settings
                my_conf_interval = st.slider("*Set Confidence Interval (%)*", min_value=1, max_value=99, value=80, step=1, help='A confidence interval is a range of values around a sample statistic, such as a mean or proportion, which is likely to contain the true population parameter with a certain degree of confidence. The level of confidence is typically expressed as a percentage, such as 95%, and represents the probability that the true parameter lies within the interval. A wider interval will generally have a higher level of confidence, while a narrower interval will have a lower level of confidence.')
                
                # define all models you want user to choose from
                models = [('Naive Model', None),
                          ('Linear Regression', LinearRegression(fit_intercept=True)), 
                          ('SARIMAX', SARIMAX(y_train)),
                          ('Prophet', Prophet())]
                # create a checkbox for each model
                selected_models = []
        
                for model_name, model in models:
                    if st.checkbox(model_name):
                        selected_models.append((model_name, model))
                    if model_name == "Naive Model":
                        custom_lag_value = None
                        with st.expander('Naive Model Hyperparameters'):
                            lag = st.selectbox('*Select seasonal **lag** for the Naive Model:*', ['None', 'Day', 'Week', 'Month', 'Year', 'Custom'])
                            if lag == 'None':
                                lag = None
                            elif lag == 'Custom':
                                lag = lag.lower()
                                custom_lag_value = st.number_input("*If seasonal **lag** set to Custom, please set lag value (in days):*", value=5)
                                if custom_lag_value != "":
                                    custom_lag_value = int(custom_lag_value)
                                else:
                                    custom_lag_value = None
                            else:
                                # lag is lowercase string of selection from user in selectbox
                                lag = lag.lower()
                    if model_name == "SARIMAX":
                        with st.expander('SARIMAX Hyperparameters', expanded=False):
                            col1, col2, col3 = st.columns([5,1,5])
                            with col1:
                                p = st.number_input("Autoregressive Order (p):", value=1, min_value=0, max_value=10)
                                d = st.number_input("Differencing (d):", value=1, min_value=0, max_value=10)
                                q = st.number_input("Moving Average (q):", value=1, min_value=0, max_value=10)   
                            with col3:
                                P = st.number_input("Seasonal Autoregressive Order (P):", value=1, min_value=0, max_value=10)
                                D = st.number_input("Seasonal Differencing (D):", value=1, min_value=0, max_value=10)
                                Q = st.number_input("Seasonal Moving Average (Q):", value=1, min_value=0, max_value=10)
                                s = st.number_input("Seasonal Periodicity (s):", value=7, min_value=1)
                            st.caption('SARIMAX Hyperparameters')
                            col1, col2, col3 = st.columns([5,1,5])
                            with col1:
                                # Add a selectbox for selecting enforce_stationarity
                                enforce_stationarity = st.selectbox('Enforce Stationarity', [True, False], index=0 )
                            with col3:
                                # Add a selectbox for selecting enforce_invertibility
                                enforce_invertibility = st.selectbox('Enforce Invertibility', [True, False], index=0)
                    if model_name == "Prophet":
                        with st.expander('Prophet Hyperparameters', expanded=False):
                            horizon_option = int(st.slider('Set Forecast Horizon (default = 30 Days):', min_value=1, max_value=365, step=1, value=30, help='The horizon for a Prophet model is typically set to the number of time periods that you want to forecast into the future. This is also known as the forecasting horizon or prediction horizon.'))
                            changepoint_prior_scale = st.slider("changepoint_prior_scale", min_value=0.001, max_value=1.0, value=0.05, step=0.01, help='This is probably the most impactful parameter. It determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints. As described in this documentation, if it is too small, the trend will be underfit and variance that should have been modeled with trend changes will instead end up being handled with the noise term. If it is too large, the trend will overfit and in the most extreme case you can end up with the trend capturing yearly seasonality. The default of 0.05 works for many time series, but this could be tuned; a range of [0.001, 0.5] would likely be about right. Parameters like this (regularization penalties; this is effectively a lasso penalty) are often tuned on a log scale.')
                            seasonality_mode = str(st.selectbox("seasonality_mode", ["additive", "multiplicative"], index=1))
                            seasonality_prior_scale = st.slider("seasonality_prior_scale", min_value=0.010, max_value=10.0, value=1.0, step=0.1)
                            holidays_prior_scale = st.slider("holidays_prior_scale", min_value=0.010, max_value=10.0, value=1.0, step=0.1)
                            yearly_seasonality = st.selectbox("yearly_seasonality", [True, False], index=0)
                            weekly_seasonality = st.selectbox("weekly_seasonality", [True, False], index=0)
                            daily_seasonality = st.selectbox("daily_seasonality", [True, False], index=0)
                            interval_width = int(my_conf_interval/100)
                    else:
                        st.sidebar.empty()
                
                col1, col2, col3 = st.columns([4,4,4])
                with col2: 
                    train_models_btn = st.form_submit_button("Submit", type="secondary")
            
        #if uploaded_file is not None:
            my_title("7. Train Models 🔢", "#0072B2")
            # if nothing is selected by user display message to user to select models to train
            if not train_models_btn and not selected_models:
                st.info("👈 Select your models to train in the sidebar!🏋️‍♂️") 
            # the code block to train the selected models will only be executed if both the button has been clicked and the list of selected models is not empty.
            elif not selected_models:
                st.warning("👈 Please select at least 1 model to train from the sidebar, when pressing the **\"Submit\"** button!🏋️‍♂️")
            
            ############################################################################### 
            # 8. Evaluate Model Performance
            ###############################################################################               
            with tab7:
                # define variables needed
                # create a list of independent variables selected by user prior used 
                # for results dataframe when evaluating models which variables were included.
                features_str = get_feature_list(X)
                with st.sidebar:
                    my_title("8. Evaluate Model Performance 🎯", "#2CB8A1")
                    # if nothing is selected by user display message to user to select models to train
                    if not train_models_btn and selected_models:
                        st.info("ℹ️ Train your models first, before results show here!")
                if not train_models_btn and selected_models:
                    st.info("ℹ️ Train your models first from the sidebar menu by pressing the **'Submit'** button, before results show here!")
                if train_models_btn and selected_models:
                    st.info("You can always retrain your models and adjust hyperparameters!")

                    # iterate over all models and if user selected checkbox for model the model(s) is/are trained
                    for model_name, model in selected_models:
# =============================================================================
#                         try:
# =============================================================================
                        if model_name == "Naive Model":
                            with st.expander('📈' + model_name, expanded=True):
                                df_preds = evaluate_regression_model(model, X_train, y_train, X_test, y_test, lag=lag, custom_lag_value=custom_lag_value)
                                display_my_metrics(df_preds, "Naive Model")
                                # plot graph with actual versus insample predictions
                                plot_actual_vs_predicted(df_preds, my_conf_interval)
                                # show the dataframe
                                st.dataframe(df_preds.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Percentage_Diff': '{:.2%}', 'MAPE': '{:.2%}'}), use_container_width=True)
                                # create download button for forecast results to .csv
                                download_csv_button(df_preds, my_file="insample_forecast_naivemodel_results.csv", help_message="Download your **Naive** model results to .CSV")
                                mape, rmse, r2 = my_metrics(df_preds, model_name=model_name)
                                # add test-results to sidebar Model Test Results dataframe
                                new_row = {'model_name': 'Naive Model',
                                           'mape': '{:.2%}'.format(metrics_dict['Naive Model']['mape']),
                                           'rmse': '{:.2f}'.format(metrics_dict['Naive Model']['rmse']),
                                           'r2': '{:.2f}'.format(metrics_dict['Naive Model']['r2']),
                                           'features':features_str}
                                results_df = pd.concat([results_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# =============================================================================
#                         except:
#                             st.warning(f'Naive Model failed to train, please check parameters set in the sidebar: lag={lag}, custom_lag_value={lag}')
# =============================================================================
                        try:
                           if model_name == "Linear Regression":
                                # train the model
                                create_streamlit_model_card(X_train, y_train, X_test, y_test, results_df, model=model, model_name=model_name)
                                # append to sidebar table the results of the model train/test
                                new_row = {'model_name': 'Linear Regression',
                                           'mape': '{:.2%}'.format(metrics_dict['Linear Regression']['mape']),
                                           'rmse': '{:.2f}'.format(metrics_dict['Linear Regression']['rmse']),
                                           'r2': '{:.2f}'.format(metrics_dict['Linear Regression']['r2']),
                                           'features':features_str}
                                results_df = pd.concat([results_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
                        except:
                            st.warning(f'Linear Regression failed to train, please contact administrator!')
# =============================================================================
#                         try:
# =============================================================================
                        if model_name == "SARIMAX":
                            with st.expander('ℹ️ ' + model_name, expanded=True):
                                with st.spinner('This model might require some time to train... you can grab a coffee ☕ or tea 🍵'):   
                                    # Assume df is your DataFrame with boolean columns - needed for SARIMAX model that does not handle boolean, but int instead
                                    bool_cols = X_train.select_dtypes(include=bool).columns
                                    X_train.loc[:, bool_cols] = X_train.loc[:, bool_cols].astype(int)
                                    bool_cols = X_test.select_dtypes(include=bool).columns
                                    X_test.loc[:, bool_cols] = X_test.loc[:, bool_cols].astype(int)
                                    
                                    # parameters have standard value but can be changed by user
                                    preds_df = evaluate_sarimax_model(order=(p,d,q), seasonal_order=(P,D,Q,s), exog_train=X_train, exog_test=X_test, endog_train=y_train, endog_test=y_test)
                                    # show metric on streamlit page
                                    display_my_metrics(preds_df, "SARIMAX")
                                    # plot graph with actual versus insample predictions
                                    plot_actual_vs_predicted(preds_df, my_conf_interval)
                                    # show the dataframe
                                    st.dataframe(preds_df.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Percentage_Diff': '{:.2%}', 'MAPE': '{:.2%}'}), use_container_width=True)
                                    # create download button for forecast results to .csv
                                    download_csv_button(preds_df, my_file="insample_forecast_sarimax_results.csv", help_message="Download your **SARIMAX** model results to .CSV")
                                    # define metrics for sarimax model
                                    mape, rmse, r2 = my_metrics(preds_df, model_name=model_name)
                                    # display evaluation results on sidebar of streamlit_model_card
                                    new_row = {'model_name': 'SARIMAX', 
                                               'mape': '{:.2%}'.format(metrics_dict['SARIMAX']['mape']),
                                               'rmse': '{:.2f}'.format(metrics_dict['SARIMAX']['rmse']), 
                                               'r2': '{:.2f}'.format(metrics_dict['SARIMAX']['r2']),
                                               'features':features_str,
                                               'model settings': f'({p},{d},{q})({P},{D},{Q},{s})'}
                                    # get the maximum index of the current results dataframe
                                    results_df = pd.concat([results_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# =============================================================================
# =============================================================================
#                         except:
#                             st.warning(f'SARIMAX failed to train, please contact administrator!')       
# =============================================================================
                        if model_name == "Prophet": 
                            with st.expander('ℹ️ ' + model_name, expanded=True):
                                # use custom fucntion that creates in-sample prediction and return a dataframe with 'Actual', 'Predicted', 'Percentage_Diff', 'MAPE' 
                                preds_df_prophet = predict_prophet(y_train,
                                                                   y_test, 
                                                                   changepoint_prior_scale=changepoint_prior_scale,
                                                                   seasonality_mode=seasonality_mode,
                                                                   seasonality_prior_scale=seasonality_prior_scale,
                                                                   holidays_prior_scale=holidays_prior_scale,
                                                                   yearly_seasonality=yearly_seasonality,
                                                                   weekly_seasonality=weekly_seasonality,
                                                                   daily_seasonality=daily_seasonality,
                                                                   interval_width=interval_width)
                                
                                display_my_metrics(preds_df_prophet, "Prophet")
                                # plot graph with actual versus insample predictions
                                plot_actual_vs_predicted(preds_df_prophet, my_conf_interval)
                                # show the dataframe
                                st.dataframe(preds_df_prophet.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Percentage_Diff': '{:.2%}', 'MAPE': '{:.2%}'}), use_container_width=True)
                                # create download button for forecast results to .csv
                                download_csv_button(preds_df_prophet, my_file="insample_forecast_prophet_results.csv", help_message="Download your **Prophet** model results to .CSV")
                                # define metrics for sarimax model
                                mape, rmse, r2 = my_metrics(preds_df_prophet, model_name=model_name)
                                # display evaluation results on sidebar of streamlit_model_card
# =============================================================================
#                                 results_df = results_df.append({'model_name': 'Prophet', 
#                                                                 'mape': '{:.2%}'.format(metrics_dict['Prophet']['mape']),
#                                                                 'rmse': '{:.2f}'.format(metrics_dict['Prophet']['rmse']), 
#                                                                 'r2': '{:.2f}'.format(metrics_dict['Prophet']['r2']),
#                                                                 'features':features_str,
#                                                                 'model settings': f' changepoint_prior_scale: {changepoint_prior_scale}, seasonality_prior_scale: {seasonality_prior_scale}, holidays_prior_scale: {holidays_prior_scale}, yearly_seasonality: {yearly_seasonality}, weekly_seasonality: {weekly_seasonality}, daily_seasonality: {daily_seasonality}, interval_width: {interval_width}'}, ignore_index=True)
# =============================================================================
                                new_row = {'model_name': 'Prophet', 
                                           'mape': '{:.2%}'.format(metrics_dict['Prophet']['mape']),
                                           'rmse': '{:.2f}'.format(metrics_dict['Prophet']['rmse']), 
                                           'r2': '{:.2f}'.format(metrics_dict['Prophet']['r2']),
                                           'features': features_str,
                                           'model settings': f' changepoint_prior_scale: {changepoint_prior_scale}, seasonality_prior_scale: {seasonality_prior_scale}, holidays_prior_scale: {holidays_prior_scale}, yearly_seasonality: {yearly_seasonality}, weekly_seasonality: {weekly_seasonality}, daily_seasonality: {daily_seasonality}, interval_width: {interval_width}'}
                                
                                results_df = pd.concat([results_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
                            # MODEL DOCUMENTATION
                            st.write('')
                            my_title('Model Documentation💡')
                            with st.expander('🗒️ Naive Model', expanded=False):
                                st.markdown('''
                                            The `Naive Model` is one of the simplest forecasting models in time series analysis. 
                                            It assumes that the value of a variable at any given time is equal to the value of the variable at the previous time period. 
                                            This means that this model is a special case of an **A**uto**R**egressive model of order 1, also known as $AR(1)$.  
                                            
                                            The Naive Model is useful as a baseline model to compare more complex forecasting models, such as ARIMA, exponential smoothing, or machine learning algorithms. 
                                            It is also useful when the underlying data generating process is highly unstable or unpredictable, and when there is no trend, seasonality, or other patterns to capture.
                                            The Naive Model can be expressed as a simple equation:
                                            
                                            $\hat{y}_{t} = y_{t-1}$
                                            
                                            where:
                                            - $y_t$ is the value of the variable at time $t$
                                            - $y_{t-1}$ is the value of the variable at time $_{t-1}$.  
                                            
                                            The Naive Model can be extended to incorporate seasonal effects, by introducing a lag period corresponding to the length of the seasonal cycle. 
                                            For example, if the time series has a weekly seasonality, the Naive Model with a lag of one week is equivalent to the model with a lag of seven days, and is given by:
                                            
                                            $\hat{y}_{t} = y_{t-7}$
                                            
                                            where:
                                            - $y_t$ is the value of the variable at time $t$
                                            - $y_{t-7}$is the value of the variable at time $_{t-7}$ (i.e., one week ago).
                                            
                                            In general, the lag value for the seasonal Naive Model should be determined based on the length of the seasonal cycle in the data, and can be estimated using visual inspection, autocorrelation analysis, or domain knowledge.
                                            ''')
                            with st.expander('🗒️ Linear Regression', expanded=False):
                                st.markdown('''
                                            `Linear regression` is a statistical method used to analyze the relationship between a dependent variable and one or more independent variables. 
                                            It involves finding a line or curve that best fits the data and can be used to make predictions. 
                                            The method assumes that the relationship between the variables is linear and that errors are uncorrelated.
                                            
                                            To find this line, we use a technique called least squares regression, which involves finding the line that minimizes the sum of the squared differences between the predicted values and the actual values. 
                                            The line is described by the equation:
                                            
                                            $$\\large Y = \\beta_0 + \\beta_1 X$$
                                        
                                            where:
                                            - $Y$ is the dependent variable
                                            - $X$ is the independent variable
                                            - $\\beta_0$ is the intercept (the value of $Y$ when $X = 0$)
                                            - $\\beta_1$ is the slope (the change in $Y$ for a unit change in $X$)
                                            ''')
                            with st.expander('🗒️ SARIMAX', expanded=False):
                                st.markdown('''
                                            `SARIMAX`, or **S**easonal **A**utoregressive **I**ntegrated **M**oving **A**verage with e**X**ogenous variables, is a popular time series forecasting model.
                                            The ARIMA model is a time series forecasting model that uses past values of a variable to predict future values. 
                                            SARIMAX extends ARIMA by incorporating seasonal patterns and adding exogenous variables that can impact the variable being forecasted.
                                            
                                            - **p:** The order of the autoregressive (AR) term, which refers to the number of lagged observations of the dependent variable in the model. A higher value of p means the model is considering more past values of the dependent variable.
                                            - **d:** The order of the differencing (I) term, which refers to the number of times the data needs to be differenced to make it stationary. Stationarity is a property of time series data where the statistical properties, such as the mean and variance, are constant over time.
                                            - **q:** The order of the moving average (MA) term, which refers to the number of lagged forecast errors in the model. A higher value of q means the model is considering more past forecast errors.
                                            - **P:** The seasonal order of the autoregressive term, which refers to the number of seasonal lags in the model.
                                            - **D:** The seasonal order of differencing, which refers to the number of times the data needs to be differenced at the seasonal lag to make it stationary.
                                            - **Q:** The seasonal order of the moving average term, which refers to the number of seasonal lags of the forecast errors in the model.
                                            - **s:** The length of the seasonal cycle, which is the number of time steps in each season. For example, if the data is monthly and the seasonality is yearly, s would be 12. The parameter s is used to determine the number of seasonal lags in the model.
                                            - ***Exogenous Variables***: These are external factors that can impact the variable being forecasted. They are included in the model as additional inputs.  
                                            ''')
                            with st.expander('🗒️ Prophet', expanded=False):
                                st.markdown('''
                                            The Facebook `Prophet` model is a popular open-source library for time series forecasting developed by Facebook's Core Data Science team.
                                            It is designed to handle time series data with strong seasonal effects and other external factors.
                                            It uses a combination of historical data and user-defined inputs to generate forecasts for future time periods.  
                                            
                                            ## Variables in the Prophet Model
                                            The main variables in the Prophet model are:
                                            - **Trend**: This is the underlying pattern in the data that represents the long-term direction of the series. It can be linear or non-linear and is modeled using a piecewise linear function.
                                            - **Seasonality**: This is the periodic pattern in the data that repeats over fixed time intervals. It can be daily, weekly, monthly, or yearly, and is modeled using Fourier series.
                                            - **Holidays**: These are user-defined events or time periods that are known to affect the time series. The model includes them as additional regressors in the forecasting equation.
                                            - **Regressors**: These are additional time-varying features that can affect the time series, such as weather, economic indicators, or other external factors.
                                            
                                            ## Math Behind the Prophet Model
                                            The math behind the Prophet model involves fitting a Bayesian additive regression model to the time series data. The model is formulated as follows:
                                            
                                            $$y_t = g_t + s_t + h_t + e_t$$
                                            
                                            where:
                                            - $y_t$ is the observed value at time $t$
                                            - $g_t$ is the trend component
                                            - $s_t$ is the seasonality component
                                            - $h_t$ is the holiday component
                                            - $e_t$ is the error term. 
                                            
                                            The **trend** component is modeled using a piecewise linear function, while the **seasonality component** is modeled using a Fourier series. The **holiday component** and any additional regressors are included as additional terms in the regression equation.
                                            
                                            The model is estimated using a Bayesian approach that incorporates prior information about the parameters and allows for uncertainty in the forecasts. The parameters are estimated using Markov Chain Monte Carlo (MCMC) sampling, which generates a large number of possible parameter values and uses them to estimate the posterior distribution of the parameters. The posterior distribution is then used to generate forecasts for future time periods.
                                            
                                            Overall, the Prophet model is a powerful tool for time series forecasting that can handle complex data patterns and external factors. Its flexible modeling approach and Bayesian framework make it a popular choice for many data scientists and analysts.
                            
                                            ''')
                    ###################################################################################################################
                    # Add results_df to session state
                    ###################################################################################################################
                    with st.sidebar:
                        with st.expander('', expanded=True):
                            # table 1: latest run results of model performance
                            my_subheader('Latest Model Test Results', my_size=4, my_style='#2CB8A1')
                            if 'results_df' not in st.session_state:
                                st.session_state.results_df = results_df
                            else:
                                #st.session_state.results_df = st.session_state.results_df.append(results_df, ignore_index=True)
                                st.session_state.results_df = pd.concat([st.session_state.results_df, results_df], ignore_index=True)
                            # Show the results dataframe in the sidebar if there is at least one model selected
                            if len(selected_models) > 0:
                                st.dataframe(results_df)
                            # table 2: ranking
                            my_subheader('Top 3 Ranking All Test Results', my_size=4, my_style='#2CB8A1')
                            # It converts the 'mape' column to floats, removes duplicates based on the 'model_name' and 'mape' columns, sorts the unique DataFrame by ascending 'mape' values, selects the top 3 rows, and displays the resulting DataFrame in Streamlit.
                            test_df = st.session_state.results_df.assign(mape=st.session_state.results_df['mape'].str.rstrip('%').astype(float)).drop_duplicates(subset=['model_name', 'mape']).sort_values(by='mape', ascending=True).iloc[:3]
                            st.write(test_df)
                
            # show on streamlit page the scoring results
            with tab8:
                my_title("8. Evaluate Model Performance 🎯", "#2CB8A1")
                # if the results dataframe is created already then you can continue code
                if 'results_df' in globals():
                    with st.expander('ℹ️ Test Results', expanded=True):
                        my_header(my_string='Modeling Test Results', my_style="#2CB8A1")
                        # create some empty newlines - for space between title and dataframe
                        st.write("")
                        st.write("")
                        # show the dataframe with all the historical results stored from 
                        # prior runs the cache e.g. called session_state in streamlit
                        # from all train/test results with models selected by user
                        st.write(st.session_state.results_df)
                        # download button
                        download_csv_button(results_df, my_file="Modeling Test Results.csv", help_message="Download your Modeling Test Results to .CSV")
                # if results_df is not created yet just tell user to train models first
                else:
                    st.info('Please select models to train first from sidebar menu and press **"Submit"**')
                 
            ###########################################################################################################################
            # 9. Hyper-parameter tuning
            ###########################################################################################################################
            with tab9:
                my_title('9. Hyper-parameter Tuning⚙️', "#88466D")
                # set variables needed
                ######################
                # set initial start time before hyper-parameter tuning is kicked-off
                start_time = time.time()
                # initialize variable for sarimax parameters p,d,q
                param_mini = None
                # initialize variable for sarimax model parameters P,D,Q,s
                param_seasonal_mini = None
            
                # sidebar hyperparameter tuning
                ################
                with st.sidebar:
                     my_title('9. Hyper-parameter Tuning⚙️', "#88466D")                    
                     with st.form("hyper_parameter_tuning"):
                         # create a multiselect checkbox with all model names selected by default
                         # create a list of the selected models by the user in the training section of the streamlit app
                         model_lst = [model_name for model_name, model in selected_models]
                         # SELECT MODEL(S): let the user select the trained model(s) in a multi-selectbox for hyper-parameter tuning
                         selected_model_names = st.multiselect('*Select Models*', model_lst, help='Selected Models are tuned utilizing **`Grid Search`**, which is a specific technique for hyperparameter tuning where a set of hyperparameters is defined and the model is trained and evaluated on all possible combinations')
                         # SELECT EVALUATION METRIC: let user set evaluation metric for the hyper-parameter tuning
                         metric = st.selectbox('*Select Evaluation Metric*', ['AIC', 'BIC', 'RMSE'], label_visibility='visible', 
                                               help='**`AIC`** (**Akaike Information Criterion**): A measure of the quality of a statistical model, taking into account the goodness of fit and the complexity of the model. A lower AIC indicates a better model fit. \
                                               \n**`BIC`** (**Bayesian Information Criterion**): Similar to AIC, but places a stronger penalty on models with many parameters. A lower BIC indicates a better model fit.  \
                                               \n**`RMSE`** (**Root Mean Squared Error**): A measure of the differences between predicted and observed values in a regression model. It is the square root of the mean of the squared differences between predicted and observed values. A lower RMSE indicates a better model fit.')
                         # Note that we set best_metric to -np.inf instead of np.inf since we want to maximize the R2 metric. 
                         if metric in ['AIC', 'BIC', 'RMSE']:
                             mini = float('+inf')
                         else:
                             # Set mini to positive infinity to ensure that the first value evaluated will become the minimum
                             # minimum metric score will be saved under variable mini while looping thorugh parameter grid
                             mini = float('-inf')
                         # SARIMAX HYPER-PARAMETER GRID TO SELECT BY USER
                         ##################################################
                         with st.expander('SARIMAX GridSearch Parameters'):
                             col1, col2, col3 = st.columns([5,1,5])
                             with col1:
                                 p_max = st.number_input("*Max Autoregressive Order (p):*", value=2, min_value=0, max_value=10)
                                 d_max = st.number_input("*Max Differencing (d):*", value=1, min_value=0, max_value=10)
                                 q_max = st.number_input("*Max Moving Average (q):*", value=2, min_value=0, max_value=10)   
                             with col3:
                                 P_max = st.number_input("*Max Seasonal Autoregressive Order (P):*", value=2, min_value=0, max_value=10)
                                 D_max = st.number_input("*Max Seasonal Differencing (D):*", value=1, min_value=0, max_value=10)
                                 Q_max = st.number_input("*Max Seasonal Moving Average (Q):*", value=2, min_value=0, max_value=10)
                                 s = st.number_input("*Set Seasonal Periodicity (s):*", value=7, min_value=1)
                         # PROPHET HYPER-PARAMETER GRID TO SELECT BY USER
                         ##################################################
                         with st.expander('Prophet GridSearch Parameters'):
                            st.write('')
                            col1, col2 = st.columns([5,1])
                            with col1:
                                # PROPHET MODEL HYPER-PARAMETER GRID WITH DOCUMENTATION
                                # usually set to forecast horizon e.g. 30 days
                                horizon_option = int(st.slider('Set Forecast Horizon (default = 30 Days):', min_value=1, max_value=365, step=1, value=30, help='The horizon for a Prophet model is typically set to the number of time periods that you want to forecast into the future. This is also known as the forecasting horizon or prediction horizon.'))
                                # This is probably the most impactful parameter. It determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints. As described in this documentation, if it is too small, the trend will be underfit and variance that should have been modeled with trend changes will instead end up being handled with the noise term. If it is too large, the trend will overfit and in the most extreme case you can end up with the trend capturing yearly seasonality. The default of 0.05 works for many time series, but this could be tuned; a range of [0.001, 0.5] would likely be about right. Parameters like this (regularization penalties; this is effectively a lasso penalty) are often tuned on a log scale.
                                changepoint_prior_scale_options = st.multiselect('*Changepoint Prior Scale*', [0.001, 0.01, 0.05, 0.1, 1], default = [0.001, 0.01, 0.1, 1], help='This is probably the most impactful parameter. It determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints. As described in this documentation, if it is too small, the trend will be underfit and variance that should have been modeled with trend changes will instead end up being handled with the noise term. If it is too large, the trend will overfit and in the most extreme case you can end up with the trend capturing yearly seasonality. The default of 0.05 works for many time series, but this could be tuned; a range of [0.001, 0.5] would likely be about right. Parameters like this (regularization penalties; this is effectively a lasso penalty) are often tuned on a log scale.')
                                # Options are ['additive', 'multiplicative']. Default is 'additive', but many business time series will have multiplicative seasonality. 
                                # This is best identified just from looking at the time series and seeing if the magnitude of seasonal fluctuations grows with the magnitude of the time series
                                seasonality_mode_options = st.multiselect('*Seasonality Modes*', [ 'additive', 'multiplicative'], default = [ 'additive', 'multiplicative'])
                                seasonality_prior_scale_options = st.multiselect('*Seasonality Prior Scales*', [0.01, 0.1, 1.0, 10.0], default = [0.01, 0.1, 1.0, 10.0])
                                holidays_prior_scale_options = st.multiselect('*Holidays Prior Scales*', [0.01, 0.1, 1.0, 10.0], default = [0.01, 0.1, 1.0, 10.0])
                                yearly_seasonality_options = st.multiselect('*Yearly Seasonality*', [True, False], default = [True, False])
                                weekly_seasonality_options = st.multiselect('*Weekly Seasonality*', [True, False], default = [True, False])
                                daily_seasonality_options = st.multiselect('*Daily Seasonality*', [True, False], default = [True, False])
                                #interval_width=interval_width
                                    
                         sarimax_tuning_results = pd.DataFrame(columns=['SARIMAX (p,d,q)x(P,D,Q,s)', 'param', 'param_seasonal', metric])
                         prophet_tuning_results = pd.DataFrame() # TEST
                         # create vertical spacing columns
                         col1, col2, col3 = st.columns([4,4,4])
                         with col2: 
                             # create submit button for the hyper-parameter tuning
                             hp_tuning_btn = st.form_submit_button("Submit", type="secondary")    
                      
                # if user clicks the hyper-parameter tuning button run code below
                if hp_tuning_btn == True and selected_model_names:
                    # Set up a progress bar
                    with st.spinner(f'Searching for optimal hyper-parameters...hold your horses 🐎🐎🐎 this might take a while to run!'):
                        ################################
                        # kick off the grid-search!
                        ################################
                        # set start time when grid-search is kicked-off to define total time it takes
                        # as computationaly intensive
                        start_time = time.time()
                        # if the name of the model selected by user in multi-selectbox is selected when pressing the submit button then run hyper-parameter search for the model
                        # note that naive model/linear regression are not added as they do not have hyper-parameters
                        for model_name in selected_model_names:
                            if model_name == "SARIMAX":
                                # Define the parameter grid to search
                                param_grid = {
                                                'order': [(p, d, q) for p, d, q in itertools.product(range(p_max+1), range(d_max+1), range(q_max+1))],
                                                'seasonal_order': [(p, d, q, s) for p, d, q in itertools.product(range(P_max+1), range(D_max+1), range(Q_max+1))]
                                              }
                                # Loop through each parameter combination in the parameter grid
                                for param, param_seasonal in itertools.product(param_grid['order'], param_grid['seasonal_order']):
                                        try:
                                            # Create a SARIMAX model with the current parameter values
                                            mod = SARIMAX(y,
                                                          order=param,
                                                          seasonal_order=param_seasonal,
                                                          exog=X, 
                                                          enforce_stationarity=enforce_stationarity,
                                                          enforce_invertibility=enforce_invertibility)
                                            # Fit the model to the data
                                            results = mod.fit()
                                            # Check if the current model has a lower AIC than the previous models
                                            if metric == 'AIC':
                                                if results.aic < mini:
                                                    # If so, update the mini value and the parameter values for the model with the lowest AIC
                                                    mini = results.aic
                                                    param_mini = param
                                                    param_seasonal_mini = param_seasonal
                                            elif metric == 'BIC':
                                                if results.bic < mini:
                                                    # If so, update the mini value and the parameter values for the model with the lowest AIC
                                                    mini = results.bic
                                                    param_mini = param
                                                    param_seasonal_mini = param_seasonal
                                            elif metric == 'RMSE':
                                                rmse = math.sqrt(results.mse)
                                                if rmse < mini:
                                                    mini = rmse
                                                    param_mini = param
                                                    param_seasonal_mini = param_seasonal
                                                                     
                                            # Append a new row to the dataframe with the parameter values and AIC score
                                            sarimax_tuning_results = sarimax_tuning_results.append({'SARIMAX (p,d,q)x(P,D,Q,s)': f'{param} x {param_seasonal}', 
                                                                           'param': param, 
                                                                           'param_seasonal': param_seasonal, 
                                                                           metric: "{:.2f}".format(mini)}, 
                                                                           ignore_index=True)
                                        # If the model fails to fit, skip it and continue to the next model
                                        except:
                                            continue
                                # set the end of runtime
                                end_time_sarimax = time.time()                  
                           
                            if model_name == "Prophet":
                                horizon_int = horizon_option
                                horizon_str = f'{horizon_int} days'  # construct horizon parameter string
                                # define cutoffs
                                cutoff_date = X_train.index[-(horizon_int+1)].strftime('%Y-%m-%d')
                                # define the parameter grid - user can select options with multi-select box in streamlit app
                                param_grid = {  
                                                'changepoint_prior_scale': changepoint_prior_scale_options,
                                                'seasonality_prior_scale': seasonality_prior_scale_options,
                                                'changepoint_prior_scale': changepoint_prior_scale_options,
                                                'seasonality_mode': seasonality_mode_options,
                                                'seasonality_prior_scale': seasonality_prior_scale_options,
                                                'holidays_prior_scale': holidays_prior_scale_options,
                                                'yearly_seasonality': yearly_seasonality_options,
                                                'weekly_seasonality': weekly_seasonality_options,
                                                'daily_seasonality': daily_seasonality_options
                                              }
                                
                                # Generate all combinations of parameters
                                all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
                                rmses = [] # Store the RMSEs for each params here
                                aics = []  # Store the AICs for each params here
                                bics = []  # Store the BICs for each params here
                                
                                # for simplicity set only a single cutoff for train/test split defined by user in the streamlit app
                                #cutoffs = pd.to_datetime(['2021-06-01', '2021-12-31']) # add list of dates 
                                cutoffs = pd.to_datetime([cutoff_date])
                                
                                # preprocess the data (y_train/y_test) for prophet model with datestamp (DS) and target (y) column
                                y_train_prophet = preprocess_data_prophet(y_train)
                                y_test_prophet = preprocess_data_prophet(y_test)
                                
                                # Use cross validation to evaluate all parameters
                                for params in all_params:
                                    m = Prophet(**params)  # Fit model with given params
                                    # train the model on the data with set parameters
                                    m.fit(y_train_prophet)
                                    
                                    # other examples of forecast horizon settings:
                                    #df_cv2 = cross_validation(m, cutoffs=cutoffs, horizon='100 days')
                                    #df_cv2 = cross_validation(m, cutoffs=cutoffs, horizon='365 days')
                                    df_cv = cross_validation(m, cutoffs=cutoffs, horizon=horizon_str, parallel=False)
                                    # rolling_window = 1 computes performance metrics using all the forecasted data to get a single performance metric number.
                                    df_p = performance_metrics(df_cv, rolling_window=1)
                                    rmses.append(df_p['rmse'].values[0])
                                    # Get residuals to compute AIC and BIC
                                    df_cv['residuals'] = df_cv['y'] - df_cv['yhat']
                                    residuals = df_cv['residuals'].values
                                    
                                    # Compute AIC and BIC
                                    nobs = len(residuals)
                                    k = len(params)
                                    loglik = -0.5 * nobs * (1 + np.log(2*np.pi) + np.log(np.sum(residuals**2)/nobs))
                                    aic = -2 * loglik + 2 * k
                                    bic = 2 * loglik + k * np.log(nobs)
                                    # add AIC score to list
                                    aics.append(aic)
                                    # add BIC score to list
                                    bics.append(bic)
                                
                                # create dataframe with parameter combinations
                                prophet_tuning_results = pd.DataFrame(all_params)
                                # add RMSE scores to dataframe
                                prophet_tuning_results['rmse'] = rmses
                                # add AIC scores to dataframe
                                prophet_tuning_results['aic'] = aics
                                # add BIC scores to dataframe
                                prophet_tuning_results['bic'] = bics
                                # set the end of runtime
                                end_time_prophet = time.time()        
                
                    # if user presses "Submit" button for hyper-parameter tuning and selected a model or multiple models in dropdown multiselectbox then run code below
                    if hp_tuning_btn == True and selected_model_names:
                        # if SARIMAX model is inside the list of modelames then run code below:
                        if 'SARIMAX' in selected_model_names:
                            with st.expander('⚙️ SARIMAX', expanded=True):                     
                                # Add a 'Rank' column based on the AIC score and sort by ascending rank
                                # rank method: min: lowest rank in the group
                                # rank method: dense: like ‘min’, but rank always increases by 1 between groups.
                                sarimax_tuning_results['Rank'] = sarimax_tuning_results[metric].rank(method='min', ascending=True).astype(int)
                                # sort values by rank
                                sarimax_tuning_results = sarimax_tuning_results.sort_values('Rank', ascending=True)
                                # show user dataframe of gridsearch results ordered by ranking
                                st.dataframe(sarimax_tuning_results.set_index('Rank'), use_container_width=True)   
                                #st.info(f"ℹ️ SARIMAX search for your optimal hyper-parameters finished in {end_time_sarimax - start_time:.2f} seconds")
                                if not sarimax_tuning_results.empty:
                                    st.write(f'🏆 **SARIMAX** set of parameters with the lowest {metric} of **{"{:.2f}".format(mini)}** found in **{end_time_sarimax - start_time:.2f}** seconds is:')  
                                    st.write(f'- **`(p,d,q)(P,D,Q,s)`**: {sarimax_tuning_results.iloc[0,1]}{sarimax_tuning_results.iloc[0,2]}')
            # =============================================================================
            #                         st.write(f'- Order (p):')
            #                         st.write('Differencing (d):')
            #                         st.write('Moving Average (q):')
            #                         st.write('Seasonal Order (P):')
            #                         st.write('Seasonal Differencing (D):')
            #                         st.write('Seasonal Moving Average (Q):')
            #                         st.write('Seasonal Periodicity (s):')
            # =============================================================================
                        if 'Prophet' in selected_model_names:
                            with st.expander('⚙️ Prophet', expanded=True):  
                                if metric.lower() in prophet_tuning_results.columns:
                                    prophet_tuning_results['Rank'] = prophet_tuning_results[metric.lower()].rank(method='min', ascending=True).astype(int)
                                    # sort values by rank
                                    prophet_tuning_results = prophet_tuning_results.sort_values('Rank', ascending=True)
                                    # show user dataframe of gridsearch results ordered by ranking
                                    st.dataframe(prophet_tuning_results.set_index('Rank'), use_container_width=True)  
                                #st.success(f"ℹ️ Prophet search for your optimal hyper-parameters finished in **{end_time_prophet - start_time:.2f}** seconds")
                                if not prophet_tuning_results.empty:
                                    st.markdown(f'🏆 **Prophet** set of parameters with the lowest {metric} of **{"{:.2f}".format(prophet_tuning_results.loc[0,metric.lower()])}** found in **{end_time_prophet - start_time:.2f}** seconds are:')
                                    st.write('\n'.join([f'- **`{param}`**: {prophet_tuning_results.loc[0, param]}' for param in prophet_tuning_results.columns[:6]]))
                else:
                    st.info('👈 Please select at least one model in the sidebar and press \"Submit\"!')
                    
            ##############################################################################
            # 10. Forecast
            ##############################################################################
            with tab10:
            #if uploaded_file is not None:
                # DEFINE VARIABLES NEEDED FOR FORECAST
                min_date = df['date'].min()
                max_date = df['date'].max()
                max_value_calendar=None
                # define maximum value in dataset for year, month day
                year = max_date.year
                month = max_date.month
                day = max_date.day
                end_date_calendar = df['date'].max()
                # end date dataframe + 1 day into future is start date of forecast
                start_date_forecast = end_date_calendar + timedelta(days=1)
                
                my_title('10. Forecast 🔮', "#48466D")   
                with st.sidebar:
                    my_title('10. Forecast 🔮', "#48466D")                    
                    with st.form("📆 "):
                        if select_dwt_features:
                            # wavelet model choice forecast
                            my_subheader('Select Model for Discrete Wavelet Feature(s) Forecast Estimates')
                            model_type_wavelet = st.selectbox('Select a model', ['Support Vector Regression', 'Linear'], label_visibility='collapsed') 
                            
                        # define all models in list as we retrain models on entire dataset anyway
                        selected_models_forecast_lst = ['Linear Regression', 'SARIMAX', 'Prophet']
                        # SELECT MODEL(S) for Forecasting
                        selected_model_names = st.multiselect('*Select Forecasting Models*', selected_models_forecast_lst, default=selected_models_forecast_lst)  
                        
                        col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 7, 7, 1, 6, 7, 1])
                        with col2: 
                            st.markdown(f'<h5 style="color: #48466D; background-color: #F0F2F6; padding: 12px; border-radius: 5px;"><center> End Date:</center></h5>', unsafe_allow_html=True)
                        with col3:
                            for model_name in selected_model_names:
                                # if model is linear regression max out the time horizon to maximum possible
                                if model_name == "Linear Regression":
                                    # max value is it depends on the length of the input data and the forecasting method used. Linear regression can only forecast as much as the input data if you're using it for time series forecasting.
                                    max_value_calendar = end_date_calendar + timedelta(days=len(df))
                                if model_name == "SARIMAX":
                                    max_value_calendar = None
                            # create user input box for entering date in a streamlit calendar widget
                            end_date_forecast = st.date_input("input forecast date", 
                                                              value=start_date_forecast,
                                                              min_value=start_date_forecast, 
                                                              max_value=max_value_calendar, 
                                                              label_visibility = 'collapsed')   
                        with col5: 
                            # set text information for dropdown frequency
                            st.markdown(f'<h5 style="color: #48466D; background-color: #F0F2F6; padding: 12px; border-radius: 5px;"><center> Frequency:</center></h5>', unsafe_allow_html=True)
                        with col6:
                            # Define a dictionary of possible frequencies and their corresponding offsets
                            forecast_freq_dict = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
                            # Ask the user to select the frequency of the data
                            forecast_freq = st.selectbox('Select the frequency of the data', list(forecast_freq_dict.keys()), label_visibility='collapsed')
                            # get the value of the key e.g. D, W, M, Q or Y
                            forecast_freq_letter = forecast_freq_dict[forecast_freq]
                      
                        # create additional linespace
                        st.write("")
                        # create vertical spacing columns
                        col1, col2, col3 = st.columns([4,4,4])
                        with col2:
                            # create submit button for the forecast
                            forecast_btn = st.form_submit_button("Submit", type="secondary")    
                   
                # when user clicks the forecast button then run below
                if forecast_btn:
                    #############################################
                    # SET VARIABLES NEEDED FOR FORECASTING MODELS
                    #############################################
                    # create a date range for your forecast
                    future_dates = pd.date_range(start=start_date_forecast, end=end_date_forecast, freq=forecast_freq_letter)
                    # first create all dates in dataframe with 'date' column
                    df_future_dates = future_dates.to_frame(index=False, name='date')
                    # add the special calendar days
                    df_future_dates = create_calendar_special_days(df_future_dates)
                    # add the year/month/day dummy variables
                    df_future_dates = create_date_features(df, year_dummies=year_dummies, month_dummies=month_dummies, day_dummies=day_dummies)
              
                    # if user wants discrete wavelet features add them
                    if select_dwt_features:
                        df_future_dates = forecast_wavelet_features(X, features_df_wavelet, future_dates, df_future_dates)
                    ##############################################
                    # DEFINE X future
                    ##############################################
                    # select only features user selected from df e.g. slice df    
                    X_future = df_future_dates.loc[:, ['date'] + [col for col in feature_selection_user if col in df_future_dates.columns]]
                    # set the 'date' column as the index again
                    X_future = copy_df_date_index(X_future, datetime_to_date=False, date_to_index=True)
                    
                    # iterate over each model name and model in list of lists
                    for model_name in selected_model_names:
                        ##############################
            # =============================================================================
            #             def add_prediction_interval(model, X_future, alpha, df):
            #                 # calculate the prediction interval for the forecast data
            #                 y_forecast = model.predict(X_future)
            #                 mse = np.mean((model.predict(model.X) - model.y) ** 2)
            #                 n = len(model.X)
            #                 dof = n - 2
            #                 t_value = stats.t.ppf(1 - alpha / 2, dof)
            #                 y_std_err = np.sqrt(mse * (1 + 1 / n + (X_future - np.mean(model.X)) ** 2 / ((n - 1) * np.var(model.X))))
            #                 lower_pi = y_forecast - t_value * y_std_err
            #                 upper_pi = y_forecast + t_value * y_std_err
            #                 # create a dataframe with the prediction interval and add it to the existing dataframe
            #                 df['lower_pi'] = lower_pi
            #                 df['upper_pi'] = upper_pi
            #                 return df
            # =============================================================================
                        ##############################
                        
                        if model_name == "Linear Regression":                
                            model = LinearRegression()
                            # train the model on all data (X) for which we have data in forecast that user feature selected
                            # e.g. if X originaly had August, but the forecast does not have August
                            model.fit(X.loc[:, [col for col in feature_selection_user if col in df_future_dates.columns]], y)
                            # forecast (y_hat with dtype numpy array)
                            y_forecast = model.predict(X_future) 
                            # convert numpy array y_forecast to a dataframe
                            df_forecast_lr = pd.DataFrame(y_forecast, columns = ['forecast']).round(0)
                            # create a dataframe with the DatetimeIndex as the index
                            df_future_dates_only = future_dates.to_frame(index=False, name='date')
                            # combine dataframe of date with y_forecast
                            df_forecast_lr = copy_df_date_index(df_future_dates_only.join(df_forecast_lr), datetime_to_date=False, date_to_index=True)
            # =============================================================================
            #                 # Add the prediction interval to the forecast dataframe
            #                 alpha = 0.05  # Level of significance for the prediction interval
            #                 df_forecast_lr = add_prediction_interval(model, X_future, alpha, df_forecast_lr)
            # =============================================================================
                            # create forecast model score card in streamlit
                            with st.expander('ℹ️' + model_name + ' Forecast', expanded=True):   
                                my_header(f'{model_name}')
                                # Create the forecast plot
                                plot_forecast(y, df_forecast_lr, title='')
                                # set note that maximum chosen date can only be up to length of input data with Linear Regression Model
                                #st.caption('Note: Linear Regression Model maximum end date depends on length of input data')
                                # show dataframe / output of forecast in streamlit linear regression
                                st.dataframe(df_forecast_lr, use_container_width=True)
                                download_csv_button(df_forecast_lr, my_file="forecast_linear_regression_results.csv") 
                        if model_name == "SARIMAX":
                            # define model parameters
                            order = (p,d,q)
                            seasonal_order = (P,D,Q,s)
                            # define model on all data (X)
                            model = SARIMAX(endog = y, order=(p, d, q),  
                                                 seasonal_order=(P, D, Q, s), 
                                                 exog=X.loc[:, [col for col in feature_selection_user if col in df_future_dates.columns]], 
                                                 enforce_invertibility=enforce_invertibility, 
                                                 enforce_stationarity=enforce_stationarity).fit()
                            # Forecast future values
                            my_forecast_steps = (end_date_forecast-start_date_forecast.date()).days
                            #y_forecast = model_fit.predict(start=start_date_forecast, end=(start_date_forecast + timedelta(days=len(X_future)-1)), exog=X_future)
                            forecast_values = model.get_forecast(steps = my_forecast_steps, exog = X_future.iloc[:my_forecast_steps,:])
            
                            # set the start date of forecasted value e.g. +7 days for new date
                            start_date = max_date + timedelta(days=1)
                            # create pandas series before appending to the forecast dataframe
                            date_series = pd.date_range(start=start_date, end=None, periods= my_forecast_steps, freq=forecast_freq_letter)
                            # create dataframe
                            df_forecast =  pd.DataFrame()
                            # add date series to forecasting pandas dataframe
                            df_forecast['date'] = date_series.to_frame(index = False)
                            # convert forecast to integers (e.g. round it)
                            df_forecast[('forecast')] = forecast_values.predicted_mean.values.astype(int).round(0)                      
                            # set 'date' as the index of the dataframe
                            df_forecast_sarimax = copy_df_date_index(df_forecast)
                            with st.expander('ℹ️ ' + model_name + ' Forecast', expanded=True):   
                                my_header(f'{model_name}')
                                # Create the forecast plot
                                plot_forecast(y, df_forecast_sarimax, title='')
                                # set note that maximum chosen date can only be up to length of input data with Linear Regression Model
                                #st.caption('Note: Linear Regression Model maximum end date depends on length of input data')
                                # show dataframe / output of forecast in streamlit linear regression
                                st.dataframe(df_forecast_sarimax, use_container_width=True)
                        if model_name == "Prophet": # NOTE: Currently no X features included in this Prophet model
                            forecast_prophet = pd.DataFrame()
                            # prep data for specificalmodelly prophet model requirements, data should have ds column and y column
                            y_prophet = preprocess_data_prophet(y)
                            # define 
                            m = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                                        seasonality_mode=seasonality_mode,
                                        seasonality_prior_scale=seasonality_prior_scale,
                                        holidays_prior_scale=holidays_prior_scale,
                                        yearly_seasonality=yearly_seasonality,
                                        weekly_seasonality=weekly_seasonality,
                                        daily_seasonality=daily_seasonality,
                                        interval_width=interval_width)
                            # train the model on the entire dataset with set parameters
                            m.fit(y_prophet)
                            # Predict on the test set
                            future = m.make_future_dataframe(periods=len(future_dates), freq='D')
                            forecast_prophet = m.predict(future)
                            # change name of yhat to forecast in df
                            forecast_prophet['forecast'] = forecast_prophet['yhat'].round(0)
                            #forecast_prophet['date'] = forecast_prophet['ds']
                            forecast_prophet['date'] = forecast_prophet['ds']
                            forecast_prophet = forecast_prophet[['date', 'forecast']]
                            # set the date column as index column
                            forecast_prophet = forecast_prophet.set_index('date')
                            # cut off the insample forecast
                            forecast_prophet= forecast_prophet[len(y):]
                            with st.expander('ℹ️ ' + model_name + ' Forecast', expanded=True):
                                my_header(f'{model_name}')
                                plot_forecast(y, forecast_prophet, title='')
                                # show dataframe / output of forecast in streamlit
                                st.dataframe(forecast_prophet, use_container_width=True)
                                download_csv_button(forecast_prophet, my_file="forecast_prophet_results.csv")