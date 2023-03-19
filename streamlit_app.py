# -*- coding: utf-8 -*-
"""
Streamlit App to Forecast Website Traffic
Created on Mon Mar 6 17:02:34 2023
@author: tholl
"""
# Import required packages
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import acf, pacf
# https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.pacf.html#statsmodels.tsa.stattools.pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
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
                   initial_sidebar_state="auto") # "auto" or "expanded" or "collapsed"

###############################################################################
# SET VARIABLES
###############################################################################
# create an empty dictionary to store the results of the models
# that I call after I train the models to display on sidebar under hedaer "Evaluate Models"
metrics_dict = {}

###############################################################################
# FUNCTIONS
###############################################################################
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
       rmse (float): Root mean squared error of the model on test data.
       r2 (float): Coefficient of determination (R-squared) of the model on test data.
       preds_df (pd.DataFrame): DataFrame of predicted and actual values on test data.
   """
   # Fit the model
   model = sm.tsa.statespace.SARIMAX(endog=endog_train, exog=exog_train, order=order, seasonal_order=seasonal_order)
   results = model.fit()
   # Generate predictions
   y_pred = results.predict(start=endog_test.index[0], end=endog_test.index[-1], exog=exog_test)
   preds_df = pd.DataFrame({'Actual': endog_test.squeeze(), 'Predicted': y_pred.squeeze()}, index=endog_test.index)
   # Calculate percentage difference between actual and predicted values and add it as a new column
   preds_df = preds_df.assign(Percentage_Diff = ((preds_df['Predicted'] - preds_df['Actual']) / preds_df['Actual']))
   # Calculate MAPE and add it as a new column
   preds_df = preds_df.assign(MAPE = abs(preds_df['Percentage_Diff']))   
   return preds_df   

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
        st.metric(':red[**MAPE:**]', delta=None, value = "{:.2%}".format(mape))
    with col2:
        st.metric(':red[**RMSE:**]', delta = None, value = round(rmse,2))
    with col3: 
        st.metric(':green[**R-squared:**]',  delta=None, value= round(r2, 2))

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
    if 'lag' in kwargs and kwargs['lag'] is not None:
        lag = kwargs['lag']
        if lag == 'day':
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
    results_df = results_df.append({'model_name': model_name, 'mape': '{:.2%}'.format(mape),'rmse': rmse, 'r2':r2}, ignore_index=True)
    
    with st.expander(':information_source: '+ model_name, expanded=True):
        display_my_metrics(my_df=df_preds, model_name=model_name)
        # plot graph with actual versus insample predictions
        plot_actual_vs_predicted(df_preds)
        # show the dataframe
        st.dataframe(df_preds.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Percentage_Diff': '{:.2%}', 'MAPE': '{:.2%}'}), use_container_width=True)
        # create download button for forecast results to .csv
        download_csv_button(df_preds, my_file="f'forecast_{model_name}_model.csv'", help_message=f'Download your **{model_name}** model results to .CSV')

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

def download_csv_button(my_df, my_file="forecast_model.csv", help_message = 'Download dataframe to .CSV'):
     # create download button for forecast results to .csv
     csv = convert_df(my_df)
     col1, col2, col3 = st.columns([2,2,2])
     with col2: 
         st.download_button(":arrow_down: Download", 
                            csv,
                            my_file,
                            "text/csv",
                            #key='', -> streamlit automatically assigns key if not defined
                            help = help_message)

def plot_actual_vs_predicted(df_preds):
    # Define the color palette
    colors = ['#5276A7', '#60B49F']
    # Create the figure with easy on eyes colors
    fig = px.line(df_preds, x=df_preds.index, y=['Actual', 'Predicted'], color_discrete_sequence=colors)
    # Update the layout of the figure
    fig.update_layout(legend_title='Legend',
                      font=dict(family='Arial', size=12, color='#707070'),
                      yaxis=dict(gridcolor='#E1E1E1', range=[0, np.maximum(df_preds['Actual'].max(), df_preds['Predicted'].max())]),
                      xaxis=dict(gridcolor='#E1E1E1'),
                      legend=dict(yanchor="bottom", y=0.0, xanchor="center", x=0.99))
    # Set the line colors
    for i, color in enumerate(colors):
        fig.data[i].line.color = color
        if fig.data[i].name == 'Predicted':
              fig.data[i].line.dash = 'dot' # dash styles options: ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
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
    # create a deepcopy of dataframe
    my_df_copy = my_df.copy(deep=True)
    if datetime_to_date == True:
        # convert the datetime to date (excl time 00:00:00)
        my_df_copy['date'] = pd.to_datetime(my_df['date']).dt.date
    if date_to_index == True:
        # set the index instead of 0,1,2... to date
        my_df_copy = my_df_copy.set_index('date')
    return my_df_copy

def resample_missing_dates(df):
    """
    Resamples a pandas DataFrame to a specified frequency, fills in missing values with NaNs,
    and inserts missing dates as rows with NaN values. Also displays a message if there are
    missing dates in the data.
    
    Parameters:
    df (pandas DataFrame): The DataFrame to be resampled
    
    Returns:
    pandas DataFrame: The resampled DataFrame with missing dates inserted
    
    """
    # Define a dictionary of possible frequencies and their corresponding offsets
    freq_dict = {'daily': 'D', 'weekly': 'W', 'monthly': 'M', 'quarterly': 'Q', 'yearly': 'Y'}

    # Ask the user to select the frequency of the data
    freq = st.sidebar.selectbox('Select the frequency of the data', list(freq_dict.keys()))
    
    # Resample the data to the specified frequency and fill in missing values with NaNs
    resampled_df = df.set_index('date').resample(freq_dict[freq]).asfreq()
    
    # Find missing dates and insert them as rows with NaN values
    missing_dates = pd.date_range(start=resampled_df.index.min(), end=resampled_df.index.max(), freq=freq_dict[freq]).difference(resampled_df.index)
    new_df = resampled_df.reindex(resampled_df.index.union(missing_dates)).sort_index()
    
    # Display a message if there are missing dates in the data
    if len(missing_dates) > 0:
        st.write("The missing dates are:")
        st.write(missing_dates)
    
    # Reset the index and rename the columns
    return new_df.reset_index().rename(columns={'index': 'date'})

def my_fill_method(df, fill_method):
    # handle missing values based on user input
    if fill_method == 'backfill':
        df.iloc[:,1] = df.iloc[:,1].bfill()
    elif fill_method == 'forwardfill':
        df.iloc[:,1] = df.iloc[:,1].ffill()
    elif fill_method == 'mean':
        df.iloc[:,1] = df.iloc[:,1].fillna(df.iloc[:,1].mean())
    elif fill_method == 'median':
        df.iloc[:,1]  = df.iloc[:,1] .fillna(df.iloc[:,1].median())
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
    
    st.markdown('<p style="text-align:center; color: #707070">Partial Autocorrelation (PACF)</p>', unsafe_allow_html=True)
    if data.isna().sum().sum() > 0:
        st.error('''**Warning** ⚠️:              
                 Data contains **NaN** values. **NaN** values were dropped in copy of dataframe to be able to plot below PACF. ''')
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
                           line=dict(color='green', width=1))
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
    st.markdown('<p style="text-align:center; color: #707070">Autocorrelation (ACF)</p>', unsafe_allow_html=True)
    if data.isna().sum().sum() > 0:
        st.error('''**Warning** ⚠️:              
                 Data contains **NaN** values. **NaN** values were dropped in copy of dataframe to be able to plot below ACF. ''')
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
                           line=dict(color='blue', width=1))
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
    
###############################################################################
# Create Left-Sidebar Streamlit App with Title + About Information
###############################################################################
# TITLE PAGE + SIDEBAR TITLE
with st.sidebar:   
        st.markdown(f'<h1 style="color:#45B8AC;"> <center> ForecastGenie™️ </center> </h1>', unsafe_allow_html=True)         
# set the title of page
st.title(":blue[]")
st.markdown(f'<h1 style="color:#45B8AC;"> <center> ForecastGenie™️ </center> </h1>', unsafe_allow_html=True)
# add vertical spacing
st.write("")

# ABOUT SIDEBAR MENU
with st.sidebar.expander(':information_source: About', expanded=False):
    st.write('''Hi :wave: **Welcome** to the ForecastGenie app created with Streamlit :smiley:
            
                \n**What does it do?**  
                - Forecast your Time Series Dataset - for example predict your future daily website `visits`!  
                \n**What do you need?**  
                - Upload your .csv file that contains your **date** (X) column and your target variable of interest (Y)  
            ''')
    st.markdown('---')
    # DISPLAY LOGO
    col1, col2, col3 = st.columns([1,3,2])
    with col2:
        st.image('./images/logo_dark.png', caption="Developed by")  
        # added spaces to align website link with logo in horizontal center
        st.markdown(f'<h5 style="color:#217CD0;"><center><a href="https://www.tonyhollaar.com/">www.tonyhollaar.com</a></center></h5>', unsafe_allow_html=True)
    st.markdown('---')

###############################################################################
# 1. Create Button to Load Dataset (.CSV format)
###############################################################################
my_title("1. Load Dataset 🚀 ", "#2CB8A1")
with st.sidebar:
    my_title("Load Dataset 🚀 ", "#2CB8A1") # 2CB8A1
    with st.expander('', expanded=True):
        uploaded_file = st.file_uploader("upload your .CSV file", label_visibility="collapsed")

# if nothing is uploaded yet by user run below code
if uploaded_file is None:
    # let user upload a file
    # inform user what template to upload
    with st.expander("", expanded=True):
        my_header("Instructions")
        st.info('''👈 **Please upload a .CSV file with:**  
                 - first column named: **$date$** with format: **$mm-dd-yyyy$**  
                 - second column the target variable: $y$''')

# if user uploaded csv file run below code
# wrap all code inside this related to data analysis / modeling
if uploaded_file is not None:   
    # define dataframe from  custom function to read from uploaded read_csv file
    df_raw = load_data()
    df_graph = df_raw.copy(deep=True)
    df_total = df_raw.copy(deep=True)
   
    # if loaded dataframe succesfully run below
    # set minimum date
    df_min = df_raw.iloc[:,0].min().date()
    # set maximum date
    df_max = df_raw.iloc[:,0].max().date()

    ## show message to user data if .csv file is loaded
    st.info('''🗨️ **Great!** your data is loaded, lets take a look :eyes: shall we...''')
    
    # set title
    my_title('2. Exploratory Data Analysis 🕵️‍♂️', my_background_color="#217CD0")
    with st.sidebar:
        my_title("Exploratory Data Analysis	🕵️‍♂️", "#217CD0")
        with st.form('eda'):
            # Create sliders in sidebar for the parameters of PACF Plot
            st.write("")
            my_subheader('ACF/PACF Plot Parameters')
            col1, col2, col3 = st.columns([4,1,4])
            # Set default values for parameters
            default_lags = 30
            default_method = "yw"  
            with col1:
                nlags = st.slider("*Select Number of lags*", min_value=1, max_value=50, value=default_lags)
            with col3:
                method = st.selectbox("*Method*", [ 'ols', 'ols-inefficient', 'ols-adjusted', 'yw', 'ywa', 'ld', 'ywadjusted', 'yw_adjusted', 'ywm', 'ywmle', 'yw_mle', 'lda', 'ldadjusted', 'ld_adjusted', 'ldb', 'ldbiased', 'ld_biased'], index=0)
            col1, col2, col3 = st.columns([4,4,4])
            with col2:
                # create button in sidebar for the ACF and PACF Plot Parameters
                acf_pacf_btn = st.form_submit_button("Submit", type="primary")
    # create expandable card with data exploration information
    with st.expander(':arrow_down: EDA', expanded=True):
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
        fig = px.line(df_graph,
                      x=df_graph.index,
                      y=df_graph.columns,
                      #labels=dict(x="Date", y="y"),
                      title='')
                      
        # Set Plotly configuration options
        fig.update_layout(width=800, height=400, xaxis=dict(title='Date'), yaxis=dict(title='', rangemode='tozero'), legend=dict(x=0.9, y=0.9))
        # set line color and width
        fig.update_traces(line=dict(color='#217CD0', width=2))
        
        # Display Plotly Express figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        #############################################################################
        # Create a button that toggles the display of the DataFrame
        #############################################################################
        col1, col2, col3 = st.columns([4,4,4])
        with col2:
            show_df_btn = st.button(f'Show DataFrame', use_container_width=True, type='secondary')
        if show_df_btn == True:
            # display the dataframe in streamlit
            st.dataframe(df_graph, use_container_width=True)
            
        #############################################################################
        # Call function for plotting Graphs of Seasonal Patterns D/W/M/Q/Y in Plotly Charts
        #############################################################################
        plot_overview(df_raw, y='total_traffic')
       
    with st.expander('ACF & PACF Plots', expanded=True): 
        if acf_pacf_btn == True:
            # use function to plot Partial ACF 
            data = df_raw.iloc[:,1]
            
            # Plot ACF        
            plot_acf(data, nlags=nlags)
            
            # Plot PACF
            plot_pacf(data,nlags=nlags, method=method)
        else:
            st.info(':arrow_left: Click \"**Submit**\" button to plot the **AutoCorrelation-** and **Partial AutoCorrelation Function**')
        # If user clicks button, more explanation on the ACF and PACF plot is displayed
        col1, col2, col3 = st.columns([4,4,4])
        with col2:
            show_pacf_btn = st.button(f'PACF Details', use_container_width=True, type='secondary')
        if show_pacf_btn == True:   
                my_subheader('Partial Autocorrelation Function (PACF)')
                st.info('''
                The :green[**partial autocorrelation function**] (PACF) is a plot of the partial correlation coefficients between a time series and its lags. 
                The PACF can help us determine the order of an autoregressive (AR) model by identifying the lag beyond which the autocorrelations are effectively zero.
                
                The :green[**PACF**] plot helps us identify the important lags that are related to a time series. It measures the correlation between a point in the time series and a lagged version of itself while controlling for the effects of all the other lags that come before it.
                In other words, the PACF plot shows us the strength and direction of the relationship between a point in the time series and a specific lag, independent of the other lags. 
                A significant partial correlation coefficient at a particular lag suggests that the lag is an important predictor of the time series.
                
                If a particular lag has a partial autocorrelation coefficient that falls outside of the :green[**95%**] or :green[**99%**] confidence interval, it suggests that this lag is a significant predictor of the time series. The next step would be to consider including that lag in the autoregressive model to improve its predictive accuracy.
                However, it is important to note that including too many lags in the model can lead to overfitting, which can reduce the model's ability to generalize to new data. Therefore, it is recommended to use a combination of statistical measures and domain knowledge to select the optimal number of lags to include in the model.
                On the other hand, if none of the lags have significant partial autocorrelation coefficients, it suggests that the time series is not well explained by an autoregressive model. In this case, alternative modeling techniques such as moving average (MA) or autoregressive integrated moving average (ARIMA) may be more appropriate.
                Or you could just flip a coin and hope for the best. But I don't recommend it...
                
                The partial autocorrelation plot (PACF) is a tool used to investigate the relationship between an observation in a time series with its lagged values, while controlling for the effects of intermediate lags. Here's a brief explanation of how to interpret a PACF plot:  
                - The horizontal axis shows the lag values (i.e., how many time steps back we're looking).
                - The vertical axis shows the correlation coefficient, which ranges from **-1** to **1**. A value of :green[**1**] indicates a :green[**perfect positive correlation**], while a value of :red[**-1**] indicates a :red[**perfect negative correlation**]. A value of **0** indicates **no correlation**.
                - Each bar in the plot represents the correlation between the observation and the corresponding lag value. The height of the bar indicates the strength of the correlation. 
                  If the bar extends beyond the dotted line (which represents the 95% confidence interval), the correlation is statistically significant.  
            
                Here are some key points to keep in mind when interpreting a PACF plot:  
                - *The first lag (lag 0) is always 1*, since an observation is perfectly correlated with itself.
                -  *A significant spike* at a particular lag indicates that there may be some **useful information** in that lagged value for predicting the current observation. 
                  This can be used to guide the selection of lag values in time series forecasting models.
                - *A sharp drop* in the PACF plot after a certain lag suggests that the lags beyond that point **are not useful** for prediction, and can be safely ignored.  
                
                An analogy would be:   
                Imagine you are watching a magic show where the magician pulls a rabbit out of a hat. Now, imagine that the magician can do this trick with different sized hats. If you were trying to figure out how the magician does this trick, you might start by looking for clues in the size of the hats.
                Similarly, the PACF plot is like a magic show where we are trying to figure out the "trick" that is causing our time series data to behave the way it does. 
                The plot shows us how strong the relationship is between each point in the time series and its past values, while controlling for the effects of all the other past values. 
                It's like looking at different sized hats to see which one the magician used to pull out the rabbit.

                If the PACF plot shows a strong relationship between a point in the time series and its past values at a certain lag (or hat size), it suggests that this past value is an important predictor of the time series. 
                On the other hand, if there is no significant relationship between a point and its past values, it suggests that the time series may not be well explained by past values alone, and we may need to look for other "tricks" to understand it.
                In summary, the PACF plot helps us identify important past values of our time series that can help us understand its behavior and make predictions about its future values.
                ''')

    ###############################################################################
    # 3. Data Cleaning
    ############################################################################### 
    my_title("2. Data Cleaning 🧹", "#440154")
    with st.sidebar:
        my_title("Data Cleaning 🧹 ", "#440154")        
        st.info('Please select options below:')    
    
    with st.expander('Missing Values', expanded=True):
        #*************************************************
        my_subheader('Handling missing values')
        #*************************************************    
        # Apply function to resample missing dates based on user set frequency
        # freq = daily, weekly, monthly, quarterly, yearly
        df_cleaned_dates = resample_missing_dates(df_raw)

        # Create matrix of missing values
        missing_matrix = df_cleaned_dates.isnull()
 
        # check if there are no dates skipped for daily data
        missing_dates = pd.date_range(start=df_raw['date'].min(), end=df_raw['date'].max()).difference(df_raw['date'])
        if missing_dates.shape[0] == 0:
            st.info('Pweh 😅, no dates are skipped in your dataframe!')
        else:
            st.warning(f'Oh No...🙁 {missing_dates.shape[0]} dates are skipped in your dataframe, lets fix this!')
            st.write(missing_dates)
            
        # Create Plotly Express figure
        my_subheader('Missing Values Matrix Plot', my_style="#333333", my_size=6)
        fig = px.imshow(missing_matrix,
                        labels=dict(x="Variables", y="Observations"),
                        x=missing_matrix.columns,
                        y=missing_matrix.index,
                        color_continuous_scale='Viridis',
                        title='')
        # Set Plotly configuration options
        fig.update_layout(width=400, height=400)
        fig.update_traces(showlegend=False)
        
        # Display Plotly Express figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)
       
        #******************************************************************
        # IMPUTE MISSING VALUES WITH FILL METHOD
        #******************************************************************
        # get user input for filling method
        fill_method = st.sidebar.selectbox('Select filling method for missing values:', ['backfill', 'forwardfill', 'mean', 'median'])
        df_clean = my_fill_method(df_cleaned_dates, fill_method)

        # Display original DataFrame with highlighted NaN cells
        # Create a copy of the original DataFrame with NaN cells highlighted in yellow
        col1, col2, col3, col4, col5 = st.columns([2, 0.5, 2, 0.5, 2])
        with col1:
            # highlight NaN values in yellow in dataframe
            highlighted_df = df_graph.style.highlight_null(null_color='yellow').format(precision=0)
            st.write('**Original DataFrame😐**')
            # show original dataframe unchanged but with highlighted missing NaN values
            st.write(highlighted_df)
        with col2:
            st.write('➡️')
        with col3:
            # Display the dates and the number of missing values associated with them
            my_subheader('Missing Values by Date😖', my_style="#333333", my_size=6)
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
    #########################################################
    with st.expander('Outliers', expanded=True):
        my_subheader('Handling outliers 😇😈😇')
    #########################################################
        
    ###############################################################################
    # 3. Feature Engineering
    ###############################################################################
    my_title("3. Feature Engineering 🧰", "#FF6F61")
    with st.sidebar:
        my_title("Feature Engineering 🧰", "#FF6F61")  
        st.info(''' Select your explanatory variables''')
        # show checkbox in middle of sidebar to select all features or none
        col1, col2, col3 = st.columns([0.1,8,3])
        with col3:
            select_all_days = st.checkbox('Select All Special Days:', value=True, label_visibility='collapsed' )
            select_all_seasonal = st.checkbox('Select All Sesaonal:', value=True, label_visibility='collapsed' )
        with col2:
            if select_all_days == True:
               st.write("*All Special Days*")
            else:
                st.write("*No Special Days*") 
            if select_all_seasonal == True:
                st.write("*All Seasonal Days*")
            else:
                st.write("*No Seasonal Days*") 
                
    with st.expander("📌", expanded=True):
        my_header('Special Calendar Days')
        start_date_calendar = df_clean['date'].min()
        end_date_calendar = df_clean['date'].max()
        st.markdown('---')
        df_exogenous_vars = pd.DataFrame({'date': pd.date_range(start = start_date_calendar, 
                                                                end = end_date_calendar)})
        cal = calendar()
        holidays = cal.holidays(start=start_date_calendar, end=end_date_calendar)
        # holiday = true/ otherwise false
        df_exogenous_vars['holiday'] = (df_exogenous_vars['date'].isin(holidays)).apply(lambda x: 1 if x==True else 0)
        
        # create column for holiday description
        df_exogenous_vars['holiday_desc'] = df_exogenous_vars['date'].apply(lambda x: my_holiday_name_func(x))
        my_subheader("🎁 Pick your special days to include: ")
        st.write("")
        col0, col1, col2, col3 = st.columns([6,12,12,1])
        
        ###############################################
        # create checkboxes for special days
        ###############################################
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
        # source: https://practicaldatascience.co.uk/data-science/how-to-create-an-ecommerce-trading-calendar-using-pandas
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
        df_total_incl_exogenous = pd.merge(df_clean, df_exogenous_vars, on='date', how='left' )
        df = df_total_incl_exogenous.copy(deep=True)
        
        ##############################
        # Add Day/Month/Year Features
        ##############################
        # create checkboxes for user to checkmark if to include features
        st.markdown('---')
        my_subheader('🌓 Pick your seasonal days to include: ')
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
        st.info(':information_source: **Note**: to prevent perfect multi-collinearity, leave-one-out is applied e.g. one year/month/day')
        ##########
        ## YEAR
        ##########
        if year_dummies:
            df['year'] = df['date'].dt.year
            dum_year =  pd.get_dummies(df['year'], columns = ['year'], drop_first=True, prefix='year', prefix_sep='_')
            df = pd.concat([df, dum_year], axis=1)   
        ##########
        ## MONTH
        ##########
        if month_dummies:
            month_dict = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
            df['month'] = df['date'].dt.month.apply(lambda x:month_dict.get(x))
            dum_month =  pd.get_dummies(df['month'], columns = ['month'], drop_first=True, prefix='', prefix_sep='')
            df = pd.concat([df, dum_month], axis=1)                            
        ##########
        # DAY
        ##########
        # date.weekday() - Return the day of the week as an integer, where Monday is 0 and Sunday is 6
        # convert 0 - 6 to weekday names
        if day_dummies:
            week_dict = {0:'Monday', 1:'Tuesday',2:'Wednesday', 3:'Thursday', 4: 'Friday', 5:'Saturday', 6:'Sunday'}
            df['day'] = df['date'].dt.weekday.apply(lambda x: week_dict.get(x))
            # get dummies
            dum_day = pd.get_dummies(df['day'], columns = ['day'], drop_first=True, prefix='', prefix_sep='')
            df = pd.concat([df, dum_day], axis=1)    
        st.markdown('---')
        # SHOW DATAFRAME
        st.write('**re-order dataframe TODO**')
        st.dataframe(df)     
        download_csv_button(df, my_file="dataframe_incl_features.csv", help_message="Download your dataset incl. features to .CSV")    
    
    ###############################################################################
    # 4. Prepare Data (split into training/test)
    ###############################################################################
    my_title('4. Prepare Data 🧪', "#FFB347")
    with st.sidebar:
        my_title('Prepare Data 🧪', "#FFB347")
    
    # 5.1) set date as index/ create local_df
    # create copy of dataframe not altering original
    local_df = df.copy(deep=True)
    # set the date as the index of the pandas dataframe
    local_df.index = pd.to_datetime(local_df['date'])
    local_df.drop(columns='date', inplace=True)

    with st.expander(':information_source: I removed the following descriptive columns automatically from analysis'):
        local_df = remove_object_columns(local_df)
    
    # USER TO SET the insample forecast days 
    with st.expander("", expanded=True):
        my_subheader('How many days you want to use as your test-set?')
        st.caption(f'<h6> <center> *NOTE: A commonly used ratio is 80:20 split between train and test </center> <h6>', unsafe_allow_html=True)

        # create sliders for user insample test-size (/train-size automatically as well)
        def update(change):
            if change == 'steps':
                st.session_state.steps = st.session_state.steps
            else:
                st.session_state.steps = math.floor((st.session_state.perc/100)*len(local_df))
        my_max_value = len(df)-1
        
        with st.sidebar:
            st.info('*Select in-sample test-size:*')
            col1, col2 = st.columns(2)
            with col1:
                my_insample_forecast_steps = st.slider('*In Days*', 
                                                       min_value = 1, 
                                                       max_value = my_max_value, 
                                                       value = int(len(df)*0.2),
                                                       step = 1,
                                                       key='steps',
                                                       on_change=update,
                                                       args=('steps',))
            with col2:
                my_insample_forecast_perc = st.slider('*As Percentage*', 
                                                      min_value = 1, 
                                                      max_value = int(my_max_value/len(df)*100), 
                                                      value = int(my_insample_forecast_steps/len(df)*100),
                                                      step = 1,
                                                      key='perc',
                                                      on_change=update,
                                                      args=('perc',))
            
        perc_test_set = "{:.2f}%".format((my_insample_forecast_steps/len(df))*100)
        perc_train_set = "{:.2f}%".format(((len(df)-my_insample_forecast_steps)/len(df))*100)
       
        ######################################################################################################
        # define dynamic user picked test-set size / train size for X,y, X_train, X_test, y_train, y_test
        # based on user picked my_insample_forecast_steps
        ######################################################################################################
        X = local_df.iloc[:, 1:]
        y = local_df.iloc[:, 0:1]
        X_train = local_df.iloc[:, 1:][:(len(df)-my_insample_forecast_steps)]
        X_test = local_df.iloc[:, 1:][(len(df)-my_insample_forecast_steps):]
        # set endogenous variable train/test split
        y_train = local_df.iloc[:, 0:1][:(len(df)-my_insample_forecast_steps)]
        y_test = local_df.iloc[:, 0:1][(len(df)-my_insample_forecast_steps):]

        # Set train/test split index
        split_index = len(local_df) - my_insample_forecast_steps      
        
        #############################################################
        # Create a figure with a scatter plot of the train/test split
        #############################################################
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=local_df.index[:len(df)-my_insample_forecast_steps], y=local_df.iloc[:len(df) - my_insample_forecast_steps, 0], mode='lines', name='Train', line=dict(color='#217CD0')))
        fig2.add_trace(go.Scatter(x=local_df.index[len(df)-my_insample_forecast_steps:], y=local_df.iloc[len(df) - my_insample_forecast_steps:, 0], mode='lines', name='Test', line=dict(color='#FFA500')))
        fig2.update_layout(title='',
                           yaxis=dict(range=[0, 
                                             local_df.iloc[:, 0].max()*1.1]),
                           shapes=[dict(type='line', 
                                        x0=local_df.index[split_index], 
                                        y0=0, 
                                        x1=local_df.index[split_index], 
                                        y1=local_df.iloc[:, 0].max()*1.1, 
                                        line=dict(color='grey', 
                                                  dash='dash'))],
                           annotations=[dict(x=local_df.index[split_index], 
                                             y=local_df.iloc[:, 0].max()*1.05, 
                                             xref='x', yref='y', 
                                             text='Train/Test<br>Split', 
                                             showarrow=True, 
                                             font = dict(color="grey", size = 15),
                                             arrowhead=1, 
                                             ax=0, 
                                             ay=-40)])
        split_date = local_df.index[len(df)-my_insample_forecast_steps-1]
        fig2.add_annotation(x = split_date, y = 0.95*y.max().values[0],
                            text = str(split_date.date()),
                            showarrow = False,
                            font = dict(color="grey", size = 15))
        st.plotly_chart(fig2, use_container_width=True)
        
        st.warning(f":information_source: train/test split equals :green[**{perc_train_set}**] and :green[**{perc_test_set}**] ")
    
    ###############################################################################
    # 5. Feature Selection
    ###############################################################################
    my_title('5. Feature Selection 🍏🍐🍋', "#CBB4D4")
    with st.sidebar:
        my_title('Feature Selection 🍏🍐🍋', "#CBB4D4")
    st.info('Let\'s review your top features to use in analysis ')
    st.info('''    Recursive Feature Elimination (RFE): This method involves recursively removing features and building a model on the remaining features. It then ranks the features based on their importance and eliminates the least important feature.
    Principal Component Analysis (PCA): This method transforms the original set of features into a smaller set of features, called principal components, that capture most of the variability in the data.
    Mutual Information: This method measures the dependence between two variables, such as the target variable and each feature. It selects the features that have the highest mutual information with the target variable.
    Lasso Regression: This method performs both feature selection and regularization by adding a penalty term to the objective function that encourages sparsity in the coefficients.''')
    try:
        with st.expander('RFECV'):
                my_subheader('Recursive Feature Elimination with Cross-Validation')
                # Scale the features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Set up the time series cross validation
                st.info(':information_source: The n_splits parameter in TimeSeriesSplit determines the number of splits to be made in the data. In other words, how many times the data is split into training and testing sets. It ensures that the testing set contains only data points that are more recent than the training set. The default value of n_splits is 5')
                timeseriessplit = st.slider('timeseries split', min_value=2, max_value=5, value=5)
                tscv = TimeSeriesSplit(n_splits=timeseriessplit)
                
                # Set up the linear regression model
                lr = LinearRegression()
                
                # Set up the recursive feature elimination with cross validation
                rfecv = RFECV(estimator=lr, step=1, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                
                # Fit the feature selection model
                rfecv.fit(X_scaled, y)
                
                # Print the selected features
                col1, col2, col3, col4 = st.columns([1,2,2,1])
                selected_features = X.columns[rfecv.support_]
                with col2:
                    st.write(':blue[**Selected features:**]', selected_features)
                
                # Print the feature rankings
                with col3: 
                    feature_rankings = pd.Series(rfecv.ranking_, index=X.columns).rename('Ranking')
                    st.write(':blue[**Feature rankings:**]')
                    st.write(feature_rankings.sort_values())
                    
                # Get the feature ranking
                feature_rankings = pd.Series(rfecv.ranking_, index=X.columns).rename('Ranking')
                
                # Create a dataframe with feature rankings and selected features
                df_ranking = pd.DataFrame({'Features': feature_rankings.index, 'Ranking': feature_rankings})
                
                # Highlight selected features
                df_ranking['Selected'] = np.where(df_ranking['Features'].isin(selected_features), 'Yes', 'No')
                
                # Create the plot
                fig = px.scatter(df_ranking, x='Features', y='Ranking', color='Selected', hover_data=['Ranking'])
                fig.update_layout(
                    title={
                        'text': 'Recursive Feature Elimination with Cross-Validation (RFECV)',
                        'x': 0.5,
                        'y': 0.95,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    xaxis_title='Features',
                    yaxis_title='Ranking',
                    legend_title='Selected'
                )
                
                # Show the plot
                st.plotly_chart(fig)
                
        # PCA feature selection
        pca = PCA(n_components=5)
        pca.fit(X)
        X_pca = pca.transform(X)
        selected_features_pca = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
        feature_names = X.columns
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pca.explained_variance_ratio_, y=feature_names[pca.components_.argmax(axis=1)], 
                             orientation='h', text=np.round(pca.explained_variance_ratio_ * 100, 2), textposition='auto'))
        fig.update_layout(title='PCA Feature Selection', xaxis_title='Explained Variance Ratio', yaxis_title='Feature Name')
        # Display plot in Streamlit
        st.plotly_chart(fig)
        # show user info about how to interpret the graph
        st.info('''When you fit a PCA model, it calculates the amount of variance that is captured by each principal component. \
                The variance ratio is the fraction of the total variance in the data that is explained by each principal component. \
                The sum of the variance ratios of all the principal components equals 1. 
                The variance ratio is expressed as a percentage by multiplying it by 100, so it can be easily interpreted. \
                For example, a variance ratio of 0.75 means that 75% of the total variance in the data is captured by the corresponding principal component.''')
    except:
        st.write(':red[Error: Recursive Feature Elimination with Cross-Validation could not execute...please adjust your selection criteria]')
    
    # Add slider to select number of top features
    st.sidebar.info('*Select number of top features:*')
    num_features = st.sidebar.slider("**value**", min_value=1, max_value=len(X.columns), value=len(X.columns), step=1, label_visibility="collapsed")
    
    # Mutual information feature selection
    mutual_info = mutual_info_classif(X, y, random_state=42)
    selected_features_mi = X.columns[np.argsort(mutual_info)[::-1]][:num_features]
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mutual_info[np.argsort(mutual_info)[::-1]][:num_features],
                         y=selected_features_mi, 
                         orientation='h',
                         text=[f'{val:.2f}' for val in mutual_info[np.argsort(mutual_info)[::-1]][:num_features]],
                         textposition='inside'))
    fig.update_layout(title=f'Top {num_features} Mutual Information Feature Selection')
    
    # Display plot in Streamlit
    st.plotly_chart(fig)
    
    #feature_selection_user = st.multiselect("favorite features", list(selected_features_mi))
    ##############################################################
    # SELECT YOUR FAVORITE FEATURES TO INCLUDE IN MODELING
    ##############################################################
    selected_cols = ["total_traffic"] + list(selected_features_mi)
    st.info(f'the columns you selected are: {selected_cols}')

###############################################################################
# 6. Select Models
###############################################################################
if uploaded_file is not None:
    my_title("6. Select Models 🔢", "#0072B2")
    with st.sidebar:
        my_title("Select Models 🔢", "#0072B2")
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
                    
                    - ***p, d, q***: These parameters refer to the autoregressive, integrated, and moving average components, respectively, in the non-seasonal part of the model. They represent the order of the AR, I, and MA terms, respectively.
                    - ***P, D, Q***: These parameters refer to the autoregressive, integrated, and moving average components, respectively, in the seasonal part of the model. They represent the order of the seasonal AR, I, and MA terms, respectively.
                    - ***m***: This parameter represents the number of time periods in a season.
                    - ***Exogenous Variables***: These are external factors that can impact the variable being forecasted. They are included in the model as additional inputs.  
            
                    $$\\large  y(t) = c + \sum_{i=1}^{p} \phi_i y(t-i) + \sum_{j=1}^{P} \delta_j y(t-jm) - \sum_{k=1}^{q}  \\theta_k e(t-k) $$   
                    
                    $$\\large - \sum_{l=1}^{Q}  \Delta_l e(t-lm) + \sum_{m=1}^{k} \\beta_m x_m(t) + e(t) $$

                    And here's a breakdown of each term in the equation:
                
                    - $y(t)$: The value of the variable being forecasted at time $t$.
                    - $c$: A constant term.
                    - $\phi_1, \dots, \phi_p$: Autoregressive coefficients for the non-seasonal component, where $p$ is the order of the non-seasonal autoregressive component.
                    - $\delta_1, \dots, \delta_P$: Autoregressive coefficients for the seasonal component, where $P$ is the order of the seasonal autoregressive component and $m$ is the number of time periods in a season.
                    - $e(t)$: The error term at time $t$.
                    - $\\theta_1, \dots, \\theta_q$: Moving average coefficients for the non-seasonal component, where $q$ is the order of the non-seasonal moving average component.
                    - $\Delta_1, \dots, \Delta_Q$: Moving average coefficients for the seasonal component, where $Q$ is the order of the seasonal moving average component and $m$ is the number of time periods in a season.
                    - $x_1(t), \dots, x_k(t)$: Exogenous variables at time $t$.
                    - $\\beta_1$, $\dots$, $\\beta_k$: Coefficients for the exogenous variables.
                    - $\sum$: The summation operator, used to add up terms over a range of values.
                    - $i, j, k, l, m$: Index variables used in the summation.
                    '''
                    )
    ################################################
    # Create a User Form to Select Model(s) to train
    ################################################
    with st.sidebar.form('model_train_form'):

        # define all models you want user to choose from
        models = [('Naive Model', None),('Linear Regression', LinearRegression(fit_intercept=True)), ('SARIMAX', SARIMAX(y_train))]
        
        # create a checkbox for each model
        selected_models = []
        for model_name, model in models:
            if st.checkbox(model_name):
                selected_models.append((model_name, model))

            if model_name == "Naive Model":
                    custom_lag_value = None
                    lag = st.sidebar.selectbox('*Select your seasonal **lag** for the Naive Model:*', ['None', 'Day', 'Week', 'Month', 'Year', 'Custom'])
                    if lag == 'None':
                        lag = None
                        if custom_lag_value != None:
                            custom_lag_value = None
                    elif lag == 'Custom':
                            lag = lag.lower()
                            custom_lag_value  = int(st.sidebar.text_input("Enter a value:", value=5))
                    else:
                        # lag is lowercase string of selection from user in selectbox
                        lag = lag.lower()
            else:
                st.sidebar.empty()
            
        # set vertical spacers
        col1, col2, col3 = st.columns([2,3,2])
        with col2:
            train_models_btn = st.form_submit_button("Submit", type="primary")
    
    #if nothing is selected by user display message to user to select models to train
    if not train_models_btn and not selected_models:
        st.warning("👈 Select your models to train in the sidebar!🏋️‍♂️") 
    # the code block to train the selected models will only be executed if both the button has been clicked and the list of selected models is not empty.
    elif not selected_models:
        st.warning("👈 Please select at least 1 model to train from the sidebar, when pressing the **\"Submit\"** button!🏋️‍♂️")
    
    ###############################################################################
    my_title("7. Evaluate Models 🔎", "#2CB8A1")
    ###############################################################################
    with st.sidebar:
        my_title("Evaluate Models 🔎", "#2CB8A1")
        
    if train_models_btn and selected_models:
        # Create a global pandas DataFrame to hold model_name and mape values
        results_df = pd.DataFrame(columns=['model_name', 'mape', 'rmse', 'r2'])
        # iterate over all models and if user selected checkbox for model the model(s) is/are trained
        for model_name, model in selected_models:
            if model_name == "Naive Model":
                with st.expander(':information_source: '+ model_name, expanded=True):
                   try:
                     df_preds = evaluate_regression_model(model, X_train, y_train, X_test, y_test, lag=lag, custom_lag_value=custom_lag_value)
                     display_my_metrics(df_preds, "Naive Model")
                     # plot graph with actual versus insample predictions
                     plot_actual_vs_predicted(df_preds)
                     # show the dataframe
                     st.dataframe(df_preds.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Percentage_Diff': '{:.2%}', 'MAPE': '{:.2%}'}), use_container_width=True)
                     # create download button for forecast results to .csv
                     download_csv_button(df_preds, my_file="f'forecast_{model_name}_model.csv'", help_message="Download your **Naive** model results to .CSV")
                     mape, rmse, r2 = my_metrics(df_preds, model_name=model_name)
                     # display evaluation results on sidebar of streamlit_model_card
                     results_df = results_df.append({'model_name': 'Naive Model', 
                                                     'mape': '{:.2%}'.format(metrics_dict['Naive Model']['mape']),
                                                     'rmse': '{:.2f}'.format(metrics_dict['Naive Model']['rmse']), 
                                                     'r2': '{:.2f}'.format(metrics_dict['Naive Model']['r2'])}, ignore_index=True)
                   except:
                       st.warning(f'Naive Model failed to train, please check parameters set in the sidebar: lag={lag}, custom_lag_value={lag}')
            if model_name == "Linear Regression":
                create_streamlit_model_card(X_train, y_train, X_test, y_test, results_df,  model=model, model_name=model_name)
                results_df = results_df.append({'model_name': 'Linear Regression', 
                                                'mape': '{:.2%}'.format(metrics_dict['Linear Regression']['mape']),
                                                'rmse': '{:.2f}'.format(metrics_dict['Linear Regression']['rmse']), 
                                                'r2': '{:.2f}'.format(metrics_dict['Linear Regression']['r2'])}, ignore_index=True)
            if model_name == "SARIMAX":
                with st.expander(':information_source: '+ model_name, expanded=True):
                    with st.spinner('This model might require some time to train... you can grab a coffee ☕ or tea 🍵'):
                        preds_df = evaluate_sarimax_model(order=(1,1,1), seasonal_order=(1,1,1,12), exog_train=X_train, exog_test=X_test, endog_train=y_train, endog_test=y_test)
                        display_my_metrics(preds_df, "SARIMAX")
                        # plot graph with actual versus insample predictions
                        plot_actual_vs_predicted(preds_df)
                        # show the dataframe
                        st.dataframe(preds_df.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Percentage_Diff': '{:.2%}', 'MAPE': '{:.2%}'}), use_container_width=True)
                        # create download button for forecast results to .csv
                        download_csv_button(preds_df, my_file="f'forecast_{model_name}_model.csv'", help_message="Download your **SARIMAX** model results to .CSV")
                        mape, rmse, r2 = my_metrics(preds_df, model_name=model_name)
                        # display evaluation results on sidebar of streamlit_model_card
                        results_df = results_df.append({'model_name': 'SARIMAX', 
                                                        'mape': '{:.2%}'.format(metrics_dict['SARIMAX']['mape']),
                                                        'rmse': '{:.2f}'.format(metrics_dict['SARIMAX']['rmse']), 
                                                        'r2': '{:.2f}'.format(metrics_dict['SARIMAX']['r2'])}, ignore_index=True)
        # Show the results dataframe in the sidebar if there is at least one model selected
        if len(selected_models) > 0:
            st.sidebar.dataframe(results_df)

##############################################################################
# 8. Forecast
##############################################################################
if uploaded_file is not None:
    my_title('7. Forecast 🔮', "#48466D")   
    with st.sidebar:
        my_title('Forecast 🔮', "#48466D")                    
        with st.expander("📆 ", expanded=True):
            #my_header("Set Date Range Forecast 📆")
            # set start date
            start_date_calendar = df['date'].min()
            end_date_calendar = df['date'].max()
            # set end date 
            max_date = df['date'].max()
            # define maximum value in dataset for year, month day
            year = max_date.year
            month = max_date.month
            day = max_date.day
            start_date_calendar = df['date'].min()
            end_date_calendar = df['date'].max()
            
            col1, col2 = st.columns(2)
            with col1: 
                st.write(" ")
                st.write(" ")
                st.markdown(f'<h4 style="color: #48466D; background-color: #F0F2F6; padding: 12px; border-radius: 5px;"><center> Select End Date:</center></h4>', unsafe_allow_html=True)
            with col2:
                end_date_calendar = st.date_input("", (datetime.date(year, month, day) + datetime.timedelta(days=90)))

