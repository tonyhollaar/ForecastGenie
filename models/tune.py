import pandas as pd
import time
import datetime
import streamlit as st
import numpy as np
import re
import itertools

import plotly.graph_objects as go
import plotly.express as px

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

from style.text import my_title, my_text_paragraph, vertical_spacer
from style.icons import load_icons
from optuna import visualization as ov
from functions import forecast_naive_model1_insample, forecast_naive_model2_insample, forecast_naive_model3_insample, \
    rank_dataframe, my_metrics

# Load icons
icons = load_icons()

def hyperparameter_tuning_form(y_train):
    # =============================================================================
    # Initiate Variables Required
    # =============================================================================
    selected_models = [('Naive Model', None),
                       ('Linear Regression', LinearRegression(fit_intercept=True)),
                       ('SARIMAX', SARIMAX(y_train)),
                       ('Prophet', Prophet())]  # Define tuples of (model name, model)

    my_title(f'{icons["tune_icon"]}', "#88466D")

    with st.form("hyper_parameter_tuning"):

        # create option for user to early stop the hyperparameter tuning process based on time
        max_wait_time = st.time_input(label='Set the maximum time allowed for tuning (in minutes)',
                                      value=datetime.time(0, 5),
                                      step=60)  # seconds /  You can also pass a datetime.timedelta object.

        # create a list of the selected models by the user in the training section of the streamlit app
        model_lst = [model_name for model_name, model in selected_models]

        # SELECT MODEL(S): let the user select the models multi-selectbox for hyper-parameter tuning
        selected_model_names = st.multiselect('Select Models',
                                              options=model_lst,
                                              help='The `Selected Models` are tuned using the chosen hyperparameter '
                                                   'search algorithm. A set of hyperparameters is defined, '
                                                   'and the models are trained on the training dataset. These models '
                                                   'are then evaluated using the selected evaluation metric on a '
                                                   'validation dataset for a range of different hyperparameter '
                                                   'combinations.')

        # SELECT EVALUATION METRIC: let user set evaluation metric for the hyperparameter tuning
        metric = st.selectbox(label='Select Evaluation Metric To Optimize',
                              options=['AIC', 'BIC', 'RMSE'],
                              label_visibility='visible',
                              help='**`AIC`** (**Akaike Information Criterion**): A measure of the quality of a '
                                   'statistical model, taking into account the goodness of fit and the complexity of '
                                   'the model. A lower AIC indicates a better model fit. \ \n**`BIC`** (**Bayesian '
                                   'Information Criterion**): Similar to AIC, but places a stronger penalty on models '
                                   'with many parameters. A lower BIC indicates a better model fit.  \ \n**`RMSE`** ('
                                   '**Root Mean Squared Error**): A measure of the differences between predicted and '
                                   'observed values in a regression model. It is the square root of the mean of the '
                                   'squared differences between predicted and observed values. A lower RMSE indicates '
                                   'a better model fit.')

        search_algorithm = st.selectbox(label='Select Search Algorithm',
                                        options=['Grid Search', 'Random Search', 'Bayesian Optimization'],
                                        index=2,
                                        help='''1. `Grid Search:` algorithm exhaustively searches through all 
                                        possible combinations of hyperparameter values within a predefined range. It 
                                        evaluates the model performance for each combination and selects the best set 
                                        of hyperparameters based on a specified evaluation metric. \n2. `Random 
                                        Search:` search randomly samples hyperparameters from predefined 
                                        distributions. It performs a specified number of iterations and evaluates the 
                                        model performance for each sampled set of hyperparameters. Random search can 
                                        be more efficient than grid search when the search space is large. \n3. 
                                        `Bayesian Optimization:` is an iterative approach to hyperparameter tuning 
                                        that uses Bayesian inference. Unlike grid search and random search, 
                                        Bayesian optimization considers past observations to make informed decisions. 
                                        It intelligently explores the hyperparameter space by selecting promising 
                                        points based on the model's predictions and uncertainty estimates. It 
                                        involves defining the search space, choosing an acquisition function, 
                                        building a surrogate model, iteratively evaluating and updating, 
                                        and terminating the optimization. It is particularly effective when the 
                                        search space is large or evaluation is computationally expensive.''')

        # set the number of combinations to search for Bayesian Optimization algorithm
        if search_algorithm == 'Bayesian Optimization':
            trial_runs = st.number_input(label='Set number of trial runs',
                                         value=10,
                                         step=1,
                                         min_value=1,
                                         help='''`Trial Runs` sets the number of maximum different combinations of 
                                         hyperparameters to be tested by the search algorithm. A larger number will 
                                         increase the hyperparameter search space, potentially increasing the 
                                         likelihood of finding better-performing hyperparameter combinations. 
                                         However, keep in mind that a larger n_trials value can also increase the 
                                         overall computational time required for the optimization process.''')
        else:
            # set default value
            trial_runs = 10

        # =============================================================================
        # SARIMAX HYPER-PARAMETER GRID TO SELECT BY USER
        # =============================================================================
        my_text_paragraph('Set Model(s) Search Space(s)', my_text_align='left', my_font_family='Ysabeau SC',
                          my_font_weight=200, my_font_size='14px')

        with st.expander('◾ Naive Models Parameters'):
            # =============================================================================
            # Naive Model I: Lag
            # =============================================================================
            my_text_paragraph('Naive Model I: Lag')

            col1, col2, col3 = st.columns([1, 12, 1])
            with col2:
                lag_options = st.multiselect(label='*select seasonal lag*',
                                             options=['Day', 'Week', 'Month', 'Year'],
                                             default=['Day', 'Week', 'Month', 'Year'])

            vertical_spacer(1)

            # =============================================================================
            # Naive Model II: Rolling Window
            # =============================================================================
            my_text_paragraph('Naive Model II: Rolling Window')

            col1, col2, col3 = st.columns([1, 12, 1])
            with col2:
                rolling_window_range = st.slider(label='Select rolling window size:',
                                                 min_value=2,
                                                 max_value=365,
                                                 value=(2, 365),
                                                 step=1)

                rolling_window_options = st.multiselect(label='*select aggregation method:*',
                                                        options=['Mean', 'Median', 'Mode'],
                                                        default=['Mean', 'Median', 'Mode'],
                                                        key='tune_options_naivemodelii')

            # =============================================================================
            # Naive Model III
            # =============================================================================
            vertical_spacer(1)
            my_text_paragraph('Naive Model III: Constant Value')

            col1, col2, col3 = st.columns([1, 12, 1])
            with col2:
                agg_options = st.multiselect(label='*select aggregation method:*',
                                             options=['Mean', 'Median', 'Mode'],
                                             default=['Mean', 'Median', 'Mode'],
                                             key='tune_options_naivemodeliii')
            vertical_spacer(1)

        with st.expander('◾ SARIMAX Hyperparameters'):

            col1, col2, col3 = st.columns([5, 1, 5])
            with col1:
                p_max = st.number_input(label="*Max Autoregressive Order (p):*",
                                        value=1,
                                        min_value=0,
                                        max_value=10)

                d_max = st.number_input(label="*Max Differencing (d):*",
                                        value=1,
                                        min_value=0,
                                        max_value=10)

                q_max = st.number_input(label="*Max Moving Average (q):*",
                                        value=1,
                                        min_value=0,
                                        max_value=10)

                trend = st.multiselect(label='trend',
                                       options=['n', 'c', 't', 'ct'],
                                       default=['n'],
                                       help='''Options for the 'trend' parameter: - `n`: No trend component is 
                                       included in the model. - `c`: A constant (i.e., a horizontal line) is included 
                                       in the model. - `t`: A linear trend component with a time-dependent slope. - 
                                       `ct`: A combination of `c` and `t`, including both a constant and a linear 
                                       time-dependent trend.''')
            with col3:
                P_max = st.number_input(label="*Max Seasonal Autoregressive Order (P):*",
                                        value=1,
                                        min_value=0,
                                        max_value=10)

                D_max = st.number_input(label="*Max Seasonal Differencing (D):*",
                                        value=1,
                                        min_value=0,
                                        max_value=10)

                Q_max = st.number_input(label="*Max Seasonal Moving Average (Q):*",
                                        value=1,
                                        min_value=0,
                                        max_value=10)

                s = st.number_input(label="*Set Seasonal Periodicity (s):*",
                                    value=7,
                                    min_value=1)

            st.write('Parameters')
            col1, col2, col3 = st.columns([5, 1, 5])

            with col1:
                enforce_stationarity = st.selectbox('*Enforce Stationarity*',
                                                    options=[True, False],
                                                    index=0,
                                                    help='Whether or not to transform the AR parameters to enforce '
                                                         'stationarity in the autoregressive component of the model.')
            with col3:
                enforce_invertibility = st.selectbox('*Enforce Invertibility*',
                                                     options=[True, False],
                                                     index=0,
                                                     help='Whether or not to transform the MA parameters to enforce '
                                                          'invertibility in the moving average component of the model.')

        # =============================================================================
        # PROPHET HYPER-PARAMETER GRID TO SELECT BY USER
        # =============================================================================
        with st.expander('◾ Prophet Hyperparameters'):

            vertical_spacer(1)

            col1, col2 = st.columns([5, 1])
            with col1:
                # usually set to forecast horizon e.g. 30 days
                horizon_option = st.slider(label='Set Forecast Horizon (default = 30 Days):',
                                           min_value=1,
                                           max_value=365,
                                           step=1, value=30,
                                           help='The horizon for a Prophet model is typically set to the number of '
                                                'time periods that you want to forecast into the future. This is also '
                                                'known as the forecasting horizon or prediction horizon.')

                changepoint_prior_scale_options = st.multiselect(label='*Changepoint Prior Scale*',
                                                                 options=[0.001, 0.01, 0.05, 0.1, 1],
                                                                 default=[0.001, 0.01, 0.1, 1],
                                                                 help='This is probably the most impactful parameter. '
                                                                      'It determines the flexibility of the trend, '
                                                                      'and in particular how much the trend changes '
                                                                      'at the trend changepoints. As described in '
                                                                      'this documentation, if it is too small, '
                                                                      'the trend will be underfit and variance that '
                                                                      'should have been modeled with trend changes '
                                                                      'will instead end up being handled with the '
                                                                      'noise term. If it is too large, the trend will '
                                                                      'overfit and in the most extreme case you can '
                                                                      'end up with the trend capturing yearly '
                                                                      'seasonality. The default of 0.05 works for '
                                                                      'many time series, but this could be tuned; a '
                                                                      'range of [0.001, 0.5] would likely be about '
                                                                      'right. Parameters like this (regularization '
                                                                      'penalties; this is effectively a lasso '
                                                                      'penalty) are often tuned on a log scale.')

                seasonality_mode_options = st.multiselect(label='*Seasonality Modes*',
                                                          options=['additive', 'multiplicative'],
                                                          default=['additive', 'multiplicative'])

                seasonality_prior_scale_options = st.multiselect(label='*Seasonality Prior Scales*',
                                                                 options=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                                                                 default=[0.01, 10.0])

                holidays_prior_scale_options = st.multiselect(label='*Holidays Prior Scales*',
                                                              options=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                                                              default=[0.01, 10.0])

                yearly_seasonality_options = st.multiselect(label='*Yearly Seasonality*',
                                                            options=[True, False],
                                                            default=[True])

                weekly_seasonality_options = st.multiselect(label='*Weekly Seasonality*',
                                                            options=[True, False],
                                                            default=[True])

                daily_seasonality_options = st.multiselect(label='*Daily Seasonality*',
                                                           options=[True, False],
                                                           default=[True])

        # create vertical spacing columns
        col1, col2, col3 = st.columns([4, 4, 4])
        with col2:
            # create submit button for the hyper-parameter tuning
            hp_tuning_btn = st.form_submit_button("Submit", type="secondary")

    return (max_wait_time, selected_model_names, metric, search_algorithm, trial_runs, trend,
            lag_options, rolling_window_range, rolling_window_options, agg_options,
            p_max, d_max, q_max, P_max, D_max, Q_max, s, enforce_stationarity, enforce_invertibility,
            horizon_option, changepoint_prior_scale_options, seasonality_mode_options, seasonality_prior_scale_options,
            holidays_prior_scale_options, yearly_seasonality_options, weekly_seasonality_options,
            daily_seasonality_options,
            hp_tuning_btn)

def run_naive_model_1(lag_options=None, validation_set=None, max_wait_time=datetime.time(0, 5),
                      rolling_window_range=None, rolling_window_options=None, agg_options=None, metrics_dict=None,
                      progress_bar=st.progress(0)):
    """
    Run the grid search for Naive Model 1 using different lag options.

    Parameters:
        lag_options (list or None): List of lag options to be tuned during grid search.
                                   Each lag option represents a specific configuration for Naive Model I.
                                   Default is None.
        validation_set (pd.DataFrame or None): Validation dataset used for forecasting and performance evaluation.
                                               It should contain the target variable.
                                               Default is None.

    Returns:
        None

    This function performs a grid search to tune Naive Model I by trying different lag options and
    evaluating their performance using the provided validation dataset. The results are displayed as a DataFrame
    with the rankings of different lag options based on their MAPE (Mean Absolute Percentage Error) metric.

    Note: The function expects a progress bar named `progress_bar` to be initialized in the Streamlit app.

    Example:
    run_naive_model_1(lag_options=[1, 2, 3], total_options=9, validation_set=validation_data)

    """
    start_time = time.time()
    max_wait_time_minutes = int(str(max_wait_time.minute))
    max_wait_time_seconds = max_wait_time_minutes * 60

    # Store total number of combinations in the parameter grid for progress bar
    total_options = len(lag_options) + (len(list(range(rolling_window_range[0], rolling_window_range[1] + 1)))
                                        * len(rolling_window_options)) + len(agg_options)

    # Set title
    my_text_paragraph('Naive Model I: Lag')

    # Define model name
    model_name = 'Naive Model I'
    # Set start time when grid-search is kicked-off to define total time it takes as computationally intensive
    start_time_naive_model_1 = time.time()
    naive_model1_tuning_results = pd.DataFrame()  # results with lag

    # iterate over grid of all possible combinations of hyperparameters
    for i, lag_option in enumerate(lag_options):

        # Check if the maximum waiting time has been exceeded
        elapsed_time_seconds = time.time() - start_time

        if elapsed_time_seconds > max_wait_time_seconds:
            st.warning("Maximum waiting time exceeded. The grid search has been stopped.")
            # exit the loop once maximum time is exceeded defined by user or default = 5 minutes
            break

        # Update the progress bar
        progress_percentage = i / total_options * 100
        progress_text = f'''Please wait up to {max_wait_time_minutes} minute(s) while parameters of Naive Models are being tuned!
                            \n{progress_percentage:.2f}% of total options within the search space reviewed ({i} out of {total_options} total options).'''
        progress_bar.progress(progress_percentage, progress_text)

        # Create a model with the current parameter values
        df_preds_naive_model1 = forecast_naive_model1_insample(validation_set, lag=lag_option.lower(),
                                                               custom_lag_value=None)

        # Retrieve metrics from difference actual - predicted
        metrics_dict = my_metrics(my_df=df_preds_naive_model1, model_name=model_name, metrics_dict=metrics_dict)

        # Create a new DataFrame with the row to append
        new_row = pd.DataFrame({'parameters': [f'lag: {lag_option}'],
                                'MAPE': metrics_dict[model_name]['mape'],
                                'SMAPE': metrics_dict[model_name]['smape'],
                                'RMSE': metrics_dict[model_name]['rmse'],
                                'R2': metrics_dict[model_name]['r2']})

        # Concatenate the original DataFrame with the new row DataFrame
        naive_model1_tuning_results = pd.concat([naive_model1_tuning_results, new_row], ignore_index=True)

        # Add rank column to dataframe and order by metric column
        ranked_naive_model1_tuning_results = rank_dataframe(naive_model1_tuning_results, 'MAPE')

        # Convert MAPE and SMAPE metrics to string with '%' percentage-sign
        ranked_naive_model1_tuning_results[['MAPE', 'SMAPE']] = ranked_naive_model1_tuning_results[
            ['MAPE', 'SMAPE']].map(lambda x: f'{x * 100:.2f}%' if not np.isnan(x) else x)

    st.dataframe(ranked_naive_model1_tuning_results, use_container_width=True, hide_index=True)

    # set the end of runtime
    end_time_naive_model_1 = time.time()

    text = str(ranked_naive_model1_tuning_results.iloc[0, 1])
    formatted_text = re.sub(r'([^:]+): ([^,]+)', r'`\1`: **\2**', text)
    final_text = re.sub(r',\s*', ', ', formatted_text)

    st.write(
        f'''💬**Naive Model I** parameter with the lowest MAPE of {ranked_naive_model1_tuning_results["MAPE"][0]} found in **{end_time_naive_model_1 - start_time_naive_model_1:.2f}** seconds is:  
             {final_text}''')


def run_naive_model_2(rolling_window_range=None, rolling_window_options=None, validation_set=None, lag_options=None,
                      agg_options=None, max_wait_time=datetime.time(0, 5), y_test=None, metrics_dict=None,
                      progress_bar=st.progress(0)):
    """
    Run the grid search for Naive Model II using different rolling window configurations.

    Parameters:
        rolling_window_range (tuple or None): A tuple (start, end) representing the range of rolling window sizes
                                              to be considered during the grid search for Naive Model II.
                                              Default is None.
        rolling_window_options (list or None): List of aggregation methods to be considered for the rolling window.
                                               Each method represents a specific configuration for Naive Model II.
                                               Default is None.
        validation_set (pd.DataFrame or None): Validation dataset used for forecasting and performance evaluation.
                                               It should contain the target variable.
                                               Default is None.

    Returns:
        None

    This function performs a grid search to tune Naive Model II by trying different combinations of rolling window
    sizes and aggregation methods, and evaluating their performance using the provided validation dataset. The results
    are displayed as a DataFrame with the rankings of different configurations based on their MAPE (Mean Absolute
    Percentage Error) metric.

    Note: The function expects a progress bar named `progress_bar` to be initialized in the Streamlit app.

    Example:
    run_naive_model_2(rolling_window_range=(3, 5), rolling_window_options=['mean', 'median'], validation_set=validation_data)

    """
    start_time = time.time()
    max_wait_time_minutes = int(str(max_wait_time.minute))
    max_wait_time_seconds = max_wait_time_minutes * 60
    # Store total number of combinations in the parameter grid for progress bar
    total_options = len(lag_options) + (len(list(range(rolling_window_range[0], rolling_window_range[1] + 1)))
                                        * len(rolling_window_options)) + len(agg_options)

    # Define title
    my_text_paragraph('Naive Model II: Rolling Window')

    # Define model name
    model_name = 'Naive Model II'

    # save evaluation results to dataframe
    naive_model2_tuning_results = pd.DataFrame()

    # Iterate over grid of all possible combinations of hyperparameters
    param_grid = {'rolling_window_range': list(range(rolling_window_range[0], rolling_window_range[1] + 1)),
                  'rolling_window_options': rolling_window_options}

    start_time_naive_model_2 = time.time()

    for i, (rolling_window_value, rolling_window_option) in enumerate(
            itertools.product(param_grid['rolling_window_range'], param_grid['rolling_window_options']), 0):

        # Check if the maximum waiting time has been exceeded
        elapsed_time_seconds = time.time() - start_time

        if elapsed_time_seconds > max_wait_time_seconds:
            st.warning("Maximum waiting time exceeded. The grid search has been stopped.")
            # exit the loop once maximum time is exceeded defined by user or default = 5 minutes
            break

        # Update the progress bar
        progress_percentage = (i + len(lag_options)) / total_options * 100
        progress_bar.progress(value=((i + len(lag_options)) / total_options),
                              text=f'''Please wait up to {max_wait_time_minutes} minute(s) while parameters of Naive Models are being tuned!
                                         \n{progress_percentage:.2f}% of total options within the search space reviewed ({i + len(lag_options)} out of {total_options} total options).''')

        # Return a prediction dataframe
        df_preds_naive_model2 = forecast_naive_model2_insample(y_test, size_rolling_window=rolling_window_value,
                                                               agg_method_rolling_window=rolling_window_option)

        # Retrieve metrics from difference actual - predicted
        metrics_dict = my_metrics(my_df=df_preds_naive_model2, model_name=model_name, metrics_dict=metrics_dict)

        # Create a new DataFrame with the row to append
        new_row = pd.DataFrame({'parameters': [
            f'rolling_window_size: {rolling_window_value}, aggregation_method: {rolling_window_option}'],
            'MAPE': metrics_dict[model_name]['mape'],
            'SMAPE': metrics_dict[model_name]['smape'],
            'RMSE': metrics_dict[model_name]['rmse'],
            'R2': metrics_dict[model_name]['r2']})

        # Concatenate the original DataFrame with the new row DataFrame
        naive_model2_tuning_results = pd.concat([naive_model2_tuning_results, new_row], ignore_index=True)

        # Add rank column to dataframe and order by metric column
        ranked_naive_model2_tuning_results = rank_dataframe(naive_model2_tuning_results, 'MAPE')

        # Convert MAPE and SMAPE evaluation metrics to '%' percentage-sign
        ranked_naive_model2_tuning_results[['MAPE', 'SMAPE']] = ranked_naive_model2_tuning_results[
            ['MAPE', 'SMAPE']].map(lambda x: f'{x * 100:.2f}%' if not np.isnan(x) else x)

    st.dataframe(ranked_naive_model2_tuning_results, use_container_width=True, hide_index=True)

    # Set the end of runtime
    end_time_naive_model_2 = time.time()

    # Clear progress bar in streamlit for user as process is completed
    progress_bar.empty()

    # Text formatting for model parameters style in user message:
    text = str(ranked_naive_model2_tuning_results.iloc[0, 1])
    formatted_text = re.sub(r'([^:]+): ([^,]+)', r'`\1`: **\2**', text)
    final_text = re.sub(r',\s*', ', ', formatted_text)

    st.write(
        f'''💬**Naive Model II** parameters with the lowest MAPE of {ranked_naive_model2_tuning_results["MAPE"][0]} found in **{end_time_naive_model_2 - start_time_naive_model_2:.2f}** seconds is:  
             {final_text}''')


def run_naive_model_3(agg_options=None, train_set=None, validation_set=None,
                      max_wait_time=datetime.time(0, 5), rolling_window_range=None,
                      rolling_window_options=None, lag_options=None, metrics_dict=None, progress_bar=st.progress(0)):
    start_time = time.time()
    max_wait_time_minutes = int(str(max_wait_time.minute))
    max_wait_time_seconds = max_wait_time_minutes * 60
    # Store total number of combinations in the parameter grid for progress bar
    total_options = len(lag_options) + (len(list(range(rolling_window_range[0], rolling_window_range[1] + 1)))
                                        * len(rolling_window_options)) + len(agg_options)

    # Define Title
    my_text_paragraph('Naive Model III: Constant Value')

    # Define Model Name
    model_name = 'Naive Model III'

    # Initiate dataframe to save results to with constant (mean/median/mode)
    naive_model3_tuning_results = pd.DataFrame()

    # Iterate over grid of all possible combinations of hyperparameters
    param_grid = {'agg_options': agg_options}

    # set the end of runtime
    start_time_naive_model_3 = time.time()

    for i, agg_option in enumerate(param_grid['agg_options']):

        # Check if the maximum waiting time has been exceeded
        elapsed_time_seconds = time.time() - start_time

        # If maximum time is exceeded, stop running loop
        if elapsed_time_seconds > max_wait_time_seconds:
            st.warning("Maximum waiting time exceeded. The grid search has been stopped.")
            # exit the loop once maximum time is exceeded defined by user or default = 5 minutes
            break

        # Update the progress bar
        progress_percentage = (i + len(agg_options) + (
                len(list(range(rolling_window_range[0], rolling_window_range[1] + 1))) * len(
            rolling_window_options))) / total_options
        progress_text = f'''Please wait up to {max_wait_time_minutes} minute(s) while parameters of Naive Models are being tuned!
        \n{progress_percentage:.2f}% of total options within the search space reviewed ({i + len(agg_options)
                                                                                         + (len(list(range(rolling_window_range[0], rolling_window_range[1] + 1))) * len(rolling_window_options))} out of {total_options} total options).'''
        progress_bar.progress(value=progress_percentage, text=progress_text)

        # return dataframe with the predictions e.g. constant of mean/median/mode of training dataset for length of
        # validation set
        df_preds_naive_model3 = forecast_naive_model3_insample(train_set,
                                                               validation_set,
                                                               agg_method=agg_option)

        # retrieve metrics from difference actual - predicted
        metrics_dict = my_metrics(my_df=df_preds_naive_model3, model_name=model_name, metrics_dict=metrics_dict)

        # Create a new DataFrame with the row to append
        new_row = pd.DataFrame({'parameters': [f'aggregation method: {agg_option}'],
                                'MAPE': metrics_dict[model_name]['mape'],
                                'SMAPE': metrics_dict[model_name]['smape'],
                                'RMSE': metrics_dict[model_name]['rmse'],
                                'R2': metrics_dict[model_name]['r2']})

        # Concatenate the original DataFrame with the new row DataFrame
        naive_model3_tuning_results = pd.concat([naive_model3_tuning_results, new_row], ignore_index=True)

        # add rank column to dataframe and order by metric column
        ranked_naive_model3_tuning_results = rank_dataframe(naive_model3_tuning_results, 'MAPE')

        # convert mape to %
        ranked_naive_model3_tuning_results[['MAPE', 'SMAPE']] = ranked_naive_model3_tuning_results[
            ['MAPE', 'SMAPE']].map(lambda x: f'{x * 100:.2f}%' if not np.isnan(x) else x)

    st.dataframe(ranked_naive_model3_tuning_results, use_container_width=True, hide_index=True)

    # set the end of runtime
    end_time_naive_model_3 = time.time()

    # clear progress bar in streamlit for user as process is completed
    progress_bar.empty()

    # text formatting for model parameters style in user message:
    text = str(ranked_naive_model3_tuning_results.iloc[0, 1])
    formatted_text = re.sub(r'([^:]+): ([^,]+)', r'`\1`: **\2**', text)
    final_text = re.sub(r',\s*', ', ', formatted_text)

    st.write(
        f'''💬**Naive Model III** parameter with the lowest MAPE of {ranked_naive_model3_tuning_results["MAPE"][0]} 
        found in **{end_time_naive_model_3 - start_time_naive_model_3:.2f}** seconds is:      {final_text}''')


def create_hyperparameter_importance_plot(data, plot_type='param_importances', grid_search=False):
    """
    Create a hyperparameter importance plot.

    Parameters:
        data (object or list): The data required to create the plot. For 'param_importances' plot_type,
                              it can be a study object from Optuna. For 'bar_chart' plot_type, it should
                              be a list of tuples containing hyperparameter and importance values.
        plot_type (str, optional): The type of plot to be created. Defaults to 'param_importances'.
        grid_search (bool, optional): Specifies if the plot is for regular grid search. Defaults to False.

    Returns:
        None

    Output:
        Streamlit Figure: Displays the hyperparameter importance plot in Streamlit.

    """
    fig = go.Figure()
    color_schema = px.colors.qualitative.Plotly

    if plot_type == 'param_importances' and not grid_search:
        fig = ov.plot_param_importances(data)
        fig.update_layout(title='', margin=dict(t=10))
        for i, trace in enumerate(fig.data):
            trace.marker.color = [color_schema[j % len(color_schema)] for j in range(len(trace.y))]

    elif plot_type == 'bar_chart':
        sorted_feature_importance = data
        for hyperparameter, importance in sorted_feature_importance:
            fig.add_trace(go.Bar(
                x=[importance],
                y=[hyperparameter],
                orientation='h',
                name=hyperparameter,
                marker=dict(color=color_schema[len(fig.data) % len(color_schema)]),
            ))
        fig.update_layout(title='', margin=dict(t=10), xaxis=dict(title='Importance Score'),
                          yaxis=dict(title='Hyperparameter'))

    # Set figure title centered in Streamlit
    my_text_paragraph('Hyperparameter Importance Plot')

    # Show the figure in Streamlit with hyperparameter importance values
    st.plotly_chart(fig, use_container_width=True)
