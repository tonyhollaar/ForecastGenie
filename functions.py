# IMPORT STANDARD LIBRARIES
import pandas as pd
import math
import time
import base64
import numpy as np
from typing import IO

from typing import Union
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
import sympy as sp

import altair as alt

from fire_state import create_store, form_update, get_state, set_state

# from sklearn.neighbors import NearestNeighbors

from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Statistical tools
from scipy import stats
from scipy.stats import mode, kurtosis, skew, shapiro
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
# from statsmodels.stats.diagnostic import acorr_ljungbox
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Data (Pre-) Processing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer, \
    QuantileTransformer
from sklearn.svm import SVR
from st_click_detector import click_detector

# from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf, adfuller
from streamlit_extras.buy_me_a_coffee import button

# Text manipulation
import re
import json

# Image Processing
# from PIL import Image

# Local Packages
from style.text import my_subheader, my_text_paragraph, my_text_header, my_header, my_title, vertical_spacer
from style.icons import load_icons

# Load icons
icons = load_icons()

def rank_dataframe(df, metric):
    """
    Rank the DataFrame based on the specified column.

    Args:
        df (pd.DataFrame): The DataFrame to be ranked.
        metric (str): The name of the column to rank the DataFrame.

    Returns:
        pd.DataFrame: The ranked DataFrame.
    """
    # Convert the specified column to numeric data type
    df[metric] = pd.to_numeric(df[metric], errors='coerce')

    # Sort the DataFrame by the specified column in ascending order
    df = df.sort_values(metric)

    # Reset the index of the DataFrame and assign ranks
    df = df.reset_index(drop=True).reset_index().rename(columns={'index': 'Rank'})
    return df


def clear_test_results():
    """
    Clears the user test results from memory.

    This function deletes both the latest user test results and the historical test results from memory.
    It updates the 'results_df' entry in the Streamlit session state with an empty DataFrame,
    resetting it to the initial state with the specified column names.

    Note:
        This function assumes that the Streamlit session state is being used and the 'results_df' key exists.

    Example usage:
        clear_test_results()
    """
    # Delete latest user test results from memory
    del st.session_state['results_df']

    # Delete historical test results from memory
    set_state("TRAIN_PAGE", ("results_df", pd.DataFrame(
        columns=['model_name', 'predicted', 'mape', 'smape', 'rmse', 'r2', 'features', 'model_settings'])))

    # do you want to remove checkboxes when user presses clear? if so add below codeblock for each model checkbox on page train sidebar
    # =============================================================================
    #                 set_state("TRAIN_PAGE", ("naive_checkbox", False))
    #                 set_state("TRAIN_PAGE", ("linreg_checkbox", False))
    #                 set_state("TRAIN_PAGE", ("sarimax_checkbox", False))
    #                 set_state("TRAIN_PAGE", ("prophet_checkbox", False))
    # =============================================================================

def plot_model_performance(df, my_metric):
    """
    Plots a bar chart showing the models with the lowest error or highest R2 score.

    Args:
        df (pd.DataFrame): The DataFrame containing the model data.
        selected_metric (str): The metric to display and compare for the models. It can be one of the following:
                               - 'mape' (mean absolute percentage error)
                               - 'rmse' (root mean squared error)
                               - 'r2' (coefficient of determination)

    Note:
        The function assumes that the 'mape', 'mse', 'rmse', and 'r2' columns exist in the DataFrame.

    Example usage:
        df = pd.DataFrame(...)
        plot_lowest_metric(df, 'mape')
    """
    # convert the metric name from user selection to metric abbreviation used in results dataframe
    metric_dict = {'Mean Absolute Percentage Error': 'mape', 'Symmetric Mean Absolute Percentage Error': 'smape',
                   'Root Mean Square Error': 'rmse', 'R-squared': 'r2'}
    selected_metric = metric_dict[my_metric]

    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Remove percentage sign from 'mape'  column and convert to float
    df_copy['mape'] = df_copy['mape'].str.replace('%', '').astype(float)
    df_copy['smape'] = df_copy['smape'].str.replace('%', '').astype(float)

    # Convert the selected metric column to numeric data type
    df_copy[selected_metric] = pd.to_numeric(df_copy[selected_metric], errors='coerce')

    # Group the dataframe by 'model_name' and find the row with the lowest score for each model and selected metric
    lowest_score_rows = df_copy.groupby('model_name')[selected_metric].idxmin()

    # Filter the dataframe to keep only the rows with the lowest scores for each model and selected metric
    filtered_df = df_copy.loc[lowest_score_rows, ['model_name', selected_metric]]

    # Extract the unique model names from the filtered dataframe
    model_names = filtered_df['model_name'].unique()

    # Create a color mapping dictionary for consistent colors based on model names
    color_schema = px.colors.qualitative.Plotly
    color_mapping = {model_name: color_schema[i % len(color_schema)] for i, model_name in enumerate(model_names)}

    # Sort the filtered dataframe based on the selected metric in ascending or descending order
    ascending_order = selected_metric != 'r2'  # Set ascending order unless 'r2' is selected
    filtered_df = filtered_df.sort_values(by=selected_metric, ascending=ascending_order)

    # Create the bar chart using Plotly Express
    fig = px.bar(
        filtered_df,
        x='model_name',
        y=selected_metric,
        color='model_name',
        color_discrete_map=color_mapping,  # Use the custom color mapping
        category_orders={'model_name': filtered_df['model_name'].tolist()},  # Set the order of model names
        text=filtered_df[selected_metric].round(2),  # Add labels to the bars with rounded metric values
    )

    fig.update_layout(
        xaxis_title='Model',
        yaxis_title=selected_metric.upper(),
        barmode='stack',
        showlegend=True,
        legend=dict(x=1, y=1),  # Adjust the position of the legend
    )
    fig.update_traces(textposition='inside', textfont=dict(color='white'),
                      insidetextfont=dict(color='white'))  # Adjust the position and color of the labels

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def generate_demo_data(seed=42):
    """
    Generate demo data with weekly, monthly, quarterly, and yearly patterns.

    Args:
        seed (int): Random seed for reproducibility.

    Returns:
        pandas.DataFrame: A DataFrame with two columns, 'date' and 'demo_data', where 'date'
        is a range of dates from 2018-01-01 to 2022-12-31, and 'demo_data' is a time series
        with weekly, monthly, quarterly, and yearly patterns plus random noise.
    """
    np.random.seed(seed)
    date_range = pd.date_range(start='1/1/2018', end='12/31/2022', freq='D')
    # generate weekly pattern
    weekly = 10 * np.sin(2 * np.pi * (date_range.dayofweek / 7))
    # generate monthly pattern
    monthly = 10 * np.sin(2 * np.pi * (date_range.month / 12))
    # generate quarterly pattern
    quarterly = 10 * np.sin(2 * np.pi * (date_range.quarter / 4))
    # generate yearly pattern
    yearly = 10 * np.sin(2 * np.pi * (date_range.year - 2018) / 4)
    # generate random variation
    noise = np.random.normal(loc=0, scale=2, size=len(date_range))
    # generate time series data
    values = weekly + monthly + quarterly + yearly + noise + 10
    demo_data = pd.DataFrame({'date': date_range, 'demo_data': values})
    return demo_data

def update_color(selected_color):
    # Save the selected color in session state
    set_state("COLORS", ("chart_color", selected_color))

def form_feature_selection_method_mifs(X_train: pd.DataFrame, streamlit_key: Union[str, int]):
    try:
        with st.form('mifs'):
            my_text_paragraph('Mutual Information')

            # Add slider to select number of top features
            num_features_mifs = st.slider(
                label="*Select number of top features to include:*",
                min_value=0,
                max_value=len(X_train.columns),
                key=streamlit_key,
                step=1
            )

            col1, col2, col3 = st.columns([4, 4, 4])
            with col2:
                mifs_btn = st.form_submit_button("Submit", type="secondary", on_click=form_update,
                                                 args=('SELECT_PAGE_MIFS',))

                # keep track of changes if mifs_btn is pressed
                if mifs_btn:
                    set_state("SELECT_PAGE_BTN_CLICKED", ('mifs_btn', True))
            return num_features_mifs
    except:
        st.error('the user form for feature selection method \'Mutual Information\' could not execute')


def form_feature_selection_method_pca(X_train: pd.DataFrame, streamlit_key: Union[str, int]):
    try:
        with st.form('pca'):

            my_text_paragraph('Principal Component Analysis')

            # Add a slider to select the number of features to be selected by the PCA algorithm
            num_features_pca = st.slider(
                label='*Select number of top features to include:*',
                min_value=0,
                max_value=len(X_train.columns),
                key=streamlit_key
            )

            col1, col2, col3 = st.columns([4, 4, 4])
            with col2:
                pca_btn = st.form_submit_button(label="Submit", type="secondary", on_click=form_update,
                                                args=('SELECT_PAGE_PCA',))
                if pca_btn:
                    set_state("SELECT_PAGE_BTN_CLICKED", ('pca_btn', True))
            return num_features_pca
    except KeyError:
        st.error('KeyError: the key used in the form or slider does not exist.')
    except ValueError:
        st.error('ValueError: an inappropriate value was used.')
    except Exception as e:
        st.error(f'Unexpected error occurred: {e}')


def form_feature_selection_method_rfe():
    try:
        # show Title in sidebar 'Feature Selection' with purple background
        my_title(f'{icons["select_icon"]}', "#3b3b3b", gradient_colors="#1A2980, #7B52AB, #FEBD2E")

        # =============================================================================
        # RFE Feature Selection - SIDEBAR FORM
        # =============================================================================
        with st.form('rfe'):
            my_text_paragraph('Recursive Feature Elimination')
            # Add a slider to select the number of features to be selected by the RFECV algorithm

            num_features_rfe = st.slider(
                label=f'Select number of top features to include*:',
                min_value=0,
                max_value=len(st.session_state['X'].columns),
                key=key1_select_page_rfe,
                help='**`Recursive Feature Elimination (RFE)`** is an algorithm that iteratively removes the least important features from the feature set until the desired number of features is reached.\
                                                  \nIt assigns a rank to each feature based on their importance scores. It is possible to have multiple features with the same ranking because the importance scores of these features are identical or very close to each other.\
                                                  \nThis can happen when the features are highly correlated or provide very similar information to the model. In such cases, the algorithm may not be able to distinguish between them and assign the same rank to multiple features.'
            )

            # set the options for the rfe (recursive feature elimination)
            with st.expander('◾', expanded=False):
                # Include duplicate ranks (True/False) because RFE can have duplicate features with the same ranking
                duplicate_ranks_rfe = st.selectbox(label='*\*allow duplicate ranks:*',
                                                   options=[True, False],
                                                   key=key5_select_page_rfe,
                                                   help='''
                                                           Allow multiple features to be added to the feature selection with the same ranking when set to `True`. otherwise, when the option is set to `False`, only 1 feature will be added to the feature selection. 
                                                           ''')

                # Add a selectbox for the user to choose the estimator
                estimator_rfe = st.selectbox(
                    label='*set estimator:*',
                    options=['Linear Regression', 'Random Forest Regression'],
                    key=key2_select_page_rfe,
                    help='''
                                                     The **`estimator`** parameter is used to specify the machine learning model that will be used to evaluate the importance of each feature. \
                                                     The estimator is essentially the algorithm used to fit the data and make predictions.
                                                     ''')

                # Add a slider to select the number of n_splits for the RFE method
                timeseriessplit_value_rfe = st.slider(label='*set number of splits for cross-validation:*',
                                                      min_value=2,
                                                      max_value=5,
                                                      key=key3_select_page_rfe,
                                                      help='**`Cross-validation`** is a statistical method used to evaluate the performance of a model by splitting the dataset into multiple "folds," where each fold is used as a holdout set for testing the model trained on the remaining folds. \
                                                              The cross-validation procedure helps to estimate the performance of the model on unseen data and reduce the risk of overfitting.  \
                                                              In the context of RFE, the cv parameter specifies the number of folds to use for the cross-validation procedure.\
                                                              The RFE algorithm fits the estimator on the training set, evaluates the performance of the estimator on the validation set, and selects the best subset of features. \
                                                              The feature selection process is repeated for each fold of the cross-validation procedure.')

                # Add a slider in the sidebar for the user to set the number of steps parameter
                num_steps_rfe = st.slider(label='*set number of steps:*',
                                          min_value=1,
                                          max_value=10,
                                          # value = 1,
                                          key=key4_select_page_rfe,
                                          help='The `step` parameter controls the **number of features** to remove at each iteration of the RFE process.')

            col1, col2, col3 = st.columns([4, 4, 4])
            with col2:
                rfe_btn = st.form_submit_button("Submit", type="secondary", on_click=form_update,
                                                args=('SELECT_PAGE_RFE',))

                if rfe_btn:
                    # update session state with user preference feature selection
                    set_state("SELECT_PAGE_BTN_CLICKED", ('rfe_btn', True))

            return num_features_rfe, duplicate_ranks_rfe, estimator_rfe, timeseriessplit_value_rfe, num_steps_rfe
    except:
        st.error('the user form for feature selection method \'Recursive Feature Selection\' could not execute')


def show_card_feature_selection_methods():
    """
   Displays feature selection methods using a Streamlit expander and carousel cards.

   Parameters:
       None

   Returns:
       None
   """
    with st.expander('', expanded=True):
        # vertical_spacer(2)
        col1, col2, col3 = st.columns([3, 8, 2])
        with col2:
            title = 'Select your top features with 3 methods!'

            # set gradient color of letters of title
            # gradient = '-webkit-linear-gradient(left, #9c27b0, #673ab7, #3f51b5, #2196f3, #03a9f4)'
            gradient = '-webkit-linear-gradient(left, #FFFFF, #7c91b4, #a9b9d2, #7b8dad, #69727c)'

            # show in streamlit the title with gradient
            st.markdown(
                f'<h1 style="text-align:center; font-family: Ysabeau SC; font-size: 30px; background: none; -webkit-background-clip: text;"> {title} </h1>',
                unsafe_allow_html=True)

            vertical_spacer(2)

            #### CAROUSEL ####
            # header_list = ['🎨', '🧮', '🎏']
            header_list = ['', '', '']
            paragraph_list_front = [
                "<b> Recursive Feature Elimination </b>",
                "<b>Principal Component Analysis</b>",
                "<b>Mutual Information</b>"
            ]
            paragraph_list_back = [
                "<b>RFE</b> is a <b>feature selection technique</b> that repeatedly removes feature(s) and each turn evaluates remaining features by ranking the features based on their importance scores and eliminates the least important feature. This process continues until a desired number of features is reached.",
                "<b>PCA</b> is a <b>feature selection technique</b> that repeatedly transforms and evaluates features based on their variance, reducing the dataset to a smaller set of uncorrelated variables called principal components. This process continues until a desired number of components is achieved.",
                "<b>MI</b> is a <b>feature selection technique</b> that calculates the mutual information between each feature and the target to determine how much information each feature provides about the target."
            ]
            # font_family = "Ysabeau SC"
            font_family = "Josefin Slab"
            font_size_front = '16px'
            font_size_back = '15px'

            # in Streamlit create and show the user defined number of carousel cards with header+text
            create_carousel_cards_v3(3, header_list, paragraph_list_front, paragraph_list_back, font_family,
                                     font_size_front, font_size_back)
            vertical_spacer(2)

        # Display a NOTE to the user about using the training set for feature selection
        my_text_paragraph(
            'NOTE: per common practice <b>only</b> the training dataset is used for feature selection to prevent <b>data leakage</b>.',
            my_font_size='12px', my_font_family='Arial')


def analyze_feature_correlations(
        selected_corr_model: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        selected_cols_rfe: list,
        selected_cols_pca: list,
        selected_cols_mifs: list,
        corr_threshold: float = 0.8,
        models: dict = {
            'Linear Regression': LinearRegression(),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=100)
        }
) -> tuple:
    """
    Analyzes feature correlations and computes importance scores.

    Args:
        X_train (pd.DataFrame): Training dataset of independent features.
        corr_threshold (float): Threshold for considering highly correlated features.
        selected_cols_rfe (list): List of selected columns from Recursive Feature Elimination (RFE).
        selected_cols_pca (list): List of selected columns from Principal Component Analysis (PCA).
        selected_cols_mifs (list): List of selected columns from Mutual Information Feature Selection (MIFS).
        models (dict): Dictionary of models with model names as keys and model objects as values.

    Returns:
        tuple: A tuple containing the following:
            - total_features (list): List of total features (selected_cols_rfe + selected_cols_pca + selected_cols_mifs).
            - importance_scores (pd.Series or np.array): Feature importance scores or permutation importance scores.
            - pairwise_features_in_total_features (list): List of pairwise features that are in total_features.
    """
    # Create correlation matrix from training dataset of independent features
    corr_matrix = X_train.corr()

    # Get the indices of the highly correlated features
    indices = np.where(abs(corr_matrix) >= corr_threshold)

    # Create a dataframe with the pairwise correlation values above the threshold
    df_pairwise = pd.DataFrame({
        'feature1': corr_matrix.columns[indices[0]],
        'feature2': corr_matrix.columns[indices[1]],
        'correlation': corr_matrix.values[indices]
    })

    # Sort feature pairs and drop duplicates
    df_pairwise = df_pairwise.assign(
        sorted_features=df_pairwise[['feature1', 'feature2']].apply(sorted, axis=1).apply(tuple))
    df_pairwise = df_pairwise.loc[df_pairwise['feature1'] != df_pairwise['feature2']].drop_duplicates(
        subset='sorted_features').drop(columns='sorted_features')

    # Sort by correlation and format output
    df_pairwise = df_pairwise.sort_values(by='correlation', ascending=False).reset_index(drop=True)
    df_pairwise['correlation'] = (df_pairwise['correlation'] * 100).apply('{:.2f}%'.format)

    # Find pairs in total_features
    total_features = np.unique(selected_cols_rfe + selected_cols_pca + selected_cols_mifs).tolist()

    pairwise_features = list(df_pairwise[['feature1', 'feature2']].itertuples(index=False, name=None))
    pairwise_features_in_total_features = [pair for pair in pairwise_features if
                                           pair[0] in total_features and pair[1] in total_features]

    # Create estimator based on user-selected model
    estimator = models[selected_corr_model]
    estimator.fit(X_train, y_train)

    # Compute feature importance scores or permutation importance scores
    importance_scores = compute_importance_scores(X_train, y_train, estimator)

    return total_features, importance_scores, pairwise_features_in_total_features, df_pairwise


def remove_lowest_importance_feature(total_features, importance_scores, pairwise_features_in_total_features):
    lowest_importance_features = []
    for feature1, feature2 in pairwise_features_in_total_features:
        if feature1 in total_features and feature2 in total_features:
            score1 = importance_scores[feature1]
            score2 = importance_scores[feature2]
            if score1 < score2:
                lowest_importance_features.append(feature1)
            else:
                lowest_importance_features.append(feature2)
    updated_total_features = [feature for feature in total_features if feature not in lowest_importance_features]
    return updated_total_features


def show_rfe_plot(rfecv, selected_features):
    """
    Show the Recursive Feature Elimination (RFE) plot and feature rankings in Streamlit.

    Parameters:
        rfecv (sklearn.feature_selection.RFECV): The fitted RFECV model.
        selected_features (list): The list of selected feature/column names.

    Returns:
        None

    """
    #############################################################
    # Scatterplot the results
    #############################################################
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

    # Create the rfe plot
    vertical_spacer(2)
    fig = create_rfe_plot(df_ranking)
    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
    # show the ranking and selected features dataframes side by side
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    with col2:
        st.write(':blue[**Selected features:**]', selected_features)
    # Print the feature rankings
    with col3:
        feature_rankings = pd.Series(rfecv.ranking_, index=X.columns).rename('Ranking')
        st.write(':blue[**Feature rankings:**]')
        st.write(feature_rankings.sort_values())

    #############################################################
    # Show in Streamlit the ranking
    #############################################################
    selected_cols_rfe = list(selected_features)
    st.info(f'Top {len(selected_cols_rfe)} features selected with RFECV: {selected_cols_rfe}')

    show_rfe_info_btn = st.button(f'About RFE plot', use_container_width=True, type='secondary')

    # if user clicks "About RFE plot" button, display information about RFE
    if show_rfe_info_btn:
        st.write('')
        # show user info about how to interpret the graph
        st.markdown('''
                    **Recursive Feature Elimination** involves recursively removing features and building a model on the remaining features. It then **ranks the features** based on their importance and **eliminates** the **least important feature**.
                    ''')


def perform_rfe(X_train, y_train, estimator_rfe, duplicate_ranks_rfe, num_steps_rfe, num_features_rfe,
                timeseriessplit_value_rfe):
    """
    Perform Recursive Feature Elimination with Cross-Validation and display the results using a scatter plot.

    Parameters:
        X_train (pandas.DataFrame): Training data features.
        y_train (pandas.Series): Training data target.
        estimator_rfe (estimator object): A supervised learning estimator with a `fit` method.
        num_steps_rfe (int): Number of features to remove at each iteration of RFE.
        duplicate_ranks_rfe (bool): allow duplicate ranks if set to True otherwise take first feature(s) from list when ordered by rank ascending
        num_features (int): Number of features to select, defaults to None.
        timeseriessplit_value_rfe (int): Number of splits in time series cross-validation.

    Returns:
        None
    """
    # Set up the estimator based on the user's selection
    if estimator_rfe == 'Linear Regression':
        est_rfe = LinearRegression()
    elif estimator_rfe == 'Random Forest Regression':
        est_rfe = RandomForestRegressor()

    #############################################################
    # Recursive Feature Elemination
    #############################################################
    # Define the time series splits set by user in sidebar slider
    tscv = TimeSeriesSplit(n_splits=timeseriessplit_value_rfe)

    # Set up the recursive feature elimination with cross validation
    rfecv = RFECV(estimator=est_rfe,
                  step=num_steps_rfe,
                  cv=tscv,
                  scoring='neg_mean_squared_error',
                  # ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error'] accuracy
                  min_features_to_select=num_features_rfe,
                  n_jobs=-1)

    # Fit the feature selection model
    rfecv.fit(X_train, y_train)

    user_message = f"note: optimal number of features suggested = {rfecv.n_features_}"

    if num_features_rfe is not None:
        selected_features = X_train.columns[rfecv.ranking_ <= num_features_rfe]
    else:
        selected_features = X_train.columns[rfecv.support_]

    # if user selected to allow duplicate rankings (default = True) use all ranks
    # example: when two features or more have rank 1, keep both features and it overrides the selected number of features by user initially set
    if duplicate_ranks_rfe == True:
        selected_cols_rfe = list(selected_features)
    # if user doesn't want duplicate ranking just get the first n items in the list with m features whereby n <= m
    elif duplicate_ranks_rfe == False:
        # Get the feature ranking
        feature_rankings = pd.Series(rfecv.ranking_, index=X_train.columns).rename('Ranking')
        # Sort the feature rankings in descending order
        sorted_rankings = feature_rankings.sort_values(ascending=True)
        # Create a dataframe with feature rankings and selected features
        df_ranking = pd.DataFrame({'Features': sorted_rankings.index, 'Ranking': sorted_rankings})
        # Sort the dataframe by ranking
        df_ranking = df_ranking.sort_values('Ranking', ascending=True)

        # only keep top x features, so slice and reindex to 0,1,2,3
        selected_features = df_ranking.iloc[:num_features_rfe, 0].reset_index(drop=True)
        selected_cols_rfe = list(selected_features)

        # st.write('selected features', selected_features, selected_cols_rfe) # TEST

    return selected_cols_rfe, selected_features, rfecv, user_message


def perform_mifs(X_train, y_train, num_features_mifs):
    """
    Perform Mutual Information Feature Selection (MIFS) on the training data.

    Parameters:
        X_train (pandas.DataFrame): The training data.
        y_train (pandas.Series or numpy.ndarray): The target variable for training.
        num_features_mifs (int): The number of top features to select based on mutual information.

    Returns:
        tuple: A tuple containing the following:
            - mutual_info (numpy.ndarray): The array of mutual information values.
            - selected_features_mi (numpy.ndarray): The selected feature/column names based on mutual information.
            - selected_cols_mifs (list): The selected feature/column names as a list.

    """
    mutual_info = mutual_info_regression(X_train, y_train, random_state=42)
    selected_features_mi = X_train.columns[np.argsort(mutual_info)[::-1]][:num_features_mifs]
    selected_cols_mifs = list(selected_features_mi)
    return mutual_info, selected_features_mi, selected_cols_mifs


def show_mifs_plot(mutual_info, selected_features_mi, num_features_mifs):
    """
    Display the Mutual Information Feature Selection (MIFS) plot.

    Parameters:
        mutual_info (numpy.ndarray): The array of mutual information values.
        selected_features_mi (list): The list of selected features based on mutual information.
        num_features_mifs (int): The number of top features to display in the plot.

    Returns:
        None

    """
    vertical_spacer(2)
    my_text_paragraph(' Mutual Information', my_font_size='26px', )
    my_text_paragraph(f'<b> TOP {num_features_mifs} </b>', my_font_size='16px', my_font_family='Segui UI')

    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mutual_info[np.argsort(mutual_info)[::-1]][:num_features_mifs],
        y=selected_features_mi,
        orientation='h',
        text=[f'{val:.2f}' for val in mutual_info[np.argsort(mutual_info)[::-1]][:num_features_mifs]],
        textposition='inside',
        marker_color='#4715EF'  # Set the desired color code
    ))

    fig.update_layout(
        title={
            'text': '',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        }, margin=dict(t=0, b=0),
        xaxis_title='Mutual Information Values',
        yaxis_title='Feature Name'
    )

    # Display plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    ##############################################################
    # SELECT YOUR FAVORITE FEATURES TO INCLUDE IN MODELING
    ##############################################################
    st.info(f'Top {num_features_mifs} features selected with MIFS: {list(selected_features_mi)}')

    # create button to display information about mutual information feature selection
    show_mifs_info_btn = st.button(f'About MIFS plot', use_container_width=True, type='secondary')

    if show_mifs_info_btn:
        st.write('')
        # show user info about how to interpret the graph
        st.markdown('''
            Mutual Information Feature Selection (MIFS) is a method for selecting the most important features in a dataset for predicting a target variable.  
            It measures the mutual information between each feature and the target variable, 
            using an entropy-based approach to quantify the amount of information that each feature provides about the target.  
            Features with high mutual information values are considered to be more important in predicting the target variable
            and features with low mutual information values are considered to be less important.  
            MIFS helps improve the accuracy of predictive models by identifying the most informative features to include in the model.
        ''')


def perform_pca(X_train, num_features_pca):
    """
    Perform Principal Component Analysis (PCA) on the training data.

    Parameters:
        X_train (pandas.DataFrame): The training data.
        num_features_pca (int): The desired number of components for PCA.

    Returns:
        list: The selected feature/column names based on PCA.

    """
    # Create a PCA object with the desired number of components
    pca = PCA(n_components=num_features_pca)
    # Fit the PCA model to the training data
    pca.fit(X_train)
    # Get the names of the features/columns in the training data
    feature_names = X_train.columns
    # Sort the features based on the explained variance ratio in descending order
    sorted_idx = np.argsort(pca.explained_variance_ratio_)[::-1]
    # Reorder the feature names based on the sorted indices
    sorted_features = feature_names[sorted_idx]
    # Convert the sorted features to a list
    selected_cols_pca = sorted_features.tolist()

    # =============================================================================
    #     # Get the explained variance for each feature
    #     variances = pca.explained_variance_
    #     st.write(variances)
    # =============================================================================
    return sorted_features, pca, sorted_idx, selected_cols_pca


def show_pca_plot(sorted_features, pca, sorted_idx, selected_cols_pca):
    """
    Display a PCA plot and information about the selected features.

    Parameters:
        - sorted_features (pandas.Series or list): A pandas Series or list containing the sorted feature names.
        - pca (sklearn.decomposition.PCA): The fitted PCA object.
        - sorted_idx (list or array-like): The sorted indices of the features.

    Returns:
        None
    """
    vertical_spacer(2)
    my_text_paragraph(f'Principal Component Analysis', my_font_size='26px')
    my_text_paragraph(f'<b>TOP {len(sorted_features)}</b>', my_font_size='16px', my_font_family='Segui UI')

    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pca.explained_variance_ratio_[sorted_idx],
        y=sorted_features,
        orientation='h',
        text=np.round(pca.explained_variance_ratio_[sorted_idx] * 100, 2),
        textposition='auto',
        marker_color='#4715EF'
    ))
    fig.update_layout(
        title={
            'text': '',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(t=0, b=0),
        xaxis_title='Explained Variance Ratio',
        yaxis_title='Feature Name'
    )

    # Display plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    st.info(f'Top {len(selected_cols_pca)} features selected with PCA: {selected_cols_pca}')

    # Show "About PCA plot" button
    show_pca_info_btn = st.button(f'About PCA plot', use_container_width=True, type='secondary')

    if show_pca_info_btn == True:
        vertical_spacer(1)
        # show user info about how to interpret the graph
        st.markdown('''When you fit a **PCA** model, it calculates the amount of variance that is captured by each principal component.
                    The variance ratio is the fraction of the total variance in the data that is explained by each principal component.
                    The sum of the variance ratios of all the principal components equals 1.
                    The variance ratio is expressed as a percentage by multiplying it by 100, so it can be easily interpreted.  
                    ''')
        st.markdown('''
                    For example, a variance ratio of 0.75 means that 75% of the total variance in the data is captured by the corresponding principal component.
                    ''')


def show_model_inputs():
    """
    Displays a set of buttons and corresponding dataframes based on user clicks.

    This function shows a set of buttons for different datasets and features (X, y).
    When a button is clicked, the corresponding dataframe is displayed below.

    Returns:
        None
    """
    my_text_header('Model Inputs')

    # =============================================================================
    #     show_lottie_animation(url="./images/86093-data-fork.json", key='inputs')
    # =============================================================================

    col1, col2, col3 = st.columns([7, 120, 1])
    with col2:

        create_flipcard_model_input(image_path_front_card='./images/model_inputs.png',
                                    my_string_back_card='Your dataset is split into two parts: a <i style="color:#67d0c4;">train dataset </i> and a <i style="color:#67d0c4;">test dataset</i>. The former is used to train the model, allowing it to learn from the features by adjusting its parameters to minimize errors and improve its predictive abilities. The latter dataset is then used to assess the performance of the trained model on new or unseen data and its ability to make accurate predictions.')
    vertical_spacer(1)

    col1, col2, col3 = st.columns([25, 40, 20])
    with col2:
        color_train = '#07080d'
        color_test = '#67d0c4'
        button_container = """
            <div style="display: flex; flex-direction: column; align-items: center; padding: 10px; margin-top: -30px;">
                <div style="display: flex; width: 100%; max-width: 400px;">
                    <div style="flex: 1; background: {color_train}; border-radius: 20px 0 0 20px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2), 0px 1px 3px rgba(0, 0, 0, 0.1);">
                        <a href='#' id='train' style="display: flex; justify-content: center; align-items: center; height: 100%; text-decoration: none; background: {color_train}; border-radius: 5px; padding: 10px;">
                            <span style="color: #FFFFFF; font-size: 14px;">Train</span>
                        </a>
                    </div>
                    <div style="flex: 1; background: {color_train}; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2), 0px 1px 3px rgba(0, 0, 0, 0.1);">
                        <a href='#' id='X_train' style="display: flex; justify-content: center; align-items: center; height: 100%; text-decoration: none; background: {color_train}; padding: 10px;">
                            <span style="color: #FFFFFF; font-size: 14px;">Train X</span>
                        </a>
                    </div>
                    <div style="flex: 1; background: {color_train}; border-radius: 0 20px 20px 0; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2), 0px 1px 3px rgba(0, 0, 0, 0.1);">
                        <a href='#' id='y_train' style="display: flex; justify-content: center; align-items: center; height: 100%; text-decoration: none; background: {color_train}; border-radius: 5px; padding: 10px;">
                            <span style="color: #FFFFFF; font-size: 14px;">Train y</span>
                        </a>
                    </div>
                </div>
                <div style="display: flex; width: 100%; max-width: 400px; margin-top: 10px;">
                    <div style="flex: 1; background: {color_test}; border-radius: 20px 0 0 20px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2), 0px 1px 3px rgba(0, 0, 0, 0.1);">
                        <a href='#' id='test' style="display: flex; justify-content: center; align-items: center; height: 100%; text-decoration: none; background: {color_test}; border-radius: 5px; padding: 10px;">
                            <span style="color: #FFFFFF; font-size: 14px;">Test</span>
                        </a>
                    </div>
                    <div style="flex: 1; background: {color_test}; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2), 0px 1px 3px rgba(0, 0, 0, 0.1);">
                        <a href='#' id='X_test' style="display: flex; justify-content: center; align-items: center; height: 100%; text-decoration: none; background: {color_test}; border-radius: 5px; padding: 10px;">
                            <span style="color: #FFFFFF; font-size: 14px;">Test X</span>
                        </a>
                    </div>
                    <div style="flex: 1; background: {color_test}; border-radius: 0 20px 20px 0; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2), 0px 1px 3px rgba(0, 0, 0, 0.1);">
                        <a href='#' id='y_test' style="display: flex; justify-content: center; align-items: center; height: 100%; text-decoration: none; background: {color_test}; border-radius: 5px;; padding: 10px;">
                            <span style="color: #FFFFFF; font-size: 14px;">Test y</span>
                        </a>
                    </div>
                </div>
                <div style="display: flex; width: 100%; max-width: 400px; margin-top: 10px;">
                    <div style="flex: 1; background: {color_train}; border-radius: 0 0 30px 30px; box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.1), 0px 1px 3px rgba(0, 0, 0, 0.1);">
                        <a href='#' id='close_btn' style="display: flex; justify-content: center; align-items: center; height: 100%; text-decoration: none; background: ; border-radius: 50%; padding: 10px; margin-bottom: -20px;">
                            <span style="color: #FFFFFF; font-size: 14px;">&times;</span>
                        </a>
                    </div>
                </div>
            </div>
        """.format(color_train=color_train, color_test=color_test)

        # Call the click_detector function with the HTML content
        clicked = click_detector(button_container)

    if clicked == "train":
        my_text_paragraph('Training Dataset', my_font_size='28px')
        vertical_spacer(1)
        st.dataframe(st.session_state['df'].iloc[:-st.session_state['insample_forecast_steps'], :],
                     use_container_width=True)
    elif clicked == "test":
        my_text_paragraph('Test Dataset', my_font_size='28px')
        vertical_spacer(1)
        # get the length of test-set based on insample_forecast_steps
        test_range = len(st.session_state['df']) - st.session_state['insample_forecast_steps']
        st.dataframe(st.session_state['df'].iloc[test_range:, :], use_container_width=True)
    elif clicked == "X_train":
        my_text_paragraph('Explanatory Variables (X)', my_font_size='28px')
        vertical_spacer(1)
        st.dataframe(X_train, use_container_width=True)
    elif clicked == "y_train":
        my_text_paragraph('Target Variable (y)', my_font_size='28px')
        vertical_spacer(1)
        st.dataframe(y_train, use_container_width=True)
    elif clicked == "X_test":
        my_text_paragraph('Explanatory Variables (X)', my_font_size='28px')
        vertical_spacer(1)
        st.dataframe(X_test, use_container_width=True)
    elif clicked == "y_test":
        my_text_paragraph('Target Variable (y)', my_font_size='28px')
        vertical_spacer(1)
        st.dataframe(y_test, use_container_width=True)
    elif clicked == "close_btn":
        pass
    else:
        pass

def stock_ticker(text, speed=15):
    """
    Displays a right-to-left scrolling text e.g. ticker animation in a Markdown format using the `st.markdown` function from the Streamlit library.

    Parameters:
        text (str): The text to be displayed in the ticker.
        speed (int, optional): The speed of the ticker animation in seconds. Default is 15.

    Returns:
        None

    Example:
        stock_ticker("Stock ABC", speed=10)
    """
    st.markdown(
        f"""
        <style>
        .ticker-outer-container {{
            width: 100%;
            overflow: hidden;
        }}

        .ticker-container {{
            display: flex;
            white-space: nowrap;
            animation: ticker-animation {speed}s linear infinite;
        }}

        .ticker-item {{
            display: inline-block;
        }}

        @keyframes ticker-animation {{
            0% {{
                transform: translateX(99%); /*text is positioned to the right and initially not visible. At 100%*/
            }}
            100% {{
                transform: translateX(-400%); /*which means the text is moved to the left by 200% of its container's width*/
            }}
        }}
        </style>
        <div class="ticker-outer-container">
            <div class="ticker-container">
                <div class="ticker-item">{text} |</div>
                <div class="ticker-item">{text} |</div>
                <div class="ticker-item">{text} |</div>
                <div class="ticker-item">{text} |</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
#
# def eda_quick_insights(df, my_string_column, my_chart_color):
#     col1, col2, col3 = st.columns([20,40,20])
#     with col2:
#         my_text_header('Quick Insights')
#         vertical_spacer(1)
#
#     #col1, col2, col3 = st.columns([20, 80, 20])
#     col1, col2, col3 = st.columns([5, 80, 5])
#     with col2:
#         # Filter out NaN and '-' values from 'Label' column
#         label_values = df[my_string_column].dropna().apply(lambda x: x.strip()).replace('-', '').tolist()
#         # Filter out any remaining '-' values from 'Label' column
#         label_values = [value for value in label_values if value != '']
#         # Create an HTML unordered list with each non-NaN and non-'-' value as a list item
#         html_list = "<div class='my-list'>"
#         for i, value in enumerate(label_values):
#             html_list += f"<li><span class='my-number'>{i + 1}</span>{value}</li>"
#         html_list += "</div>"
#         # Display the HTML list using Streamlit
#         st.markdown(
#             f"""
#             <style>
#                 .my-list {{
#                     font-size: 16px;
#                     color: black; /* Add your desired font color here */
#                     line-height: 1.4;
#                     margin-bottom: 10px;
#                     margin-left: 0px;
#                     margin-right: 0px;
#                     background-color: white;
#                     border-radius: 10px;
#                     box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.3);
#                     padding: 20px;
#
#                 }}
#                 .my-list li {{
#                     margin: 10px 10px 10px 10px;
#                     padding-left: 30px;
#                     position: relative;
#                 }}
#                 .my-number {{
#                     font-weight: bold;
#                     color: white;
#                     background-color: {my_chart_color};
#                     border-radius: 50%;
#                     text-align: center;
#                     width: 20px;
#                     height: 20px;
#                     line-height: 20px;
#                     display: inline-block;
#                     position: absolute;
#                     left: 0;
#                     top: 0;
#                 }}
#             </style>
#             {html_list}
#             """,
#             unsafe_allow_html=True
#         )
#         # vertical spacer
#         vertical_spacer(1)
# =============================================================================

def create_flipcard_quick_insights(num_cards, header_list, paragraph_list_front, paragraph_list_back, font_family,
                                   font_size_front, font_size_back, image_path_front_card=None, df=None,
                                   my_string_column='Label', **kwargs):
    col1, col2, col3 = st.columns([20, 40, 20])
    with col2:
        my_text_header('Quick Insights')
        vertical_spacer(1)

    # Filter out NaN and '-' values from 'Label' column
    label_values = df[my_string_column].dropna().apply(lambda x: x.strip()).replace('-', '').tolist()
    # Filter out any remaining '-' values from 'Label' column
    label_values = [value for value in label_values if value != '']
    # Create an HTML unordered list with each non-NaN and non-'-' value as a list item
    html_list = "<div class='my-list'>"
    for i, value in enumerate(label_values):
        html_list += f"<li><span class='my-number'>{i + 1}</span>{value}</li>"
    html_list += "</div>"

    # open the image for the front of the card
    with open(image_path_front_card, 'rb') as file:
        contents = file.read()
        data_url = base64.b64encode(contents).decode("utf-8")

    # create empty list that will keep the html code needed for each card with header+text
    card_html = []

    # iterate over cards specified by user and join the headers and text of the lists
    for i in range(num_cards):
        card_html.append(f"""<div class="flashcard">
                                <div class='front'>
                                    <img src="data:image/png;base64,{data_url}"style="width: 100%; height: 100%; object-fit: cover; border-radius: 10px;">
                                    <h1 style='text-align:center;color:white; margin-bottom: 10px;padding: 35px;'>{header_list[i]}</h1>
                                    <p style='text-align:center; font-family: {font_family}; font-size: {font_size_front};'>{paragraph_list_front[i]}</p>
                                </div>
                                <div class="back">
                                    <h2>{header_list[i]}</h2>
                                    {html_list}
                                    <p style='text-align:center; font-family: {font_family}; font-size: {font_size_back};'>{paragraph_list_back[i]}</p>
                                </div>
                            </div>
                            """)

    # join all the html code for each card and join it into single html code with carousel wrapper
    carousel_html = "<div class='carousel'>" + "".join(card_html) + "</div>"

    # Display the carousel in streamlit
    st.markdown(carousel_html, unsafe_allow_html=True)

    # Create the CSS styling for the carousel
    st.markdown(
        f"""
        <style>
        /* back of card styling */
        .my-list {{
            font-size: 16px;
            color: black; /* Add your desired font color here */
            line-height: 1.4;
            margin-bottom: 10px;
            margin-left: 0px;
            margin-right: 0px;
            margin-bottom: 0px;
            padding: 0px;
            text-align: left;
        }}
        .my-list li {{
            margin: 10px 10px 10px 10px;
            padding-left: 50px;
            position: relative;
        }}
        .my-number {{
            font-weight: lighter;
            color: white;
            background-color: #48555e;
            border-radius: 100%;
            text-align: center;
            width: 25px;
            height: 25px;
            line-height: 20px;
            display: inline-block;
            position: absolute;
            left: 0;
            top: 0;
        }}

        /* Carousel Styling */
        .carousel {{
          grid-gap: 10px;
          overflow-x: auto;
          scroll-snap-type: x mandatory;
          scroll-behavior: smooth;
          -webkit-overflow-scrolling: touch;
          width: 100%;
          margin: auto;
        }}
       .flashcard {{
          display: inline-block; /* Display cards inline */
          width: 600px;
          height: 600px;
          background-color: white;
          border-radius: 10px;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
          perspective: 100px;
          margin-bottom: 0px; /* Add space between cards */
          padding: 0px;
          scroll-snap-align: center;
        }}
        .front, .back {{
          position: absolute;
          top: 0;
          left: 0;
          width: 600px;
          height: 600px;
          border-radius: 10px;
          backface-visibility: hidden;
          font-family: 'Ysabeau SC', sans-serif;
          text-align: center;
          overflow: hidden; /* Hide the scroll bar */
        }}
        .front {{
          background-color: #E9EBE1; /* Change the background color here */
          color: #333333;
          transform: rotateY(0deg);
        }}
        .back {{
          color: #333333;
          background-color: #E9EBE1; /* Change the background color here */
          transform: rotateY(180deg);
          display: flex;
          justify-content: center;
          align-items: center;
          flex-direction: column;
          border: 1px solid #48555e; /* Change the border color here */
        }}                                
        .flashcard:hover .front {{
          transform: rotateY(180deg);
        }}
        .flashcard:hover .back {{
          transform: rotateY(0deg);
        }}
        .front h1 {{
          padding-top: 10px;
          line-height: 1.5;
        }}
        .back h2 {{
          line-height: 2;
        }}
        .back p {{
          margin: 20px; /* Add margin for paragraph text */
        }}
        /* Carousel Navigation Styling */
        .carousel-nav {{
          margin: 10px 0px;
          text-align: center;
        }}
        </style>
        """, unsafe_allow_html=True)


def create_flipcard_quick_summary(header_list, paragraph_list_front, paragraph_list_back, font_family, font_size_front,
                                  font_size_back, image_path_front_card=None, **kwargs):
    # Compute and display the metrics for the first column
    rows = st.session_state.df_raw.shape[0]
    min_date = str(st.session_state.df_raw.iloc[:, 0].min().date())
    percent_missing = "{:.2%}".format(round((st.session_state.df_raw.iloc[:, 1].isna().mean()), 2))
    mean_val = np.round(st.session_state.df_raw.iloc[:, 1].mean(), 2)
    min_val = np.round(st.session_state.df_raw.iloc[:, 1].min(skipna=True), 2)
    std_val = np.round(np.nanstd(st.session_state.df_raw.iloc[:, 1], ddof=0), 2)

    # Compute and display the metrics for the second column
    cols = st.session_state.df_raw.shape[1]
    max_date = str(st.session_state.df_raw.iloc[:, 0].max().date())
    dataframe_freq, dataframe_freq_name = determine_df_frequency(st.session_state.df_raw, column_name='date')
    median_val = np.round(st.session_state.df_raw.iloc[:, 1].median(skipna=True), 2)
    max_val = np.round(st.session_state.df_raw.iloc[:, 1].max(skipna=True), 2)
    mode_val = st.session_state.df_raw.iloc[:, 1].dropna().mode().round(2).iloc[0]

    # open the image for the front of the card
    with open(image_path_front_card, 'rb') as file:
        contents = file.read()
        data_url = base64.b64encode(contents).decode("utf-8")

    # create empty list that will keep the html code needed for each card with header+text
    card_html = []

    header_color = 'white'

    card_html.append(f"""<div class="flashcard">
                            <div class='front_summary'>
                                <img src="data:image/png;base64,{data_url}"style="width: 100%; height: 100%; object-fit: cover; border-radius: 10px;">
                                <h1 style='text-align:center;color:#F5F5F5; margin-bottom: 10px;padding: 35px;'>{header_list}</h1>
                                <p style='text-align:center; font-family: Lato; font-size: {font_size_front};'>{paragraph_list_front} </p>
                            </div>
                            <div class="back_summary">
                                <h2>{header_list}</h2>
                                   <div style="display: flex; justify-content: space-between; margin-bottom: -20px; margin-left: 20px; margin-right: 20px; margin-top: -20px;">
                                   <div style="text-align: center; margin-right: 80px;">
                                   <div style="margin-bottom: 0px; "><b style="color: {header_color};">rows</b></div><div>{rows}</div><br/>
                                   <div style="margin-bottom: 0px;"><b style="color: {header_color};">start date</b></div><div>{min_date}</div><br/>
                                   <div style="margin-bottom: 0px;"><b style="color: {header_color};">missing</b></div><div>{percent_missing}</div><br/>
                                   <div style="margin-bottom: 0px;"><b style="color: {header_color};">mean</b></div><div>{mean_val}</div><br/>
                                   <div style="margin-bottom: 0px;"><b style="color: {header_color};">minimum</b></div><div>{min_val}</div><br/>
                                   <div style="margin-bottom: 0px;"><b style="color: {header_color};">stdev</b></div><div>{std_val}</div><br/>
                                   </div>
                                   <div style="text-align: center;">
                                   <div style="margin-bottom: 0px; "><b style="color: {header_color};">columns</b></div><div>{cols}</div><br/>
                                   <div style="margin-bottom: 0px; "><b style="color: {header_color};">end date</b></div><div>{max_date}</div><br/>
                                   <div style="margin-bottom: 0px; "><b style="color: {header_color};">frequency</b></div><div>{dataframe_freq_name}</div><br/>
                                   <div style="margin-bottom: 0px; "><b style="color: {header_color};">median</b></div><div>{median_val}</div><br/>
                                   <div style="margin-bottom: 0px; "><b style="color: {header_color};">maximum</b></div><div>{max_val}</div><br/>
                                   <div style="margin-bottom: 0px; "><b style="color: {header_color};">mode</b></div><div>{mode_val}</div><br/>
                                <p style='text-align:center; font-family: Lato; font-size: {font_size_back};'>{paragraph_list_back}</p>
                            </div>
                        </div>
                        """)

    # join all the html code for each card and join it into single html code with carousel wrapper
    carousel_html = "<div class='carousel'>" + "".join(card_html) + "</div>"

    # Display the carousel in streamlit
    st.markdown(carousel_html, unsafe_allow_html=True)
    # Create the CSS styling for the carousel
    st.markdown(
        f"""
        <style>   

        /* Carousel Styling */
        .carousel {{
          display: flex;
          justify-content: center;
          overflow-x: auto;
          scroll-snap-type: x mandatory;
          scroll-behavior: smooth;
          -webkit-overflow-scrolling: touch;
          width: 100%;
        }}
       .flashcard {{
          width: 600px;
          height: 600px;
          background-color: white;
          border-radius: 10px;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
          perspective: 100px;
          margin-bottom: 10px;
          scroll-snap-align: center;
        }}
        .front_summary, .back_summary {{
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          border-radius: 10px;
          backface-visibility: hidden;
          font-family: 'Ysabeau SC', sans-serif;
          font-size: {font_size_back};
          text-align: center;
        }}
        .front_summary {{
            background: linear-gradient(to bottom, #383e56, #383e56, #383e56, #383e56, #383e56, #383e56);
            color: #F5F5F5;
            transform: rotateY(0deg);
        }}
        .back_summary {{
            border: 1px solid #48555e; /* Change the border color here */
            background-color: #4d5466 ; /* Change the background color here */
            color: white;
            transform: rotateY(180deg);
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            box-shadow: none; /* Remove the shadow */
            font-family: 'Ysabeau SC', sans-serif;
            font-size: 18px;
        }}                       
        .flashcard:hover .front_summary {{
          transform: rotateY(180deg);
        }}
        .flashcard:hover .back_summary {{
          transform: rotateY(0deg);
        }}
        .front_summary h1 {{
          padding-top: 10px;
          line-height: 1.5;
        }}
        .back_summary h2 {{
          line-height: 2;
        }}
        .back_summary p {{
          margin: 10px; /* Add margin for paragraph text */
        }}
        /* Carousel Navigation Styling */
        .carousel-nav {{
          margin: 10px 0px;
          text-align: center;
        }}
        </style>
        """, unsafe_allow_html=True)


def create_flipcard_model_input(image_path_front_card=None, font_size_back='10px', my_string_back_card='', **kwargs):
    # Open the image for the front of the card
    with open(image_path_front_card, 'rb') as file:
        contents = file.read()
        data_url = base64.b64encode(contents).decode("utf-8")

    # Create empty list that will keep the HTML code needed for each card with header+text
    card_html = []

    # Append HTML code to list
    card_html.append(f"""
        <div class="flashcard_model_input">
            <div class='front_model_input'>
                <img src="data:image/png;base64,{data_url}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 10px;">
            </div>
            <div class="back_model_input" style="margin-top: 20px; display: flex; align-items: center;">
                <!-- Add your text for the back of the card here -->
                <h3> {my_string_back_card} </h3>
            </div>
        </div>
    """)

    # Create the CSS styling for the carousel
    st.markdown(
        f"""
        <style>
        .flipcard_stats {{
          display: flex;
          justify-content: center;
          overflow-x: auto;
          scroll-snap-type: x mandatory;
          scroll-behavior: smooth;
          -webkit-overflow-scrolling: touch;
          width: 100%;
        }}
        .flashcard_model_input {{
          width: 600px;
          height: 600px;
          background-color: white;
          border-radius: 10px;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
          perspective: 100px;
          margin-bottom: 10px;
          scroll-snap-align: center;
        }}
        .front_model_input, .back_model_input {{
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          border-radius: 10px;
          backface-visibility: hidden;
          font-family: 'Ysabeau SC', sans-serif;
          font-size: {font_size_back};
          text-align: center;
        }}
        .front_model_input {{
          /* background: linear-gradient(to bottom left, #4e3fce, #7a5dc7, #9b7cc2, #bb9bbd, #c2b1c4); */ 
          color: white;
          transform: rotateY(0deg);
        }}
        .back_model_input {{
            color: #333333;
            background: white;
            transform: rotateY(180deg);
            display: flex;
            justify-content: flex-start;
            border: none;
            margin: 0;
            padding: 60px;
            text-align: left;
            text-justify: inter-word;
            overflow: auto;
            position: relative;
            /* border-radius: 10px; */
            /* border: 10px solid black; */
        }}
        .back_model_input h6 {{
          margin-bottom: 0px;
          margin-top: 0px;
        }}
        .flashcard_model_input:hover .front_model_input {{
          transform: rotateY(180deg);
        }}
        .flashcard_model_input:hover .back_model_input {{
          transform: rotateY(0deg);
        }}
        .back_model_input p {{
          margin: 10px 0;
        }}
        footer {{
          text-align: center;
          margin-top: 20px;
          font-size: 12px;
          margin-bottom: 20px;
        }}
        </style>
        """, unsafe_allow_html=True)

    # Join the card HTML code list and display the carousel in Streamlit
    st.markdown("".join(card_html), unsafe_allow_html=True)

#################################
# FORMATTING DATAFRAMES FUNCTIONS
#################################
def train_models_carousel(my_title=''):
    # gradient title
    vertical_spacer(2)

    title = my_title

    # =============================================================================
    #     # set gradient color of letters of title
    #     gradient = '-webkit-linear-gradient(left, #0072B2, #673ab7, #3f51b5, #2196f3, #03a9f4)'
    # =============================================================================
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        # show in streamlit the title with gradient
        st.markdown(f'<h2 style="text-align:center; font-family: Rock Salt; color: black;"> {title} </h2>',
                    unsafe_allow_html=True)

    vertical_spacer(2)

    # show carousel of models
    paragraph_list_back = [
        'The <b> Naive models </b> serve as a simple baseline or benchmark. Model I assumes the next observation is equal to a lagged previous value which can be either previous day, week, month, quarter, year or custom defined. Model II aggregates a set of past values using a rolling window to predict the next value. Model III has a fixed single value by taking the average, median or mode.',
        'The <b> Linear Regression Model </b> is a statistical technique used to analyze the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship, aiming to find the best-fit line that minimizes the differences between observed and predicted values.',
        '<b>SARIMAX</b>, short for <b>Seasonal Autoregressive Integrated Moving Average with Exogenous Variables</b>, is a powerful time series forecasting model that incorporates seasonal patterns and exogenous variables. It combines <i> autoregressive </i> (past values), <i> moving average </i> (averages of certain time spans), and <i> integrated </i> (calculating differences of subsequent values) components.',
        '<b>Prophet</b> utilizes an additive model (sum of individual factors) that decomposes time series data into: <i>trend</i>, <i>seasonality</i>, and <i>holiday components</i>. It incorporates advanced statistical techniques and automatic detection of changepoints to handle irregularities in the data. It offers flexibility in handling missing data and outliers making it a powerful forecasting model.']

    # create carousel cards for each model
    header_list = ['Naive Models', 'Linear Regression', 'SARIMAX', 'Prophet']
    paragraph_list_front = ['', '', '', '']

    # define the font family to display the text of paragraph
    font_family = 'Ysabeau SC'
    # font_family = 'Rubik Dirt'

    # define the paragraph text size
    font_size_front = '14px'
    font_size_back = '15px'

    # apply carousel function to show 'flashcards'
    create_carousel_cards_v2(4, header_list, paragraph_list_front, paragraph_list_back, font_family, font_size_front,
                             font_size_back)
    vertical_spacer(2)


def create_carousel_cards(num_cards, header_list, paragraph_list, font_family, font_size):
    # note: #purple gradient: background: linear-gradient(to bottom right, #F08A5D, #FABA63, #2E9CCA, #4FB99F, #dababd);
    # create empty list that will keep the html code needed for each card with header+text
    card_html = []
    # iterate over cards specified by user and join the headers and text of the lists
    for i in range(num_cards):
        card_html.append(
            f"<div class='card'><h1 style='text-align:center;color:white; margin-bottom: 10px;'>{header_list[i]}</h1><p style='text-align:center; font-family: {font_family}; font-size: {font_size};'>{paragraph_list[i]}</p></div>")
    # join all the html code for each card and join it into single html code with carousel wrapper
    carousel_html = "<div class='carousel'>" + "".join(card_html) + "</div>"
    # Display the carousel in streamlit
    st.markdown(carousel_html, unsafe_allow_html=True)
    # Create the CSS styling for the carousel
    st.markdown(
        """
        <style>
        /* Carousel Styling */
        .carousel {
          display: flex;
          overflow-x: auto;
          scroll-snap-type: x mandatory;
          scroll-behavior: smooth;
          -webkit-overflow-scrolling: touch;
          width: 80%;
          margin: auto;
        }
        .card {
          display: inline-block;
          margin-right: 10px;
          vertical-align: top;
          scroll-snap-align: center;
          flex: 0 0 auto;
          width: 75%;
          height: 200px;
          margin: 30px;
          padding: 20px;
          background: linear-gradient(to bottom left, #F08A5D, #FABA63);
                                      #  '-webkit-linear-gradient(left, #F08A5D, #FABA63, #2E9CCA, #4FB99F)'
          box-shadow: 10px 10px 10px 0px rgba(0, 0, 0, 0.1);
          border-radius: 20px;
          text-align: center;
          font-family: """ + font_family + """, sans-serif;
          font-size: """ + str(font_size) + """px;
          color: white;
          transition: transform 0.2s ease-in-out;
        }
        .card:hover {
          transform: scale(1.1);
          box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.5);
        }
        /* Carousel Navigation Styling */
        .carousel-nav {
          margin: 10px 0px;
          text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)


# this is the independent cards with purple gradient version
def create_carousel_cards_v3(num_cards, header_list, paragraph_list_front, paragraph_list_back, font_family,
                             font_size_front, font_size_back):
    # note removing display: flex; inside the css code for .flashcard -> puts cards below eachother
    # create empty list that will keep the html code needed for each card with header+text
    card_html = []
    # iterate over cards specified by user and join the headers and text of the lists
    for i in range(num_cards):
        card_html.append(f"""<div class="flashcard">                     
                                <div class='front'>
                                    <h1 style='text-align:center;color:white; margin-bottom: 10px;padding: 35px;'>{header_list[i]}</h1>
                                    <p style='text-align:center; font-family: {font_family}; font-size: {font_size_front};'>{paragraph_list_front[i]}</p>
                                </div>
                                <div class="back">
                                    <p style=font-family: {font_family}; font-size: {font_size_back};'>{paragraph_list_back[i]}</p>
                                </div>
                            </div>
                            """)
    # join all the html code for each card and join it into single html code with carousel wrapper
    carousel_html = "<div class='carousel'>" + "".join(card_html) + "</div>"
    # Display the carousel in streamlit
    st.markdown(carousel_html, unsafe_allow_html=True)
    # Create the CSS styling for the carousel
    st.markdown(
        f"""
        <style>
        /* Carousel Styling */
        .carousel {{
          grid-gap: 10px;
          overflow-x: auto;
          scroll-snap-type: x mandatory;
          scroll-behavior: smooth;
          -webkit-overflow-scrolling: touch;
          width: 100%;
          margin: auto;
          border: 1px solid #000000; 
          border-radius: 10px;
        }}
       .flashcard {{
          display: inline-block; /* Display cards inline */
          width: 400px;
          height: 200px;
          background-color: white;
          border-radius: 10px;
          perspective: 100px;
          margin-bottom: 5px; /* Add space between cards */
          padding: 0px;
          scroll-snap-align: center;
        }}
        .front, .back {{
          position: absolute;
          top: 0;
          left: 0;
          width: 400px;
          height: 200px;
          border-radius: 10px;
          backface-visibility: hidden;
          font-family: {font_family};
          text-align: justify;
        }}
        .front {{
          background: linear-gradient(to bottom, #F5F5F5, #F5F5F5);
          color: black; /* text color */
          transform: rotateY(0deg);
        }}
        .back {{   
          background-color: #F5F5F5; /* Set the background color to off-white */
          color: black;
        transform: rotateY(180deg);
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        /* border: 1px solid #000000; */ /* Add 1px border with black color */ 
        }}                                
        .flashcard:hover .front {{
          transform: rotateY(180deg);
        }}
        .flashcard:hover .back {{
          transform: rotateY(0deg);
        }}
        .front h2, .back h2 {{
          color: white;
          text-align: center;
          margin-top: 10%;
          transform: translateY(-10%);
          font-family: {font_family};
          font-size: {font_size_front}px;
        }}
        .front h2 {{
          padding-top: 10px;
          line-height: 1.5;
        }}
        .back h2 {{
          line-height: 2;
        }}
        .back p {{
          margin: 20px; /* Add margin for paragraph text */
          }}
        /* Carousel Navigation Styling */
        .carousel-nav {{
          margin: 10px 0px;
          text-align: center;
        }}
        </style>
        """, unsafe_allow_html=True)


# v2 with front and back of a flashcard as well with text on both sides and white background and gradient letters
def create_carousel_cards_v2(num_cards, header_list, paragraph_list_front, paragraph_list_back, font_family,
                             font_size_front, font_size_back):
    # note removing display: flex; inside the css code for .flashcard -> puts cards below eachother
    # create empty list that will keep the html code needed for each card with header+text
    card_html = []
    # iterate over cards specified by user and join the headers and text of the lists
    for i in range(num_cards):
        card_html.append(f"""<div class="flashcard">                     
                                <div class='front'>
                                    <h1 style='text-align: center; color: #62e1d3; margin-top: 10px; margin-bottom: -10px; padding: 35px;'>{header_list[i]}</h1>
                                    <p style='text-align:center; font-family: {font_family}; font-size: {font_size_front};'>{paragraph_list_front[i]}</p>
                                </div>
                                <div class="back">
                                    <p style='text-align:justify;  word-spacing: 1px; font-family: {font_family}; font-size: {font_size_back};'>{paragraph_list_back[i]}</p>
                                </div>
                            </div>
                            """)
    # join all the html code for each card and join it into single html code with carousel wrapper
    carousel_html = "<div class='carousel'>" + "".join(card_html) + "</div>"
    # Display the carousel in streamlit
    st.markdown(carousel_html, unsafe_allow_html=True)
    # Create the CSS styling for the carousel
    st.markdown(
        f"""
        <style>
        /* Carousel Styling */
        .carousel {{
          grid-gap: 10px;
          justify-content: center; /* Center horizontally */
          align-items: center; /* Center vertically */
          overflow-x: auto;
          scroll-snap-type: x mandatory;
          scroll-behavior: smooth;
          -webkit-overflow-scrolling: touch;
          width: 400px;
          margin: 0 0; /* Center horizontally by setting left and right margins to auto */
          background-color: white; /* Add black background */
          padding: 0px; /* Add padding for the black background */
          border-radius: 10px; /* Add border-radius for rounded edges */
          box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2), 0px -4px 8px rgba(0, 0, 0, 0.1); /* Add box shadow */
        }}
       .flashcard {{
          display: inline-block; /* Display cards inline */
          width: 400px;
          height: 200px;
          background-color: white;
          border-radius: 5px;
          perspective: 100px;
          margin-bottom: 0px; /* Add space between cards */
          padding: 0px;
          scroll-snap-align: center;
        }}
        .front, .back {{
          position: absolute;
          top: 0;
          left: 0;
          width: 400px;
          height: 200px;
          border-radius: 8px;
          backface-visibility: hidden;
          font-family: {font_family};
          text-align: center;
        }}
        .front {{
          background: white;
          color: black; /* Make the text transparent */
          transform: rotateY(0deg);
          border: 4px;
          border-color: white;
        }}
        .back {{
            /* ... other styles ... */
            background: black;
            color: transparent; /* Make the text transparent */
            background-clip: text; /* Apply the background gradient to the text */
            -webkit-background-clip: text; /* For Safari */
            -webkit-text-fill-color: transparent; /* For Safari */
            background-image: linear-gradient(to bottom left, #000000); /* linear-gradient(to bottom left, #941c8e, #763a9a, #4e62a3, #2e81ad, #12a9b4); */ /* Set the background gradient */
            transform: rotateY(180deg);
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }}                               
        .flashcard:hover .front {{
          transform: rotateY(180deg);
        }}
        .flashcard:hover .back {{
          transform: rotateY(0deg);
          cursor: default; /* Change cursor to pointer on hover */
        }}
        .front h2, .back h2 {{
          color: black;
          text-align: center;
          margin-top: 10%;
          transform: translateY(-10%);
          font-family: {font_family};
          font-size: {font_size_front}px;
        }}
        .front h2 {{
          padding-top: 10px;
          line-height: 1.5;
        }}
        .back h2 {{
          line-height: 2;
        }}
        .back p {{
          margin: 24px; /* Add margin for paragraph text */
          }}
        /* Carousel Navigation Styling */
        .carousel-nav {{
          margin: 10px 0px;
          text-align: center;
        }}
        </style>
        """, unsafe_allow_html=True)


# ******************************************************************************
# STATISTICAL TEST FUNCTIONS
# ******************************************************************************
################################################
# SUMMARY STATISTICS DESCRIPTIVE LABEL FUNCTIONS
################################################
def eda_quick_summary(my_chart_color):
    """
    Displays a quick summary of the exploratory data analysis (EDA) metrics.

    This function calculates and displays various metrics based on the provided DataFrame `df_raw`. The metrics include:
    - Number of rows
    - Minimum date
    - Percentage of missing data
    - Mean of the second column
    - Number of columns
    - Maximum date
    - Frequency of time-series data
    - Median of the first column
    - Minimum of the fourth column

    The metrics are displayed in a visually appealing format using Streamlit columns and CSS styling.

    Parameters:
    None

    Returns:
    None
    """
    # Define CSS style for the metrics container
    container_style = '''
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0px 0px 15px -5px rgba(0,0,0,0.3);
        padding: 25px;
        margin: 10px;
        width: 100%;
        max-width: 700px;
        display: flex; /* add this line to enable flexbox */
        justify-content: center; /* add this line to horizontally center-align the metrics */
        align-items: center; /* add this line to vertically center-align the metrics */
        text-align: center; /* add this line to center-align the metrics */
        font-family: Arial;
        font-size: 16px;
        color: #555555;
    '''
    try:
        col1, col2, col3 = st.columns([19, 80, 20])
        with col2:
            # Compute and display the metrics for the first column
            rows = st.session_state.df_raw.shape[0]
            min_date = str(st.session_state.df_raw.iloc[:, 0].min().date())
            percent_missing = "{:.2%}".format(round((st.session_state.df_raw.iloc[:, 1].isna().mean()), 2))
            mean_val = np.round(st.session_state.df_raw.iloc[:, 1].mean(), 2)
            min_val = np.round(st.session_state.df_raw.iloc[:, 1].min(skipna=True), 2)
            std_val = np.round(np.nanstd(st.session_state.df_raw.iloc[:, 1], ddof=0), 2)

            # Compute and display the metrics for the second column
            cols = st.session_state.df_raw.shape[1]
            max_date = str(st.session_state.df_raw.iloc[:, 0].max().date())
            dataframe_freq, dataframe_freq_name = determine_df_frequency(st.session_state.df_raw, column_name='date')
            median_val = np.round(st.session_state.df_raw.iloc[:, 1].median(skipna=True), 2)
            max_val = np.round(st.session_state.df_raw.iloc[:, 1].max(skipna=True), 2)
            mode_val = st.session_state.df_raw.iloc[:, 1].dropna().mode().round(2).iloc[0]
            st.write(
                f'<div style="{container_style}">'
                f'<div style="display: flex; justify-content: space-between; margin-bottom: -20px;margin-left: 90px; margin-right: 0px; margin-top: -10px;">'
                f'<div style="text-align: center; margin-right: 50px;">'
                f'<div><b style="color: {my_chart_color};">Rows</b></div><div>{rows}</div><br/>'
                f'<div><b style="color: {my_chart_color};">Start Date</b></div><div>{min_date}</div><br/>'
                f'<div><b style="color: {my_chart_color};">Missing</b></div><div>{percent_missing}</div><br/>'
                f'<div><b style="color: {my_chart_color};">Mean</b></div><div>{mean_val}</div><br/>'
                f'<div><b style="color: {my_chart_color};">Minimum</b></div><div>{min_val}</div><br/>'
                f'<div><b style="color: {my_chart_color};">StDev</b></div><div>{std_val}</div><br/>'
                f'</div>'
                f'<div style="text-align: center;">'
                f'<div><b style="color: {my_chart_color};">Columns</b></div><div>{cols}</div><br/>'
                f'<div><b style="color: {my_chart_color};">End Date</b></div><div>{max_date}</div><br/>'
                f'<div><b style="color: {my_chart_color};">Frequency</b></div><div>{dataframe_freq_name}</div><br/>'
                f'<div><b style="color: {my_chart_color};">Median</b></div><div>{median_val}</div><br/>'
                f'<div><b style="color: {my_chart_color};">Maximum</b></div><div>{max_val}</div><br/>'
                f'<div><b style="color: {my_chart_color};">Mode</b></div><div>{mode_val}</div><br/>'
                f'</div>'
                f'</div>'
                f'</div>'
                ,
                unsafe_allow_html=True)
            # vertical spacer
            vertical_spacer(1)
    except:
        st.write('Error: could not show the quick summary stats...please contact admin')


def create_summary_df(data):
    """
    Create a DataFrame with summary statistics for the input data.

    Parameters
    ----------
    data : pandas DataFrame or pandas Series
        The data for which summary statistics are to be computed.

    Returns
    -------
    summary_df : pandas DataFrame
        A DataFrame with summary statistics for the input data.
    """
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    if not isinstance(data, pd.DataFrame):
        raise TypeError('Input data must be a pandas DataFrame or Series')

    ####################################
    # Compute summary statistics
    ####################################
    length = data.shape[0]
    length_label = dataset_size_function(data)
    percent_missing = (data.isna().mean())[0]

    ####################################
    # drop rows with missing values
    ####################################
    num_missing = data.isnull().sum()[0]
    missing_label_value, missing_label_perc = missing_value_function(data)
    if num_missing > 0:
        data.dropna(inplace=True)
    # =============================================================================
    #         st.warning(f'''**Warning** ⚠️:
    #                  **{num_missing}** NaN values were excluded to be able to calculate metrics such as: *skewness*, *kurtosis* and *White-Noise* test.''')
    # =============================================================================
    else:
        pass

    ########################
    # Continued Code
    # Summary Statistics
    ########################
    mean = data.mean()
    mean_label, mean_position = mean_label_function(data)
    median = data.median()
    median_label, median_position = median_label_function(data)
    mode_val = mode(data)[0][0]
    std_dev = data.std()
    std_label = std_label_function(data)
    variance = data.var()
    var_label = var_label_function(data)
    num_distinct = data.nunique()
    distinct_label = distinct_values_label_function(data)
    kurt = kurtosis(data)[0]
    kurtosis_label = kurtosis_label_function(data)
    skewness = skew(data)[0]

    # Define skewness labels
    if skewness < -1.0:
        skewness_type = "Highly negatively skewed"
    elif -1.0 <= skewness < -0.5:
        skewness_type = "Moderately negatively skewed"
    elif -0.5 <= skewness <= 0.5:
        skewness_type = "Approximately symmetric"
    elif 0.5 < skewness <= 1.0:
        skewness_type = "Moderately positively skewed"
    else:
        skewness_type = "Highly positively skewed"

        # =============================================================================

    # White Noise - Ljung Box Test
    # =============================================================================

    def white_noise_label(white_noise, lag_value):
        """
        Description
        ----------
        Get descriptive label for white noise test the quick insights dataframe

        Parameters
        ----------
        white_noise: boolean
            True or False
        lag_value: int
        """
        # initiate empty string container
        white_noise_label = ''
        if white_noise == 'True':
            white_noise_label = f'residuals independently distributed for lag={lag_value}'
        elif white_noise == 'False':
            white_noise_label = f'residuals not independently distributed for lag={lag_value}'
        else:
            white_noise_label = '-'
        return white_noise_label

    try:
        if len(data) >= 24:
            # define the model
            model_lag24 = sm.tsa.AutoReg(data, lags=24, trend='c', old_names=False)
            # train model on the residuals
            res_lag24 = model_lag24.fit()

            # define the model
            model_lag48 = sm.tsa.AutoReg(data, lags=48, trend='c', old_names=False)
            # train model on the residuals
            res_lag48 = model_lag48.fit()

            # Perform Ljung-Box test on residuals with lag=24 and lag=48
            result_ljungbox24 = sm.stats.acorr_ljungbox(res_lag24.resid, lags=[24], return_df=True)
            result_ljungbox48 = sm.stats.acorr_ljungbox(res_lag48.resid, lags=[48], return_df=True)

            test_statistic_ljungbox_24 = result_ljungbox24.iloc[0]['lb_stat']
            test_statistic_ljungbox_48 = result_ljungbox48.iloc[0]['lb_stat']

            p_value_ljungbox_24 = result_ljungbox24.iloc[0]['lb_pvalue']
            p_value_ljungbox_48 = result_ljungbox48.iloc[0]['lb_pvalue']

            white_noise_24 = "True" if p_value_ljungbox_24 <= 0.05 else "False"
            white_noise_48 = "True" if p_value_ljungbox_48 <= 0.05 else "False"

            white_noise_24_lbl = white_noise_label(white_noise_24, 24)
            white_noise_48_lbl = white_noise_label(white_noise_48, 48)
        else:
            # define lags
            lag1 = int((len(data) - 1) / 2)
            lag2 = int(len(data) - 1)

            st.write(lag1, lag2)  # test

            # define the model
            model_lag24 = sm.tsa.AutoReg(data, lags=lag1, trend='c', old_names=False)
            # train model on the residuals
            res_lag24 = model_lag24.fit()

            # define the model
            model_lag48 = sm.tsa.AutoReg(data, lags=lag2, trend='c', old_names=False)
            # train model on the residuals
            res_lag48 = model_lag48.fit()

            # Perform Ljung-Box test on residuals with lag=24 and lag=48
            result_ljungbox24 = sm.stats.acorr_ljungbox(res_lag24.resid, lags=[lag1], return_df=True)
            result_ljungbox48 = sm.stats.acorr_ljungbox(res_lag48.resid, lags=[lag2], return_df=True)

            test_statistic_ljungbox_24 = result_ljungbox24.iloc[0]['lb_stat']
            test_statistic_ljungbox_48 = result_ljungbox48.iloc[0]['lb_stat']

            p_value_ljungbox_24 = result_ljungbox24.iloc[0]['lb_pvalue']
            p_value_ljungbox_48 = result_ljungbox48.iloc[0]['lb_pvalue']

            white_noise_24 = "True" if p_value_ljungbox_24 <= 0.05 else "False"
            white_noise_48 = "True" if p_value_ljungbox_48 <= 0.05 else "False"

            white_noise_24_lbl = white_noise_label(white_noise_24, lag1)
            white_noise_48_lbl = white_noise_label(white_noise_48, lag2)
    except:
        result_ljungbox = np.nan
        test_statistic_ljungbox_24 = np.nan
        test_statistic_ljungbox_48 = np.nan
        p_value_ljungbox_24 = np.nan
        p_value_ljungbox_48 = np.nan
        white_noise_24 = np.nan
        white_noise_48 = np.nan
        white_noise_24_lbl = '-'
        white_noise_48_lbl = '-'
    # =============================================================================

    # ******************************
    # Augmented Dickey-Fuller Test
    # ******************************
    result = adfuller(data)
    stationarity = "Stationary" if result[0] < result[4]["5%"] else "Non-Stationary"
    p_value = result[1]
    test_statistic = result[0]
    critical_value_1 = result[4]["1%"]
    critical_value_5 = result[4]["5%"]
    critical_value_10 = result[4]["10%"]

    # ***********************
    # Perform Shapiro Test
    # ***********************
    shapiro_stat, shapiro_pval = shapiro(data)
    normality = "True" if shapiro_pval > 0.05 else "False"

    ###########################
    # Create summary DataFrame
    ###########################
    summary_df = pd.DataFrame(
        columns=['Test', 'Test Name', 'Property', 'Settings', str('Column: ' + data.columns[0]), 'Label'])
    summary_df.loc[0] = ['Summary', 'Statistics', 'Length', '-', length, length_label]
    summary_df.loc[1] = ['', '', '# Missing Values', '-', num_missing, missing_label_value]
    summary_df.loc[2] = ['', '', '% Missing Values', '-', f"{percent_missing:.2%}", missing_label_perc]
    summary_df.loc[3] = ['', '', 'Mean', '-', round(mean[0], 2), f'{mean_label} and {mean_position}']
    summary_df.loc[4] = ['', '', 'Median', '-', round(median[0], 2), f'{median_label} and {median_position}']
    summary_df.loc[5] = ['', '', 'Standard Deviation', '-', round(std_dev[0], 2), std_label]
    summary_df.loc[6] = ['', '', 'Variance', '-', round(variance[0], 2), var_label]
    summary_df.loc[7] = ['', '', 'Kurtosis', '-', round(kurt, 2), kurtosis_label]
    summary_df.loc[8] = ['', '', 'Skewness', '-', round(skewness, 2), skewness_type]
    summary_df.loc[9] = ['', '', '# Distinct Values', '-', num_distinct[0], distinct_label]
    summary_df.loc[10] = ['White Noise', 'Ljung-Box', 'Test Statistic', '24', round(test_statistic_ljungbox_24, 4), '-']
    summary_df.loc[11] = ['', '', 'Test Statistic', '48', round(test_statistic_ljungbox_48, 4), '-']
    summary_df.loc[12] = ['', '', 'p-value', '24', round(p_value_ljungbox_24, 4), white_noise_24_lbl]
    summary_df.loc[13] = ['', '', '', '48', round(p_value_ljungbox_48, 4), white_noise_48_lbl]
    summary_df.loc[14] = ['', '', 'White Noise', '24', white_noise_24, '-']
    summary_df.loc[15] = ['', '', '', '48', white_noise_48, '-']
    summary_df.loc[16] = ['Stationarity', 'ADF', 'Stationarity', '0.05', stationarity, '-']
    summary_df.loc[17] = ['', '', 'p-value', '0.05', round(p_value, 4), '-']
    summary_df.loc[18] = ['', '', 'Test-Statistic', '0.05', round(test_statistic, 4), '-']
    summary_df.loc[19] = ['', '', 'Critical Value 1%', '0.05', round(critical_value_1, 4), '-']
    summary_df.loc[20] = ['', '', 'Critical Value 5%', '0.05', round(critical_value_5, 4), '-']
    summary_df.loc[21] = ['', '', 'Critical Value 10%', '0.05', round(critical_value_10, 4), '-']
    summary_df.loc[22] = ['Normality', 'Shapiro', 'Normality', '0.05', normality, '-']
    summary_df.loc[23] = ['', '', 'p-value', '0.05', round(shapiro_pval, 4), '-']
    return summary_df


def dataset_size_function(data):
    """
    This function takes in a dataset and returns a label describing the size of the dataset.
    """
    num_rows = data.shape[0]
    if num_rows < 50:
        size_label = "Small dataset"
    elif num_rows < 1000:
        size_label = "Medium dataset"
    else:
        size_label = "Large dataset"
    return size_label


def missing_value_function(data):
    """
    This function takes in a dataset and returns a label based on the percentage of missing values in the dataset.
    """
    num_missing = data.isnull().sum()[0]
    num_total = data.size
    missing_ratio = num_missing / num_total
    if missing_ratio == 0:
        missing_label_value = "No missing values"
        missing_label_perc = "No percentage missing"
    elif missing_ratio < 0.01:
        missing_label_value = "Very few missing values"
        missing_label_perc = "Very Low percentage missing"
    elif missing_ratio < 0.05:
        missing_label_value = "Some missing values"
        missing_label_perc = "Low percentage missing"
    elif missing_ratio < 0.1:
        missing_label_value = "Many missing values"
        missing_label_perc = "High percentage missing"
    else:
        missing_label_value = "A large percentage of missing values"
        missing_label_perc = "Very high percentage missing"
    return missing_label_value, missing_label_perc


def std_label_function(data):
    """
    This function takes in a dataset, calculates the standard deviation, and returns a label based on the size of the standard deviation relative to the range of the dataset.
    """
    std = data.std()
    range_data = data.min() - data.max()
    std_ratio = (std / range_data)[0]
    if std_ratio <= 0.1:
        std_label = "Very small variability"
    elif std_ratio <= 0.3:
        std_label = "Small variability"
    elif std_ratio <= 0.5:
        std_label = "Moderate variability"
    elif std_ratio <= 0.7:
        std_label = "Large variability"
    else:
        std_label = "Very large variability"
    return std_label


def var_label_function(data):
    """
    This function takes in a dataset, calculates the variance, and returns a label based on the size of the variance relative to the range of the dataset.
    """
    var = data.var()
    range_data = data.min() - data.max()
    var_ratio = (var / range_data)[0]
    if var_ratio <= 0.1:
        var_label = "Very small variance"
    elif var_ratio <= 0.3:
        var_label = "Small variance"
    elif var_ratio <= 0.5:
        var_label = "Moderate variance"
    elif var_ratio <= 0.7:
        var_label = "Large variance"
    else:
        var_label = "Very large variance"
    return var_label


def mean_label_function(data):
    """
    This function takes in a dataset, calculates the mean, and returns a label based on the position of the mean relative to the range of the dataset.
    """
    median = data.median()[0]
    mean = data.mean()[0]
    range_data = data.max() - data.min()
    mean_ratio = ((mean - data.min()) / range_data)[0]
    if mean_ratio < 0.25:
        mean_label = "Mean skewed left"
    elif mean_ratio > 0.75:
        mean_label = "Mean skewed right"
    else:
        mean_label = "Mean balanced"
    if mean > median:
        mean_position = "Mean above median"
    elif mean < median:
        mean_position = "Mean below median"
    else:
        mean_position = "Mean equal to median"
    return mean_label, mean_position


def median_label_function(data):
    """
    This function takes in a dataset, calculates the median, and returns a label based on the position of the median relative to the range of the dataset.
    """
    median = data.median()[0]
    range_data = data.max() - data.min()
    median_ratio = ((median - data.min()) / range_data)[0]
    mean = data.mean()[0]
    mean_ratio = (mean - data.min()) / range_data
    if median_ratio < 0.25:
        median_label = "Median Skewed left"
    elif median_ratio > 0.75:
        median_label = "Median Skewed right"
    else:
        median_label = "Median balanced"
    if median > mean:
        median_position = "Median above mean"
    elif median < mean:
        median_position = "Median below mean"
    else:
        median_position = "Median equal to mean"
    return median_label, median_position


def kurtosis_label_function(data):
    """
    This function takes in a dataset, calculates the kurtosis, and returns a label based on the kurtosis value.
    """
    kurt = data.kurtosis()[0]
    if kurt < -2.0:
        kurtosis_label = 'Highly platykurtic (thin tails)'
    elif -2.0 <= kurt < -0.5:
        kurtosis_label = 'Moderately platykurtic (thin tails)'
    elif -0.5 <= kurt <= 0.5:
        kurtosis_label = 'Mesokurtic (medium tails)'
    elif 0.5 < kurt <= 2.0:
        kurtosis_label = 'Moderately leptokurtic (fat tails)'
    else:
        kurtosis_label = 'Highly leptokurtic (fat tails)'
    return kurtosis_label


def distinct_values_label_function(data):
    """
    This function takes in a dataset and returns a label based on the number of distinct values in the dataset.
    """
    num_distinct = len(data.drop_duplicates())
    num_total = len(data)
    distinct_ratio = num_distinct / num_total
    if distinct_ratio < 0.05:
        distinct_values_label = "Very low amount of distinct values"
    elif distinct_ratio < 0.1:
        distinct_values_label = "Low amount of distinct values"
    elif distinct_ratio < 0.5:
        distinct_values_label = "Moderate amount of distinct values"
    elif distinct_ratio < 0.9:
        distinct_values_label = "High amount of distinct values"
    else:
        distinct_values_label = "Very high amount of distinct values"
    return distinct_values_label





# ******************************************************************************
# GRAPH FUNCTIONS | PLOT FUNCTIONS
# ******************************************************************************




def chart_title(title, subtitle, font, font_size):
    """
    Creates a dictionary containing the properties for a chart title and subtitle.

    Parameters:
    -----------
    title : str
        The main title of the chart.
    subtitle : str
        The subtitle of the chart.
    font : str
        The name of the font to be used for the title and subtitle.
    font_size : int
        The font size to be used for the title and subtitle.

    Returns:
    --------
    dict
        A dictionary containing the properties for the chart title and subtitle,
        including the text, font size, font family, and anchor position.
    """
    return {
        "text": title,
        "subtitle": subtitle,
        "fontSize": font_size,
        "font": font,
        "anchor": "middle"
    }


def altair_correlation_chart(total_features, importance_scores, pairwise_features_in_total_features, corr_threshold):
    """
    Creates an Altair chart object for visualizing pairwise feature importance scores.

    Args:
        total_features (list): List of all features.
        importance_scores (dict): Dictionary mapping feature names to importance scores.
        pairwise_features_in_total_features (list): List of pairs of features.
        corr_threshold (float): Threshold for displaying correlation scores.

    Returns:
        altair.Chart: Altair chart object for visualizing pairwise feature importance scores.
    """
    # Set title font style and size
    title_font = "Helvetica"
    title_font_size = 12
    num_features = len(total_features)
    num_cols = min(3, num_features)
    num_rows = math.ceil(num_features / num_cols)
    charts = []
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
                        alt.value('#b8a5f9'),  # red
                        alt.value('#6137f1')  # green
                    )
                ).properties(width=100, height=100,
                             title=chart_title("Removing", feature2, title_font, title_font_size))
            elif score2 > score1:
                total_features.remove(feature1)
                chart = alt.Chart(data).mark_bar().encode(
                    x='Score:Q',
                    y=alt.Y('Feature:O', sort='-x'),
                    color=alt.condition(
                        alt.datum.Feature == feature1,
                        alt.value('#b8a5f9'),  # red #DC143C
                        alt.value('#6137f1')  # green #00FF7F
                    )
                ).properties(width=100, height=100,
                             title=chart_title("Removing", feature1, title_font, title_font_size))
            else:
                total_features.remove(feature1)
                chart = alt.Chart(data).mark_bar().encode(
                    x='Score:Q',
                    y=alt.Y('Feature:O', sort='-x'),
                    color=alt.condition(
                        alt.datum.Feature == feature1,
                        alt.value('#b8a5f9'),  # red
                        alt.value('#6137f1')  # green
                    )
                ).properties(width=100, height=100,
                             title=chart_title("Removing", feature1, title_font, title_font_size))
            charts.append(chart)
    # Combine all charts into a grid
    grid_charts = []
    for i in range(num_rows):
        row_charts = []
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < len(charts):
                row_charts.append(charts[idx])
        if row_charts:
            grid_charts.append(alt.hconcat(*row_charts))

    grid_chart = alt.vconcat(*grid_charts, spacing=10)

    # create a streamlit container with a title and caption
    my_text_paragraph("Removing Highly Correlated Features", my_font_size='26px')

    col1, col2, col3 = st.columns([5.5, 4, 5])
    with col2:
        st.caption(f'for pair-wise features >={corr_threshold * 100:.0f}%')

    # show altair chart with pairwise correlation importance scores and in red lowest and green highest
    st.altair_chart(grid_chart, use_container_width=True)


def display_dataframe_graph(df, key=0, my_chart_color='#217CD0'):
    """
    Displays a line chart of a Pandas DataFrame using Plotly Express.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to display.
    key : int or str, optional
        An optional identifier used to cache the output of the function when called with the same arguments,
        by default 0.

    Returns
    -------
    None
        The function displays the chart in the Streamlit app.

    Args:
        my_chart_color: set hex-color string e.g. '#000000'

    """
    fig = px.line(df,
                  x=df.index,
                  y=df.columns,
                  # labels=dict(x="Date", y="y"),
                  title='',
                  )
    # Set Plotly configuration options
    fig.update_layout(width=800, height=400, xaxis=dict(title='Date'), yaxis=dict(title='', rangemode='tozero'),
                      legend=dict(x=0.9, y=0.9))
    # set line color and width
    fig.update_traces(line=dict(color=my_chart_color, width=2, dash='solid'))
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
                x=0.35,
                y=1.2,
                yanchor='auto',  # top
                font=dict(size=10),
            ),
            rangeslider=dict(  # bgcolor='45B8AC',
                visible=True,
                range=[df.index.min(), df.index.max()]  # Set range of slider based on data
            ),
            type='date'
        )
    )

    # Display Plotly Express figure in Streamlit
    st.plotly_chart(fig, use_container_width=True, key=key)


def create_rfe_plot(df_ranking):
    """
    Create a scatter plot of feature rankings and selected features.

    Parameters:
        df_ranking (pandas.DataFrame): A DataFrame with feature rankings and selected features.

    Returns:
        None
    """
    # calculate number of top features chosen by user in slider
    num_top_features = (len(df_ranking[df_ranking['Selected'] == 'Yes']))

    # streamlit title rfe plot
    my_text_paragraph('Recursive Feature Elimination', my_font_size='26px')
    # streamlit subtitle rfe plot
    my_text_paragraph('with Cross-Validation (RFECV)', my_font_size='16px')

    rank_str = get_state("SELECT_PAGE_RFE", "num_features_rfe")
    my_text_paragraph(f'<b> TOP {num_top_features}</b>', my_font_size='16px', my_font_family='Segui UI')

    fig = px.scatter(df_ranking, x='Features', y='Ranking', color='Selected', hover_data=['Ranking'],
                     color_discrete_map={'Yes': '#4715EF', 'No': '#000000'})
    fig.update_layout(
        title={
            'text': '',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Features',
        yaxis_title='Ranking',
        legend_title='Selected',
        xaxis_tickangle=-45  # set the tickangle to x degrees
    )
    return fig


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


def plot_train_test_split(df, split_index):
    """
    Plot the train-test split of the given dataframe.

    Parameters:
    -----------
    df: pandas.DataFrame
        A pandas dataframe containing the time series data.
    split_index: int
        The index position that represents the split between the training and test set.

    Returns:
    --------
    fig: plotly.graph_objs._figure.Figure
        A plotly Figure object containing the train-test split plot.
    """
    try:
        train_linecolor = get_state('COLORS', 'chart_color')
        test_linecolor = adjusted_color = adjust_brightness(train_linecolor, 2)
    except:
        train_linecolor = '#4715ef'
        test_linecolor = '#efac15'

    # Get the absolute maximum value of the data
    max_value = df.iloc[:, 0].max()
    min_value = min(df.iloc[:, 0].min(), 0)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index[:split_index],
            y=df.iloc[:split_index, 0],
            mode='lines',
            name='Train',
            line=dict(color=train_linecolor)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index[split_index:],
            y=df.iloc[split_index:, 0],
            mode='lines',
            name='Test',
            line=dict(color=test_linecolor)
        )
    )
    fig.update_layout(
        title='',
        yaxis=dict(range=[min_value * 1.1, max_value * 1.1]),
        # Set y-axis range to include positive and negative values
        shapes=[dict(type='line',
                     x0=df.index[split_index],
                     y0=-max_value * 1.1,  # Set y0 to -max_value*1.1
                     x1=df.index[split_index],
                     y1=max_value * 1.1,  # Set y1 to max_value*1.1
                     line=dict(color='grey',
                               dash='dash'))],
        annotations=[dict(
            x=df.index[split_index],
            y=max_value * 1.05,
            xref='x',
            yref='y',
            text='Train/Test<br>Split',
            showarrow=True,
            font=dict(color="grey", size=15),
            arrowhead=1,
            ax=0,
            ay=-40
        )
        ]
    )
    split_date = df.index[split_index - 1]
    fig.add_annotation(
        x=split_date,
        y=0.99 * max_value,
        text=str(split_date.date()),
        showarrow=False,
        font=dict(color="grey", size=16)
    )
    return fig


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
                    color_continuous_scale=[[0, '#FFFFFF'], [1, '#4715EF']],  # "Blues",
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
    fig.update_layout(width=400,
                      height=400,
                      margin=dict(l=200, r=200, t=0, b=0))

    # show plotly figure in streamlit
    st.plotly_chart(fig, use_container_width=True)


# ******************************************************************************
# DOCUMENTATION FUNCTIONS
# ******************************************************************************
def model_documentation():
    ''' SHOW MODEL DOCUMENTATION
        - Naive Models
        - Linear Regression
        - SARIMAX
        - Prophet
    '''
    vertical_spacer(2)
    # =============================================================================
    # ADD MOON MENU BUTTONS
    # =============================================================================
    # returns the lowercase with no spaces but underscore your button_names when clicked
    # clicked_moon_btn = create_moon_clickable_btns(button_names = ['Home', 'Naive Model', 'Linear Regression', 'SARIMAX', 'Prophet'])
    clicked_moon_btn = st.selectbox(label='Select Model',
                                    options=['-', 'Naive Model', 'Linear Regression', 'SARIMAX', 'Prophet'])

    with st.expander('', expanded=True):
        col1, col2, col3 = st.columns([2, 8, 2])
        if clicked_moon_btn == '-':
            st.image('./images/model_details_info.png')

        elif clicked_moon_btn == 'Naive Model':
            with col2:
                my_text_header('Naive Model')
                vertical_spacer(1)
                st.markdown('''
                <p style="text-align: justify;">
                The <strong>Naive Model</strong> is one of the simplest forecasting models in time series analysis. 
                It assumes that the value of a variable at any given time is equal to the value of the variable at the previous time period. 
                This means that this model is a special case of an <strong>A</strong>uto<strong>R</strong>egressive model of order 1, also known as <b> AR(1)</b>.
                </p>

                <p style="text-align: justify;">
                The Naive Model is useful as a baseline model to compare more complex forecasting models, such as ARIMA, exponential smoothing, or machine learning algorithms. 
                It is also useful when the underlying data generating process is highly unstable or unpredictable, and when there is no trend, seasonality, or other patterns to capture.
                The Naive Model can be expressed as a simple equation: 
                </p>

                &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; $\hat{y}_{t} = y_{t-1}$

                <p style="text-align: justify;">
                where:
                </p>

                - $y_t$ is the value of the variable at time $t$
                <br> 
                - $y_{t-1}$ is the value of the variable at time $_{t-1}$

                <br>

                <p style="text-align: justify;">
                The Naive Model can be extended to incorporate seasonal effects, by introducing a lag period corresponding to the length of the seasonal cycle. 
                For example, if the time series has a weekly seasonality, the Naive Model with a lag of one week is equivalent to the model with a lag of seven days, and is given by:
                </p>

                &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$\hat{y}_{t} = y_{t-7}$

                <p style="text-align: justify;">
                where:
                </p>

                - $y_t$ is the value of the variable at time $t$
                <br>
                - $y_{t-7}$ is the value of the variable at time $_{t-7}$

                <br>
                <p style="text-align: justify;">
                In general, the lag value for the seasonal Naive Model should be determined based on the length of the seasonal cycle in the data, and can be estimated using visual inspection, autocorrelation analysis, or domain knowledge.
                </p>
                ''', unsafe_allow_html=True)

        elif clicked_moon_btn == 'Linear Regression':
            with col2:
                my_text_header('Linear Regression')
                vertical_spacer(1)
                st.markdown('''
                            <p style="text-align: justify;"> 
                            <strong> The Linear Regression Model </strong> is used to analyze the relationship between a dependent variable and one or more independent variables. 
                            It involves finding a line or curve that best fits the data and can be used to make predictions. 
                            The method assumes that the relationship between the variables is linear and that errors are uncorrelated.


                            To find this line, we use a technique called least squares regression, which involves finding the line that minimizes the sum of the squared differences between the predicted values and the actual values. 
                            The line is described by the equation:
                            </p>                           

                            &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; $$Y = β₀ + β₁ X$$

                            where:
                            - $Y$ is the dependent variable</li>
                            - $X$ is the independent variable</li>
                            - $β₀$ is the intercept &rarr; the value of $Y$ when $X = 0$
                            - $β₁$ is the slope &rarr; the change in $Y$ for a unit change in $X$
                            <br>
                            ''', unsafe_allow_html=True)
        elif clicked_moon_btn == 'SARIMAX':
            with col2:
                my_text_header('SARIMAX')
                vertical_spacer(1)
                st.markdown('''
                            <p style="text-align: justify;">
                            <strong>SARIMAX</strong>, or <b>S</b>easonal <b>A</b>utoregressive <b>I</b>ntegrated <b>M</b>oving <b>A</b>verage with e<b>X</b>ogenous variables, is a popular time series forecasting model.
                            The ARIMA model is a time series forecasting model that uses past values of a variable to predict future values. 
                            SARIMAX extends ARIMA by incorporating seasonal patterns and adding exogenous variables <b>X</b> that can impact the variable being forecasted. The model can be represented as follows:
                            <br>
                            <br>
                            &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <strong> SARIMA(p, d, q)(P, D, Q)s </strong>

                            Non-Seasonal Terms    
                            - <b>p </b> &emsp;The order of the autoregressive (AR) term, which refers to the number of lagged observations of the dependent variable in the model. A higher value of p means the model is considering more past values of the dependent variable.
                            <br>
                            - <b>d</b> &emsp;The order of the differencing (I) term, which refers to the number of times the data needs to be differenced to make it stationary. Stationarity is a property of time series data where the statistical properties, such as the mean and variance, are constant over time.
                            <br>
                            - <b>q</b> &emsp;The order of the moving average (MA) term, which refers to the number of lagged forecast errors in the model. A higher value of q means the model is considering more past forecast errors.
                            <br>

                            <br>Seasonal Terms:
                            - <b>P</b> &emsp;The seasonal order of the autoregressive term, which refers to the number of seasonal lags in the model.
                            <br>
                            - <b>D</b> &emsp;The seasonal order of differencing, which refers to the number of times the data needs to be differenced at the seasonal lag to make it stationary.
                            <br>
                            - <b>Q</b> &emsp;The seasonal order of the moving average term, which refers to the number of seasonal lags of the forecast errors in the model.
                            <br>
                            - <b>s:</b> &emsp;The length of the seasonal cycle, which is the number of time steps in each season. For example, if the data is monthly and the seasonality is yearly, s would be 12. The parameter s is used to determine the number of seasonal lags in the model.
                            <br>
                            <br>Exogenous Variables:
                            <br>
                            - <b>X</b>  &emsp; These are external factors that can impact the variable being forecasted. They are included in the model as additional inputs.  
                            </p>
                            ''', unsafe_allow_html=True)
        elif clicked_moon_btn == 'Prophet':
            with col2:
                my_text_header('Prophet')
                vertical_spacer(1)
                st.markdown('''
                            The Facebook <strong> Prophet </strong> model is a popular open-source library for time series forecasting developed by Facebook's Core Data Science team.
                            It is designed to handle time series data with strong seasonal effects and other external factors.
                            It uses a combination of historical data and user-defined inputs to generate forecasts for future time periods.  
                            <br>
                            <center><h6>Variables in the Prophet Model</h6></center>
                            The main variables in the Prophet model are:
                            <br> 
                            - <b>Trend</b>: This is the underlying pattern in the data that represents the long-term direction of the series. It can be linear or non-linear and is modeled using a piecewise linear function.
                            <br> 
                            - <b>Seasonality</b>: This is the periodic pattern in the data that repeats over fixed time intervals. It can be daily, weekly, monthly, or yearly, and is modeled using Fourier series.
                            <br> 
                            - <b>Holidays</b>: These are user-defined events or time periods that are known to affect the time series. The model includes them as additional regressors in the forecasting equation.
                            <br> 
                            - <b>Regressors</b>: These are additional time-varying features that can affect the time series, such as weather, economic indicators, or other external factors.
                            <br>                               
                            <br>
                            <center><h6>Math Behind the Prophet Model</h6></center>
                            The math behind the Prophet model involves fitting a Bayesian additive regression model to the time series data. The model is formulated as follows:
                            <br>
                            <br>

                            &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; $$y_t = g_t + s_t + h_t + e_t$$

                            where:
                            - $y_t$ is the observed value at time $t$
                            - $g_t$ is the trend component
                            - $s_t$ is the seasonality component
                            - $h_t$ is the holiday component
                            - $e_t$ is the error term. 

                            <br>
                            The <b>trend</b> component is modeled using a piecewise linear function, while the <b>seasonality component</b> is modeled using a Fourier series. The <b>holiday component</b> and any additional regressors are included as additional terms in the regression equation.
                            <br>
                            <br>
                            The model is estimated using a Bayesian approach that incorporates prior information about the parameters and allows for uncertainty in the forecasts. The parameters are estimated using Markov Chain Monte Carlo (MCMC) sampling, which generates a large number of possible parameter values and uses them to estimate the posterior distribution of the parameters. The posterior distribution is then used to generate forecasts for future time periods.
                            <br>
                            <br>
                            Overall, the Prophet model is a powerful tool for time series forecasting that can handle complex data patterns and external factors. Its flexible modeling approach and Bayesian framework make it a popular choice for many data scientists and analysts.
                            <br>
                            <br>
                            ''', unsafe_allow_html=True)
        else:
            st.image('./images/train_info.png')

        # ******************************************************************************


# OTHER FUNCTIONS
# ******************************************************************************
def display_summary_statistics(df):
    """
    Calculate summary statistics for a given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame for which to calculate summary statistics.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the minimum, maximum, mean, median, standard deviation,
        and data type for each column of the input DataFrame.
    """
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



def forecast_wavelet_features(X, features_df_wavelet, future_dates, df_future_dates):
    """
    Forecast the wavelet features of a time series using historical data and a selected model.

    Parameters
    ----------
    X : pandas DataFrame
        The historical time series data, with each column representing a different feature.

    features_df_wavelet : pandas DataFrame
        The features to be forecasted using the discrete wavelet transform.

    future_dates : pandas DatetimeIndex
        The dates for which to make the wavelet feature forecasts.

    df_future_dates : pandas DataFrame
        The forecasted values for the wavelet features, with a column for each feature and a row for each future date.

    Returns
    -------
    pandas DataFrame
        The forecasted values for the wavelet features, combined with the original dataframe, with a column for each feature and a row for each future date.

    Raises
    ------
    StreamlitAPIException
        If an error occurs during the forecast.

    Notes
    -----
    This function uses the historical data and a selected model to forecast the wavelet features of a time series. The wavelet features are obtained using the discrete wavelet transform, and a separate model is fitted and used to forecast each feature. The resulting forecasts are combined with the original dataframe to produce the final forecast.

    """
    ########### START WAVELET CODE ###########
    # if user selected checkbox for Discrete Wavelet Features run code run prediction for wavelet features
    if dwt_features_checkbox:
        try:
            # only extract wavelet features
            X_wavelet_features = X.loc[:, features_df_wavelet.columns.intersection(X.columns)]
            X_future_wavelet = pd.DataFrame(index=future_dates, columns=X_wavelet_features.columns)
            # iterate over all wavelet feature column names
            for col in X_wavelet_features.columns:
                # define the length of the dataframe
                n = len(X_wavelet_features)
                # just use the date range as integers as independent feature (1 through n)
                X_wavelet = np.arange(1, n + 1).reshape(-1, 1)
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
                prediction_wavelet = model_wavelet.predict(np.arange(n + 1, n + len(future_dates) + 1).reshape(-1, 1))
                X_future_wavelet[col] = prediction_wavelet
            # reset the index and rename the datetimestamp column to 'date' - which is now a column that can be used to merge dataframes
            X_future_wavelet = X_future_wavelet.reset_index().rename(columns={'index': 'date'})
            # combine the independent features with the forecast dataframe
            df_future_dates = pd.merge(df_future_dates, X_future_wavelet, on='date', how='left')
            return df_future_dates
        except:
            st.warning(
                'Error: Discrete Wavelet Features are not created correctly, please remove from selection criteria')


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
    # issue with switching data frequency e.g. testcase: daily to quarterly data commented out code for now...
    # =============================================================================
    #     if my_insample_forecast_steps >= len(df):
    #         raise ValueError("Test-set size must be less than the total number of rows in the dataset.")
    # =============================================================================
    # Initialize X_train_numeric_scaled with a default value
    X_train_numeric_scaled = pd.DataFrame()  # TEST
    X_test_numeric_scaled = pd.DataFrame()  # TEST
    X_numeric_scaled = pd.DataFrame()  # TEST

    # Split the data into training and testing sets
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0:1]
    # =============================================================================
    #     # Find the index of the 'date' column
    #     date_column_index = df.columns.get_loc('date')
    #     # Get the date column + all columns
    #     # except the target feature which is assumed to be column after 'date' column
    #     X = df.iloc[:, :date_column_index+1].join(df.iloc[:, date_column_index+2:])
    #     y = df.iloc[:, date_column_index+1: date_column_index+2]
    # =============================================================================

    X_train = X.iloc[:-my_insample_forecast_steps, :]
    X_test = X.iloc[-my_insample_forecast_steps:, :]
    y_train = y.iloc[:-my_insample_forecast_steps, :]
    y_test = y.iloc[-my_insample_forecast_steps:, :]

    # initialize variable
    scaler = ""

    # Scale the data if user selected a scaler choice in the normalization / standardization in streamlit sidebar
    if scaler_choice != "None":
        # check if there are numerical features in dataframe, if so run code
        if numerical_features:
            # Select only the numerical features to be scaled
            X_train_numeric = X_train[numerical_features]
            X_test_numeric = X_test[numerical_features]
            X_numeric = X[numerical_features]

            # Create the scaler based on the selected choice
            scaler_choices = {
                "MinMaxScaler": MinMaxScaler(),
                "RobustScaler": RobustScaler(),
                "MaxAbsScaler": MaxAbsScaler(),
                "PowerTransformer": PowerTransformer(),
                "QuantileTransformer": QuantileTransformer(n_quantiles=100, output_distribution="normal")
            }

            if scaler_choice not in scaler_choices:
                raise ValueError("Invalid scaler choice. Please choose from: MinMaxScaler, RobustScaler, MaxAbsScaler, "
                                 "PowerTransformer, QuantileTransformer")

            scaler = scaler_choices[scaler_choice]

            # Fit the scaler on the training set and transform both the training and test sets
            X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
            X_train_numeric_scaled = pd.DataFrame(X_train_numeric_scaled, columns=X_train_numeric.columns,
                                                  index=X_train_numeric.index)

            # note: you do not want to fit_transform on the test set else the distribution of the entire dataset is
            # used and is data leakage
            X_test_numeric_scaled = scaler.transform(X_test_numeric)
            X_test_numeric_scaled = pd.DataFrame(X_test_numeric_scaled, columns=X_test_numeric.columns,
                                                 index=X_test_numeric.index)

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


def perform_train_test_split_standardization(X, y, X_train, X_test, y_train, y_test, my_insample_forecast_steps,
                                             scaler_choice=None, numerical_features=[]):
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
    # Check If the specified test-set size is greater than or equal to the total number of rows in the dataset.
    if my_insample_forecast_steps >= len(y):
        raise ValueError("Test-set size must be less than the total number of rows in the dataset.")
    if scaler_choice != "None":
        # Check if there are numerical features in the dataframe
        if numerical_features:
            # Select only the numerical features to be scaled
            X_train_numeric = X_train[numerical_features]
            X_test_numeric = X_test[numerical_features]
            X_numeric = X[numerical_features]
            # if user selected in sidebar menu 'standardscaler'
            if scaler_choice == "StandardScaler":
                scaler = StandardScaler()
            else:
                raise ValueError("Invalid scaler choice. Please choose from: StandardScaler")
            # Fit the scaler on the training set and transform both the training and test sets
            X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
            X_train_numeric_scaled = pd.DataFrame(X_train_numeric_scaled, columns=X_train_numeric.columns,
                                                  index=X_train_numeric.index)
            X_test_numeric_scaled = scaler.transform(X_test_numeric)
            X_test_numeric_scaled = pd.DataFrame(X_test_numeric_scaled, columns=X_test_numeric.columns,
                                                 index=X_test_numeric.index)
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


def train_test_split_slider(df):
    """
   Creates a slider for the user to choose the number of days or percentage for train/test split.

   Args:
       df (pd.DataFrame): Input DataFrame.

   Returns:
       tuple: A tuple containing the number of days and percentage for the in-sample forecast.
   """

    # update the session state of the form and make it persistent when switching app_pages
    def form_callback():
        st.session_state['percentage'] = st.session_state['percentage']
        st.session_state['steps'] = st.session_state['steps']

    with st.sidebar:
        with st.form('train/test split'):
            my_text_paragraph('Train/Test Split')
            col1, col2 = st.columns(2)
            with col1:
                split_type = st.radio(label="*Select split type:*",
                                      options=("Steps", "Percentage"),
                                      index=1,
                                      help="""
                                             Set your preference for how you want to **split** the training data and test data:
                                             \n- as a `percentage` (between 1% and 99%)  \
                                             \n- in `steps` (for example number of days with daily data, number of weeks with weekly data, etc.)
                                             """)

                if split_type == "Steps":
                    with col2:
                        insample_forecast_steps = st.slider(label='*Size of the test-set in steps:*',
                                                            min_value=1,
                                                            max_value=len(df) - 1,
                                                            step=1,
                                                            key='steps')

                        insample_forecast_perc = st.session_state['percentage']
                else:
                    with col2:
                        insample_forecast_perc = st.slider('*Size of the test-set as percentage*', min_value=1,
                                                           max_value=99, step=1, key='percentage')
                        insample_forecast_steps = round((insample_forecast_perc / 100) * len(df))

            # show submit button in streamlit centered in sidebar
            col1, col2, col3 = st.columns([4, 4, 4])
            with col2:
                train_test_split_btn = st.form_submit_button("Submit", type="secondary", on_click=form_callback)
    return insample_forecast_steps, insample_forecast_perc


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
    """
    Plot the actual and forecasted time series data on a line chart using Plotly.

    Args:
    - df_actual (pd.DataFrame): A Pandas DataFrame containing the actual time series data with DatetimeIndex as the index.
    - df_forecast (pd.DataFrame): A Pandas DataFrame containing the forecasted time series data with DatetimeIndex as the index and a 'forecast' column.
    - title (str, optional): The title of the chart. Default is an empty string.

    Returns:
    - None: Displays the chart in Streamlit using `st.plotly_chart`.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_actual.index, y=df_actual.iloc[:, 0], name='Actual', mode='lines'))
    fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast['forecast'], name='Forecast', mode='lines',
                             line=dict(dash='dot',
                                       color='#87CEEB')))  # dash styles: ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
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
    # show plot in streamlit
    st.plotly_chart(fig, use_container_width=True)

def my_subheader_metric(string1, color1="#cfd7c2", metric=0, color2="#FF0000", my_style="#000000", my_size=5):
    metric_rounded = "{:.2%}".format(metric)
    metric_formatted = f"<span style='color:{color2}'>{metric_rounded}</span>"
    string1 = string1.replace(f"{metric}", f"<span style='color:{color1}'>{metric_rounded}</span>")
    st.markdown(f'<h{my_size} style="color:{my_style};"> <center> {string1} {metric_formatted} </center> </h{my_size}>',
                unsafe_allow_html=True)


def wait(seconds):
    start_time = time.time()
    with st.spinner(f"Please wait... {int(time.time() - start_time)} seconds passed"):
        time.sleep(seconds)


# def my_holiday_name_func(my_date):
#     """
#     This function takes a date as input and returns the name of the holiday that falls on that date.
#
#     Parameters:
#     -----------
#     my_date : str
#         The date for which the holiday name is to be returned. The date should be in the format 'YYYY-MM-DD'.
#
#     Returns:
#     --------
#     str:
#         The name of the holiday that falls on the given date. If there is no holiday on that date, an empty string is returned.
#
#     Examples:
#     ---------
#     >>> my_holiday_name_func('2022-07-04')
#     'Independence Day'
#     >>> my_holiday_name_func('2022-12-25')
#     'Christmas Day'
#     >>> my_holiday_name_func('2022-09-05')
#     'Labor Day'
#     """
#     holiday_name = calendar().holidays(start=my_date, end=my_date, return_name=True)
#     if len(holiday_name) < 1:
#         holiday_name = ""
#         return holiday_name
#     else:
#         return holiday_name[0]


def my_metrics(my_df, model_name, metrics_dict):
    """
    Calculates performance metrics for a given model.

    Args:
        my_df (pandas.DataFrame): DataFrame containing actual and predicted values for evaluation.
            Columns:
                - 'Actual': Actual target variable values.
                - 'Predicted': Predicted target variable values.
        model_name (str): Name of the model.

    Returns:
        mape (float): Mean Absolute Percentage Error (MAPE) for the model.
        rmse (float): Root Mean Squared Error (RMSE) for the model.
        r2 (float): R-squared (coefficient of determination) for the model.
        smape (float): Symmetric Mean Absolute Percentage Error (SMAPE) for the model.

    Note:
        - The function calculates MAPE, RMSE, R-squared, and SMAPE metrics based on the provided DataFrame.
        - MAPE is the average percentage difference between predicted and actual values.
        - RMSE is the square root of the mean squared error between predicted and actual values.
        - R-squared measures the proportion of the variance in the target variable explained by the model.
        - SMAPE is the symmetric mean absolute percentage error between predicted and actual values.
        - The function updates the 'metrics_dict' dictionary with the calculated metrics for the given model.

    Example:
        result = my_metrics(df_preds, 'MyModel')
        mape_value, rmse_value, r2_value, smape_value = result
    """
    # Check if not empty dataframe
    if not my_df.empty:
        # Calculate metrics
        mape = np.mean(np.abs((my_df['Actual'] - my_df['Predicted']) / my_df['Actual']))
        smape = (1 / len(my_df)) * np.sum(
            np.abs(my_df['Actual'] - my_df['Predicted']) / (np.abs(my_df['Actual']) + np.abs(my_df['Predicted'])))
        mse = mean_squared_error(my_df['Actual'], my_df['Predicted'])
        rmse = np.sqrt(mse)
        r2 = r2_score(my_df['Actual'], my_df['Predicted'])

    else:
        # Set metrics equal to none if df is empty
        mape = None
        smape = None
        mse = None
        rmse = None
        r2 = None

    # Add the results to the dictionary
    metrics_dict[model_name] = {'mape': mape, 'smape': smape, 'mse': mse, 'rmse': rmse, 'r2': r2}

    return metrics_dict


def display_my_metrics(my_df, model_name="", my_subtitle=None):
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
    # st.markdown(f'<h2 style="text-align:center">{model_name}</h2></p>', unsafe_allow_html=True)
    my_text_header(f'{model_name}')

    if my_subtitle is not None:
        my_text_paragraph(my_string=my_subtitle, my_text_align='center', my_font_weight=200, my_font_size='14px')

    vertical_spacer(2)

    # define vertical spacings
    col0, col1, col2, col3, col4, col5 = st.columns([2, 3, 3, 3, 3, 1])

    # Get the evaluation metrics for your model
    metrics_dict = my_metrics(my_df, model_name)

    # define the font size of the st.metric
    st.markdown(
        """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 18px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Show the evaluation metrics on Model Card
    with col1:
        st.metric(label=':red[MAPE]',
                  value="{:.2%}".format(metrics_dict[model_name]['mape']),
                  help='The `Mean Absolute Percentage Error` (MAPE) measures the average absolute percentage difference between the predicted values and the actual values. It provides a relative measure of the accuracy of a forecasting model. A lower MAPE indicates better accuracy.')
    with col2:
        st.metric(label=':red[SMAPE]',
                  value="{:.2%}".format(metrics_dict[model_name]['smape']),
                  help='The `Symmetric Mean Absolute Percentage Error` (SMAPE) is similar to MAPE but addresses the issue of scale dependency. It calculates the average of the absolute percentage differences between the predicted values and the actual values, taking into account the magnitude of both values. SMAPE values range between 0% and 100%, where lower values indicate better accuracy.')
    with col3:
        st.metric(label=':red[RMSE]',
                  value=round(metrics_dict[model_name]['rmse'], 2),
                  help='The `Root Mean Square Error` (RMSE) is a commonly used metric to evaluate the accuracy of a prediction model. It calculates the square root of the average of the squared differences between the predicted values and the actual values. RMSE is sensitive to outliers and penalizes larger errors.')
    with col4:
        st.metric(label=':green[R²]',
                  value=round(metrics_dict[model_name]['r2'], 2),
                  help='`R-squared` (R²) is a statistical measure that represents the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, where 1 indicates a perfect fit. R-squared measures how well the predicted values fit the actual values and is commonly used to assess the goodness of fit of a regression model.')


def forecast_naive_model1_insample(y_test, lag=None, custom_lag_value=None):
    """
    Implements Naive Model I for time series forecasting.

    Args:
        y_test: The target variable of the test set.
        lag: The lag value to use for shifting the target variable. Options are 'day', 'week', 'month', 'year', or 'custom'.
        custom_lag_value: The custom lag value to use if lag='custom'.

    Returns:
        df_preds (pandas.DataFrame): DataFrame containing actual and predicted values for evaluation.
            Columns:
                - 'Actual': Actual target variable values.
                - 'Predicted': Predicted target variable values.
                - 'Percentage_Diff': Percentage difference between predicted and actual values.
                - 'MAPE': Mean Absolute Percentage Error between predicted and actual values.

    Raises:
        ValueError: If an invalid value for 'lag' is provided, or if custom_lag_value is not provided when lag='custom'.

    Note:
        - The function applies a lag to the target variable based on the 'lag' argument.
        - It creates a DataFrame with actual and predicted values.
        - The index is set to the date portion of the datetime index.
        - Rows with missing values are dropped.
        - Percentage difference and MAPE (Mean Absolute Percentage Error) are calculated.

    Example:
        lag_value = 'day'
        custom_lag_value = 5
        result = forecast_naive_model1_insample(y_test, lag=lag_value, custom_lag_value=5)
    """

    if lag == 'day':
        y_pred = y_test.shift(1)
    elif lag == 'week':
        y_pred = y_test.shift(7)
    elif lag == 'month':
        y_pred = y_test.shift(30)
    elif lag == 'year':
        y_pred = y_test.shift(365)
    elif lag == 'custom':
        if custom_lag_value is None:
            raise ValueError('Custom lag value is required when lag is set to "custom".')
        y_pred = y_test.shift(custom_lag_value)
    else:
        raise ValueError('Invalid value for "lag". Must be "day", "week", "month", "year", or "custom".')

    # Create a DataFrame for insample predictions versus actual
    df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})

    # Set the index to just the date portion of the datetime index
    df_preds.index = df_preds.index.date

    # Drop rows with N/A values
    df_preds.dropna(inplace=True)

    # Calculate the percentage difference between actual and predicted values and add it as a new column
    df_preds['Error'] = (df_preds['Actual'] - df_preds['Predicted'])
    df_preds['Error (%)'] = (df_preds['Error'] / df_preds['Actual'])

    return df_preds


def forecast_naive_model2_insample(y_test, size_rolling_window, agg_method_rolling_window):
    """
    Implements Naive Model 2 for time series forecasting with a rolling window approach.

    Args:
        y_test: The target variable of the test set.
        size_rolling_window (int): The size of the rolling window.
        agg_method_rolling_window (str): The aggregation method to apply within the rolling window. Options are
                                          'mean', 'median', 'mode', or custom aggregation functions.

    Returns:
        df_preds (pandas.DataFrame): DataFrame containing actual and predicted values for evaluation.
            Columns:
                - 'Actual': Actual target variable values.
                - 'Predicted': Predicted target variable values.
                - 'Percentage_Diff': Percentage difference between predicted and actual values.
                - 'MAPE': Mean Absolute Percentage Error between predicted and actual values.

    Raises:
        ValueError: If an invalid value for 'agg_method_rolling_window' is provided.

    Note:
        - The function uses a rolling window approach to make predictions.
        - The rolling window size and aggregation method are specified by the arguments.
        - It creates a DataFrame with actual and predicted values.
        - The index is set to the date portion of the datetime index.
        - Rows with missing values are dropped.
        - Percentage difference and MAPE (Mean Absolute Percentage Error) are calculated.

    Example:
        size_window = 7
        agg_method = 'mean'
        result = forecast_naive_model2_insample(y_test, size_rolling_window=size_window, agg_method_rolling_window=agg_method)
    """
    agg_method_rolling_window = agg_method_rolling_window.lower()

    # Apply rolling window and aggregation
    if agg_method_rolling_window == 'mean':
        y_pred = y_test.rolling(size_rolling_window).mean()
    elif agg_method_rolling_window == 'median':
        y_pred = y_test.rolling(size_rolling_window).median()
    elif agg_method_rolling_window == 'mode':
        y_pred = y_test.rolling(size_rolling_window).apply(lambda x: x.value_counts().index[0])
    elif callable(agg_method_rolling_window):
        y_pred = y_test.rolling(size_rolling_window).apply(agg_method_rolling_window)
    else:
        raise ValueError('Invalid value for "agg_method_rolling_window". Must be "mean", "median", "mode", '
                         'or a valid custom aggregation function.')

    # Create dataframe for insample predictions versus actual
    df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})

    # Set the index to just the date portion of the datetime index
    df_preds.index = df_preds.index.date

    # Drop rows with N/A values
    df_preds.dropna(inplace=True)

    # Calculate the percentage difference between actual and predicted values and add it as a new column
    df_preds['Error'] = (df_preds['Actual'] - df_preds['Predicted'])

    df_preds['Error (%)'] = (df_preds['Error'] / df_preds['Actual'])

    return df_preds


def forecast_naive_model3_insample(y_train, y_test, agg_method):
    """
    Implements Naive Model for time series forecasting without a rolling window approach.

    Args:
        X_train: The input features of the training set.
        y_train: The target variable of the training set.
        X_test: The input features of the test set.
        y_test: The target variable of the test set.
        agg_method (str): The aggregation method to apply to the entire time series. Options are 'mean', 'median', or 'mode'.

    Returns:
        df_preds (pandas.DataFrame): DataFrame containing actual and predicted values for evaluation.
            Columns:
                - 'Actual': Actual target variable values.
                - 'Predicted': Predicted target variable values.
                - 'Percentage_Diff': Percentage difference between predicted and actual values.
                - 'MAPE': Mean Absolute Percentage Error between predicted and actual values.

    Raises:
        ValueError: If an invalid value for 'agg_method' is provided.

    Note:
        - The function calculates the mean, median, or mode for the entire time series.
        - It creates a DataFrame with actual and predicted values.
        - The index is set to the date portion of the datetime index.
        - Rows with missing values are dropped.
        - Percentage difference and MAPE (Mean Absolute Percentage Error) are calculated.

    Example:
        agg_method = 'mean'
        result = forecast_naive_model_insample(X_train, y_train, X_test, y_test, agg_method)
    """
    agg_method = agg_method.lower()

    # Apply aggregation
    if agg_method == 'mean':
        my_metric = y_train.mean().values[0]
    elif agg_method == 'median':
        my_metric = y_train.median().values[0]
    elif agg_method == 'mode':
        my_metric = y_train.iloc[0].mode().iloc[0]  # TEST IF CORRECTLY CALCULATED
    else:
        raise ValueError('Invalid value for "agg_method". Must be "mean", "median", or "mode".')

    # Create DataFrame with datetime index and metric values
    y_pred = pd.DataFrame(index=y_test.index, data=my_metric, columns=['Values'])
    # =============================================================================
    #     st.write(y_train, y_test) # TEST
    #     st.write(y_pred) # TEST
    # =============================================================================

    # Create dataframe for insample predictions versus actual
    df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})

    # st.write(df_preds)

    # Set the index to just the date portion of the datetime index
    df_preds.index = df_preds.index.date

    # Drop rows with N/A values
    df_preds.dropna(inplace=True)

    # Calculate the percentage difference between actual and predicted values and add it as a new column
    df_preds['Error'] = (df_preds['Actual'] - df_preds['Predicted'])
    df_preds['Error (%)'] = (df_preds['Error'] / df_preds['Actual'])

    return df_preds


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
    # Train the model using the training/test sets if selectbox equal to 'Yes' to include features (default option)
    # if not then only use the target feature and the date column converted to numeric
    if get_state("TRAIN_PAGE", "include_feature_selection") == 'No':
        # create numeric date values e.g. 0,1,2,3 through N of your data (whereby X_test (e.g. first value = 101) continues counting from last value of X_train (e.g. last value = 100))
        X_train = pd.DataFrame(range(len(X_train)), index=X_train.index, columns=['date_numeric'])
        X_test = pd.DataFrame(range(len(X_train), len(X_train) + len(X_test)), index=X_test.index,
                              columns=['date_numeric'])
    else:
        pass

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # =============================================================================
    # COEFFICIENTS TABLE
    # =============================================================================
    # Extract the coefficient values and feature names
    coefficients = model.coef_
    # st.write(coefficients) # TEST

    intercept = model.intercept_
    # st.write(intercept) # TEST

    feature_names = X_train.columns.tolist()
    # st.write(feature_names) # TEST

    # Create a table to display the coefficients and intercept
    coefficients_table = pd.DataFrame(
        {'Feature': ['Intercept'] + feature_names, 'Coefficient': np.insert(coefficients, 0, intercept)})
    # st.write(coefficients_table)

    # =============================================================================
    # LINEAR REGRESSION EQUATION
    # =============================================================================
    # Get the intercept and coefficients from the table
    intercept = coefficients_table.loc[0, 'Coefficient'].round(2)
    weights = coefficients_table.loc[1:, 'Coefficient'].round(2)
    feature_names = coefficients_table.loc[1:, 'Feature']

    # Reverse the weights and feature_names lists
    weights = weights[::-1]
    feature_names = feature_names[::-1]

    # Create the symbols for the features and coefficients
    x_symbols = sp.symbols(feature_names)
    coefficients_symbols = sp.symbols(['b0'] + [f'b{i}' for i in range(1, len(x_symbols) + 1)])

    # Create the custom equation string
    equation_parts = [f"{coeff} * {feat}" for coeff, feat in zip(weights, feature_names)]
    equation_str = f"y_hat = {intercept} + " + " + ".join(equation_parts)

    # Create a dictionary to substitute the coefficient symbols with their values
    subs_dict = {coefficients_symbols[i]: coeff for i, coeff in enumerate(weights)}
    subs_dict[coefficients_symbols[0]] = intercept

    # Substitute the coefficient symbols in the equation expression
    for symbol, value in subs_dict.items():
        equation_str = equation_str.replace(str(symbol), str(value))

    # Create dataframe for insample predictions versus actual
    df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})

    # set the index to just the date portion of the datetime index
    df_preds.index = df_preds.index.date

    # Drop rows with N/A values
    df_preds.dropna(inplace=True)

    # Calculate the percentage difference between actual and predicted values and add it as a new column
    df_preds['Error'] = (df_preds['Actual'] - df_preds['Predicted'])
    df_preds['Error (%)'] = df_preds['Error'] / df_preds['Actual']

    return df_preds, coefficients_table, equation_str


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
    # Step 0: check if user wants to include feature selection and if not create a date_numeric for X_train and X_test (exogenous variable)
    if get_state("TRAIN_PAGE", "include_feature_selection") == 'No':
        # create numeric date values e.g. 0,1,2,3 through N of your data (whereby X_test (e.g. first value = 101) continues counting from last value of X_train (e.g. last value = 100))
        exog_train = pd.DataFrame(range(len(exog_train)), index=exog_train.index, columns=['date_numeric'])
        exog_test = pd.DataFrame(range(len(exog_train), len(exog_train) + len(exog_test)), index=exog_test.index,
                                 columns=['date_numeric'])
    else:
        pass

    # Step 1: Define the Model
    model = sm.tsa.statespace.SARIMAX(endog=endog_train,
                                      exog=exog_train,
                                      order=order,
                                      seasonal_order=seasonal_order)

    # Step 2: Fit the model e.g. Train the model
    results = model.fit()

    # Step 3: Generate predictions
    y_pred = results.predict(start=endog_test.index[0],
                             end=endog_test.index[-1],
                             exog=exog_test)

    # Step 4: Combine Actual values and Prediction in a single dataframe
    df_preds = pd.DataFrame({'Actual': endog_test.squeeze(),
                             'Predicted': y_pred.squeeze()},
                            index=endog_test.index)

    # Step 5: Calculate Difference
    # Calculate the percentage difference between actual and predicted values and add it as a new column
    df_preds['Error'] = (df_preds['Actual'] - df_preds['Predicted'])

    df_preds['Error (%)'] = (df_preds['Error'] / df_preds['Actual'])

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
    df_preds, coefficients_table, equation_str = evaluate_regression_model(model, X_train, y_train, X_test, y_test)

    with st.expander('📈' + model_name, expanded=True):
        display_my_metrics(my_df=df_preds,
                           model_name=model_name)

        # plot graph with actual versus insample predictions
        plot_actual_vs_predicted(df_preds,
                                 my_conf_interval)

        # =============================================================================
        #         # =============================================================================
        #         #  Show/Hide Button to download dataframe
        #         # =============================================================================
        #         # have button available for user and if clicked it expands with the dataframe
        #         col1, col2, col3 = st.columns([100,50,95])
        #         with col2:
        #
        #             # create empty placeholder for button show/hide
        #             placeholder = st.empty()
        #
        #             # create button (enabled to click e.g. disabled=false with unique key)
        #             btn = placeholder.button('Show Details', disabled=False,  key = "show_linreg_model_btn")
        #
        #         # if button is clicked run below code
        #         if btn == True:
        #
        #             # display button with text "click me again", with unique key
        #             placeholder.button('Hide Details', disabled=False, key = "hide_linreg_model_btn")
        #
        #             st.markdown('---')
        # =============================================================================

        # show the dataframe
        st.dataframe(df_preds.style.format(
            {'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Percentage_Diff': '{:.2%}', 'MAPE': '{:.2%}'}),
            use_container_width=True)

        # create download button for forecast results to .csv
        download_csv_button(df_preds,
                            my_file="insample_forecast_linear_regression_results.csv",
                            help_message=f'Download your **{model_name}** model results to .CSV',
                            my_key='download_btn_linreg_df_preds')

        st.markdown('---')

        my_text_paragraph('Coefficients Table', my_font_size='24px')

        vertical_spacer(1)

        st.dataframe(coefficients_table, use_container_width=True)

        st.markdown('---')

        my_text_paragraph('Regression Equation', my_font_size='24px')

        vertical_spacer(1)

        st.write(equation_str)
        # INTERCEPT ROUNDING SEEMS OFF? # TEST

    vertical_spacer(1)

    return df_preds


def preprocess_X_prophet(X):
    """
    Preprocess the X dataframe for Prophet modeling.

    Args:
        X (pd.DataFrame): Dataset with date index and independent features.

    Returns:
        pd.DataFrame: Preprocessed X dataframe with 'ds' column and regressor columns.
    """
    if isinstance(X.index, pd.DatetimeIndex):
        X = X.reset_index()

    X_prophet = X.copy()
    X_prophet = X_prophet.rename(columns={X_prophet.columns[0]: 'ds'})
    return X_prophet


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


def predict_prophet(y_train, y_test, X, **kwargs):
    """
    Predict future values using Prophet model.

    Args:
        y_train (pd.DataFrame): Training dataset.
        y_test (pd.DataFrame): Test dataset.
        **kwargs: Keyword arguments used to adjust the Prophet model.
            Allowed keyword arguments:
            - changepoint_prior_scale (float): Parameter for changepoint prior scale.
            - seasonality_mode (str): Parameter for seasonality mode.
            - seasonality_prior_scale (float): Parameter for seasonality prior scale.
            - holidays_prior_scale (float): Parameter for holidays prior scale.
            - yearly_seasonality (bool): Whether to include yearly seasonality.
            - weekly_seasonality (bool): Whether to include weekly seasonality.
            - daily_seasonality (bool): Whether to include daily seasonality.
            - interval_width (float): Width of the uncertainty interval.

    Returns:
        pd.DataFrame: A dataframe with the following columns:
            - Actual: The actual values of the test dataset.
            - Predicted: The predicted values of the test dataset.
            - Error: The difference between actual and predicted values.
            - Error (%): The percentage difference between actual and predicted values.
    """
    # if user set selectbox: 'include feature selection' -> 'Yes' then include the additional explanatory variables/features
    if get_state("TRAIN_PAGE", "include_feature_selection") == 'Yes':

        # Step 1: Preprocess Data for Prophet (required are at least 'ds' column and 'y' column)
        #######################
        X_train_prophet = preprocess_X_prophet(X_train)
        y_train_prophet = preprocess_data_prophet(y_train)
        joined_train_data = pd.merge(y_train_prophet, X_train_prophet, on='ds')
        # st.write('joined train data', joined_train_data) #TEST

        X_test_prophet = preprocess_X_prophet(X_test)
        y_test_prophet = preprocess_data_prophet(y_test)
        joined_test_data = pd.merge(y_test_prophet, X_test_prophet, on='ds')
        # st.write('joined test data', joined_test_data) # TEST

        # merge train and test data together in 1 dataframe
        merged_data = joined_train_data.append(joined_test_data, ignore_index=True)
        # st.write('merged data', merged_data) # TEST

        # Step 2: Define Model
        #######################
        # get the parameters from the settings either preset or adjusted by user and user pressed submit button
        m = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_mode=seasonality_mode,
                    seasonality_prior_scale=seasonality_prior_scale,
                    holidays_prior_scale=holidays_prior_scale,
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=daily_seasonality,
                    interval_width=interval_width)

        # Step 3: Add independent features/regressors to model
        #######################
        try:
            # ADD COUNTRY SPECIFIC HOLIDAYS IF AVAILABLE FOR SPECIFIC COUNTRY CODE AND USER HAS SELECTBOX SET TO TRUE FOR PROPHET HOLIDAYS
            if get_state("TRAIN_PAGE", "prophet_holidays") == True:
                m.add_country_holidays(country_name=get_state("ENGINEER_PAGE_COUNTRY_HOLIDAY", "country_code"))
        except:
            st.warning(
                'FORECASTGENIE WARNING: Could not add Prophet Holiday Features to Dataframe. Insample train/test will continue without these features and might lead to less accurate results.')

        # iterate over the names of features found from training dataset (X_train) and add them to prophet model as regressors
        for column in X_train.columns:
            m.add_regressor(column)

        # Step 4: Fit the model
        #######################
        # train the model on the data with set parameters
        # m.fit(y_train_prophet)
        m.fit(joined_train_data)

        # Step 5: Create current date range + future date range
        #######################
        # Predict on the test set
        future = m.make_future_dataframe(periods=len(y_test_prophet), freq='D')

        # step 6: Add regressors to future
        #######################
        for column in merged_data.columns:
            future[column] = merged_data[column]

        # step 7: forecast
        #######################
        forecast = m.predict(future)

    elif get_state("TRAIN_PAGE", "include_feature_selection") == 'No':

        # Step 1: Preprocess Data for Prophet (required are at least 'ds' column and 'y' column)
        #######################
        y_train_prophet = preprocess_data_prophet(y_train)
        y_test_prophet = preprocess_data_prophet(y_test)

        # Step 2: Define Model
        # Note: get the parameters from the settings either preset or adjusted by user and user pressed submit button
        #######################
        m = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_mode=seasonality_mode,
                    seasonality_prior_scale=seasonality_prior_scale,
                    holidays_prior_scale=holidays_prior_scale,
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=daily_seasonality,
                    interval_width=interval_width)

        # Step 3: Add Prophet Holiday if User set to True
        #######################
        try:
            # ADD COUNTRY SPECIFIC HOLIDAYS IF AVAILABLE FOR SPECIFIC COUNTRY CODE AND USER HAS SELECTBOX SET TO TRUE FOR PROPHET HOLIDAYS
            if get_state("TRAIN_PAGE", "prophet_holidays") == True:
                m.add_country_holidays(country_name=get_state("ENGINEER_PAGE_COUNTRY_HOLIDAY", "country_code"))
        except:
            st.warning(
                'FORECASTGENIE WARNING: Could not add Prophet Holiday Features to Dataframe. Insample train/test will continue without these features and might lead to less accurate results.')

        # Step 4: Fit the model/Train the model on the data with set parameters
        #######################
        m.fit(y_train_prophet)

        # Step 5: Create (current date range + future date range_
        # Note: Prophet model needs entire date range (train+test) versus just test date range used in others models
        #######################
        future = m.make_future_dataframe(periods=len(y_test), freq='D')

        # step 6: Add additional regressors to future
        #######################
        # not performed when 'include feature selection' == False

        # step 7: forecast
        #######################
        forecast = m.predict(future)

    # slice the test-set of the forecast - exclude the forecast on the training set although prophet model does supply it
    # Prophet model provides it to check for overfitting the model, however to prevent user from thinking it trained on whole dataset clearer to provide forecast of test set only
    yhat_test = forecast['yhat'][-len(y_test):]
    preds_df_prophet = pd.DataFrame({'Actual': y_test_prophet['y'].values, 'Predicted': yhat_test.values},
                                    index=y_test_prophet['ds'])

    # create column date and set the datetime index to date without the time i.e. 00:00:00
    preds_df_prophet['date'] = preds_df_prophet.index.strftime('%Y-%m-%d')

    # set the date column as index column
    preds_df_prophet = preds_df_prophet.set_index('date')

    # Calculate absolute error and add it as a new column
    preds_df_prophet['Error'] = preds_df_prophet['Predicted'] - preds_df_prophet['Actual']
    # Calculate percentage difference between actual and predicted values and add it as a new column
    preds_df_prophet['Error (%)'] = (preds_df_prophet['Error'] / preds_df_prophet['Actual'])

    return preds_df_prophet


def load_data(uploaded_file: IO) -> pd.DataFrame:
    """
    This function loads data from a CSV or XLS file and returns a pandas DataFrame object.
    """
    try:
        df_raw = pd.read_csv(uploaded_file, parse_dates=['date'])
    except:
        df_raw = pd.read_excel(uploaded_file, parse_dates=['date'])
    return df_raw


def create_df_pred(y_test, y_pred, show_df_streamlit=True):
    try:
        df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
        # set the index to just the date portion of the datetime index
        df_preds.index = df_preds.index.date

        # Calculate percentage difference between actual and predicted values and add it as a new column
        df_preds['Error'] = df_preds['Actual'] - df_preds['Predicted']
        df_preds['Error (%)'] = df_preds['Error'] / df_preds['Actual']

        # show the predictions versus actual results
        my_df = df_preds.style.format(
            {'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Error': '{:.2f}', 'Error (%)': '{:.2f%}'})

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


def download_csv_button(my_df, my_file="forecast_model.csv", help_message='Download dataframe to .CSV', set_index=False,
                        my_key='define_unique_key'):
    """
    Create a download button for a pandas DataFrame in CSV format.

    Parameters:
    -----------
    my_df : pandas.DataFrame
        The DataFrame to be downloaded.
    my_file : str, optional (default="forecast_model.csv")
        The name of the downloaded file.
    help_message : str, optional (default="Download dataframe to .CSV")
        The text displayed when hovering over the download button.
    set_index : bool, optional (default=False)
        If True, sets the DataFrame index as the first column in the output file.

    Returns:
    --------
    None
    """
    # create download button for forecast results to .csv
    if set_index:
        csv = convert_df_with_index(my_df)
    else:
        csv = convert_df(my_df)
    col1, col2, col3 = st.columns([54, 30, 50])
    with col2:
        st.download_button("Download",
                           csv,
                           my_file,
                           "text/csv",
                           # key='', -> streamlit automatically assigns key if not defined
                           help=help_message,
                           key=my_key)


def plot_actual_vs_predicted(df_preds, my_conf_interval):
    """
    Plots the actual and predicted values from a dataframe, along with a shaded confidence interval.

    Parameters:
    -----------
    df_preds : pandas.DataFrame
        The dataframe containing the actual and predicted values to plot.
    my_conf_interval : float
        The level of confidence for the confidence interval to display, as a percentage (e.g. 95 for 95% confidence interval).

    Returns:
    --------
    None
    """
    # Define the color palette
    colors = ['#5276A7', '#ff7700']

    # set color shading of confidence interval
    my_fillcolor = 'rgba(222,235,247,0.5)'

    # Create the figure with easy on eyes colors
    fig = px.line(df_preds, x=df_preds.index, y=['Actual', 'Predicted'], color_discrete_sequence=colors)

    # Update the layout of the figure
    fig.update_layout(
        legend_title='Legend',
        font=dict(family='Arial',
                  size=12,
                  color='#707070'
                  ),
        yaxis=dict(
            gridcolor='#E1E1E1',
            range=[np.minimum(df_preds['Actual'].min(), df_preds['Predicted'].min()) - (
                    df_preds['Predicted'].max() - df_preds['Predicted'].min()),
                   np.maximum(df_preds['Actual'].max(), df_preds['Predicted'].max()) + (
                           df_preds['Predicted'].max() - df_preds['Predicted'].min())],
            zeroline=False,  # remove the x-axis line at y=0
        ),
        xaxis=dict(gridcolor='#E1E1E1'),
        legend=dict(yanchor="bottom", y=0.0, xanchor="center", x=0.99, bgcolor='rgba(0,0,0,0)')
    )

    # Set the line colors
    for i, color in enumerate(colors):
        fig.data[i].line.color = color
        if fig.data[i].name == 'Predicted':
            fig.data[
                i].line.dash = 'dot'  # dash styles options: ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']

    # Compute the level of confidence
    confidence = float(my_conf_interval / 100)
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


# remove datatypes object - e.g. descriptive columns not used in modeling
def remove_object_columns(df, message_columns_removed=False):
    """
     Remove columns with 'object' datatype from the input dataframe.

     Parameters:
     -----------
     df : pandas.DataFrame
         Input dataframe.

     Returns:
     --------
     pandas.DataFrame
         Dataframe with columns of 'object' datatype removed.

     Example:
     --------
     >>> df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c'], 'col3': [4.5, 5.2, 6.1]})
     >>> remove_object_columns(df)
          col1  col3
     0     1    4.5
     1     2    5.2
     2     3    6.1
     """
    # background color number
    try:
        background_color_number_lst = get_state("COLORS", "chart_color")
    except:
        background_color_number_lst = '#efac15'

    # Get a list of column names with object datatype
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    # Remove the columns that are not needed
    if message_columns_removed:
        # Create an HTML unordered list with each non-NaN and non-'-' value as a list item
        html_list = "<div class='my-list'>"
        for i, value in enumerate(obj_cols):
            html_list += f"<li><span class='my-number'>{i + 1}</span>{value}</li>"
        html_list += "</div>"
        # Display the HTML list using Streamlit
        col1, col2, col3 = st.columns([295, 800, 400])
        with col2:
            st.markdown(
                f"""
                <style>
                    .my-list {{
                        font-size: 16px;
                        line-height: 1.4;
                        margin-bottom: 10px;
                        margin-left: 50px;
                        background-color: white;
                        border-radius: 10px;
                        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.3);
                        padding: 20px;
                    }}
                    .my-list li {{
                        margin: 10px 10px 10px 10px;
                        padding-left: 30px;
                        position: relative;
                        color: grey;
                    }}
                    .my-number {{
                        font-weight: bold;
                        color: white;
                        background-color: {background_color_number_lst};
                        border-radius: 50%;
                        text-align: center;
                        width: 20px;
                        height: 20px;
                        line-height: 20px;
                        display: inline-block;
                        position: absolute;
                        left: 0;
                        top: 0;
                    }}
                </style>
                {html_list}
                """,
                unsafe_allow_html=True
            )
    df = df.drop(columns=obj_cols)
    return df


def copy_df_date_index(my_df, datetime_to_date=True, date_to_index=True):
    """
    Create a deep copy of a DataFrame and optionally convert the 'date' column to a date datatype and set it as the index.

    Parameters:
    -----------
    my_df : pandas.DataFrame
        The input DataFrame.
    datetime_to_date : bool, optional (default=True)
        Whether to convert the 'date' column to a date datatype.
    date_to_index : bool, optional (default=True)
        Whether to set the 'date' column as the index.

    Returns:
    --------
    pandas.DataFrame
        A new copy of the input DataFrame, with the 'date' column optionally converted to a date datatype and/or set as the index.

    Examples:
    ---------
    >>> df = pd.DataFrame({'date': ['2022-01-01 12:00:00', '2022-01-02 12:00:00', '2022-01-03 12:00:00'], 'value': [1, 2, 3]})
    >>> copy_df_date_index(df)
               value
    date
    2022-01-01      1
    2022-01-02      2
    2022-01-03      3
    """
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






def infer_frequency(date_df_series):
    if len(date_df_series) >= 2:
        first_date = date_df_series.iloc[0]
        second_date = date_df_series.iloc[1]
        diff = second_date - first_date
        if diff.days == 1:
            my_freq = 'D'  # Daily frequency
            my_freq_name = 'Daily'
        elif diff.days == 7:
            my_freq = 'W'
            my_freq_name = 'Weekly'
        elif diff.days == 30:
            my_freq = 'M'
            my_freq_name = 'Monthly'
        elif diff.days >= 90 and diff.days < 92:
            my_freq = 'Q'
            my_freq_name = 'Quarterly'
        elif diff.days == 365:
            my_freq = 'Y'
            my_freq_name = 'Yearly'
        else:
            my_freq = None
            my_freq_name = None
    return my_freq, my_freq_name


def determine_df_frequency(df, column_name='date'):
    try:
        # initialize variables
        my_freq = None
        my_freq_name = '-'
        # infer frequency with pandas function infer_freq that has outputs possible below
        freq = pd.infer_freq(df[column_name])
        # DAILY
        if freq in ['D', 'B', 'BS']:
            my_freq = 'D'
            my_freq_name = 'Daily'
        # WEEKLY
        elif freq in ['W']:
            my_freq = 'W'
            my_freq_name = 'Weekly'
        # MONTHLY
        elif freq in ['M', 'MS', 'BM', 'BMS']:
            my_freq = 'M'
            my_freq_name = 'Monthly'
        # QUARTERLY
        elif freq in ['Q', 'QS', 'BQS', 'Q-JAN', 'Q-FEB', 'Q-MAR', 'Q-APR', 'Q-MAY', 'Q-JUN', 'Q-JUL', 'Q-AUG', 'Q-SEP',
                      'Q-OCT', 'Q-NOV', 'Q-DEC', 'QS-JAN', 'QS-FEB', 'QS-MAR', 'QS-APR', 'QS-MAY', 'QS-JUN', 'QS-JUL',
                      'QS-AUG', 'QS-SEP', 'QS-OCT', 'QS-NOV', 'QS-DEC', 'BQ-JAN', 'BQ-FEB', 'BQ-MAR', 'BQ-APR',
                      'BQ-MAY', 'BQ-JUN', 'BQ-JUL', 'BQ-AUG', 'BQ-SEP', 'BQ-OCT', 'BQ-NOV', 'BQ-DEC', 'BQS-JAN',
                      'BQS-FEB', 'BQS-MAR', 'BQS-APR', 'BQS-MAY', 'BQS-JUN', 'BQS-JUL', 'BQS-AUG', 'BQS-SEP', 'BQS-OCT',
                      'BQS-NOV', 'BQS-DEC']:
            my_freq = 'Q'
            my_freq_name = 'Quarterly'
        # YEARLY
        elif freq in ['A', 'AS', 'Y', 'BYS', 'YS', 'A-JAN', 'A-FEB', 'A-MAR', 'A-APR', 'A-MAY', 'A-JUN', 'A-JUL',
                      'A-AUG', 'A-SEP', 'A-OCT', 'A-NOV', 'A-DEC', 'AS-JAN', 'AS-FEB', 'AS-MAR', 'AS-APR', 'AS-MAY',
                      'AS-JUN', 'AS-JUL', 'AS-AUG', 'AS-SEP', 'AS-OCT', 'AS-NOV', 'AS-DEC', 'BA-JAN', 'BA-FEB',
                      'BA-MAR', 'BA-APR', 'BA-MAY', 'BA-JUN', 'BA-JUL', 'BA-AUG', 'BA-SEP', 'BA-OCT', 'BA-NOV',
                      'BA-DEC', 'BAS-JAN', 'BAS-FEB', 'BAS-MAR', 'BAS-APR', 'BAS-MAY', 'BAS-JUN', 'BAS-JUL', 'BAS-AUG',
                      'BAS-SEP', 'BAS-OCT', 'BAS-NOV', 'BAS-DEC']:
            my_freq = 'Y'
            my_freq_name = 'Yearly'
        else:
            # check if missing dates creating gaps, because pandas infer_freq function does not work well then use custom function to determine frequency
            my_freq, my_freq_name = infer_frequency(df[column_name])
        return my_freq, my_freq_name
    except:
        print(f'Error in function "determine_df_frequency()": could not determine frequency of uploaded data.')


def adjust_brightness(hex_color, brightness_factor):
    """
    Adjusts the brightness of a given hex color.

    Args:
        hex_color (str): The hex color code (e.g., '#RRGGBB').
        brightness_factor (float): The factor by which to adjust the brightness.
                                  Values less than 1.0 decrease brightness,
                                  while values greater than 1.0 increase brightness.

    Returns:
        str: The adjusted hex color code.

    """
    # Convert hex color to RGB
    rgb_color = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    # Adjust brightness
    adjusted_rgb_color = tuple(max(0, min(255, int(round(c * brightness_factor)))) for c in rgb_color)
    # Convert RGB back to hex color
    adjusted_hex_color = '#' + ''.join(format(c, '02x') for c in adjusted_rgb_color)
    return adjusted_hex_color














