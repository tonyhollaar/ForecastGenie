# -*- coding: utf-8 -*-
"""
_____ _____ _____ _____ _____ _____ _____ _____ _____ _____
\____\\____\\____\\____\\____\\____\\____\\____\\____\\____\
  ___ ___  ___ ___ ___   _   ___ _____ ___ ___ _  _ ___ ___
 | __/ _ \| _ \ __/ __| /_\ / __|_   _/ __| __| \| |_ _| __|
 | _| (_) |   / _| (__ / _ \\__ \ | || (_ | _|| .` || || _|
 |_| \___/|_|_\___\___/_/ \_\___/ |_| \___|___|_|\_|___|___|
_____ _____ _____ _____ _____ _____ _____ _____ _____ _____
\____\\____\\____\\____\\____\\____\\____\\____\\____\\____\
Streamlit App: ForecastGenie
Forecast y based on X timeseries data
@author: Tony Hollaar
Date: 12/26/2023
Version 2.1
"""
# import standard libraries
import streamlit as st
import os

import itertools
import optuna
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

# SET PAGE CONFIGURATIONS STREAMLIT
st.set_page_config(page_title="ForecastGenie™️",
                   layout="centered",  # "centered" or "wide"
                   page_icon="🕓",
                   initial_sidebar_state="expanded")  # "auto" or "expanded" or "collapsed"

# local Libraries
from functions import *
from models.tune import run_naive_model_1, run_naive_model_2, run_naive_model_3
from models.tune import create_hyperparameter_importance_plot, hyperparameter_tuning_form
from style.animations import show_lottie_animation
from style.text import font_style, my_subheader, my_text_paragraph, my_text_header, my_header, my_title, vertical_spacer
from style.icons import load_icons
from app_pages.about import AboutPage
from app_pages.faq import FaqPage
from app_pages.doc import DocPage
from app_pages.load import LoadPage
from app_pages.explore import ExplorePage
from app_pages.clean import CleanPage
from app_pages.engineer import EngineerPage
from app_pages.prepare import PreparePage
from utils.config import SessionState
from utils.app_navigation import AppNavigation
from utils.logger import setup_logger

# INITIATE GLOBAL VARIABLES
(key1_chart_color, key2_chart_patterns, key3_run,
 metrics_dict, random_state, results_df, custom_fill_value,
 key1_load, key2_load, key3_load,
 key1_explore, key2_explore, key3_explore, key4_explore,
 key1_missing, key2_missing, key3_missing,
 key1_outlier, key2_outlier, key3_outlier, key4_outlier, key5_outlier, key6_outlier, key7_outlier,
 key1_engineer, key2_engineer, key3_engineer, key4_engineer, key5_engineer, key6_engineer, key7_engineer,
 key1_engineer_page_country, key2_engineer_page_country, key3_engineer_page_country,
 key1_prepare_normalization, key2_prepare_standardization,
 key1_select_page_user_selection,
 key1_select_page_pca,
 key1_select_page_rfe, key2_select_page_rfe, key3_select_page_rfe, key4_select_page_rfe, key5_select_page_rfe,
 key6_select_page_rfe,
 key1_select_page_mifs,
 key1_select_page_corr, key2_select_page_corr,
 key1_train, key2_train, key3_train, key4_train, key5_train, key6_train, key7_train, key8_train, key9_train,
 key10_train,
 key11_train, key12_train, key13_train, key14_train, key15_train, key16_train, key17_train, key18_train, key19_train,
 key20_train,
 key21_train, key22_train, key23_train, key24_train, key25_train, key26_train, key27_train, key28_train, key29_train,
 key30_train, key31_train, key32_train,
 key1_evaluate, key2_evaluate) = SessionState.initiate_global_variables()

# LOGGING
logger = setup_logger()
loglevel = os.getenv('LOGLEVEL', 'INFO').upper()  # Set log level from environment variable
logger.setLevel(loglevel)
logger.info('Starting the application...')  # Log application start

# Load Fonts
st.markdown(font_style, unsafe_allow_html=True)

# Load Icons
icons = load_icons()

# Set Logo
st.sidebar.image('./images/forecastgenie_logo.png')

# Create an instance of AppNavigation
app_navigation = AppNavigation()

# Create Menus
menu_item = app_navigation.create_main_menu()
sidebar_menu_item = app_navigation.create_sidebar_menu()

# =============================================================================
# SIDEBAR MENU PAGES
# =============================================================================
if sidebar_menu_item == 'ABOUT':
    about_page = AboutPage()
    about_page.render()

if sidebar_menu_item == 'FAQ':
    faq_page = FaqPage()
    faq_page.render()

if sidebar_menu_item == 'DOC':
    doc_page = DocPage()
    doc_page.render()

# =============================================================================
# INITIATE ALL PAGE CLASSES TO BE ABLE TO CALL METHODS FROM EACH PAGE
# SUCH AS RENDER THE PAGE, UPDATE THE PAGE, ETC.
# =============================================================================
# Instantiate the CleanPage class
clean_page = CleanPage(st.session_state, key1_missing, key2_missing, key3_missing, random_state, key1_outlier,
                       key2_outlier, key3_outlier, key4_outlier, key5_outlier, key6_outlier, key7_outlier)

# Instantiate the EngineerPage class
engineer_page = EngineerPage(st.session_state, key1_engineer, key2_engineer, key3_engineer, key4_engineer,
                             key5_engineer, key6_engineer, key7_engineer, key1_engineer_page_country,
                             key2_engineer_page_country, key3_engineer_page_country)

# =============================================================================
# BELOW RUN PROCEDURES THAT MANIPULATE THE DATAFRAME
# EITHER DEMO DATA OR USER UPLOADED FILE
# REGARDLESS IF USER IS ON A CERTAIN PAGE WITH STANDARD PARAMETERS SET IN CONFIG.PY
# =============================================================================
# Call the perform_data_cleaning method that performs all the data cleaning procedures
clean_page.perform_data_cleaning()

# Call the perform_data_engineering method that performs all the data engineering procedures
engineer_page.perform_data_engineering()

# this df is available and needs to be used in prepare
st.write('dataframes - df', get_state("DATAFRAMES", "df_cleaned_outliers_with_index"))
st.write(f'dataframes - df: {get_state("DATAFRAMES", "df_cleaned_outliers_with_index").shape[0]} rows x {get_state("DATAFRAMES", "df_cleaned_outliers_with_index").shape[1]} columns')
# =============================================================================
# MAIN MENU PAGES
# =============================================================================
if menu_item == 'Load' and sidebar_menu_item == 'HOME':
    load_page = LoadPage(st.session_state)
    load_page.render()

if menu_item == 'Explore' and sidebar_menu_item == 'HOME':
    explore_page = ExplorePage(st.session_state)
    explore_page.render()

if menu_item == 'Clean' and sidebar_menu_item == 'HOME':
    clean_page.render()

if menu_item == 'Engineer' and sidebar_menu_item == 'HOME':
    engineer_page.render()

if menu_item == 'Prepare' and sidebar_menu_item == 'HOME':
    prepare_page = PreparePage(get_state("DATAFRAMES", "df_cleaned_outliers_with_index"),
                               key1_prepare_normalization, key2_prepare_standardization)
    prepare_page.render()

#st.write('dataframes - df', get_state("DATAFRAMES", "df_cleaned_outliers_with_index"))


# else:
#     # =============================================================================
#     # - if user not in prepare screen then update the dataframe with preselected choices e.g. 80/20 split
#     # - do not normalize and do not standardize
#     # =============================================================================
#
#     #################################
#     # 1. REMOVE OBJECT DTYPE FEATURES
#     #################################
#     # remove dtype = object features from dataframe
#     # these are/should be the descriptive columns like holiday description, day of week -> for which default dummy variables are created etc.
#     st.session_state['df'] = remove_object_columns(st.session_state['df'], message_columns_removed=False)
#
#     ##########################
#     # 2. SET DATE AS INDEX COLUMN
#     # 3. TRAIN/TEST SPLIT
#     ##########################
#     st.session_state['insample_forecast_steps'] = round((st.session_state['insample_forecast_perc'] / 100) * len(df))
#
#     if 'date' in st.session_state['df']:
#         # st.write('date is in the session state of dataframe df and will now be set as index') # TEST
#         X, y, X_train, X_test, y_train, y_test, scaler = perform_train_test_split(
#             st.session_state['df'].set_index('date'),
#             st.session_state['insample_forecast_steps'],
#             st.session_state['normalization_choice'],
#             numerical_features=numerical_features)
#     else:
#         X, y, X_train, X_test, y_train, y_test, scaler = perform_train_test_split(st.session_state['df'],
#                                                                                   st.session_state[
#                                                                                       'insample_forecast_steps'],
#                                                                                   st.session_state[
#                                                                                       'normalization_choice'],
#                                                                                   numerical_features=numerical_features)
#
# # =============================================================================
# # Update Session States of dataset including: Target variable (y), independent features (X) for both train & test-set
# # =============================================================================
# st.session_state['X'] = X
# st.session_state['y'] = y
# st.session_state['X_train'] = X_train
# st.session_state['X_test'] = X_test
# st.session_state['y_train'] = y_train
# st.session_state['y_test'] = y_test
#
# # =============================================================================
# # # If user changes engineered features to lower number than default e.g. 5,
# # # then check the number of features in X_train and also check if a user did not set slider to lower number than maximum of e.g. 3 features:
# # =============================================================================
# if st.session_state['X_train'].shape[1] < 5:
#     st.write('number of features in X_train is smaller than default value of 5')
#
#     # =============================================================================
#     # PCA
#     # =============================================================================
#     # if user changed slider set it to that value
#     if get_state("SELECT_PAGE_PCA", "num_features_pca") < st.session_state['X_train'].shape[1]:
#         st.write('number of features is smaller than X_train for pca')
#     else:
#         # PCA UPDATE SESSION STATE
#         set_state("SELECT_PAGE_PCA", ("num_features_pca", min(5, st.session_state['X_train'].shape[1])))
#
#     # =============================================================================
#     # RFE
#     # =============================================================================
#     if get_state("SELECT_PAGE_RFE", "num_features_rfe") < st.session_state['X_train'].shape[1]:
#         st.write('number of features is smaller than X_train for rfe')
#     else:
#         # RFE UPDATE SESSION STATE
#         set_state("SELECT_PAGE_RFE", ("num_features_rfe", min(5, st.session_state['X_train'].shape[1])))
#         # =============================================================================
#         # set_state("SELECT_PAGE_RFE", ("estimator_rfe", "Linear Regression"))
#         # set_state("SELECT_PAGE_RFE", ("timeseriessplit_value_rfe", 5))
#         # set_state("SELECT_PAGE_RFE", ("num_steps_rfe", 1))
#         # =============================================================================
#
#     # =============================================================================
#     # MIFS
#     # =============================================================================
#     if get_state("SELECT_PAGE_MIFS", "num_features_mifs") < st.session_state['X_train'].shape[1]:
#         st.write('number of features is smaller than X_train for mifs')
#     else:
#         # MIFS UPDATE SESSION STATE
#         set_state("SELECT_PAGE_MIFS", ("num_features_mifs", min(5, st.session_state['X_train'].shape[1])))
#
# # =============================================================================
# # IF USER UPLOADS NEW FILE THEN RESET THE FEATURE SELECTION LIST TO EMPTY LIST
# # =============================================================================
# if get_state("DATA_OPTION", "upload_new_data") == True:
#     # set equal to empty list again if user uploads new file
#     set_state("SELECT_PAGE_USER_SELECTION", ("feature_selection_user", []))
#
# # =============================================================================
# #    _____ ______ _      ______ _____ _______
# #   / ____|  ____| |    |  ____/ ____|__   __|
# #  | (___ | |__  | |    | |__ | |       | |
# #   \___ \|  __| | |    |  __|| |       | |
# #   ____) | |____| |____| |___| |____   | |
# #  |_____/|______|______|______\_____|  |_|
# #
# # =============================================================================
# # FEATURE SELECTION OF INDEPENDENT FEATURES / EXOGENOUS VARIABLES
# if menu_item == 'Select' and sidebar_menu_item == 'HOME':
#     # =============================================================================
#     # SHOW USER FORMS FOR PARAMETERS OF FEATURE SELECTION METHODS
#     # =============================================================================
#     with st.sidebar:
#         num_features_rfe, duplicate_ranks_rfe, estimator_rfe, timeseriessplit_value_rfe, num_steps_rfe = form_feature_selection_method_rfe()
#
#         num_features_pca = form_feature_selection_method_pca(X_train=X_train, streamlit_key=key1_select_page_pca)
#
#         num_features_mifs = form_feature_selection_method_mifs(X_train=X_train, streamlit_key=key1_select_page_mifs)
#
#     # define user tabs to seperate SELECT PAGE sections
#     tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
#         ["▫️info", "▫️rfe", "▫️pca", "▫️mi", "▫️pairwise correlation", "▫️feature selection"])
#
#     # =============================================================================
#     # SHOW INFORMATION CARD ABOUT FEATURE SELECTION METHODS
#     # =============================================================================
#     with tab1:
#         try:
#             show_card_feature_selection_methods()
#
#             st.image('./images/feature_info.png')
#         except:
#             pass
#
#     # =============================================================================
#     # RFE Feature Selection
#     # =============================================================================
#     with tab2:
#         try:
#             with st.expander('', expanded=True):
#                 selected_cols_rfe, selected_features, rfecv, user_message = perform_rfe(
#                     X_train=st.session_state['X_train'],
#                     y_train=st.session_state['y_train'],
#                     duplicate_ranks_rfe=duplicate_ranks_rfe,
#                     estimator_rfe=estimator_rfe,
#                     num_steps_rfe=num_steps_rfe,
#                     num_features_rfe=num_features_rfe,
#                     timeseriessplit_value_rfe=timeseriessplit_value_rfe)
#                 show_rfe_plot(rfecv, selected_features)
#                 # note the optimal features to user noted by RFE method
#                 st.write(user_message)
#         except:
#             selected_cols_rfe = []
#             st.error(
#                 '**ForecastGenie Error**: *Recursive Feature Elimination with Cross-Validation* could not execute. Need at least 2 features to be able to apply Feature Elimination. Please adjust your selection criteria.')
#
#     # =============================================================================
#     # PCA Feature Selection
#     # =============================================================================
#     with tab3:
#         try:
#             with st.expander('', expanded=True):
#                 sorted_features, pca, sorted_idx, selected_cols_pca = perform_pca(X_train=X_train,
#                                                                                   num_features_pca=num_features_pca)
#                 show_pca_plot(sorted_features=sorted_features, pca=pca, sorted_idx=sorted_idx,
#                               selected_cols_pca=selected_cols_pca)
#         except:
#             # IF PCA could not execute show user error
#             selected_cols_pca = []
#             st.error(
#                 '**ForecastGenie Error**: *Principal Component Analysis* could not execute. Please adjust your selection criteria.')
#
#     # =============================================================================
#     # Mutual Information Feature Selection
#     # =============================================================================
#     with tab4:
#         try:
#             with st.expander('', expanded=True):
#                 mutual_info, selected_features_mi, selected_cols_mifs = perform_mifs(X_train, y_train,
#                                                                                      num_features_mifs)
#                 show_mifs_plot(mutual_info=mutual_info, selected_features_mi=selected_features_mi,
#                                num_features_mifs=num_features_mifs)
#         except:
#             selected_cols_mifs = []
#             st.warning(
#                 ':red[**ERROR**: Mutual Information Feature Selection could not execute...please adjust your selection criteria]')
#
#     # =============================================================================
#     # Removing Highly Correlated Independent Features
#     # =============================================================================
#     with tab5:
#         try:
#             with st.sidebar:
#                 with st.form('correlation analysis'):
#                     my_text_paragraph('Correlation Analysis')
#
#                     corr_threshold = st.slider("*Select Correlation Threshold*",
#                                                min_value=0.0,
#                                                max_value=1.0,
#                                                key=key1_select_page_corr,
#                                                step=0.05,
#                                                help='Set `Correlation Threshold` to determine which pair(s) of variables in the dataset are strongly correlated e.g. no correlation = 0, perfect correlation = 1')
#
#                     # define models for computing importance scores of highly correlated independent features
#                     models = {'Linear Regression': LinearRegression(),
#                               'Random Forest Regressor': RandomForestRegressor(n_estimators=100)}
#
#                     # provide user option to select model (default = Linear Regression)
#                     selected_corr_model = st.selectbox(
#                         label='*Select **model** for computing **importance scores** for highly correlated feature pairs, to drop the **least important** feature of each pair which is highly correlated*:',
#                         options=list(models.keys()),
#                         key=key2_select_page_corr)
#
#                     col1, col2, col3 = st.columns([4, 4, 4])
#                     with col2:
#                         corr_btn = st.form_submit_button("Submit", type="secondary", on_click=form_update,
#                                                          args=('SELECT_PAGE_CORR',))
#
#                         # if button is pressed
#                         if corr_btn:
#                             set_state("SELECT_PAGE_BTN_CLICKED", ('correlation_btn', True))
#
#             with st.expander('', expanded=True):
#                 vertical_spacer(1)
#
#                 my_text_paragraph('Pairwise Correlation', my_font_size='26px')
#
#                 col1, col2, col3 = st.columns([5, 3, 5])
#                 with col2:
#                     st.caption(f'with threshold >={corr_threshold * 100:.0f}%')
#
#                 ################################################################
#                 # PLOT HEATMAP WITH PAIRWISE CORRELATION OF INDEPENDENT FEATURES.
#                 ################################################################
#                 # Generate correlation heatmap for independent features based on threshold from slider set by user e.g. default to 0.8
#                 correlation_heatmap(X_train, correlation_threshold=corr_threshold)
#
#                 # Apply Function to analyse feature correlations and computes importance scores.
#                 total_features, importance_scores, pairwise_features_in_total_features, df_pairwise = analyze_feature_correlations(
#                     selected_corr_model,
#                     X_train,
#                     y_train,
#                     selected_cols_rfe,
#                     selected_cols_pca,
#                     selected_cols_mifs,
#                     corr_threshold,
#                     models)
#                 # =============================================================================
#                 # CHECK IF USER HAS SELECTED ANY FEATURES - IF ALL 3 SLIDERS ARE SET TO 0 FOR RFE, PCA, MIFS                                                                                                                               models)
#                 # =============================================================================
#                 if not total_features:
#                     st.info(
#                         f'**NOTE**: please adjust your selection criteria as the list of features is currently empty!')
#                     st.stop()
#
#                 # Display message with pairs in total_features
#                 if df_pairwise.empty:
#                     st.info(
#                         f'There are no **pairwise combinations** in the selected features with a **correlation** larger than or equal to the user defined threshold of **{corr_threshold * 100:.0f}%**')
#                     vertical_spacer(1)
#                 else:
#                     my_text_paragraph(
#                         f'The below pairwise combinations of features have a correlation >= {corr_threshold * 100:.0f}% threshold:',
#                         my_font_size='16px')
#                     vertical_spacer(1)
#
#                     # show dataframe with pairwise features
#                     st.dataframe(df_pairwise, use_container_width=True)
#
#                     # download button for dataframe of pairwise correlation
#                     download_csv_button(df_pairwise, my_file="pairwise_correlation.csv",
#                                         help_message='Download the pairwise correlations to .CSV', set_index=False)
#
#                     # insert divider line
#                     st.markdown('---')
#
#                 # SHOW CORRELATION CHART
#                 altair_correlation_chart(total_features, importance_scores, pairwise_features_in_total_features,
#                                          corr_threshold)
#
#                 # Remove features with lowest importance scores from each pair
#                 # note: only remove one of highly correlated features based on importance score after plot is shown
#                 total_features_default = remove_lowest_importance_feature(total_features, importance_scores,
#                                                                           pairwise_features_in_total_features)
#
#         except:
#             total_features_default = ['date_numeric']
#             st.warning(
#                 ':red[**ERROR**]: could not execute the correlation analysis...please adjust your selection criteria.')
#
#     # =============================================================================
#     # Top Features
#     # =============================================================================
#     with st.sidebar:
#         with st.form('top_features'):
#             my_text_paragraph('Selected Features')
#
#             # if rfe pca, mifs or correlation SUBMIT button is pressed,
#             # or if on ENGINEER PAGE button is pressed in sidebar (when e.g. features are removed or added)
#             # set total_features equal again to recommended/calculated total_features from feature selection methods
#             # if user changes one of the selection method parameters -> reset to feature selection method selection and remove user feature selection preference
#             if get_state("SELECT_PAGE_BTN_CLICKED", "rfe_btn") == True \
#                     or get_state("SELECT_PAGE_BTN_CLICKED", "pca_btn") == True \
#                     or get_state("SELECT_PAGE_BTN_CLICKED", "mifs_btn") == True \
#                     or get_state("SELECT_PAGE_BTN_CLICKED", "correlation_btn") == True \
#                     or get_state("ENGINEER_PAGE_FEATURES_BTN", "engineering_form_submit_btn") == True:
#
#                 st.write('option 1')  # TEST1
#
#                 # reset state (back to False)
#                 set_state("SELECT_PAGE_BTN_CLICKED", ("rfe_btn", False))
#                 set_state("SELECT_PAGE_BTN_CLICKED", ("pca_btn", False))
#                 set_state("SELECT_PAGE_BTN_CLICKED", ("mifs_btn", False))
#                 set_state("SELECT_PAGE_BTN_CLICKED", ("correlation_btn", False))
#                 set_state("ENGINEER_PAGE_FEATURES_BTN", ("engineering_form_submit_btn", False))
#
#                 # deleting states set with fire-state package doesn't work need to use set_state to my knowledge
#                 # =============================================================================
#                 #                 # delete session states as it still otherwise evaluates to True
#                 #                 del st.session_state["__SELECT_PAGE_BTN_CLICKED-rfe_btn__"]
#                 #                 del st.session_state["__SELECT_PAGE_BTN_CLICKED-pca_btn__"]
#                 #                 del st.session_state["__SELECT_PAGE_BTN_CLICKED-mifs_btn__"]
#                 #                 del st.session_state["__SELECT_PAGE_BTN_CLICKED-correlation_btn__"]
#                 # =============================================================================
#
#                 # =============================================================================
#                 #                 st.write('rfe_btn', get_state("SELECT_PAGE_BTN_CLICKED", "rfe_btn"))   # TEST1
#                 #                 st.write('pca_btn', get_state("SELECT_PAGE_BTN_CLICKED", "pca_btn"))   # TEST1
#                 #                 st.write('mifs_btn', get_state("SELECT_PAGE_BTN_CLICKED", "mifs_btn")) # TEST1
#                 #                 st.write('correlation_btn', get_state("SELECT_PAGE_BTN_CLICKED", "correlation_btn")) # TEST1
#                 # =============================================================================
#
#                 # set default feature selection
#                 total_features = total_features_default
#
#                 # set features equal to default (based on 3 feature selection methods)
#                 set_state("SELECT_PAGE_USER_SELECTION", ("feature_selection_user", total_features))
#
#             # if user selected features, use those! else use the recommended features by feature selection methods
#             elif "__SELECT_PAGE_USER_SELECTION-feature_selection_user__" in st.session_state and get_state(
#                     "SELECT_PAGE_USER_SELECTION", "feature_selection_user") != []:
#                 st.write('option 2')  # TEST
#                 total_features = get_state("SELECT_PAGE_USER_SELECTION", "feature_selection_user")
#
#             else:
#                 st.write('option 3')  # TEST
#                 # set feature selection to default based result of 3 feature selection methods)
#
#                 total_features = total_features_default
#
#                 total_features = set_state("SELECT_PAGE_USER_SELECTION", ("feature_selection_user", total_features))
#
#             feature_selection_user = st.multiselect(label="favorite features",
#                                                     options=X_train.columns,
#                                                     key=key1_select_page_user_selection,
#                                                     label_visibility="collapsed")
#
#             col1, col2, col3 = st.columns([4, 4, 4])
#             with col2:
#                 top_features_btn = st.form_submit_button("Submit", type="secondary", on_click=form_update,
#                                                          args=("SELECT_PAGE_USER_SELECTION",))
#
#     # IS THIS CODE REDUNDANT? I CAN DEFINE IT OUTSIDE OF SELECT PAGE WITH SESSION STATES
#     ######################################################################################################
#     # Redefine dynamic user picked features for X, y, X_train, X_test, y_train, y_test
#     ######################################################################################################
#     # Apply columns user selected to X
#     X = X.loc[:, feature_selection_user]
#     y = y
#     X_train = X[:(len(df) - st.session_state['insample_forecast_steps'])]
#     X_test = X[(len(df) - st.session_state['insample_forecast_steps']):]
#
#     # set endogenous variable train/test split
#     y_train = y[:(len(df) - st.session_state['insample_forecast_steps'])]
#     y_test = y[(len(df) - st.session_state['insample_forecast_steps']):]
#
#     with tab6:
#         with st.expander('🥇 Top Features Selected', expanded=True):
#             my_text_paragraph('Your Feature Selection', my_font_size='26px')
#
#             # =============================================================================
#             #             show_lottie_animation(url = "./images/astronaut_star_in_hand.json",
#             #                                   key = 'lottie_animation_austronaut_star',
#             #                                   width=200,
#             #                                   height=200,
#             #                                   col_sizes=[5,4,5],
#             #                                   margin_before=1,
#             #                                   margin_after=2)
#             # =============================================================================
#
#             # TEST - show recommended features from 3 selection methods
#             # st.write(selected_cols_rfe, selected_cols_pca, selected_cols_mifs) # TEST1
#
#             df_total_features = pd.DataFrame(feature_selection_user, columns=['Top Features'])
#
#             # Create the rating column and update it based on the presence in the lists
#             df_total_features['rating'] = df_total_features['Top Features'].apply(lambda feature: '⭐' * (
#                     int(feature in selected_cols_rfe) + int(feature in selected_cols_pca) + int(
#                 feature in selected_cols_mifs)) if any(
#                 feature in lst for lst in [selected_cols_rfe, selected_cols_pca, selected_cols_mifs]) else '-')
#
#             # Create boolean columns 'rfe', 'pca', and 'mi' based on the presence of features in each list
#             df_total_features['rfe'] = df_total_features['Top Features'].apply(
#                 lambda feature: feature in selected_cols_rfe)
#             df_total_features['pca'] = df_total_features['Top Features'].apply(
#                 lambda feature: feature in selected_cols_pca)
#             df_total_features['mi'] = df_total_features['Top Features'].apply(
#                 lambda feature: feature in selected_cols_mifs)
#
#             # =============================================================================
#             # THIS CODE RETRIEVES DIFFERENCE WHAT WAS RECOMMENDED BY FEATURE SELECTION METHODS
#             # AND WHAT USER ADDS AS ADDITIONAL VARIABLES OUTSIDE OF RECOMMENDATION
#             # Note: first time app starts the total_features is None
#             # =============================================================================
#             if total_features is not None:
#                 difference_lsts = list(set(feature_selection_user) - set(total_features_default))
#             else:
#                 # if no features chosen by user then return empty list -> no cells need to be highlighted yellow
#                 difference_lsts = []
#
#
#             def highlight_cols_feature_selection(row):
#                 """
#                 A function that highlights the cells of a DataFrame based on their values.
#
#                 Args:
#                 row: A pandas Series representing a row of the DataFrame.
#
#                 Returns:
#                 list: A list of CSS styles to be applied to the row cells.
#                 """
#                 styles = ['background-color: #f0f6ff' if row['Top Features'] in difference_lsts else '' for _ in row]
#                 return styles
#
#
#             # Apply the custom function to highlight rows that have features in difference_lsts, including the 'rating' column
#             styled_df = df_total_features.style.apply(highlight_cols_feature_selection, axis=1)
#
#             # show dataframe with top features + user selection in Streamlit
#             st.dataframe(styled_df, use_container_width=True)
#
#             ###
#
#             # Display the dataframe with independent features (X) and values in Streamlit
#             st.dataframe(X, use_container_width=True)
#
#             # Create download button for forecast results to .CSV
#             download_csv_button(X, my_file="features_dataframe.csv",
#                                 help_message="Download your **features** to .CSV",
#                                 my_key='features_df_download_btn')
#
#         st.image('./images/feature_selection2.png')
#     # Set session state variable 'upload_new_data' back to False
#     set_state("DATA_OPTION", ("upload_new_data", False))
# else:
#     # =============================================================================
#     # ELSE WHEN USER IS NOT ON SELECT PAGE:
#     # 0. RUN THE FEATURE SELECTION TO SELECT TOP FEATURES
#     # 1. REMOVE FROM HIGHLY CORRELATED INDEPENDENT FEATURES PAIRS THE ONE WITH LOWEST IMPORTANCE SCORES
#     # =============================================================================
#
#     # =============================================================================
#     # Apply 3 Feature Selection Techniques
#     # =============================================================================
#     ################
#     # 1. RFE
#     ################
#     try:
#         selected_cols_rfe, selected_features, rfecv, message = perform_rfe(X_train=st.session_state['X_train'],
#                                                                            y_train=st.session_state['y_train'],
#                                                                            estimator_rfe=get_state("SELECT_PAGE_RFE",
#                                                                                                    "estimator_rfe"),
#                                                                            duplicate_ranks_rfe=get_state(
#                                                                                "SELECT_PAGE_RFE",
#                                                                                "duplicate_ranks_rfe"),
#                                                                            num_steps_rfe=get_state("SELECT_PAGE_RFE",
#                                                                                                    "num_steps_rfe"),
#                                                                            num_features_rfe=get_state("SELECT_PAGE_RFE",
#                                                                                                       "num_features_rfe"),
#                                                                            timeseriessplit_value_rfe=get_state(
#                                                                                "SELECT_PAGE_RFE",
#                                                                                "timeseriessplit_value_rfe"))
#     except:
#         selected_cols_rfe = []
#
#     ################
#     # 2. PCA
#     ################
#     try:
#         sorted_features, pca, sorted_idx, selected_cols_pca = perform_pca(X_train=st.session_state['X_train'],
#                                                                           num_features_pca=get_state("SELECT_PAGE_PCA",
#                                                                                                      "num_features_pca"))
#     except:
#         selected_cols_pca = []
#     ################
#     # 3. MIFS
#     ################
#     try:
#         mutual_info, selected_features_mi, selected_cols_mifs = perform_mifs(X_train=st.session_state['X_train'],
#                                                                              y_train=st.session_state['y_train'],
#                                                                              num_features_mifs=get_state(
#                                                                                  "SELECT_PAGE_MIFS",
#                                                                                  "num_features_mifs"))
#     except:
#         selected_cols_mifs = []
#
#     # =============================================================================
#     # Remove Highly Correlated Features >= threshold ( default = 80% pairwise-correlation)
#     # remove one of each pair based on importance scores
#     # =============================================================================
#     total_features, importance_scores, pairwise_features_in_total_features, df_pairwise = analyze_feature_correlations(
#         selected_corr_model=get_state("SELECT_PAGE_CORR", "selected_corr_model"),
#         X_train=X_train,
#         y_train=y_train,
#         selected_cols_rfe=selected_cols_rfe,
#         selected_cols_pca=selected_cols_pca,
#         selected_cols_mifs=selected_cols_mifs,
#         models={'Linear Regression': LinearRegression(),
#                 'Random Forest Regressor': RandomForestRegressor(n_estimators=100)},
#         corr_threshold=get_state("SELECT_PAGE_CORR", "corr_threshold"))
#     # NOTE: only remove one of highly correlated features based on importance score after plot is shown
#     total_features_default = remove_lowest_importance_feature(total_features, importance_scores,
#                                                               pairwise_features_in_total_features)
#
#     # if user selected features, use those! else use the recommended features by feature selection methods
#     if "__SELECT_PAGE_USER_SELECTION-feature_selection_user__" in st.session_state and get_state(
#             "SELECT_PAGE_USER_SELECTION", "feature_selection_user") != []:
#         total_features = get_state("SELECT_PAGE_USER_SELECTION", "feature_selection_user")
#     else:
#         total_features = total_features_default
#
#     X = X.loc[:, total_features]
#     y = y
#     X_train = X[:(len(df) - st.session_state['insample_forecast_steps'])]
#     X_test = X[(len(df) - st.session_state['insample_forecast_steps']):]
#     y_train = y[:(len(df) - st.session_state['insample_forecast_steps'])]
#     y_test = y[(len(df) - st.session_state['insample_forecast_steps']):]
#
# # =============================================================================
# #   _______ _____            _____ _   _
# #  |__   __|  __ \     /\   |_   _| \ | |
# #     | |  | |__) |   /  \    | | |  \| |
# #     | |  |  _  /   / /\ \   | | | . ` |
# #     | |  | | \ \  / ____ \ _| |_| |\  |
# #     |_|  |_|  \_\/_/    \_\_____|_| \_|
# #
# # =============================================================================
# # TRAIN MODELS
# if menu_item == 'Train' and sidebar_menu_item == 'HOME':
#     # =============================================================================
#     # USER FORM: TRAIN MODELS (MODEL CHECKBOXES, PARAMETERS AND HYPERPARAMETERS)
#     # =============================================================================
#     with st.sidebar:
#         my_title(f"""{icons["train_icon"]}""", "#3b3b3b")
#         with st.form('model_train_form'):
#
#             vertical_spacer(1)
#
#             my_text_paragraph('Settings')
#             vertical_spacer(2)
#
#             col1, col2, col3, col4, col5 = st.columns([1, 7, 1, 7, 1])
#             with col2:
#                 include_feature_selection = st.selectbox(label='*include feature selection:*',
#                                                          options=['Yes', 'No'],
#                                                          key=key28_train,
#                                                          help='''if `feature selection` is enabled, the explanatory variables or features will be incorporated as additional inputs during the model training process. These features can be either the default recommendations based on the feature selection process (refer to the Select Page for more details), or if you have made changes to the selection, it will reflect the features you have chosen to include.
#                                                                    \nBy enabling feature selection, you allow the model to utilize these explanatory features with the goal to enhance its predictive capabilities to improve the overall performance of the model(s).''')
#             with col4:
#                 # Generic Graph Settings
#                 my_conf_interval = st.slider(label="*confidence interval (%)*",
#                                              min_value=1,
#                                              max_value=99,
#                                              step=1,
#                                              key=key1_train,
#                                              help='a **`confidence interval`** is a range of values around a sample statistic, such as a mean or proportion, which is likely to contain the true population parameter with a certain degree of confidence.\
#                                                     The level of confidence is typically expressed as a percentage, such as 95%, and represents the probability that the true parameter lies within the interval.\
#                                                     A wider interval will generally have a higher level of confidence, while a narrower interval will have a lower level of confidence.')
#             vertical_spacer(2)
#
#             my_text_paragraph('Model Selection')
#
#             # *****************************************************************************
#             # 1. NAIVE MODELS
#             # *****************************************************************************
#             naive_model_checkbox = st.checkbox(label='Naive Models',
#                                                key=key2_train)
#
#             # *****************************************************************************
#             # NAIVE MODELS PARAMETERS
#             # *****************************************************************************
#             custom_lag_value = None  # initiate custom lag value
#
#             with st.expander('◾'):
#                 # =============================================================================
#                 # 1.1 NAIVE MODEL I: LAG / CUSTOM LAG FOR NAIVE MODEL
#                 # ============================================================================
#                 my_text_paragraph('Naive Model I: Lag')
#
#                 col1, col2, col3 = st.columns([1, 2, 1])
#                 with col2:
#                     lag = st.selectbox(label='*select seasonal lag*',
#                                        options=['Day', 'Week', 'Month', 'Year', 'Custom'],
#                                        key=key7_train,
#                                        label_visibility='visible')
#
#                     lag = lag.lower()  # make lag name lowercase e.g. 'Week' becomes 'week'
#
#                     if lag == 'custom':
#                         custom_lag_value = st.number_input(
#                             label="*if seasonal **lag** set to Custom, please set lag value (in days):*",
#                             step=1,
#                             key=key8_train)
#
#                         custom_lag_value = custom_lag_value if custom_lag_value is not None else None
#
#                 # =============================================================================
#                 # 1.2 NAIVE MODEL II: ROLLING WINDOW
#                 # =============================================================================
#                 vertical_spacer(1)
#                 my_text_paragraph('Naive Model II: Rolling Window')
#                 col1, col2, col3 = st.columns([1, 2, 1])
#                 with col2:
#                     size_rolling_window_naive_model = st.number_input(label='*select rolling window size:*',
#                                                                       min_value=1,
#                                                                       max_value=None,
#                                                                       step=1,
#                                                                       format='%i',
#                                                                       key=key29_train)
#
#                     agg_method_rolling_window_naive_model = st.selectbox(label='*select aggregation method:*',
#                                                                          options=['Mean', 'Median', 'Mode'],
#                                                                          key=key30_train)
#
#                 # =============================================================================
#                 # 1.3 NAIVE MODEL III: Straight Line (Mean, Median, Mode)
#                 # =============================================================================
#                 vertical_spacer(1)
#                 my_text_paragraph('Naive Model III: Constant Value')
#
#                 col1, col2, col3 = st.columns([1, 2, 1])
#                 with col2:
#                     agg_method_baseline_naive_model = st.selectbox(label='*select aggregation method:*',
#                                                                    options=['Mean', 'Median', 'Mode'],
#                                                                    key=key31_train)
#                 vertical_spacer(1)
#
#             # *****************************************************************************
#             # 2. LINEAR REGRESSION MODEL
#             # *****************************************************************************
#             linreg_checkbox = st.checkbox('Linear Regression',
#                                           key=key3_train)
#
#             # *****************************************************************************
#             # 3. SARIMAX MODEL
#             # *****************************************************************************
#             sarimax_checkbox = st.checkbox('SARIMAX',
#                                            key=key4_train)
#
#             # *****************************************************************************
#             # 3.1 SARIMAX MODEL PARAMETERS
#             # *****************************************************************************
#             with st.expander('◾', expanded=False):
#                 col1, col2, col3 = st.columns([5, 1, 5])
#                 with col1:
#                     p = st.number_input(label="Autoregressive Order (p):",
#                                         min_value=0,
#                                         max_value=10,
#                                         key=key9_train)
#
#                     d = st.number_input(label="Differencing (d):",
#                                         min_value=0,
#                                         max_value=10,
#                                         key=key10_train)
#
#                     q = st.number_input(label="Moving Average (q):",
#                                         min_value=0,
#                                         max_value=10,
#                                         key=key11_train)
#                 with col3:
#                     P = st.number_input(label="Seasonal Autoregressive Order (P):",
#                                         min_value=0,
#                                         max_value=10,
#                                         key=key12_train)
#
#                     D = st.number_input(label="Seasonal Differencing (D):",
#                                         min_value=0,
#                                         max_value=10,
#                                         key=key13_train)
#
#                     Q = st.number_input(label="Seasonal Moving Average (Q):",
#                                         # value=1,
#                                         min_value=0,
#                                         max_value=10,
#                                         key=key14_train)
#
#                     s = st.number_input(label="Seasonal Periodicity (s):",
#                                         min_value=1,
#                                         key=key15_train,
#                                         help='`seasonal periodicity` i.e. **$s$** in **SARIMAX** refers to the **number of observations per season**.\
#                                                \n\nfor example, if we have daily data with a weekly seasonal pattern, **s** would be 7 because there are 7 days in a week.\
#                                                similarly, for monthly data with an annual seasonal pattern, **s** would be 12 because there are 12 months in a year.\
#                                                here are some common values for **$s$**:\
#                                                \n- **Daily data** with **weekly** seasonality: **$s=7$** \
#                                                \n- **Monthly data** with **quarterly** seasonality: **$s=3$**\
#                                                \n- **Monthly data** with **yearly** seasonality: **$s=12$**\
#                                                \n- **Quarterly data** with **yearly** seasonality: **$s=4$**')
#
#                 col1, col2, col3 = st.columns([5, 1, 5])
#                 with col1:
#                     # Add a selectbox for selecting enforce_stationarity
#                     enforce_stationarity = st.selectbox(label='Enforce Stationarity',
#                                                         options=[True, False],
#                                                         key=key16_train)
#                 with col3:
#                     # Add a selectbox for selecting enforce_invertibility
#                     enforce_invertibility = st.selectbox(label='Enforce Invertibility',
#                                                          options=[True, False],
#                                                          key=key17_train)
#
#             # *****************************************************************************
#             # 4 PROPHET MODEL
#             # *****************************************************************************
#             prophet_checkbox = st.checkbox('Prophet',
#                                            key=key5_train)
#
#             # *****************************************************************************
#             # 4.1 PROPHET MODEL PARAMETERS
#             # *****************************************************************************
#             with st.expander('◾', expanded=False):
#                 # used to have int(st.slider) -> not needed? # TEST
#                 horizon_option = st.slider(label='Set Forecast Horizon (default = 30 Days):',
#                                            min_value=1,
#                                            max_value=365,
#                                            step=1,
#                                            # value = 30,
#                                            key=key18_train,
#                                            help='The horizon for a Prophet model is typically set to the number of time periods that you want to forecast into the future. This is also known as the forecasting horizon or prediction horizon.')
#
#                 changepoint_prior_scale = st.slider(label="changepoint_prior_scale",
#                                                     min_value=0.001,
#                                                     max_value=1.0,
#                                                     # value = 0.05,
#                                                     step=0.01,
#                                                     key=key19_train,
#                                                     help='This is probably the most impactful parameter. It determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints. As described in this documentation, if it is too small, the trend will be underfit and variance that should have been modeled with trend changes will instead end up being handled with the noise term. If it is too large, the trend will overfit and in the most extreme case you can end up with the trend capturing yearly seasonality. The default of 0.05 works for many time series, but this could be tuned; a range of [0.001, 0.5] would likely be about right. Parameters like this (regularization penalties; this is effectively a lasso penalty) are often tuned on a log scale.')
#
#                 # used to be str(selectbox) -> not needed? # TEST
#                 seasonality_mode = st.selectbox(label="seasonality_mode",
#                                                 options=["additive", "multiplicative"],
#                                                 # index = 1,
#                                                 key=key20_train)
#
#                 seasonality_prior_scale = st.slider(label="seasonality_prior_scale",
#                                                     min_value=0.010,
#                                                     max_value=10.0,
#                                                     # value = 1.0,
#                                                     step=0.1,
#                                                     key=key21_train)
#
#                 # retrieve from the session state the country code to display to user what country holidays are currently added to Prophet Model if set to TRUE
#                 country_holidays = st.selectbox(
#                     label=f'country_holidays ({get_state("ENGINEER_PAGE_COUNTRY_HOLIDAY", "country_code")})',
#                     options=[True, False],
#                     key=key27_train,
#                     help="""Setting this to `True` adds Country Specific Holidays - whereby Prophet assigns each holiday a specific weight""")
#
#                 holidays_prior_scale = st.slider(label="holidays_prior_scale",
#                                                  min_value=0.010,
#                                                  max_value=10.0,
#                                                  # value = 1.0,
#                                                  step=0.1,
#                                                  key=key22_train)
#
#                 yearly_seasonality = st.selectbox(label="yearly_seasonality",
#                                                   options=[True, False],
#                                                   # index = 0,
#                                                   key=key23_train)
#
#                 weekly_seasonality = st.selectbox(label="weekly_seasonality",
#                                                   options=[True, False],
#                                                   # index = 0,
#                                                   key=key24_train)
#
#                 daily_seasonality = st.selectbox(label="daily_seasonality",
#                                                  options=[True, False],
#                                                  # index = 0,
#                                                  key=key25_train)
#
#                 interval_width = int(my_conf_interval / 100)
#
#             col1, col2, col3 = st.columns([4, 4, 4])
#             with col2:
#                 # submit button of user form for training models and updating session states on click utilizing custom package fire-state
#                 train_models_btn = st.form_submit_button(label="Submit",
#                                                          type="secondary",
#                                                          on_click=form_update,
#                                                          args=('TRAIN_PAGE',)
#                                                          )
#                 # update session_state
#                 st.session_state['train_models_btn'] = train_models_btn
#
#         # *****************************************************************************
#         # ALL MODELS
#         # *****************************************************************************
#         # Define all models you want user to choose from (<model_name>, <actual model to be called when fitting the model>)
#         models = [('Naive Models', None),
#                   ('Linear Regression', LinearRegression(fit_intercept=True)),
#                   ('SARIMAX', SARIMAX(y_train)),
#                   ('Prophet', Prophet())]
#
#         # initiate empty list to hold user's checked model names and models from selectboxes
#         selected_models = []
#
#         # if user pressed submit button train the models else do not e.g. every time you go back to train page dont automatically kick off the training
#         if st.session_state['train_models_btn']:
#
#             for model_name, model in models:
#
#                 # Add model name and model if checkbox of model is checked into a list
#                 if model_name == 'Naive Models' and naive_model_checkbox == True:
#                     selected_models.append((model_name, model))
#
#                 # Add model name and model if checkbox of model is checked into a list
#                 if model_name == 'Linear Regression' and linreg_checkbox == True:
#                     selected_models.append((model_name, model))
#
#                 # add model name and model if checkbox of model is checked into a list
#                 if model_name == 'SARIMAX' and sarimax_checkbox == True:
#                     selected_models.append((model_name, model))
#
#                 # add model name and model if checkbox of model is checked into a list
#                 if model_name == 'Prophet' and prophet_checkbox == True:
#                     selected_models.append((model_name, model))
#
#                     # add the user selected trained models to the session state
#     set_state("TRAIN_PAGE", ("selected_models", selected_models))
#
#     # =============================================================================
#     # CREATE TABS
#     # =============================================================================
#     tab1, tab2, tab3, tab4 = st.tabs(['▫️info', '▫️inputs', '▫️outputs', '▫️model details'])
#
#     with tab1:
#         with st.expander('', expanded=True):
#             col1, col2, col3 = st.columns([1, 3, 1])
#             with col2:
#                 train_models_carousel(my_title='Select your models to train!')
#
#         st.image('./images/train_page_info.png')
#
#     with tab2:
#         with st.expander('', expanded=True):
#             # Show buttons with Training Data/Test Data
#             show_model_inputs()
#
#     with tab3:
#         # if no models are selected by user show message
#         if selected_models == []:
#             with st.expander('', expanded=True):
#                 my_text_header('Model Outputs')
#
#                 col1, col2, col3 = st.columns([7, 120, 1])
#                 with col2:
#                     create_flipcard_model_input(image_path_front_card='./images/train_info.png',
#                                                 my_string_back_card='If you train your models, the output will show up here! You can use the checkboxes from the sidebar menu to select your model(s) and hit <i style="color:#9d625e;">\"Submit\"</i> to check out your test results!')
#
#                 vertical_spacer(4)
#
#         # if models are selected by user run code
#         if selected_models:
#
#             # iterate over all models and if user selected checkbox for model the model(s) is/are trained
#             for model_name, model in selected_models:
#                 # =============================================================================
#                 # FOR RESULTS DATAFRAME - CHECK WHICH FEATURES ARE USED
#                 # IF NO FEATURES ARE USED THE DATE AS NUMERIC VALUES (INTEGERS 1,2,3 ... N are used)
#                 # =============================================================================
#                 # create a list of independent variables selected by user prior used
#                 # for results dataframe when evaluating models which variables were included.
#                 if get_state("TRAIN_PAGE", "include_feature_selection") == "Yes":
#                     features_str = get_feature_list(X)
#                 elif get_state("TRAIN_PAGE", "include_feature_selection") == "No":
#                     features_str = 'date_numeric'
#
#                 # =============================================================================
#                 #  NAIVE MODELS
#                 # =============================================================================
#                 try:
#                     if model_name == "Naive Models":
#                         with st.expander('📈' + model_name, expanded=True):
#                             # =============================================================================
#                             # Naive Model I
#                             # =============================================================================
#                             # set model name
#                             model_name = 'Naive Model I'
#
#                             df_preds_naive_model1 = forecast_naive_model1_insample(y_test,
#                                                                                    lag=lag,
#                                                                                    custom_lag_value=custom_lag_value)
#
#                             # define for subtitle dictionary to tell user what lag frequency was chosen
#                             lag_word = {'day': 'daily', 'week': 'weekly', 'month': 'monthly', 'year': 'yearly'}.get(lag)
#
#                             # Show metrics on model card
#                             display_my_metrics(df_preds_naive_model1,
#                                                model_name="Naive Model I",
#                                                my_subtitle=f'custom lag: {custom_lag_value}' if custom_lag_value is not None else f'{lag_word} lag')
#
#                             # Plot graph with actual versus insample predictions
#                             plot_actual_vs_predicted(df_preds_naive_model1, my_conf_interval)
#
#                             # Show the dataframe
#                             st.dataframe(df_preds_naive_model1.style.format(
#                                 {'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Error': '{:.2f}', 'Error (%)': '{:.2%}'}),
#                                 use_container_width=True)
#
#                             # Create download button for forecast results to .csv
#                             download_csv_button(df_preds_naive_model1,
#                                                 my_file="insample_forecast_naivemodel_results.csv",
#                                                 help_message="Download your Naive Model I results to .CSV",
#                                                 my_key='naive_trained_modeli_download_btn',
#                                                 set_index=True)
#
#                             # =============================================================================
#                             # SAVE TEST RESULTS FOR EVALUATION PAGE BY ADDING ROW TO RESULTS_DF
#                             # =============================================================================
#                             metrics_dict = my_metrics(df_preds_naive_model1, model_name="Naive Model I")
#
#                             # Retrieve the feature for Naive Model for results_df 'features' column
#                             lag_int = {'day': '1', 'week': '7', 'month': '30', 'year': '-365'}.get(lag)
#
#                             naive_model1_feature_str = f"t-{custom_lag_value}" if custom_lag_value else f"yt-{lag_int}"
#
#                             # Create new row with test result details
#                             new_row = {'model_name': model_name,
#                                        'mape': '{:.2%}'.format(metrics_dict[model_name]['mape']),
#                                        'smape': '{:.2%}'.format(metrics_dict[model_name]['smape']),
#                                        'rmse': round(metrics_dict[model_name]['rmse'], 2),
#                                        'r2': round(metrics_dict[model_name]['r2'], 2),
#                                        'features': naive_model1_feature_str,
#                                        'model_settings': f"lag: {lag}",
#                                        'predicted': np.ravel(df_preds_naive_model1['Predicted'])
#                                        }
#
#                             # Turn new row into a dataframe
#                             new_row_df = pd.DataFrame([new_row])
#
#                             # Concatenate new row DataFrame with results_df
#                             results_df = pd.concat([results_df, new_row_df], ignore_index=True)
#
#                             # update session state with latest results with new row added
#                             set_state("TRAIN_PAGE", ("results_df", results_df))
#
#                             # =============================================================================
#                             # Naive Model II
#                             # =============================================================================
#                             # set model name
#                             model_name = 'Naive Model II'
#
#                             st.markdown('---')
#
#                             df_preds_naive_model2 = forecast_naive_model2_insample(y_test,
#                                                                                    size_rolling_window_naive_model,
#                                                                                    agg_method_rolling_window_naive_model)
#                             # Show metrics on model card
#                             display_my_metrics(df_preds_naive_model2,
#                                                model_name='Naive Model II',
#                                                my_subtitle=f'{agg_method_rolling_window_naive_model.lower()} rolling window of {size_rolling_window_naive_model}')
#
#                             # Plot graph with actual versus insample predictions
#                             plot_actual_vs_predicted(df_preds_naive_model2, my_conf_interval)
#
#                             # Show the dataframe
#                             st.dataframe(df_preds_naive_model2.style.format(
#                                 {'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Error': '{:.2f}', 'Error (%)': '{:.2%}'}),
#                                 use_container_width=True)
#
#                             # Create download button for forecast results to .csv
#                             download_csv_button(df_preds_naive_model2,
#                                                 my_file="insample_forecast_naivemodel_results.csv",
#                                                 help_message="Download your Naive Model II results to .CSV",
#                                                 my_key='naive_trained_modelii_download_btn',
#                                                 set_index=True)
#
#                             # =============================================================================
#                             # SAVE TEST RESULTS FOR EVALUATION PAGE BY ADDING ROW TO RESULTS_DF
#                             # =============================================================================
#                             metrics_dict = my_metrics(df_preds_naive_model2, model_name=model_name)
#
#                             # Get a scientific notation for the rolling window (Note: Latex not supported in
#                             # Streamlit st.dataframe() so not currently possible)
#                             if agg_method_rolling_window_naive_model == 'Mean':
#                                 naive_model2_feature_str = f"(1/{size_rolling_window_naive_model}) * ∑yt-1, ..., yt-{size_rolling_window_naive_model}"
#                             elif agg_method_rolling_window_naive_model == 'Median':
#                                 naive_model2_feature_str = f"median(yt-1, ..., yt-{size_rolling_window_naive_model})"
#                             elif agg_method_rolling_window_naive_model == 'Mode':
#                                 naive_model2_feature_str = f"mode(yt-1, ..., yt-{size_rolling_window_naive_model})"
#                             else:
#                                 naive_model2_feature_str = '-'
#
#                             # Create new row with test result details
#                             new_row = {'model_name': 'Naive Model II',
#                                        'mape': '{:.2%}'.format(metrics_dict[model_name]['mape']),
#                                        'smape': '{:.2%}'.format(metrics_dict[model_name]['smape']),
#                                        'rmse': round(metrics_dict[model_name]['rmse'], 2),
#                                        'r2': round(metrics_dict[model_name]['r2'], 2),
#                                        'features': naive_model2_feature_str,
#                                        'model_settings': f"rolling window: {size_rolling_window_naive_model}, aggregation method: {agg_method_rolling_window_naive_model}",
#                                        'predicted': np.ravel(df_preds_naive_model2['Predicted'])}
#
#                             # Turn new row into a dataframe
#                             new_row_df = pd.DataFrame([new_row])
#
#                             # Concatenate new row DataFrame with results_df
#                             results_df = pd.concat([results_df, new_row_df], ignore_index=True)
#
#                             # Update session state with latest results with new row added
#                             set_state("TRAIN_PAGE", ("results_df", results_df))
#
#                             # =============================================================================
#                             # Naive Model III
#                             # =============================================================================
#                             # set model name
#                             model_name = 'Naive Model III'
#
#                             st.markdown('---')
#
#                             # retrieve dataframe with insample prediction results from train/test of naive model III
#                             df_preds_naive_model3 = forecast_naive_model3_insample(y_train,
#                                                                                    y_test,
#                                                                                    agg_method=agg_method_baseline_naive_model)
#                             # Show metrics on model card
#                             display_my_metrics(df_preds_naive_model3,
#                                                model_name='Naive Model III',
#                                                my_subtitle=f'{agg_method_baseline_naive_model.lower()}')
#
#                             # Plot graph with actual versus insample predictions
#                             plot_actual_vs_predicted(df_preds_naive_model3,
#                                                      my_conf_interval)
#
#                             # Show the dataframe
#                             st.dataframe(df_preds_naive_model3.style.format(
#                                 {'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Error': '{:.2f}', 'Error (%)': '{:.2%}'}),
#                                 use_container_width=True)
#
#                             # Create download button for forecast results to .csv
#                             download_csv_button(df_preds_naive_model3,
#                                                 my_file="insample_forecast_naivemodeliii_results.csv",
#                                                 help_message="Download your Naive Model III results to .CSV",
#                                                 my_key='naive_trained_modeliii_download_btn',
#                                                 set_index=True)
#
#                             # =============================================================================
#                             # SAVE TEST RESULTS FOR EVALUATION PAGE BY ADDING ROW TO RESULTS_DF
#                             # =============================================================================
#                             metrics_dict = my_metrics(df_preds_naive_model3, model_name="Naive Model III")
#
#                             # Get a scientific notation for the rolling window (Note: Latex not supported in Streamlit st.dataframe() so not currently possible)
#                             if agg_method_baseline_naive_model == 'Mean':
#                                 naive_model3_feature_str = f"(1/{len(y_train)}) * ∑yt, ..., yt-{len(y_train)}"
#                             elif agg_method_baseline_naive_model == 'Median':
#                                 naive_model3_feature_str = f"median(yt, ..., yt-{len(y_train)})"
#                             elif agg_method_baseline_naive_model == 'Mode':
#                                 naive_model3_feature_str = f"mode(yt, ..., yt-{len(y_train)})"
#                             else:
#                                 naive_model3_feature_str = '-'
#
#                             # Create new row with test result details
#                             new_row = {'model_name': model_name,
#                                        'mape': '{:.2%}'.format(metrics_dict[model_name]['mape']),
#                                        'smape': '{:.2%}'.format(metrics_dict[model_name]['smape']),
#                                        'rmse': round(metrics_dict[model_name]['rmse'], 2),
#                                        'r2': round(metrics_dict[model_name]['r2'], 2),
#                                        'features': naive_model3_feature_str,
#                                        'model_settings': f"aggregation method: {agg_method_baseline_naive_model.lower()}",
#                                        'predicted': np.ravel(df_preds_naive_model3['Predicted'])}
#
#                             # Turn new row into a dataframe
#                             new_row_df = pd.DataFrame([new_row])
#
#                             # Concatenate new row DataFrame with results_df
#                             results_df = pd.concat([results_df, new_row_df], ignore_index=True)
#
#                             # Update session state with latest results with new row added
#                             set_state("TRAIN_PAGE", ("results_df", results_df))
#                 except:
#                     st.warning(f'Naive Model(s) failed to train, please check parameters...!')
#                 try:
#                     # =============================================================================
#                     # LINEAR REGRESSION MODEL
#                     # =============================================================================
#                     if model_name == "Linear Regression":
#                         # create card with model insample prediction with linear regression model
#                         df_preds_linreg = create_streamlit_model_card(X_train=X_train,
#                                                                       y_train=y_train,
#                                                                       X_test=X_test,
#                                                                       y_test=y_test,
#                                                                       results_df=results_df,
#                                                                       model=model,
#                                                                       model_name=model_name)
#
#                         # =============================================================================
#                         # SAVE TEST RESULTS FOR EVALUATION PAGE BY ADDING ROW TO RESULTS_DF
#                         # =============================================================================
#                         # Define new row of train/test results
#                         new_row = {'model_name': 'Linear Regression',
#                                    'mape': '{:.2%}'.format(metrics_dict[model_name]['mape']),
#                                    'smape': '{:.2%}'.format(metrics_dict[model_name]['smape']),
#                                    'rmse': round(metrics_dict[model_name]['rmse'], 2),
#                                    'r2': round(metrics_dict[model_name]['r2'], 2),
#                                    'features': features_str,
#                                    'model_settings': '-',
#                                    'predicted': np.ravel(df_preds_linreg['Predicted']),
#                                    }
#
#                         # Turn new row into a dataframe
#                         new_row_df = pd.DataFrame([new_row])
#
#                         # Concatenate new row DataFrame with results_df
#                         results_df = pd.concat([results_df, new_row_df], ignore_index=True)
#
#                         # Update session state with latest results with new row added
#                         set_state("TRAIN_PAGE", ("results_df", results_df))
#                 except:
#                     st.warning(f'Linear Regression failed to train, please contact your administrator!')
#
#                 # =============================================================================
#                 # SARIMAX MODEL
#                 # =============================================================================
#                 try:
#                     if model_name == "SARIMAX":
#                         with st.expander('📈' + model_name, expanded=True):
#                             with st.spinner(
#                                     'This model might require some time to train... you can grab a coffee ☕ or tea 🍵'):
#                                 # =============================================================================
#                                 # PREPROCESS
#                                 # =============================================================================
#                                 # Assume DF is your DataFrame with boolean columns - needed for SARIMAX model that does not handle boolean, but integers (int) datatype instead
#                                 bool_cols = X_train.select_dtypes(include=bool).columns
#                                 X_train.loc[:, bool_cols] = X_train.loc[:, bool_cols].astype(int)
#                                 bool_cols = X_test.select_dtypes(include=bool).columns
#                                 X_test.loc[:, bool_cols] = X_test.loc[:, bool_cols].astype(int)
#
#                                 # Parameters have standard value but can be changed by user
#                                 preds_df_sarimax = evaluate_sarimax_model(order=(p, d, q),
#                                                                           seasonal_order=(P, D, Q, s),
#                                                                           exog_train=X_train,
#                                                                           exog_test=X_test,
#                                                                           endog_train=y_train,
#                                                                           endog_test=y_test)
#
#                                 # Show metrics on Model Card
#                                 display_my_metrics(preds_df_sarimax,
#                                                    model_name="SARIMAX",
#                                                    my_subtitle=f'(p, d, q)(P, D, Q) = ({p}, {d}, {q})({P}, {D}, {Q})')
#
#                                 # Plot graph with actual versus insample predictions on Model Card
#                                 plot_actual_vs_predicted(preds_df_sarimax, my_conf_interval)
#
#                                 # Show the dataframe on Model Card
#                                 st.dataframe(preds_df_sarimax.style.format(
#                                     {'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Error': '{:.2f}',
#                                      'Error (%)': '{:.2%}'}), use_container_width=True)
#
#                                 # Create download button for forecast results to .csv
#                                 download_csv_button(preds_df_sarimax, my_file="insample_forecast_sarimax_results.csv",
#                                                     help_message="Download your **SARIMAX** model results to .CSV")
#
#                                 # =============================================================================
#                                 # SAVE TEST RESULTS FOR EVALUATION PAGE BY ADDING ROW TO RESULTS_DF
#                                 # =============================================================================
#                                 # Define metrics for sarimax model
#                                 metrics_dict = my_metrics(preds_df_sarimax, model_name=model_name)
#
#                                 # Create new row
#                                 new_row = {'model_name': 'SARIMAX',
#                                            'mape': '{:.2%}'.format(metrics_dict[model_name]['mape']),
#                                            'smape': '{:.2%}'.format(metrics_dict[model_name]['smape']),
#                                            'rmse': round(metrics_dict[model_name]['rmse'], 2),
#                                            'r2': round(metrics_dict[model_name]['r2'], 2),
#                                            'features': features_str,
#                                            'model_settings': f'({p},{d},{q})({P},{D},{Q},{s})',
#                                            'predicted': np.ravel(preds_df_sarimax['Predicted'])}
#
#                                 # turn new row into a dataframe
#                                 new_row_df = pd.DataFrame([new_row])
#
#                                 # Concatenate new row DataFrame with results_df
#                                 results_df = pd.concat([results_df, new_row_df], ignore_index=True)
#
#                                 # Update session state with latest results with new row added
#                                 set_state("TRAIN_PAGE", ("results_df", results_df))
#                 except:
#                     st.warning(f'SARIMAX failed to train, please contact administrator!')
#
#                     # =============================================================================
#                 # PROPHET MODEL
#                 # =============================================================================
#                 if model_name == "Prophet":
#                     with st.expander('📈' + model_name, expanded=True):
#                         # Use custom function that creates in-sample prediction and return a dataframe with 'Actual', 'Predicted', 'Percentage_Diff', 'MAPE'
#                         preds_df_prophet = predict_prophet(y_train,
#                                                            y_test,
#                                                            changepoint_prior_scale=changepoint_prior_scale,
#                                                            seasonality_mode=seasonality_mode,
#                                                            seasonality_prior_scale=seasonality_prior_scale,
#                                                            holidays_prior_scale=holidays_prior_scale,
#                                                            yearly_seasonality=yearly_seasonality,
#                                                            weekly_seasonality=weekly_seasonality,
#                                                            daily_seasonality=daily_seasonality,
#                                                            interval_width=interval_width,
#                                                            X=X)  # TO EXTRACT FEUATURE(S) AND HOLIDAYS BASED ON USER FEATURE SELECTOIN TRUE/FALSE
#
#                         display_my_metrics(preds_df_prophet, "Prophet")
#
#                         # Plot graph with actual versus insample predictions
#                         plot_actual_vs_predicted(preds_df_prophet, my_conf_interval)
#
#                         # Show the dataframe
#                         st.dataframe(preds_df_prophet.style.format(
#                             {'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Error': '{:.2f}', 'Error (%)': '{:.2%}'}),
#                             use_container_width=True)
#
#                         # Create download button for forecast results to .csv
#                         download_csv_button(preds_df_prophet,
#                                             my_file="insample_forecast_prophet_results.csv",
#                                             help_message="Download your **Prophet** model results to .CSV",
#                                             my_key='download_btn_prophet_df_preds')
#
#                         # =============================================================================
#                         # SAVE TEST RESULTS FOR EVALUATION PAGE BY ADDING ROW TO RESULTS_DF
#                         # =============================================================================
#                         # define metrics for prophet model
#                         metrics_dict = my_metrics(preds_df_prophet, model_name=model_name)
#
#                         # display evaluation results on sidebar of streamlit_model_card
#                         new_row = {'model_name': 'Prophet',
#                                    'mape': '{:.2%}'.format(metrics_dict[model_name]['mape']),
#                                    'smape': '{:.2%}'.format(metrics_dict[model_name]['smape']),
#                                    'rmse': round(metrics_dict[model_name]['rmse'], 2),
#                                    'r2': round(metrics_dict[model_name]['r2'], 2),
#                                    'features': features_str,
#                                    'predicted': np.ravel(preds_df_prophet['Predicted']),
#                                    'model_settings': f' changepoint_prior_scale: {changepoint_prior_scale}, seasonality_prior_scale: {seasonality_prior_scale}, country_holidays: {country_holidays}, holidays_prior_scale: {holidays_prior_scale}, yearly_seasonality: {yearly_seasonality}, weekly_seasonality: {weekly_seasonality}, daily_seasonality: {daily_seasonality}, interval_width: {interval_width}'}
#
#                         # turn new row into a dataframe
#                         new_row_df = pd.DataFrame([new_row])
#
#                         # Concatenate new row DataFrame with results_df
#                         results_df = pd.concat([results_df, new_row_df], ignore_index=True)
#
#                         # update session state with latest results with new row added
#                         set_state("TRAIN_PAGE", ("results_df", results_df))
#
#             # show friendly user reminder message they can compare results on the evaluation page
#             st.markdown(
#                 f'<h2 style="text-align:center; font-family: Ysabeau SC, sans-serif; font-size: 18px ; color: black; border: 1px solid #d7d8d8; padding: 10px; border-radius: 5px;">💡 Vist the Evaluation Page for a comparison of your test results! </h2>',
#                 unsafe_allow_html=True)
#
#     with tab4:
#         # SHOW MODEL DETAILED DOCUMENTATION
#         # model_documentation(st.session_state['selected_model_info'])
#         model_documentation()
#
# # =============================================================================
# #   ________      __     _     _    _      _______ ______
# #  |  ____\ \    / /\   | |   | |  | |  /\|__   __|  ____|
# #  | |__   \ \  / /  \  | |   | |  | | /  \  | |  | |__
# #  |  __|   \ \/ / /\ \ | |   | |  | |/ /\ \ | |  |  __|
# #  | |____   \  / ____ \| |___| |__| / ____ \| |  | |____
# #  |______|   \/_/    \_\______\____/_/    \_\_|  |______|
# #
# # =============================================================================
# # EVALUATE
# if menu_item == 'Evaluate' and sidebar_menu_item == 'HOME':
#     # =============================================================================
#     # SIDEBAR OF EVALUATE PAGE
#     # =============================================================================
#     with st.sidebar:
#
#         my_title(f"""{icons["evaluate_icon"]}""", "#000000")
#
#         with st.form('Plot Metric'):
#
#             my_text_paragraph('Plot')
#
#             col1, col2, col3 = st.columns([2, 8, 2])
#             with col2:
#                 # define list of metrics user can choose from in drop-down selectbox
#                 metrics_list = ['Mean Absolute Percentage Error', 'Symmetric Mean Absolute Percentage Error',
#                                 'Root Mean Square Error', 'R-squared']
#
#                 selected_metric = st.selectbox(label="*Select evaluation metric to sort by:*",
#                                                options=metrics_list,
#                                                key=key1_evaluate,
#                                                help='''Choose which evaluation metric is used to sort the top score of each model by:
#                                                 \n- `Mean Absolute Percentage Error` (MAPE):
#                                                 \nThe Mean Absolute Percentage Error measures the average absolute percentage difference between the predicted values and the actual values. It provides a relative measure of the accuracy of a forecasting model. A lower MAPE indicates better accuracy.
#                                                 \n- `Symmetric Mean Absolute Percentage Error` (SMAPE):
#                                                 \n The Symmetric Mean Absolute Percentage Error (SMAPE) is similar to MAPE but addresses the issue of scale dependency. It calculates the average of the absolute percentage differences between the predicted values and the actual values, taking into account the magnitude of both values. SMAPE values range between 0% and 100%, where lower values indicate better accuracy.
#                                                 \n- `Root Mean Square Error` (RMSE):
#                                                 \n The Root Mean Square Error is a commonly used metric to evaluate the accuracy of a prediction model. It calculates the square root of the average of the squared differences between the predicted values and the actual values. RMSE is sensitive to outliers and penalizes larger errors.
#                                                 \n- `R-squared` (R2):
#                                                 \nR-squared is a statistical measure that represents the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, where 1 indicates a perfect fit. R-squared measures how well the predicted values fit the actual values and is commonly used to assess the goodness of fit of a regression model.
#                                                 ''')
#
#             col1, col2, col3 = st.columns([5, 4, 4])
#             with col2:
#                 metric_btn = st.form_submit_button('Submit', type="secondary", on_click=form_update,
#                                                    args=('EVALUATE_PAGE',))
#
#         with st.expander('', expanded=True):
#             my_text_paragraph('Latest Test Results')
#
#             if 'results_df' not in st.session_state:
#                 st.session_state['results_df'] = results_df
#             else:
#                 # if user just reloads the page - do not add the same results to the dataframe (e.g. avoid adding duplicates)
#                 # by checking if the values of the dataframes are the same
#                 if get_state("TRAIN_PAGE", "results_df").equals(st.session_state['results_df']):
#                     pass
#                 else:
#                     # add new test results to results dataframe
#                     st.session_state['results_df'] = pd.concat([st.session_state['results_df'],
#                                                                 get_state("TRAIN_PAGE", "results_df")],
#                                                                ignore_index=True)
#             # =============================================================================
#             # 1. Show Last Test Run Results in a Dataframe
#             # =============================================================================
#             st.dataframe(data=get_state("TRAIN_PAGE", "results_df"),
#                          column_config={"predicted": st.column_config.LineChartColumn("predicted", width="small",
#                                                                                       help="insample prediction",
#                                                                                       y_min=None, y_max=None)},
#                          hide_index=True)
#
#             my_text_paragraph('Top 3 Test Results')
#
#             # Convert the 'mape' column to floats, removes duplicates based on the 'model_name' and 'mape' columns, sorts the unique DataFrame by ascending 'mape' values, selects the top 3 rows, and displays the resulting DataFrame in Streamlit.
#             test_df = st.session_state['results_df'].assign(
#                 mape=st.session_state.results_df['mape'].str.rstrip('%').astype(float)).drop_duplicates(
#                 subset=['model_name', 'mape']).sort_values(by='mape', ascending=True).iloc[:3]
#
#             test_df['mape'] = test_df['mape'].astype(str) + '%'
#
#             # =============================================================================
#             # 2. Show Top 3 test Results in Dataframe
#             # =============================================================================
#             st.dataframe(data=test_df,
#                          column_config={"predicted": st.column_config.LineChartColumn("predicted", width="small",
#                                                                                       help="insample prediction",
#                                                                                       y_min=None, y_max=None)},
#                          use_container_width=True,
#                          hide_index=True)
#
#     # =============================================================================
#     # MAIN PAGE OF EVALUATE PAGE
#     # =============================================================================
#     with st.expander('', expanded=True):
#
#         col0, col1, col2, col3 = st.columns([10, 90, 8, 5])
#         with col1:
#             my_text_header('Model Performance')
#
#             my_text_paragraph(f'{selected_metric}')
#
#         # Plot for each model the expirement run with the lowest error (MAPE or RMSE) or highest r2 based on user's chosen metric
#         plot_model_performance(st.session_state['results_df'], my_metric=selected_metric)
#
#         with col2:
#             # Clear session state results_df button to remove prior test results and start fresh
#             clear_results_df = st.button(label='🗑️', on_click=clear_test_results,
#                                          help='Clear Results')  # on click run function to clear results
#
#         # =============================================================================
#         # 3. Show Historical Test Runs Dataframe
#         # =============================================================================
#         # see for line-chart: https://docs.streamlit.io/library/api-reference/data/st.column_config/st.column_config.linechartcolumn
#         st.dataframe(data=st.session_state['results_df'],
#                      column_config={"predicted": st.column_config.LineChartColumn("predicted", width="small",
#                                                                                   help="insample prediction",
#                                                                                   y_min=None, y_max=None)},
#                      hide_index=False,
#                      use_container_width=True)
#
#         # Download Button for test results to .csv
#         download_csv_button(my_df=st.session_state['results_df'],
#                             my_file="Modeling Test Results.csv",
#                             help_message="Download your Modeling Test Results to .CSV")
#
#     st.image('./images/evaluation_page.png')
#
# # =============================================================================
# #   _______ _    _ _   _ ______
# #  |__   __| |  | | \ | |  ____|
# #     | |  | |  | |  \| | |__
# #     | |  | |  | | . ` |  __|
# #     | |  | |__| | |\  | |____
# #     |_|   \____/|_| \_|______|
# #
# # =============================================================================
# # 9. Hyper-parameter tuning
# if menu_item == 'Tune' and sidebar_menu_item == 'HOME':
#     # # =============================================================================
#     # # Initiate Variables Required
#     # # =============================================================================
#     # selected_models = [('Naive Model', None),
#     #                    ('Linear Regression', LinearRegression(fit_intercept=True)),
#     #                    ('SARIMAX', SARIMAX(y_train)),
#     #                    ('Prophet', Prophet())]  # Define tuples of (model name, model)
#
#     # =============================================================================
#     # CREATE USER FORM FOR HYPERPARAMETER TUNING
#     # =============================================================================
#     with st.sidebar:
#         # return parameters set for hyper-parameter tuning either default or overwritten by user options
#         max_wait_time, selected_model_names, metric, search_algorithm, trial_runs, trend, \
#             lag_options, rolling_window_range, rolling_window_options, agg_options, \
#             p_max, d_max, q_max, P_max, D_max, Q_max, s, enforce_stationarity, enforce_invertibility, \
#             horizon_option, changepoint_prior_scale_options, seasonality_mode_options, seasonality_prior_scale_options, holidays_prior_scale_options, yearly_seasonality_options, weekly_seasonality_options, daily_seasonality_options, \
#             hp_tuning_btn = hyperparameter_tuning_form(y_train)
#
#     # #########################################################################################################
#     # # INITIATE VARIABLES
#     # #########################################################################################################
#     # progress_bar = st.progress(0)
#     #
#     # # Store total number of combinations in the parameter grid for progress bar
#     # total_options = len(lag_options) + (len(list(range(rolling_window_range[0], rolling_window_range[1] + 1)))
#     #                                     * len(rolling_window_options)) + len(agg_options)
#
#     # =============================================================================
#     # CREATE DATAFRAMES TO STORE HYPERPARAMETER TUNING RESULTS
#     # =============================================================================
#     sarimax_tuning_results = pd.DataFrame()
#     prophet_tuning_results = pd.DataFrame()
#
#     # if user clicks the hyperparameter tuning button start hyperparameter tuning code below for selected models
#     if hp_tuning_btn == True and selected_model_names:
#
#         # iterate over user selected models from multi-selectbox when user pressed SUBMIT button for hyperparameter
#         # tuning
#         for model_name in selected_model_names:
#
#             # Set start time when grid-search is kicked-off to define total time it takes as computationally intensive
#             start_time = time.time()
#
#             if model_name == 'Naive Model':
#                 with st.expander('⚙️ Naive Models', expanded=True):
#                     progress_bar = st.progress(0)
#
#                     #########################################################################################################
#                     # Run Naive Models: I (LAG), II (ROLLING WINDOW), III (CONSTANT: MEAN/MEDIAN/MODE)
#                     #########################################################################################################
#                     run_naive_model_1(lag_options=lag_options,
#                                       validation_set=y_test,
#                                       max_wait_time=max_wait_time,
#                                       rolling_window_range=rolling_window_range,
#                                       rolling_window_options=rolling_window_options,
#                                       agg_options=agg_options,
#                                       metrics_dict=metrics_dict,
#                                       progress_bar=progress_bar
#                                       )
#
#                     run_naive_model_2(lag_options=lag_options,
#                                       rolling_window_range=rolling_window_range,
#                                       rolling_window_options=rolling_window_options,
#                                       validation_set=y_test,
#                                       agg_options=agg_options,
#                                       max_wait_time=max_wait_time,
#                                       y_test=y_test,
#                                       metrics_dict=metrics_dict,
#                                       progress_bar=progress_bar
#                                       )
#
#                     run_naive_model_3(agg_options=agg_options,
#                                       train_set=y_train,
#                                       validation_set=y_test,
#                                       max_wait_time=max_wait_time,
#                                       rolling_window_range=rolling_window_range,
#                                       lag_options=lag_options,
#                                       rolling_window_options=rolling_window_options,
#                                       metrics_dict=metrics_dict,
#                                       progress_bar=progress_bar
#                                       )
#
#             if model_name == "SARIMAX":
#                 if search_algorithm == 'Grid Search':
#
#                     start_time = time.time()  # set start time when grid-search is kicked-off to define total time it takes as computationaly intensive
#                     progress_bar = st.progress(0)  # initiate progress bar for SARIMAX Grid Search runtime
#
#                     # =============================================================================
#                     # Define Parameter Grid for Grid Search
#                     # =============================================================================
#                     # check if trend has at least 1 option (not an empty list) / e.g. when user clears all options in sidebar - set default to 'n'
#                     if trend == []:
#                         trend = ['n']
#
#                     param_grid = {'order': [(p, d, q) for p, d, q in
#                                             itertools.product(range(p_max + 1), range(d_max + 1), range(q_max + 1))],
#                                   'seasonal_order': [(p, d, q, s) for p, d, q in
#                                                      itertools.product(range(P_max + 1), range(D_max + 1),
#                                                                        range(Q_max + 1))],
#                                   'trend': trend}
#
#                     # store total number of combinations in the parameter grid for progress bar
#                     total_combinations = len(param_grid['order']) * len(param_grid['seasonal_order']) * len(
#                         param_grid['trend'])
#
#                     # Convert max_wait_time to an integer
#                     max_wait_time_minutes = int(str(max_wait_time.minute))
#
#                     # Convert the maximum waiting time from minutes to seconds
#                     max_wait_time_seconds = max_wait_time_minutes * 60
#
#                     # iterate over grid of all possible combinations of hyperparameters
#                     for i, (param, param_seasonal, trend) in enumerate(
#                             itertools.product(param_grid['order'], param_grid['seasonal_order'], param_grid['trend']),
#                             0):
#
#                         # Check if the maximum waiting time has been exceeded
#                         elapsed_time_seconds = time.time() - start_time
#
#                         if elapsed_time_seconds > max_wait_time_seconds:
#                             st.warning("Maximum waiting time exceeded. The grid search has been stopped.")
#                             # exit the loop once maximum time is exceeded defined by user or default = 5 minutes
#                             break
#
#                         # Update the progress bar
#                         progress_percentage = i / total_combinations * 100
#                         progress_bar.progress(value=(i / total_combinations),
#                                               text=f'''Please wait up to {max_wait_time_minutes} minute(s) while hyperparameters of {model_name} model are being tuned!
#                                                          \n{progress_percentage:.2f}% of total combinations within the search space reviewed ({i} out of {total_combinations} combinations).''')
#
#                         try:
#                             # Create a SARIMAX model with the current parameter values
#                             model = SARIMAX(y_train,
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             exog=X_train,
#                                             enforce_stationarity=enforce_stationarity,
#                                             enforce_invertibility=enforce_invertibility)
#
#                             # Train/Fit the model to the data
#                             mdl = model.fit(disp=0)
#
#                             # Append a new row to the dataframe with the parameter values and AIC score
#                             sarimax_tuning_results = sarimax_tuning_results.append(
#                                 {'SARIMAX (p,d,q)x(P,D,Q,s)': f'{param} x {param_seasonal}',
#                                  'order': param,
#                                  'seasonal_order': param_seasonal,
#                                  'trend': trend,
#                                  'AIC': "{:.2f}".format(mdl.aic),
#                                  'BIC': "{:.2f}".format(mdl.bic),
#                                  'RMSE': "{:.2f}".format(math.sqrt(mdl.mse))
#                                  }, ignore_index=True)
#
#                         # If the model fails to fit, skip it and continue to the next model
#                         except:
#                             continue
#
#                     # set the end of runtime
#                     end_time_sarimax = time.time()
#
#                     # clear progress bar in streamlit for user as process is completed
#                     progress_bar.empty()
#
#                     # add a rank column and order dataframe ascending based on the chosen metric
#                     gridsearch_sarimax_results_df = rank_dataframe(df=sarimax_tuning_results, metric=metric)
#
#                     # ============================================================================
#                     # Show Results in Streamlit
#                     # =============================================================================
#                     with st.expander('⚙️ SARIMAX', expanded=True):
#
#                         # Display the result dataframe
#                         st.dataframe(gridsearch_sarimax_results_df, hide_index=True, use_container_width=True)
#                         st.write(
#                             f'🏆 **SARIMAX** hyperparameters with the lowest {metric} of **{gridsearch_sarimax_results_df.iloc[0, 5]}** found in **{end_time_sarimax - start_time:.2f}** seconds is:')
#                         st.write(f'- **`(p,d,q)(P,D,Q,s)`**: {gridsearch_sarimax_results_df.iloc[0, 1]}')
#                         st.markdown('---')
#
#                         # =============================================================================
#                         # Hyperparameter Importance
#                         # =============================================================================
#                         # Step 1: Perform the grid search and store the evaluation metric values for each combination of hyperparameters.
#                         columns = ['p', 'd', 'q', 'P', 'D', 'Q', 's', 'trend', 'AIC', 'BIC', 'RMSE']
#                         df_gridsearch = pd.DataFrame(columns=columns)
#
#                         for _, row in sarimax_tuning_results.iterrows():
#                             order = tuple(map(int, str(row['order']).strip('()').split(',')))
#                             seasonal_order = tuple(map(int, str(row['seasonal_order']).strip('()').split(',')))
#                             aic = float(row['AIC'])
#                             bic = float(row['BIC'])
#                             rmse = float(row['RMSE'])
#                             trend = row['trend']
#
#                             df_gridsearch = df_gridsearch.append({'p': order[0], 'd': order[1], 'q': order[2],
#                                                                   'P': seasonal_order[0], 'D': seasonal_order[1],
#                                                                   'Q': seasonal_order[2], 's': seasonal_order[3],
#                                                                   'trend': trend, 'AIC': aic, 'BIC': bic, 'RMSE': rmse},
#                                                                  ignore_index=True)
#
#                         # =============================================================================
#                         # Step 2: Calculate the baseline metric as the minimum value of the evaluation metric.
#                         # =============================================================================
#                         baseline_metric = df_gridsearch[metric].astype(float).min()
#                         # st.write(baseline_metric) # TEST
#
#                         # =============================================================================
#                         # Step 3: Define / Train Model on hyperparameters (X) and target metric (y)
#                         # =============================================================================
#                         # Prepare the data
#                         df_gridsearch['trend'] = df_gridsearch['trend'].replace({None: 'None'})
#                         df_gridsearch['trend'] = df_gridsearch['trend'].map(
#                             {'n': 0, 'c': 1, 't': 2, 'ct': 3, 'None': 4})
#
#                         X = df_gridsearch[['p', 'd', 'q', 'P', 'D', 'Q', 'trend']]  # Features (hyperparameters)
#                         y = df_gridsearch[metric]  # Target (evaluation metric values)
#
#                         # Define Model for Importance Score testing
#                         rf_model = RandomForestRegressor()
#
#                         # Define X and y (X = parameters, y = evaluation metric scores)
#                         rf_model.fit(X, y)
#
#                         # =============================================================================
#                         # Step 4: Calculate the relative importance of each hyperparameter
#                         # =============================================================================
#                         feature_importances = rf_model.feature_importances_
#                         importance_scores = {param: score for param, score in zip(X.columns, feature_importances)}
#
#                         # Sort the feature importances in descending order
#                         sorted_feature_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=False)
#
#                         # =============================================================================
#                         # STEP 5: Plot results
#                         # =============================================================================
#                         create_hyperparameter_importance_plot(sorted_feature_importance, plot_type='bar_chart',
#                                                               grid_search=True)
#
#                 if search_algorithm == 'Bayesian Optimization':
#                     # =============================================================================
#                     # Step 0: Define variables in order to retrieve time it takes to run hyperparameter tuning
#                     # =============================================================================
#                     # set start time when grid-search is kicked-off to define total time it takes as computationaly intensive
#                     start_time = time.time()
#
#                     # Convert max_wait_time to an integer
#                     max_wait_time_minutes = int(str(max_wait_time.minute))
#
#                     # Convert the maximum waiting time from minutes to seconds
#                     max_wait_time_seconds = max_wait_time_minutes * 60
#
#                     # =============================================================================
#                     # Step 1: Define Dataframe to save results in
#                     # =============================================================================
#                     # Create a dataframe to store the tuning results
#                     optuna_results = pd.DataFrame(
#                         columns=['SARIMAX (p,d,q)x(P,D,Q,s)', 'order', 'seasonal_order', 'trend', 'AIC', 'BIC', 'RMSE'])
#
#                     # =============================================================================
#                     # Step 2: Define progress bar for user to visually see progress
#                     # =============================================================================
#                     # Set progress bar for Optuna runtime
#                     progress_bar = st.progress(0)
#
#                     # =============================================================================
#                     # Step 3: Define parameter grid
#                     # =============================================================================
#                     pdq = [(p, d, q) for p, d, q in
#                            itertools.product(range(p_max + 1), range(d_max + 1), range(q_max + 1))]
#                     pdqs = [(p, d, q, s) for p, d, q in
#                             itertools.product(range(P_max + 1), range(D_max + 1), range(Q_max + 1))]
#
#
#                     # =============================================================================
#                     # Step 4: Define the objective function (REQUIREMENT FOR OPTUNA PACKAGE)
#                     # =============================================================================
#
#                     def objective_sarima(trial, trend=trend):
#                         # define parameters of grid for optuna package with syntax: suggest_categorical()
#                         p = trial.suggest_int('p', 0, p_max)
#                         d = trial.suggest_int('d', 0, d_max)
#                         q = trial.suggest_int('q', 0, q_max)
#                         P = trial.suggest_int('P', 0, P_max)
#                         D = trial.suggest_int('D', 0, D_max)
#                         Q = trial.suggest_int('Q', 0, Q_max)
#                         trend = trial.suggest_categorical('trend', trend)
#
#                         order = (p, d, q)
#                         seasonal_order = (P, D, Q, s)
#
#                         try:
#                             # Your SARIMAX model code here
#                             model = SARIMAX(y_train,
#                                             order=order,
#                                             seasonal_order=seasonal_order,
#                                             trend=trend,
#                                             # initialization='approximate_diffuse' # Initialization method for the initial state. If a string, must be one of {‘diffuse’, ‘approximate_diffuse’, ‘stationary’, ‘known’}.
#                                             )
#
#                         except ValueError as e:
#                             # Handle the specific exception for overlapping moving average terms
#                             if 'moving average lag(s)' in str(e):
#                                 st.warning(
#                                     f'Skipping parameter combination: {order}, {seasonal_order}, {trend} due to invalid model.')
#                             else:
#                                 raise e
#
#                         # Train the model
#                         mdl = model.fit(disp=0)
#
#                         # Note that we set best_metric to -np.inf instead of np.inf since we want to maximize the R2 metric.
#                         if metric in ['AIC', 'BIC', 'RMSE']:
#                             # Check if the current model has a lower AIC than the previous models
#                             if metric == 'AIC':
#                                 my_metric = mdl.aic
#                             elif metric == 'BIC':
#                                 my_metric = mdl.bic
#                             elif metric == 'RMSE':
#                                 my_metric = math.sqrt(mdl.mse)
#
#                         # Append the result to the dataframe
#                         optuna_results.loc[len(optuna_results)] = [f'{order}x{seasonal_order}', order, seasonal_order,
#                                                                    trend, mdl.aic, mdl.bic, math.sqrt(mdl.mse)]
#
#                         # Update the progress bar
#                         progress_percentage = len(optuna_results) / trial_runs * 100
#                         progress_bar.progress(value=len(optuna_results) / trial_runs,
#                                               text=f'Please Wait! Tuning {model_name} hyperparameters... {progress_percentage:.2f}% completed ({len(optuna_results)} of {trial_runs} total combinations).')
#                         return my_metric
#
#
#                     # =============================================================================
#                     # Step 5: Create the study (hyperparameter tuning experiment)
#                     # =============================================================================
#                     study = optuna.create_study(direction="minimize")
#
#                     # =============================================================================
#                     # Step 6: Optimize the study with the objective function (run experiment)
#                     # =============================================================================
#                     study.optimize(objective_sarima,
#                                    n_trials=trial_runs,
#                                    timeout=max_wait_time_seconds)
#
#                     # =============================================================================
#                     # Step 8: Clear progress bar and stop time when optimization has completed
#                     # =============================================================================
#                     # Clear the progress bar in Streamlit
#                     progress_bar.empty()
#
#                     # set the end of runtime
#                     end_time_sarimax = time.time()
#
#                     # =============================================================================
#                     # Step 9: Rank the results dataframe on the evaluation metric
#                     # =============================================================================
#                     bayesian_sarimax_results_df = rank_dataframe(df=optuna_results, metric=metric)
#
#                     # round to 2 decimals
#                     bayesian_sarimax_results_df[['AIC', 'BIC', 'RMSE']] = bayesian_sarimax_results_df[
#                         ['AIC', 'BIC', 'RMSE']].round(2)
#
#                     # =============================================================================
#                     # Step 10: Show Results in Streamlit
#                     # =============================================================================
#                     with st.expander('⚙️ SARIMAX', expanded=True):
#
#                         # Display the result dataframe
#                         st.dataframe(bayesian_sarimax_results_df, hide_index=True, use_container_width=True)
#
#                         # user messages about the results
#                         st.write(
#                             f'🏆 **SARIMAX** hyperparameters with the lowest {metric} of **{bayesian_sarimax_results_df.iloc[0, 5]}** found in **{end_time_sarimax - start_time:.2f}** seconds is:')
#                         st.write(f'- **`(p,d,q)(P,D,Q,s)`**: {bayesian_sarimax_results_df.iloc[0, 1]}')
#                         st.markdown('---')
#
#                         # =============================================================================
#                         # Plot Hyperparameter Importance
#                         # =============================================================================
#                         # Optuna Parameter Importance Plot
#                         create_hyperparameter_importance_plot(study, plot_type='param_importances', grid_search=False)
#
#             if model_name == "Prophet":
#                 horizon_int = horizon_option
#                 horizon_str = f'{horizon_int} days'  # construct horizon parameter string
#
#                 # define cutoffs
#                 cutoff_date = X_train.index[-(horizon_int + 1)].strftime('%Y-%m-%d')
#
#                 # define the parameter grid - user can select options with multi-select box in streamlit app
#                 param_grid = {'changepoint_prior_scale': changepoint_prior_scale_options,
#                               'seasonality_prior_scale': seasonality_prior_scale_options,
#                               'changepoint_prior_scale': changepoint_prior_scale_options,
#                               'seasonality_mode': seasonality_mode_options,
#                               'seasonality_prior_scale': seasonality_prior_scale_options,
#                               'holidays_prior_scale': holidays_prior_scale_options,
#                               'yearly_seasonality': yearly_seasonality_options,
#                               'weekly_seasonality': weekly_seasonality_options,
#                               'daily_seasonality': daily_seasonality_options}
#
#                 # Generate all combinations of parameters
#                 all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
#                 rmses = []  # Store the RMSEs for each params here
#                 aics = []  # Store the AICs for each params here
#                 bics = []  # Store the BICs for each params here
#
#                 # for simplicity set only a single cutoff for train/test split defined by user in the streamlit app
#                 # cutoffs = pd.to_datetime(['2021-06-01', '2021-12-31']) # add list of dates
#                 cutoffs = pd.to_datetime([cutoff_date])
#
#                 # preprocess the data (y_train/y_test) for prophet model with datestamp (DS) and target (y) column
#                 y_train_prophet = preprocess_data_prophet(y_train)
#                 y_test_prophet = preprocess_data_prophet(y_test)
#
#                 # Use cross validation to evaluate all parameters
#                 for params in all_params:
#                     m = Prophet(**params)  # Fit model with given params
#                     # train the model on the data with set parameters
#                     m.fit(y_train_prophet)
#
#                     # other examples of forecast horizon settings:
#                     # df_cv2 = cross_validation(m, cutoffs=cutoffs, horizon='100 days')
#                     # df_cv2 = cross_validation(m, cutoffs=cutoffs, horizon='365 days')
#                     df_cv = cross_validation(m, cutoffs=cutoffs, horizon=horizon_str, parallel=False)
#
#                     # rolling_window = 1 computes performance metrics using all the forecasted data to get a single performance metric number.
#                     df_p = performance_metrics(df_cv, rolling_window=1)
#
#                     rmses.append(df_p['rmse'].values[0])
#
#                     # Get residuals to compute AIC and BIC
#                     df_cv['residuals'] = df_cv['y'] - df_cv['yhat']
#
#                     residuals = df_cv['residuals'].values
#
#                     # Compute AIC and BIC
#                     nobs = len(residuals)
#                     k = len(params)
#                     loglik = -0.5 * nobs * (1 + np.log(2 * np.pi) + np.log(np.sum(residuals ** 2) / nobs))
#                     aic = -2 * loglik + 2 * k
#                     bic = 2 * loglik + k * np.log(nobs)
#
#                     # Add AIC score to list
#                     aics.append(aic)
#
#                     # Add BIC score to list
#                     bics.append(bic)
#
#                 # create dataframe with parameter combinations
#                 prophet_tuning_results = pd.DataFrame(all_params)
#                 # add RMSE scores to dataframe
#                 prophet_tuning_results['rmse'] = rmses
#                 # add AIC scores to dataframe
#                 prophet_tuning_results['aic'] = aics
#                 # add BIC scores to dataframe
#                 prophet_tuning_results['bic'] = bics
#                 # set the end of runtime
#                 end_time_prophet = time.time()
#
#         if 'Prophet' in selected_model_names:
#             with st.expander('⚙️ Prophet', expanded=True):
#
#                 if metric.lower() in prophet_tuning_results.columns:
#                     prophet_tuning_results['Rank'] = prophet_tuning_results[metric.lower()].rank(method='min',
#                                                                                                  ascending=True).astype(
#                         int)
#
#                     # sort values by rank
#                     prophet_tuning_results = prophet_tuning_results.sort_values('Rank', ascending=True)
#
#                     # show user dataframe of gridsearch results ordered by ranking
#                     st.dataframe(prophet_tuning_results.set_index('Rank'), use_container_width=True)
#
#                     # provide button for user to download the hyperparameter tuning results
#                     download_csv_button(prophet_tuning_results, my_file='Prophet_Hyperparameter_Gridsearch.csv',
#                                         help_message='Download your Hyperparameter tuning results to .CSV')
#
#                 # st.success(f"ℹ️ Prophet search for your optimal hyper-parameters finished in **{end_time_prophet - start_time:.2f}** seconds")
#
#                 if not prophet_tuning_results.empty:
#                     st.markdown(
#                         f'🏆 **Prophet** set of parameters with the lowest {metric} of **{"{:.2f}".format(prophet_tuning_results.loc[0, metric.lower()])}** found in **{end_time_prophet - start_time:.2f}** seconds are:')
#                     st.write('\n'.join([f'- **`{param}`**: {prophet_tuning_results.loc[0, param]}' for param in
#                                         prophet_tuning_results.columns[:6]]))
#     else:
#
#         with st.expander('', expanded=True):
#             st.image('./images/tune_models.png')
#             my_text_paragraph('Please select at least one model in the sidebar and press \"Submit\"!',
#                               my_font_size='16px')
#
# # =============================================================================
# #   ______ ____  _____  ______ _____           _____ _______
# #  |  ____/ __ \|  __ \|  ____/ ____|   /\    / ____|__   __|
# #  | |__ | |  | | |__) | |__ | |       /  \  | (___    | |
# #  |  __|| |  | |  _  /|  __|| |      / /\ \  \___ \   | |
# #  | |   | |__| | | \ \| |___| |____ / ____ \ ____) |  | |
# #  |_|    \____/|_|  \_\______\_____/_/    \_\_____/   |_|
# #
# # =============================================================================
# if menu_item == 'Forecast':
#
#     # =============================================================================
#     # DEFINE VARIABLES NEEDED FOR FORECAST
#     # =============================================================================
#     min_date = df['date'].min()
#     max_date = df['date'].max()
#     max_value_calendar = None
#
#     # define maximum value in dataset for year, month day
#     year = max_date.year
#     month = max_date.month
#     day = max_date.day
#     end_date_calendar = df['date'].max()
#
#     # end date dataframe + 1 day into future is start date of forecast
#     start_date_forecast = end_date_calendar + timedelta(days=1)
#
#     # =============================================================================
#     # FORECAST SIDEBAR
#     # =============================================================================
#     with st.sidebar:
#
#         my_title(f'{icons["forecast_icon"]}', "#48466D")
#
#         with st.form("📆 "):
#             if dwt_features_checkbox:
#                 # wavelet model choice forecast
#                 my_subheader('Select Model for Discrete Wavelet Feature(s) Forecast Estimates')
#
#                 model_type_wavelet = st.selectbox('Select a model', ['Support Vector Regression', 'Linear'],
#                                                   label_visibility='collapsed')
#
#                 # define all models in list as we retrain models on entire dataset anyway
#             selected_models_forecast_lst = ['Linear Regression', 'SARIMAX', 'Prophet']
#
#             # SELECT MODEL(S) for Forecasting
#             selected_model_names = st.multiselect('*Select Forecasting Models*', selected_models_forecast_lst,
#                                                   default=selected_models_forecast_lst)
#
#             # create column spacers
#             col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 7, 7, 1, 6, 7, 1])
#
#             with col2:
#                 st.markdown(
#                     f'<h5 style="color: #48466D; background-color: #F0F2F6; padding: 12px; border-radius: 5px;"><center> End Date:</center></h5>',
#                     unsafe_allow_html=True)
#             with col3:
#                 for model_name in selected_model_names:
#                     # if model is linear regression max out the time horizon to maximum possible
#                     if model_name == "Linear Regression":
#                         # max value is it depends on the length of the input data and the forecasting method used. Linear regression can only forecast as much as the input data if you're using it for time series forecasting.
#                         max_value_calendar = end_date_calendar + timedelta(days=len(df))
#
#                     if model_name == "SARIMAX":
#                         max_value_calendar = None
#
#                 # create user input box for entering date in a streamlit calendar widget
#                 end_date_forecast = st.date_input(label="input forecast date",
#                                                   value=start_date_forecast,
#                                                   min_value=start_date_forecast,
#                                                   max_value=max_value_calendar,
#                                                   label_visibility='collapsed')
#             with col5:
#                 # set text information for dropdown frequency
#                 st.markdown(
#                     f'<h5 style="color: #48466D; background-color: #F0F2F6; padding: 12px; border-radius: 5px;"><center> Frequency:</center></h5>',
#                     unsafe_allow_html=True)
#
#             with col6:
#                 # Define a dictionary of possible frequencies and their corresponding offsets
#                 forecast_freq_dict = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
#
#                 # Ask the user to select the frequency of the data
#                 forecast_freq = st.selectbox('Select the frequency of the data', list(forecast_freq_dict.keys()),
#                                              label_visibility='collapsed')
#
#                 # get the value of the key e.g. D, W, M, Q or Y
#                 forecast_freq_letter = forecast_freq_dict[forecast_freq]
#
#             vertical_spacer(1)
#
#             # create vertical spacing columns
#             col1, col2, col3 = st.columns([4, 4, 4])
#             with col2:
#                 # create submit button for the forecast
#                 forecast_btn = st.form_submit_button("Submit", type="secondary")
#
#                 # =============================================================================
#     # RUN FORECAST - when user clicks the forecast button then run below
#     # =============================================================================
#     if forecast_btn:
#         # create a date range for your forecast
#         future_dates = pd.date_range(start=start_date_forecast, end=end_date_forecast, freq=forecast_freq_letter)
#
#         # first create all dates in dataframe with 'date' column
#         df_future_dates = future_dates.to_frame(index=False, name='date')
#
#         # add the special calendar days
#         df_future_dates = create_calendar_special_days(df_future_dates)
#
#         # add the year/month/day dummy variables
#         df_future_dates = create_date_features(df, year_dummies=year_dummies_checkbox,
#                                                month_dummies=month_dummies_checkbox, day_dummies=day_dummies_checkbox)
#
#         # if user wants discrete wavelet features add them
#         if dwt_features_checkbox:
#             df_future_dates = forecast_wavelet_features(X, features_df_wavelet, future_dates, df_future_dates)
#
#         ##############################################
#         # DEFINE X future
#         ##############################################
#         # select only features user selected from df e.g. slice df
#         X_future = df_future_dates.loc[:,
#                    ['date'] + [col for col in feature_selection_user if col in df_future_dates.columns]]
#         # set the 'date' column as the index again
#         X_future = copy_df_date_index(X_future, datetime_to_date=False, date_to_index=True)
#         # iterate over each model name and model in list of lists
#         for model_name in selected_model_names:
#
#             # =============================================================================
#             #             def add_prediction_interval(model, X_future, alpha, df):
#             #                 # calculate the prediction interval for the forecast data
#             #                 y_forecast = model.predict(X_future)
#             #                 mse = np.mean((model.predict(model.X) - model.y) ** 2)
#             #                 n = len(model.X)
#             #                 dof = n - 2
#             #                 t_value = stats.t.ppf(1 - alpha / 2, dof)
#             #                 y_std_err = np.sqrt(mse * (1 + 1 / n + (X_future - np.mean(model.X)) ** 2 / ((n - 1) * np.var(model.X))))
#             #                 lower_pi = y_forecast - t_value * y_std_err
#             #                 upper_pi = y_forecast + t_value * y_std_err
#             #                 # create a dataframe with the prediction interval and add it to the existing dataframe
#             #                 df['lower_pi'] = lower_pi
#             #                 df['upper_pi'] = upper_pi
#             #                 return df
#             # =============================================================================
#
#             if model_name == "Linear Regression":
#                 model = LinearRegression()
#                 # train the model on all data (X) for which we have data in forecast that user feature selected
#                 # e.g. if X originaly had August, but the forecast does not have August
#                 model.fit(X.loc[:, [col for col in feature_selection_user if col in df_future_dates.columns]], y)
#                 # forecast (y_hat with dtype numpy array)
#                 y_forecast = model.predict(X_future)
#                 # convert numpy array y_forecast to a dataframe
#                 df_forecast_lr = pd.DataFrame(y_forecast, columns=['forecast']).round(0)
#                 # create a dataframe with the DatetimeIndex as the index
#                 df_future_dates_only = future_dates.to_frame(index=False, name='date')
#                 # combine dataframe of date with y_forecast
#                 df_forecast_lr = copy_df_date_index(df_future_dates_only.join(df_forecast_lr), datetime_to_date=False,
#                                                     date_to_index=True)
#                 # =============================================================================
#                 #                 # Add the prediction interval to the forecast dataframe
#                 #                 alpha = 0.05  # Level of significance for the prediction interval
#                 #                 df_forecast_lr = add_prediction_interval(model, X_future, alpha, df_forecast_lr)
#                 # =============================================================================
#
#                 # create forecast model score card in streamlit
#                 with st.expander('ℹ️' + model_name + ' Forecast', expanded=True):
#                     my_header(f'{model_name}')
#                     # Create the forecast plot
#                     plot_forecast(y, df_forecast_lr, title='')
#                     # set note that maximum chosen date can only be up to length of input data with Linear Regression Model
#                     # st.caption('Note: Linear Regression Model maximum end date depends on length of input data')
#                     # show dataframe / output of forecast in streamlit linear regression
#                     st.dataframe(df_forecast_lr, use_container_width=True)
#                     download_csv_button(df_forecast_lr, my_file="forecast_linear_regression_results.csv")
#
#             if model_name == "SARIMAX":
#                 # =============================================================================
#                 # Define Model Parameters
#                 # =============================================================================
#                 order = (p, d, q)
#                 seasonal_order = (P, D, Q, s)
#
#                 # Define model on all data (X)
#                 # Assume df is your DataFrame with boolean columns - needed for SARIMAX model that does not handle boolean, but int instead
#                 # X bool to int dtypes
#                 bool_cols = X.select_dtypes(include=bool).columns
#                 X.loc[:, bool_cols] = X.loc[:, bool_cols].astype(int)
#
#                 # X_future bool to int dtypes
#                 bool_cols = X_future.select_dtypes(include=bool).columns
#                 X_future.loc[:, bool_cols] = X_future.loc[:, bool_cols].astype(int)
#
#                 # define model SARIMAX
#                 model = SARIMAX(endog=y,
#                                 order=(p, d, q),
#                                 seasonal_order=(P, D, Q, s),
#                                 exog=X.loc[:,
#                                      [col for col in feature_selection_user if col in df_future_dates.columns]],
#                                 enforce_invertibility=enforce_invertibility,
#                                 enforce_stationarity=enforce_stationarity
#                                 ).fit()
#
#                 # Forecast future values
#                 my_forecast_steps = (end_date_forecast - start_date_forecast.date()).days
#
#                 # y_forecast = model_fit.predict(start=start_date_forecast, end=(start_date_forecast + timedelta(days=len(X_future)-1)), exog=X_future)
#                 forecast_values = model.get_forecast(steps=my_forecast_steps, exog=X_future.iloc[:my_forecast_steps, :])
#
#                 # set the start date of forecasted value e.g. +7 days for new date
#                 start_date = max_date + timedelta(days=1)
#                 # create pandas series before appending to the forecast dataframe
#                 date_series = pd.date_range(start=start_date, end=None, periods=my_forecast_steps,
#                                             freq=forecast_freq_letter)
#
#                 # create dataframe
#                 df_forecast = pd.DataFrame()
#
#                 # add date series to forecasting pandas dataframe
#                 df_forecast['date'] = date_series.to_frame(index=False)
#
#                 # convert forecast to integers (e.g. round it)
#                 df_forecast[('forecast')] = forecast_values.predicted_mean.values.astype(int).round(0)
#
#                 # set 'date' as the index of the dataframe
#                 df_forecast_sarimax = copy_df_date_index(df_forecast)
#
#                 with st.expander('ℹ️ ' + model_name + ' Forecast', expanded=True):
#                     my_header(f'{model_name}')
#                     # Create the forecast plot
#                     plot_forecast(y, df_forecast_sarimax, title='')
#                     # set note that maximum chosen date can only be up to length of input data with Linear Regression Model
#                     # st.caption('Note: Linear Regression Model maximum end date depends on length of input data')
#                     # show dataframe / output of forecast in streamlit linear regression
#                     st.dataframe(df_forecast_sarimax, use_container_width=True)
#
#             if model_name == "Prophet":  # NOTE: Currently no X features included in this Prophet model
#                 forecast_prophet = pd.DataFrame()
#
#                 # prep data for specificalmodelly prophet model requirements, data should have ds column and y column
#                 y_prophet = preprocess_data_prophet(y)
#
#                 # define
#                 m = Prophet(changepoint_prior_scale=changepoint_prior_scale,
#                             seasonality_mode=seasonality_mode,
#                             seasonality_prior_scale=seasonality_prior_scale,
#                             holidays_prior_scale=holidays_prior_scale,
#                             yearly_seasonality=yearly_seasonality,
#                             weekly_seasonality=weekly_seasonality,
#                             daily_seasonality=daily_seasonality,
#                             interval_width=interval_width)
#
#                 # train the model on the entire dataset with set parameters
#                 m.fit(y_prophet)
#
#                 future = m.make_future_dataframe(periods=len(future_dates), freq='D')
#
#                 # Predict on the test set
#                 forecast_prophet = m.predict(future)
#
#                 # change name of yhat to forecast in df
#                 forecast_prophet['forecast'] = forecast_prophet['yhat'].round(0)
#                 # forecast_prophet['date'] = forecast_prophet['ds']
#                 forecast_prophet['date'] = forecast_prophet['ds']
#                 forecast_prophet = forecast_prophet[['date', 'forecast']]
#                 # set the date column as index column
#                 forecast_prophet = forecast_prophet.set_index('date')
#                 # cut off the insample forecast
#                 forecast_prophet = forecast_prophet[len(y):]
#                 with st.expander('ℹ️ ' + model_name + ' Forecast', expanded=True):
#                     my_header(f'{model_name}')
#                     plot_forecast(y, forecast_prophet, title='')
#                     # show dataframe / output of forecast in streamlit
#                     st.dataframe(forecast_prophet, use_container_width=True)
#                     download_csv_button(forecast_prophet, my_file="forecast_prophet_results.csv")
