# standard libraries
import streamlit as st
import pandas as pd

# local libraries
from fire_state import create_store, form_update, get_state, set_state
from app_pages.load import generate_demo_data


class SessionState:
    @staticmethod
    def initiate_global_variables():
        # ================================ TEST GLOBAL DATAFRAME AS VARIABLES ====================================
        create_store("DATAFRAMES", [("df_cleaned_outliers_with_index", pd.DataFrame())])

        # ================================ COLORS =====================================
        # store color scheme for app
        key1_chart_color, key2_chart_patterns, key3_run = create_store("COLORS", [("chart_color", "#F900EC"), ("chart_patterns", "#4d5466"), ("run", 0)])

        # ================================ LOAD PAGE =======================================
        if "df_raw" not in st.session_state:
            st.session_state["df_raw"] = pd.DataFrame()
            st.session_state.df_raw = generate_demo_data()
            df_graph = st.session_state.df_raw.copy(deep=True)

            # set minimum date
            df_min = st.session_state.df_raw.iloc[:, 0].min().date()

            # set maximum date
            df_max = st.session_state.df_raw.iloc[:, 0].max().date()

        if "df_graph" not in st.session_state:
            df_graph = st.session_state.df_raw.copy(deep=True)

        if "my_data_choice" not in st.session_state:
            st.session_state['my_data_choice'] = "Demo Data"

        # create session state for if demo data (default) or user uploaded a file
        create_store("DATA_OPTION", [("upload_new_data", False)])

        # Save the Data Choice of User
        key1_load, key2_load, key3_load = create_store("LOAD_PAGE", [
            ("my_data_choice", "Demo Data"),  # key1_load,
            ("user_data_uploaded", False),  # key2_load,
            ("uploaded_file_name", None)  # key3_load
        ])

        # ================================ EXPLORE PAGE ====================================
        key1_explore, key2_explore, key3_explore, key4_explore, key5_explore = create_store("EXPLORE_PAGE", [
            ("lags_acf", min(30, int((len(st.session_state.df_raw) - 1)))),  # key1_explore
            ("lags_pacf", min(30, int((len(st.session_state.df_raw) - 2) / 2))),  # key2_explore
            ("default_pacf_method", "yw"),  # key3_explore
            ("order_of_differencing_series", "Original Series"),  # key4_explore
            ("run", 0)  # key5_explore
        ])

        # HISTOGRAM PARAMETER FOR STORING RADIO_BUTTON USER PREFERENCE
        key_hist = create_store("HIST", [("histogram_freq_type", "Absolute"), ("run", 0)])

        # ================================ CLEAN PAGE ======================================
        # set the random state
        random_state = 10

        # for missing values custom fill value can be set by user (initiate variable with None)
        custom_fill_value = None

        if 'freq_dict' not in st.session_state:
            st.session_state['freq_dict'] = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}

        if 'freq' not in st.session_state:
            # assume frequency is daily data for now -> expand to automated frequency detection later
            st.session_state['freq'] = 'Daily'

        # define keys and store them in memory with their associated values
        key1_missing, key2_missing, key3_missing, key4_missing = create_store("CLEAN_PAGE_MISSING",
                                                                              [
                                                                                  ("missing_fill_method", "Backfill"),
                                                                                  # key1
                                                                                  ("missing_custom_fill_value", "1"),
                                                                                  # key2
                                                                                  ("data_frequency", 'Daily'),  # key3
                                                                                  ("run", 0)  # key4
                                                                              ]
                                                                              )

        key1_outlier, key2_outlier, key3_outlier, key4_outlier, key5_outlier, key6_outlier, key7_outlier, key8_outlier = create_store(
            "CLEAN_PAGE", [
                ("outlier_detection_method", "None"),  # key1_outlier
                ("outlier_isolationforest_contamination", 0.01),  # key2_outlier
                ("outlier_zscore_threshold", 3.0),  # key3_outlier
                ("outlier_iqr_q1", 25.0),  # key4_outlier
                ("outlier_iqr_q3", 75.0),  # key5_outlier
                ("outlier_iqr_multiplier", 1.5),  # key6_outlier
                ("outlier_replacement_method", "Interpolation"),  # key7_outlier
                ("run", 0)  # key8_outlier
            ])
        # ================================ ENGINEER PAGE ===================================
        key1_engineer, key2_engineer, key3_engineer, key4_engineer, key5_engineer, key6_engineer, key7_engineer, key8_engineer = create_store(
            "ENGINEER_PAGE",
            [
                ("calendar_dummies_checkbox", True),  # key1_engineer
                ("calendar_holidays_checkbox", True),  # key2_engineer
                ("special_calendar_days_checkbox", True),  # key3_engineer
                ("dwt_features_checkbox", False),  # key4_engineer
                ("wavelet_family_selectbox", "db4"),  # key5_engineer
                ("wavelet_level_decomposition_selectbox", 3),  # key6_engineer
                ("wavelet_window_size_slider", 7),  # key7_engineer
                ("run", 0)  # key8_engineer
            ]
        )

        key1_engineer_var, key2_engineer_var, key3_engineer_var, key4_engineer_var, key5_engineer_var, key6_engineer_var, key7_engineer_var, key8_engineer_var, key9_engineer_var, key10_engineer_var, \
            key11_engineer_var, key12_engineer_var, key13_engineer_var, key14_engineer_var, key15_engineer_var, key16_engineer_var, key17_engineer_var = create_store(
            "ENGINEER_PAGE_VARS",
            [("year_dummies_checkbox", True),  # key1_engineer_var
             ("month_dummies_checkbox", True),  # key2_engineer_var
             ("day_dummies_checkbox", True),  # key3_engineer_var
             ("jan_sales", True),  # key4_engineer_var
             ("val_day_lod", True),  # key5_engineer_var
             ("val_day", True),  # key6_engineer_var
             ("mother_day_lod", True),  # key7_engineer_var
             ("mother_day", True),  # key8_engineer_var
             ("father_day_lod", True),  # key9_engineer_var
             ("pay_days", True),  # key10_engineer_var
             ("father_day", True),  # key11_engineer_var
             ("black_friday_lod", True),  # key12_engineer_var
             ("black_friday", True),  # key13_engineer_var
             ("cyber_monday", True),  # key14_engineer_var
             ("christmas_day", True),  # key15_engineer_var
             ("boxing_day", True),  # key16_engineer_var
             ("run", 0)  # key17_engineer_var
             ]
        )

        # set initial country holidays to country_code US and country_name "United States of America"
        key1_engineer_page_country, key2_engineer_page_country, key3_engineer_page_country = create_store(
            "ENGINEER_PAGE_COUNTRY_HOLIDAY", [
                ("country_name", "United States of America"),
                # key1_engineer_page_country
                ("country_code", "US"),
                # key2_engineer_page_country
                ("run", 0)])  # key3_engineer_page_country

        # THIS SESSION STATE IS USED TO DEAL WITH RESETTING DEFAULT FEATURE SELECTION OR TO KEEP USER SELECTION ON SELECT PAGE
        create_store("ENGINEER_PAGE_FEATURES_BTN", [("engineering_form_submit_btn", False)])

        # ================================ PREPARE PAGE ====================================
        # define my_insample_forecast_steps used for train/test split
        if 'percentage' not in st.session_state:
            st.session_state['percentage'] = 20

        if 'steps' not in st.session_state:
            st.session_state['steps'] = 1

        # save user choice in session state of in sample test-size percentage
        if 'insample_forecast_perc' not in st.session_state:
            st.session_state['insample_forecast_perc'] = 20

        # save user choice in session state of in sample test-size in steps (e.g. days if daily frequency data)
        if 'insample_forecast_steps' not in st.session_state:
            st.session_state['insample_forecast_steps'] = 1

        # set default value for normalization method to 'None'
        if 'normalization_choice' not in st.session_state:
            st.session_state['normalization_choice'] = 'None'

        # set the normalization and standardization to default None which has index 0
        key1_prepare_normalization, key2_prepare_standardization, key3_prepare = create_store("PREPARE", [
            ("normalization_choice", "None"),  # key1_prepare_normalization
            ("standardization_choice", "None"),  # key2_prepare_standardization
            ("run", 0)])  # key3_prepare

        # ================================ SELECT PAGE ====================================
        # TO DEAL WITH EITHER FEATURE SELECTION METHODS FEATURES OR TO REVERT TO USER SELECTION
        create_store("SELECT_PAGE_BTN_CLICKED", [("rfe_btn", False),
                                                 ("mifs_btn", False),
                                                 ("pca_btn", False),
                                                 ("correlation_btn", False)])
        # FEATURE SELECTION BY USER
        key1_select_page_user_selection, key2_select_page_user_selection = create_store("SELECT_PAGE_USER_SELECTION", [
            ("feature_selection_user", []),  # key1_select_page_user_selection
            ("run", 0)])  # key2_select_page_user_selection

        # PCA
        key1_select_page_pca, key2_select_page_pca = create_store("SELECT_PAGE_PCA", [
            ("num_features_pca", 5),
            # key1_select_page_pca # default of PCA features set to minimum of either 5 or the number of independent features in the dataframe
            ("run", 0)])  # key2_select_page_pca

        # RFE
        key1_select_page_rfe, key2_select_page_rfe, key3_select_page_rfe, key4_select_page_rfe, key5_select_page_rfe, key6_select_page_rfe = create_store(
            "SELECT_PAGE_RFE", [
                ("num_features_rfe", 5),  # key1_select_page_rfe
                ("estimator_rfe", "Linear Regression"),  # key2_select_page_rfe
                ("timeseriessplit_value_rfe", 5),  # key3_select_page_rfe
                ("num_steps_rfe", 1),  # key4_select_page_rfe
                ("duplicate_ranks_rfe", True),  # key5_select_page_rfe
                ("run", 0)])  # key6_select_page_rfe

        # MIFS
        key1_select_page_mifs, key2_select_page_mifs = create_store("SELECT_PAGE_MIFS", [
            ("num_features_mifs", 5),  # key1_select_page_mifs
            ("run", 0)])  # key2_select_page_mifs

        # CORR
        key1_select_page_corr, key2_select_page_corr, key3_select_page_corr = create_store("SELECT_PAGE_CORR", [
            ("corr_threshold", 0.8),  # key1_select_page_corr
            ("selected_corr_model", 'Linear Regression'),  # key2_select_page_corr
            ("run", 0)])  # key3_select_page_corr

        # ================================ TRAIN PAGE ===================================
        # TRAIN MENU TEST
        if 'train_models_btn' not in st.session_state:
            st.session_state['train_models_btn'] = False

        # =============================================================================
        #     if 'selected_model_info' not in st.session_state:
        #         st.session_state['selected_model_info'] = '-'
        # =============================================================================

        # set session states for the buttons when models are trained,
        # to expand dataframe below graph
        create_store("TRAIN", [("naive_model_btn_show", False),
                               ("naive_model_btn_hide", False),
                               ("linreg_model_btn_show", False),
                               ("sarimax_model_btn_show", False),
                               ("prophet_model_btn_show", False)])

        # Training models parameters
        key1_train, key2_train, key3_train, key4_train, key5_train, key6_train, key7_train, key8_train, key9_train, key10_train, \
            key11_train, key12_train, key13_train, key14_train, key15_train, key16_train, key17_train, key18_train, key19_train, key20_train, \
            key21_train, key22_train, key23_train, key24_train, key25_train, key26_train, key27_train, key28_train, key29_train, key30_train, \
            key31_train, key32_train = create_store("TRAIN_PAGE", [
            ("my_conf_interval", 80),  # key1_train
            ("naive_checkbox", False),  # key2_train
            ("linreg_checkbox", False),  # key3_train
            ("sarimax_checkbox", False),  # key4_train
            ("prophet_checkbox", False),  # key5_train
            ("selected_models", []),  # key6_train
            ("lag", 'Week'),  # key7_train
            ("custom_lag_value", 5),  # key8_train
            ("p", 1),  # key9_train
            ("d", 1),  # key10_train
            ("q", 1),  # key11_train
            ("P", 1),  # key12_train
            ("D", 1),  # key13_train
            ("Q", 1),  # key14_train
            ("s", 7),  # key15_train
            ("enforce_stationarity", True),  # key16_train
            ("enforce_invertibility", True),  # key17_train
            ("horizon_option", 30),  # key18_train
            ("changepoint_prior_scale", 0.01),  # key19_train
            ("seasonality_mode", "multiplicative"),  # key20_train
            ("seasonality_prior_scale", 0.01),  # key21_train
            ("holidays_prior_scale", 0.01),  # key22_train
            ("yearly_seasonality", True),  # key23_train
            ("weekly_seasonality", True),  # key24_train
            ("daily_seasonality", True),  # key25_train
            ("results_df", pd.DataFrame(
                columns=['model_name', 'predicted', 'mape', 'smape', 'rmse', 'r2', 'features', 'model_settings'])),
            # key26_train
            ("prophet_holidays", True),  # key27_train
            ("include_feature_selection", 'Yes'),  # key28_train
            ("size_rolling_window_naive_model", 7),  # key29_train
            ("method_rolling_window_naive_model", "Mean"),  # key30_train
            ("method_baseline_naive_model", "Mean"),  # key31_train
            ("run", 0)  # key32_train
        ])

        # ================================ EVALUATE PAGE ===================================
        # create an empty dictionary to store the results of the models
        # that I call after I train the models to display on sidebar under hedaer "Evaluate Models"
        metrics_dict = {}

        # Initialize results_df in global scope that has model test evaluation results
        results_df = pd.DataFrame(
            columns=['model_name', 'predicted', 'mape', 'smape', 'rmse', 'r2', 'features', 'model_settings'])

        if 'results_df' not in st.session_state:
            st.session_state['results_df'] = pd.DataFrame(
                columns=['model_name', 'predicted', 'mape', 'smape', 'rmse', 'r2', 'features', 'model_settings'])

            # save user's chosen metric in persistent session state - initiate default metric (MAPE)
        key1_evaluate, key2_evaluate = create_store("EVALUATE_PAGE", [
            ("selected_metric", 'Mean Absolute Percentage Error'),  # key1_evaluate
            ("run", 0)])  # key2_evaluate

        # ================================ TUNE PAGE ===================================
        #
        #

        # ================================ FORECAST PAGE ===================================
        #
        #

        # Logging
        print('ForecastGenie Print: Loaded Global Variables')

        return (key1_chart_color, key2_chart_patterns, key3_run,
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
                key1_select_page_rfe, key2_select_page_rfe, key3_select_page_rfe, key4_select_page_rfe, key5_select_page_rfe, key6_select_page_rfe,
                key1_select_page_mifs,
                key1_select_page_corr, key2_select_page_corr,
                key1_train, key2_train, key3_train, key4_train, key5_train, key6_train, key7_train, key8_train, key9_train, key10_train,
                key11_train, key12_train, key13_train, key14_train, key15_train, key16_train, key17_train, key18_train, key19_train, key20_train,
                key21_train, key22_train, key23_train, key24_train, key25_train, key26_train, key27_train, key28_train, key29_train, key30_train, key31_train, key32_train,
                key1_evaluate, key2_evaluate)

    @staticmethod
    def reset_session_states():
        """
        Resets the session states for variables by deleting all items in Session state, except for 'data_choice' and 'my_data_choice'.

        This function is useful when a new file is uploaded, and you want to clear the existing session state.

        Returns:
            None

        Example:
            reset_session_states()
        """
        # Delete all the items in Session state
        for key in st.session_state.keys():
            ############### NEW TEST ############################
            # FIX: review need of 2 session states
            if key == 'data_choice' or key == 'my_data_choice' or key == "__LOAD_PAGE-my_data_choice__" or key == "__LOAD_PAGE-user_data_uploaded__":
                pass
            else:
                del st.session_state[key]

        # reset session states to default -> initiate global variables
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
         key11_train, key12_train, key13_train, key14_train, key15_train, key16_train, key17_train, key18_train,
         key19_train,
         key20_train,
         key21_train, key22_train, key23_train, key24_train, key25_train, key26_train, key27_train, key28_train,
         key29_train,
         key30_train, key31_train, key32_train,
         key1_evaluate, key2_evaluate) = SessionState.initiate_global_variables()
