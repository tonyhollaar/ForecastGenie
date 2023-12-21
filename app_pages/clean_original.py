
# =============================================================================
#    _____ _      ______          _   _
#   / ____| |    |  ____|   /\   | \ | |
#  | |    | |    | |__     /  \  |  \| |
#  | |    | |    |  __|   / /\ \ | . ` |
#  | |____| |____| |____ / ____ \| |\  |
#   \_____|______|______/_/    \_\_| \_|
#
# =============================================================================
# CLEAN PAGE
if menu_item == 'Clean' and sidebar_menu_item == 'HOME':
    # =============================================================================
    # Define Progress bar
    # =============================================================================
    progress_text = "Dusting off cleaning methods... Please wait!"
    my_bar = st.progress(0, text=progress_text)

    with st.sidebar:
        my_title(f"""{icons["clean_icon"]}""", "#3b3b3b", gradient_colors="#440154, #2C2A6B, #FDE725")

        with st.form('data_cleaning'):
            my_text_paragraph('Handling Missing Data')

            # get user input for filling method
            fill_method = st.selectbox(label='*Select filling method for missing values:*',
                                       options=['Backfill', 'Forwardfill', 'Mean', 'Median', 'Mode', 'Custom'],
                                       key=key1_missing)

            if fill_method == 'Custom':
                custom_fill_value = st.text_input(label='*Insert custom value to replace missing value(s) with:*',
                                                  key=key2_missing,
                                                  help='Please enter your **`custom value`** to impute missing values with, you can use a whole number or decimal point number')

            # Define a dictionary of possible frequencies and their corresponding offsets
            freq_dict = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}

            # infer and return the original data frequency e.g. 'M' and name e.g. 'Monthly'
            original_freq, original_freq_name = determine_df_frequency(st.session_state['df_raw'],
                                                                       column_name='date')

            # determine the position of original frequency in the freq_dict to use as the index as chosen frequency in drop-down selectbox for user when loading page
            position = list(freq_dict.values()).index(original_freq)

            # Ask the user to select the frequency of the data
            # set the frequency that is inferred from the data itself with custom function to pre-selected e.g. index = ...
            freq = st.selectbox('*Select the frequency of the data:*',
                                list(freq_dict.keys()),
                                key=key3_missing)

            col1, col2, col3 = st.columns([4, 4, 4])
            with col2:
                data_cleaning_btn = st.form_submit_button("Submit", type="secondary", on_click=form_update,
                                                          args=("CLEAN_PAGE_MISSING",))

    # =========================================================================
    # 1. IMPUTE MISSING DATES WITH RESAMPLE METHOD
    # =========================================================================
    my_bar.progress(25, text='checking for missing dates...')

    # Apply function to resample missing dates based on user set frequency
    df_cleaned_dates = resample_missing_dates(df=st.session_state.df_raw,
                                              freq_dict=freq_dict,
                                              original_freq=original_freq,
                                              freq=freq)

    # =========================================================================
    # 2. IMPUTE MISSING VALUES WITH FILL METHOD
    # =========================================================================
    my_bar.progress(50, text='checking for missing values...')

    df_clean = my_fill_method(df_cleaned_dates, fill_method, custom_fill_value)
    # Convert datetime column to date AND set date column as index column
    df_clean_show = copy_df_date_index(df_clean, datetime_to_date=True, date_to_index=True)
    # =========================================================================

    # create user tabs
    tab1_clean, tab2_clean = st.tabs(['missing data', 'outliers'])

    with tab1_clean:

        with st.expander('', expanded=True):
            my_text_header('Handling missing data')

            # show_lottie_animation(url = "./images/ufo.json", key='jumping_dots', width=300, height=300, speed = 1, col_sizes=[2,4,2])

            # check if there are no dates skipped for frequency e.g. daily data missing days in between dates
            missing_dates = pd.date_range(start=st.session_state.df_raw['date'].min(),
                                          end=st.session_state.df_raw['date'].max()).difference(
                st.session_state.df_raw['date'])

            # check if there are no missing values (NaN) in dataframe
            missing_values = st.session_state.df_raw.iloc[:, 1].isna().sum()

            # Plot missing values matrix with custom function
            plot_missing_values_matrix(df=df_cleaned_dates)

            # check if in continous time-series dataset no dates are missing in between
            if missing_dates.shape[0] == 0:
                st.success('Pweh 😅, no dates were skipped in your dataframe!')
            else:
                st.warning(
                    f'💡 **{missing_dates.shape[0]}** dates were skipped in your dataframe, don\'t worry though! I will **fix** this by **imputing** the dates into your cleaned dataframe!')
            if missing_values != 0 and fill_method == 'Backfill':
                st.warning(
                    f'💡 **{missing_values}** missing values are filled with the next available value in the dataset (i.e. backfill method), optionally you can change the *filling method* and press **\"Submit\"** from the sidebar menu.')
            elif missing_values != 0 and fill_method != 'Custom':
                st.warning(
                    f'💡 **{missing_values}** missing values are replaced by the **{fill_method}**, optionally you can change the *filling method* and press **\"Submit\"** from the sidebar menu.')
            elif missing_values != 0 and fill_method == 'Custom':
                st.warning(
                    f'💡 **{missing_values}** missing values are replaced by custom value **{custom_fill_value}**, optionally you can change the *filling method* and press **\"Submit\"** from the sidebar menu.')

            col1, col2, col3, col4, col5 = st.columns([2, 0.5, 2, 0.5, 2])
            with col1:
                df_graph = copy_df_date_index(my_df=st.session_state['df_raw'],
                                              datetime_to_date=True,
                                              date_to_index=True)

                highlighted_df = df_graph.style.highlight_null(color='yellow').format(precision=2)

                my_subheader('Original DataFrame', my_style="#333333", my_size=6)

                st.dataframe(highlighted_df, use_container_width=True)

            with col2:
                st.markdown(icons["arrow_right_icon"], unsafe_allow_html=True)

            with col3:
                my_subheader('Skipped Dates', my_style="#333333", my_size=6)

                # Convert the DatetimeIndex to a dataframe with a single column named 'Date'
                df_missing_dates = pd.DataFrame({'Skipped Dates': missing_dates})

                # change datetime to date
                df_missing_dates['Skipped Dates'] = df_missing_dates['Skipped Dates'].dt.date

                # show missing dates
                st.dataframe(df_missing_dates, use_container_width=True)

                # Display the dates and the number of missing values associated with them
                my_subheader('Missing Values', my_style="#333333", my_size=6)
                # Filter the DataFrame to include only rows with missing values
                missing_df = copy_df_date_index(st.session_state.df_raw.loc[st.session_state.df_raw.iloc[:,
                                                                            1].isna(), st.session_state.df_raw.columns],
                                                datetime_to_date=True, date_to_index=True)
                st.write(missing_df)

            with col4:
                st.markdown(icons["arrow_right_icon"], unsafe_allow_html=True)

            with col5:
                my_subheader('Cleaned Dataframe', my_style="#333333", my_size=6)

                # Show the cleaned dataframe with if needed dates inserted if skipped to NaN and then the values inserted with impute method user selected backfill/forward fill/mean/median
                st.dataframe(df_clean_show, use_container_width=True)

            # Create and show download button in streamlit to user to download the dataframe with imputations performed to missing values
            download_csv_button(df_clean_show, my_file="df_imputed_missing_values.csv", set_index=True,
                                help_message='Download cleaned dataframe to .CSV')

    my_bar.progress(75, text='loading outlier detection methods and cleaning methods')
    #########################################################
    # Handling Outliers
    #########################################################
    with st.sidebar:
        # call the function to get the outlier form in Streamlit 'Handling Outliers'
        outlier_form(key1_outlier, key2_outlier, key3_outlier, key4_outlier, key5_outlier, key6_outlier, key7_outlier)

        # retrieve the parameters from default in-memory value or overwritten by user from streamlit widgets
        outlier_detection_method = get_state("CLEAN_PAGE", "outlier_detection_method")
        outlier_zscore_threshold = get_state("CLEAN_PAGE", "outlier_zscore_threshold")
        outlier_iqr_q1 = get_state("CLEAN_PAGE", "outlier_iqr_q1")
        outlier_iqr_q3 = get_state("CLEAN_PAGE", "outlier_iqr_q1")
        outlier_replacement_method = get_state("CLEAN_PAGE", "outlier_replacement_method")
        outlier_isolationforest_contamination = get_state("CLEAN_PAGE", "outlier_isolationforest_contamination")
        outlier_iqr_multiplier = get_state("CLEAN_PAGE", "outlier_iqr_multiplier")

        # Create form and sliders for outlier detection and handling
        ##############################################################################
        df_cleaned_outliers, outliers = handle_outliers(df_clean_show,
                                                        outlier_detection_method,
                                                        outlier_zscore_threshold,
                                                        outlier_iqr_q1,
                                                        outlier_iqr_q3,
                                                        outlier_replacement_method,
                                                        outlier_isolationforest_contamination,
                                                        random_state,
                                                        outlier_iqr_multiplier)

        with tab2_clean:
            with st.expander('', expanded=True):
                # Set page subheader with custum function
                my_text_header('Handling outliers')
                my_text_paragraph('Outlier Plot')
                if outliers is not None and any(outliers):

                    outliers_df = copy_df_date_index(df_clean[outliers], datetime_to_date=True,
                                                     date_to_index=True).add_suffix('_outliers')

                    df_cleaned_outliers = df_cleaned_outliers.add_suffix('_outliers_replaced')

                    # inner join two dataframes of outliers and imputed values
                    outliers_df = outliers_df.join(df_cleaned_outliers, how='inner', rsuffix='_outliers_replaced')

                    ## OUTLIER FIGURE CODE
                    fig_outliers = go.Figure()
                    fig_outliers.add_trace(
                        go.Scatter(
                            x=df_clean['date'],
                            y=df_clean.iloc[:, 1],
                            mode='markers',
                            name='Before',
                            marker=dict(color='#440154'), opacity=0.5
                        )
                    )
                    # add scatterplot
                    fig_outliers.add_trace(
                        go.Scatter(
                            x=df_cleaned_outliers.index,
                            y=df_cleaned_outliers.iloc[:, 0],
                            mode='markers',
                            name='After',
                            marker=dict(color='#45B8AC'), opacity=1
                        )
                    )

                    df_diff = df_cleaned_outliers.loc[outliers]
                    # add scatterplot
                    fig_outliers.add_trace(go.Scatter(
                        x=df_diff.index,
                        y=df_diff.iloc[:, 0],
                        mode='markers',
                        name='Outliers After',
                        marker=dict(color='#FFC300'), opacity=1
                    )
                    )

                    # show the outlier plot
                    st.session_state['fig_outliers'] = fig_outliers

                    st.plotly_chart(fig_outliers, use_container_width=True)

                    #  show the dataframe of outliers
                    st.info(
                        f'ℹ️ You replaced **{len(outliers_df)} outlier(s)** with their respective **{outlier_replacement_method}(s)** utilizing **{outlier_detection_method}**.')

                    # Apply the color scheme to the dataframe, round values by 2 decimals and display it in streamlit using full size of expander window
                    st.dataframe(outliers_df.style.format("{:.2f}").apply(highlight_cols, axis=0),
                                 use_container_width=True)

                    # add download button for user to be able to download outliers
                    download_csv_button(outliers_df, my_file="df_outliers.csv", set_index=True,
                                        help_message='Download outlier dataframe to .CSV', my_key='df_outliers')

                # if outliers are NOT found or None is selected as outlier detection method
                # ... run code...
                # Show scatterplot data without outliers
                else:
                    vertical_spacer(1)
                    fig_no_outliers = go.Figure()
                    fig_no_outliers.add_trace(go.Scatter(x=df_clean['date'],
                                                         y=df_clean.iloc[:, 1],
                                                         mode='markers',
                                                         name='Before',
                                                         marker=dict(color='#440154'), opacity=0.5))

                    st.plotly_chart(fig_no_outliers, use_container_width=True)
                    my_text_paragraph(f'No <b> outlier detection </b> or <b> outlier replacement </b> method selected.',
                                      my_font_size='14px')

    my_bar.progress(100, text='100%')
    my_bar.empty()
else:
    ##################################################################################################################################
    # ************************* PREPROCESSING DATA - CLEANING MISSING AND IF USER SELECTED IT ALSO IMPUTE OUTLIERS ********************
    ##################################################################################################################################
    # Retrieve the date frequency of the timeseries
    freq_dict = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}

    # Infer and return the original data frequency e.g. 'M' and name e.g. 'Monthly'
    original_freq, original_freq_name = determine_df_frequency(st.session_state['df_raw'], column_name='date')

    # =========================================================================
    # 1. IMPUTE MISSING DATES WITH RESAMPLE METHOD
    # ******************************************************************
    # Apply function to resample missing dates based on user set frequency
    df_cleaned_dates = resample_missing_dates(df=st.session_state['df_raw'],
                                              freq_dict=freq_dict,
                                              original_freq=original_freq,
                                              freq=st.session_state['freq'])

    # =========================================================================
    # 2. IMPUTE MISSING VALUES WITH FILL METHOD
    # =========================================================================
    df_clean = my_fill_method(df=df_cleaned_dates,
                              fill_method=get_state('CLEAN_PAGE_MISSING', 'missing_fill_method'),
                              custom_fill_value=get_state('CLEAN_PAGE_MISSING', 'missing_custom_fill_value'))

    df_clean_show = copy_df_date_index(df_clean, datetime_to_date=True, date_to_index=True)

    # =========================================================================
    # 3. IMPUTE OUTLIERS DETECTED WITH OUTLIER REPLACEMENT METHOD
    # =========================================================================
    df_cleaned_outliers, outliers = handle_outliers(data=df_clean_show,
                                                    method=get_state('CLEAN_PAGE', 'outlier_detection_method'),
                                                    outlier_threshold=get_state('CLEAN_PAGE',
                                                                                'outlier_zscore_threshold'),
                                                    q1=get_state('CLEAN_PAGE', 'outlier_iqr_q1'),
                                                    q3=get_state('CLEAN_PAGE', 'outlier_iqr_q3'),
                                                    outlier_replacement_method=get_state('CLEAN_PAGE',
                                                                                         'outlier_replacement_method'),
                                                    contamination=get_state('CLEAN_PAGE',
                                                                            'outlier_isolationforest_contamination'),
                                                    random_state=random_state,
                                                    # defined variable random_state top of script e.g. 10
                                                    iqr_multiplier=get_state('CLEAN_PAGE', 'outlier_iqr_multiplier'))

    # create a copy of the dataframe
    df_cleaned_outliers_with_index = df_cleaned_outliers.copy(deep=True)

    # reset the index again to have index instead of date column as index for further processing
    df_cleaned_outliers_with_index.reset_index(inplace=True)

    # convert 'date' column to datetime in DataFrame
    df_cleaned_outliers_with_index['date'] = pd.to_datetime(df_cleaned_outliers_with_index['date'])

    # =========================================================================
    # 4. SAVE TRANSFORMED (CLEANED) DATAFRAME TO SESSION STATE (IN-MEMORY)
    # =========================================================================
    # TEST replaced if statement with always updating the dataframe in session state e.g. if user changed from demo data to uploading data
    st.session_state['df_cleaned_outliers_with_index'] = df_cleaned_outliers_with_index