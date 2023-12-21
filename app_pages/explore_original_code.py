    # =============================================================================
    # Define Progress bar
    # =============================================================================
    progress_text = "Preparing your exploratory data analysis... Please wait!"
    my_bar = st.progress(0, text=progress_text)
    # with st.spinner('Loading your summary, insights, patterns, statistical tests, lag analysis...'):

    # define tabs for page sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["overview",
                                            "insights",
                                            "patterns",
                                            "statistical tests",
                                            "lag analysis"])

    ####################################################
    # Sidebar EDA parameters / buttons
    ####################################################
    with st.sidebar:
        my_title(f'{icons["explore_icon"]}', my_background_color="#3b3b3b", gradient_colors="#217CD0,#555555")

        with st.form('ljung-box'):
            my_text_paragraph('White Noise')

            lag1_ljung_box = st.number_input(label='*Enter maximum lag:*',
                                             min_value=1,
                                             value=min(24, len(st.session_state.df_raw) - 2),
                                             max_value=len(st.session_state.df_raw) - 2,
                                             key='lag1_ljung_box',
                                             help='the lag parameter in the Ljung-Box test determines the number of time periods over which the autocorrelation of residuals is evaluated to assess the presence of significant autocorrelation in the time series.')

            col1, col2, col3 = st.columns([5, 4, 4])
            with col2:
                # create button in sidebar for the white noise (Ljung-Box) test
                vertical_spacer(1)
                ljung_box_btn = st.form_submit_button("Submit", type="secondary")

        # Autocorrelation parameters form
        with st.form('autocorrelation'):
            # Create sliders in sidebar for the parameters of PACF Plot
            my_text_paragraph('Autocorrelation ')
            col1, col2, col3 = st.columns([4, 1, 4])

            # If dataset is very small (less then 31 rows), then update the key1_explore and key2_explore
            if len(st.session_state['df_raw']) < 31:
                set_state("EXPLORE_PAGE", ("lags_acf", int((len(st.session_state['df_raw']) - 1))))
                set_state("EXPLORE_PAGE", ("lags_pacf", int((len(st.session_state['df_raw']) - 1) / 2)))

            # create slider for number of lags for ACF Plot
            nlags_acf = st.slider(label="*Lags ACF*",
                                  min_value=1,
                                  max_value=(len(st.session_state['df_raw']) - 1),
                                  key=key1_explore)

            col1, col2, col3 = st.columns([4, 1, 4])
            with col1:
                # create slider for number of lags for PACF Plot
                nlags_pacf = st.slider(label="*Lags PACF*",
                                       min_value=1,
                                       max_value=int((len(st.session_state['df_raw']) - 2) / 2),
                                       key=key2_explore)
            with col3:
                # create dropdown menu for method to calculate the PACF Plot
                method_pacf = st.selectbox(label="*Method PACF*",
                                           options=['ols', 'ols-inefficient', 'ols-adjusted', 'yw', 'ywa', 'ld',
                                                    'ywadjusted', 'yw_adjusted', 'ywm', 'ywmle', 'yw_mle', 'lda',
                                                    'ldadjusted', 'ld_adjusted', 'ldb', 'ldbiased', 'ld_biased'],
                                           key=key3_explore)

            # create dropdown menu to select if you want the original series or differenced series that will also impact the ACF & PACF Plots
            # next to having a preview in sidebar with the differenced series plotted if user selects 1st ,2nd or 3rd order difference
            selection = st.selectbox('*Apply Differencing [Optional]:*',
                                     options=['Original Series', 'First Order Difference', 'Second Order Difference',
                                              'Third Order Difference'],
                                     key=key4_explore)

            col1, col2, col3 = st.columns([5, 4, 4])
            with col2:
                # create button in sidebar for the ACF and PACF Plot Parameters
                vertical_spacer(1)
                acf_pacf_btn = st.form_submit_button("Submit", type="secondary", on_click=form_update,
                                                     args=('EXPLORE_PAGE',))

    ####################################################
    # Explore MAIN PAGE (EDA)
    ####################################################
    with tab1:
        # my_title(f'{explore_icon} Exploratory Data Analysis ', my_background_color="#217CD0", gradient_colors="#217CD0,#555555")
        # create expandable card with data exploration information
        with st.expander('', expanded=True):
            col0, col1, col2, col3 = st.columns([18, 90, 8, 1])
            with col2:
                my_chart_color = st.color_picker(label='Color',
                                                 value=get_state("COLORS", "chart_patterns"),
                                                 label_visibility='collapsed',
                                                 help='Set the **`color`** of the charts and styling elements. It will revert back to the **default** color when switching app_pages.')

            #############################################################################
            # Quick Summary Results
            #############################################################################
            with col1:
                my_text_header('Quick Summary')

            ## show the flashcard with the quick summary results on page
            # eda_quick_summary(my_chart_color)

            col1, col2, col3 = st.columns([7, 120, 1])
            with col2:
                vertical_spacer(1)

                create_flipcard_quick_summary(header_list='',
                                              paragraph_list_front='',
                                              paragraph_list_back='',
                                              font_family='Arial',
                                              font_size_front='16px',
                                              font_size_back='16px',
                                              image_path_front_card='./images/quick_summary.png',
                                              my_chart_color='#FFFFFF')

            # show button and if clicked, show dataframe
            col1, col2, col3 = st.columns([100, 50, 95])
            with col2:
                placeholder = st.empty()
                # create button (enabled to click e.g. disabled=false with unique key)
                btn_summary_stats = placeholder.button('Show Details', disabled=False,
                                                       key="summary_statistics_show_btn")

            # if button is clicked run below code
            if btn_summary_stats == True:
                # display button with text "click me again", with unique key
                placeholder.button('Hide Details', disabled=False, key="summary_statistics_hide_btn")

                # Display summary statistics table
                summary_stats_df = display_summary_statistics(st.session_state.df_raw)

                st.dataframe(summary_stats_df, use_container_width=True)

                download_csv_button(summary_stats_df, my_file="summary_statistics.csv",
                                    help_message='Download your Summary Statistics Dataframe to .CSV',
                                    my_key="summary_statistics_download_btn")

            vertical_spacer(1)

            # Show Summary Statistics and statistical test results of dependent variable (y)
            summary_statistics_df = create_summary_df(data=st.session_state.df_raw.iloc[:, 1])

    my_bar.progress(20, text='loading quick insights...')
    #######################################
    # Quick Insights Results
    #####################################
    with tab2:
        with st.expander('', expanded=True):
            # old metrics card
            # eda_quick_insights(df=summary_statistics_df, my_string_column='Label', my_chart_color = my_chart_color)

            col1, col2, col3 = st.columns([7, 120, 1])
            with col2:
                create_flipcard_quick_insights(1, header_list=[''],
                                               paragraph_list_front=[''],
                                               paragraph_list_back=[''],
                                               font_family='Arial',
                                               font_size_front='12px',
                                               font_size_back='12px',
                                               image_path_front_card='./images/futuristic_city_robot.png',
                                               df=summary_statistics_df,
                                               my_string_column='Label',
                                               my_chart_color='#FFFFFF')

            # have button available for user and if clicked it expands with the dataframe
            col1, col2, col3 = st.columns([100, 50, 95])
            with col2:
                placeholder = st.empty()

                # create button (enabled to click e.g. disabled=false with unique key)
                btn_insights = placeholder.button('Show Details', disabled=False, key="insights_statistics_show_btn")

                vertical_spacer(1)

            # if button is clicked run below code
            if btn_insights == True:
                # display button with text "click me again", with unique key
                placeholder.button('Hide Details', disabled=False, key="insights_statistics_hide_btn")

                st.dataframe(summary_statistics_df, use_container_width=True)

                download_csv_button(summary_statistics_df, my_file="insights.csv",
                                    help_message='Download your Insights Dataframe to .CSV',
                                    my_key="insights_download_btn")

    my_bar.progress(40, text='loading patterns')
    with tab3:
        #############################################################################
        # Call function for plotting Graphs of Seasonal Patterns D/W/M/Q/Y in Plotly Charts
        #############################################################################
        with st.expander('', expanded=True):
            # Update layout
            my_text_header('Patterns')

            # show all graphs with patterns in streamlit
            plot_overview(df=st.session_state.df_raw,
                          y=st.session_state.df_raw.columns[1],
                          my_chart_color=my_chart_color)

            # radio button for user to select frequency of hist: absolute values or relative
            col1, col2, col3 = st.columns([10, 10, 7])
            with col2:
                # Add radio button for frequency type of histogram
                frequency_type = st.radio(label="*histogram frequency type:*",
                                          options=("Absolute", "Relative"),
                                          index=1 if get_state("HIST", "histogram_freq_type") == "Relative" else 0,
                                          on_change=hist_change_freq,
                                          horizontal=True)

                vertical_spacer(3)

        st.image('./images/patterns_banner.png')

    my_bar.progress(60, text='loading statistical tests')
    with tab4:
        # =============================================================================
        # ################################## TEST #####################################
        # =============================================================================
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([7, 120, 1])
            with col2:
                # apply function flipcard
                create_flipcard_stats(image_path_front_card='./images/statistical_test.png', my_header='Test Results')

        ###################################################################
        # LJUNG-BOX STATISTICAL TEST FOR WHITE NOISE e.g. random residuals
        ###################################################################
        # Perform the Ljung-Box test on the residuals
        with st.expander('White Noise', expanded=False):

            my_text_header('White Noise')

            my_text_paragraph('Ljung-Box')

            col1, col2, col3 = st.columns([18, 44, 10])
            with col2:
                vertical_spacer(2)

                res, result_ljungbox = ljung_box_test(df=st.session_state.df_raw,
                                                      variable_loc=1,
                                                      lag=lag1_ljung_box,
                                                      model_type="AutoReg")

            ljung_box_plots(df=st.session_state['df_raw'],
                            variable_loc=1,
                            lag=lag1_ljung_box,
                            res=res,
                            result_ljungbox=result_ljungbox,
                            my_chart_color=my_chart_color)

        ###################################################################
        # AUGMENTED DICKEY-FULLER TEST
        ###################################################################
        # Show Augmented Dickey-Fuller Statistical Test Result with hypotheses
        with st.expander('Stationarity', expanded=False):

            my_text_header('Stationarity')
            my_text_paragraph('Augmented Dickey Fuller')

            # Augmented Dickey-Fuller (ADF) test results
            adf_result = adf_test(st.session_state['df_raw'], 1)

            col1, col2, col3 = st.columns([18, 40, 10])
            col2.write(adf_result)

            vertical_spacer(2)


        ###################################################################
        # SHAPIRO TEST
        ###################################################################



        with st.expander('Normality', expanded=False):
            my_text_header('Normality')
            my_text_paragraph('Shapiro')
            shapiro_result = shapiro_test(st.session_state['df_raw'], 1, alpha=0.05)
            col1, col2, col3 = st.columns([18, 40, 10])
            col2.write(shapiro_result)
            vertical_spacer(2)

    my_bar.progress(80, text='loading autocorrelation')
    with tab5:
        ###################################################################
        # AUTOCORRELATION PLOTS - Autocorrelation Plots (ACF & PACF) with optional Differencing applied
        ###################################################################
        with st.expander('Autocorrelation', expanded=True):
            my_text_header('Autocorrelation')
            ############################## ACF & PACF ################################
            # Display the original or data differenced Plot based on the user's selection
            # my_text_paragraph(f'{selection}')
            st.markdown(f'<p style="text-align:center; color: #707070"> {selection} </p>', unsafe_allow_html=True)

            # get the differenced dataframe and original figure
            original_fig, df_select_diff = df_differencing(st.session_state.df_raw, selection, my_chart_color)
            st.plotly_chart(original_fig, use_container_width=True)

            # set data equal to the second column e.g. expecting first column 'date'
            data = df_select_diff
            # Plot ACF
            plot_acf(data, nlags=nlags_acf, my_chart_color=my_chart_color)
            # Plot PACF
            plot_pacf(data, my_chart_color=my_chart_color, nlags=nlags_pacf, method=method_pacf)
            # create 3 buttons, about ACF/PACF/Difference for more explanation on the ACF and PACF plots
            acf_pacf_info()

    # =============================================================================
    # PROGRESS BAR - LOAD COMPLETED
    # =============================================================================
    my_bar.progress(100, text='"All systems go!"')
    my_bar.empty()
# logging
print('ForecastGenie Print: Ran Explore')
