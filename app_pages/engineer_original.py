# =============================================================================
#   ______ _   _  _____ _____ _   _ ______ ______ _____
#  |  ____| \ | |/ ____|_   _| \ | |  ____|  ____|  __ \
#  | |__  |  \| | |  __  | | |  \| | |__  | |__  | |__) |
#  |  __| | . ` | | |_ | | | | . ` |  __| |  __| |  _  /
#  | |____| |\  | |__| |_| |_| |\  | |____| |____| | \ \
#  |______|_| \_|\_____|_____|_| \_|______|______|_|  \_\
#
# =============================================================================
# FEATURE ENGINEERING
if menu_item == 'Engineer' and sidebar_menu_item == 'HOME':
    # =============================================================================
    # Define Progress bar
    # =============================================================================
    progress_text = "Loading tabs... Please wait!"
    my_bar = st.progress(0, text=progress_text)

    tab1_engineer, tab2_engineer = st.tabs(['engineer', 'result'])

    with st.sidebar:
        my_title(f"""{icons["engineer_icon"]}""", "#3b3b3b", gradient_colors="#1A2980, #FF6F61, #FEBD2E")

    with st.sidebar.form('feature engineering sidebar'):

        my_text_paragraph('Features')

        vertical_spacer(1)

        # show checkboxes in middle of sidebar to select all features or none
        col1, col2, col3 = st.columns([0.1, 8, 3])
        with col3:
            # create checkbox for all seasonal days e.g. dummy variables for day/month/year
            calendar_dummies_checkbox = st.checkbox(label=' ',
                                                    label_visibility='visible',
                                                    key=key1_engineer,
                                                    help='Include independent features, namely create dummy variables for each `day` of the week, `month` and `year` whereby the leave-1-out principle is applied to not have `perfect multi-collinearity` i.e. the sum of the dummy variables for each observation will otherwise always be equal to one.')
            # create checkbox for country holidays
            calendar_holidays_checkbox = st.checkbox(label=' ',
                                                     label_visibility='visible',
                                                     key=key2_engineer,
                                                     help='Include **`official holidays`** of a specified country  \
                                                          \n**(default = USA)**')

            # create checkbox for all special calendar days
            special_calendar_days_checkbox = st.checkbox(label=' ',
                                                         label_visibility='visible',
                                                         key=key3_engineer,
                                                         help='Include independent features including: **`pay-days`** and significant **`sales`** dates.')

            # create checkbox for Discrete Wavelet Transform features which automatically is checked
            dwt_features_checkbox = st.checkbox(label=' ',
                                                label_visibility='visible',
                                                key=key4_engineer,
                                                help='In feature engineering, wavelet transform can be used to extract useful information from a time series by decomposing it into different frequency bands. This is done by applying a mathematical function called the wavelet function to the time series data. The resulting wavelet coefficients can then be used as features in machine learning models.')
        with col2:
            # show checkbox message if True/False for user options for Feature Selection
            st.write("*🌓 All Seasonal Periods*" if calendar_dummies_checkbox else "*🌓 No Seasonal Periods*")
            st.write("*⛱️ All Holiday Periods*" if calendar_holidays_checkbox else "*⛱️ No Holiday Periods*")
            st.write(
                "*🎁 All Special Calendar Days*" if special_calendar_days_checkbox else "*🎁 No Special Calendar Days*")
            st.write("*🌊 All Wavelet Features*" if dwt_features_checkbox else "*🌊 No Wavelet Features*")

        # provide option for user in streamlit to adjust/set wavelet parameters
        with st.expander('🔽 Wavelet settings'):
            wavelet_family_selectbox = st.selectbox(label='*Select Wavelet Family*',
                                                    options=['db4', 'sym4', 'coif4'],
                                                    label_visibility='visible',
                                                    key=key5_engineer,
                                                    help=' A wavelet family is a set of wavelet functions that have different properties and characteristics.  \
                                          \n**`db4`** wavelet is commonly used for signals with *smooth variations* and *short-duration* pulses  \
                                          \n**`sym4`** wavelet is suited for signals with *sharp transitions* and *longer-duration* pulses.  \
                                          \n**`coif4`** wavelet, on the other hand, is often used for signals with *non-linear trends* and *abrupt* changes.  \
                                          \nIn general, the **`db4`** wavelet family is a good starting point, as it is a popular choice for a wide range of applications and has good overall performance.')

            # set standard level of decomposition to 3
            wavelet_level_decomposition_selectbox = st.selectbox('*Select Level of Decomposition*',
                                                                 [1, 2, 3, 4, 5],
                                                                 label_visibility='visible',
                                                                 key=key6_engineer,
                                                                 help='The level of decomposition refers to the number of times the signal is decomposed recursively into its approximation coefficients and detail coefficients.  \
                                                             \nIn wavelet decomposition, the signal is first decomposed into two components: a approximation component and a detail component.\
                                                             The approximation component represents the coarsest level of detail in the signal, while the detail component represents the finer details.  \
                                                             \nAt each subsequent level of decomposition, the approximation component from the previous level is decomposed again into its own approximation and detail components.\
                                                             This process is repeated until the desired level of decomposition is reached.  \
                                                             \nEach level of decomposition captures different frequency bands and details in the signal, with higher levels of decomposition capturing finer and more subtle details.  \
                                                             However, higher levels of decomposition also require more computation and may introduce more noise or artifacts in the resulting representation of the signal.  \
                                                             \nThe choice of the level of decomposition depends on the specific application and the desired balance between accuracy and computational efficiency.')

            # add slider or text input to choose window size
            wavelet_window_size_slider = st.slider(label='*Select Window Size (in days)*',
                                                   label_visibility='visible',
                                                   min_value=1,
                                                   max_value=30,
                                                   step=1,
                                                   key=key7_engineer)

        col1, col2, col3 = st.columns([4, 4, 4])
        with col2:
            # add submit button to form, when user presses it it updates the selection criteria
            engineering_form_submit_btn = st.form_submit_button('Submit', on_click=form_update, args=("ENGINEER_PAGE",))
            # if user presses the button on ENGINEER PAGE in sidebar
            if engineering_form_submit_btn:
                # update session state to True -> which is used to determine if default feature selection is chosen or user selection in SELECT PAGE
                set_state("ENGINEER_PAGE_FEATURES_BTN", ('engineering_form_submit_btn', True))

    with tab1_engineer:
        with st.expander("", expanded=True):

            show_lottie_animation(url="./images/aJ7Ra5vpQB.json", key="robot_engineering", width=350, height=350,
                                  speed=1, col_sizes=[1, 3, 1])

            ##############################
            # Add Day/Month/Year Features
            # create checkboxes for user to checkmark if to include features
            ##############################
            my_text_header('Dummy Variables')
            my_text_paragraph('🌓 Pick your time-based features to include: ')
            vertical_spacer(1)

            # select all or none of the individual dummy variable checkboxes based on sidebar checkbox
            if calendar_dummies_checkbox == False:
                # update the individual variables checkboxes if not true
                set_state("ENGINEER_PAGE_VARS", ('year_dummies_checkbox', False))
                set_state("ENGINEER_PAGE_VARS", ('month_dummies_checkbox', False))
                set_state("ENGINEER_PAGE_VARS", ('day_dummies_checkbox', False))
            else:
                # update the individual variables checkboxes if calendar_dummies_checkbox is True
                set_state("ENGINEER_PAGE_VARS", ('year_dummies_checkbox', True))
                set_state("ENGINEER_PAGE_VARS", ('month_dummies_checkbox', True))
                set_state("ENGINEER_PAGE_VARS", ('day_dummies_checkbox', True))

            if special_calendar_days_checkbox == False:
                set_state("ENGINEER_PAGE_VARS", ('jan_sales', False))
                set_state("ENGINEER_PAGE_VARS", ('val_day_lod', False))
                set_state("ENGINEER_PAGE_VARS", ('val_day', False))
                set_state("ENGINEER_PAGE_VARS", ('mother_day_lod', False))
                set_state("ENGINEER_PAGE_VARS", ('mother_day', False))
                set_state("ENGINEER_PAGE_VARS", ('father_day_lod', False))
                set_state("ENGINEER_PAGE_VARS", ('pay_days', False))
                set_state("ENGINEER_PAGE_VARS", ('father_day', False))
                set_state("ENGINEER_PAGE_VARS", ('black_friday_lod', False))
                set_state("ENGINEER_PAGE_VARS", ('black_friday', False))
                set_state("ENGINEER_PAGE_VARS", ('cyber_monday', False))
                set_state("ENGINEER_PAGE_VARS", ('christmas_day', False))
                set_state("ENGINEER_PAGE_VARS", ('boxing_day', False))
            else:
                # update the individual variables checkboxes if special_calendar_days_checkbox is True
                set_state("ENGINEER_PAGE_VARS", ('jan_sales', True))
                set_state("ENGINEER_PAGE_VARS", ('val_day_lod', True))
                set_state("ENGINEER_PAGE_VARS", ('val_day', True))
                set_state("ENGINEER_PAGE_VARS", ('mother_day_lod', True))
                set_state("ENGINEER_PAGE_VARS", ('mother_day', True))
                set_state("ENGINEER_PAGE_VARS", ('father_day_lod', True))
                set_state("ENGINEER_PAGE_VARS", ('pay_days', True))
                set_state("ENGINEER_PAGE_VARS", ('father_day', True))
                set_state("ENGINEER_PAGE_VARS", ('black_friday_lod', True))
                set_state("ENGINEER_PAGE_VARS", ('black_friday', True))
                set_state("ENGINEER_PAGE_VARS", ('cyber_monday', True))
                set_state("ENGINEER_PAGE_VARS", ('christmas_day', True))
                set_state("ENGINEER_PAGE_VARS", ('boxing_day', True))

            # create columns for aligning in middle the checkboxes
            col0, col1, col2, col3, col4 = st.columns([2, 2, 2, 2, 1])
            with col1:
                year_dummies_checkbox = st.checkbox(label='Year',
                                                    value=get_state("ENGINEER_PAGE_VARS", "year_dummies_checkbox"))
                set_state("ENGINEER_PAGE_VARS", ("year_dummies_checkbox", year_dummies_checkbox))
            with col2:
                month_dummies_checkbox = st.checkbox('Month',
                                                     value=get_state("ENGINEER_PAGE_VARS", "month_dummies_checkbox"))
                set_state("ENGINEER_PAGE_VARS", ("year_dummies_checkbox", month_dummies_checkbox))
            with col3:
                day_dummies_checkbox = st.checkbox('Day',
                                                   value=get_state("ENGINEER_PAGE_VARS", "day_dummies_checkbox"))
                set_state("ENGINEER_PAGE_VARS", ("year_dummies_checkbox", day_dummies_checkbox))

            ###############################################
            # create selectbox for country holidays
            ###############################################
            vertical_spacer(1)
            my_text_header('Holidays')
            my_text_paragraph('⛱️ Select country-specific holidays to include:')

            if calendar_holidays_checkbox:
                # TEST
                st.write(st.session_state['df_cleaned_outliers_with_index'])
                # apply function to create country specific holidays in columns is_holiday (boolean 1 if holiday otherwise 0) and holiday_desc for holiday_name
                df = create_calendar_holidays(df=st.session_state['df_cleaned_outliers_with_index'])

                # update the session_state
                st.session_state['df_cleaned_outliers_with_index'] = df
            else:
                my_text_paragraph('<i> no country-specific holiday selected </i>')

            ###############################################
            # create checkboxes for special days on page
            ###############################################
            my_text_header('Special Calendar Days')
            my_text_paragraph("🎁 Pick your special days to include: ")
            vertical_spacer(1)

            col0, col1, col2, col3 = st.columns([6, 12, 12, 1])
            with col1:
                jan_sales = st.checkbox(label='January Sale',
                                        value=get_state("ENGINEER_PAGE_VARS", "jan_sales"))

                set_state("ENGINEER_PAGE_VARS", ("jan_sales", jan_sales))

                val_day_lod = st.checkbox(label="Valentine's Day [last order date]",
                                          value=get_state("ENGINEER_PAGE_VARS", "val_day_lod"))

                set_state("ENGINEER_PAGE_VARS", ("val_day_lod", val_day_lod))

                val_day = st.checkbox(label="Valentine's Day",
                                      value=get_state("ENGINEER_PAGE_VARS", "val_day"))

                set_state("ENGINEER_PAGE_VARS", ("val_day", val_day))

                mother_day_lod = st.checkbox(label="Mother's Day [last order date]",
                                             value=get_state("ENGINEER_PAGE_VARS", "mother_day_lod"))

                set_state("ENGINEER_PAGE_VARS", ("mother_day_lod", mother_day_lod))

                mother_day = st.checkbox(label="Mother's Day",
                                         value=get_state("ENGINEER_PAGE_VARS", "mother_day"))

                set_state("ENGINEER_PAGE_VARS", ("mother_day", mother_day))

                father_day_lod = st.checkbox(label="Father's Day [last order date]",
                                             value=get_state("ENGINEER_PAGE_VARS", "father_day_lod"))

                set_state("ENGINEER_PAGE_VARS", ("father_day_lod", father_day_lod))

                pay_days = st.checkbox(label='Monthly Pay Days (4th Friday of month)',
                                       value=get_state("ENGINEER_PAGE_VARS", "pay_days"))

                set_state("ENGINEER_PAGE_VARS", ("pay_days", pay_days))

            with col2:
                father_day = st.checkbox(label="Father's Day",
                                         value=get_state("ENGINEER_PAGE_VARS", "father_day"))

                set_state("ENGINEER_PAGE_VARS", ("father_day", father_day))

                black_friday_lod = st.checkbox(label='Black Friday [sale starts]',
                                               value=get_state("ENGINEER_PAGE_VARS", "black_friday_lod"))

                set_state("ENGINEER_PAGE_VARS", ("black_friday_lod", black_friday_lod))

                black_friday = st.checkbox(label='Black Friday',
                                           value=get_state("ENGINEER_PAGE_VARS", "black_friday"))

                set_state("ENGINEER_PAGE_VARS", ("black_friday", black_friday))

                cyber_monday = st.checkbox('Cyber Monday',
                                           value=get_state("ENGINEER_PAGE_VARS", "cyber_monday"))

                set_state("ENGINEER_PAGE_VARS", ("cyber_monday", cyber_monday))

                christmas_day = st.checkbox(label='Christmas Day [last order date]',
                                            value=get_state("ENGINEER_PAGE_VARS", "christmas_day"))

                set_state("ENGINEER_PAGE_VARS", ("christmas_day", christmas_day))

                boxing_day = st.checkbox(label='Boxing Day sale',
                                         value=get_state("ENGINEER_PAGE_VARS", "boxing_day"))
                set_state("ENGINEER_PAGE_VARS", ("boxing_day", boxing_day))

            vertical_spacer(3)

        # user checkmarked the box for all seasonal periods
        if special_calendar_days_checkbox:

            # call very extensive function to create all days selected by users as features
            df = create_calendar_special_days(st.session_state['df_cleaned_outliers_with_index'])

            # update the session_state
            st.session_state['df_cleaned_outliers_with_index'] = df

        else:

            df = st.session_state['df_cleaned_outliers_with_index']

        # user checkmarked the box for all seasonal periods
        if calendar_dummies_checkbox:
            # apply function to add year/month and day dummy variables
            df = create_date_features(df,
                                      year_dummies=year_dummies_checkbox,
                                      month_dummies=month_dummies_checkbox,
                                      day_dummies=day_dummies_checkbox)
            # update the session_state
            st.session_state['df_cleaned_outliers_with_index'] = df
        else:
            pass

        # if user checkmarked checkbox: Discrete Wavelet Transform
        if dwt_features_checkbox:
            with st.expander('🌊 Wavelet Features', expanded=True):
                my_text_header('Discrete Wavelet Transform')
                my_text_paragraph('Feature Extraction')

                ########## CREATE WAVELET FEATURES ##################
                # define wavelet and level of decomposition

                # TEST REPLACE WITH SESSION STATES THE USER CHOICES
                # =============================================================================
                #             wavelet = wavelet_family_selectbox
                #             level = wavelet_level_decomposition_selectbox
                #             window_size = wavelet_window_size_slider
                # =============================================================================
                wavelet = get_state("ENGINEER_PAGE", "wavelet_family_selectbox")
                level = get_state("ENGINEER_PAGE", "wavelet_level_decomposition_selectbox")
                window_size = get_state("ENGINEER_PAGE", "wavelet_window_size_slider")
                st.write('wavelet', wavelet, 'wavelet_level_decomposition_selectbox', level, 'window_size', window_size)

                # create empty list to store feature vectors
                feature_vectors = []

                # loop over each window in the data
                for i in range(window_size, len(df)):
                    # extract data for current window
                    data_in_window = df.iloc[i - window_size:i, 1].values

                    # perform DWT on data
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
                               [f'detail{i + 1}_mean' for i in range(level)] + \
                               [f'detail{i + 1}_std' for i in range(level)] + \
                               [f'detail{i + 1}_max' for i in range(level)] + \
                               [f'detail{i + 1}_min' for i in range(level)]

                # create a dataframe with the created features with discrete wavelet transform on target variable with timewindow set by user
                features_df_wavelet = pd.DataFrame(feature_vectors, columns=feature_cols,
                                                   index=df.iloc[:, 0].index[window_size:])

                # merge features dataframe with original data
                df = pd.merge(df, features_df_wavelet, left_index=True, right_index=True)
                #################################################################

                # PLOT WAVELET FEATURES
                #######################
                # create a dataframe again with the index set as the first column
                # assumption used: the 'date' column is the first column of the dataframe
                features_df_plot = pd.DataFrame(feature_vectors, columns=feature_cols, index=df.iloc[:, 0])

                fig = px.line(features_df_plot,
                              x=features_df_plot.index,
                              y=['approx_mean'] + [f'detail{i + 1}_mean' for i in range(level)],
                              title='',
                              labels={'value': 'Coefficient Mean', 'variable': 'Subband'})

                fig.update_layout(xaxis_title='Date')

                st.plotly_chart(fig, use_container_width=True)

                # SHOW WAVELETE FEATURES
                ########################
                # Show Dataframe with features
                my_text_paragraph('Wavelet Features Dataframe')
                st.dataframe(features_df_wavelet, use_container_width=True)

                # update the session state
                st.session_state['df_cleaned_outliers_with_index'] = df
        else:
            pass

        # =============================================================================
        # Add the date column but only as numeric feature
        # =============================================================================
        df['date_numeric'] = (df['date'] - df['date'].min()).dt.days

    my_bar.progress(50, text='Combining Features into a presentable format...Please Wait!')
    #################################################################
    # ALL FEATURES COMBINED INTO A DATAFRAME
    #################################################################
    with tab2_engineer:
        with st.expander('', expanded=True):
            my_text_header('Engineered Features')

            # Retrieve number of features to show on page
            # -2 -> because of datetime index and target variable
            num_features_df = len(df.columns) - 2
            my_text_paragraph(f'{num_features_df}')

            show_lottie_animation(url="./images/features_round.json", key="features_round", width=400, height=400)

            # Show dataframe in streamlit
            st.dataframe(copy_df_date_index(my_df=df, datetime_to_date=True, date_to_index=True),
                         use_container_width=True)

            # add download button
            download_csv_button(df, my_file="dataframe_incl_features.csv",
                                help_message="Download your dataset incl. features to .CSV")

    my_bar.progress(100, text='All systems go!')
    my_bar.empty()

# Else e.g. if user is not within menu_item == 'Engineer' and sidebar_menu_item is not 'Home':
else:
    # set dataframe equal to the cleaned dataframe
    df = st.session_state['df_cleaned_outliers_with_index']

    # Check if checkboxes are True or False for feature engineering options
    calendar_holidays_checkbox = get_state("ENGINEER_PAGE", "calendar_holidays_checkbox")
    special_calendar_days_checkbox = get_state("ENGINEER_PAGE", "special_calendar_days_checkbox")
    calendar_dummies_checkbox = get_state("ENGINEER_PAGE", "calendar_dummies_checkbox")
    dwt_features_checkbox = get_state("ENGINEER_PAGE", "dwt_features_checkbox")

    # if feature engineering option is checked for holidays, add features:
    if calendar_holidays_checkbox:
        df = create_calendar_holidays(df=st.session_state['df_cleaned_outliers_with_index'], slider=False)

    # if feature engineering option is checked for special calendar days (e.g. black friday etc.), add features
    if special_calendar_days_checkbox:
        df = create_calendar_special_days(df)

    # if featue engineering option is checked for the year/month/day dummy variables, add features
    if calendar_dummies_checkbox:
        df = create_date_features(df,
                                  year_dummies=get_state("ENGINEER_PAGE_VARS", "year_dummies_checkbox"),
                                  month_dummies=get_state("ENGINEER_PAGE_VARS", "month_dummies_checkbox"),
                                  day_dummies=get_state("ENGINEER_PAGE_VARS", "day_dummies_checkbox"))

    # =============================================================================
    #     # TEST OUTSIDE FUNCTION TO RUN WAVELET FUNCTION
    #     # df = <insert wavelet function code>
    #     if dwt_features_checkbox:
    #         df =
    # =============================================================================

    # Add the date column but only as numeric feature
    df['date_numeric'] = (df['date'] - df['date'].min()).dt.days
    # TIMESTAMP NUMERIC Unix gives same result
    # df['date_numeric'] = (df['date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    # =============================================================================
    # FIX DATA TYPES FOR ENGINEERED FEATURES
    # =============================================================================
    # columns_to_convert = {'holiday': 'uint8', 'calendar_event': 'uint8', 'pay_day': 'uint8', 'year': 'int32', 'is_holiday': 'uint8'}
    columns_to_convert = {'holiday': 'uint8', 'calendar_event': 'uint8', 'pay_day': 'uint8', 'year': 'str',
                          'is_holiday': 'uint8'}
    for column, data_type in columns_to_convert.items():
        if column in df:
            df[column] = df[column].astype(data_type)

    else:
        # cleaned df is unchanged e.g. no features added with feature engineering page
        # st.write('no features engineered') # TEST
        pass

# =============================================================================
# # update dataframe's session state with transformations:
# # - Added calendar days
# # - Date dummy variables
# # - Wavelet features
# # - Datatypes updated
# =============================================================================
st.session_state['df'] = df

# Assumption: date column and y column are at index 0 and index 1 so start from column 3 e.g. index 2 to count potential numerical_features
# e.g. changed float64 to float to include other floats such as float32 and float16 data types
numerical_features = list(st.session_state['df'].iloc[:, 2:].select_dtypes(include=['float', 'int']).columns)

# create copy of dataframe not altering original
local_df = st.session_state['df'].copy(deep=True)
