
# =============================================================================
#   _____  _____  ______ _____        _____  ______
#  |  __ \|  __ \|  ____|  __ \ /\   |  __ \|  ____|
#  | |__) | |__) | |__  | |__) /  \  | |__) | |__
#  |  ___/|  _  /|  __| |  ___/ /\ \ |  _  /|  __|
#  | |    | | \ \| |____| |  / ____ \| | \ \| |____
#  |_|    |_|  \_\______|_| /_/    \_\_|  \_\______|
#
# =============================================================================
# PREPARE PAGE
# GOAL: PREPARE DATASET (REMOVE OBJECT DTYPE FEATURES, TRAIN/TEST SPLIT, NORMALIZE, STANDARDIZE)

if menu_item == 'Prepare' and sidebar_menu_item == 'HOME':

    # define user tabs for the prepare page
    tab1_prepare, tab2_prepare, tab3_prepare, tab4_prepare = st.tabs(
        ['preprocess', 'train/test split', 'normalization', 'standardization'])

    with st.sidebar:

        my_title(f'{icons["prepare_icon"]}', "#3b3b3b", gradient_colors="#1A2980, #FF9F00, #FEBD2E")

    ############################################
    # 5.0.1 PREPROCESS (remove redundant features)
    # Assumption: if object datatype only descriptive of other columns -> not needed in train/test
    ############################################
    with tab1_prepare:
        obj_cols = local_df.select_dtypes(include='object').columns.tolist()

        # if not an empty list e.g. only show if there are variables removed with dtype = object
        if obj_cols:

            # show user which descriptive variables are removed, that just had the purpose to inform user what dummy was from e.g. holiday days such as Martin Luther King Day
            with st.expander('', expanded=True):
                my_text_header('Preprocess')
                my_text_paragraph('*removing redundant features (dtype = object)', my_font_size='12px')

                local_df = remove_object_columns(local_df, message_columns_removed=True)

                # have button available for user and if clicked it expands with the dataframe
                col1, col2, col3 = st.columns([130, 60, 120])
                with col2:
                    # initiate placeholder
                    placeholder = st.empty()

                    # create button (enabled to click e.g. disabled=false with unique key)
                    btn = placeholder.button('Show Data', disabled=False, key="preprocess_df_show_btn")

                # if button is clicked run below code
                if btn == True:
                    # display button with text "click me again", with unique key
                    placeholder.button('Hide Data', disabled=False, key="preprocess_df_hide_btn")

                    # show dataframe to user in streamlit
                    st.dataframe(local_df, use_container_width=True)

                vertical_spacer(1)
        st.image('./images/train_test_split_banner.png')

        ############################################
        # 5.0.2 PREPROCESS (set date feature as index column)
        ############################################
        # Check if 'date' column exists
        if 'date' in local_df.columns:
            # set the date as the index of the pandas dataframe
            local_df.index = pd.to_datetime(local_df['date'])
            local_df.drop(columns='date', inplace=True)

        # update df in session state without descriptive columns
        st.session_state['df'] = local_df

    ######################
    # 5.1 TRAIN/TEST SPLIT
    ######################
    with tab2_prepare:
        with st.expander("", expanded=True):
            my_text_header('Train/Test Split')

            # create sliders for user insample test-size (/train-size automatically as well)
            my_insample_forecast_steps, my_insample_forecast_perc = train_test_split_slider(df=st.session_state['df'])

            # update the session_state with train/test split chosen by user from sidebar slider
            # note: insample_forecast_steps is used in train/test split as variable
            st.session_state['insample_forecast_steps'] = my_insample_forecast_steps
            st.session_state['insample_forecast_perc'] = my_insample_forecast_perc

            # SHOW USER MESSAGE OF TRAIN/TEST SPLIT
            #######################################
            # SET VARIABLES
            length_df = len(st.session_state['df'])

            # format as new variables insample_forecast steps in days/as percentage e.g. the test set to predict for
            perc_test_set = "{:.2f}%".format((st.session_state['insample_forecast_steps'] / length_df) * 100)
            perc_train_set = "{:.2f}%".format(
                ((length_df - st.session_state['insample_forecast_steps']) / length_df) * 100)
            my_text_paragraph(f"{perc_train_set} / {perc_test_set}", my_font_size='16px')

            # PLOT TRAIN/TEST SPLIT
            ############################
            # Set train/test split index
            split_index = min(length_df - st.session_state['insample_forecast_steps'], length_df - 1)

            # Create a figure with a scatter plot of the train/test split
            train_test_fig = plot_train_test_split(st.session_state['df'], split_index)

            # show the plot inside streamlit app on page
            st.plotly_chart(train_test_fig, use_container_width=True)
            # show user the train test split currently set by user or default e.g. 80:20 train/test split
            # st.warning(f"ℹ️ train/test split currently equals :green[**{perc_train_set}**] and :green[**{perc_test_set}**] ")

            my_text_paragraph('NOTE: a commonly used ratio is <b> 80:20 </b> split between the train- and test set.',
                              my_font_size='12px', my_font_family='Arial')

            vertical_spacer(2)

    ##############################
    # 5.2 Normalization
    ##############################
    with tab3_prepare:
        with st.sidebar:
            with st.form('normalization'):

                my_text_paragraph('Normalization')

                # Assumption: date column and y column are at index 0 and index 1 so start from column 3 e.g. index 2 to count potential numerical_features
                # e.g. changed float64 to float to include other floats such as float32 and float16 data types
                numerical_features = list(
                    st.session_state['df'].iloc[:, 2:].select_dtypes(include=['float', 'int']).columns)

                # Add selectbox for normalization choices
                if numerical_features:
                    # define dictionary of normalization choices as keys and values are descriptions of each choice
                    normalization_choices = {
                        "None": "Do not normalize the data",
                        "MinMaxScaler": "Scale features to a given range (default range is [0, 1]).",
                        "RobustScaler": "Scale features using statistics that are robust to outliers.",
                        "MaxAbsScaler": "Scale each feature by its maximum absolute value.",
                        "PowerTransformer": "Apply a power transformation to make the data more Gaussian-like.",
                        "QuantileTransformer": "Transform features to have a uniform or Gaussian distribution."
                    }

                    # create a dropdown menu for user in sidebar to choose from a list of normalization methods
                    normalization_choice = st.selectbox(label="*Select normalization method:*",
                                                        options=list(normalization_choices.keys()),
                                                        format_func=lambda x: f"{x} - {normalization_choices[x]}",
                                                        key=key1_prepare_normalization,
                                                        help='**`Normalization`** is a data pre-processing technique to transform the numerical data in a dataset to a standard scale or range.\
                                                                This process involves transforming the features of the dataset so that they have a common scale, which makes it easier for data scientists to analyze, compare, and draw meaningful insights from the data.')
                    # save user normalization choice in memory
                    st.session_state['normalization_choice'] = normalization_choice
                else:
                    # if no numerical features show user a message to inform
                    st.warning("No numerical features to normalize, you can try adding features!")
                    # set normalization_choice to None
                    normalization_choice = "None"

                # create form button centered on sidebar to submit user choice for normalization method
                col1, col2, col3 = st.columns([4, 4, 4])
                with col2:
                    normalization_btn = st.form_submit_button("Submit", type="secondary", on_click=form_update,
                                                              args=("PREPARE",))

            # apply function for normalizing the dataframe if user choice
            # IF user selected a normalization_choice other then "None" the X_train and X_test will be scaled
            X, y, X_train, X_test, y_train, y_test, scaler = perform_train_test_split(df=st.session_state['df'],
                                                                                      my_insample_forecast_steps=
                                                                                      st.session_state[
                                                                                          'insample_forecast_steps'],
                                                                                      scaler_choice=st.session_state[
                                                                                          'normalization_choice'],
                                                                                      numerical_features=numerical_features)

        # if user did not select normalization (yet) then show user message to select normalization method in sidebar
        if normalization_choice == "None":
            with st.expander('Normalization ', expanded=True):
                my_text_header('Normalization')

                vertical_spacer(4)

                # show_lottie_animation(url="./images/monster-2.json", key="rocket_night_day", width=200, height=200, col_sizes=[6,6,6])
                st.image('./images/train_test_split.png')
                my_text_paragraph(f'Method: {normalization_choice}')

                vertical_spacer(12)

                st.info(
                    '👈 Please choose in the sidebar your normalization method for numerical columns. Note: columns with booleans will be excluded.')

        # else show user the dataframe with the features that were normalized
        else:
            with st.expander('Normalization', expanded=True):

                my_text_header('Normalized Features')

                my_text_paragraph(f'Method: {normalization_choice}')

                # need original (unnormalized) X_train as well for figure in order to show before/after normalization
                X_unscaled_train = df.iloc[:, 1:].iloc[:-st.session_state['insample_forecast_steps'], :]

                # with custom function create the normalization plot with numerical features i.e. before/after scaling
                plot_scaling_before_after(X_unscaled_train, X_train, numerical_features)

                st.success(
                    f'🎉 Good job! **{len(numerical_features)}** numerical feature(s) are normalized with **{normalization_choice}**!')

                st.dataframe(X[numerical_features].assign(date=X.index.date).reset_index(drop=True).set_index('date'),
                             use_container_width=True)  # TEST

                # create download button for user, to download the standardized features dataframe with dates as index i.e. first column
                download_csv_button(X[numerical_features],
                                    my_file='standardized_features.csv',
                                    help_message='Download standardized features to .CSV',
                                    set_index=True,
                                    my_key='normalization_download_btn')

    ##############################
    # 5.3 Standardization
    ##############################
    with tab4_prepare:
        with st.sidebar:
            with st.form('standardization'):
                my_text_paragraph('Standardization')
                if numerical_features:
                    standardization_choices = {
                        "None": "Do not standardize the data",
                        "StandardScaler": "Standardize features by removing the mean and scaling to unit variance.",
                    }
                    standardization_choice = st.selectbox(label="*Select standardization method:*",
                                                          options=list(standardization_choices.keys()),
                                                          format_func=lambda x: f"{x} - {standardization_choices[x]}",
                                                          key=key2_prepare_standardization,
                                                          help='**`Standardization`** is a preprocessing technique used to transform the numerical data to have zero mean and unit variance.\
                                                                  This is achieved by subtracting the mean from each value and then dividing by the standard deviation.\
                                                                  The resulting data will have a mean of zero and a standard deviation of one.\
                                                                  The distribution of the data is changed by centering and scaling the values, which can make the data more interpretable and easier to compare across different features')
                else:
                    # if no numerical features show user a message to inform
                    st.warning("No numerical features to standardize, you can try adding features!")

                    # set normalization_choice to None
                    standardization_choice = "None"

                # create form button centered on sidebar to submit user choice for standardization method
                col1, col2, col3 = st.columns([4, 4, 4])
                with col2:
                    standardization_btn = st.form_submit_button("Submit", type="secondary", on_click=form_update,
                                                                args=("PREPARE",))

            # apply function for normalizing the dataframe if user choice
            # IF user selected a normalization_choice other then "None" the X_train and X_test will be scaled
            X, y, X_train, X_test, y_train, y_test = perform_train_test_split_standardization(X, y, X_train, X_test,
                                                                                              y_train, y_test,
                                                                                              st.session_state[
                                                                                                  'insample_forecast_steps'],
                                                                                              scaler_choice=standardization_choice,
                                                                                              numerical_features=numerical_features)

        # if user did not select standardization (yet) then show user message to select normalization method in sidebar
        if standardization_choice == "None":
            # on page create expander
            with st.expander('Standardization ', expanded=True):

                my_text_header('Standardization')
                my_text_paragraph(f'Method: {standardization_choice}')

                # show_lottie_animation(url="./images/2833-green-monster.json", key="green-monster", width=200, height=200, col_sizes=[6,6,6], speed=0.8)
                st.image('./images/standardization.png')
                st.info(
                    '👈 Please choose in the sidebar your Standardization method for numerical columns. Note: columns with booleans will be excluded.')

        # ELSE show user the dataframe with the features that were normalized
        else:
            with st.expander('Standardization', expanded=True):

                my_text_header('Standardization')
                my_text_paragraph(f'Method: {standardization_choice}')

                # need original (unnormalized) X_train as well for figure in order to show before/after Standardization
                # TEST or do i need st.session_state['df'] instead of df? -> replaced df with st.session_state['df']
                X_unscaled_train = st.session_state['df'].iloc[:, 1:].iloc[
                                   :-st.session_state['insample_forecast_steps'], :]

                # with custom function create the Standardization plot with numerical features i.e. before/after scaling
                plot_scaling_before_after(X_unscaled_train, X_train, numerical_features)

                st.success(
                    f'⚖️ Great, you balanced the scales! **{len(numerical_features)}** numerical feature(s) standardized with **{standardization_choice}**')

                st.dataframe(X[numerical_features], use_container_width=True)

                # create download button for user, to download the standardized features dataframe with dates as index i.e. first column
                download_csv_button(X[numerical_features],
                                    my_file='standardized_features.csv',
                                    help_message='Download standardized features to .CSV',
                                    set_index=True,
                                    my_key='standardization_download_btn')