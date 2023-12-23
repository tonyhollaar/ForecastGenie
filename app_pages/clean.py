# standard libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Union

# third-party libraries
from fire_state import get_state, set_state, form_update

# local packages
from functions import determine_df_frequency, copy_df_date_index, download_csv_button
from style.text import my_title, my_text_header, my_text_paragraph, my_subheader, vertical_spacer
from style.icons import load_icons

from sklearn.ensemble import IsolationForest
from scipy import stats

# load icons
icons = load_icons()


class CleanPage:
    def __init__(self, state, key1_missing, key2_missing, key3_missing, random_state, key1_outlier, key2_outlier,
                 key3_outlier, key4_outlier, key5_outlier, key6_outlier, key7_outlier):
        """
        Initializes the CleanPage object.

        Parameters:
        - state (dict): The state dictionary containing data and configuration.
        - key1_missing (str): Key for missing data method selection.
        - key2_missing (str): Key for custom fill value input.
        - key3_missing (str): Key for frequency selection.
        - random_state (int): Random state for reproducibility.
        """
        self.my_bar = None
        self.state = state
        self.key1_missing = key1_missing
        self.key2_missing = key2_missing
        self.key3_missing = key3_missing
        self.random_state = random_state
        self.key1_outlier = key1_outlier
        self.key2_outlier = key2_outlier
        self.key3_outlier = key3_outlier
        self.key4_outlier = key4_outlier
        self.key5_outlier = key5_outlier
        self.key6_outlier = key6_outlier
        self.key7_outlier = key7_outlier
        self.fill_method = get_state("CLEAN_PAGE_MISSING", "missing_fill_method")
        self.custom_fill_value = get_state("CLEAN_PAGE_MISSING", "missing_custom_fill_value")

        # infer and return the original data frequency e.g. 'M' and name e.g. 'Monthly'
        self.original_freq, self.original_freq_name = determine_df_frequency(self.state['df_raw'], column_name='date')

        # Initialize df_clean_show as an empty DataFrame or with appropriate initial value
        self.df_clean_show = pd.DataFrame()

        # initialize the variable before using it
        self.outliers = None

        # Initialize df_cleaned_outliers as an empty DataFrame
        self.df_cleaned_outliers = pd.DataFrame()

        self.freq = st.session_state['freq']

    def render(self):
        """
        Renders the CleanPage interface.
        """
        self.my_bar = st.progress(0, text="Dusting off cleaning methods... Please wait!")

        self.render_sidebar()  # Renders the sidebar for user input.

        self.my_bar.progress(10, text='10%')

        # create user tabs
        tab1_clean, tab2_clean = st.tabs(['missing data', 'outliers'])

        with tab1_clean:
            self.handle_missing_data(tab1_clean)

        with tab2_clean:
            self.handle_outliers_tab(tab2_clean)

        self.my_bar.progress(100, text='100%')
        self.my_bar.empty()

    def perform_data_cleaning(self):
        """
        Performs data cleaning on the dataset.
        """
        self.impute_missing_dates()  # Applies function to resample missing dates based on the user-set frequency.
        self.impute_missing_values()  # Applies the selected method to impute missing values.
        self.retrieve_parameters() # Retrieves the parameters from the state dictionary.
        #self.execute_outlier_handling() # Handles outliers in the dataset.
        self.deal_with_outliers()

    def render_sidebar(self):
        """
        Renders the sidebar for user input.
        """
        with st.sidebar:
            self.render_sidebar_title()
            self.render_data_cleaning_form()
            self.render_outlier_form()

    def render_sidebar_title(self):
        my_title(f"""{icons["clean_icon"]}""",
                 "#3b3b3b",
                 gradient_colors="#440154, #2C2A6B, #FDE725")

    def highlight_cols(self, s):
        """
        A function that highlights the cells of a DataFrame based on their values.

        Args:
        s (pd.Series): A Pandas Series object representing the columns of a DataFrame.

        Returns:
        list: A list of CSS styles to be applied to each cell in the input Series object.
        """
        if isinstance(s, pd.Series):
            if s.name == self.outliers_df.columns[0]:
                return ['background-color: lavender'] * len(s)
            elif s.name == self.outliers_df.columns[1]:
                return ['background-color: lightyellow'] * len(s)
            else:
                return [''] * len(s)
        else:
            return [''] * len(s)


    def render_data_cleaning_form(self):
        with st.form('data_cleaning'):
            my_text_paragraph('Handling Missing Data')
            self.fill_method = self.get_fill_method()
            if self.fill_method == 'Custom':
                self.custom_fill_value = self.get_custom_fill_value()
            self.freq = self.get_data_frequency()
            self.data_cleaning_btn = self.get_data_cleaning_button()

    def get_fill_method(self):
        return st.selectbox(label='*Select filling method for missing values:*',
                            options=['Backfill', 'Forwardfill', 'Mean', 'Median', 'Mode', 'Custom'],
                            key=self.key1_missing)

    def get_custom_fill_value(self):
        return st.text_input(label='*Insert custom value to replace missing value(s) with:*',
                             key=self.key2_missing,
                             help='Please enter your **`custom value`** to impute missing values with, you can use a whole number or decimal point number')

    def get_data_frequency(self):
        freq_dict = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
        return st.selectbox('*Select the frequency of the data:*',
                            list(freq_dict.keys()),
                            key=self.key3_missing)

    def get_data_cleaning_button(self):
        col1, col2, col3 = st.columns([4, 4, 4])
        with col2:
            return st.form_submit_button(label="Submit",
                                         type="secondary",
                                         on_click=form_update,
                                         args=("CLEAN_PAGE_MISSING",))

    def render_outlier_form(self):
        self.outlier_form(self.key1_outlier, self.key2_outlier, self.key3_outlier, self.key4_outlier, self.key5_outlier,
                          self.key6_outlier, self.key7_outlier)

    def retrieve_parameters(self):
        self.outlier_detection_method = get_state("CLEAN_PAGE", "outlier_detection_method")
        self.outlier_zscore_threshold = get_state("CLEAN_PAGE", "outlier_zscore_threshold")
        self.outlier_iqr_q1 = get_state("CLEAN_PAGE", "outlier_iqr_q1")
        self.outlier_iqr_q3 = get_state("CLEAN_PAGE", "outlier_iqr_q1")
        self.outlier_replacement_method = get_state("CLEAN_PAGE", "outlier_replacement_method")
        self.outlier_isolationforest_contamination = get_state("CLEAN_PAGE", "outlier_isolationforest_contamination")
        self.outlier_iqr_multiplier = get_state("CLEAN_PAGE", "outlier_iqr_multiplier")

    def execute_outlier_handling(self):
        self.df_cleaned_outliers, self.outliers = self.handle_outliers(data=self.df_clean_show,
                                                                  method=self.outlier_detection_method,
                                                                  outlier_threshold=self.outlier_zscore_threshold,
                                                                  q1=self.outlier_iqr_q1,
                                                                  q3=self.outlier_iqr_q3,
                                                                  outlier_replacement_method=self.outlier_replacement_method,
                                                                  contamination=self.outlier_isolationforest_contamination,
                                                                  random_state=self.random_state,
                                                                  iqr_multiplier=self.outlier_iqr_multiplier)

    def impute_missing_dates(self):
        """
        Applies function to resample missing dates based on the user-set frequency.
        """
        # Apply function to resample missing dates based on user set frequency
        self.df_cleaned_dates = self.resample_missing_dates(df=self.state['df_raw'],
                                                      freq_dict=self.state['freq_dict'],
                                                      original_freq=self.original_freq)

        #self.state['df_cleaned_dates'] = self.df_cleaned_dates

    def impute_missing_values(self):
        """
        Applies the selected method to impute missing values.
        """
        self.df_clean = self.my_fill_method(df=self.df_cleaned_dates,
                                  fill_method=get_state('CLEAN_PAGE_MISSING', 'missing_fill_method'),
                                  custom_fill_value=self.custom_fill_value)

        self.df_clean_show = copy_df_date_index(self.df_clean, datetime_to_date=True, date_to_index=True)
        #self.state['df_clean_show'] = self.df_clean_show

    def deal_with_outliers(self):
        """
        Handles outliers in the dataset.
        """
        self.df_cleaned_outliers, self.outliers = self.handle_outliers(data=self.df_clean_show ,
                                                        method=get_state('CLEAN_PAGE', 'outlier_detection_method'),
                                                        outlier_threshold=get_state('CLEAN_PAGE',
                                                                                    'outlier_zscore_threshold'),
                                                        q1=get_state('CLEAN_PAGE', 'outlier_iqr_q1'),
                                                        q3=get_state('CLEAN_PAGE', 'outlier_iqr_q3'),
                                                        outlier_replacement_method=get_state('CLEAN_PAGE',
                                                                                             'outlier_replacement_method'),
                                                        contamination=get_state('CLEAN_PAGE',
                                                                                'outlier_isolationforest_contamination'),
                                                        random_state=self.random_state,
                                                        iqr_multiplier=get_state('CLEAN_PAGE', 'outlier_iqr_multiplier'))

        # create a copy of df_cleaned_outliers with index
        df_cleaned_outliers_with_index = self.df_cleaned_outliers.copy(deep=True)

        # reset index
        df_cleaned_outliers_with_index.reset_index(inplace=True)

        # change date column to datetime
        df_cleaned_outliers_with_index['date'] = pd.to_datetime(df_cleaned_outliers_with_index['date'])

        # save to session state
        st.session_state['df_cleaned_outliers_with_index'] = df_cleaned_outliers_with_index

    def my_fill_method(self, df, fill_method, custom_fill_value=None):
        """
        Handle missing values in the DataFrame based on the user's specified fill method.

        Args:
            df (pandas.DataFrame): The input DataFrame to be processed.
            fill_method (str): The fill method to be used. Can be one of 'Backfill', 'Forwardfill', 'Mean', 'Median',
                'Mode', or 'Custom'.
            custom_fill_value (optional): The custom value to be used for filling missing values. Only applicable when
                fill_method is set to 'Custom'. Defaults to None.

        Returns:
            pandas.DataFrame: A copy of the input DataFrame with missing values filled according to the specified fill method.
        """
        # create a copy of df
        df = df.copy(deep=True)
        # handle missing values based on user input
        if fill_method == 'Backfill':
            df.iloc[:, 1] = df.iloc[:, 1].bfill()
        elif fill_method == 'Forwardfill':
            df.iloc[:, 1] = df.iloc[:, 1].ffill()
        elif fill_method == 'Mean':
            # rounding occurs to nearest decimal for filling in the average value of y
            df.iloc[:, 1] = df.iloc[:, 1].fillna(df.iloc[:, 1].mean().round(0))
        elif fill_method == 'Median':
            df.iloc[:, 1] = df.iloc[:, 1].fillna(df.iloc[:, 1].median())
        elif fill_method == 'Mode':
            # if True, only apply to numeric columns (numeric_only=False)
            # Don’t consider counts of NaN/NaT (dropna=True)
            df.iloc[:, 1] = df.iloc[:, 1].fillna(df.iloc[:, 1].mode(dropna=True)[0])
        elif fill_method == 'Custom':
            df.iloc[:, 1] = df.iloc[:, 1].fillna(self.custom_fill_value)
        return df

    def resample_missing_dates(self, df, freq_dict, original_freq):
        """
        Resamples a pandas DataFrame to a specified frequency, fills in missing values with NaNs,
        and inserts missing dates as rows with NaN values. Also displays a message if there are
        missing dates in the data.

        Parameters:
        df (pandas DataFrame): The DataFrame to be resampled

        Returns:
        pandas DataFrame: The resampled DataFrame with missing dates inserted

        """
        # =============================================================================
        #     # TESTING
        #     # do test with original freq
        #     st.write(freq_dict)
        #     st.write(freq)
        #     st.write(df)
        # =============================================================================
        # =============================================================================
        #     # If resampling from higher frequency to lower frequency (e.g., monthly to quarterly),
        #     # fill the missing values with appropriate methods (e.g., mean)
        #     if freq_dict[freq] < freq_dict[original_freq]:
        #         new_df = new_df.set_index('date').resample(freq_dict[freq]).mean().asfreq()
        # =============================================================================

        # Resample the data to the specified frequency and fill in missing values with NaNs or forward-fill
        resampled_df = df.set_index('date').resample(freq_dict[self.freq]).asfreq()

        # Fill missing values with a specified method for non-daily data
        if freq_dict[self.freq] != 'D':
            # pad = forward fill
            resampled_df = resampled_df.fillna(method='pad')

        # Find skipped dates and insert them as rows with NaN values
        missing_dates = pd.date_range(start=resampled_df.index.min(), end=resampled_df.index.max(),
                                      freq=freq_dict[self.freq]).difference(resampled_df.index)
        new_df = resampled_df.reindex(resampled_df.index.union(missing_dates)).sort_index()

        # Display a message if there are skipped dates in the data
        if len(missing_dates) > 0:
            st.write("The skipped dates are:")
            st.write(missing_dates)

        # Reset the index and rename the columns
        return new_df.reset_index().rename(columns={'index': 'date'})

    def handle_outliers(self, data, method, outlier_threshold, q1, q3,
                        outlier_replacement_method='Median', contamination=0.01,
                        random_state=10, iqr_multiplier=1.5):
        """
        Detects and replaces outlier values in a dataset using various methods.

        Args:
            data (pd.DataFrame): The input dataset.
            method (str): The method used to detect and replace outliers. Possible values are:
            - 'Isolation Forest': uses the Isolation Forest algorithm to detect and replace outliers
            - 'Z-score': uses the Z-score method to detect and replace outliers
            - 'IQR': uses Tukey's method to detect and replace outliers
            outlier_threshold (float): The threshold used to determine outliers. This parameter is used only if method is 'Z-score'.
            outlier_replacement_method (str, optional): The method used to replace outliers. Possible values are 'Mean' and 'Median'. Defaults to 'Median'.
            contamination (float, optional): The expected amount of contamination in the dataset. This parameter is used only if method is 'Isolation Forest'. Defaults to 0.01.
            random_state (int, optional): The random state used for reproducibility. This parameter is used only if method is 'Isolation Forest'. Defaults to 10.
            iqr_multiplier (float, optional): The multiplier used to compute the lower and upper bounds for Tukey's method. This parameter is used only if method is 'IQR'. Defaults to 1.5.

        Returns:
            pd.DataFrame: A copy of the input dataset with outlier values replaced by either the mean or median of their respective columns, depending on the value of outlier_replacement_method.
        """
        # initialize the variable before using it
        #outliers = None
        if method == 'Isolation Forest':
            # detect and replace outlier values using Isolation Forest
            model = IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=random_state)
            model.fit(data)
            self.outliers = model.predict(data) == -1
        elif method == 'Z-score':
            # detect and replace outlier values using Z-score method
            z_scores = np.abs(stats.zscore(data))
            # The first array contains the indices of the outliers in your data variable.
            # The second array contains the actual z-scores of these outliers.
            outliers = np.where(z_scores > outlier_threshold)[0]
            # create a boolean mask indicating whether each row is an outlier or not
            is_outlier = np.zeros(data.shape[0], dtype=bool)
            is_outlier[outliers] = True
            self.outliers = is_outlier
        elif method == 'IQR':
            # detect and replace outlier values using Tukey's method
            q1, q3 = np.percentile(data, [25, 75], axis=0)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            outliers = ((data < lower_bound) | (data > upper_bound)).any(axis=1)
            # create a boolean mask indicating whether each row is an outlier or not
            is_outlier = np.zeros(data.shape[0], dtype=bool)
            is_outlier[outliers] = True
            self.outliers = is_outlier

        # select the rows that are outliers and create a new dataframe
        if outlier_replacement_method == 'Mean':
            means = data.mean()
            for col in data.columns:
                data[col][self.outliers] = means[col]
        elif outlier_replacement_method == 'Median':
            medians = data.median()
            for col in data.columns:
                data[col][self.outliers] = medians[col]
        elif outlier_replacement_method == 'Interpolation':
            # iterate over each column present in the dataframe
            for col in data.columns:
                # replace outliers with NaNs
                data[col][self.outliers] = np.nan
                # interpolate missing values using linear method
                data[col] = data[col].interpolate(method='linear')
                if pd.isnull(data[col].iloc[0]):
                    # Note: had an edge-case with quarterly data that had an outlier as first value and replaced it with NaN with interpolation and this imputes that NaN with second datapoint's value
                    st.warning(
                        f"⚠️ Note: The first value in column '{col}' is **NaN** and will be replaced during interpolation. This introduces some bias into the data.")
                # replace the first NaN value with the first non-NaN value in the column
                first_non_nan = data[col].dropna().iloc[0]
                data[col].fillna(first_non_nan, inplace=True)
        else:
            raise ValueError(f"Invalid outlier_replacement_method: {outlier_replacement_method}")
        return data, self.outliers

    def handle_missing_data(self, tab):
        with tab:
            with st.expander('', expanded=True):
                my_text_header('Handling missing data')

                # check if there are no dates skipped for frequency e.g. daily data missing days in between dates
                missing_dates = pd.date_range(start=st.session_state.df_raw['date'].min(),
                                              end=st.session_state.df_raw['date'].max()).difference(
                    st.session_state.df_raw['date'])

                # check if there are no missing values (NaN) in dataframe
                missing_values = st.session_state.df_raw.iloc[:, 1].isna().sum()

                # Plot missing values matrix with custom function
                self.plot_missing_values_matrix(df=self.df_cleaned_dates)

                # check if in continous time-series dataset no dates are missing in between
                if missing_dates.shape[0] == 0:
                    st.success('Pweh 😅, no dates were skipped in your dataframe!')
                else:
                    st.warning(
                        f'💡 **{missing_dates.shape[0]}** dates were skipped in your dataframe, don\'t worry though! I will **fix** this by **imputing** the dates into your cleaned dataframe!')
                if missing_values != 0 and self.fill_method == 'Backfill':
                    st.warning(
                        f'💡 **{missing_values}** missing values are filled with the next available value in the dataset (i.e. backfill method), optionally you can change the *filling method* and press **\"Submit\"** from the sidebar menu.')
                elif missing_values != 0 and self.fill_method != 'Custom':
                    st.warning(
                        f'💡 **{missing_values}** missing values are replaced by the **{self.fill_method}**, optionally you can change the *filling method* and press **\"Submit\"** from the sidebar menu.')
                elif missing_values != 0 and self.fill_method == 'Custom':
                    st.warning(
                        f'💡 **{missing_values}** missing values are replaced by custom value **{self.custom_fill_value}**, optionally you can change the *filling method* and press **\"Submit\"** from the sidebar menu.')

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
                    missing_df = copy_df_date_index(st.session_state.df_raw.loc[st.session_state.df_raw.iloc[:, 1].isna(), st.session_state.df_raw.columns],
                                                    datetime_to_date=True,
                                                    date_to_index=True)
                    st.write(missing_df)

                with col4:
                    st.markdown(icons["arrow_right_icon"], unsafe_allow_html=True)

                with col5:
                    my_subheader('Cleaned Dataframe', my_style="#333333", my_size=6)

                    # Show the cleaned dataframe with if needed dates inserted if skipped to NaN and then the values inserted with impute method user selected backfill/forward fill/mean/median
                    st.dataframe(self.df_clean_show, use_container_width=True)

                # Create and show download button in streamlit to user to download the dataframe with imputations performed to missing values
                download_csv_button(self.df_clean_show, my_file="df_imputed_missing_values.csv", set_index=True,
                                    help_message='Download cleaned dataframe to .CSV')

    def plot_missing_values_matrix(self, df):
        """
        Generates a Plotly Express figure of the missing values matrix for a DataFrame.

        Parameters:
        df (pandas DataFrame): The DataFrame to visualize the missing values matrix.

        """
        # Create Plotly Express figure
        # Create matrix of missing values
        missing_matrix = df.isnull()
        my_text_paragraph('Missing Values Matrix Plot')
        fig = px.imshow(missing_matrix,
                        labels=dict(x="Variables", y="Observations"),
                        x=missing_matrix.columns,
                        y=missing_matrix.index,
                        color_continuous_scale='Viridis',  # Viridis
                        title='')

        # Set Plotly configuration options
        fig.update_layout(width=400, height=400, margin=dict(l=50, r=50, t=0, b=50))
        fig.update_traces(showlegend=False)
        # Display Plotly Express figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def handle_outliers_tab(self, tab):
        with tab:
            with st.expander('', expanded=True):
                my_text_header('Handling outliers')
                my_text_paragraph('Outlier Plot')
                if self.outliers is not None and any(self.outliers):

                    self.outliers_df = copy_df_date_index(self.df_clean[self.outliers], datetime_to_date=True,
                                                     date_to_index=True).add_suffix('_outliers')

                    self.df_cleaned_outliers = self.df_cleaned_outliers.add_suffix('_outliers_replaced')

                    # inner join two dataframes of outliers and imputed values
                    self.outliers_df = self.outliers_df.join(self.df_cleaned_outliers, how='inner', rsuffix='_outliers_replaced')

                    ## OUTLIER FIGURE CODE
                    fig_outliers = go.Figure()
                    fig_outliers.add_trace(
                        go.Scatter(
                            x=self.df_clean['date'],
                            y=self.df_clean.iloc[:, 1],
                            mode='markers',
                            name='Before',
                            marker=dict(color='#440154'), opacity=0.5
                        )
                    )
                    # add scatterplot
                    fig_outliers.add_trace(
                        go.Scatter(
                            x=self.df_cleaned_outliers.index,
                            y=self.df_cleaned_outliers.iloc[:, 0],
                            mode='markers',
                            name='After',
                            marker=dict(color='#45B8AC'), opacity=1
                        )
                    )

                    df_diff = self.df_cleaned_outliers.loc[self.outliers]
                    # add scatterplot
                    fig_outliers.add_trace(go.Scatter(
                        x=df_diff.index,
                        y=df_diff.iloc[:, 0],
                        mode='markers',
                        name='Outliers After',
                        marker=dict(color='#FFC300'), opacity=1
                    ))

                    # show the outlier plot
                    st.session_state['fig_outliers'] = fig_outliers

                    st.plotly_chart(fig_outliers, use_container_width=True)

                    #  show the dataframe of outliers
                    st.info(
                        f'ℹ️ You replaced **{len(self.outliers_df)} outlier(s)** with their respective **{self.outlier_replacement_method}(s)** utilizing **{self.outlier_detection_method}**.')

                    # Apply the color scheme to the dataframe, round values by 2 decimals and display it in streamlit using full size of expander window
                    st.dataframe(self.outliers_df.style.format("{:.2f}").apply(self.highlight_cols, axis=0), use_container_width=True)

                    # add download button for user to be able to download outliers
                    download_csv_button(self.outliers_df, my_file="df_outliers.csv", set_index=True,
                                        help_message='Download outlier dataframe to .CSV', my_key='df_outliers')

                # if outliers are NOT found or None is selected as outlier detection method
                # ... run code...
                # Show scatterplot data without outliers
                else:
                    vertical_spacer(1)
                    fig_no_outliers = go.Figure()
                    fig_no_outliers.add_trace(go.Scatter(x=self.df_clean['date'],
                                                         y=self.df_clean.iloc[:, 1],
                                                         mode='markers',
                                                         name='Before',
                                                         marker=dict(color='#440154'), opacity=0.5))

                    st.plotly_chart(fig_no_outliers, use_container_width=True)
                    my_text_paragraph(f'No <b> outlier detection </b> or <b> outlier replacement </b> method selected.',
                                      my_font_size='14px')

    def outlier_form(self, key1: Union[str, int], key2: Union[str, int], key3: Union[str, int], key4: Union[str, int],
                     key5: Union[str, int], key6: Union[str, int], key7: Union[str, int]):
        """
        This function creates a streamlit form for handling outliers in a dataset using different outlier detection methods.
        It allows the user to select one of the following methods: Isolation Forest, Z-score, or IQR.
        - If Isolation Forest is selected, the user can specify the contamination parameter, which determines the proportion of samples in the dataset that are considered to be outliers.
        The function also sets a standard value for the random state parameter.
        - If Z-score is selected, the user can specify the outlier threshold, which is the number of standard deviations away from the mean beyond which a data point is considered an outlier.
        - If IQR is selected, the user can specify the first and third quartiles of the dataset and an IQR multiplier value, which is used to detect outliers by multiplying the interquartile range.
        - If K-means clustering is selected, the user can specify the number of clusters to form and the maximum number of iterations to run.

        The function also allows the user to select a replacement method for each outlier detection method, either mean or median.

        Returns:
            method (str): the selected outlier detection method
            contamination (float or None): the contamination parameter for Isolation Forest, or None if another method is selected
            outlier_replacement_method (str or None): the selected replacement method, or None if no method is selected
            random_state (int or None): the standard value for the random state parameter for Isolation Forest, or None if another method is selected
            outlier_threshold (float or None): the outlier threshold for Z-score, or None if another method is selected
            n_clusters (int or None): the number of clusters to form for K-means clustering, or None if another method is selected
            max_iter (int or None): the maximum number of iterations to run for K-means clustering, or None if another method is selected
        """
        # form for user to select outlier detection methods and parameters
        with st.form('outlier_form'):
            my_text_paragraph('Handling Outliers')
            # create user selectbox for outlier detection methods to choose from in Streamlit
            outlier_method = st.selectbox(
                label='*Select outlier detection method:*',
                options=('None', 'Isolation Forest', 'Z-score', 'IQR'),
                key=key1
            )

            # load when user selects "Isolation Forest" and presses 'Submit' detection algorithm parameters
            if outlier_method == 'Isolation Forest':
                col1, col2, col3 = st.columns([1, 12, 1])
                with col2:
                    contamination = st.slider(
                        label='Contamination:',
                        min_value=0.01,
                        max_value=0.5,
                        step=0.01,
                        key=key2,
                        help='''**`Contamination`** determines the *proportion of samples in the dataset that are considered to be outliers*.
                                                     It represents the expected fraction of the contamination within the data, which means it should be set to a value close to the percentage of outliers present in the data.  
                                                     A **higher** value of **contamination** will result in a **higher** number of **outliers** being detected, while a **lower** value will result in a **lower** number of **outliers** being detected.
                                                     '''
                    )
            # load when user selects "Z-Score" and presses 'Submit' detection algorithm parameters
            elif outlier_method == 'Z-score':
                col1, col2, col3 = st.columns([1, 12, 1])
                with col2:
                    outlier_threshold = st.slider(
                        label='Threshold:',
                        min_value=1.0,
                        max_value=10.0,
                        key=key3,
                        step=0.1,
                        help='Using a threshold of 3 for the z-score outlier detection means that any data point +3 standard deviations or -3 standard deviations away from the mean is considered an outlier'
                    )

            # load when user selects "IQR" and presses 'Submit' detection algorithm parameters
            elif outlier_method == 'IQR':
                col1, col2, col3 = st.columns([1, 12, 1])
                with col2:
                    q1 = st.slider(
                        label='Q1:',
                        min_value=0.0,
                        max_value=100.0,
                        step=1.0,
                        key=key4,
                        help='Determines the value of the first quantile. If you have a Streamlit slider set at 25%, it represents the value below which e.g. 25% of the data points fall.'
                    )
                    # value=75.0
                    q3 = st.slider(
                        label='Q3:',
                        min_value=0.0,
                        max_value=100.0,
                        step=1.0,
                        key=key5,
                        help='Determine the value of the third quantile. If you have a Streamlit slider set at 75%, it represents the value below which e.g. 75% of the data points fall.'
                    )

                    # value=1.5
                    iqr_multiplier = st.slider(
                        label='IQR multiplier:',
                        min_value=1.0,
                        max_value=5.0,
                        step=0.1,
                        key=key6,
                        help='''**`IQR multiplier`** determines the value used to multiply the **Interquartile range** to detect outliers.   
                                                      For example, a value of 1.5 means that any value outside the range is considered an outlier, see formula:  
                                                      \n$Q_1 - 1.5*IQR < outlier < Q_3 + 1.5*IQR$
                                                      \nWhere `Q1` and `Q3` are the first and third quartiles, respectively, and `IQR` is the `interquartile range`, which is equal to $Q3 - Q1$.  
                                                      Quantiles are calculated by sorting a dataset in ascending order and then dividing it into equal parts based on the desired quantile value.   
                                                      For example, to calculate the first quartile `Q1`, the dataset is divided into four equal parts, and the value at which 25% of the data falls below is taken as the first quartile. 
                                                      The same process is repeated to calculate the third quartile `Q3`, which is the value at which 75% of the data falls below.
                                                      '''
                    )
            elif outlier_method == 'None':
                # do nothing if preset or set by user the detection method to None
                pass
            else:
                st.write('outlier_form function has an issue with the parameters, please contact your administrator')

            # form to select outlier replacement method for each outlier detection method
            if outlier_method != 'None':
                outlier_replacement_method = st.selectbox(label='*Select outlier replacement method:*',
                                                          options=('Interpolation', 'Mean', 'Median'),
                                                          key=key7,
                                                          help='''**`Replacement method`** determines the actual value(s) to replace detected outlier(s) with.   
                                                            You can replace your outlier(s) with one of the following replacement methods:    
                                                            - *linear interpolation algorithm* **(default option)**  
                                                            - *mean*  
                                                            - *median*
                                                            ''')

            col1, col2, col3 = st.columns([4, 4, 4])
            with col2:
                st.form_submit_button(label='Submit',
                                      on_click=form_update,
                                      args=("CLEAN_PAGE",))

