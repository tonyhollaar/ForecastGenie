# standard libraries
import streamlit as st
import pandas as pd
import holidays
from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday, SU, FR, SA)
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from pandas.tseries.offsets import BDay
import plotly.express as px

# third-party libraries
from fire_state import get_state, set_state, form_update, create_store

# local packages
from functions import download_csv_button, copy_df_date_index
from style.text import my_title, my_text_header, my_text_paragraph, vertical_spacer
from style.animations import show_lottie_animation
from style.icons import load_icons
import pywt

# load icons
icons = load_icons()


class EngineerPage:
    # CONSTANTS

    # Available countries and their country codes
    COUNTRY_DATA = [
        ('Albania', 'AL'),
        # ('Algeria', 'DZ'), -> Algeria has issues with holiday package
        ('American Samoa', 'AS'),
        ('Andorra', 'AD'),
        ('Angola', 'AO'),
        ('Argentina', 'AR'),
        ('Armenia', 'AM'),
        ('Aruba', 'AW'),
        ('Australia', 'AU'),
        ('Austria', 'AT'),
        ('Azerbaijan', 'AZ'),
        ('Bahrain', 'BH'),
        ('Bangladesh', 'BD'),
        ('Belarus', 'BY'),
        ('Belgium', 'BE'),
        ('Bolivia', 'BO'),
        ('Bosnia and Herzegovina', 'BA'),
        ('Botswana', 'BW'),
        ('Brazil', 'BR'),
        ('Bulgaria', 'BG'),
        ('Burundi', 'BI'),
        ('Canada', 'CA'),
        ('Chile', 'CL'),
        ('China', 'CN'),
        ('Colombia', 'CO'),
        ('Costa Rica', 'CR'),
        ('Croatia', 'HR'),
        ('Cuba', 'CU'),
        ('Curacao', 'CW'),
        ('Cyprus', 'CY'),
        ('Czechia', 'CZ'),
        ('Denmark', 'DK'),
        ('Djibouti', 'DJ'),
        ('Dominican Republic', 'DO'),
        ('Ecuador', 'EC'),
        ('Egypt', 'EG'),
        ('Estonia', 'EE'),
        ('Eswatini', 'SZ'),
        ('Ethiopia', 'ET'),
        ('Finland', 'FI'),
        ('France', 'FR'),
        ('Georgia', 'GE'),
        ('Germany', 'DE'),
        ('Greece', 'GR'),
        ('Guam', 'GU'),
        ('Honduras', 'HN'),
        ('Hong Kong', 'HK'),
        ('Hungary', 'HU'),
        ('Iceland', 'IS'),
        ('India', 'IN'),
        ('Indonesia', 'ID'),
        ('Ireland', 'IE'),
        ('Isle of Man', 'IM'),
        ('Israel', 'IL'),
        ('Italy', 'IT'),
        ('Jamaica', 'JM'),
        ('Japan', 'JP'),
        ('Kazakhstan', 'KZ'),
        ('Kenya', 'KE'),
        ('Kyrgyzstan', 'KG'),
        ('Latvia', 'LV'),
        ('Lesotho', 'LS'),
        ('Liechtenstein', 'LI'),
        ('Lithuania', 'LT'),
        ('Luxembourg', 'LU'),
        ('Madagascar', 'MG'),
        ('Malawi', 'MW'),
        ('Malaysia', 'MY'),
        ('Malta', 'MT'),
        ('Marshall Islands', 'MH'),
        ('Mexico', 'MX'),
        ('Moldova', 'MD'),
        ('Monaco', 'MC'),
        ('Montenegro', 'ME'),
        ('Morocco', 'MA'),
        ('Mozambique', 'MZ'),
        ('Namibia', 'NA'),
        ('Netherlands', 'NL'),
        ('New Zealand', 'NZ'),
        ('Nicaragua', 'NI'),
        ('Nigeria', 'NG'),
        ('Northern Mariana Islands', 'MP'),
        ('North Macedonia', 'MK'),
        ('Norway', 'NO'),
        ('Pakistan', 'PK'),
        ('Panama', 'PA'),
        ('Paraguay', 'PY'),
        ('Peru', 'PE'),
        ('Philippines', 'PH'),
        ('Poland', 'PL'),
        ('Portugal', 'PT'),
        ('Puerto Rico', 'PR'),
        ('Romania', 'RO'),
        ('Russia', 'RU'),
        ('San Marino', 'SM'),
        ('Saudi Arabia', 'SA'),
        ('Serbia', 'RS'),
        ('Singapore', 'SG'),
        ('Slovakia', 'SK'),
        ('Slovenia', 'SI'),
        ('South Africa', 'ZA'),
        ('South Korea', 'KR'),
        ('Spain', 'ES'),
        ('Sweden', 'SE'),
        ('Switzerland', 'CH'),
        ('Taiwan', 'TW'),
        ('Thailand', 'TH'),
        ('Tunisia', 'TN'),
        ('Turkey', 'TR'),
        ('Ukraine', 'UA'),
        ('United Arab Emirates', 'AE'),
        ('United Kingdom', 'GB'),
        ('United States Minor Outlying Islands', 'UM'),
        ('United States of America', 'US'),
        ('United States Virgin Islands', 'VI'),
        ('Uruguay', 'UY'),
        ('Uzbekistan', 'UZ'),
        ('Vatican City', 'VA'),
        ('Venezuela', 'VE'),
        ('Vietnam', 'VN'),
        ('Virgin Islands (U.S.)', 'VI'),
        ('Zambia', 'ZM'),
        ('Zimbabwe', 'ZW')
    ]

    def __init__(self, state, key1_engineer, key2_engineer, key3_engineer, key4_engineer, key5_engineer, key6_engineer,
                 key7_engineer, key1_engineer_page_country, key2_engineer_page_country, key3_engineer_page_country, ):
        self.my_bar = None
        self.state = state

        self.key1_engineer = key1_engineer
        self.key2_engineer = key2_engineer
        self.key3_engineer = key3_engineer
        self.key4_engineer = key4_engineer
        self.key5_engineer = key5_engineer
        self.key6_engineer = key6_engineer
        self.key7_engineer = key7_engineer

        self.key1_engineer_page_country = key1_engineer_page_country
        self.key2_engineer_page_country = key2_engineer_page_country
        self.key3_engineer_page_country = key3_engineer_page_country

        self.dwt_features_checkbox = None
        self.special_calendar_days_checkbox = None
        self.calendar_holidays_checkbox = None
        self.day_dummies_checkbox = None
        self.month_dummies_checkbox = None
        self.engineering_form_submit_btn = None
        self.wavelet_window_size_slider = None
        self.wavelet_level_decomposition_selectbox = None
        self.wavelet_family_selectbox = None
        self.year_dummies_checkbox = None
        self.calendar_dummies_checkbox = None

        self.df = pd.DataFrame()  # Initialize df as an empty DataFrame

    def render(self):
        self.my_bar = st.progress(0, text="Loading tabs... Please wait!")

        self.render_sidebar()
        self.render_tabs()

        self.my_bar.progress(100, text='100%')
        self.my_bar.empty()

    def render_sidebar(self):
        with st.sidebar:
            # show gear icon
            my_title(f"""{icons["engineer_icon"]}""",
                     "#3b3b3b",
                     gradient_colors="#1A2980, #FF6F61, #FEBD2E")

            # create sidebar user form
            with st.sidebar.form('feature engineering sidebar'):
                self.render_features()
                self.render_wavelet_settings()

    def render_tabs(self):
        tab1_engineer, tab2_engineer = st.tabs(['engineer', 'result'])

        with tab1_engineer:
            with st.expander("", expanded=True):
                self.render_tab1_content()

            with st.expander('🌊 Wavelet Features', expanded=True):
                # if user check-marked the box add wavelet features to dataframe
                self.create_wavelet_features()

        with tab2_engineer:
            with st.expander("", expanded=True):
                self.render_tab2_content()

    def render_tab1_content(self):
        show_lottie_animation(url="./images/aJ7Ra5vpQB.json", key="robot_engineering", width=350, height=350, speed=1,
                              col_sizes=[1, 3, 1])

        self.add_dummy_variables_section()
        self.add_country_holidays_section()
        self.add_special_calendar_days_checkboxes()
        self.add_special_calendar_days_section()
        self.add_numeric_date_feature()



    def render_tab2_content(self):
        my_text_header('Engineered Features')

        # Retrieve number of features to show on page
        # -2 -> because of datetime index and target variable
        num_features_df = len(self.df.columns) - 2
        my_text_paragraph(f'{num_features_df}')

        show_lottie_animation(url="./images/features_round.json", key="features_round", width=400, height=400)

        # Show dataframe in streamlit
        st.dataframe(copy_df_date_index(my_df=self.df, datetime_to_date=True, date_to_index=True),
                     use_container_width=True)

        # add download button
        download_csv_button(self.df, my_file="dataframe_incl_features.csv",
                            help_message="Download your dataset incl. features to .CSV")

    def render_features(self):
        my_text_paragraph('Features')
        vertical_spacer(1)
        col1, col2, col3 = st.columns([0.1, 8, 3])
        with col3:
            self.render_checkboxes()
        with col2:
            self.render_checkbox_messages()

    def render_wavelet_settings(self):
        with st.expander('🔽 Wavelet settings'):
            self.wavelet_family_selectbox = st.selectbox(label='*Select Wavelet Family*',
                                                         options=['db4', 'sym4', 'coif4'],
                                                         label_visibility='visible',
                                                         key=self.key5_engineer,
                                                         help='A wavelet family is a set of wavelet functions that '
                                                              'have different properties and characteristics.  \ '
                                                              '\n**`db4`** wavelet is commonly used for signals with '
                                                              '*smooth variations* and *short-duration* pulses  \ '
                                                              '\n**`sym4`** wavelet is suited for signals with *sharp '
                                                              'transitions* and *longer-duration* pulses.  \ '
                                                              '\n**`coif4`** wavelet, on the other hand, '
                                                              'is often used for signals with *non-linear trends* and '
                                                              '*abrupt* changes.  \ \nIn general, the **`db4`** '
                                                              'wavelet family is a good starting point, as it is a '
                                                              'popular choice for a wide range of applications and '
                                                              'has good overall performance.')

            # set standard level of decomposition to 3
            self.wavelet_level_decomposition_selectbox = st.selectbox('*Select Level of Decomposition*',
                                                                      [1, 2, 3, 4, 5],
                                                                      label_visibility='visible',
                                                                      key=self.key6_engineer,
                                                                      help='The level of decomposition refers to the '
                                                                           'number of times the signal is decomposed '
                                                                           'recursively into its approximation '
                                                                           'coefficients and detail coefficients.  \ '
                                                                           '\nIn wavelet decomposition, the signal is '
                                                                           'first decomposed into two components: a '
                                                                           'approximation component and a detail '
                                                                           'component.\ The approximation component '
                                                                           'represents the coarsest level of detail '
                                                                           'in the signal, while the detail component '
                                                                           'represents the finer details.  \ \nAt '
                                                                           'each subsequent level of decomposition, '
                                                                           'the approximation component from the '
                                                                           'previous level is decomposed again into '
                                                                           'its own approximation and detail '
                                                                           'components.\ This process is repeated '
                                                                           'until the desired level of decomposition '
                                                                           'is reached.  \ \nEach level of '
                                                                           'decomposition captures different '
                                                                           'frequency bands and details in the '
                                                                           'signal, with higher levels of '
                                                                           'decomposition capturing finer and more '
                                                                           'subtle details.  \ However, higher levels '
                                                                           'of decomposition also require more '
                                                                           'computation and may introduce more noise '
                                                                           'or artifacts in the resulting '
                                                                           'representation of the signal.  \ \nThe '
                                                                           'choice of the level of decomposition '
                                                                           'depends on the specific application and '
                                                                           'the desired balance between accuracy and '
                                                                           'computational efficiency.')

            # add slider or text input to choose window size
            self.wavelet_window_size_slider = st.slider(label='*Select Window Size (in days)*',
                                                        label_visibility='visible',
                                                        min_value=1,
                                                        max_value=30,
                                                        step=1,
                                                        key=self.key7_engineer)

        col1, col2, col3 = st.columns([4, 4, 4])
        with col2:
            # add submit button to form, when user presses it it updates the selection criteria
            self.engineering_form_submit_btn = st.form_submit_button('Submit', on_click=form_update,
                                                                     args=("ENGINEER_PAGE",))
            # if user presses the button on ENGINEER PAGE in sidebar
            if self.engineering_form_submit_btn:
                # update session state to True -> which is used to determine if default feature selection is chosen
                # or user selection in SELECT PAGE
                set_state("ENGINEER_PAGE_FEATURES_BTN", ('engineering_form_submit_btn', True))

    def render_checkboxes(self):

        # create checkbox for all seasonal days e.g. dummy variables for day/month/year
        self.calendar_dummies_checkbox = st.checkbox(label=' ',
                                                     label_visibility='visible',
                                                     key=self.key1_engineer,
                                                     help='Include independent features, namely create dummy '
                                                          'variables for'
                                                          'each `day` of the week, `month` and `year` whereby the '
                                                          'leave-1-out principle is applied to not have `perfect '
                                                          'multi-collinearity` i.e. the sum of the dummy variables for '
                                                          'each observation will otherwise always be equal to one.')
        # create checkbox for country holidays
        self.calendar_holidays_checkbox = st.checkbox(label=' ',
                                                      label_visibility='visible',
                                                      key=self.key2_engineer,
                                                      help='Include **`official holidays`** of a specified country  \
                                                      \n**(default = USA)**')

        # create checkbox for all special calendar days
        self.special_calendar_days_checkbox = st.checkbox(label=' ',
                                                          label_visibility='visible',
                                                          key=self.key3_engineer,
                                                          help='Include independent features including: '
                                                               '**`pay-days`** and'
                                                               'significant **`sales`** dates.')

        # create checkbox for Discrete Wavelet Transform features which automatically is checked
        self.dwt_features_checkbox = st.checkbox(label=' ',
                                                 label_visibility='visible',
                                                 key=self.key4_engineer,
                                                 help='''In feature engineering, wavelet transform is used to extract
                                                      useful information from a time series by decomposing it into
                                                      different frequency bands. This is done by applying a
                                                      mathematical function called the wavelet function to the time 
                                                      series data. The resulting wavelet coefficients can then be used
                                                      as features in machine learning models.''')

    def render_checkbox_messages(self):
        # show checkbox message if True/False for user options for Feature Selection
        st.write("*🌓 All Seasonal Periods*" if self.calendar_dummies_checkbox else "*🌓 No Seasonal Periods*")
        st.write("*⛱️ All Holiday Periods*" if self.calendar_holidays_checkbox else "*⛱️ No Holiday Periods*")
        st.write(
            "*🎁 All Special Calendar Days*" if self.special_calendar_days_checkbox else "*🎁 No Special Calendar Days*")
        st.write("*🌊 All Wavelet Features*" if self.dwt_features_checkbox else "*🌊 No Wavelet Features*")

    def add_dummy_variables_section(self):
        my_text_header('Dummy Variables')
        my_text_paragraph('🌓 Pick your time-based features to include:')
        vertical_spacer(1)

        if not self.calendar_dummies_checkbox:
            self.update_dummy_variables_checkboxes(False)
        else:
            self.update_dummy_variables_checkboxes(True)

        # create checkboxes for user to checkmark if to include features
        col0, col1, col2, col3, col4 = st.columns([2, 2, 2, 2, 1])
        with col1:
            self.year_dummies_checkbox = st.checkbox(label='Year',
                                                     value=get_state("ENGINEER_PAGE_VARS", "year_dummies_checkbox"))

            # add year dummies checkbox to session state
            set_state("ENGINEER_PAGE_VARS", ("year_dummies_checkbox", self.year_dummies_checkbox))

        with col2:
            self.month_dummies_checkbox = st.checkbox('Month', value=get_state("ENGINEER_PAGE_VARS",
                                                                               "month_dummies_checkbox"))
            # add month dummies checkbox to session state
            set_state("ENGINEER_PAGE_VARS", ("year_dummies_checkbox", self.month_dummies_checkbox))

        with col3:
            self.day_dummies_checkbox = st.checkbox('Day',
                                                    value=get_state("ENGINEER_PAGE_VARS", "day_dummies_checkbox"))

            # add day dummies checkbox to session state
            set_state("ENGINEER_PAGE_VARS", ("year_dummies_checkbox", self.day_dummies_checkbox))

    def update_dummy_variables_checkboxes(self, value):
        set_state("ENGINEER_PAGE_VARS", ('year_dummies_checkbox', value))
        set_state("ENGINEER_PAGE_VARS", ('month_dummies_checkbox', value))
        set_state("ENGINEER_PAGE_VARS", ('day_dummies_checkbox', value))

    def add_special_calendar_days_checkboxes(self):
        # Your existing logic for special calendar days checkboxes
        if not self.special_calendar_days_checkbox:
            self.update_special_calendar_days_checkboxes(False)
        else:
            self.update_special_calendar_days_checkboxes(True)

    def update_special_calendar_days_checkboxes(self, value):
        set_state("ENGINEER_PAGE_VARS", ('jan_sales', value))
        set_state("ENGINEER_PAGE_VARS", ('val_day_lod', value))
        set_state("ENGINEER_PAGE_VARS", ('val_day', value))
        set_state("ENGINEER_PAGE_VARS", ('mother_day_lod', value))
        set_state("ENGINEER_PAGE_VARS", ('mother_day', value))
        set_state("ENGINEER_PAGE_VARS", ('father_day_lod', value))
        set_state("ENGINEER_PAGE_VARS", ('pay_days', value))
        set_state("ENGINEER_PAGE_VARS", ('father_day', value))
        set_state("ENGINEER_PAGE_VARS", ('black_friday_lod', value))
        set_state("ENGINEER_PAGE_VARS", ('black_friday', value))
        set_state("ENGINEER_PAGE_VARS", ('cyber_monday', value))
        set_state("ENGINEER_PAGE_VARS", ('christmas_day', value))
        set_state("ENGINEER_PAGE_VARS", ('boxing_day', value))

    def add_country_holidays_section(self):
        vertical_spacer(1)
        my_text_header('Holidays')
        my_text_paragraph('⛱️ Select country-specific holidays to include:')

        selected_country_name = self.render_country_holiday_form(self.COUNTRY_DATA)

        if self.calendar_holidays_checkbox:
            self.apply_country_holidays()
        else:
            my_text_paragraph('<i> no country-specific holiday selected </i>')

    def apply_country_holidays(self):
        # apply function to create country-specific holidays in columns
        # is_holiday (boolean 1 if holiday otherwise 0) and holiday_desc for holiday_name
        self.df = self.create_calendar_holidays(df=self.state['df_cleaned_outliers_with_index'])

        # update the session_state
        #self.state['df_cleaned_outliers_with_index'] = self.df

        return self.df

    def add_special_calendar_days_section(self):
        ###############################################
        # create checkboxes for special days on page
        ###############################################
        my_text_header('Special Calendar Days')
        my_text_paragraph("🎁 Pick your special days to include: ")
        vertical_spacer(1)

        col0, col1, col2, col3 = st.columns([6, 12, 12, 1])
        with col1:
            self.add_special_checkbox("jan_sales", "January Sale")
            self.add_special_checkbox("val_day_lod", "Valentine's Day [last order date]")
            self.add_special_checkbox("val_day", "Valentine's Day")
            self.add_special_checkbox("mother_day_lod", "Mother's Day [last order date]")
            self.add_special_checkbox("mother_day", "Mother's Day")
            self.add_special_checkbox("father_day_lod", "Father's Day [last order date]")
            self.add_special_checkbox("pay_days", "Monthly Pay Days (4th Friday of month)")

        with col2:
            self.add_special_checkbox("father_day", "Father's Day")
            self.add_special_checkbox("black_friday_lod", "Black Friday [sale starts]")
            self.add_special_checkbox("black_friday", "Black Friday")
            self.add_special_checkbox("cyber_monday", "Cyber Monday")
            self.add_special_checkbox("christmas_day", "Christmas Day [last order date]")
            self.add_special_checkbox("boxing_day", "Boxing Day sale")
            vertical_spacer(3)

        # user checkmarked the box for all seasonal periods
        if self.special_calendar_days_checkbox:
            self.apply_special_calendar_days()
        else:
            self.df = self.state['df_cleaned_outliers_with_index']

        # if user check-marked the box for all seasonal periods
        if self.calendar_dummies_checkbox:
            # apply function to add year/month and day dummy variables
            self.df = self.create_date_features(self.df,
                                                year_dummies=self.year_dummies_checkbox,
                                                month_dummies=self.month_dummies_checkbox,
                                                day_dummies=self.day_dummies_checkbox)
            # update the session_state
            self.state['df_cleaned_outliers_with_index'] = self.df
        else:
            pass

    def add_special_checkbox(self, state_key, label):
        # create checkbox for special days in streamlit
        checkbox_value = st.checkbox(label=label,
                                     value=get_state("ENGINEER_PAGE_VARS", state_key),
                                     key=state_key)  # use state_key as the unique key for the checkbox

        # add checkbox value to session state e.g. True or False
        set_state("ENGINEER_PAGE_VARS", (state_key, checkbox_value))

    def apply_special_calendar_days(self):
        # call very extensive function to create all days selected by users as features
        self.df = self.create_calendar_special_days(self.state['df_cleaned_outliers_with_index'])

        # # update the session_state
        # self.state['df_cleaned_outliers_with_index'] = self.df
        set_state("DATAFRAMES", ("df_cleaned_outliers_with_index", self.df))

        return self.df

    def add_numeric_date_feature(self):
        """
        Returns: Add the date column but only as numeric feature
        """
        self.df['date_numeric'] = (self.df['date'] - self.df['date'].min()).dt.days
        set_state("DATAFRAMES", ("df_cleaned_outliers_with_index", self.df))

    def create_wavelet_features(self):
        """
        Create wavelet features if the dwt_features_checkbox is checked.
        """
        if self.dwt_features_checkbox:
            my_text_header('Discrete Wavelet Transform')
            my_text_paragraph('Feature Extraction')

            # CREATE WAVELET FEATURES
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
            # st.write('wavelet', wavelet, 'wavelet_level_decomposition_selectbox', level, 'window_size', window_size)

            # create empty list to store feature vectors
            feature_vectors = []

            # loop over each window in the data
            for i in range(window_size, len(self.df)):
                # extract data for current window
                data_in_window = self.df.iloc[i - window_size:i, 1].values

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

            # create a dataframe with the created features with discrete wavelet transform on target variable with
            # timewindow set by user
            features_df_wavelet = pd.DataFrame(feature_vectors, columns=feature_cols,
                                               index=self.df.iloc[:, 0].index[window_size:])

            # merge features dataframe with original data
            self.df = pd.merge(self.df, features_df_wavelet, left_index=True, right_index=True)

            # PLOT WAVELET FEATURES
            #######################
            # create a dataframe again with the index set as the first column
            # assumption used: the 'date' column is the first column of the dataframe
            features_df_plot = pd.DataFrame(feature_vectors, columns=feature_cols, index=self.df.iloc[:, 0])

            fig = px.line(features_df_plot,
                          x=features_df_plot.index,
                          y=['approx_mean'] + [f'detail{i + 1}_mean' for i in range(level)],
                          title='',
                          labels={'value': 'Coefficient Mean', 'variable': 'Subband'})

            fig.update_layout(xaxis_title='Date')

            st.plotly_chart(fig, use_container_width=True)

            # SHOW WAVELET FEATURES DATAFRAME
            # Show Dataframe with features
            my_text_paragraph('Wavelet Features Dataframe')
            st.dataframe(features_df_wavelet, use_container_width=True)

            # update the session state
            # self.state['df_cleaned_outliers_with_index'] = self.df
            set_state("DATAFRAMES", ("df_cleaned_outliers_with_index", self.df))
        else:
            pass

    @staticmethod
    def create_date_features(df, year_dummies=True, month_dummies=True, day_dummies=True):
        """
        This function creates dummy variables for year, month, and day of week from a date column in a pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): Input DataFrame containing a date column.
        year_dummies (bool): Flag to indicate if year dummy variables are needed. Default is True.
        month_dummies (bool): Flag to indicate if month dummy variables are needed. Default is True.
        day_dummies (bool): Flag to indicate if day of week dummy variables are needed. Default is True.

        Returns:
        pandas.DataFrame: A new DataFrame with added dummy variables.
        """
        if year_dummies:
            df['year'] = df['date'].dt.year
            dum_year = pd.get_dummies(df['year'], columns=['year'], drop_first=True, prefix='year', prefix_sep='_')
            df = pd.concat([df, dum_year], axis=1)
        if month_dummies:
            month_dict = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July',
                          8: 'August',
                          9: 'September', 10: 'October', 11: 'November', 12: 'December'}
            df['month'] = df['date'].dt.month.apply(lambda x: month_dict.get(x))
            dum_month = pd.get_dummies(df['month'], columns=['month'], drop_first=True, prefix='', prefix_sep='')
            df = pd.concat([df, dum_month], axis=1)
        if day_dummies:
            week_dict = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday',
                         6: 'Sunday'}
            df['day'] = df['date'].dt.weekday.apply(lambda x: week_dict.get(x))
            dum_day = pd.get_dummies(df['day'], columns=['day'], drop_first=True, prefix='', prefix_sep='')
            df = pd.concat([df, dum_day], axis=1)
        # Drop any duplicate columns based on their column names
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def render_country_holiday_form(self, country_data):
        """

        Args:
            country_data: a list of tuples of country name and country 2-letter code

        Returns:

        """
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            with st.form('country_holiday'):
                selected_country_name = st.selectbox(label="Select a country",
                                                     options=[country[0] for country in country_data],
                                                     key=self.key1_engineer_page_country,
                                                     label_visibility='collapsed')

                col1, col2, col3 = st.columns([5, 4, 4])
                with col2:
                    country_holiday_btn = st.form_submit_button('Apply', on_click=form_update,
                                                                args=("ENGINEER_PAGE_COUNTRY_HOLIDAY",))

                    # update country code as well in session state -
                    # which is used for prophet model holiday feature as well
                    set_state("ENGINEER_PAGE_COUNTRY_HOLIDAY",
                              ("country_code", dict(country_data).get(selected_country_name)))

        return selected_country_name

    def create_calendar_holidays(self, df: pd.DataFrame, slider: bool = True):
        """
        Create a calendar of holidays for a given DataFrame.

        Parameters:
        - df (pandas.DataFrame): The input DataFrame containing a 'date' column.

        Returns: - pandas.DataFrame: The input DataFrame with additional columns 'is_holiday' (1 if the date is a
        holiday, 0 otherwise) and 'holiday_desc' (description of the holiday, empty string if not a holiday).
        """
        # try:
        # Note: create_calendar_holidays FUNCTION BUILD ON TOP OF HOLIDAY PACKAGE
        # some countries like Algeria have issues therefore if/else statement to catch it
        # Define variables
        start_date = df['date'].min()
        end_date = df['date'].max()

        # retrieve index of default country e.g. 'United States of America'
        us_index = self.COUNTRY_DATA.index(('United States of America', 'US'))

        selected_country_name = get_state("ENGINEER_PAGE_COUNTRY_HOLIDAY", "country_name")

        # create empty container for the calendar
        country_calendars = {}

        # iterate over all countries and try-except block for if holiday for country is not found
        # in holiday python package
        for name, code in self.COUNTRY_DATA:
            try:
                country_calendars[name] = getattr(holidays, code)()
            except AttributeError:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    print(f"No holiday calendar found for country: {name}")
                continue

        # Retrieve the country code for the selected country
        selected_country_code = dict(self.COUNTRY_DATA).get(get_state("ENGINEER_PAGE_COUNTRY_HOLIDAY", "country_name"))
        # st.write(selected_country_code, selected_country_name) # TEST

        # Check if the selected country has a holiday calendar
        if selected_country_name in country_calendars.keys():

            country_holidays = holidays.country_holidays(selected_country_code)
            # Set the start and end date for the date range
            range_of_dates = pd.date_range(start_date, end_date)

            # create a dataframe for the date range
            df_country_holidays = pd.DataFrame(index=range_of_dates)
            # add boolean (1 if holiday date else 0)
            df_country_holidays['is_holiday'] = [1 if date in country_holidays else 0 for date in range_of_dates]
            # add holiday description
            df_country_holidays['holiday_desc'] = [country_holidays.get(date, '') for date in range_of_dates]

            # st.write(df_country_holidays) # TEST IF TWO COLUMNS ARE CREATED CORRECTLY FOR COUNTRY HOLIDAYS

            # merge dataframe of index with dates, is_holiday, holiday_desc with original df
            df = pd.merge(df, df_country_holidays, left_on='date', right_index=True, how='left')
            # st.write(df) # TEST IF MERGE WAS SUCCESFULL
            return df

        else:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                return st.error(f"⚠️No holiday calendar found for country: **{selected_country_name}**")
        # except:
        #     st.error(
        #         'Forecastgenie Error: the function create_calendar_holidays() could not execute correctly, please contact the administrator...')

    def create_calendar_special_days(self, df, start_date_calendar=None, end_date_calendar=None):
        """
        # source: https://practicaldatascience.co.uk/data-science/how-to-create-an-ecommerce-trading-calendar-using-pandas
        Create a trading calendar for an ecommerce business in the UK.

        Parameters:
        df (pd.DataFrame): Cleaned DataFrame containing order data with index
        special_calendar_days_checkbox (bool): Whether to select all days or only specific special days

        Returns:
        df_exogenous_vars (pd.DataFrame): DataFrame containing trading calendar with holiday and event columns
        """
        ###############################################
        # Define variables
        ###############################################
        if start_date_calendar is None:
            # set start date to minimum date in dataframe
            start_date_calendar = df['date'].min()
        else:
            # set start date to user input
            start_date_calendar = start_date_calendar
        if end_date_calendar is None:
            # set end date to maximum date in dataframe
            end_date_calendar = df['date'].max()
        else:
            # set end date to user input
            end_date_calendar = end_date_calendar

        # create a pandas dataframe with a date column containing all dates between start and end date
        df_exogenous_vars = pd.DataFrame({'date': pd.date_range(start=start_date_calendar, end=end_date_calendar)})

        class UKEcommerceTradingCalendar(AbstractHolidayCalendar):
            # create an empty list to add rules for the calendar
            rules = []

            # Seasonal trading events
            # only add Holiday if user checkmarked checkbox (e.g. equals True)
            special_days = {
                "jan_sales": Holiday('January sale', month=1, day=1),
                "val_day_lod": Holiday('Valentine\'s Day [last order date]', month=2, day=14, offset=BDay(-2)),
                "val_day": Holiday('Valentine\'s Day', month=2, day=14),
                "mother_day_lod": Holiday('Mother\'s Day [last order date]', month=5, day=1, offset=BDay(-2)),
                "mother_day": Holiday('Mother\'s Day', month=5, day=1, offset=pd.DateOffset(weekday=SU(2))),
                "father_day_lod": Holiday('Father\'s Day [last order date]', month=6, day=1, offset=BDay(-2)),
                "father_day": Holiday('Father\'s Day', month=6, day=1, offset=pd.DateOffset(weekday=SU(3))),
                "black_friday_lod": Holiday("Black Friday [sale starts]", month=11, day=1,
                                            offset=[pd.DateOffset(weekday=SA(4)), BDay(-5)]),
                "black_friday": Holiday('Black Friday', month=11, day=1, offset=pd.DateOffset(weekday=FR(4))),
                "cyber_monday": Holiday("Cyber Monday", month=11, day=1,
                                        offset=[pd.DateOffset(weekday=SA(4)), pd.DateOffset(2)]),
                "christmas_day": Holiday('Christmas Day [last order date]', month=12, day=25, offset=BDay(-2)),
                "boxing_day": Holiday('Boxing Day sale', month=12, day=26)
            }

            for key, holiday in special_days.items():
                if get_state("ENGINEER_PAGE_VARS", key):
                    rules.append(holiday)

        calendar = UKEcommerceTradingCalendar()
        start = df_exogenous_vars.date.min()
        end = df_exogenous_vars.date.max()
        events = calendar.holidays(start=start, end=end, return_name=True)
        events = events.reset_index(name='calendar_event_desc').rename(columns={'index': 'date'})
        df_exogenous_vars = df_exogenous_vars.merge(events, on='date', how='left').fillna('')

        # source: https://splunktool.com/holiday-calendar-in-pandas-dataframe

        ###############################################
        # Create Pay Days
        ###############################################
        class UKEcommerceTradingCalendar(AbstractHolidayCalendar):
            rules = []
            # Pay days(based on fourth Friday of the month)
            if get_state("ENGINEER_PAGE_VARS", "pay_days") == True:
                rules = [
                    Holiday('January Pay Day', month=1, day=31, offset=BDay(-1)),
                    Holiday('February Pay Day', month=2, day=28, offset=BDay(-1)),
                    Holiday('March Pay Day', month=3, day=31, offset=BDay(-1)),
                    Holiday('April Pay Day', month=4, day=30, offset=BDay(-1)),
                    Holiday('May Pay Day', month=5, day=31, offset=BDay(-1)),
                    Holiday('June Pay Day', month=6, day=30, offset=BDay(-1)),
                    Holiday('July Pay Day', month=7, day=31, offset=BDay(-1)),
                    Holiday('August Pay Day', month=8, day=31, offset=BDay(-1)),
                    Holiday('September Pay Day', month=9, day=30, offset=BDay(-1)),
                    Holiday('October Pay Day', month=10, day=31, offset=BDay(-1)),
                    Holiday('November Pay Day', month=11, day=30, offset=BDay(-1)),
                    Holiday('December Pay Day', month=12, day=31, offset=BDay(-1))
                ]

        # create calendar
        calendar = UKEcommerceTradingCalendar()
        # create date range
        start = df_exogenous_vars.date.min()
        end = df_exogenous_vars.date.max()

        # create dataframe with pay days
        events = calendar.holidays(start=start, end=end, return_name=True)
        events = events.reset_index(name='pay_day_desc').rename(columns={'index': 'date'})

        # merge dataframe with pay days with original dataframe
        df_exogenous_vars = df_exogenous_vars.merge(events, on='date', how='left').fillna('')

        # create boolean column for pay days
        df_exogenous_vars['pay_day'] = df_exogenous_vars['pay_day_desc'].apply(lambda x: 1 if len(x) > 1 else 0)
        df_exogenous_vars['calendar_event'] = df_exogenous_vars['calendar_event_desc'].apply(
            lambda x: 1 if len(x) > 1 else 0)

        ###############################################################################
        # Reorder Columns to logical order e.g. value | description of value
        ###############################################################################
        df_exogenous_vars = df_exogenous_vars[
            ['date', 'calendar_event', 'calendar_event_desc', 'pay_day', 'pay_day_desc']]

        ###############################################################################
        # combine exogenous vars with df
        ###############################################################################
        df_total_incl_exogenous = pd.merge(df, df_exogenous_vars, on='date', how='left')

        # create a copy of the df
        df = df_total_incl_exogenous.copy(deep=True)

        return df

    def perform_data_engineering(self):
        # TODO - create conditionals

        # Apply special calendar days if checkbox is checked
        self.df = self.apply_special_calendar_days()
        self.state['df_cleaned_outliers_with_index'] = self.df  # update the session_state

        # Apply country holidays if checkbox is checked
        self.df = self.apply_country_holidays()
        self.state['df_cleaned_outliers_with_index'] = self.df  # update the session_state

        # Create dummy variables if checkbox is checked
        self.df = self.create_date_features(self.state['df_cleaned_outliers_with_index'],
                                            year_dummies=self.year_dummies_checkbox,
                                            month_dummies=self.month_dummies_checkbox,
                                            day_dummies=self.day_dummies_checkbox)
        self.state['df_cleaned_outliers_with_index'] = self.df # update the session_state

        # Create Wavelet features
        #self.create_wavelet_features()

        #set_state("DATAFRAMES", ("df_cleaned_outliers_with_index", self.df))
