# standard libraries
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer, \
    QuantileTransformer

# third-party libraries
from fire_state import get_state, set_state, form_update

# local packages
from style.text import my_title, my_text_header, my_text_paragraph, my_subheader, vertical_spacer
from style.animations import show_lottie_animation
from style.icons import load_icons

# load icons
icons = load_icons()


class PreparePage:
    """
    This class contains all the methods and functions to render the prepare page.
    """
    def __init__(self, state, key1_prepare_normalization, key2_prepare_standardization):
        # initialize session state
        self.my_insample_forecast_perc = None
        self.my_insample_forecast_steps = None
        self.state = state

        self.key1_prepare_normalization = key1_prepare_normalization
        self.key2_prepare_standardization = key2_prepare_standardization

        # =============================================================================
        # TEST WHICH DATAFRAME IS SHOULD USE AND NAMING CONVENTION
        # =============================================================================
        # create copy of dataframe not altering original
        # self.local_df = state['df'].copy(deep=True)
        self.local_df = get_state("DATAFRAMES", "df_cleaned_outliers_with_index")
        #st.write('self.local_df ja', get_state("DATAFRAMES", "df_cleaned_outliers_with_index"))

        #self.df = st.session_state['df']
        self.df = get_state("DATAFRAMES", "df_cleaned_outliers_with_index")
        #st.write('self df', self.df)

        # Assumption: date column and y column are at index 0 and index 1 so start from column 3 e.g. index 2 to
        # count potential numerical_features e.g. changed float64 to float to include other floats such as float32
        # and float16 data types
        self.numerical_features = list(self.df.iloc[:, 2:].select_dtypes(include=['float', 'int']).columns)
        st.write('test numerical features', self.numerical_features)

    def render(self):
        # create 4 tabs
        tab1_prepare, tab2_prepare, tab3_prepare, tab4_prepare = st.tabs(
            ['preprocess', 'train/test split', 'normalization', 'standardization'])

        # create sidebar with icon and train/test split user-form
        with st.sidebar:
            my_title(f'{icons["prepare_icon"]}', "#3b3b3b", gradient_colors="#1A2980, #FF9F00, #FEBD2E")

            # create train test user form
            self.my_insample_forecast_steps, self.my_insample_forecast_perc = self.train_test_split_slider(df=self.df)

        with tab1_prepare:
            self.preprocess()

        # with tab2_prepare:
        #     self.train_test_split()
        #
        # with tab3_prepare:
        #     X, y, X_train, X_test, y_train, y_test = self.normalize()
        #
        # with tab4_prepare:
        #     X, y, X_train, X_test, y_train, y_test = self.standardize(X, y, X_train, X_test, y_train, y_test)

    def preprocess(self):

        # NOTE TO SELF -> replaced with self.local_df might change naming convention
        # find the columns in the dataframe which have object as datatype
        obj_cols = self.local_df.select_dtypes(include='object').columns.tolist()

        # if there are any columns remove them
        if obj_cols:

            # with streamlit expander
            with st.expander('', expanded=True):

                # set header and subheader
                my_text_header('Preprocess')
                my_text_paragraph('*removing redundant features (dtype = object)', my_font_size='12px')

                # Call remove_object_columns method and get the modified DataFrame
                df = self.remove_object_columns(self.local_df, message_columns_removed=True)

                # have button available for user and if clicked it expands with the dataframe
                col1, col2, col3 = st.columns([130, 60, 120])
                with col2:
                    # initiate placeholder
                    placeholder = st.empty()

                    # create button (enabled to click e.g. disabled=false with unique key)
                    btn = placeholder.button('Show Data', disabled=False, key="preprocess_df_show_btn")

                # if button is clicked run below code
                if btn:
                    # display button with text "click me again", with unique key
                    placeholder.button('Hide Data', disabled=False, key="preprocess_df_hide_btn")

                    # show dataframe to user in streamlit
                    st.dataframe(df, use_container_width=True)

            vertical_spacer(1)

        st.image('./images/train_test_split_banner.png')

    def train_test_split(self):

        self.state['insample_forecast_steps'] = self.my_insample_forecast_steps
        self.state['insample_forecast_perc'] = self.my_insample_forecast_perc

    def normalize(self):
        normalization_choice = self.key1_prepare_normalization
        if normalization_choice != "None":
            X, y, X_train, X_test, y_train, y_test = self.perform_train_test_split()
            # Use X and y for further processing in normalization
            return X, y, X_train, X_test, y_train, y_test

    def standardize(self, X, y, X_train, X_test, y_train, y_test):
        standardization_choice = get_state("PREPARE", "standardization_choice")
        if standardization_choice != "None":
            # Use instance variables and methods from the class
            X, y, X_train, X_test, y_train, y_test = self.perform_train_test_split_standardization(
                X, y, X_train, X_test, y_train, y_test,
                self.state['insample_forecast_steps'],
                scaler_choice=standardization_choice,
                numerical_features=self.numerical_features
            )

    def remove_object_columns(self, df, message_columns_removed=False):
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

    def train_test_split_slider(self, df):
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

    def perform_train_test_split(self, scaler_choice=None, numerical_features=[]):
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
        X = self.df.iloc[:, 1:]
        y = self.df.iloc[:, 0:1]
        # =============================================================================
        #     # Find the index of the 'date' column
        #     date_column_index = df.columns.get_loc('date')
        #     # Get the date column + all columns
        #     # except the target feature which is assumed to be column after 'date' column
        #     X = df.iloc[:, :date_column_index+1].join(df.iloc[:, date_column_index+2:])
        #     y = df.iloc[:, date_column_index+1: date_column_index+2]
        # =============================================================================

        X_train = X.iloc[:-self.my_insample_forecast_steps, :]
        X_test = X.iloc[-self.my_insample_forecast_steps:, :]
        y_train = y.iloc[:-self.my_insample_forecast_steps, :]
        y_test = y.iloc[-self.my_insample_forecast_steps:, :]

        # initialize variable
        scaler = ""

        # Scale the data if user selected a scaler choice in the normalization / standardization in streamlit sidebar
        if get_state("PREPARE", "normalization_choice") != "None":  # changed to get_state from -> self.scaler_choice
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
                    raise ValueError(
                        "Invalid scaler choice. Please choose from: MinMaxScaler, RobustScaler, MaxAbsScaler, "
                        "PowerTransformer, QuantileTransformer")

                scaler = scaler_choices[scaler_choice]

                # Fit the scaler on the training set and transform both the training and test sets
                X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
                X_train_numeric_scaled = pd.DataFrame(X_train_numeric_scaled, columns=X_train_numeric.columns,
                                                      index=X_train_numeric.index)

                # note: you do not want to fit_transform on the test set else the distribution of the entire dataset is
                # used and is data leakage
                X_test_numeric_scaled = scaler.transform(X_test_numeric)
                X_test_numeric_scaled = pd.DataFrame(X_test_numeric_scaled,
                                                     columns=X_test_numeric.columns,
                                                     index=X_test_numeric.index)

                # refit the scaler on the entire exogenous features e.g. X which is used for forecasting beyond
                # train/test sets
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

    def perform_train_test_split_standardization(self, X, y, X_train, X_test, y_train, y_test,
                                                 my_insample_forecast_steps,
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
        # Initialize variables
        X_train_numeric_scaled = None
        X_test_numeric_scaled = None
        X_numeric_scaled = None

        # Check If the specified test-set size is greater than or equal to the total number of rows in the dataset.
        if self.my_insample_forecast_steps >= len(y):
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

        # Check if X_train_numeric_scaled is not None before subscripting
        if X_train_numeric_scaled is not None:
            X_train[numerical_features] = X_train_numeric_scaled[numerical_features]
        if X_test_numeric_scaled is not None:
            X_test[numerical_features] = X_test_numeric_scaled[numerical_features]
        if X_numeric_scaled is not None:
            X[numerical_features] = X_numeric_scaled[numerical_features]

        # Return the training and testing sets as well as the scaler used (if any)
        return X, y, X_train, X_test, y_train, y_test

