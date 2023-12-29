# standard libraries
import pandas as pd
import plotly.express as px
import streamlit as st

# third-party libraries
from streamlit_extras.dataframe_explorer import dataframe_explorer
from fire_state import get_state, set_state

# local packages
from style.text import my_title, my_text_header, my_text_paragraph, vertical_spacer
from style.icons import load_icons
from functions import (generate_demo_data, load_data, copy_df_date_index, display_dataframe_graph, download_csv_button,
                       update_color)

from utils.config import SessionState

# load icons
icons = load_icons()


class LoadPage:
    # headers constants
    UPLOAD_DATA_HEADER = "Upload Data"
    DEMO_DATA_HEADER = "Demo Data"

    # supported file types constants
    SUPPORTED_FILE_TYPES = ["csv", "xls", "xlsx", "xlsm", "xlsb"]

    # data related constants
    DATE_COLUMN = 'date'
    DEMO_DATA_FILENAME = 'demo_data.csv'

    # file paths constants
    LOAD_PAGE_IMAGE_PATH = "./images/load.png"

    def __init__(self, state):
        self.state = state

    def render(self):
        tab1_load, tab2_load = st.tabs(['Load', 'Dashboard'])
        uploaded_file = self.render_sidebar()
        with tab1_load:
            self.render_load_tab(uploaded_file)
        with tab2_load:
            with st.expander('', expanded=True):
                self.create_relative_change_plot()
                self.create_relative_change_table(value_column=self.state['df_raw'].iloc[:, 1].name)

    def render_load_tab(self, uploaded_file):
        # If the user has selected "Demo Data", call the method to render demo data
        if get_state('LOAD_PAGE', "my_data_choice") == "Demo Data":
            self.render_demo_data()

        # If the user has selected "Upload Data", and either a file has been uploaded or a file was previously uploaded
        # (indicated by the "user_data_uploaded" state being True), and 'demo_data' is not a column in the dataframe
        # (indicating that the data in the dataframe is not demo data), call the method to render the uploaded data
        if (get_state('LOAD_PAGE', "my_data_choice") == "Upload Data" and
                uploaded_file is not None or
                get_state("LOAD_PAGE", "user_data_uploaded") == True and
                'demo_data' not in st.session_state['df_raw'].columns):
            self.render_upload_data(uploaded_file)

        # If the user has selected "Upload Data", but no file has been uploaded and no file was previously uploaded
        # (indicated by the "user_data_uploaded" state being False), call the method to render instructions
        elif (get_state('LOAD_PAGE', "my_data_choice") == "Upload Data" and
              ((uploaded_file is None) or (get_state("LOAD_PAGE", "user_data_uploaded") == False))):
            self.render_instructions()

    def load_change(self):
        try:
            data_option = get_state('LOAD_PAGE', "my_data_choice")
            index_data_option = ("Upload Data" if data_option == "Demo Data" else "Demo Data")
            set_state("LOAD_PAGE", ("my_data_choice", index_data_option))
        except:
            set_state("LOAD_PAGE", ("my_data_choice", "Demo Data"))

    def render_sidebar(self):
        uploaded_file = None  # Initialize with a default value
        with st.sidebar:

            my_title(f"""{icons["load_icon"]}""", "#3b3b3b")

            with st.expander('', expanded=True):
                col1, col2, col3 = st.columns(3)
                with col2:
                    # =============================================================================
                    #  STEP 1: SAVE USER DATA CHOICE (DEMO/UPLOAD) WITH RADIO BUTTON
                    # =============================================================================
                    data_option = st.radio(
                        label="*Choose an option:*",
                        options=["Demo Data", "Upload Data"],
                        index=0 if get_state('LOAD_PAGE', "my_data_choice") == "Demo Data" else 1,
                        on_change=self.load_change  # run function
                    )

                    # update session state that user uploaded a file
                    set_state("LOAD_PAGE", ("my_data_choice", data_option))

                    vertical_spacer(1)

                # =============================================================================
                # STEP 2: IF UPLOADED FILE THEN SHOW FILE-UPLOADER WIDGET
                # =============================================================================
                if data_option == "Upload Data":
                    # show file-uploader widget from streamlit
                    uploaded_file = st.file_uploader(label="Upload your file",
                                                     type=LoadPage.SUPPORTED_FILE_TYPES,
                                                     accept_multiple_files=False,
                                                     label_visibility='collapsed',
                                                     on_change=SessionState.reset_session_states)

                    # if a file is uploaded by user then save the file in session state
                    if uploaded_file is not None:
                        # update session state that user uploaded a file
                        set_state("LOAD_PAGE", ("uploaded_file_name", uploaded_file.name))

                    # if prior a file was uploaded by user (note: st.file_uploader after page refresh does not keep
                    # the object in it)
                    if get_state('LOAD_PAGE', "user_data_uploaded") == True:
                        # retrieve filename from session state
                        file_name = get_state("LOAD_PAGE", "uploaded_file_name")

                        # show filename
                        my_text_paragraph(f'{file_name}', my_font_weight=100, my_font_size='14px')
        return uploaded_file

    def render_demo_data(self):
        # update session state that user uploaded a file
        set_state("LOAD_PAGE", ("user_data_uploaded", False))

        # GENERATE THE DEMO DATA WITH FUNCTION
        st.session_state['df_raw'] = generate_demo_data()

        # df_raw = generate_demo_data()
        df_raw = st.session_state['df_raw']
        df_graph = df_raw.copy(deep=True)
        df_min = df_raw.iloc[:, 0].min().date()
        df_max = df_raw.iloc[:, 0].max().date()

        with st.expander('', expanded=True):
            col0, col1, col2, col3 = st.columns([10, 90, 8, 1])

            with col2:
                my_chart_color = st.color_picker(
                    label='Color',
                    value=get_state("COLORS", "chart_color"),
                    on_change=update_color(get_state("COLORS", "chart_color")),
                    label_visibility='collapsed'
                )

            with col1:
                my_text_header(LoadPage.DEMO_DATA_HEADER)

            # create 3 columns for spacing
            col1, col2, col3 = st.columns([1, 3, 1])

            # short message about dataframe that has been loaded with shape (# rows, # columns)
            col2.markdown(
                f"<center>Your <b>dataframe</b> has <b><font color='#555555'>{st.session_state.df_raw.shape[0]}</b></font> \
                                               rows and <b><font color='#555555'>{st.session_state.df_raw.shape[1]}</b></font> columns <br> with date range: \
                                               <b><font color='#555555'>{df_min}</b></font> to <b><font color='#555555'>{df_max}</font></b>.</center>",
                unsafe_allow_html=True)

            # create aa deep copy of dataframe which will be manipulated for graphs
            df_graph = copy_df_date_index(my_df=df_graph, datetime_to_date=True, date_to_index=True)

            # Display Plotly Express figure in Streamlit
            display_dataframe_graph(df=df_graph, key=1, my_chart_color=my_chart_color)

            # try to use add-on package of streamlit dataframe_explorer
            try:
                df_explore = dataframe_explorer(st.session_state['df_raw'])
                st.dataframe(df_explore, use_container_width=True)
            # if add-on package does not work use the regular dataframe without index
            except:
                st.dataframe(df_graph, use_container_width=True)

            # download csv button
            download_csv_button(df_graph, my_file=LoadPage.DEMO_DATA_FILENAME,
                                help_message='Download dataframe to .CSV',
                                set_index=True)

            vertical_spacer(1)

    def render_upload_data(self, uploaded_file):

        if uploaded_file is not None:
            # define dataframe from custom function to read from uploaded read_csv file
            st.session_state['df_raw'] = load_data(uploaded_file)

            # update session state that user uploaded a file
            set_state("LOAD_PAGE", ("user_data_uploaded", True))

            # update session state that new file is uploaded / set to True -> use variable reset user selection of features list to empty list
            set_state("DATA_OPTION", ("upload_new_data", True))

        # for multi-page app if user previously uploaded data uploaded_file will return to None but then get the persistent session state stored
        if get_state('LOAD_PAGE', "my_data_choice") == "Upload Data" and ((uploaded_file is not None) or (
                get_state("LOAD_PAGE", "user_data_uploaded") == True)) and 'demo_data' not in st.session_state[
            'df_raw'].columns:

            # define dataframe copy used for graphs
            df_graph = st.session_state['df_raw'].copy(deep=True)
            # set minimum date
            df_min = st.session_state['df_raw'].iloc[:, 0].min().date()
            # set maximum date
            df_max = st.session_state['df_raw'].iloc[:, 0].max().date()

            # let user select color for graph
            with st.expander('', expanded=True):
                col0, col1, col2, col3 = st.columns([15, 90, 8, 1])
                with col2:
                    my_chart_color = st.color_picker(label='Color',
                                                     value=get_state("COLORS", "chart_color"),
                                                     on_change=update_color(get_state("COLORS", "chart_color")),
                                                     label_visibility='collapsed')
                with col1:
                    my_text_header(LoadPage.UPLOAD_DATA_HEADER)

                # create 3 columns for spacing
                col1, col2, col3 = st.columns([1, 3, 1])

                # display df shape and date range min/max for user
                col2.markdown(
                    f"<center>Your <b>dataframe</b> has <b><font color='#555555'>{st.session_state.df_raw.shape[0]}</b></font> \
                                      rows and <b><font color='#555555'>{st.session_state.df_raw.shape[1]}</b></font> columns <br> with date range: \
                                      <b><font color='#555555'>{df_min}</b></font> to <b><font color='#555555'>{df_max}</font></b>.</center>",
                    unsafe_allow_html=True)

                vertical_spacer(1)

                df_graph = copy_df_date_index(my_df=df_graph, datetime_to_date=True, date_to_index=True)

                vertical_spacer(1)

                # display/plot graph of dataframe
                display_dataframe_graph(df=df_graph, key=2, my_chart_color=my_chart_color)

                try:
                    # show dataframe below graph
                    df_explore = dataframe_explorer(st.session_state['df_raw'])
                    st.dataframe(df_explore, use_container_width=True)
                except:
                    st.dataframe(df_graph, use_container_width=True)

                # download csv button
                download_csv_button(df_graph, my_file="raw_data.csv", help_message='Download dataframe to .CSV',
                                    set_index=True)

    def render_instructions(self):
        with st.expander("", expanded=True):
            my_text_header("Instructions")
            vertical_spacer(2)
            col1, col2, col3 = st.columns([1, 8, 1])
            with col2:
                my_text_paragraph('''👈 Please upload a file with your <b><span style="color:#000000">dates</span></b> and <b><span style="color:#000000">values</span></b> in below order:<br><br>
                                            - first column named: <b><span style="color:#000000">date</span></b> in format: mm/dd/yyyy e.g. 12/31/2023<br>
                                            - second column named:  <b><span style="color:#000000">&#60;insert variable name&#62;</span></b> e.g. revenue<br>
                                            - supported frequencies: Daily/Weekly/Monthly/Quarterly/Yearly <br>
                                            - supported file extensions: .CSV, .XLS, .XLSX, .XLSM, .XLSB
                                            ''', my_font_weight=300, my_text_align='left')
                vertical_spacer(2)

            # Display Doodle image of Load
            st.image(self.LOAD_PAGE_IMAGE_PATH, caption="", use_column_width=True)

    def calculate_relative_change(self, date_column, value_column, relative_change_type):
        """
        Calculate the relative change of a given value column based on the selected type of relative change.

        Args:
            date_column (str): Name of the date column.
            value_column (str): Name of the value column.
            relative_change_type (str): Type of relative change to calculate.

        Returns:
            pandas.Series: Relative change of the value column based on the selected type of relative change.
        """
        # Convert the date column to a pandas DateTimeIndex
        data = self.state['df_raw'].copy(deep=True)
        data[date_column] = pd.to_datetime(data[date_column])
        data.set_index(date_column, inplace=True)

        # Calculate the relative change based on the selected type
        if relative_change_type == 'DoD%Δ':
            relative_change = data[value_column].pct_change() * 100
        elif relative_change_type == 'WoW%Δ':
            relative_change = data[value_column].pct_change(7) * 100
        elif relative_change_type == 'MoM%Δ':
            relative_change = data[value_column].pct_change(30) * 100
        elif relative_change_type == 'YoY%Δ':
            relative_change = data[value_column].pct_change(365) * 100
        else:
            relative_change = pd.Series()
        return relative_change

    def create_relative_change_plot(self):
        col0, col1, col2, col3 = st.columns([15, 90, 8, 1])
        with col2:
            my_chart_color = st.color_picker(label='Color2',
                                             value=get_state("COLORS", "chart_color"),
                                             on_change=update_color(get_state("COLORS", "chart_color")),
                                             label_visibility='collapsed')

        with col1:
            if get_state('LOAD_PAGE', "my_data_choice") == "Demo Data":
                my_text_header(LoadPage.DEMO_DATA_HEADER)
            elif get_state('LOAD_PAGE', "my_data_choice") == "Upload Data":
                my_text_header(LoadPage.UPLOAD_DATA_HEADER)

        # Retrieve the column name based on its relative position (position 1)
        value_column = self.state['df_raw'].iloc[:, 1].name

        # Call the function to calculate the relative change
        relative_change_type = st.radio(label='Select relative change type:',
                                        options=['DoD%Δ', 'WoW%Δ', 'MoM%Δ', 'YoY%Δ'],
                                        horizontal=True,
                                        help="""The `relative change` is calculated as the percentage difference between 
                                        the value of a specific day and that of a prior period.  
                                            - `DoD%Δ`: Day-over-day percentage change   
                                            - `WoW%Δ`: Week-over-week percentage change   
                                            - `MoM%Δ`: Month-over-month percentage change   
                                            - `YoY%Δ`: Year-over-year percentage change""")

        # Calculate the relative change
        relative_change = self.calculate_relative_change(
            date_column=LoadPage.DATE_COLUMN,
            value_column=value_column,
            relative_change_type=relative_change_type
        )

        # Filter out empty (NaN) values when comparing relative change
        relative_change = relative_change.dropna()

        # Convert the relative_change Series to a DataFrame
        relative_change = relative_change.to_frame(name=value_column)

        # Add a new column for the day of the week
        relative_change['day_of_week'] = relative_change.index.day_name()

        # Create the bar chart using Plotly Express
        fig = px.bar(relative_change, x=relative_change.index, y=value_column, hover_data=['day_of_week'])

        fig.update_layout(
            xaxis_title='Day of the Week',
            yaxis_title='Relative Change (%)',
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
                    y=0.9,
                    yanchor='auto',
                    font=dict(size=10),
                ),
                type='date'
            ),
            yaxis=dict(autorange=True),
            height=600,  # Adjust the height value as needed
            title={
                'text': f'Relative Change by {relative_change_type}',
                'x': 0.5,  # Center the title horizontally
                'xanchor': 'center',  # Center the title horizontally
                'y': 0.9  # Adjust the vertical position of the title if needed
            }
        )

        # Add labels inside the bars
        fig.update_traces(
            texttemplate='%{y:.2f}%',
            textposition='inside',
            insidetextfont=dict(color='white'),
            marker_color=my_chart_color  # Set the color of the plot line
        )

        st.plotly_chart(fig, use_container_width=True)

    def create_relative_change_table(self, value_column):
        # Add a radio button for the user to select the period
        period = st.radio('Select period:', ['1 week', '1 month', '3 months', '6 months', '1 year'], horizontal=True)

        # Create a dictionary that maps the period to the corresponding number of days
        period_to_days = {'1 week': 7, '1 month': 30, '3 months': 90, '6 months': 180, '1 year': 365}

        # Calculate the relative changes based on the selected period
        relative_change_period = self.state['df_raw'][value_column].pct_change(period_to_days[period]) * 100

        # Get the actual result from the specific period ago
        actual_result_period_ago = self.state['df_raw'][value_column].shift(period_to_days[period])

        # Get the day of the week from the 'Date' column
        day_of_week = self.state['df_raw']['date'].dt.day_name()

        # Convert 'Date' column to date format (remove time part)
        date_without_time = self.state['df_raw']['date'].dt.date

        date_column = pd.to_datetime(self.state['df_raw']['date'])

        # Calculate the date for the old period
        old_date = date_column - pd.to_timedelta(period_to_days[period], unit='D')

        # Get the day of the week for the old date
        day_of_week_old_date = pd.to_datetime(old_date).dt.day_name()

        # Create a DataFrame with the actual results, the actual result from the specific period ago, the relative changes, and the day of the week
        results_df = pd.DataFrame({
            'Date': date_without_time,
            'Actual Results': self.state['df_raw'][value_column],
            f'Actual Results {period} Ago': actual_result_period_ago,
            'Relative Change (%)': relative_change_period,
            'Day of Week': day_of_week,
            'Day of Week Old Date': day_of_week_old_date
        })

        # Sort the DataFrame in descending order based on 'Date'
        results_df = results_df.sort_values(by='Date', ascending=False)

        # Display the DataFrame
        st.dataframe(results_df, hide_index=True, use_container_width=True)
