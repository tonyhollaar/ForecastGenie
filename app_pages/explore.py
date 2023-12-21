# standard libraries
import streamlit as st
import pandas as pd
import base64
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import pacf, adfuller
import statsmodels.api as sm
from scipy.stats import shapiro

# third-party libraries
from fire_state import form_update, get_state, set_state

# local packages
from style.text import my_title, my_text_header, my_text_paragraph, vertical_spacer, my_subheader
from style.icons import load_icons
from functions import (adjust_brightness, determine_df_frequency, display_summary_statistics,
                       create_summary_df, download_csv_button, )

# load icons
icons = load_icons()


class ExplorePage:
    # define paths to images
    QUICK_SUMMARY_IMAGE_PATH = './images/quick_summary.png'
    INSIGHTS_IMAGE_PATH = './images/futuristic_city_robot.png'
    PATTERNS_BANNER_PATH = './images/patterns_banner.png'
    STATISTICS_IMAGE_PATH = './images/statistical_test.png'

    def __init__(self, state):
        self.state = state
        self.my_bar = st.progress(0, text="Preparing your exploratory data analysis... Please wait!")
        self.tab1, self.tab2, self.tab3, self.tab4, self.tab5 = st.tabs(
            ["overview", "insights", "patterns", "statistical tests", "lag analysis"])

    def render(self):
        self.render_sidebar()
        self.render_tab1()
        self.my_bar.progress(20, text='loading quick insights...')
        self.render_tab2()
        self.my_bar.progress(40, text='loading patterns')
        self.render_tab3()
        self.my_bar.progress(60, text='loading statistical tests')
        self.render_tab4()
        self.my_bar.progress(80, text='loading lag analysis')
        self.render_tab5()
        self.my_bar.progress(100, text='"All systems go!"')
        self.my_bar.empty()

    def render_sidebar(self):
        with st.sidebar:
            my_title(f'{icons["explore_icon"]}', my_background_color="#3b3b3b", gradient_colors="#217CD0,#555555")
            with st.form('ljung-box'):
                my_text_paragraph('White Noise')
                lag1_ljung_box = st.number_input(label='*Enter maximum lag:*', min_value=1,
                                                 value=min(24, len(self.state.df_raw) - 2),
                                                 max_value=len(self.state.df_raw) - 2,
                                                 key='lag1_ljung_box',
                                                 help='the lag parameter in the Ljung-Box test determines the number of time periods over which the autocorrelation of residuals is evaluated to assess the presence of significant autocorrelation in the time series.')
                col1, col2, col3 = st.columns([5, 4, 4])
                with col2:
                    vertical_spacer(1)
                    ljung_box_btn = st.form_submit_button("Submit", type="secondary")

            # Autocorrelation parameters form
            with st.form('autocorrelation'):
                my_text_paragraph('Autocorrelation ')
                col1, col2, col3 = st.columns([4, 1, 4])

                # If dataset is very small (less then 31 rows), then update the key1_explore and key2_explore
                if len(self.state['df_raw']) < 31:
                    set_state("EXPLORE_PAGE", ("lags_acf", int((len(self.state['df_raw']) - 1))))
                    set_state("EXPLORE_PAGE", ("lags_pacf", int((len(self.state['df_raw']) - 1) / 2)))

                # create slider for number of lags for ACF Plot
                self.state['nlags_acf'] = st.slider(label="*Lags ACF*",
                                                    min_value=1,
                                                    max_value=(len(self.state['df_raw']) - 1),
                                                    key='key1_explore')

                col1, col2, col3 = st.columns([4, 1, 4])
                with col1:
                    # create slider for number of lags for PACF Plot
                    self.state['nlags_pacf'] = st.slider(label="*Lags PACF*",
                                                         min_value=1,
                                                         max_value=int((len(self.state['df_raw']) - 2) / 2),
                                                         key='key2_explore')
                with col3:
                    # create dropdown menu for method to calculate the PACF Plot
                    self.state['method_pacf'] = st.selectbox(label="*Method PACF*",
                                                             options=['ols', 'ols-inefficient', 'ols-adjusted', 'yw',
                                                                      'ywa', 'ld',
                                                                      'ywadjusted', 'yw_adjusted', 'ywm', 'ywmle',
                                                                      'yw_mle', 'lda',
                                                                      'ldadjusted', 'ld_adjusted', 'ldb', 'ldbiased',
                                                                      'ld_biased'],
                                                             key='key3_explore')

                # create dropdown menu to select if you want the original series or differenced series that will also impact the ACF & PACF Plots
                # next to having a preview in sidebar with the differenced series plotted if user selects 1st ,2nd or 3rd order difference
                self.state['selection'] = st.selectbox('*Apply Differencing [Optional]:*',
                                                       options=['Original Series', 'First Order Difference',
                                                                'Second Order Difference',
                                                                'Third Order Difference'],
                                                       key='key4_explore')

                col1, col2, col3 = st.columns([5, 4, 4])
                with col2:
                    # create button in sidebar for the ACF and PACF Plot Parameters
                    vertical_spacer(1)
                    self.state['acf_pacf_btn'] = st.form_submit_button("Submit", type="secondary", on_click=form_update,
                                                                       args=('EXPLORE_PAGE',))

    def render_tab1(self):
        with self.tab1:
            with st.expander('', expanded=True):
                col0, col1, col2, col3 = st.columns([18, 90, 8, 1])
                with col2:
                    self.state['my_chart_color'] = st.color_picker(label='Color',
                                                                   value=get_state("COLORS", "chart_patterns"),
                                                                   label_visibility='collapsed',
                                                                   help='Set the **`color`** of the charts and styling elements. It will revert back to the **default** color when switching app_pages.')
                with col1:
                    my_text_header('Quick Summary')
                col1, col2, col3 = st.columns([7, 120, 1])
                with col2:
                    vertical_spacer(1)
                    self.create_flipcard_quick_summary(header_list='',
                                                       paragraph_list_front='',
                                                       paragraph_list_back='',
                                                       font_family='Arial',
                                                       font_size_front='16px',
                                                       font_size_back='16px',
                                                       image_path_front_card=self.QUICK_SUMMARY_IMAGE_PATH,
                                                       my_chart_color='#FFFFFF')

                col1, col2, col3 = st.columns([100, 50, 95])
                with col2:
                    placeholder = st.empty()
                    btn_summary_stats = placeholder.button('Show Details', disabled=False,
                                                           key="summary_statistics_show_btn")
                if btn_summary_stats == True:
                    placeholder.button('Hide Details', disabled=False, key="summary_statistics_hide_btn")
                    summary_stats_df = display_summary_statistics(self.state.df_raw)
                    st.dataframe(summary_stats_df, use_container_width=True)
                    download_csv_button(summary_stats_df, my_file="summary_statistics.csv",
                                        help_message='Download your Summary Statistics Dataframe to .CSV',
                                        my_key="summary_statistics_download_btn")
                vertical_spacer(1)
                self.state['summary_statistics_df'] = create_summary_df(data=self.state.df_raw.iloc[:, 1])

    def render_tab2(self):
        with self.tab2:
            with st.expander('', expanded=True):
                col1, col2, col3 = st.columns([7, 120, 1])
                with col2:
                    self.create_flipcard_quick_insights(1, header_list=[''],
                                                        paragraph_list_front=[''],
                                                        paragraph_list_back=[''],
                                                        font_family='Arial',
                                                        font_size_front='12px',
                                                        font_size_back='12px',
                                                        image_path_front_card=self.INSIGHTS_IMAGE_PATH,
                                                        df=self.state['summary_statistics_df'],
                                                        my_string_column='Label',
                                                        my_chart_color='#FFFFFF')

                # have button available for user and if clicked it expands with the dataframe
                col1, col2, col3 = st.columns([100, 50, 95])
                with col2:
                    placeholder = st.empty()

                    # create button (enabled to click e.g. disabled=false with unique key)
                    btn_insights = placeholder.button('Show Details', disabled=False,
                                                      key="insights_statistics_show_btn")

                    vertical_spacer(1)

                # if button is clicked run below code
                if btn_insights:
                    # display button with text "click me again", with unique key
                    placeholder.button('Hide Details', disabled=False, key="insights_statistics_hide_btn")

                    st.dataframe(self.state['summary_statistics_df'], use_container_width=True, hide_index=True)

                    download_csv_button(self.state['summary_statistics_df'], my_file="insights.csv",
                                        help_message='Download your Insights Dataframe to .CSV',
                                        my_key="insights_download_btn")

    def render_tab3(self):
        with self.tab3:
            with st.expander('', expanded=True):
                my_text_header('Patterns')

                # show all graphs with patterns in streamlit
                self.plot_overview(df=self.state.df_raw,
                                   y=self.state.df_raw.columns[1],
                                   my_chart_color=self.state['my_chart_color'])

                # radio button for user to select frequency of hist: absolute values or relative
                col1, col2, col3 = st.columns([10, 10, 7])
                with col2:
                    # Add radio button for frequency type of histogram
                    frequency_type = st.radio(label="*histogram frequency type:*",
                                              options=("Absolute", "Relative"),
                                              index=1 if get_state("HIST", "histogram_freq_type") == "Relative" else 0,
                                              on_change=self.hist_change_freq,
                                              horizontal=True)

                    vertical_spacer(3)

                st.image(self.PATTERNS_BANNER_PATH)

    def render_tab4(self):
        with self.tab4:
            with st.expander('', expanded=True):
                col1, col2, col3 = st.columns([7, 120, 1])
                with col2:
                    self.create_flipcard_stats(image_path_front_card=self.STATISTICS_IMAGE_PATH,
                                               my_header='Test Results')

            with st.expander('White Noise', expanded=False):
                my_text_header('White Noise')
                my_text_paragraph('Ljung-Box')
                col1, col2, col3 = st.columns([18, 44, 10])
                with col2:
                    vertical_spacer(2)
                    res, result_ljungbox = self.ljung_box_test(df=self.state.df_raw,
                                                               variable_loc=1,
                                                               lag=self.state['lag1_ljung_box'],
                                                               model_type="AutoReg")

                self.ljung_box_plots(df=self.state['df_raw'],
                                     variable_loc=1,
                                     lag=self.state['lag1_ljung_box'],
                                     res=res,
                                     result_ljungbox=result_ljungbox,
                                     my_chart_color=self.state['my_chart_color'])

            with st.expander('Stationarity', expanded=False):
                my_text_header('Stationarity')
                my_text_paragraph('Augmented Dickey Fuller')
                adf_result = self.adf_test(self.state['df_raw'], 1)
                col1, col2, col3 = st.columns([18, 40, 10])
                col2.write(adf_result)
                vertical_spacer(2)

            with st.expander('Normality', expanded=False):
                my_text_header('Normality')
                my_text_paragraph('Shapiro')
                shapiro_result = self.shapiro_test(self.state['df_raw'], 1, alpha=0.05)
                col1, col2, col3 = st.columns([18, 40, 10])
                col2.write(shapiro_result)
                vertical_spacer(2)

    def render_tab5(self):
        with self.tab5:
            with st.expander('Autocorrelation', expanded=True):
                my_text_header('Autocorrelation')
                st.markdown(f'<p style="text-align:center; color: #707070"> {self.state["selection"]} </p>',
                            unsafe_allow_html=True)
                original_fig, df_select_diff = self.df_differencing(self.state.df_raw, self.state["selection"],
                                                                    self.state['my_chart_color'])
                st.plotly_chart(original_fig, use_container_width=True)
                data = df_select_diff
                self.plot_acf(data,
                              nlags=self.state["nlags_acf"],
                              my_chart_color=self.state['my_chart_color'])

                self.plot_pacf(data,
                               my_chart_color=self.state['my_chart_color'],
                               nlags=self.state['nlags_pacf'],
                               method=self.state["method_pacf"])

                self.acf_pacf_info()

    def acf_pacf_info(self):
        background_colors = ["#456689", "#99c9f4"]

        col1, col2, col3 = st.columns([5, 5, 5])
        with col1:
            show_acf_info_btn = st.button(f'about acf plot', use_container_width=True, type='secondary')
        with col2:
            show_pacf_info_btn = st.button(f'about pacf plot', use_container_width=True, type='secondary')
        with col3:
            diff_acf_pacf_info_btn = st.button(f'difference acf/pacf', use_container_width=True, type='secondary')

        if show_acf_info_btn == True:
            vertical_spacer(1)
            my_subheader('Autocorrelation Function (ACF)', my_background_colors=background_colors)
            col1, col2, col3 = st.columns([1, 8, 1])
            with col2:
                st.markdown('''
                            The **Autocorrelation Function (ACF)** plot is a statistical tool used to identify patterns of correlation between observations in a time series dataset. 
                            It is commonly used in time series analysis to determine the extent to which a given observation is related to its previous observations.  
                            \nThe **ACF** plot displays the correlation coefficient between the time series and its own lagged values (i.e., the correlation between the series at time $t$ and the series at times $t_{-1}$, $t_{-2}$, $t_{-3}$, etc.).  
                            The horizontal axis of the plot shows the lag or time difference between observations, while the vertical axis represents the correlation coefficient, ranging from -1 to 1.
                            ''')
            vertical_spacer(1)
            my_subheader('How to interpret a ACF plot', my_background_colors=background_colors)
            col1, col2, col3 = st.columns([1, 8, 1])
            with col2:
                st.markdown('''Interpreting the **ACF** plot involves looking for significant peaks or spikes above the horizontal dashed lines (which represent the confidence interval) to determine if there is any correlation between the current observation and the lagged observations. 
                            If there is a significant peak at a particular lag value, it indicates that there is a strong correlation between the observation and its lagged values up to that point.
                            ''')
            vertical_spacer(1)
            my_subheader('Key Points:', my_background_colors=background_colors)
            col1, col2, col3 = st.columns([1, 8, 1])
            with col2:
                st.markdown('''
                            Some key takeaways when looking at an **ACF** plot include:  
                            1. If there are no significant peaks, then there is no significant correlation between the observations and their lagged values.
                            2. A significant peak at lag $k$ means that the observation at time $t$ is significantly correlated with the observation at time $t_{-k}$.
                            3. A rapid decay of autocorrelation towards zero suggests a stationary time series, while a slowly decaying or persistent non-zero autocorrelations, suggests a non-stationary time series.
                            ''')
            vertical_spacer(1)
        if show_pacf_info_btn == True:
            vertical_spacer(1)
            my_subheader('Partial Autocorrelation Function (PACF)', my_background_colors=background_colors)
            col1, col2, col3 = st.columns([1, 8, 1])
            with col2:
                st.markdown('''
                            The **Partial Autocorrelation Function (PACF)** is a plot of the partial correlation coefficients between a time series and its lags. 
                            The PACF can help us determine the order of an autoregressive (AR) model by identifying the lag beyond which the autocorrelations are effectively zero.

                            The **PACF plot** helps us identify the important lags that are related to a time series. It measures the correlation between a point in the time series and a lagged version of itself while controlling for the effects of all the other lags that come before it.
                            In other words, the PACF plot shows us the strength and direction of the relationship between a point in the time series and a specific lag, independent of the other lags. 
                            A significant partial correlation coefficient at a particular lag suggests that the lag is an important predictor of the time series.

                            If a particular lag has a partial autocorrelation coefficient that falls outside of the **95%** or **99%** confidence interval, it suggests that this lag is a significant predictor of the time series. 
                            The next step would be to consider including that lag in the autoregressive model to improve its predictive accuracy.
                            However, it is important to note that including too many lags in the model can lead to overfitting, which can reduce the model's ability to generalize to new data. 
                            Therefore, it is recommended to use a combination of statistical measures and domain knowledge to select the optimal number of lags to include in the model.

                            On the other hand, if none of the lags have significant partial autocorrelation coefficients, it suggests that the time series is not well explained by an autoregressive model. 
                            In this case, alternative modeling techniques such as **Moving Average (MA)** or **Autoregressive Integrated Moving Average (ARIMA)** may be more appropriate. 
                            Or you could just flip a coin and hope for the best. But I don\'t recommend the latter...
                            ''')
            vertical_spacer(1)

            my_subheader('How to interpret a PACF plot', my_background_colors=background_colors)
            col1, col2, col3 = st.columns([1, 8, 1])
            col1, col2, col3 = st.columns([1, 8, 1])
            with col2:
                st.markdown('''
                            The partial autocorrelation plot (PACF) is a tool used to investigate the relationship between an observation in a time series with its lagged values, while controlling for the effects of intermediate lags. Here's a brief explanation of how to interpret a PACF plot:  

                            - The horizontal axis shows the lag values (i.e., how many time steps back we\'re looking).
                            - The vertical axis shows the correlation coefficient, which ranges from **-1** to **1**. 
                              A value of :green[**1**] indicates a :green[**perfect positive correlation**], while a value of :red[**-1**] indicates a :red[**perfect negative correlation**]. A value of **0** indicates **no correlation**.
                            - Each bar in the plot represents the correlation between the observation and the corresponding lag value. The height of the bar indicates the strength of the correlation. 
                              If the bar extends beyond the dotted line (which represents the 95% confidence interval), the correlation is statistically significant.  
                            ''')
            vertical_spacer(1)
            my_subheader('Key Points:', my_background_colors=background_colors)
            col1, col2, col3 = st.columns([1, 8, 1])
            with col2:
                st.markdown('''                            
                            - **The first lag (lag 0) is always 1**, since an observation is perfectly correlated with itself.
                            - A significant spike at a particular lag indicates that there may be some **useful information** in that lagged value for predicting the current observation. 
                              This can be used to guide the selection of lag values in time series forecasting models.
                            - A sharp drop in the PACF plot after a certain lag suggests that the lags beyond that point **are not useful** for prediction, and can be safely ignored.
                            ''')
            vertical_spacer(1)
            my_subheader('An analogy', my_background_colors=background_colors)
            col1, col2, col3 = st.columns([1, 8, 1])
            with col2:
                st.markdown('''
                            Imagine you are watching a magic show where the magician pulls a rabbit out of a hat. Now, imagine that the magician can do this trick with different sized hats. If you were trying to figure out how the magician does this trick, you might start by looking for clues in the size of the hats.
                            Similarly, the PACF plot is like a magic show where we are trying to figure out the "trick" that is causing our time series data to behave the way it does. 
                            The plot shows us how strong the relationship is between each point in the time series and its past values, while controlling for the effects of all the other past values. 
                            It's like looking at different sized hats to see which one the magician used to pull out the rabbit.

                            If the **PACF** plot shows a strong relationship between a point in the time series and its past values at a certain lag (or hat size), it suggests that this past value is an important predictor of the time series. 
                            On the other hand, if there is no significant relationship between a point and its past values, it suggests that the time series may not be well explained by past values alone, and we may need to look for other "tricks" to understand it.
                            In summary, the **PACF** plot helps us identify important past values of our time series that can help us understand its behavior and make predictions about its future values.
                            ''')
            vertical_spacer(1)

        if diff_acf_pacf_info_btn == True:
            vertical_spacer(1)
            my_subheader('Differences explained between ACF and PACF', my_background_colors=background_colors)
            col1, col2, col3 = st.columns([1, 8, 1])
            with col2:
                st.markdown('''
                            - The **ACF** plot measures the correlation between an observation and its lagged values.
                            - The **PACF** plot measures the correlation between an observation and its lagged values while controlling for the effects of intermediate observations.
                            - The **ACF** plot is useful for identifying the order of a moving average **(MA)** model, while the **PACF** plot is useful for identifying the order of an autoregressive **(AR)** model.  
                            ''')
            vertical_spacer(1)

    def hist_change_freq(self):
        try:
            # update in memory value for radio button / save user choice of histogram freq
            frequency_type = get_state("HIST", "histogram_freq_type")
            index_freq_type = ("Relative" if frequency_type == "Absolute" else "Absolute")
            set_state("HIST", ("histogram_freq_type", index_freq_type))
        except:
            set_state("HIST", ("histogram_freq_type", "Absolute"))

    def plot_overview(self, df, y, my_chart_color):
        """
        Plot an overview of daily, weekly, monthly, quarterly, and yearly patterns
        for a given dataframe and column.
        """
        # initiate variables
        ####################
        num_graph_start = 1
        freq, my_freq_name = determine_df_frequency(df, column_name='date')

        # Determine what pattern graphs should be shown e.g. if weekly do not show daily but rest, if monthly do not show daily/weekly etc.
        # DAILY
        if freq == 'D':
            num_graph_start = 1
        # WEEKLY
        elif freq == 'W':
            num_graph_start = 2
        # MONTHLY
        elif freq == 'M':
            num_graph_start = 3
        # QUARTERLY
        elif freq == 'Q':
            num_graph_start = 4
        # YEARLY
        elif freq == 'Y':
            num_graph_start = 5
        else:
            print('Error could not define the number of graphs to plot')

        y_column_index = df.columns.get_loc(y)
        y_colname = df.columns[y_column_index]

        # Create subplots
        all_subplot_titles = ('Daily Pattern',
                              'Weekly Pattern',
                              'Monthly Pattern',
                              'Quarterly Pattern',
                              'Yearly Pattern',
                              'Boxplot',
                              'Histogram',
                              )

        my_subplot_titles = all_subplot_titles[num_graph_start - 1:]

        # set figure with 6 rows and 1 column with needed subplot titles and set the row_ieght
        fig = make_subplots(rows=len(my_subplot_titles), cols=1,
                            subplot_titles=my_subplot_titles,
                            # the row_heights parameter is set to [0.2] * 5 + [0.5]
                            # which means that the first five rows will have a height of 0.2 each
                            # and the last row will have a height of 0.5 for the histogram with larger height.
                            row_heights=[0.2] * (len(my_subplot_titles) - 2) + [0.7] + [0.5])

        # define intervals for resampling
        df_weekly = df.resample('W', on='date').mean().reset_index()
        df_monthly = df.resample('M', on='date').mean().reset_index()
        df_quarterly = df.resample('Q', on='date').mean().reset_index()
        df_yearly = df.resample('Y', on='date').mean().reset_index()

        if num_graph_start == 1:
            # Daily Pattern
            fig.add_trace(px.line(df, x='date', y=y_colname, title='Daily Pattern').data[0], row=1, col=1)
            fig.add_trace(px.line(df_weekly, x='date', y=y_colname, title='Weekly Pattern').data[0], row=2, col=1)
            fig.add_trace(px.line(df_monthly, x='date', y=y_colname, title='Monthly Pattern').data[0], row=3, col=1)
            fig.add_trace(px.line(df_quarterly, x='date', y=y_colname, title='Quarterly Pattern').data[0], row=4, col=1)
            fig.add_trace(px.line(df_yearly, x='date', y=y_colname, title='Yearly Pattern').data[0], row=5, col=1)
            fig.add_trace(px.box(df, y=y_colname, title='Boxplot of {}'.format(y_colname)).data[0], row=6, col=1)
            self.plot_histogram(df=df, y_colname=y_colname, fig=fig, row=7, col=1, my_chart_color=my_chart_color)

            # Set color for graphs
            fig.update_traces(line_color=my_chart_color, row=1, col=1)
            fig.update_traces(line_color=my_chart_color, row=2, col=1)
            fig.update_traces(line_color=my_chart_color, row=3, col=1)
            fig.update_traces(line_color=my_chart_color, row=4, col=1)
            fig.update_traces(line_color=my_chart_color, row=5, col=1)
            fig.update_traces(marker_color=my_chart_color, row=6, col=1)
            fig.update_traces(marker_color=my_chart_color, row=7, col=1)

        if num_graph_start == 2:
            # Weekly Pattern
            fig.add_trace(px.line(df_weekly, x='date', y=y_colname, title='Weekly Pattern').data[0], row=1, col=1)
            fig.add_trace(px.line(df_monthly, x='date', y=y_colname, title='Monthly Pattern').data[0], row=2, col=1)
            fig.add_trace(px.line(df_quarterly, x='date', y=y_colname, title='Quarterly Pattern').data[0], row=3, col=1)
            fig.add_trace(px.line(df_yearly, x='date', y=y_colname, title='Yearly Pattern').data[0], row=4, col=1)
            fig.add_trace(px.box(df, y=y_colname, title='Boxplot of {}'.format(y_colname)).data[0], row=5, col=1)
            self.plot_histogram(df=df, y_colname=y_colname, fig=fig, row=6, col=1, my_chart_color=my_chart_color)

            # set color for graphs
            fig.update_traces(line_color=my_chart_color, row=1, col=1)
            fig.update_traces(line_color=my_chart_color, row=2, col=1)
            fig.update_traces(line_color=my_chart_color, row=3, col=1)
            fig.update_traces(line_color=my_chart_color, row=4, col=1)
            fig.update_traces(marker_color=my_chart_color, row=5, col=1)
            fig.update_traces(marker_color=my_chart_color, row=6, col=1)

        if num_graph_start == 3:
            # Monthly Pattern
            fig.add_trace(px.line(df_monthly, x='date', y=y_colname, title='Monthly Pattern').data[0], row=1, col=1)
            fig.add_trace(px.line(df_monthly, x='date', y=y_colname, title='Quarterly Pattern').data[0], row=2, col=1)
            fig.add_trace(px.line(df_monthly, x='date', y=y_colname, title='Yearly Pattern').data[0], row=3, col=1)
            fig.add_trace(px.box(df, y=y_colname, title='Boxplot of {}'.format(y_colname)).data[0], row=4, col=1)
            self.plot_histogram(df=df, y_colname=y_colname, fig=fig, row=5, col=1, my_chart_color=my_chart_color)

            # set color for graphs
            fig.update_traces(line_color=my_chart_color, row=1, col=1)
            fig.update_traces(line_color=my_chart_color, row=2, col=1)
            fig.update_traces(line_color=my_chart_color, row=3, col=1)
            fig.update_traces(marker_color=my_chart_color, row=4, col=1)
            fig.update_traces(marker_color=my_chart_color, row=5, col=1)

        if num_graph_start == 4:
            # Quarterly Pattern
            fig.add_trace(px.line(df_quarterly, x='date', y=y_colname, title='Quarterly Pattern').data[0], row=1, col=1)
            fig.add_trace(px.line(df_yearly, x='date', y=y_colname, title='Yearly Pattern').data[0], row=2, col=1)
            fig.add_trace(px.box(df, y=y_colname, title='Boxplot of {}'.format(y_colname)).data[0], row=3, col=1)
            self.plot_histogram(df=df, y_colname=y_colname, fig=fig, row=4, col=1, my_chart_color=my_chart_color)

            # set color for graphs
            fig.update_traces(line_color=my_chart_color, row=1, col=1)
            fig.update_traces(line_color=my_chart_color, row=2, col=1)
            fig.update_traces(marker_color=my_chart_color, row=3, col=1)
            fig.update_traces(marker_color=my_chart_color, row=4, col=1)

        if num_graph_start == 5:
            # Yearly Pattern
            fig.add_trace(px.line(df_yearly, x='date', y=y_colname, title='Yearly Pattern').data[0], row=1, col=1)
            fig.add_trace(px.box(df, y=y_colname, title='Boxplot of {}'.format(y_colname)).data[0], row=2, col=1)
            self.plot_histogram(df=df, y_colname=y_colname, fig=fig, row=3, col=1, my_chart_color=my_chart_color)

            # set color for graphs
            fig.update_traces(line_color=my_chart_color, row=1, col=1)
            fig.update_traces(marker_color=my_chart_color, row=2, col=1)
            fig.update_traces(marker_color=my_chart_color, row=3, col=1)

        # define height of graph
        my_height = len(my_subplot_titles) * 266

        # set height dynamically e.g. 6 graphs maximum but less if frequency is not daily data and x 266 (height) per graph
        fig.update_layout(height=my_height)

        # Display in Streamlit app
        st.plotly_chart(fig, use_container_width=True)

    def plot_histogram(self, df, y_colname, fig, row, col, my_chart_color):
        """
        Plot a histogram with normal distribution curve for a given dataframe and column.
        Adds the histogram and normal distribution curve to the specified plotly figure.
        The figure is updated in place.

        Parameters:
        - df (pandas.DataFrame): The dataframe containing the data.
        - y_colname (str): The column name for the histogram.
        - fig (plotly.graph_objects.Figure): The plotly figure to update.
        - row (int): The row position of the subplot to add the histogram.
        - col (int): The column position of the subplot to add the histogram.
        """
        # Histogram's Normal Distribution Curve Calculation & Color
        # Scott's Rule: calculate number of bins based on st. dev.
        bin_width = 3.5 * np.std(df[y_colname]) / (len(df[y_colname]) ** (1 / 3))
        num_bins = math.ceil((np.max(df[y_colname]) - np.min(df[y_colname])) / bin_width)

        mean = df[y_colname].mean()
        std = df[y_colname].std()

        x_vals = np.linspace(df[y_colname].min(), df[y_colname].max(), 100)  # Generate x-values
        y_vals = (np.exp(-(x_vals - mean) ** 2 / (2 * std ** 2)) / (np.sqrt(2 * np.pi) * std) * len(
            df[y_colname]) * bin_width)  # Calculate y-values

        # Define adjusted brightness color of normal distribution trace
        adjusted_color = adjust_brightness(my_chart_color, 2)

        # Check for NaN values in the column
        if df[y_colname].isnull().any():
            vertical_spacer(2)
            st.info(
                '**ForecastGenie message**: replaced your missing values with zero in a copy of original dataframe, in order to plot the Histogram. No worries I kept the original dataframe in one piece.')
            # Handle missing values in copy of dataframe -> do not want to change original df
            df = df.copy(deep=True)
            df[y_colname].fillna(0, inplace=True)  # Replace NaN values with zero
        else:
            pass

        # Plot histogram based on frequency type
        freq_type = get_state("HIST", "histogram_freq_type")

        if freq_type == "Absolute":
            histogram_trace = px.histogram(df, x=y_colname, title='Histogram', nbins=num_bins)  # Define Histogram Trace
            fig.add_trace(histogram_trace.data[0], row=row, col=col)  # Histogram
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line_color=adjusted_color, showlegend=False),
                          row=row, col=col)  # Normal Dist Curve
            fig.add_vline(x=mean, line_dash="dash", line_color=adjusted_color, row=row, col=col)  # Mean line

        elif freq_type == "Relative":
            hist, bins = np.histogram(df[y_colname], bins=num_bins)
            rel_freq = hist / np.sum(hist)
            fig.add_trace(go.Bar(x=bins, y=rel_freq, name='Relative Frequency', showlegend=False), row=row, col=col)

        else:
            st.error(
                'FORECASTGENIE ERROR: Could not execute the plot of the Histogram, please contact your administrator.')

    def ljung_box_test(self, df, variable_loc, lag, model_type="AutoReg"):
        """
        Perform the Ljung-Box test for white noise on a time series using the AutoReg model.

        Parameters
        ----------
        df : pandas DataFrame
            The time series data.
        variable_loc : int or tuple or list or pandas.Series
            The location of the variable to test for white noise, which can be a single integer (if the variable
            is located in a single-column DataFrame), a tuple or list of integers (if the variable is located
            in a multi-column DataFrame), or a pandas Series.
        lag : int
            The lag value to consider for the Ljung-Box test.
        model_type : str, optional
            The type of model to use for the Ljung-Box test (default: "AutoReg").

        Returns
        -------
        result : str
            A string containing the results of the Ljung-Box test.
        """
        # =============================================================================
        #     try:
        # =============================================================================
        # Select the variable to test for white noise
        if isinstance(variable_loc, int):
            variable = df.iloc[:, variable_loc]
        elif isinstance(variable_loc, (tuple, list)):
            variable = df.iloc[:, variable_loc]
        elif isinstance(variable_loc, pd.Series):
            variable = variable_loc
        else:
            raise ValueError("The 'variable_loc' argument must be an integer, tuple, list, or pandas Series.")

        # =============================================================================
        # Replace NaN with 0 values ELSE WHITENOISE TEST WILL THROW ERROR
        # =============================================================================
        if variable.isnull().any():
            # make a deepcopy
            variable = variable.copy(deep=True)
            # replace with 0
            variable = variable.fillna(0)
            # show user message
            st.info(
                '**ForecastGenie message**: replaced your missing values with zero in a copy of original dataframe, in order to perform the ljung box test and can introduce bias into the results.')

        # Drop missing values
        variable = variable.dropna()

        if model_type == "AutoReg":
            # Fit AutoReg model to the data
            model = sm.tsa.AutoReg(variable, lags=lag, trend='c', old_names=False)
            res = model.fit()

            # Perform Ljung-Box test on residuals for all lags including up to lag integer
            result_ljungbox = sm.stats.acorr_ljungbox(res.resid, lags=lag, return_df=True)
        else:
            raise ValueError("Invalid model type selected.")

        # test_statistic = result_ljungbox.iloc[0]['lb_stat']
        # p_value = result_ljungbox.iloc[0]['lb_pvalue']
        p_values = result_ljungbox['lb_pvalue']

        # white_noise = "True" if p_value > 0.05 else "False"

        alpha = 0.05  # significance level
        lag_with_autocorr = None

        # Check p-value for each lag
        for i, p_value in enumerate(p_values):
            lag_number = i + 1
            if p_value <= alpha:
                lag_with_autocorr = lag_number
                break

        # if p value is less than or equal to significance level reject zero hypothesis
        if lag_with_autocorr is not None:
            st.markdown(
                f'❌ $H_0$: the residuals have **:green[no autocorrelation]** for one or more lags up to a maximum lag of **{lag}**.')  # h0
            st.markdown(
                f'✅ $H_1$: the residuals **:red[have autocorrelation]** for one or more lags up to a maximum lag of **{lag}**.')  # h1
        else:
            st.markdown(
                f'✅ $H_0$: the residuals have **:green[no autocorrelation]** for one or more lags up to a maximum lag of **{lag}**.')  # h0
            st.markdown(
                f'❌ $H_1$: the residuals **:red[have autocorrelation]** for one or more lags up to a maximum lag of **{lag}**.')  # h1

        alpha = 0.05  # Significance level

        if lag_with_autocorr is None:
            st.markdown(
                f"**conclusion:** the null hypothesis **cannot** be rejected for all lags up to maximum lag of **{lag}**. the residuals show **no significant autocorrelation.**")
            vertical_spacer(2)
        else:
            st.markdown(
                f"**conclusion:** the null hypothesis can be **rejected** for at least one lag up to a maximum lag of **{lag}** with a p-value of **`{p_value:.2e}`**, which is smaller than the significance level of **`{alpha:}`**. this suggests presence of serial dependence in the time series.")

        # show dataframe with test results
        st.dataframe(result_ljungbox, use_container_width=True)

        return res, result_ljungbox

    def adf_test(self, df, variable_loc, max_diffs=3):
        """
        Perform the Augmented Dickey-Fuller (ADF) test for stationarity on a time series.

        Parameters
        ----------
        df : pandas DataFrame
            The time series data.
        variable_loc : int or tuple or list or pandas.Series
            The location of the variable to test for stationarity, which can be a single integer (if the variable
            is located in a single-column DataFrame), a tuple or list of integers (if the variable is located
            in a multi-column DataFrame), or a pandas Series.
        max_diffs : int, optional
            The maximum number of times to difference the time series if it is non-stationary.
            Defaults to 3.

        Returns
        -------
        result : str
            A string containing the results of the ADF test.
        """

        # Select the variable to test for stationarity
        if isinstance(variable_loc, int):
            variable = df.iloc[:, variable_loc]
        elif isinstance(variable_loc, (tuple, list)):
            variable = df.iloc[:, variable_loc]
        elif isinstance(variable_loc, pd.Series):
            variable = variable_loc
        else:
            raise ValueError("The 'variable_loc' argument must be an integer, tuple, list, or pandas Series.")

        # Drop missing values
        variable = variable.dropna()
        col1, col2, col3 = st.columns([18, 40, 10])
        # Check if the time series is stationary
        p_value = adfuller(variable, autolag='AIC')[1]
        # Check if the p-value is less than or equal to 0.05
        if p_value <= 0.05:
            # If the p-value is less than 0.001, use scientific notation with 2 decimal places
            if p_value < 1e-3:
                p_value_str = f"{p_value:.2e}"
            # Otherwise, use regular decimal notation with 3 decimal places
            else:
                p_value_str = f"{p_value:.3f}"
        if p_value <= 0.05:
            with col2:
                vertical_spacer(1)  # newline vertical space
                h0 = st.markdown(
                    r'❌$H_0$: the time series has a unit root, meaning it is **non-stationary**. It has some time dependent structure.')
                vertical_spacer(1)
                h1 = st.markdown(
                    r'✅$H_1$: the time series does **not** have a unit root, meaning it is **stationary**. it does not have time-dependent structure.')
                vertical_spacer(1)
                result = f'**conclusion:**\
                          the null hypothesis can be :red[**rejected**] with a p-value of **`{p_value:.5f}`**, which is smaller than the 95% confidence interval (p-value = `0.05`).'
        else:
            # If the time series remains non-stationary after max_diffs differencings, return the non-stationary result
            for i in range(1, max_diffs + 1):
                # Difference the time series
                diff_variable = variable.diff(i).dropna()
                # Check if the differenced time series is stationary
                p_value = adfuller(diff_variable, autolag='AIC')[1]
                # Check if the p-value is less than or equal to 0.05
                if p_value <= 0.05:
                    # If the p-value is less than 0.001, use scientific notation with 2 decimal places
                    if p_value < 1e-3:
                        p_value_str = f"{p_value:.2e}"
                    # Otherwise, use regular decimal notation with 3 decimal places
                    else:
                        p_value_str = f"{p_value:.3f}"

                    # If the differenced time series is stationary, return the result
                    with col2:
                        vertical_spacer(1)
                        h0 = st.markdown(
                            f'❌$H_0$: the time series has a unit root, meaning it is :red[**non-stationary**]. it has some time dependent structure.')
                        vertical_spacer(1)
                        h1 = st.markdown(
                            f'✅$H_1$: the time series does **not** have a unit root, meaning it is :green[**stationary**]. it does not have time-dependent structure.')
                        vertical_spacer(1)
                        result = f'**conclusion:**\
                                  After *differencing* the time series **`{i}`** time(s), the null hypothesis can be :red[**rejected**] with a p-value of **`{p_value_str}`**, which is smaller than `0.05`.'
                    break
                else:
                    # If the time series remains non-stationary after max_diffs differencings, return the non-stationary result
                    result = f'the {variable.name} time series is non-stationary even after **differencing** up to **{max_diffs}** times. last ADF p-value: {p_value:.5f}'

                    with col2:
                        max_diffs = st.slider(':red[[Optional]] *Adjust maximum number of differencing:*', min_value=0,
                                              max_value=10, value=3, step=1,
                                              help='Adjust maximum number of differencing if Augmented Dickey-Fuller Test did not become stationary after differencing the data e.g. 3 times (default value)')
        return result

    def ljung_box_plots(self, df, variable_loc, res, lag, result_ljungbox, my_chart_color):
        adjusted_color = adjust_brightness(my_chart_color, 2)
        st.markdown('---')
        # 1st GRAPH - Residual Plot
        ###################################
        # Plot the residuals
        column_name = df.iloc[:, variable_loc].name
        my_text_paragraph(f'Residuals of {column_name}')

        # Create the line plot with specified x, y, and labels
        fig = px.line(x=df['date'][lag:],
                      y=res.resid,
                      labels={"x": "Date", "y": "Residuals"})

        # Update the line color and name
        fig.update_traces(line_color=my_chart_color,
                          name="Residuals")

        # Add a dummy trace to create the legend for the first graph
        fig.add_trace(go.Scatter(x=[None], y=[None], name="Residuals", line=dict(color="#4185c4")))

        # Position the legend at the top right inside the graph
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='top'),
                          margin=dict(t=10), yaxis=dict(automargin=True))

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('---')

        # 2nd GRAPH -  p-values of the lags
        ###################################

        # Compute the p-values of the lags
        p_values = result_ljungbox['lb_pvalue']

        # Plot the p-values
        my_text_paragraph('P-values for Ljung-Box Statistic')

        fig_pvalues = go.Figure(data=go.Scatter(x=df.index[0:lag],
                                                y=p_values.iloc[0:lag],
                                                mode="markers",
                                                name='p-value',
                                                marker=dict(symbol="circle-open", color=my_chart_color)))

        # Add a blue dotted line for the significance level
        fig_pvalues.add_trace(go.Scatter(x=[df.index[0], df.index[lag - 1]],
                                         y=[0.05, 0.05],
                                         mode="lines",
                                         line=dict(color=adjusted_color, dash="dot"),
                                         name="α = 0.05"))

        # Position the legend at the top right inside the graph and update y-axis range and tick settings
        fig_pvalues.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='top'), xaxis_title="Lag",
                                  yaxis_title="P-value", yaxis=dict(range=[-0.1, 1], dtick=0.1), margin=dict(t=10),
                                  showlegend=True)

        # Show plotly chart
        st.plotly_chart(fig_pvalues, use_container_width=True)

    def shapiro_test(self, df, variable_loc, alpha=0.05):
        """
        Perform the Shapiro-Wilk test for normality on a dataset.

        Parameters
        ----------
        data : array-like
            The data to test for normality.
        alpha : float, optional
            The significance level for the test. Defaults to 0.05.

        Returns
        -------
        result : str
            A string containing the results of the Shapiro-Wilk test.
        """
        # Select the variable to test for stationarity
        if isinstance(variable_loc, int):
            variable = df.iloc[:, variable_loc]
        elif isinstance(variable_loc, (tuple, list)):
            variable = df.iloc[:, variable_loc]
        elif isinstance(variable_loc, pd.Series):
            variable = variable_loc
        else:
            raise ValueError("The 'variable_loc' argument must be an integer, tuple, list, or pandas Series.")
        # =============================================================================
        #         # Drop missing values
        #         variable = variable.dropna()
        # =============================================================================

        # Perform the Shapiro-Wilk test
        statistic, p_value = shapiro(variable)

        # Check if the p-value is less than or equal to the significance level
        if p_value <= alpha:
            # If the p-value is less than 0.001, use scientific notation with 2 decimal places
            if p_value < 1e-3:
                p_value_str = f"{p_value:.2e}"
            # Otherwise, use regular decimal notation with 3 decimal places
            else:
                p_value_str = f"{p_value:.3f}"

            # H0 and H1 hypotheses
            h0 = r"❌$H_0$: the data is normally distributed."
            h1 = r"✅$H_1$: the data is **not** normally distributed."

            # Conclusion when H0 is rejected
            conclusion = f"**conclusion:** the null hypothesis can be **:red[rejected]** with a p-value of **`{p_value_str}`**, which is smaller than or equal to the significance level of **`{alpha}`**. therefore, the data is **not** normally distributed."
        else:
            # H0 and H1 hypotheses
            h0 = r"✅$H_0$: The data is normally distributed."
            h1 = r"❌$H_1$: The data is **not** normally distributed."

            # Conclusion when H0 is not rejected
            conclusion = f"**conclusion:** the null hypothesis cannot be rejected with a p-value of **`{p_value:.5f}`**, which is greater than the significance level of **`{alpha}`**. therefore, the data is normally distributed."

        # Combine H0, H1, and conclusion
        result = f"{h0}\n\n{h1}\n\n{conclusion}"

        return result

    def df_differencing(self, df, selection, my_chart_color):
        """
        Perform differencing on a time series DataFrame up to third order.

        Parameters:
        -----------
        df: pandas.DataFrame
            The DataFrame containing the time series.
        selection: str
            The type of differencing to perform. Must be one of ['Original Series', 'First Order Difference',
            'Second Order Difference', 'Third Order Difference'].

        Returns:
        --------
        fig: plotly.graph_objs._figure.Figure
            A Plotly figure object showing the selected type of differencing.
        df_select_diff: pandas.Series
            The resulting differenced series based on the selected type of differencing.
        """
        # Calculate the first three differences of the data

        df_diff1 = df.iloc[:, 1].diff()
        df_diff2 = df_diff1.diff()
        df_diff3 = df_diff2.diff()

        # Replace any NaN values with 0
        df_diff1.fillna(0, inplace=True)
        df_diff2.fillna(0, inplace=True)
        df_diff3.fillna(0, inplace=True)

        if selection == 'Original Series':
            fig = px.line(df, x='date', y=df.columns[1], title='Original Series',
                          color_discrete_sequence=[my_chart_color])
            df_select_diff = df.iloc[:, 1]
            fig.update_layout(
                title_x=0.5,
                title_font=dict(size=14, family="Arial"),
                yaxis_title=''
            )
        elif selection == 'First Order Difference':
            fig = px.line(pd.concat([df.iloc[:, 0], df_diff1], axis=1), x='date', y=df_diff1.name,
                          color_discrete_sequence=['#87CEEB'])
            df_select_diff = df_diff1
        elif selection == 'Second Order Difference':
            fig = px.line(pd.concat([df.iloc[:, 0], df_diff2], axis=1), x='date', y=df_diff2.name,
                          color_discrete_sequence=['#1E90FF'])
            df_select_diff = df_diff2
        elif selection == 'Third Order Difference':
            fig = px.line(pd.concat([df.iloc[:, 0], df_diff3], axis=1), x='date', y=df_diff3.name,
                          color_discrete_sequence=['#000080'])
            df_select_diff = df_diff3
        else:
            raise ValueError("Invalid selection. Must be one of ['Original Series', 'First Order Difference', "
                             "'Second Order Difference', 'Third Order Difference']")

        fig.update_layout(title=' ',
                          title_x=0.5,
                          title_font=dict(size=14, family="Arial"),
                          yaxis=dict(title=df.columns[1], showticklabels=False, fixedrange=True),
                          margin=dict(l=0, r=20, t=0, b=0),
                          height=200
                          )
        return fig, df_select_diff

    def plot_acf(self, data, nlags, my_chart_color):
        '''
        Plots the autocorrelation function (ACF) for a given time series data.

        Parameters:
        -----------
        data : pandas.DataFrame
            The time series data to plot the ACF for.
        nlags : int
            The number of lags to include in the ACF plot.

        Returns:
        --------
        None.

        Notes:
        ------
        This function drops any rows from the input data that contain NaN values before calculating the ACF.
        The ACF plot includes shaded regions representing the 95% and 99% confidence intervals.
        '''
        # Define adjusted brightness color of normal distribution trace
        adjusted_color = adjust_brightness(my_chart_color, 2)

        if data.isna().sum().sum() > 0:
            st.warning('''**Warning** ⚠️:              
                     Data contains **NaN** values. **NaN** values were dropped in copy of dataframe to be able to plot below ACF. ''')
        st.markdown('<p style="text-align:center; color: #707070">Autocorrelation (ACF)</p>', unsafe_allow_html=True)
        # Drop NaN values if any
        data = data.dropna(axis=0)
        data = data.to_numpy()
        # Calculate ACF
        acf_vals = self.calc_acf(data, nlags)
        # Create trace for ACF plot
        traces = []
        for i in range(nlags + 1):
            trace = go.Scatter(x=[i, i], y=[0, acf_vals[i]],
                               mode='lines+markers', name='Lag {}'.format(i),
                               line=dict(color='grey', width=1))
            # Color lines based on confidence intervals
            conf95 = 1.96 / np.sqrt(len(data))
            conf99 = 2.58 / np.sqrt(len(data))
            if abs(acf_vals[i]) > conf99:
                trace.line.color = my_chart_color  # 'darkred'
                trace.name += ' (>|99%|)'
            elif abs(acf_vals[i]) > conf95:
                trace.line.color = adjusted_color  # 'lightcoral'
                trace.name += ' (>|95%|)'
            traces.append(trace)
        # define the background shape and color for the 95% confidence band
        conf_interval_95_background = go.layout.Shape(
            type='rect',
            xref='x',
            yref='y',
            x0=0.5,
            # lag0 is y with itself so confidence interval starts from lag1 and I want to show a little over lag1 visually so 0.5
            y0=-conf95,
            x1=nlags + 1,
            y1=conf95,
            fillcolor='rgba(68, 114, 196, 0.3)',
            line=dict(width=0),
            opacity=0.5
        )
        # define the background shape and color for the 99% confidence band
        conf_interval_99_background = go.layout.Shape(
            type='rect',
            xref='x',
            yref='y',
            x0=0.5,
            # lag0 is y with itself so confidence interval starts from lag1 and I want to show a little over lag1 visually so 0.5
            y0=-conf99,
            x1=nlags + 1,
            y1=conf99,
            fillcolor='rgba(68, 114, 196, 0.3)',
            line=dict(width=0),
            opacity=0.4
        )
        # Set layout of ACF plot
        layout = go.Layout(
            title='',
            xaxis=dict(title='Lag'),
            yaxis=dict(title='Autocorrelation'),
            margin=dict(l=50, r=50, t=0, b=50),
            shapes=[
                {'type': 'line', 'x0': -1, 'y0': conf95,
                 'x1': nlags + 1, 'y1': conf95,
                 'line': {'color': 'gray', 'dash': 'dash', 'width': 1},
                 'name': '95% Confidence Interval'},
                {'type': 'line', 'x0': -1, 'y0': -conf95,
                 'x1': nlags + 1, 'y1': -conf95,
                 'line': {'color': 'gray', 'dash': 'dash', 'width': 1}},
                {'type': 'line', 'x0': -1, 'y0': conf99,
                 'x1': nlags + 1, 'y1': conf99,
                 'line': {'color': 'gray', 'dash': 'dot', 'width': 1},
                 'name': '99% Confidence Interval'},
                {'type': 'line', 'x0': -1, 'y0': -conf99,
                 'x1': nlags + 1, 'y1': -conf99,
                 'line': {'color': 'gray', 'dash': 'dot', 'width': 1}},
                conf_interval_95_background,
                conf_interval_99_background,
            ]
        )
        # Define Figure
        fig = go.Figure(data=traces, layout=layout)
        # Add Legend
        fig.update_layout(legend=dict(title='Lag (conf. interval)'))
        # Plot ACF with Streamlit Plotly
        st.plotly_chart(fig, use_container_width=True)

    def calc_pacf(self, data, nlags, method):
        return pacf(data, nlags=nlags, method=method)

    # Define function to plot PACF
    def plot_pacf(self, data, nlags, method, my_chart_color):
        '''
        Plots the partial autocorrelation function (PACF) for a given time series data.

        Parameters:
        -----------
        data : pandas.DataFrame
            The time series data to plot the PACF for.
        nlags : int
            The number of lags to include in the PACF plot.
        method : str
            The method to use for calculating the PACF. Can be one of "yw" (default), "ols", "ywunbiased", "ywmle", or "ld".

        Returns:
        --------
        None.

        Notes:
        ------
        This function drops any rows from the input data that contain NaN values before calculating the PACF.
        The PACF plot includes shaded regions representing the 95% and 99% confidence intervals.
        '''
        # Define adjusted brightness color of normal distribution trace
        adjusted_color = adjust_brightness(my_chart_color, 2)

        if data.isna().sum().sum() > 0:
            st.warning('''**Warning** ⚠️:              
                     Data contains **NaN** values. **NaN** values were dropped in copy of dataframe to be able to plot below PACF. ''')
        st.markdown('<p style="text-align:center; color: #707070">Partial Autocorrelation (PACF)</p>',
                    unsafe_allow_html=True)
        # Drop NaN values if any
        data = data.dropna(axis=0)
        data = data.to_numpy()
        # Calculate PACF
        pacf_vals = self.calc_pacf(data, nlags, method)
        # Create trace for PACF plot
        traces = []
        for i in range(nlags + 1):
            trace = go.Scatter(x=[i, i], y=[0, pacf_vals[i]],
                               mode='lines+markers', name='Lag {}'.format(i),
                               line=dict(color='grey', width=1))
            # Color lines based on confidence intervals
            conf95 = 1.96 / np.sqrt(len(data))
            conf99 = 2.58 / np.sqrt(len(data))
            # define the background shape and color for the 95% confidence band
            conf_interval_95_background = go.layout.Shape(
                type='rect',
                xref='x',
                yref='y',
                x0=0.5,
                # lag0 is y with itself so confidence interval starts from lag1 and I want to show a little over lag1 visually so 0.5
                y0=-conf95,
                x1=nlags + 1,
                y1=conf95,
                fillcolor='rgba(68, 114, 196, 0.3)',
                line=dict(width=0),
                opacity=0.5
            )
            # define the background shape and color for the 99% confidence band
            conf_interval_99_background = go.layout.Shape(
                type='rect',
                xref='x',
                yref='y',
                x0=0.5,
                # lag0 is y with itself so confidence interval starts from lag1 and I want to show a little over lag1 visually so 0.5
                y0=-conf99,
                x1=nlags + 1,
                y1=conf99,
                fillcolor='rgba(68, 114, 196, 0.3)',
                line=dict(width=0),
                opacity=0.4
            )
            # if absolute value of lag is larger than confidence band 99% then color 'darkred'
            if abs(pacf_vals[i]) > conf99:
                trace.line.color = my_chart_color  # 'darkred'
                trace.name += ' (>|99%|)'
            # else if absolute value of lag is larger than confidence band 95% then color 'lightcoral'
            elif abs(pacf_vals[i]) > conf95:
                trace.line.color = adjusted_color  # 'lightcoral'
                trace.name += ' (>|95%|)'
            traces.append(trace)
        # Set layout of PACF plot
        layout = go.Layout(title='',
                           xaxis=dict(title='Lag'),
                           yaxis=dict(title='Partial Autocorrelation'),
                           margin=dict(l=50, r=50, t=0, b=50),
                           shapes=[{'type': 'line', 'x0': -1, 'y0': conf95,
                                    'x1': nlags + 1, 'y1': conf95,
                                    'line': {'color': 'gray', 'dash': 'dash', 'width': 1},
                                    'name': '95% Confidence Interval'},
                                   {'type': 'line', 'x0': -1, 'y0': -conf95,
                                    'x1': nlags + 1, 'y1': -conf95,
                                    'line': {'color': 'gray', 'dash': 'dash', 'width': 1}},
                                   {'type': 'line', 'x0': -1, 'y0': conf99,
                                    'x1': nlags + 1, 'y1': conf99,
                                    'line': {'color': 'gray', 'dash': 'dot', 'width': 1},
                                    'name': '99% Confidence Interval'},
                                   {'type': 'line', 'x0': -1, 'y0': -conf99,
                                    'x1': nlags + 1, 'y1': -conf99,
                                    'line': {'color': 'gray', 'dash': 'dot', 'width': 1}},
                                   conf_interval_95_background,
                                   conf_interval_99_background],
                           showlegend=True,
                           )
        # Create figure with PACF plot
        fig = go.Figure(data=traces, layout=layout)
        # add legend
        fig.update_layout(
            legend=dict(title='Lag (conf. interval)'))
        st.plotly_chart(fig, use_container_width=True)

    def calc_acf(self, data, nlags):
        '''
        Calculates the autocorrelation function (ACF) for a given time series data.

        Parameters:
        -----------
        data : numpy.ndarray
            The time series data to calculate the ACF for.
        nlags : int
            The number of lags to include in the ACF calculation.

        Returns:
        --------
        acf_vals : numpy.ndarray
            The ACF values for the specified number of lags.
        '''
        # Calculate the mean of the input data
        mean = np.mean(data)
        # Calculate the autocovariance for each lag
        acovf = np.zeros(nlags + 1)
        for k in range(nlags + 1):
            sum = 0.0
            for i in range(k, len(data)):
                sum += (data[i] - mean) * (data[i - k] - mean)
            acovf[k] = sum / (len(data) - k)
        # Calculate the ACF by normalizing the autocovariance with the variance
        acf_vals = np.zeros(nlags + 1)
        var = np.var(data)
        acf_vals[0] = 1.0
        for k in range(1, nlags + 1):
            acf_vals[k] = acovf[k] / var
        return acf_vals

    def create_flipcard_quick_insights(self, num_cards, header_list, paragraph_list_front, paragraph_list_back,
                                       font_family,
                                       font_size_front, font_size_back, image_path_front_card=None, df=None,
                                       my_string_column='Label', **kwargs):
        col1, col2, col3 = st.columns([20, 40, 20])
        with col2:
            my_text_header('Quick Insights')
            vertical_spacer(1)

        # Filter out NaN and '-' values from 'Label' column
        label_values = df[my_string_column].dropna().apply(lambda x: x.strip()).replace('-', '').tolist()
        # Filter out any remaining '-' values from 'Label' column
        label_values = [value for value in label_values if value != '']
        # Create an HTML unordered list with each non-NaN and non-'-' value as a list item
        html_list = "<div class='my-list'>"
        for i, value in enumerate(label_values):
            html_list += f"<li><span class='my-number'>{i + 1}</span>{value}</li>"
        html_list += "</div>"

        # open the image for the front of the card
        with open(image_path_front_card, 'rb') as file:
            contents = file.read()
            data_url = base64.b64encode(contents).decode("utf-8")

        # create empty list that will keep the html code needed for each card with header+text
        card_html = []

        # iterate over cards specified by user and join the headers and text of the lists
        for i in range(num_cards):
            card_html.append(f"""<div class="flashcard">
                                    <div class='front'>
                                        <img src="data:image/png;base64,{data_url}"style="width: 100%; height: 100%; object-fit: cover; border-radius: 10px;">
                                        <h1 style='text-align:center;color:white; margin-bottom: 10px;padding: 35px;'>{header_list[i]}</h1>
                                        <p style='text-align:center; font-family: {font_family}; font-size: {font_size_front};'>{paragraph_list_front[i]}</p>
                                    </div>
                                    <div class="back">
                                        <h2>{header_list[i]}</h2>
                                        {html_list}
                                        <p style='text-align:center; font-family: {font_family}; font-size: {font_size_back};'>{paragraph_list_back[i]}</p>
                                    </div>
                                </div>
                                """)

        # join all the html code for each card and join it into single html code with carousel wrapper
        carousel_html = "<div class='carousel'>" + "".join(card_html) + "</div>"

        # Display the carousel in streamlit
        st.markdown(carousel_html, unsafe_allow_html=True)

        # Create the CSS styling for the carousel
        st.markdown(
            f"""
            <style>
            /* remove bullet points from list as I have custom css styled numbered list */
            .my-list li {{
                list-style-type: none; /* Add this line */
                margin: 10px 10px 10px 10px;
                padding-left: 30px;
                position: relative;
            }}            
            /* back of card styling */
            .my-list {{
                font-size: 16px;
                color: black; /* Add your desired font color here */
                line-height: 1.4;
                margin-bottom: 10px;
                margin-left: 0px;
                margin-right: 0px;
                margin-bottom: 0px;
                padding: 0px;
                text-align: left;
            }}
            .my-list li {{
                margin: 10px 10px 10px 10px;
                padding-left: 50px;
                position: relative;
            }}
            .my-number {{
                font-weight: lighter;
                color: white;
                background-color: #48555e;
                border-radius: 100%;
                text-align: center;
                width: 25px;
                height: 25px;
                line-height: 20px;
                display: inline-block;
                position: absolute;
                left: 0;
                top: 0;
            }}

            /* Carousel Styling */
            .carousel {{
              grid-gap: 10px;
              overflow-x: auto;
              scroll-snap-type: x mandatory;
              scroll-behavior: smooth;
              -webkit-overflow-scrolling: touch;
              width: 100%;
              margin: auto;
            }}
           .flashcard {{
              display: inline-block; /* Display cards inline */
              width: 600px;
              height: 600px;
              background-color: white;
              border-radius: 10px;
              box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
              perspective: 100px;
              margin-bottom: 0px; /* Add space between cards */
              padding: 0px;
              scroll-snap-align: center;
            }}
            .front, .back {{
              position: absolute;
              top: 0;
              left: 0;
              width: 600px;
              height: 600px;
              border-radius: 10px;
              backface-visibility: hidden;
              font-family: 'Ysabeau SC', sans-serif;
              text-align: center;
              overflow: hidden; /* Hide the scroll bar */
            }}
            .front {{
              background-color: #E9EBE1; /* Change the background color here */
              color: #333333;
              transform: rotateY(0deg);
            }}
            .back {{
              color: #333333;
              background-color: #E9EBE1; /* Change the background color here */
              transform: rotateY(180deg);
              display: flex;
              justify-content: center;
              align-items: center;
              flex-direction: column;
              border: 1px solid #48555e; /* Change the border color here */
            }}                                
            .flashcard:hover .front {{
              transform: rotateY(180deg);
            }}
            .flashcard:hover .back {{
              transform: rotateY(0deg);
            }}
            .front h1 {{
              padding-top: 10px;
              line-height: 1.5;
            }}
            .back h2 {{
              line-height: 2;
            }}
            .back p {{
              margin: 20px; /* Add margin for paragraph text */
            }}
            /* Carousel Navigation Styling */
            .carousel-nav {{
              margin: 10px 0px;
              text-align: center;
            }}
            </style>
            """, unsafe_allow_html=True)

    def create_flipcard_quick_summary(self, header_list, paragraph_list_front, paragraph_list_back, font_family,
                                      font_size_front,
                                      font_size_back, image_path_front_card=None, **kwargs):
        # Compute and display the metrics for the first column
        rows = st.session_state.df_raw.shape[0]
        min_date = str(st.session_state.df_raw.iloc[:, 0].min().date())
        percent_missing = "{:.2%}".format(round((st.session_state.df_raw.iloc[:, 1].isna().mean()), 2))
        mean_val = np.round(st.session_state.df_raw.iloc[:, 1].mean(), 2)
        min_val = np.round(st.session_state.df_raw.iloc[:, 1].min(skipna=True), 2)
        std_val = np.round(np.nanstd(st.session_state.df_raw.iloc[:, 1], ddof=0), 2)

        # Compute and display the metrics for the second column
        cols = st.session_state.df_raw.shape[1]
        max_date = str(st.session_state.df_raw.iloc[:, 0].max().date())
        dataframe_freq, dataframe_freq_name = determine_df_frequency(st.session_state.df_raw, column_name='date')
        median_val = np.round(st.session_state.df_raw.iloc[:, 1].median(skipna=True), 2)
        max_val = np.round(st.session_state.df_raw.iloc[:, 1].max(skipna=True), 2)
        mode_val = st.session_state.df_raw.iloc[:, 1].dropna().mode().round(2).iloc[0]

        # open the image for the front of the card
        with open(image_path_front_card, 'rb') as file:
            contents = file.read()
            data_url = base64.b64encode(contents).decode("utf-8")

        # create empty list that will keep the html code needed for each card with header+text
        card_html = []

        header_color = 'white'

        card_html.append(f"""<div class="flashcard">
                                <div class='front_summary'>
                                    <img src="data:image/png;base64,{data_url}"style="width: 100%; height: 100%; object-fit: cover; border-radius: 10px;">
                                    <h1 style='text-align:center;color:#F5F5F5; margin-bottom: 10px;padding: 35px;'>{header_list}</h1>
                                    <p style='text-align:center; font-family: Lato; font-size: {font_size_front};'>{paragraph_list_front} </p>
                                </div>
                                <div class="back_summary">
                                    <h2>{header_list}</h2>
                                       <div style="display: flex; justify-content: space-between; margin-bottom: -20px; margin-left: 20px; margin-right: 20px; margin-top: -20px;">
                                       <div style="text-align: center; margin-right: 80px;">
                                       <div style="margin-bottom: 0px; "><b style="color: {header_color};">rows</b></div><div>{rows}</div><br/>
                                       <div style="margin-bottom: 0px;"><b style="color: {header_color};">start date</b></div><div>{min_date}</div><br/>
                                       <div style="margin-bottom: 0px;"><b style="color: {header_color};">missing</b></div><div>{percent_missing}</div><br/>
                                       <div style="margin-bottom: 0px;"><b style="color: {header_color};">mean</b></div><div>{mean_val}</div><br/>
                                       <div style="margin-bottom: 0px;"><b style="color: {header_color};">minimum</b></div><div>{min_val}</div><br/>
                                       <div style="margin-bottom: 0px;"><b style="color: {header_color};">stdev</b></div><div>{std_val}</div><br/>
                                       </div>
                                       <div style="text-align: center;">
                                       <div style="margin-bottom: 0px; "><b style="color: {header_color};">columns</b></div><div>{cols}</div><br/>
                                       <div style="margin-bottom: 0px; "><b style="color: {header_color};">end date</b></div><div>{max_date}</div><br/>
                                       <div style="margin-bottom: 0px; "><b style="color: {header_color};">frequency</b></div><div>{dataframe_freq_name}</div><br/>
                                       <div style="margin-bottom: 0px; "><b style="color: {header_color};">median</b></div><div>{median_val}</div><br/>
                                       <div style="margin-bottom: 0px; "><b style="color: {header_color};">maximum</b></div><div>{max_val}</div><br/>
                                       <div style="margin-bottom: 0px; "><b style="color: {header_color};">mode</b></div><div>{mode_val}</div><br/>
                                    <p style='text-align:center; font-family: Lato; font-size: {font_size_back};'>{paragraph_list_back}</p>
                                </div>
                            </div>
                            """)

        # join all the html code for each card and join it into single html code with carousel wrapper
        carousel_html = "<div class='carousel'>" + "".join(card_html) + "</div>"

        # Display the carousel in streamlit
        st.markdown(carousel_html, unsafe_allow_html=True)
        # Create the CSS styling for the carousel
        st.markdown(
            f"""
            <style>   

            /* Carousel Styling */
            .carousel {{
              display: flex;
              justify-content: center;
              overflow-x: auto;
              scroll-snap-type: x mandatory;
              scroll-behavior: smooth;
              -webkit-overflow-scrolling: touch;
              width: 100%;
            }}
           .flashcard {{
              width: 600px;
              height: 600px;
              background-color: white;
              border-radius: 10px;
              box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
              perspective: 100px;
              margin-bottom: 10px;
              scroll-snap-align: center;
            }}
            .front_summary, .back_summary {{
              position: absolute;
              top: 0;
              left: 0;
              width: 100%;
              height: 100%;
              border-radius: 10px;
              backface-visibility: hidden;
              font-family: 'Ysabeau SC', sans-serif;
              font-size: {font_size_back};
              text-align: center;
            }}
            .front_summary {{
                background: linear-gradient(to bottom, #383e56, #383e56, #383e56, #383e56, #383e56, #383e56);
                color: #F5F5F5;
                transform: rotateY(0deg);
            }}
            .back_summary {{
                border: 1px solid #48555e; /* Change the border color here */
                background-color: #4d5466 ; /* Change the background color here */
                color: white;
                transform: rotateY(180deg);
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
                box-shadow: none; /* Remove the shadow */
                font-family: 'Ysabeau SC', sans-serif;
                font-size: 18px;
            }}                       
            .flashcard:hover .front_summary {{
              transform: rotateY(180deg);
            }}
            .flashcard:hover .back_summary {{
              transform: rotateY(0deg);
            }}
            .front_summary h1 {{
              padding-top: 10px;
              line-height: 1.5;
            }}
            .back_summary h2 {{
              line-height: 2;
            }}
            .back_summary p {{
              margin: 10px; /* Add margin for paragraph text */
            }}
            /* Carousel Navigation Styling */
            .carousel-nav {{
              margin: 10px 0px;
              text-align: center;
            }}
            </style>
            """, unsafe_allow_html=True)

    def create_flipcard_stats(self, image_path_front_card=None, font_size_back='10px', my_header=None, **kwargs):
        # Header in Streamlit
        col1, col2, col3 = st.columns([20, 40, 20])
        with col2:
            my_text_header(my_header)
            vertical_spacer(1)

        # Open the image for the front of the card
        with open(image_path_front_card, 'rb') as file:
            contents = file.read()
            data_url = base64.b64encode(contents).decode("utf-8")

        # Create empty list that will keep the HTML code needed for each card with header+text
        card_html = []

        # Append HTML code to list
        card_html.append(f"""
            <div class="flashcard">
                <div class='front'>
                    <img src="data:image/png;base64,{data_url}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 10px;">
                </div>
                <div class="back">
                    <h6>White Noise Test</h6>
                    <p><b>use case:</b> the <i>ljung-box</i> test is utilized to assess whether a forecasting model effectively
                    captures the inherent patterns present in the time series. it accomplishes this by first calculating the
                    differences between the actual values and the predicted values and tests if these unexplained differences,
                    known as <b>'residuals'</b> are either:
                    </p>
                    <ul>
                        <li><b>random fluctuations:</b> also referred to as <i>white noise</i> and exhibits no correlation or
                        predictable pattern over time. this is true when the expected total sum of both positive and negative
                        fluctuations will average out to zero and the variance remains constant/does not systematically increase
                        or decrease over time.</li>
                        <li><b>autocorrelated:</b> residuals exhibit a correlation with their own lagged values, indicating the
                        presence of a pattern that is currently not captured by the existing forecasting model.</li>
                    </ul>
                    <br>
                    <p><b>recommendation:</b> if the residuals are found to be 'just' random fluctuations, continue using the
                    current forecasting model<sup><a href="#footnote-1">1</a></sup> as it captures the underlying patterns.
                    Otherwise, if the presence of autocorrelation is found, consider using a more complex forecasting model or
                    increase the number of lags considered.</p>
                    <footer>
                        <div id="footnote-1">
                            <sup>1</sup> ForecastGenie uses an autoregressive model for the white noise test, which is the same
                            as using SARIMAX(p, 0, 0)(0, 0, 0), whereby p is the lag range (default=24).
                        </div>
                    </footer>
                    <br>
                    <h6>Stationarity Test</h6>
                    <p><b>use case:</b> The Augmented Dickey-Fuller (ADF) test is used to assess the stationarity of a time
                    series. Stationarity indicates that a time-series has a stable mean and variance over time, which is an
                    assumption for models such as Autoregressive Moving Average (ARMA) to forecast accurately. However, if
                    initially, your time-series is non-stationary, differencing could be applied to try and make a non-stationary
                    time-series stationary.</p>
                    <h6>Normality Test</h6>
                    <p><b>use case:</b> The Shapiro test examines residual distribution's normality in a model, where residuals
                    are observed-predicted differences. Non-normality may necessitate alternative models (e.g., Neural Networks or
                    Random Forests)</p>
                </div>
            </div>
        """)

        # Join all the HTML code for each card and join it into single HTML code with carousel wrapper
        carousel_html = "<div class='flipcard_stats'>" + "".join(card_html) + "</div>"
        # Display the carousel in Streamlit
        st.markdown(carousel_html, unsafe_allow_html=True)
        # Create the CSS styling for the carousel
        st.markdown(
            f"""
            <style>
            .flipcard_stats {{
              display: flex;
              justify-content: center;
              overflow-x: auto;
              scroll-snap-type: x mandatory;
              scroll-behavior: smooth;
              -webkit-overflow-scrolling: touch;
              width: 100%;
            }}
            .flashcard {{
              width: 600px;
              height: 600px;
              background-color: white;
              border-radius: 10px;
              box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
              perspective: 100px;
              margin-bottom: 10px;
              scroll-snap-align: center;
            }}
            .front, .back {{
              position: absolute;
              top: 0;
              left: 0;
              width: 100%;
              height: 100%;
              border-radius: 10px;
              backface-visibility: hidden;
              font-family: 'Ysabeau SC', sans-serif;
              font-size: {font_size_back};
              text-align: center;
            }}
            .front {{
              background: linear-gradient(to bottom left, #4e3fce, #7a5dc7, #9b7cc2, #bb9bbd, #c2b1c4);
              color: white;
              transform: rotateY(0deg);
            }}
            .back {{
              color: #333333;
              background-color: #E9EBE1;
              transform: rotateY(180deg);
              display: flex;
              justify-content: flex-start;
              border: 1px solid #48555e;
              margin: 0;
              padding: 60px;
              text-align: left;
              text-justify: inter-word;
              overflow: auto;
            }}
            .back h6 {{
              margin-bottom: 0px;
              margin-top: 0px;
            }}
            .flashcard:hover .front {{
              transform: rotateY(180deg);
            }}
            .flashcard:hover .back {{
              transform: rotateY(0deg);
            }}
            .back p {{
              margin: 10px 0;
            }}
            footer {{
              text-align: center;
              margin-top: 20px;
              font-size: 12px;
              margin-bottom: 20px;
            }}
            </style>
            """, unsafe_allow_html=True)
