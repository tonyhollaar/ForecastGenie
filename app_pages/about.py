# Standard Library Imports
import os
from typing import Literal

# Third-Party Library Imports
import streamlit as st
from streamlit_marquee import streamlit_marquee

# Local Imports
from site_links.social_media import SocialMediaLinks
from streamlit_extras.buy_me_a_coffee import button as buymeacoffee_btn
from style.icons import load_icons
from style.text import vertical_spacer, my_text_paragraph, my_text_header, my_bubbles

# Load icons
icons = load_icons()

# CONSTANTS
#######################################################################################################################
# File Paths
IMAGES_FOLDER = 'images'
IMAGES_PATH = os.path.join(".", IMAGES_FOLDER, "")

# Scrolling/marquee text effect
BACKGROUND_COLOR = "#f5f5f5"  # Scrolling/marquee text effect
FONT_SIZE = '16px'  # Scrolling/marquee text effect
TEXT_COLOR = "#000000"  # Scrolling/marquee text effect
LINE_HEIGHT = "0px"  # Scrolling/marquee text effect
CONTAINER_WIDTH = '800px'  # Scrolling/marquee text effect
ANIMATION_DURATION = '60s'

# Constants for Buy Me a Coffee button
BMC_USERNAME = "tonyhollaar"
BMC_FLOATING = False
BMC_WIDTH = 221
BMC_TEXT = 'Buy me a coffee'
BMC_BG_COLOR = '#FFFFFF'
BMC_FONT: Literal["Cookie", "Lato", "Arial", "Comic", "Inter", "Bree", "Poppins"] = 'Cookie'
BMC_FONT_COLOR = 'black'
BMC_COFFEE_COLOR = 'black'
MARGIN_AFTER = 14

# App version and release date
APP_VERSION = '2.0'
RELEASE_DATE = '06-30-2023'


class AboutPage:
    """Class representing the about page of ForecastGenie."""
    FRONT_COVER_IMAGE = os.path.join(IMAGES_PATH, 'about_page.png')

    FRONT_COVER = '-'
    WHAT_DOES_IT_DO = 'What does it do?'
    WHAT_DO_I_NEED = 'What do I need?'
    WHO_IS_THIS_FOR = 'Who is this for?'
    ORIGIN_STORY = "Origin Story"
    SHOW_YOUR_SUPPORT = "Show your support"
    VERSION = "Version"

    PAGES = [FRONT_COVER, WHAT_DOES_IT_DO, WHAT_DO_I_NEED, WHO_IS_THIS_FOR, ORIGIN_STORY, SHOW_YOUR_SUPPORT, VERSION]

    def __init__(self):
        """Initialize the AboutPage class."""
        self.selected_about_page = None

    def render(self):
        """Render the about page."""
        try:
            with st.sidebar:
                self._render_sidebar()

            if self.selected_about_page:
                self._render_selected_page()

        except Exception as e:
            st.error(f'ForecastGenie Error: "About" in sidebar-menu did not load properly. Error: {e}')

    def _render_sidebar(self):
        """Render the sidebar for the documentation page."""
        vertical_spacer(2)
        with (st.sidebar):
            my_text_paragraph(f'{icons["book_icon"]}')

            # Use st.sidebar.radio for the selected page index
            self.selected_about_page = st.sidebar.radio(
                label="*Select Page*:",
                options=self.PAGES,
                label_visibility='collapsed',
                index=self.PAGES.index(self.selected_about_page) if self.selected_about_page else 0
            )

            # Adjust the alignment of the label in the sidebar to center
            st.sidebar.markdown(
                """
                <style>
                    div[data-testid="stRadio"] {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

        # SHOW SOCIAL MEDIA ICONS/LINKS AT BOTTOM OF STREAMLIT SIDEBAR
        social_media = SocialMediaLinks()  # Instantiate the SocialMediaLinks class
        social_media.render()  # Render the social media links

    def _render_selected_page(self):
        """Render the selected page."""
        if self.selected_about_page == self.FRONT_COVER:
            self._render_front_cover()
        elif self.selected_about_page == self.WHAT_DOES_IT_DO:
            self._render_what_does_it_do()
        elif self.selected_about_page == self.WHAT_DO_I_NEED:
            self._render_what_do_i_need()
        elif self.selected_about_page == self.WHO_IS_THIS_FOR:
            self._render_who_is_this_for()
        elif self.selected_about_page == self.ORIGIN_STORY:
            self._render_origin_story()
        elif self.selected_about_page == self.SHOW_YOUR_SUPPORT:
            self._render_show_your_support()
        elif self.selected_about_page == self.VERSION:
            self._render_version()
        # Add other selected app_pages as needed...

    @staticmethod
    def _render_front_cover():
        """Render the content for 'Front Cover'"""
        with st.expander('', expanded=True):
            # Show the cover image in about page
            st.image(AboutPage.FRONT_COVER_IMAGE)

    @staticmethod
    def _render_what_does_it_do():
        """Render the content for 'What does it do?'"""
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                my_text_header('What does it do?')
                my_text_paragraph('🕵️‍♂️ <b> Analyze data:', add_border=True, border_color="#F08A5D")
                my_text_paragraph('Inspect seasonal patterns and distribution of the data', add_border=False)
                my_text_paragraph('🧹 <b> Cleaning data: </b>', add_border=True, border_color="#F08A5D")
                my_text_paragraph('Automatic detection and replacing missing data points and remove outliers')
                my_text_paragraph('🧰 <b> Feature Engineering: </b>', add_border=True, border_color="#F08A5D")
                my_text_paragraph(' Add holidays, calendar day/week/month/year and optional wavelet features')
                my_text_paragraph('⚖️ <b> Normalization and Standardization </b>', add_border=True,
                                  border_color="#F08A5D")
                my_text_paragraph('Select from industry standard techniques')
                my_text_paragraph('🍏 <b> Feature Selection: </b>', add_border=True, border_color="#F08A5D")
                my_text_paragraph('</b> Only keep relevant features based on feature selection techniques')
                my_text_paragraph('🍻 <b> Correlation Analysis:</b> ', add_border=True, border_color="#F08A5D")
                my_text_paragraph('Automatically remove highly correlated features')
                my_text_paragraph('🔢 <b> Train Models:</b>', add_border=True, border_color="#F08A5D")
                my_text_paragraph('Including Naive, Linear, SARIMAX and Prophet Models')
                my_text_paragraph('🎯 <b> Evaluate Model Performance:', add_border=True, border_color="#F08A5D")
                my_text_paragraph('Benchmark models performance with evaluation metrics')
                my_text_paragraph('🔮  <b> Forecast:', add_border=True, border_color="#F08A5D")
                my_text_paragraph(
                    'Forecast your variable of interest with ease by selecting your desired end-date from the calendar')

    @staticmethod
    def _render_what_do_i_need():
        """Render the content for 'What do I need?'"""
        with st.expander('', expanded=True):
            # Render the content for "What do I need?"
            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                my_text_header('What do I need?')
                my_text_paragraph('<b> File: </b>', add_border=True, border_color="#0262ac")
                my_text_paragraph('''From the <i> Load </i> menu you can select the radio-button 'Upload Data' and 
                drag-and-drop or left-click to browse the file from your local computer.''', my_text_align='justify')
                my_text_paragraph('<b> File Requirements: </b>', add_border=True, border_color="#0262ac")
                my_text_paragraph('''- the 1st column should have a header with the following lowercase name: <b> 
                'date' </b> with dates in ascending order in format: <b>mm/dd/yyyy</b>.  For example 12/31/2024. <br> 
                - the 2nd column should have a header as well with a name of your choosing. This column should 
                contain the historical values of your variable of interest, which you are trying to forecast, 
                based on historical data.''', my_text_align='left')
                my_text_paragraph('<b> Supported File Formats: </b>', add_border=True, border_color="#0262ac")
                my_text_paragraph('Common file-formats are supported, namely: .csv, .xls .xlsx, xlsm and xlsb.',
                                  my_text_align='justify')
                vertical_spacer(25)

    @staticmethod
    def _render_who_is_this_for():
        """Render the content for 'Who is this for?'"""
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([2, 6, 2])
            with col2:
                my_text_header('Who is this for?')
                my_text_paragraph("<b>Target Groups</b>", add_border=True, border_color="#bc62f6")
                my_text_paragraph('ForecastGenie is designed to cater to a wide range of users, including <i>Business '
                                  'Analysts, Data Scientists, Statisticians, and Enthusiasts.</i>Whether you are a '
                                  'seasoned professional or an aspiring data enthusiast, this app aims to provide you '
                                  'with powerful data analysis and forecasting capabilities! If you are curious about '
                                  'the possibilities, some use cases are listed below.', my_text_align='justify')
                my_text_paragraph("<b>Business Analysts:</b>", add_border=True, border_color="#bc62f6")
                my_text_paragraph('''- <b>Sales Forecasting:</b> Predict future sales volumes and trends to optimize 
                inventory management, plan marketing campaigns, and make informed business decisions.<br> - <b>Demand 
                Forecasting:</b> Estimate future demand for products or services to streamline supply chain 
                operations, improve production planning, and optimize resource allocation.<br> - <b>Financial 
                Forecasting:</b> Forecast revenues, expenses, and cash flows to facilitate budgeting, 
                financial planning, and investment decisions.''', my_text_align='left')
                my_text_paragraph("<b>Data Scientists:</b>", add_border=True, border_color="#bc62f6")
                my_text_paragraph('''- <b>Time Series Analysis:</b> Analyze historical data patterns and trends to 
                identify seasonality, trend components, and anomalies for various applications such as finance, 
                energy, stock market, or weather forecasting.<br> - <b>Predictive Maintenance:</b> Forecast equipment 
                failures or maintenance needs based on sensor data, enabling proactive maintenance scheduling and 
                minimizing downtime.<br> - <b>Customer Churn Prediction:</b> Predict customer churn probabilities, 
                allowing companies to take preventive measures, retain customers, and enhance customer satisfaction.''',
                                  my_text_align='left')
                my_text_paragraph("<b>Statisticians:</b>", add_border=True, border_color="#bc62f6")
                my_text_paragraph('''- <b>Economic Forecasting:</b> Forecast macroeconomic indicators such as GDP, 
                inflation rates, or employment levels to support economic policy-making, investment strategies, 
                and financial planning.<br> - <b>Population Forecasting:</b> Predict future population sizes and 
                demographic changes for urban planning, resource allocation, and infrastructure development.<br> - 
                <b>Epidemiological Forecasting:</b> Forecast disease spread and outbreak patterns to aid in public 
                health planning, resource allocation, and implementation of preventive measures.''',
                                  my_text_align='left')
                my_text_paragraph("<b>Enthusiasts:</b>", add_border=True, border_color="#bc62f6")
                my_text_paragraph('''- <b>Personal Budget Forecasting:</b> Forecast your personal income and expenses 
                to better manage your finances, plan savings goals, and make informed spending decisions.<br> - 
                <b>Weather Forecasting:</b> Generate short-term or long-term weather forecasts for personal planning, 
                outdoor activities, or agricultural purposes.<br> - <b>Energy Consumption Forecasting:</b> Forecast 
                your household or individual energy consumption patterns based on historical data, helping you 
                optimize energy usage, identify potential savings opportunities, and make informed decisions about 
                energy-efficient upgrades or renewable energy investments.''', my_text_align='left')

    @staticmethod
    def _render_origin_story():
        """Render the content for 'Origin Story'"""
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                my_text_header('Origin Story')
                my_text_paragraph(
                    'ForecastGenie is a forecasting app created by Tony Hollaar with a background in data analytics '
                    'with the goal of making accurate forecasting accessible to businesses of all sizes.')
                my_text_paragraph(
                    'As the developer of the app, I saw a need for a tool that simplifies the process of forecasting, '
                    'without sacrificing accuracy, therefore ForecastGenie was born.')
                my_text_paragraph(
                    'With ForecastGenie, you can upload your data and quickly generate forecasts using '
                    'state-of-the-art machine learning algorithms. The app also provides various evaluation metrics '
                    'to help you benchmark your models and ensure their performance meets your needs.')

                # DISPLAY LOGO
                col1, col2, col3 = st.columns([2, 8, 2])
                with col2:
                    st.image('./images/coin_logo.png')

            # scrolling/marquee text effect
            scroll_text = """
                            In a world ravaged by chaos,
                            Humans and robots intertwine.

                            Humans bring emotions and dreams,
                            Robots bring logic and precision.

                            Together, they mend their broken realm,
                            Contributing unique strengths.

                            Side by side, they stand as guardians,
                            Defending their fragile society.

                            Humans rely on machine strength,
                            Robots depend on human empathy.

                            Bound by an unbreakable code,
                            They blend their destinies.

                            Hope emerges amidst the chaos,
                            Balance and harmony prevail. - Quote from OpenAI's GPT-3.5-based ChatGPT
                            """

            streamlit_marquee(**{
                'background': BACKGROUND_COLOR,
                'fontSize': FONT_SIZE,
                'color': TEXT_COLOR,
                'content': scroll_text,
                'width': CONTAINER_WIDTH,
                'lineHeight': LINE_HEIGHT,
                'animationDuration': ANIMATION_DURATION,
            })

    @staticmethod
    def _render_show_your_support():
        """Render the content for 'Show your support'"""
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([2, 4, 2])
            with col2:
                st.markdown(
                    '<h1 style="text-align:center; font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif; '
                    'font-weight: 200; font-size: 32px; line-height: 1.5;">Show your support {}</h1>'.format(
                        icons["balloon_heart_svg"]), unsafe_allow_html=True)
                vertical_spacer(2)
                my_text_paragraph(
                    'If you find this app useful, please consider supporting it by buying me a coffee. Your support '
                    'helps me continue developing and maintaining this app. Thank you!')

            col1, col2, col3 = st.columns([2, 2, 2])
            with col2:
                vertical_spacer(2)
                my_bubbles('')
                buymeacoffee_btn(username=BMC_USERNAME, floating=BMC_FLOATING, width=BMC_WIDTH, text=BMC_TEXT,
                                 bg_color=BMC_BG_COLOR, font=BMC_FONT, font_color=BMC_FONT_COLOR,
                                 coffee_color=BMC_COFFEE_COLOR)
                vertical_spacer(MARGIN_AFTER)

    @staticmethod
    def _render_version():
        """Render the content for 'Version'"""
        with st.expander('', expanded=True):
            my_text_header('Version')
            st.caption(f'<h7><center> ForecastGenie version: `{APP_VERSION}` <br>  Release date: `{RELEASE_DATE}`  </center></h7>',
                       unsafe_allow_html=True)
            vertical_spacer(30)

    @staticmethod
    def social_media_links():
        """Render the social media links."""
        # Define how social media links are rendered
        social_media = SocialMediaLinks()
        social_media.render()
