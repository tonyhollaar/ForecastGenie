# faq.py
import streamlit as st
from style.text import vertical_spacer, my_text_paragraph
from style.icons import load_icons

# Load icons
icons = load_icons()


class FaqPage:
    IMAGES_PATH = "./images/"
    FRONT_COVER_IMAGE = IMAGES_PATH + "faq_page.png"

    """Class representing the FAQ page of your app."""
    SECTION_FRONT_COVER = '-'
    SECTION_WHAT_IS_FORECAST_GENIE = 'What is ForecastGenie?'
    SECTION_DATA_USAGE = 'What data can I use with ForecastGenie?'
    SECTION_FORECASTING_MODELS = 'What forecasting models does ForecastGenie offer?'
    SECTION_METRICS_USED = 'What metrics does ForecastGenie use?'
    SECTION_IS_FREE = 'Is ForecastGenie really free?'
    SECTION_NON_TECHNICAL_USERS = 'Is ForecastGenie suitable for non-technical users?'
    SECTION_OTHER_QUESTIONS = 'Other Questions?'

    PAGES = [SECTION_FRONT_COVER, SECTION_WHAT_IS_FORECAST_GENIE, SECTION_DATA_USAGE, SECTION_FORECASTING_MODELS,
             SECTION_METRICS_USED, SECTION_IS_FREE, SECTION_NON_TECHNICAL_USERS, SECTION_OTHER_QUESTIONS]

    def __init__(self):
        """Initialize the FAQPage class."""
        self.selected_faq_page = None

    def render(self):
        """Render the FAQ page."""
        try:
            with st.sidebar:
                self._render_sidebar()

            if self.selected_faq_page:
                self._render_selected_page()

        except Exception as e:
            st.error(f'ForecastGenie Error: "FAQ" in sidebar-menu did not load properly. Error: {e}')

    def _render_sidebar(self):
        """Render the sidebar for the FAQ page."""
        vertical_spacer(2)
        with st.sidebar:
            my_text_paragraph(f'{icons["book_icon"]}')

        # Use st.sidebar.radio for the selected page index
        self.selected_faq_page = st.sidebar.radio(
            label="*Select Page*:",
            options=self.PAGES,
            label_visibility='collapsed',
            index=self.PAGES.index(self.selected_faq_page) if self.selected_faq_page else 0
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

    def _render_selected_page(self):
        """Render the selected section."""
        if self.selected_faq_page == self.SECTION_FRONT_COVER:
            self._render_front_cover()
        elif self.selected_faq_page == self.SECTION_WHAT_IS_FORECAST_GENIE:
            self._render_what_is_forecast_genie()
        elif self.selected_faq_page == self.SECTION_DATA_USAGE:
            self._render_data_usage()
        elif self.selected_faq_page == self.SECTION_FORECASTING_MODELS:
            self._render_forecasting_models()
        elif self.selected_faq_page == self.SECTION_METRICS_USED:
            self._render_metrics_used()
        elif self.selected_faq_page == self.SECTION_IS_FREE:
            self._render_is_free()
        elif self.selected_faq_page == self.SECTION_NON_TECHNICAL_USERS:
            self._render_non_technical_users()
        elif self.selected_faq_page == self.SECTION_OTHER_QUESTIONS:
            self._render_other_questions()

    @staticmethod
    def _render_front_cover():
        """Render the content for 'Front Cover'"""
        with st.expander('', expanded=True):
            st.image(FaqPage.FRONT_COVER_IMAGE)  # Tree Image

    @staticmethod
    def _render_what_is_forecast_genie():
        """Render the content for 'What is ForecastGenie?'"""
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                vertical_spacer(10)
                my_text_paragraph('<b>Question:</b> <i> What is ForecastGenie? </i>',
                                  my_text_align='justify')
                my_text_paragraph(
                    '<b> Answer:</b> ForecastGenie is a free, open-source application that enables users to perform '
                    'time-series forecasting on their data. The application offers a range of advanced features and '
                    'models to help users generate accurate forecasts and gain insights into their data.',
                    my_text_align='justify')
                vertical_spacer(20)

    @staticmethod
    def _render_data_usage():
        """Render the content for 'What data can I use with ForecastGenie?'"""
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                vertical_spacer(10)
                my_text_paragraph('<b>Question:</b><i> What data can I use with ForecastGenie?</i>',
                                  my_text_align='justify')
                my_text_paragraph(
                    '<b>Answer:</b> ForecastGenie accepts data in the form of common file types such as .CSV or .XLS. '
                    'Hereby the first column should contain the dates and the second column containing the target '
                    'variable of interest. The application can handle a wide range of time-series data, '
                    'including financial data, sales data, weather data, and more.',
                    my_text_align='justify')
                vertical_spacer(20)

    @staticmethod
    def _render_forecasting_models():
        """Render the content for 'What forecasting models does ForecastGenie offer?'"""
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                vertical_spacer(10)
                my_text_paragraph('<b>Question:</b> <i>What forecasting models does ForecastGenie offer?</i>',
                                  my_text_align='justify')
                my_text_paragraph(
                    '<b>Answer:</b> ForecastGenie offers a range of models to suit different data and use cases, '
                    'including Naive, SARIMAX, and Prophet. The application also includes hyper-parameter tuning, '
                    'enabling users to optimize the performance of their models and achieve more accurate forecasts.',
                    my_text_align='justify')
                vertical_spacer(20)

    @staticmethod
    def _render_metrics_used():
        """Render the content for 'What metrics does ForecastGenie use?'"""
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                vertical_spacer(10)
                my_text_paragraph('<b>Question: </b><i>What metrics does ForecastGenie use?</i>',
                                  my_text_align='justify')
                my_text_paragraph(
                    '<b>Answer: </b> ForecastGenie uses a range of business-standard evaluation metrics to assess the '
                    'accuracy of forecasting models, including Mean Absolute Error (MAE), Mean Squared Error (MSE), '
                    'Root Mean Squared Error (RMSE), and more. These metrics provide users with a reliable and '
                    'objective measure of their model\'s performance.',
                    my_text_align='justify')
                vertical_spacer(20)

    @staticmethod
    def _render_is_free():
        """Render the content for 'Is ForecastGenie really free?'"""
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                vertical_spacer(10)
                my_text_paragraph('<b>Question: </b><i>Is ForecastGenie really free?</i>', my_text_align='justify')
                my_text_paragraph('<b>Answer:</b> Yes! ForecastGenie is completely free and open-source.',
                                  my_text_align='justify')
                vertical_spacer(20)

    @staticmethod
    def _render_non_technical_users():
        """Render the content for 'Is ForecastGenie suitable for non-technical users?'"""
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                vertical_spacer(10)
                my_text_paragraph('<b>Question:</b><i> Is ForecastGenie suitable for non-technical users?</i>',
                                  my_text_align='justify')
                my_text_paragraph(
                    '<b>Answer:</b> Yes! ForecastGenie is designed to be user-friendly and intuitive, even for users '
                    'with little or no technical experience. The application includes automated data cleaning and '
                    'feature engineering, making it easy to prepare your data for forecasting. Additionally, '
                    'the user interface is simple and easy to navigate, with clear instructions and prompts '
                    'throughout the process.',
                    my_text_align='justify')
                vertical_spacer(20)

    @staticmethod
    def _render_other_questions():
        """Render the content for 'Other Questions?'"""
        with st.expander('', expanded=True):
            vertical_spacer(8)

            # Flashcard front / back
            col1, col2, col3 = st.columns([4, 12, 4])
            with col2:
                my_code = """<div class="flashcard_faq">
                              <div class="front_faq">
                                <div class="content">
                                  <h2>Other Questions?</h2>
                                </div>
                              </div>
                              <div class="back_faq">
                                <div class="content">
                                  <h2>info@forecastgenie.com</h2>
                                </div>
                              </div>
                            </div>
                            <style>
                            .flashcard_faq {
                              position: relative;
                              width: 400px;
                              height: 400px;
                              background-color: white;
                              border-radius: 10px;
                              box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                              perspective: 1000px;
                              transition: transform 0.6s;
                              transform-style: preserve-3d;
                            }

                            .front_faq, .back_faq {
                              position: absolute;
                              top: 0;
                              left: 0;
                              width: 100%;
                              height: 100%;
                              border-radius: 10px;
                              backface-visibility: hidden;
                              font-family: "Roboto", Arial, sans-serif; 
                              display: flex;
                              justify-content: center;
                              align-items: center;
                              text-align: center;
                            }

                            .front_faq {
                              background: linear-gradient(to top, #f74996 , #3690c0);
                              color: white;
                              transform: rotateY(0deg);
                            }

                            .back_faq {
                              background:linear-gradient(to top, #f74996 , #3690c0);
                              transform: rotateY(180deg);
                            }

                            .flashcard_faq:hover .front_faq {
                              transform: rotateY(180deg);
                            }

                            .flashcard_faq:hover .back_faq {
                              transform: rotateY(0deg);
                            }

                            .content h2 {
                              margin: 0;
                              font-size: 26px;
                              line-height: 1.5;
                            }

                            .back_faq h2 {
                              color: white;  /* Add this line to set text color to white */
                            }
                            .front_faq h2 {
                              color: white;  /* Add this line to set text color to white */
                            }
                            </style>"""

                # show flashcard in streamlit
                st.markdown(my_code, unsafe_allow_html=True)
                vertical_spacer(10)