# -*- coding: utf-8 -*-
"""
_____ _____ _____ _____ _____ _____ _____ _____ _____ _____ 
\____\\____\\____\\____\\____\\____\\____\\____\\____\\____\
  ___ ___  ___ ___ ___   _   ___ _____ ___ ___ _  _ ___ ___ 
 | __/ _ \| _ \ __/ __| /_\ / __|_   _/ __| __| \| |_ _| __|
 | _| (_) |   / _| (__ / _ \\__ \ | || (_ | _|| .` || || _| 
 |_| \___/|_|_\___\___/_/ \_\___/ |_| \___|___|_|\_|___|___|    
_____ _____ _____ _____ _____ _____ _____ _____ _____ _____ 
\____\\____\\____\\____\\____\\____\\____\\____\\____\\____\
    
Streamlit App: ForecastGenie
Forecast y based on X timeseries data
@author: Tony Hollaar
Date: 06/30/2023
Version 2.0
source ASCII ART: http://patorjk.com/software/taag/#p=display&f=Graffiti&t=Type%20Something%20
"""
# =============================================================================
#   _      _____ ____  _____            _____  _____ ______  _____ 
#  | |    |_   _|  _ \|  __ \     /\   |  __ \|_   _|  ____|/ ____|
#  | |      | | | |_) | |__) |   /  \  | |__) | | | | |__  | (___  
#  | |      | | |  _ <|  _  /   / /\ \ |  _  /  | | |  __|  \___ \ 
#  | |____ _| |_| |_) | | \ \  / ____ \| | \ \ _| |_| |____ ____) |
#  |______|_____|____/|_|  \_\/_/    \_\_|  \_\_____|______|_____/ 
#                                                                  
# =============================================================================
#**************************
# Import basic packages
#**************************
import pandas as pd
import numpy as np
import streamlit as st
       
import json
import base64

#**************************
# Streamlit add-on packages
#**************************
from streamlit_option_menu import option_menu
from streamlit_extras.buy_me_a_coffee import button
from streamlit_extras.dataframe_explorer import dataframe_explorer
from st_click_detector import click_detector
from streamlit_lottie import st_lottie

# source: https://github.com/inspurer/streamlit-marquee
# scrolling text
from streamlit_marquee import streamlit_marquee

# source: https://github.com/Mr-Milk/streamlit-fire-state
# keep user preferences of streamlit st.form submissions in session state 
# make available when switching between pages e.g. menu items
from fire_state import create_store, form_update, get_state, set_state, get_store, set_store
        
#**************************
# Import datetime 
#**************************
import datetime
from datetime import timedelta

#**************************
# Import itertools 
#**************************
import itertools

#**************************
# Time functions
#**************************
import time

#**************************
# Math package
#**************************
import math
import sympy as sp

#**************************
# Image Processing 
#**************************
#from PIL import Image

#**************************
# Text manipulation
#**************************
import re

#***********************************
# Data Visualization 
#***********************************
import altair as alt
#import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#****************************
# Statistical tools
#****************************
from scipy import stats
#from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf, adfuller
from scipy.stats import mode, kurtosis, skew, shapiro
#from statsmodels.stats.diagnostic import acorr_ljungbox
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

#********************************
# Data (Pre-) Processing
#********************************
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from prophet.diagnostics import cross_validation, performance_metrics    
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
import pywt
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
#from sklearn.neighbors import NearestNeighbors
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from pandas.tseries.offsets import BDay
import holidays
from pandas.tseries.holiday import(
                                    AbstractHolidayCalendar, Holiday, DateOffset, \
                                    SU, MO, TU, WE, TH, FR, SA, \
                                    next_monday, nearest_workday, sunday_to_monday,
                                    EasterMonday, GoodFriday, Easter
                                  )
    
#********************************
# Hyperparameter tuning framework
#********************************
import optuna
from optuna import visualization as ov

# =============================================================================
#   _____        _____ ______    _____ ______ _______ _    _ _____  
#  |  __ \ /\   / ____|  ____|  / ____|  ____|__   __| |  | |  __ \ 
#  | |__) /  \ | |  __| |__    | (___ | |__     | |  | |  | | |__) |
#  |  ___/ /\ \| | |_ |  __|    \___ \|  __|    | |  | |  | |  ___/ 
#  | |  / ____ \ |__| | |____   ____) | |____   | |  | |__| | |     
#  |_| /_/    \_\_____|______| |_____/|______|  |_|   \____/|_|     
#                                                                                                                                    
# =============================================================================
# SET PAGE CONFIGURATIONS STREAMLIT
st.set_page_config(page_title = "ForecastGenie™️", 
                   layout = "centered", # "centered" or "wide"
                   page_icon = "🌀", 
                   initial_sidebar_state = "expanded") # "auto" or "expanded" or "collapsed"

# SET FONT STYLES
font_style = """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Ysabeau+SC:wght@200&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Rubik+Dirt&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Rock+Salt&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Josefin+Slab:wght@200&display=swap');
            
            /* Set the font family for paragraph elements */
            p {
               font-family: 'Ysabeau SC', sans-serif;
            }
            
            /* Set the font family, size, and weight for unordered list and ordered list elements */
            ul, ol {
                font-family: 'Ysabeau SC', sans-serif;
                font-size: 16px;
                font-weight: normal;
            }
            
            /* Set the font family, size, weight, and margin for list item elements */
            li {
                font-family: 'Ysabeau SC', sans-serif;
                font-size: 16px;
                font-weight: normal;
                margin-top: 5px;
            }
            </style>
            """

# =============================================================================
# # Render the font styles in Streamlit
# =============================================================================
st.markdown(font_style, unsafe_allow_html=True)

# =============================================================================
#   _____   ____   ____  _____  _      ______  _____ 
#  |  __ \ / __ \ / __ \|  __ \| |    |  ____|/ ____|
#  | |  | | |  | | |  | | |  | | |    | |__  | (___  
#  | |  | | |  | | |  | | |  | | |    |  __|  \___ \ 
#  | |__| | |__| | |__| | |__| | |____| |____ ____) |
#  |_____/ \____/ \____/|_____/|______|______|_____/ 
#                                                    
# =============================================================================
# =============================================================================
#             # image outliers
#             st.markdown('---')
#             vertical_spacer(2)
#             col1, col2, col3 = st.columns([1,3,1])
#             with col2:
#                 image_path = './images/outliers.png'
#                 caption = 'Doodle: A Slippery Slope'
#                 st.image(image_path, caption=caption)
# =============================================================================
# =============================================================================
#         # Doodle Dickey-Fuller Test
#         image = Image.open("./images/adf_test.png")
#         # Display the image in Streamlit
#         st.image(image, caption="", use_column_width=True)
#         #my_text_paragraph('Doodle: Dickey-Fuller Test', my_font_size='12px')
# =============================================================================  
# =============================================================================
#         # AUTOCORRELATION DOODLE #
#         # canva drawing autocorrelation
#         vertical_spacer(2)
#         st.markdown('---')
#         col1, col2, col3 = st.columns([1,3,1])
#         with col2:
#             # Load the canva demo_data image from subfolder images
#             image = Image.open("./images/autocorrelation.png")
#             # Display the image in Streamlit
#             st.image(image, caption="", use_column_width='auto')
#             my_text_paragraph('Doodle: Autocorrelation', my_font_size='12px')
# =============================================================================
# =============================================================================
#    _____  _____ _____            _____  ____   ____   ____  _  __
#   / ____|/ ____|  __ \     /\   |  __ \|  _ \ / __ \ / __ \| |/ /
#  | (___ | |    | |__) |   /  \  | |__) | |_) | |  | | |  | | ' / 
#   \___ \| |    |  _  /   / /\ \ |  ___/|  _ <| |  | | |  | |  <  
#   ____) | |____| | \ \  / ____ \| |    | |_) | |__| | |__| | . \ 
#  |_____/ \_____|_|  \_\/_/    \_\_|    |____/ \____/ \____/|_|\_\
#                                                                  
# =============================================================================

# TEST
# =============================================================================
# def create_flipcards_model_cards(num_cards, header_list, paragraph_list_front, paragraph_list_back, font_family, font_size_front, font_size_back):
#     # note removing display: flex; inside the css code for .flashcard -> puts cards below eachother
#     # create empty list that will keep the html code needed for each card with header+text
#     card_html = []
#     # iterate over cards specified by user and join the headers and text of the lists
#     for i in range(num_cards):
#         card_html.append(f"""<div class="flashcard">                     
#                                 <div class='front'>
#                                     <h1 style='text-align:center;'>{header_list[i]}</h1>
#                                     <p style='text-align:center; font-size: {font_size_front};'>{paragraph_list_front[i]}</p>
#                                 </div>
#                                 <div class="back">
#                                     <p style='text-align:justify; word-spacing: 1px; font-size: {font_size_back}; margin-right: 60px; margin-left: 60px'>{paragraph_list_back[i]}</p>
#                                 </div>
#                             </div>
#                             """)
#     # join all the html code for each card and join it into single html code with carousel wrapper
#     carousel_html = "<div class='carousel'>" + "".join(card_html) + "</div>"
#     
#     # Display the carousel in streamlit
#     st.markdown(carousel_html, unsafe_allow_html=True)
#     
#     # Create the CSS styling for the carousel
#     st.markdown(
#         f"""
#         <style>
#         /* Carousel Styling */
#         .carousel {{
#           grid-gap: 0px; /* Reduce the gap between cards */
#           justify-content: center; /* Center horizontally */
#           align-items: center; /* Center vertically */
#           overflow-x: auto;
#           scroll-snap-type: x mandatory;
#           scroll-behavior: smooth;
#           -webkit-overflow-scrolling: touch;
#           width: 500px;
#           margin: 0 auto; /* Center horizontally by setting left and right margins to auto */
#           background-color: transparent; /* Remove the background color */
#           padding: 0px; /* Remove padding */
#           border-radius: 0px; /* Add border-radius for rounded edges */
#         }}
#         .flashcard {{
#           display: inline-block; /* Display cards inline */
#           width: 500px;
#           height: 200px;
#           background-color: transparent; /* Remove the background color */
#           border-radius: 0px;
#           border: 2px solid black; /* Add black border */
#           perspective: 0px;
#           margin-bottom: 0px; /* Remove space between cards */
#           padding: 0px;
#           scroll-snap-align: center;
#         }}
#         .front, .back {{
#           position: absolute;
#           top: 0;
#           left: 0;
#           width: 500px;
#           height: 200px;
#           border-radius: 0px;
#           backface-visibility: hidden;
#           font-family: 'Ysabeau SC', sans-serif; /* font-family: font-family: 'Ysabeau SC', sans-serif; */
#           text-align: center;
#           margin: 0px;
#           background: white;
#         }}
#         .back {{
#             /* ... other styles ... */
#             background: none; /* Remove the background */
#             color: transparent; /* Make the text transparent */
#             background-clip: text; /* Apply the background gradient to the text */
#             -webkit-background-clip: text; /* For Safari */
#             -webkit-text-fill-color: transparent; /* For Safari */
#             background-image: linear-gradient(to bottom left, #941c8e, #763a9a, #4e62a3, #2e81ad, #12a9b4); /* Set the background gradient */
#             transform: rotateY(180deg);
#             display: flex;
#             justify-content: center;
#             align-items: center;
#             flex-direction: column;
#         }}                               
#         .flashcard:hover .front {{
#           transform: rotateY(180deg);
#         }}
#         .flashcard:hover .back {{
#           transform: rotateY(0deg);
#           cursor: default; /* Change cursor to pointer on hover */
#         }}
#         .front h1, .back p {{
#           color: black;
#           text-align: center;
#           margin: 0;
#           font-family: {font_family};
#           font-size: {font_size_front}px;
#         }}
#         .back p {{
#           line-height: 1.5;
#           margin: 0;
#         }}
#         /* Carousel Navigation Styling */
#         .carousel-nav {{
#           margin: 0px 0px;
#           text-align: center;
#         }}
#         </style>
#         """, unsafe_allow_html=True)
# =============================================================================

# =============================================================================
# def create_moon_clickable_btns(button_names = ['MY_BUTTON_TEXT']):
#     """
#     Generates HTML code for creating clickable moon buttons.
#     
#     Args:
#         button_names (list): A list of button names.
#     
#     Returns:
#         str: The lowercase button name if a moon button is clicked with added lowercase if spaces in the name
#         
#     Example:
#         clicked_btn_name = create_moon_clickable_btns(button_names = [my button name'])                                                            
#         if clicked_btn_name == 'my_button_name':
#             print(f'you clicked the button {my_button_name}!')        
#     
#     Requirements:
#         streamlit
#         from st_click_detector import click_detector #source: https://github.com/vivien000/st-click-detector
#         
#     """
#     button_colors = [
#                      'radial-gradient(circle at 50% 50%, #000000)', 
#                      'conic-gradient(from 180deg at 50% 50%, #f5f5f5 0deg 180deg, #000000 180deg)', 
#                      'radial-gradient(circle at 50% 50%, #000000)', 
#                      'conic-gradient(from 180deg at 50% 50%, #000000 0deg 180deg, #f5f5f5 180deg)'
#                     ]
#     
#     # MOON BUTTONS
#     button_container = """
#     <style>
#         .button-container {{
#             display: flex;
#             flex-direction: row;
#             align-items: center;
#             justify-content: center;
#             padding: 10px;
#             margin-top: -30px;
#             position: relative;
#             perspective: 1000px;
#             transform-style: preserve-3d;
#         }}
#     
#         .button {{
#             width: 70px;
#             height: 70px;
#             background: #292929;
#             border-radius: 50%;
#             margin: 0 10px;
#             position: relative;
#             perspective: 1000px;
#             transform-style: preserve-3d;
#             overflow: hidden;
#         }}
#     
#         .button:before {{
#             content: "";
#             position: absolute;
#             top: 50%;
#             left: 50%;
#             width: 120%;
#             height: 120%;
#             border-radius: 50%;
#             background: transparent;
#             border: 2px solid rgba(255, 255, 255, 0.3);
#             transform-style: preserve-3d;
#             transform: translate(-50%, -50%) rotateY(-90deg);
#             animation: planet-ring-rotation 8s linear infinite reverse;
#             z-index: -1;
#         }}
#     
#         .button:after {{
#             content: "";
#             position: absolute;
#             top: 50%;
#             left: 50%;
#             width: 120%;
#             height: 2px;
#             background: rgba(255, 255, 255, 0.3);
#             transform-style: preserve-3d;
#             transform: translate(-50%, -50%) rotateX(90deg);
#         }}
#     
#         .button a {{
#             position: absolute;
#             top: 0;
#             left: 0;
#             width: 100%;
#             height: 100%;
#             display: flex;
#             justify-content: center;
#             align-items: center;
#             text-decoration: 'none';
#             color: #f5f5f5;
#             text-transform: uppercase;
#             font-size: 16px;
#             font-weight: bold;
#             white-space: nowrap;
#             animation: text-rotation 8s linear infinite reverse;
#             transform-style: preserve-3d;
#             perspective: 1000px;
#         }}
#     
#         @keyframes planet-ring-rotation {{
#             0% {{
#                 transform: translate(100%, -50%) rotateY(0deg);
#                 clip-path: polygon(50%  0%, 0% 0%, 0% 0%, 0% 100%);
#             }}
#             50% {{
#                 transform: translate(-70%, -50%) rotateY(-180deg);
#                 clip-path: polygon(50%  0%, 100% 0%, 100% 100%, 50% 100%);
#             }}
#             100% {{
#                 transform: translate(-50%, -50%) rotateY(-360deg);
#                 clip-path: polygon(50% 0%, 100% 0%, 100% 100%, 50% 100%);
#             }}
#         }}
#     
#         @keyframes text-rotation {{
#             0% {{
#                 transform: translateX(-100%) scale(0.5);
#                 opacity: 0;
#             }}
#             10% {{
#                 transform: translateX(-70%) scale(0.7) perspective(1000px);
#                 opacity: 1;
#             }}
#             90% {{
#                 transform: translateX(70%) scale(0.7) perspective(1000px);
#                 opacity: 1;
#             }}
#             100% {{
#                 transform: translateX(100%) scale(0.5) perspective(1000px) rotateY(180deg);
#                 opacity: 0;
#             }}
#         }}
#     </style>
#     <div class="button-container">
#         {buttons_html}
#     </div>
#     """
# 
#     # Define the HTML code for a single button
#     button_template = """
#     <div class="button" style="background: {color};">
#         <a href="#" id="{id}">
#             {name}
#         </a>
#     </div>
#     """
#     
#     # Generate the HTML code for the buttons
#     buttons_html = ""
#     for i, model_name in enumerate(button_names):
#         button_html = button_template.format(
#             id=model_name.lower().replace(" ", "_"),
#             name=model_name,
#             color=button_colors[i % len(button_colors)]
#         )
#         buttons_html += button_html
#     
#     # Combine the buttons HTML with the container HTML
#     full_html = button_container.format(buttons_html=buttons_html)
#  
#     clicked_model_btn = click_detector(full_html)
#     return clicked_model_btn
# =============================================================================

# =============================================================================
# # =============================================================================
# #  Show/Hide Button to download dataframe                   
# # =============================================================================
# # have button available for user and if clicked it expands with the dataframe
# col1, col2, col3 = st.columns([100,50,95])
# with col2:    
#     # create empty placeholder for button show/hide
#     placeholder = st.empty()
#     
#     # create button (enabled to click e.g. disabled=false with unique key)
#     btn = placeholder.button('Show Details', 
#                              disabled=False,
#                              key = "show_naive_trained_model_btn")
#    
# # if button is clicked run below code
# if btn == True:  
#     
#     # display button with text "click me again", with unique key
#     placeholder.button('Hide Details', disabled=False, key = "hide_naive_trained_model_btn")
# =============================================================================
    
# =============================================================================
# show_lottie_animation(url="./images/89601-solar-system.json", key='solar_system', speed = 1, width=400, reverse=False, height=400, margin_before = 2, margin_after=10)
# =============================================================================
            
# =============================================================================
# 
# # left / right arrow you can create with click event from st-click-detector package
# button_container2 = """
# <style>
# .arrow-container {
#     display: flex;
#     justify-content: center;
#     align-items: center;
# }
# 
# .arrow {
#     width: 30px;
#     height: 30px;
#     border-radius: 50%;
#     background-color: black;
#     display: flex;
#     justify-content: center;
#     align-items: center;
#     cursor: pointer;
# }
# 
# .left-arrow {
#     margin-right: 10px;
# }
# 
# .right-arrow {
#     margin-left: 10px;
# }
# 
# .arrow-icon {
#     width: 10px;
#     height: 10px;
#     border-top: 2px solid white;
#     border-right: 2px solid white;
#     transform: rotate(225deg);
# }
# .rarrow-icon {
#     width: 10px;
#     height: 10px;
#     border-top: 2px solid white;
#     border-right: 2px solid white;
#     transform: rotate(45deg);
# }
# </style>
# 
# <div class="arrow-container">
#     <a href="#" id="left-arrow" class="arrow left-arrow" onclick="handleClick('left-arrow')">
#         <span class="arrow-icon"></span>
#     </a>
#     <a href="#" id="right-arrow" class="arrow right-arrow" onclick="handleClick('right-arrow')">
#         <span class="rarrow-icon"></span>
#     </a>
# </div>
# """
# =============================================================================

# =============================================================================
#             title = '\"Hi 👋 Welcome to the ForecastGenie app!\"'
#             # set gradient color of letters of title
#             gradient = '-webkit-linear-gradient(left, #F08A5D, #FABA63, #2E9CCA, #4FB99F)'
#             # show in streamlit the title with gradient
#             st.markdown(f'<h1 style="text-align:center; background: none; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-image: {gradient};"> {title} </h1>', unsafe_allow_html=True)
#             # vertical spacer
#             st.write('')
# =============================================================================
# =============================================================================
#                 # Create Carousel Cards
#                 # define for each card the header in the header list
#                 header_list = ["📈", 
#                                "🔍",  
#                                "🧹",  
#                                "🧰",  
#                                "🔢"]
#                 # define for each card the corresponding paragraph in the list
#                 paragraph_list = ["Forecasting made easy", 
#                                   "Professional data analysis", 
#                                   "Automated data cleaning", 
#                                   "Intelligent feature engineering", 
#                                   "User-friendly interface", 
#                                   "Versatile model training"]
#                 # define the font family to display the text of paragraph
#                 font_family = "Trebuchet MS"
#                 # define the paragraph text size
#                 font_size = '18px'
#                 # in streamlit create and show the user defined number of carousel cards with header+text
#                 create_carousel_cards(4, header_list, paragraph_list, font_family, font_size)
# =============================================================================

# =============================================================================
#   ______ _    _ _   _  _____ _______ _____ ____  _   _  _____ 
#  |  ____| |  | | \ | |/ ____|__   __|_   _/ __ \| \ | |/ ____|
#  | |__  | |  | |  \| | |       | |    | || |  | |  \| | (___  
#  |  __| | |  | | . ` | |       | |    | || |  | | . ` |\___ \ 
#  | |    | |__| | |\  | |____   | |   _| || |__| | |\  |____) |
#  |_|     \____/|_| \_|\_____|  |_|  |_____\____/|_| \_|_____/ 
#                                                               
# =============================================================================
def run_naive_model_1(lag_options = None, validation_set = None):
    """
    Run the grid search for Naive Model I using different lag options.
    
    Parameters:
        lag_options (list or None): List of lag options to be tuned during grid search.
                                   Each lag option represents a specific configuration for Naive Model I.
                                   Default is None.
        total_options (int or None): Total number of options in the grid search.
                                     Default is None.
        validation_set (pd.DataFrame or None): Validation dataset used for forecasting and performance evaluation.
                                               It should contain the target variable.
                                               Default is None.
    
    Returns:
        None
    
    This function performs a grid search to tune Naive Model I by trying different lag options and
    evaluating their performance using the provided validation dataset. The results are displayed as a DataFrame
    with the rankings of different lag options based on their MAPE (Mean Absolute Percentage Error) metric.
    
    Note: The function expects a progress bar named `progress_bar` to be initialized in the Streamlit app.
    
    Example:
    run_naive_model_1(lag_options=[1, 2, 3], total_options=9, validation_set=validation_data)
    
    """
    # Set title
    my_text_paragraph('Naive Model I: Lag')
    
    # Define model name 
    model_name = 'Naive Model I'
    
    # Set start time when grid-search is kicked-off to define total time it takes as computationaly intensive
    start_time_naive_model_1 = time.time()
    
    naive_model1_tuning_results = pd.DataFrame() # results with lag
    
    # iterate over grid of all possible combinations of hyperparameters
    for i, lag_option in enumerate(lag_options):

        # Check if the maximum waiting time has been exceeded
        elapsed_time_seconds = time.time() - start_time
            
        if elapsed_time_seconds > max_wait_time_seconds:
            st.warning("Maximum waiting time exceeded. The grid search has been stopped.")
            # exit the loop once maximum time is exceeded defined by user or default = 5 minutes
            break
        
        # Update the progress bar
        progress_percentage = i / total_options * 100
        progress_bar.progress(value = (i / total_options), 
                              text = f'''Please wait up to {max_wait_time_minutes} minute(s) while parameters of Naive Models are being tuned!  
                                         \n{progress_percentage:.2f}% of total options within the search space reviewed ({i} out of {total_options} total options).''')
                                         
        # Create a model with the current parameter values
        df_preds_naive_model1 = forecast_naive_model1_insample(validation_set, lag = lag_option.lower(), custom_lag_value = None)
        
        # Retrieve metrics from difference actual - predicted    
        metrics_dict = my_metrics(my_df = df_preds_naive_model1, model_name = model_name)
                                                       
        # Create a new DataFrame with the row to append
        new_row = pd.DataFrame({'parameters': [f'lag: {lag_option}'],
                                'MAPE':  metrics_dict[model_name]['mape'],
                                'SMAPE': metrics_dict[model_name]['smape'],
                                'RMSE': metrics_dict[model_name]['rmse'],
                                'R2': metrics_dict[model_name]['r2']})
                                 
        # Concatenate the original DataFrame with the new row DataFrame
        naive_model1_tuning_results = pd.concat([naive_model1_tuning_results, new_row], ignore_index=True)

        # Add rank column to dataframe and order by metric column
        ranked_naive_model1_tuning_results = rank_dataframe(naive_model1_tuning_results, 'MAPE')
        
        # Convert MAPE and SMAPE metrics to string with '%' percentage-sign
        ranked_naive_model1_tuning_results[['MAPE', 'SMAPE']] = ranked_naive_model1_tuning_results[['MAPE', 'SMAPE']].applymap(lambda x: f'{x*100:.2f}%' if not np.isnan(x) else x)

    st.dataframe(ranked_naive_model1_tuning_results, use_container_width = True, hide_index = True)
   
    # set the end of runtime
    end_time_naive_model_1 = time.time()
    
    text = str(ranked_naive_model1_tuning_results.iloc[0, 1])
    formatted_text = re.sub(r'([^:]+): ([^,]+)', r'`\1`: **\2**', text)
    final_text = re.sub(r',\s*', ', ', formatted_text)
    
    st.write(f'''💬**Naive Model I** parameter with the lowest MAPE of {ranked_naive_model1_tuning_results["MAPE"][0]} found in **{end_time_naive_model_1 - start_time_naive_model_1:.2f}** seconds is:  
             {final_text}''')

def run_naive_model_2(rolling_window_range = None, rolling_window_options = None, validation_set = None):
    """
    Run the grid search for Naive Model II using different rolling window configurations.
    
    Parameters:
        rolling_window_range (tuple or None): A tuple (start, end) representing the range of rolling window sizes
                                              to be considered during the grid search for Naive Model II.
                                              Default is None.
        rolling_window_options (list or None): List of aggregation methods to be considered for the rolling window.
                                               Each method represents a specific configuration for Naive Model II.
                                               Default is None.
        validation_set (pd.DataFrame or None): Validation dataset used for forecasting and performance evaluation.
                                               It should contain the target variable.
                                               Default is None.
    
    Returns:
        None
    
    This function performs a grid search to tune Naive Model II by trying different combinations of rolling window
    sizes and aggregation methods, and evaluating their performance using the provided validation dataset. The results
    are displayed as a DataFrame with the rankings of different configurations based on their MAPE (Mean Absolute
    Percentage Error) metric.
    
    Note: The function expects a progress bar named `progress_bar` to be initialized in the Streamlit app.
    
    Example:
    run_naive_model_2(rolling_window_range=(3, 5), rolling_window_options=['mean', 'median'], validation_set=validation_data)
    
    """
    # Define title
    my_text_paragraph('Naive Model II: Rolling Window')
    
    # Define model name
    model_name = 'Naive Model II'
    
    # save evaluation results to dataframe
    naive_model2_tuning_results = pd.DataFrame()
    
    # Iterate over grid of all possible combinations of hyperparameters
    param_grid = {'rolling_window_range': list(range(rolling_window_range[0], rolling_window_range[1] + 1)),
                  'rolling_window_options': rolling_window_options}
              
    start_time_naive_model_2 = time.time()
    
    for i, (rolling_window_value, rolling_window_option) in enumerate(itertools.product(param_grid['rolling_window_range'], param_grid['rolling_window_options']), 0):
        
        # Check if the maximum waiting time has been exceeded
        elapsed_time_seconds = time.time() - start_time
            
        if elapsed_time_seconds > max_wait_time_seconds:
            st.warning("Maximum waiting time exceeded. The grid search has been stopped.")
            # exit the loop once maximum time is exceeded defined by user or default = 5 minutes
            break  
        
        # Update the progress bar
        progress_percentage = (i + len(lag_options)) / total_options * 100
        progress_bar.progress(value = ((i  + len(lag_options) ) / total_options), 
                               text = f'''Please wait up to {max_wait_time_minutes} minute(s) while parameters of Naive Models are being tuned!  
                                         \n{progress_percentage:.2f}% of total options within the search space reviewed ({i + len(lag_options)} out of {total_options} total options).''')

        # Return a prediction dataframe 
        df_preds_naive_model2 = forecast_naive_model2_insample(y_test, size_rolling_window = rolling_window_value, agg_method_rolling_window = rolling_window_option)
         
        # Retrieve metrics from difference actual - predicted
        metrics_dict = my_metrics(my_df = df_preds_naive_model2, model_name = model_name)
         
        # Create a new DataFrame with the row to append
        new_row = pd.DataFrame({'parameters': [f'rolling_window_size: {rolling_window_value}, aggregation_method: {rolling_window_option}'],
                                'MAPE':  metrics_dict[model_name]['mape'],
                                'SMAPE': metrics_dict[model_name]['smape'],
                                'RMSE': metrics_dict[model_name]['rmse'],
                                'R2': metrics_dict[model_name]['r2']})
                                 
        # Concatenate the original DataFrame with the new row DataFrame
        naive_model2_tuning_results = pd.concat([naive_model2_tuning_results, new_row], ignore_index=True)

        # Add rank column to dataframe and order by metric column
        ranked_naive_model2_tuning_results = rank_dataframe(naive_model2_tuning_results, 'MAPE')
   
        # Convert MAPE and SMAPE evaluation metrics to '%' percentage-sign
        ranked_naive_model2_tuning_results[['MAPE', 'SMAPE']] = ranked_naive_model2_tuning_results[['MAPE', 'SMAPE']].applymap(lambda x: f'{x*100:.2f}%' if not np.isnan(x) else x)
    
    st.dataframe(ranked_naive_model2_tuning_results, use_container_width=True, hide_index = True)
    
    # Set the end of runtime
    end_time_naive_model_2 = time.time()
    
    # Clear progress bar in streamlit for user as process is completed
    progress_bar.empty()
    
    # Text formatting for model parameters style in user message:
    text = str(ranked_naive_model2_tuning_results.iloc[0, 1])
    formatted_text = re.sub(r'([^:]+): ([^,]+)', r'`\1`: **\2**', text)
    final_text = re.sub(r',\s*', ', ', formatted_text)
    
    st.write(f'''💬**Naive Model II** parameters with the lowest MAPE of {ranked_naive_model2_tuning_results["MAPE"][0]} found in **{end_time_naive_model_2 - start_time_naive_model_2:.2f}** seconds is:  
             {final_text}''')
                
def run_naive_model_3(agg_options = None, train_set = None, validation_set = None):

    # Define Title
    my_text_paragraph('Naive Model III: Constant Value')
    
    # Define Model Name    
    model_name = 'Naive Model III'
    
    # Initiate dataframe to save results to with constant (mean/median/mode)
    naive_model3_tuning_results = pd.DataFrame() 
    
    # Iterate over grid of all possible combinations of hyperparameters
    param_grid = {'agg_options': agg_options}

    # set the end of runtime
    start_time_naive_model_3 = time.time()
                                    
    for i, agg_option in enumerate(param_grid['agg_options']):
        
        # Check if the maximum waiting time has been exceeded
        elapsed_time_seconds = time.time() - start_time
       
        # If maximum time is exceeded, stop running loop
        if elapsed_time_seconds > max_wait_time_seconds:
            st.warning("Maximum waiting time exceeded. The grid search has been stopped.")
            # exit the loop once maximum time is exceeded defined by user or default = 5 minutes
            break  
        
        # Update the progress bar
        progress_percentage = (i + len(agg_options) + (len(list(range(rolling_window_range[0], rolling_window_range[1] + 1)))*len(rolling_window_options))) / total_options * 100
        progress_bar.progress(value = ((i  + len(agg_options) + (len(list(range(rolling_window_range[0], rolling_window_range[1] + 1)))*len(rolling_window_options))) / total_options), 
                              text = f'''Please wait up to {max_wait_time_minutes} minute(s) while parameters of Naive Models are being tuned!  
                                        \n{progress_percentage:.2f}% of total options within the search space reviewed ({i + len(agg_options) + (len(list(range(rolling_window_range[0], rolling_window_range[1] + 1)))*len(rolling_window_options))} out of {total_options} total options).''')                         
        
        # return dataframe with the predictions e.g. constant of mean/median/mode of training dataset for length of validation set
        df_preds_naive_model3 = forecast_naive_model3_insample(train_set, 
                                                               validation_set, 
                                                               agg_method = agg_option)
        
        # retrieve metrics from difference actual - predicted
        metrics_dict = my_metrics(my_df = df_preds_naive_model3, model_name = model_name)
        
        # Create a new DataFrame with the row to append
        new_row = pd.DataFrame({'parameters': [f'aggregation method: {agg_option}'],
                                'MAPE':  metrics_dict[model_name]['mape'],
                                'SMAPE': metrics_dict[model_name]['smape'],
                                'RMSE': metrics_dict[model_name]['rmse'],
                                'R2': metrics_dict[model_name]['r2']})
                                
        # Concatenate the original DataFrame with the new row DataFrame
        naive_model3_tuning_results = pd.concat([naive_model3_tuning_results, new_row], ignore_index=True)

        # add rank column to dataframe and order by metric column
        ranked_naive_model3_tuning_results = rank_dataframe(naive_model3_tuning_results, 'MAPE')
  
        # convert mape to %
        ranked_naive_model3_tuning_results[['MAPE', 'SMAPE']] = ranked_naive_model3_tuning_results[['MAPE', 'SMAPE']].applymap(lambda x: f'{x*100:.2f}%' if not np.isnan(x) else x)

    st.dataframe(ranked_naive_model3_tuning_results, use_container_width=True, hide_index = True)
    
    # set the end of runtime
    end_time_naive_model_3 = time.time()
    
    # clear progress bar in streamlit for user as process is completed
    progress_bar.empty()
    
    # text formatting for model parameters style in user message:
    text = str(ranked_naive_model3_tuning_results.iloc[0, 1])
    formatted_text = re.sub(r'([^:]+): ([^,]+)', r'`\1`: **\2**', text)
    final_text = re.sub(r',\s*', ', ', formatted_text)
    
    st.write(f'''💬**Naive Model III** parameter with the lowest MAPE of {ranked_naive_model3_tuning_results["MAPE"][0]} found in **{end_time_naive_model_3 - start_time_naive_model_3:.2f}** seconds is:  
             {final_text}''')

def create_hyperparameter_importance_plot(data, plot_type='param_importances', grid_search=False):
    """
    Create a hyperparameter importance plot.
    
    Parameters:
        data (object or list): The data required to create the plot. For 'param_importances' plot_type,
                              it can be a study object from Optuna. For 'bar_chart' plot_type, it should
                              be a list of tuples containing hyperparameter and importance values.
        plot_type (str, optional): The type of plot to be created. Defaults to 'param_importances'.
        grid_search (bool, optional): Specifies if the plot is for regular grid search. Defaults to False.
    
    Returns:
        None
    
    Output:
        Streamlit Figure: Displays the hyperparameter importance plot in Streamlit.
    
    """
    fig = go.Figure()
    color_schema = px.colors.qualitative.Plotly

    if plot_type == 'param_importances' and not grid_search:
        fig = ov.plot_param_importances(data)
        fig.update_layout(title='', margin=dict(t=10))
        for i, trace in enumerate(fig.data):
            trace.marker.color = [color_schema[j % len(color_schema)] for j in range(len(trace.y))]

    elif plot_type == 'bar_chart':
        sorted_feature_importance = data
        for hyperparameter, importance in sorted_feature_importance:
            fig.add_trace(go.Bar(
                x=[importance],
                y=[hyperparameter],
                orientation='h',
                name=hyperparameter,
                marker=dict(color=color_schema[len(fig.data) % len(color_schema)]),
            ))
        fig.update_layout(title='', margin=dict(t=10), xaxis=dict(title='Importance Score'), yaxis=dict(title='Hyperparameter'))

    # Set figure title centered in Streamlit
    my_text_paragraph('Hyperparameter Importance Plot')

    # Show the figure in Streamlit with hyperparameter importance values
    st.plotly_chart(fig, use_container_width=True)

def rank_dataframe(df, metric):
    """
    Rank the DataFrame based on the specified column.

    Args:
        df (pd.DataFrame): The DataFrame to be ranked.
        metric (str): The name of the column to rank the DataFrame.

    Returns:
        pd.DataFrame: The ranked DataFrame.
    """
    # Convert the specified column to numeric data type
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    
    # Sort the DataFrame by the specified column in ascending order
    df = df.sort_values(metric)
    
    # Reset the index of the DataFrame and assign ranks
    df = df.reset_index(drop=True).reset_index().rename(columns={'index': 'Rank'})
    return df


def clear_test_results():
    """
    Clears the user test results from memory.
    
    This function deletes both the latest user test results and the historical test results from memory.
    It updates the 'results_df' entry in the Streamlit session state with an empty DataFrame,
    resetting it to the initial state with the specified column names.
    
    Note:
        This function assumes that the Streamlit session state is being used and the 'results_df' key exists.
    
    Example usage:
        clear_test_results()
    """
    # Delete latest user test results from memory
    del st.session_state['results_df']
   
    # Delete historical test results from memory
    set_state("TRAIN_PAGE", ("results_df", pd.DataFrame(columns=['model_name', 'predicted', 'mape', 'smape', 'rmse', 'r2', 'features', 'model_settings'])))
    
    # do you want to remove checkboxes when user presses clear? if so add below codeblock for each model checkbox on page train sidebar
    # =============================================================================
    #                 set_state("TRAIN_PAGE", ("naive_checkbox", False))
    #                 set_state("TRAIN_PAGE", ("linreg_checkbox", False))
    #                 set_state("TRAIN_PAGE", ("sarimax_checkbox", False))
    #                 set_state("TRAIN_PAGE", ("prophet_checkbox", False))
    # =============================================================================
def plot_model_performance(df, my_metric):
    """
    Plots a bar chart showing the models with the lowest error or highest R2 score.

    Args:
        df (pd.DataFrame): The DataFrame containing the model data.
        selected_metric (str): The metric to display and compare for the models. It can be one of the following:
                               - 'mape' (mean absolute percentage error)
                               - 'rmse' (root mean squared error)
                               - 'r2' (coefficient of determination)

    Note:
        The function assumes that the 'mape', 'mse', 'rmse', and 'r2' columns exist in the DataFrame.

    Example usage:
        df = pd.DataFrame(...)
        plot_lowest_metric(df, 'mape')
    """
# =============================================================================
#     try:
# =============================================================================
    # convert the metric name from user selection to metric abbreviation used in results dataframe
    metric_dict = {'Mean Absolute Percentage Error': 'mape', 'Symmetric Mean Absolute Percentage Error': 'smape', 'Root Mean Square Error':'rmse', 'R-squared':'r2'}    
    selected_metric = metric_dict[my_metric]
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Remove percentage sign from 'mape'  column and convert to float
    df_copy['mape'] = df_copy['mape'].str.replace('%', '').astype(float)
    df_copy['smape'] = df_copy['smape'].str.replace('%', '').astype(float)

    # Convert the selected metric column to numeric data type
    df_copy[selected_metric] = pd.to_numeric(df_copy[selected_metric], errors='coerce')

    # Group the dataframe by 'model_name' and find the row with the lowest score for each model and selected metric
    lowest_score_rows = df_copy.groupby('model_name')[selected_metric].idxmin()

    # Filter the dataframe to keep only the rows with the lowest scores for each model and selected metric
    filtered_df = df_copy.loc[lowest_score_rows, ['model_name', selected_metric]]

    # Extract the unique model names from the filtered dataframe
    model_names = filtered_df['model_name'].unique()

    # Create a color mapping dictionary for consistent colors based on model names
    color_schema = px.colors.qualitative.Plotly
    color_mapping = {model_name: color_schema[i % len(color_schema)] for i, model_name in enumerate(model_names)}

    # Sort the filtered dataframe based on the selected metric in ascending or descending order
    ascending_order = selected_metric != 'r2'  # Set ascending order unless 'r2' is selected
    filtered_df = filtered_df.sort_values(by=selected_metric, ascending=ascending_order)

    # Create the bar chart using Plotly Express
    fig = px.bar(
        filtered_df,
        x = 'model_name',
        y = selected_metric,
        color='model_name',
        color_discrete_map = color_mapping,  # Use the custom color mapping
        category_orders = {'model_name': filtered_df['model_name'].tolist()},  # Set the order of model names
        text = filtered_df[selected_metric].round(2),  # Add labels to the bars with rounded metric values
    )

    fig.update_layout(
        xaxis_title='Model',
        yaxis_title=selected_metric.upper(),
        barmode='stack',
        showlegend=True,
        legend=dict(x=1, y=1),  # Adjust the position of the legend
    )

    fig.update_traces(textposition='inside', textfont=dict(color='white'), insidetextfont=dict(color='white'))  # Adjust the position and color of the labels

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)
# =============================================================================
#     except:
#         st.error('FORECASTGENIE ERROR: Could not plot model performance. Please contact an administrator.')
# =============================================================================
    
def social_media_links(margin_before = 30):
    vertical_spacer(margin_before)
    # Get the URL to link to
    github_url = "https://github.com/tonyhollaar/ForecastGenie"  # Replace with your GitHub URL
    medium_url = "https://medium.com/@thollaar"  # Replace with your Medium URL
    logo_url = "https://tonyhollaar.com"  # Replace with the URL to your Website logo
    twitter_url = "https://twitter.com/tonyhollaar"  # Replace with your Twitter URL
    buymeacoffee_url = "https://www.buymeacoffee.com/tonyhollaar"  # Replace with your BuyMeACoffee URL
    
    # Generate the HTML code for the GitHub icon
    github_code = f'<a href="{github_url}"><img src="https://raw.githubusercontent.com/tonyhollaar/ForecastGenie/main/images/github-mark.png" alt="GitHub" width="32"></a>'
    
    # Generate the HTML code for the Medium icon
    medium_code = f'<a href="{medium_url}"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Medium_logo_Monogram.svg/512px-Medium_logo_Monogram.svg.png" alt="Medium" width="32"></a>'
    
    # Generate the HTML code for the logo
    logo_code = f'<a href="{logo_url}"><img src="https://raw.githubusercontent.com/tonyhollaar/ForecastGenie/main/images/logo_website.png" alt="Logo" width="32"></a>'
    
    twitter_code = f'<a href="{twitter_url}"><img src="https://raw.githubusercontent.com/tonyhollaar/ForecastGenie/main/images/twitter_logo_black.png" alt="Logo" width="32"></a>'
    
    buymeacoffee_code = f'<a href="{buymeacoffee_url}"><img src="https://raw.githubusercontent.com/tonyhollaar/ForecastGenie/main/images/buymeacoffee_logo.png" alt="buymeacoffee" height="32"></a>'
    # Render the icons using st.markdown()
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1])
    with col2:
        st.markdown(logo_code, unsafe_allow_html=True)
       
    with col4:
        st.markdown(medium_code, unsafe_allow_html=True)
        
    with col6:
        st.markdown(twitter_code, unsafe_allow_html=True)
        
    with col8:
        st.markdown(github_code, unsafe_allow_html=True)

    with col10:
        st.markdown(buymeacoffee_code, unsafe_allow_html=True)

def form_feature_selection_method_mifs():
    try:
        with st.form('mifs'):
            my_text_paragraph('Mutual Information')
            
            # Add slider to select number of top features
            num_features_mifs = st.slider(
                                          label = "*Select number of top features to include:*", 
                                          min_value = 0, 
                                          max_value = len(X.columns), 
                                          #value = 5,
                                          key = key1_select_page_mifs,
                                          step = 1
                                         )
            
            col1, col2, col3 = st.columns([4,4,4])
            with col2:
                mifs_btn = st.form_submit_button("Submit", type="secondary", on_click=form_update, args=('SELECT_PAGE_MIFS',))
                
                # keep track of changes if mifs_btn is pressed
                if mifs_btn:
                    set_state("SELECT_PAGE_BTN_CLICKED", ('mifs_btn', True))
            return num_features_mifs
    except:
        st.error('the user form for feature selection method \'Mutual Information\' could not execute')

def form_feature_selection_method_pca():
    try:
        with st.form('pca'):

            my_text_paragraph('Principal Component Analysis')

            # Add a slider to select the number of features to be selected by the PCA algorithm
            num_features_pca = st.slider(
                                         label = '*Select number of top features to include:*', 
                                         min_value = 0, 
                                         max_value = len(X_train.columns), 
                                         key = key1_select_page_pca
                                        )

            col1, col2, col3 = st.columns([4,4,4])
            with col2:       
                pca_btn = st.form_submit_button(label = "Submit", type = "secondary", on_click=form_update, args=('SELECT_PAGE_PCA',))  
                if pca_btn:
                    set_state("SELECT_PAGE_BTN_CLICKED", ('pca_btn', True))
            return num_features_pca
    except:
        st.error('the user form for feature selection method \'Principal Component Analaysis\' could not execute')

def form_feature_selection_method_rfe():
    try:
        # show Title in sidebar 'Feature Selection' with purple background
        my_title(f'{select_icon}', "#3b3b3b", gradient_colors="#1A2980, #7B52AB, #FEBD2E")

        # =============================================================================
        # RFE Feature Selection - SIDEBAR FORM
        # =============================================================================
        with st.form('rfe'):
             my_text_paragraph('Recursive Feature Elimination')
             # Add a slider to select the number of features to be selected by the RFECV algorithm
             
             num_features_rfe = st.slider(
                                          label = f'Select number of top features to include*:', 
                                          min_value = 0, 
                                          max_value = len(st.session_state['X'].columns), 
                                          key = key1_select_page_rfe,
                                          help = '**`Recursive Feature Elimination (RFE)`** is an algorithm that iteratively removes the least important features from the feature set until the desired number of features is reached.\
                                                  \nIt assigns a rank to each feature based on their importance scores. It is possible to have multiple features with the same ranking because the importance scores of these features are identical or very close to each other.\
                                                  \nThis can happen when the features are highly correlated or provide very similar information to the model. In such cases, the algorithm may not be able to distinguish between them and assign the same rank to multiple features.'
                                          )


             # set the options for the rfe (recursive feature elimination)
             with st.expander('◾', expanded=False):
                 # Include duplicate ranks (True/False) because RFE can have duplicate features with the same ranking
                 duplicate_ranks_rfe = st.selectbox(label = '*\*allow duplicate ranks:*',
                                                    options = [True, False],
                                                    key = key5_select_page_rfe,
                                                    help = '''
                                                           Allow multiple features to be added to the feature selection with the same ranking when set to `True`. otherwise, when the option is set to `False`, only 1 feature will be added to the feature selection. 
                                                           ''')

                 # Add a selectbox for the user to choose the estimator
                 estimator_rfe = st.selectbox(
                                              label = '*set estimator:*', 
                                              options = ['Linear Regression', 'Random Forest Regression'], 
                                              key = key2_select_page_rfe,
                                              help = '''
                                                     The **`estimator`** parameter is used to specify the machine learning model that will be used to evaluate the importance of each feature. \
                                                     The estimator is essentially the algorithm used to fit the data and make predictions.
                                                     ''')
                                              
                 # Add a slider to select the number of n_splits for the RFE method
                 timeseriessplit_value_rfe = st.slider(label = '*set number of splits for cross-validation:*', 
                                                       min_value = 2, 
                                                       max_value = 5, 
                                                       key = key3_select_page_rfe,
                                                       help = '**`Cross-validation`** is a statistical method used to evaluate the performance of a model by splitting the dataset into multiple "folds," where each fold is used as a holdout set for testing the model trained on the remaining folds. \
                                                              The cross-validation procedure helps to estimate the performance of the model on unseen data and reduce the risk of overfitting.  \
                                                              In the context of RFE, the cv parameter specifies the number of folds to use for the cross-validation procedure.\
                                                              The RFE algorithm fits the estimator on the training set, evaluates the performance of the estimator on the validation set, and selects the best subset of features. \
                                                              The feature selection process is repeated for each fold of the cross-validation procedure.')
                 
                 # Add a slider in the sidebar for the user to set the number of steps parameter
                 num_steps_rfe = st.slider(label = '*set number of steps:*', 
                                           min_value = 1, 
                                           max_value = 10, 
                                           #value = 1, 
                                           key = key4_select_page_rfe,
                                           help = 'The `step` parameter controls the **number of features** to remove at each iteration of the RFE process.')

             col1, col2, col3 = st.columns([4,4,4])
             with col2:       
                 
                 rfe_btn = st.form_submit_button("Submit", type="secondary", on_click=form_update, args=('SELECT_PAGE_RFE',))   
                 
                 if rfe_btn:
                     # update session state with user preference feature selection
                     set_state("SELECT_PAGE_BTN_CLICKED", ('rfe_btn', True))
                     
             return num_features_rfe, duplicate_ranks_rfe, estimator_rfe, timeseriessplit_value_rfe, num_steps_rfe
    except:
        st.error('the user form for feature selection method \'Recursive Feature Selection\' could not execute')
         
def show_card_feature_selection_methods():
    """
   Displays feature selection methods using a Streamlit expander and carousel cards.
   
   Parameters:
       None
   
   Returns:
       None
   """
    with st.expander('', expanded=True):
        #vertical_spacer(2)
        col1, col2, col3 = st.columns([3,8,2])
        with col2:
            
            title = 'Select your top features with 3 methods!'
            
            # set gradient color of letters of title
            #gradient = '-webkit-linear-gradient(left, #9c27b0, #673ab7, #3f51b5, #2196f3, #03a9f4)'
            gradient = '-webkit-linear-gradient(left, #FFFFF, #7c91b4, #a9b9d2, #7b8dad, #69727c)'
            
            # show in streamlit the title with gradient
            st.markdown(f'<h1 style="text-align:center; font-family: Ysabeau SC; font-size: 30px; background: none; -webkit-background-clip: text;"> {title} </h1>', unsafe_allow_html=True)
           
            vertical_spacer(2)
            
            #### CAROUSEL ####
            #header_list = ['🎨', '🧮', '🎏']
            header_list = ['', '', '']
            paragraph_list_front = [
                                    "<b> Recursive Feature Elimination </b>", 
                                    "<b>Principal Component Analysis</b>", 
                                    "<b>Mutual Information</b>"
                                   ]
            paragraph_list_back = [
                                   "<b>RFE</b> is a <b>feature selection technique</b> that repeatedly removes feature(s) and each turn evaluates remaining features by ranking the features based on their importance scores and eliminates the least important feature. This process continues until a desired number of features is reached.", 
                                   "<b>PCA</b> is a <b>feature selection technique</b> that repeatedly transforms and evaluates features based on their variance, reducing the dataset to a smaller set of uncorrelated variables called principal components. This process continues until a desired number of components is achieved.", 
                                   "<b>MI</b> is a <b>feature selection technique</b> that calculates the mutual information between each feature and the target to determine how much information each feature provides about the target."
                                  ]
            #font_family = "Ysabeau SC"
            font_family = "Josefin Slab"
            font_size_front = '16px'
            font_size_back = '15px'    
            
            # in Streamlit create and show the user defined number of carousel cards with header+text
            create_carousel_cards_v3(3, header_list, paragraph_list_front, paragraph_list_back, font_family, font_size_front, font_size_back)
            vertical_spacer(2)

        # Display a NOTE to the user about using the training set for feature selection
        my_text_paragraph('NOTE: per common practice <b>only</b> the training dataset is used for feature selection to prevent <b>data leakage</b>.',my_font_size='12px', my_font_family='Arial')   
        
def analyze_feature_correlations(
    selected_corr_model: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_cols_rfe: list,
    selected_cols_pca: list,
    selected_cols_mifs: list,
    corr_threshold: float = 0.8,
    models: dict = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100)
        }
    ) -> tuple:
    """
    Analyzes feature correlations and computes importance scores.

    Args:
        X_train (pd.DataFrame): Training dataset of independent features.
        corr_threshold (float): Threshold for considering highly correlated features.
        selected_cols_rfe (list): List of selected columns from Recursive Feature Elimination (RFE).
        selected_cols_pca (list): List of selected columns from Principal Component Analysis (PCA).
        selected_cols_mifs (list): List of selected columns from Mutual Information Feature Selection (MIFS).
        models (dict): Dictionary of models with model names as keys and model objects as values.

    Returns:
        tuple: A tuple containing the following:
            - total_features (list): List of total features (selected_cols_rfe + selected_cols_pca + selected_cols_mifs).
            - importance_scores (pd.Series or np.array): Feature importance scores or permutation importance scores.
            - pairwise_features_in_total_features (list): List of pairwise features that are in total_features.
    """
    # Create correlation matrix from training dataset of independent features
    corr_matrix = X_train.corr()

    # Get the indices of the highly correlated features
    indices = np.where(abs(corr_matrix) >= corr_threshold)

    # Create a dataframe with the pairwise correlation values above the threshold
    df_pairwise = pd.DataFrame({
        'feature1': corr_matrix.columns[indices[0]],
        'feature2': corr_matrix.columns[indices[1]],
        'correlation': corr_matrix.values[indices]
    })

    # Sort feature pairs and drop duplicates
    df_pairwise = df_pairwise.assign(sorted_features=df_pairwise[['feature1', 'feature2']].apply(sorted, axis=1).apply(tuple))
    df_pairwise = df_pairwise.loc[df_pairwise['feature1'] != df_pairwise['feature2']].drop_duplicates(subset='sorted_features').drop(columns='sorted_features')

    # Sort by correlation and format output
    df_pairwise = df_pairwise.sort_values(by='correlation', ascending=False).reset_index(drop=True)
    df_pairwise['correlation'] = (df_pairwise['correlation'] * 100).apply('{:.2f}%'.format)

    # Find pairs in total_features
    total_features = np.unique(selected_cols_rfe + selected_cols_pca + selected_cols_mifs).tolist()

    pairwise_features = list(df_pairwise[['feature1', 'feature2']].itertuples(index=False, name=None))
    pairwise_features_in_total_features = [pair for pair in pairwise_features if pair[0] in total_features and pair[1] in total_features]

    # Create estimator based on user-selected model
    estimator = models[selected_corr_model]
    estimator.fit(X_train, y_train)
    
    # Compute feature importance scores or permutation importance scores
    importance_scores = compute_importance_scores(X_train, y_train, estimator)
    
    return total_features, importance_scores, pairwise_features_in_total_features, df_pairwise   

def remove_lowest_importance_feature(total_features, importance_scores, pairwise_features_in_total_features):
    lowest_importance_features = []
    for feature1, feature2 in pairwise_features_in_total_features:
        if feature1 in total_features and feature2 in total_features:
            score1 = importance_scores[feature1]
            score2 = importance_scores[feature2]
            if score1 < score2:
                lowest_importance_features.append(feature1)
            else:
                lowest_importance_features.append(feature2)
    updated_total_features = [feature for feature in total_features if feature not in lowest_importance_features]
    return updated_total_features
            
def show_rfe_plot(rfecv, selected_features):
    """
    Show the Recursive Feature Elimination (RFE) plot and feature rankings in Streamlit.

    Parameters:
        rfecv (sklearn.feature_selection.RFECV): The fitted RFECV model.
        selected_features (list): The list of selected feature/column names.

    Returns:
        None

    """
    #############################################################
    # Scatterplot the results
    #############################################################
    # Get the feature ranking
    feature_rankings = pd.Series(rfecv.ranking_, index=X_train.columns).rename('Ranking')
    # Sort the feature rankings in descending order
    sorted_rankings = feature_rankings.sort_values(ascending=True)
    # Create a dataframe with feature rankings and selected features
    df_ranking = pd.DataFrame({'Features': sorted_rankings.index, 'Ranking': sorted_rankings})
    # Sort the dataframe by ranking
    df_ranking = df_ranking.sort_values('Ranking', ascending=True)
    # Highlight selected features
    df_ranking['Selected'] = np.where(df_ranking['Features'].isin(selected_features), 'Yes', 'No')
    
    # Create the rfe plot
    vertical_spacer(2)
    fig = create_rfe_plot(df_ranking)
    # Show the plot
    st.plotly_chart(fig, use_container_width=True)                       
    # show the ranking and selected features dataframes side by side
    col1, col2, col3, col4 = st.columns([1,2,2,1])
    with col2:
        st.write(':blue[**Selected features:**]', selected_features)
    # Print the feature rankings
    with col3: 
        feature_rankings = pd.Series(rfecv.ranking_, index=X.columns).rename('Ranking')
        st.write(':blue[**Feature rankings:**]')
        st.write(feature_rankings.sort_values())
    
    #############################################################
    # Show in Streamlit the ranking
    #############################################################
    selected_cols_rfe = list(selected_features)
    st.info(f'Top {len(selected_cols_rfe)} features selected with RFECV: {selected_cols_rfe}')
    
    show_rfe_info_btn = st.button(f'About RFE plot', use_container_width=True, type='secondary')
    
    # if user clicks "About RFE plot" button, display information about RFE
    if show_rfe_info_btn:
        st.write('')
        # show user info about how to interpret the graph
        st.markdown('''
                    **Recursive Feature Elimination** involves recursively removing features and building a model on the remaining features. It then **ranks the features** based on their importance and **eliminates** the **least important feature**.
                    ''')

def perform_rfe(X_train, y_train, estimator_rfe, duplicate_ranks_rfe, num_steps_rfe, num_features_rfe, timeseriessplit_value_rfe):
    """
    Perform Recursive Feature Elimination with Cross-Validation and display the results using a scatter plot.

    Parameters:
        X_train (pandas.DataFrame): Training data features.
        y_train (pandas.Series): Training data target.
        est_rfe (estimator object): A supervised learning estimator with a `fit` method.
        num_steps_rfe (int): Number of features to remove at each iteration of RFE.
        duplicate_ranks_rfe (bool): allow duplicate ranks if set to True otherwise take first feature(s) from list when ordered by rank ascending
        num_features (int): Number of features to select, defaults to None.
        timeseriessplit_value_rfe (int): Number of splits in time series cross-validation.

    Returns:
        None
    """
    # Set up the estimator based on the user's selection
    if estimator_rfe == 'Linear Regression':
        est_rfe = LinearRegression()
    elif estimator_rfe == 'Random Forest Regression':
        est_rfe = RandomForestRegressor()
                  
    #############################################################
    # Recursive Feature Elemination
    #############################################################
    # Define the time series splits set by user in sidebar slider      
    tscv = TimeSeriesSplit(n_splits=timeseriessplit_value_rfe)
    
    # Set up the recursive feature elimination with cross validation
    rfecv = RFECV(estimator = est_rfe, 
                  step = num_steps_rfe, 
                  cv = tscv, 
                  scoring = 'neg_mean_squared_error',  #['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error'] accuracy
                  min_features_to_select = num_features_rfe,
                  n_jobs = -1)
    
    # Fit the feature selection model
    rfecv.fit(X_train, y_train)
    
    user_message = f"note: optimal number of features suggested = {rfecv.n_features_}"
        
    if num_features_rfe is not None:
        selected_features = X_train.columns[rfecv.ranking_ <= num_features_rfe]
    else:
        selected_features = X_train.columns[rfecv.support_]

    # if user selected to allow duplicate rankings (default = True) use all ranks
    # example: when two features or more have rank 1, keep both features and it overrides the selected number of features by user initially set
    if duplicate_ranks_rfe == True:
        selected_cols_rfe = list(selected_features)
    # if user doesn't want duplicate ranking just get the first n items in the list with m features whereby n <= m
    elif duplicate_ranks_rfe == False:
        # Get the feature ranking
        feature_rankings = pd.Series(rfecv.ranking_, index=X_train.columns).rename('Ranking')
        # Sort the feature rankings in descending order
        sorted_rankings = feature_rankings.sort_values(ascending=True)
        # Create a dataframe with feature rankings and selected features
        df_ranking = pd.DataFrame({'Features': sorted_rankings.index, 'Ranking': sorted_rankings})
        # Sort the dataframe by ranking
        df_ranking = df_ranking.sort_values('Ranking', ascending=True)
                
        # only keep top x features, so slice and reindex to 0,1,2,3
        selected_features = df_ranking.iloc[:num_features_rfe, 0].reset_index(drop=True)
        selected_cols_rfe = list(selected_features)
        
        #st.write('selected features', selected_features, selected_cols_rfe) # TEST

    return selected_cols_rfe, selected_features, rfecv, user_message

def perform_mifs(X_train, y_train, num_features_mifs):
    """
    Perform Mutual Information Feature Selection (MIFS) on the training data.

    Parameters:
        X_train (pandas.DataFrame): The training data.
        y_train (pandas.Series or numpy.ndarray): The target variable for training.
        num_features_mifs (int): The number of top features to select based on mutual information.

    Returns:
        tuple: A tuple containing the following:
            - mutual_info (numpy.ndarray): The array of mutual information values.
            - selected_features_mi (numpy.ndarray): The selected feature/column names based on mutual information.
            - selected_cols_mifs (list): The selected feature/column names as a list.

    """
    mutual_info = mutual_info_regression(X_train, y_train, random_state=42)
    selected_features_mi = X_train.columns[np.argsort(mutual_info)[::-1]][:num_features_mifs]
    selected_cols_mifs = list(selected_features_mi)
    return mutual_info, selected_features_mi, selected_cols_mifs

def show_mifs_plot(mutual_info, selected_features_mi, num_features_mifs):
    """
    Display the Mutual Information Feature Selection (MIFS) plot.

    Parameters:
        mutual_info (numpy.ndarray): The array of mutual information values.
        selected_features_mi (list): The list of selected features based on mutual information.
        num_features_mifs (int): The number of top features to display in the plot.

    Returns:
        None

    """
    vertical_spacer(2)
    my_text_paragraph(' Mutual Information', my_font_size='26px',)
    my_text_paragraph(f'<b> TOP {num_features_mifs} </b>', my_font_size='16px', my_font_family='Segui UI')
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mutual_info[np.argsort(mutual_info)[::-1]][:num_features_mifs],
        y=selected_features_mi,
        orientation='h',
        text=[f'{val:.2f}' for val in mutual_info[np.argsort(mutual_info)[::-1]][:num_features_mifs]],
        textposition='inside',
        marker_color='#4715EF'  # Set the desired color code
    ))
    
    fig.update_layout(
        title={
            'text': '',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        }, margin=dict(t=0, b=0),
        xaxis_title='Mutual Information Values',
        yaxis_title='Feature Name'
    )
    
    # Display plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    ##############################################################
    # SELECT YOUR FAVORITE FEATURES TO INCLUDE IN MODELING
    ##############################################################            
    st.info(f'Top {num_features_mifs} features selected with MIFS: {list(selected_features_mi)}')
    
    # create button to display information about mutual information feature selection
    show_mifs_info_btn = st.button(f'About MIFS plot', use_container_width=True, type='secondary')            
    
    if show_mifs_info_btn:
        st.write('')
        # show user info about how to interpret the graph
        st.markdown('''
            Mutual Information Feature Selection (MIFS) is a method for selecting the most important features in a dataset for predicting a target variable.  
            It measures the mutual information between each feature and the target variable, 
            using an entropy-based approach to quantify the amount of information that each feature provides about the target.  
            Features with high mutual information values are considered to be more important in predicting the target variable
            and features with low mutual information values are considered to be less important.  
            MIFS helps improve the accuracy of predictive models by identifying the most informative features to include in the model.
        ''')

def perform_pca(X_train, num_features_pca):
    """
    Perform Principal Component Analysis (PCA) on the training data.

    Parameters:
        X_train (pandas.DataFrame): The training data.
        num_features_pca (int): The desired number of components for PCA.

    Returns:
        list: The selected feature/column names based on PCA.

    """
    # Create a PCA object with the desired number of components
    pca = PCA(n_components = num_features_pca)
    # Fit the PCA model to the training data
    pca.fit(X_train)
    # Get the names of the features/columns in the training data
    feature_names = X_train.columns
    # Sort the features based on the explained variance ratio in descending order
    sorted_idx = np.argsort(pca.explained_variance_ratio_)[::-1]
    # Reorder the feature names based on the sorted indices
    sorted_features = feature_names[sorted_idx]
    # Convert the sorted features to a list
    selected_cols_pca = sorted_features.tolist()
    
    # =============================================================================
    #     # Get the explained variance for each feature
    #     variances = pca.explained_variance_
    #     st.write(variances)
    # =============================================================================
    return sorted_features, pca, sorted_idx, selected_cols_pca

def show_pca_plot(sorted_features, pca, sorted_idx, selected_cols_pca):
    """
    Display a PCA plot and information about the selected features.

    Parameters:
        - sorted_features (pandas.Series or list): A pandas Series or list containing the sorted feature names.
        - pca (sklearn.decomposition.PCA): The fitted PCA object.
        - sorted_idx (list or array-like): The sorted indices of the features.

    Returns:
        None
    """            
    vertical_spacer(2)
    my_text_paragraph(f'Principal Component Analysis', my_font_size='26px')
    my_text_paragraph(f'<b>TOP {len(sorted_features)}</b>', my_font_size='16px', my_font_family='Segui UI')

    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x= pca.explained_variance_ratio_[sorted_idx],
        y= sorted_features,
        orientation = 'h',
        text = np.round(pca.explained_variance_ratio_[sorted_idx] * 100, 2),
        textposition = 'auto',
        marker_color='#4715EF'
    ))
    fig.update_layout(
        title={
            'text': '',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(t=0, b=0), 
        xaxis_title='Explained Variance Ratio',
        yaxis_title='Feature Name'
    )
    
    # Display plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    st.info(f'Top {len(selected_cols_pca)} features selected with PCA: {selected_cols_pca}')

    # Show "About PCA plot" button
    show_pca_info_btn = st.button(f'About PCA plot', use_container_width=True, type='secondary')
    
    if show_pca_info_btn == True:
        vertical_spacer(1)
        # show user info about how to interpret the graph
        st.markdown('''When you fit a **PCA** model, it calculates the amount of variance that is captured by each principal component.
                    The variance ratio is the fraction of the total variance in the data that is explained by each principal component.
                    The sum of the variance ratios of all the principal components equals 1.
                    The variance ratio is expressed as a percentage by multiplying it by 100, so it can be easily interpreted.  
                    ''')
        st.markdown('''
                    For example, a variance ratio of 0.75 means that 75% of the total variance in the data is captured by the corresponding principal component.
                    ''')
                    
def show_model_inputs():
    """
    Displays a set of buttons and corresponding dataframes based on user clicks.
    
    This function shows a set of buttons for different datasets and features (X, y).
    When a button is clicked, the corresponding dataframe is displayed below.
    
    Returns:
        None
    """
    my_text_header('Model Inputs')
    
    # =============================================================================
    #     show_lottie_animation(url="./images/86093-data-fork.json", key='inputs')
    # =============================================================================

    col1, col2, col3 = st.columns([7,120,1])
    with col2:  
        
        create_flipcard_model_input(image_path_front_card='./images/model_inputs.png', 
                                    my_string_back_card = 'Your dataset is split into two parts: a <i style="color:#67d0c4;">train dataset </i> and a <i style="color:#67d0c4;">test dataset</i>. The former is used to train the model, allowing it to learn from the features by adjusting its parameters to minimize errors and improve its predictive abilities. The latter dataset is then used to assess the performance of the trained model on new or unseen data and its ability to make accurate predictions.')
    vertical_spacer(1)
    
    col1, col2, col3 = st.columns([25, 40, 20])
    with col2:
        color_train = '#07080d'  
        color_test = '#67d0c4'  
        button_container = """
            <div style="display: flex; flex-direction: column; align-items: center; padding: 10px; margin-top: -30px;">
                <div style="display: flex; width: 100%; max-width: 400px;">
                    <div style="flex: 1; background: {color_train}; border-radius: 20px 0 0 20px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2), 0px 1px 3px rgba(0, 0, 0, 0.1);">
                        <a href='#' id='train' style="display: flex; justify-content: center; align-items: center; height: 100%; text-decoration: none; background: {color_train}; border-radius: 5px; padding: 10px;">
                            <span style="color: #FFFFFF; font-size: 14px;">Train</span>
                        </a>
                    </div>
                    <div style="flex: 1; background: {color_train}; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2), 0px 1px 3px rgba(0, 0, 0, 0.1);">
                        <a href='#' id='X_train' style="display: flex; justify-content: center; align-items: center; height: 100%; text-decoration: none; background: {color_train}; padding: 10px;">
                            <span style="color: #FFFFFF; font-size: 14px;">Train X</span>
                        </a>
                    </div>
                    <div style="flex: 1; background: {color_train}; border-radius: 0 20px 20px 0; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2), 0px 1px 3px rgba(0, 0, 0, 0.1);">
                        <a href='#' id='y_train' style="display: flex; justify-content: center; align-items: center; height: 100%; text-decoration: none; background: {color_train}; border-radius: 5px; padding: 10px;">
                            <span style="color: #FFFFFF; font-size: 14px;">Train y</span>
                        </a>
                    </div>
                </div>
                <div style="display: flex; width: 100%; max-width: 400px; margin-top: 10px;">
                    <div style="flex: 1; background: {color_test}; border-radius: 20px 0 0 20px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2), 0px 1px 3px rgba(0, 0, 0, 0.1);">
                        <a href='#' id='test' style="display: flex; justify-content: center; align-items: center; height: 100%; text-decoration: none; background: {color_test}; border-radius: 5px; padding: 10px;">
                            <span style="color: #FFFFFF; font-size: 14px;">Test</span>
                        </a>
                    </div>
                    <div style="flex: 1; background: {color_test}; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2), 0px 1px 3px rgba(0, 0, 0, 0.1);">
                        <a href='#' id='X_test' style="display: flex; justify-content: center; align-items: center; height: 100%; text-decoration: none; background: {color_test}; border-radius: 5px; padding: 10px;">
                            <span style="color: #FFFFFF; font-size: 14px;">Test X</span>
                        </a>
                    </div>
                    <div style="flex: 1; background: {color_test}; border-radius: 0 20px 20px 0; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2), 0px 1px 3px rgba(0, 0, 0, 0.1);">
                        <a href='#' id='y_test' style="display: flex; justify-content: center; align-items: center; height: 100%; text-decoration: none; background: {color_test}; border-radius: 5px;; padding: 10px;">
                            <span style="color: #FFFFFF; font-size: 14px;">Test y</span>
                        </a>
                    </div>
                </div>
                <div style="display: flex; width: 100%; max-width: 400px; margin-top: 10px;">
                    <div style="flex: 1; background: {color_train}; border-radius: 0 0 30px 30px; box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.1), 0px 1px 3px rgba(0, 0, 0, 0.1);">
                        <a href='#' id='close_btn' style="display: flex; justify-content: center; align-items: center; height: 100%; text-decoration: none; background: ; border-radius: 50%; padding: 10px; margin-bottom: -20px;">
                            <span style="color: #FFFFFF; font-size: 14px;">&times;</span>
                        </a>
                    </div>
                </div>
            </div>
        """.format(color_train=color_train, color_test=color_test)
   
        # Call the click_detector function with the HTML content
        clicked = click_detector(button_container)

    if clicked == "train":
        my_text_paragraph('Training Dataset', my_font_size='28px')
        vertical_spacer(1)
        st.dataframe(st.session_state['df'].iloc[:-st.session_state['insample_forecast_steps'], :], use_container_width=True)
    elif clicked == "test":
        my_text_paragraph('Test Dataset', my_font_size='28px')
        vertical_spacer(1)
        # get the length of test-set based on insample_forecast_steps
        test_range = len(st.session_state['df']) - st.session_state['insample_forecast_steps']
        st.dataframe(st.session_state['df'].iloc[test_range:, :], use_container_width=True)
    elif clicked == "X_train":
        my_text_paragraph('Explanatory Variables (X)', my_font_size='28px')
        vertical_spacer(1)
        st.dataframe(X_train, use_container_width=True)
    elif clicked == "y_train":
        my_text_paragraph('Target Variable (y)', my_font_size='28px')
        vertical_spacer(1)
        st.dataframe(y_train, use_container_width=True)
    elif clicked == "X_test":
        my_text_paragraph('Explanatory Variables (X)', my_font_size='28px')
        vertical_spacer(1)
        st.dataframe(X_test, use_container_width=True)
    elif clicked == "y_test":
        my_text_paragraph('Target Variable (y)', my_font_size='28px')
        vertical_spacer(1)
        st.dataframe(y_test, use_container_width=True)
    elif clicked == "close_btn":
        pass
    else:
        pass
    
def hist_change_freq():
    try:
        # update in memory value for radio button / save user choice of histogram freq
        index_freq_type = ("Relative" if frequency_type == "Absolute" else "Absolute")
        set_state("HIST", ("histogram_freq_type", index_freq_type))
    except:
        set_state("HIST", ("histogram_freq_type", "Absolute"))
        
def stock_ticker(text, speed=15):
    """
    Displays a right-to-left scrolling text e.g. ticker animation in a Markdown format using the `st.markdown` function from the Streamlit library.

    Parameters:
        text (str): The text to be displayed in the ticker.
        speed (int, optional): The speed of the ticker animation in seconds. Default is 15.

    Returns:
        None

    Example:
        stock_ticker("Stock ABC", speed=10)
    """
    st.markdown(
        f"""
        <style>
        .ticker-outer-container {{
            width: 100%;
            overflow: hidden;
        }}

        .ticker-container {{
            display: flex;
            white-space: nowrap;
            animation: ticker-animation {speed}s linear infinite;
        }}

        .ticker-item {{
            display: inline-block;
        }}

        @keyframes ticker-animation {{
            0% {{
                transform: translateX(99%); /*text is positioned to the right and initially not visible. At 100%*/
            }}
            100% {{
                transform: translateX(-400%); /*which means the text is moved to the left by 200% of its container's width*/
            }}
        }}
        </style>
        <div class="ticker-outer-container">
            <div class="ticker-container">
                <div class="ticker-item">{text} |</div>
                <div class="ticker-item">{text} |</div>
                <div class="ticker-item">{text} |</div>
                <div class="ticker-item">{text} |</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def show_lottie_animation(url, key, reverse=False, height=400, width=400, speed=1, loop=True, quality='high', col_sizes=[1, 3, 1], margin_before = 0, margin_after = 0):
    with open(url, "r") as file:
        animation_url = json.load(file)

    col1, col2, col3 = st.columns(col_sizes)
    with col2:
        vertical_spacer(margin_before)
        
        st_lottie(animation_url,
                  reverse=reverse,
                  height=height,
                  width=width,
                  speed=speed,
                  loop=loop,
                  quality=quality,
                  key=key
                  )
        vertical_spacer(margin_after)

def vertical_spacer(n):
    for i in range(n):
        st.write("")
# =============================================================================
# 
# def eda_quick_insights(df, my_string_column, my_chart_color):
#     col1, col2, col3 = st.columns([20,40,20])
#     with col2:
#         my_text_header('Quick Insights')
#         vertical_spacer(1)
#         
#     #col1, col2, col3 = st.columns([20, 80, 20])
#     col1, col2, col3 = st.columns([5, 80, 5])
#     with col2:
#         # Filter out NaN and '-' values from 'Label' column
#         label_values = df[my_string_column].dropna().apply(lambda x: x.strip()).replace('-', '').tolist()
#         # Filter out any remaining '-' values from 'Label' column
#         label_values = [value for value in label_values if value != '']
#         # Create an HTML unordered list with each non-NaN and non-'-' value as a list item
#         html_list = "<div class='my-list'>"
#         for i, value in enumerate(label_values):
#             html_list += f"<li><span class='my-number'>{i + 1}</span>{value}</li>"
#         html_list += "</div>"
#         # Display the HTML list using Streamlit
#         st.markdown(
#             f"""
#             <style>
#                 .my-list {{
#                     font-size: 16px;
#                     color: black; /* Add your desired font color here */
#                     line-height: 1.4;
#                     margin-bottom: 10px;
#                     margin-left: 0px;
#                     margin-right: 0px;
#                     background-color: white;
#                     border-radius: 10px;
#                     box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.3);
#                     padding: 20px;
#                     
#                 }}
#                 .my-list li {{
#                     margin: 10px 10px 10px 10px;
#                     padding-left: 30px;
#                     position: relative;
#                 }}
#                 .my-number {{
#                     font-weight: bold;
#                     color: white;
#                     background-color: {my_chart_color};
#                     border-radius: 50%;
#                     text-align: center;
#                     width: 20px;
#                     height: 20px;
#                     line-height: 20px;
#                     display: inline-block;
#                     position: absolute;
#                     left: 0;
#                     top: 0;
#                 }}
#             </style>
#             {html_list}
#             """,
#             unsafe_allow_html=True
#         )
#         # vertical spacer
#         vertical_spacer(1)
# =============================================================================

def create_flipcard_quick_insights(num_cards, header_list, paragraph_list_front, paragraph_list_back, font_family, font_size_front, font_size_back, image_path_front_card = None, df = None, my_string_column = 'Label', **kwargs):    
    
    col1, col2, col3 = st.columns([20,40,20])
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

def create_flipcard_quick_summary(header_list, paragraph_list_front, paragraph_list_back, font_family, font_size_front, font_size_back, image_path_front_card = None, **kwargs):    
   
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
 
def create_flipcard_model_input(image_path_front_card=None, font_size_back='10px', my_string_back_card = '', **kwargs):

    # Open the image for the front of the card
    with open(image_path_front_card, 'rb') as file:
        contents = file.read()
        data_url = base64.b64encode(contents).decode("utf-8")

    # Create empty list that will keep the HTML code needed for each card with header+text
    card_html = []

    # Append HTML code to list
    card_html.append(f"""
        <div class="flashcard_model_input">
            <div class='front_model_input'>
                <img src="data:image/png;base64,{data_url}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 10px;">
            </div>
            <div class="back_model_input" style="margin-top: 20px; display: flex; align-items: center;">
                <!-- Add your text for the back of the card here -->
                <h3> {my_string_back_card} </h3>
            </div>
        </div>
    """)
        
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
        .flashcard_model_input {{
          width: 600px;
          height: 600px;
          background-color: white;
          border-radius: 10px;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
          perspective: 100px;
          margin-bottom: 10px;
          scroll-snap-align: center;
        }}
        .front_model_input, .back_model_input {{
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
        .front_model_input {{
          /* background: linear-gradient(to bottom left, #4e3fce, #7a5dc7, #9b7cc2, #bb9bbd, #c2b1c4); */ 
          color: white;
          transform: rotateY(0deg);
        }}
        .back_model_input {{
            color: #333333;
            background: white;
            transform: rotateY(180deg);
            display: flex;
            justify-content: flex-start;
            border: none;
            margin: 0;
            padding: 60px;
            text-align: left;
            text-justify: inter-word;
            overflow: auto;
            position: relative;
            /* border-radius: 10px; */
            /* border: 10px solid black; */
        }}
        .back_model_input h6 {{
          margin-bottom: 0px;
          margin-top: 0px;
        }}
        .flashcard_model_input:hover .front_model_input {{
          transform: rotateY(180deg);
        }}
        .flashcard_model_input:hover .back_model_input {{
          transform: rotateY(0deg);
        }}
        .back_model_input p {{
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

    # Join the card HTML code list and display the carousel in Streamlit
    st.markdown("".join(card_html), unsafe_allow_html=True)
 
def create_flipcard_stats(image_path_front_card=None, font_size_back='10px', my_header=None, **kwargs):
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

#################################
# FORMATTING DATAFRAMES FUNCTIONS
#################################
def highlight_cols(s):
    """
    A function that highlights the cells of a DataFrame based on their values.

    Args:
    s (pd.Series): A Pandas Series object representing the columns of a DataFrame.

    Returns:
    list: A list of CSS styles to be applied to each cell in the input Series object.
    """
    if isinstance(s, pd.Series):
        if s.name == outliers_df.columns[0]:
            return ['background-color: lavender']*len(s)
        elif s.name == outliers_df.columns[1]:
            return ['background-color: lightyellow']*len(s)
        else:
            return ['']*len(s)
    else:
        return ['']*len(s)
    
############################
# FORMATTING TEXT FUNCTIONS
############################
def my_title(my_string, my_background_color="#my_title", gradient_colors=None):
    if gradient_colors is None:
        gradient_colors = f"{my_background_color}, #2CB8A1, #0072B2"
    gradient = f"-webkit-linear-gradient(45deg, {gradient_colors})"
    st.markdown(f'<h3 style="background: none; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-image: {gradient}; padding:10px; border-radius: 10px; border: 1px solid {my_background_color}; box-shadow: 0px 4px 4px rgba(0, 0, 0, 0.25);"> <center> {my_string} </center> </h3>', unsafe_allow_html=True)

def my_header(my_string, my_style="#217CD0"):
    st.markdown(f'<h2 style="color:{my_style};"> <center> {my_string} </center> </h2>', unsafe_allow_html=True)


def my_subheader(my_string, my_background_colors=["#45B8AC", "#2CB8A1", "#0072B2"], my_style="#FFFFFF", my_size=3):
    gradient = f"-webkit-linear-gradient(45deg, {', '.join(my_background_colors)})"
    st.markdown(f'<h{my_size} style="background: none; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-image: {gradient}; font-family: sans-serif; font-weight: bold; text-align: center; color: {my_style};"> {my_string} </h{my_size}>', unsafe_allow_html=True)

def my_text_header(my_string,
                   my_text_align='center', 
                   my_font_family='Segoe UI, Tahoma, Geneva, Verdana, sans-serif',
                   my_font_weight=200,
                   my_font_size='36px',
                   my_line_height=1.5):
    text_header = f'<h1 style="text-align:{my_text_align}; font-family: {my_font_family}; font-weight: {my_font_weight}; font-size: {my_font_size}; line-height: {my_line_height};">{my_string}</h1>'
    st.markdown(text_header, unsafe_allow_html=True)
    
def my_text_paragraph(my_string,
                       my_text_align='center',
                       my_font_family='Segoe UI, Tahoma, Geneva, Verdana, sans-serif',
                       my_font_weight=200,
                       my_font_size='18px',
                       my_line_height=1.5,
                       add_border=False,
                       border_color = "#45B8AC"):
    if add_border:
        border_style = f'border: 2px solid {border_color}; border-radius: 10px; padding: 10px; box-shadow: 0px 4px 4px rgba(0, 0, 0, 0.25);'
    else:
        border_style = ''
    paragraph = f'<p style="text-align:{my_text_align}; font-family:{my_font_family}; font-weight:{my_font_weight}; font-size:{my_font_size}; line-height:{my_line_height}; background-color: rgba(255, 255, 255, 0); {border_style}">{my_string}</p>'
    st.markdown(paragraph, unsafe_allow_html=True)

def my_bubbles(my_string, my_background_color="#2CB8A1"):
    gradient = f"-webkit-linear-gradient(45deg, {my_background_color}, #2CB8A1, #0072B2)"
    text_style = f"text-align: center; font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif; font-weight: 200; font-size: 36px; line-height: 1.5; -webkit-background-clip: text; -webkit-text-fill-color: black; padding: 20px; position: relative;"
    st.markdown(f'''
        <div style="position: relative;">
            <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; z-index: -1; opacity: 0.2;"></div>
            <h1 style="{text_style}">
                <center>{my_string}</center>
                <div style="position: absolute; top: 20px; left: 80px;">
                    <div style="background-color: #0072B2; width: 8px; height: 8px; border-radius: 50%; animation: bubble 3s infinite;"></div>
                </div>
                <div style="position: absolute; top: -20px; right: 100px;">
                    <div style="background-color: #FF0000; width: 14px; height: 14px; border-radius: 50%; animation: bubble 4s infinite;"></div>
                </div>
                <div style="position: absolute; top: 10px; right: 50px;">
                    <div style="background-color: #0072B2; width: 8px; height: 8px; border-radius: 50%; animation: bubble 5s infinite;"></div>
                </div>
                <div style="position: absolute; top: -20px; left: 60px;">
                    <div style="background-color: #88466D; width: 8px; height: 8px; border-radius: 50%; animation: bubble 6s infinite;"></div>
                </div>
                <div style="position: absolute; top: 0px; left: -10px;">
                    <div style="background-color: #2CB8A1; width: 12px; height: 12px; border-radius: 50%; animation: bubble 7s infinite;"></div>
                </div>
                <div style="position: absolute; top: 10px; right: -20px;">
                    <div style="background-color: #7B52AB; width: 10px; height: 10px; border-radius: 50%; animation: bubble 10s infinite;"></div>
                </div>
                <div style="position: absolute; top: -20px; left: 150px;">
                    <div style="background-color: #FF9F00; width: 8px; height: 8px; border-radius: 50%; animation: bubble 20s infinite;"></div>
                </div>
                <div style="position: absolute; top: 25px; right: 170px;">
                    <div style="background-color: #FF6F61; width: 12px; height: 12px; border-radius: 50%; animation: bubble 4s infinite;"></div>
                </div>
                <div style="position: absolute; top: -30px; right: 120px;">
                <div style="background-color: #440154; width: 10px; height: 10px; border-radius: 50%; animation: bubble 5s infinite;"></div>
                </div>
                <div style="position: absolute; top: -20px; left: 150px;">
                <div style="background-color: #2CB8A1; width: 8px; height: 8px; border-radius: 50%; animation: bubble 6s infinite;"></div>
                </div>
                <div style="position: absolute; top: -10px; right: 20px;">
                <div style="background-color: #FFC300; width: 12px; height: 12px; border-radius: 50%; animation: bubble 7s infinite;"></div>
                </div>
                </h1>
                <style>
                @keyframes bubble {{
                0% {{
                transform: translateY(0);
                }}
                50% {{
                transform: translateY(+50px);
                }}
                100% {{
                transform: translateY(0);
                }}
                }}
                .bubble-container div {{
                margin: 10px;
                }}
                </style>
                </div>
                ''', unsafe_allow_html=True)
                
def train_models_carousel(my_title= ''):
    # gradient title
    vertical_spacer(2)

    title = my_title
    
# =============================================================================
#     # set gradient color of letters of title
#     gradient = '-webkit-linear-gradient(left, #0072B2, #673ab7, #3f51b5, #2196f3, #03a9f4)'
# =============================================================================
    col1, col2, col3 = st.columns([1,8,1])
    with col2:
        # show in streamlit the title with gradient
        st.markdown(f'<h2 style="text-align:center; font-family: Rock Salt; color: black;"> {title} </h2>', unsafe_allow_html=True)

    vertical_spacer(2)

    # show carousel of models   
    paragraph_list_back = ['The <b> Naive models </b> serve as a simple baseline or benchmark. Model I assumes the next observation is equal to a lagged previous value which can be either previous day, week, month, quarter, year or custom defined. Model II aggregates a set of past values using a rolling window to predict the next value. Model III has a fixed single value by taking the average, median or mode.', 
                           'The <b> Linear Regression Model </b> is a statistical technique used to analyze the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship, aiming to find the best-fit line that minimizes the differences between observed and predicted values.', 
                           '<b>SARIMAX</b>, short for <b>Seasonal Autoregressive Integrated Moving Average with Exogenous Variables</b>, is a powerful time series forecasting model that incorporates seasonal patterns and exogenous variables. It combines <i> autoregressive </i> (past values), <i> moving average </i> (averages of certain time spans), and <i> integrated </i> (calculating differences of subsequent values) components.', 
                           '<b>Prophet</b> utilizes an additive model (sum of individual factors) that decomposes time series data into: <i>trend</i>, <i>seasonality</i>, and <i>holiday components</i>. It incorporates advanced statistical techniques and automatic detection of changepoints to handle irregularities in the data. It offers flexibility in handling missing data and outliers making it a powerful forecasting model.']
    
    # create carousel cards for each model
    header_list = ['Naive Models', 'Linear Regression', 'SARIMAX', 'Prophet']    
    paragraph_list_front = ['', '', '', '']
   
    # define the font family to display the text of paragraph
    font_family = 'Ysabeau SC'
    #font_family = 'Rubik Dirt'
    
    # define the paragraph text size
    font_size_front = '14px'
    font_size_back = '15px'       
    
    # apply carousel function to show 'flashcards'
    create_carousel_cards_v2(4, header_list, paragraph_list_front, paragraph_list_back, font_family, font_size_front, font_size_back)
    vertical_spacer(2)
                        
def create_carousel_cards(num_cards, header_list, paragraph_list, font_family, font_size):
    # note: #purple gradient: background: linear-gradient(to bottom right, #F08A5D, #FABA63, #2E9CCA, #4FB99F, #dababd);
    # create empty list that will keep the html code needed for each card with header+text
    card_html = []
    # iterate over cards specified by user and join the headers and text of the lists
    for i in range(num_cards):
        card_html.append(f"<div class='card'><h1 style='text-align:center;color:white; margin-bottom: 10px;'>{header_list[i]}</h1><p style='text-align:center; font-family: {font_family}; font-size: {font_size};'>{paragraph_list[i]}</p></div>")
    # join all the html code for each card and join it into single html code with carousel wrapper
    carousel_html = "<div class='carousel'>" + "".join(card_html) + "</div>"
    # Display the carousel in streamlit
    st.markdown(carousel_html, unsafe_allow_html=True)
    # Create the CSS styling for the carousel
    st.markdown(
        """
        <style>
        /* Carousel Styling */
        .carousel {
          display: flex;
          overflow-x: auto;
          scroll-snap-type: x mandatory;
          scroll-behavior: smooth;
          -webkit-overflow-scrolling: touch;
          width: 80%;
          margin: auto;
        }
        .card {
          display: inline-block;
          margin-right: 10px;
          vertical-align: top;
          scroll-snap-align: center;
          flex: 0 0 auto;
          width: 75%;
          height: 200px;
          margin: 30px;
          padding: 20px;
          background: linear-gradient(to bottom left, #F08A5D, #FABA63);
                                      #  '-webkit-linear-gradient(left, #F08A5D, #FABA63, #2E9CCA, #4FB99F)'
          box-shadow: 10px 10px 10px 0px rgba(0, 0, 0, 0.1);
          border-radius: 20px;
          text-align: center;
          font-family: """ + font_family + """, sans-serif;
          font-size: """ + str(font_size) + """px;
          color: white;
          transition: transform 0.2s ease-in-out;
        }
        .card:hover {
          transform: scale(1.1);
          box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.5);
        }
        /* Carousel Navigation Styling */
        .carousel-nav {
          margin: 10px 0px;
          text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
        
# this is the independent cards with purple gradient version
def create_carousel_cards_v3(num_cards, header_list, paragraph_list_front, paragraph_list_back, font_family, font_size_front, font_size_back):
    # note removing display: flex; inside the css code for .flashcard -> puts cards below eachother
    # create empty list that will keep the html code needed for each card with header+text
    card_html = []
    # iterate over cards specified by user and join the headers and text of the lists
    for i in range(num_cards):
        card_html.append(f"""<div class="flashcard">                     
                                <div class='front'>
                                    <h1 style='text-align:center;color:white; margin-bottom: 10px;padding: 35px;'>{header_list[i]}</h1>
                                    <p style='text-align:center; font-family: {font_family}; font-size: {font_size_front};'>{paragraph_list_front[i]}</p>
                                </div>
                                <div class="back">
                                    <p style=font-family: {font_family}; font-size: {font_size_back};'>{paragraph_list_back[i]}</p>
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
          grid-gap: 10px;
          overflow-x: auto;
          scroll-snap-type: x mandatory;
          scroll-behavior: smooth;
          -webkit-overflow-scrolling: touch;
          width: 100%;
          margin: auto;
          border: 1px solid #000000; 
          border-radius: 10px;
        }}
       .flashcard {{
          display: inline-block; /* Display cards inline */
          width: 400px;
          height: 200px;
          background-color: white;
          border-radius: 10px;
          perspective: 100px;
          margin-bottom: 5px; /* Add space between cards */
          padding: 0px;
          scroll-snap-align: center;
        }}
        .front, .back {{
          position: absolute;
          top: 0;
          left: 0;
          width: 400px;
          height: 200px;
          border-radius: 10px;
          backface-visibility: hidden;
          font-family: {font_family};
          text-align: justify;
        }}
        .front {{
          background: linear-gradient(to bottom, #F5F5F5, #F5F5F5);
          color: black; /* text color */
          transform: rotateY(0deg);
        }}
        .back {{   
          background-color: #F5F5F5; /* Set the background color to off-white */
          color: black;
        transform: rotateY(180deg);
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        /* border: 1px solid #000000; */ /* Add 1px border with black color */ 
        }}                                
        .flashcard:hover .front {{
          transform: rotateY(180deg);
        }}
        .flashcard:hover .back {{
          transform: rotateY(0deg);
        }}
        .front h2, .back h2 {{
          color: white;
          text-align: center;
          margin-top: 10%;
          transform: translateY(-10%);
          font-family: {font_family};
          font-size: {font_size_front}px;
        }}
        .front h2 {{
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

# v2 with front and back of a flashcard as well with text on both sides and white background and gradient letters
def create_carousel_cards_v2(num_cards, header_list, paragraph_list_front, paragraph_list_back, font_family, font_size_front, font_size_back):
    # note removing display: flex; inside the css code for .flashcard -> puts cards below eachother
    # create empty list that will keep the html code needed for each card with header+text
    card_html = []
    # iterate over cards specified by user and join the headers and text of the lists
    for i in range(num_cards):
        card_html.append(f"""<div class="flashcard">                     
                                <div class='front'>
                                    <h1 style='text-align: center; color: #62e1d3; margin-top: 10px; margin-bottom: -10px; padding: 35px;'>{header_list[i]}</h1>
                                    <p style='text-align:center; font-family: {font_family}; font-size: {font_size_front};'>{paragraph_list_front[i]}</p>
                                </div>
                                <div class="back">
                                    <p style='text-align:justify;  word-spacing: 1px; font-family: {font_family}; font-size: {font_size_back};'>{paragraph_list_back[i]}</p>
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
          grid-gap: 10px;
          justify-content: center; /* Center horizontally */
          align-items: center; /* Center vertically */
          overflow-x: auto;
          scroll-snap-type: x mandatory;
          scroll-behavior: smooth;
          -webkit-overflow-scrolling: touch;
          width: 400px;
          margin: 0 0; /* Center horizontally by setting left and right margins to auto */
          background-color: white; /* Add black background */
          padding: 0px; /* Add padding for the black background */
          border-radius: 10px; /* Add border-radius for rounded edges */
          box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2), 0px -4px 8px rgba(0, 0, 0, 0.1); /* Add box shadow */
        }}
       .flashcard {{
          display: inline-block; /* Display cards inline */
          width: 400px;
          height: 200px;
          background-color: white;
          border-radius: 5px;
          perspective: 100px;
          margin-bottom: 0px; /* Add space between cards */
          padding: 0px;
          scroll-snap-align: center;
        }}
        .front, .back {{
          position: absolute;
          top: 0;
          left: 0;
          width: 400px;
          height: 200px;
          border-radius: 8px;
          backface-visibility: hidden;
          font-family: {font_family};
          text-align: center;
        }}
        .front {{
          background: white;
          color: black; /* Make the text transparent */
          transform: rotateY(0deg);
          border: 4px;
          border-color: white;
        }}
        .back {{
            /* ... other styles ... */
            background: black;
            color: transparent; /* Make the text transparent */
            background-clip: text; /* Apply the background gradient to the text */
            -webkit-background-clip: text; /* For Safari */
            -webkit-text-fill-color: transparent; /* For Safari */
            background-image: linear-gradient(to bottom left, #000000); /* linear-gradient(to bottom left, #941c8e, #763a9a, #4e62a3, #2e81ad, #12a9b4); */ /* Set the background gradient */
            transform: rotateY(180deg);
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }}                               
        .flashcard:hover .front {{
          transform: rotateY(180deg);
        }}
        .flashcard:hover .back {{
          transform: rotateY(0deg);
          cursor: default; /* Change cursor to pointer on hover */
        }}
        .front h2, .back h2 {{
          color: black;
          text-align: center;
          margin-top: 10%;
          transform: translateY(-10%);
          font-family: {font_family};
          font-size: {font_size_front}px;
        }}
        .front h2 {{
          padding-top: 10px;
          line-height: 1.5;
        }}
        .back h2 {{
          line-height: 2;
        }}
        .back p {{
          margin: 24px; /* Add margin for paragraph text */
          }}
        /* Carousel Navigation Styling */
        .carousel-nav {{
          margin: 10px 0px;
          text-align: center;
        }}
        </style>
        """, unsafe_allow_html=True)
 
#******************************************************************************
# STATISTICAL TEST FUNCTIONS
#******************************************************************************
################################################
# SUMMARY STATISTICS DESCRIPTIVE LABEL FUNCTIONS
################################################
def eda_quick_summary(my_chart_color):
    """
    Displays a quick summary of the exploratory data analysis (EDA) metrics.
    
    This function calculates and displays various metrics based on the provided DataFrame `df_raw`. The metrics include:
    - Number of rows
    - Minimum date
    - Percentage of missing data
    - Mean of the second column
    - Number of columns
    - Maximum date
    - Frequency of time-series data
    - Median of the first column
    - Minimum of the fourth column
    
    The metrics are displayed in a visually appealing format using Streamlit columns and CSS styling.
    
    Parameters:
    None
    
    Returns:
    None
    """
    # Define CSS style for the metrics container
    container_style = '''
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0px 0px 15px -5px rgba(0,0,0,0.3);
        padding: 25px;
        margin: 10px;
        width: 100%;
        max-width: 700px;
        display: flex; /* add this line to enable flexbox */
        justify-content: center; /* add this line to horizontally center-align the metrics */
        align-items: center; /* add this line to vertically center-align the metrics */
        text-align: center; /* add this line to center-align the metrics */
        font-family: Arial;
        font-size: 16px;
        color: #555555;
    '''
    try:        
        col1, col2, col3 = st.columns([19, 80, 20])
        with col2:
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
            st.write(
                    f'<div style="{container_style}">'
                    f'<div style="display: flex; justify-content: space-between; margin-bottom: -20px;margin-left: 90px; margin-right: 0px; margin-top: -10px;">'
                    f'<div style="text-align: center; margin-right: 50px;">'
                    f'<div><b style="color: {my_chart_color};">Rows</b></div><div>{rows}</div><br/>'
                    f'<div><b style="color: {my_chart_color};">Start Date</b></div><div>{min_date}</div><br/>'
                    f'<div><b style="color: {my_chart_color};">Missing</b></div><div>{percent_missing}</div><br/>'
                    f'<div><b style="color: {my_chart_color};">Mean</b></div><div>{mean_val}</div><br/>'
                    f'<div><b style="color: {my_chart_color};">Minimum</b></div><div>{min_val}</div><br/>'
                    f'<div><b style="color: {my_chart_color};">StDev</b></div><div>{std_val}</div><br/>'
                    f'</div>'
                    f'<div style="text-align: center;">'
                    f'<div><b style="color: {my_chart_color};">Columns</b></div><div>{cols}</div><br/>'
                    f'<div><b style="color: {my_chart_color};">End Date</b></div><div>{max_date}</div><br/>'
                    f'<div><b style="color: {my_chart_color};">Frequency</b></div><div>{dataframe_freq_name}</div><br/>'
                    f'<div><b style="color: {my_chart_color};">Median</b></div><div>{median_val}</div><br/>'
                    f'<div><b style="color: {my_chart_color};">Maximum</b></div><div>{max_val}</div><br/>'
                    f'<div><b style="color: {my_chart_color};">Mode</b></div><div>{mode_val}</div><br/>'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                    ,
                unsafe_allow_html=True)
                    # vertical spacer
            vertical_spacer(1)
    except:
        st.write('Error: could not show the quick summary stats...please contact admin')

def create_summary_df(data):
    """
    Create a DataFrame with summary statistics for the input data.

    Parameters
    ----------
    data : pandas DataFrame or pandas Series
        The data for which summary statistics are to be computed.

    Returns
    -------
    summary_df : pandas DataFrame
        A DataFrame with summary statistics for the input data.
    """
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    if not isinstance(data, pd.DataFrame):
        raise TypeError('Input data must be a pandas DataFrame or Series')
    
    ####################################                        
    # Compute summary statistics
    ####################################
    length = data.shape[0]
    length_label = dataset_size_function(data)
    percent_missing = (data.isna().mean())[0]
    
    ####################################
    # drop rows with missing values
    ####################################
    num_missing = data.isnull().sum()[0]
    missing_label_value, missing_label_perc = missing_value_function(data)
    if num_missing > 0:
        data.dropna(inplace=True)    
# =============================================================================
#         st.warning(f'''**Warning** ⚠️:  
#                  **{num_missing}** NaN values were excluded to be able to calculate metrics such as: *skewness*, *kurtosis* and *White-Noise* test.''')
# =============================================================================
    else:
        pass
    
    ########################
    # Continued Code 
    # Summary Statistics
    ########################
    mean = data.mean()
    mean_label, mean_position = mean_label_function(data)
    median = data.median()
    median_label, median_position = median_label_function(data)
    mode_val = mode(data)[0][0]
    std_dev = data.std()
    std_label = std_label_function(data)
    variance = data.var()
    var_label = var_label_function(data)
    num_distinct = data.nunique()
    distinct_label = distinct_values_label_function(data)
    kurt = kurtosis(data)[0]
    kurtosis_label = kurtosis_label_function(data)
    skewness = skew(data)[0]
   
    
    # Define skewness labels
    if skewness < -1.0:
        skewness_type = "Highly negatively skewed"
    elif -1.0 <= skewness < -0.5:
        skewness_type = "Moderately negatively skewed"
    elif -0.5 <= skewness <= 0.5:
        skewness_type = "Approximately symmetric"
    elif 0.5 < skewness <= 1.0:
        skewness_type = "Moderately positively skewed"
    else:
        skewness_type = "Highly positively skewed"                    
    
    # =============================================================================
    # White Noise - Ljung Box Test    
    # =============================================================================
              
    def white_noise_label(white_noise, lag_value):
        """
        Description
        ----------
        Get descriptive label for white noise test the quick insights dataframe
        
        Parameters
        ----------
        white_noise: boolean
            True or False
        lag_value: int
        """
        # initiate empty string container
        white_noise_label = ''
        if white_noise == 'True':
            white_noise_label = f'residuals independently distributed for lag={lag_value}' 
        elif white_noise == 'False':
            white_noise_label = f'residuals not independently distributed for lag={lag_value}'
        else:
            white_noise_label  = '-'
        return white_noise_label
    
    try:        
        if len(data) >= 24:
            # define the model
            model_lag24 = sm.tsa.AutoReg(data, lags=24, trend='c', old_names=False)
            # train model on the residuals
            res_lag24 = model_lag24.fit()
        
            # define the model
            model_lag48 = sm.tsa.AutoReg(data, lags=48, trend='c', old_names=False)
            # train model on the residuals
            res_lag48 = model_lag48.fit()
            
            # Perform Ljung-Box test on residuals with lag=24 and lag=48
            result_ljungbox24 = sm.stats.acorr_ljungbox(res_lag24.resid, lags=[24], return_df=True)
            result_ljungbox48 = sm.stats.acorr_ljungbox(res_lag48.resid, lags=[48], return_df=True)
                
            test_statistic_ljungbox_24 = result_ljungbox24.iloc[0]['lb_stat']
            test_statistic_ljungbox_48 = result_ljungbox48.iloc[0]['lb_stat']
            
            p_value_ljungbox_24 = result_ljungbox24.iloc[0]['lb_pvalue']
            p_value_ljungbox_48 = result_ljungbox48.iloc[0]['lb_pvalue']
            
            white_noise_24 = "True" if p_value_ljungbox_24 <= 0.05 else "False"
            white_noise_48 = "True" if p_value_ljungbox_48 <= 0.05 else "False"
            
            white_noise_24_lbl = white_noise_label(white_noise_24, 24)
            white_noise_48_lbl = white_noise_label(white_noise_48, 48)
        else:
            #define lags
            lag1 = int((len(data)-1)/2)
            lag2 = int(len(data)-1)
            
            st.write(lag1, lag2)#test
            
            # define the model
            model_lag24 = sm.tsa.AutoReg(data, lags = lag1, trend='c', old_names=False)
            # train model on the residuals
            res_lag24 = model_lag24.fit()
        
            # define the model
            model_lag48 = sm.tsa.AutoReg(data, lags = lag2, trend='c', old_names=False)
            # train model on the residuals
            res_lag48 = model_lag48.fit()
            
            # Perform Ljung-Box test on residuals with lag=24 and lag=48
            result_ljungbox24 = sm.stats.acorr_ljungbox(res_lag24.resid, lags=[lag1], return_df=True)
            result_ljungbox48 = sm.stats.acorr_ljungbox(res_lag48.resid, lags=[lag2], return_df=True)
                
            test_statistic_ljungbox_24 = result_ljungbox24.iloc[0]['lb_stat']
            test_statistic_ljungbox_48 = result_ljungbox48.iloc[0]['lb_stat']
            
            p_value_ljungbox_24 = result_ljungbox24.iloc[0]['lb_pvalue']
            p_value_ljungbox_48 = result_ljungbox48.iloc[0]['lb_pvalue']
            
            white_noise_24 = "True" if p_value_ljungbox_24 <= 0.05 else "False"
            white_noise_48 = "True" if p_value_ljungbox_48 <= 0.05 else "False"
            
            white_noise_24_lbl = white_noise_label(white_noise_24, lag1)
            white_noise_48_lbl = white_noise_label(white_noise_48, lag2)
    except:
        result_ljungbox = np.nan
        test_statistic_ljungbox_24 = np.nan
        test_statistic_ljungbox_48 = np.nan
        p_value_ljungbox_24 = np.nan
        p_value_ljungbox_48 = np.nan
        white_noise_24 = np.nan
        white_noise_48 = np.nan
        white_noise_24_lbl = '-'
        white_noise_48_lbl = '-'
    # =============================================================================
    
    #******************************
    # Augmented Dickey-Fuller Test
    #******************************
    result = adfuller(data)
    stationarity = "Stationary" if result[0] < result[4]["5%"] else "Non-Stationary"
    p_value = result[1]
    test_statistic = result[0]
    critical_value_1 = result[4]["1%"]
    critical_value_5 = result[4]["5%"]
    critical_value_10 = result[4]["10%"]
    
    #***********************
    # Perform Shapiro Test
    #***********************
    shapiro_stat, shapiro_pval = shapiro(data)
    normality = "True" if shapiro_pval > 0.05 else "False"
    
    ###########################
    # Create summary DataFrame
    ###########################
    summary_df = pd.DataFrame(columns=['Test', 'Test Name', 'Property', 'Settings', str('Column: ' + data.columns[0]), 'Label'])
    summary_df.loc[0] = ['Summary', 'Statistics', 'Length', '-', length, length_label]	
    summary_df.loc[1] = ['', '', '# Missing Values', '-', num_missing, missing_label_value]
    summary_df.loc[2] = ['', '', '% Missing Values', '-', f"{percent_missing:.2%}", missing_label_perc]
    summary_df.loc[3] = ['', '', 'Mean', '-', round(mean[0], 2), f'{mean_label} and {mean_position}']
    summary_df.loc[4] = ['', '', 'Median', '-', round(median[0], 2), f'{median_label} and {median_position}']
    summary_df.loc[5] = ['', '', 'Standard Deviation', '-', round(std_dev[0],2), std_label]
    summary_df.loc[6] = ['', '', 'Variance', '-', round(variance[0],2), var_label]
    summary_df.loc[7] = ['', '', 'Kurtosis', '-', round(kurt,2), kurtosis_label]
    summary_df.loc[8] = ['', '', 'Skewness', '-', round(skewness,2), skewness_type]
    summary_df.loc[9] = ['', '', '# Distinct Values', '-', num_distinct[0], distinct_label]
    summary_df.loc[10] = ['White Noise', 'Ljung-Box', 'Test Statistic', '24', round(test_statistic_ljungbox_24, 4), '-']
    summary_df.loc[11] = ['', '', 'Test Statistic', '48', round(test_statistic_ljungbox_48, 4), '-']
    summary_df.loc[12] = ['', '', 'p-value', '24', round(p_value_ljungbox_24, 4), white_noise_24_lbl]
    summary_df.loc[13] = ['', '', '', '48', round(p_value_ljungbox_48, 4), white_noise_48_lbl]
    summary_df.loc[14] = ['', '', 'White Noise', '24', white_noise_24, '-']
    summary_df.loc[15] = ['', '', '', '48', white_noise_48, '-']
    summary_df.loc[16] = ['Stationarity', 'ADF', 'Stationarity', '0.05', stationarity, '-']
    summary_df.loc[17] = ['', '', 'p-value', '0.05', round(p_value, 4), '-']
    summary_df.loc[18] = ['', '', 'Test-Statistic', '0.05', round(test_statistic, 4), '-']
    summary_df.loc[19] = ['', '', 'Critical Value 1%', '0.05', round(critical_value_1, 4), '-']
    summary_df.loc[20] = ['', '', 'Critical Value 5%', '0.05', round(critical_value_5, 4), '-']
    summary_df.loc[21] = ['', '', 'Critical Value 10%', '0.05', round(critical_value_10, 4), '-']
    summary_df.loc[22] = ['Normality', 'Shapiro', 'Normality', '0.05', normality, '-']
    summary_df.loc[23] = ['', '', 'p-value', '0.05', round(shapiro_pval, 4), '-']
    return summary_df

def dataset_size_function(data):
    """
    This function takes in a dataset and returns a label describing the size of the dataset.
    """
    num_rows = data.shape[0]
    if num_rows < 50:
        size_label = "Small dataset"
    elif num_rows < 1000:
        size_label = "Medium dataset"
    else:
        size_label = "Large dataset"
    return size_label

def missing_value_function(data):
    """
    This function takes in a dataset and returns a label based on the percentage of missing values in the dataset.
    """
    num_missing = data.isnull().sum()[0]
    num_total = data.size
    missing_ratio = num_missing / num_total
    if missing_ratio == 0:
        missing_label_value = "No missing values"
        missing_label_perc = "No percentage missing"
    elif missing_ratio < 0.01:
        missing_label_value = "Very few missing values"
        missing_label_perc = "Very Low percentage missing"
    elif missing_ratio < 0.05:
        missing_label_value = "Some missing values"
        missing_label_perc = "Low percentage missing"
    elif missing_ratio < 0.1:
        missing_label_value = "Many missing values"
        missing_label_perc = "High percentage missing"
    else:
        missing_label_value = "A large percentage of missing values"
        missing_label_perc = "Very high percentage missing"
    return missing_label_value, missing_label_perc

def std_label_function(data):
    """
    This function takes in a dataset, calculates the standard deviation, and returns a label based on the size of the standard deviation relative to the range of the dataset.
    """
    std = data.std()
    range_data = data.min() - data.max()
    std_ratio = (std / range_data)[0]
    if std_ratio <= 0.1:
        std_label = "Very small variability"
    elif std_ratio <= 0.3:
        std_label = "Small variability"
    elif std_ratio <= 0.5:
        std_label  = "Moderate variability"
    elif std_ratio <= 0.7:
        std_label  = "Large variability"
    else:
        std_label = "Very large variability"
    return std_label
    
def var_label_function(data):
    """
    This function takes in a dataset, calculates the variance, and returns a label based on the size of the variance relative to the range of the dataset.
    """
    var = data.var()
    range_data = data.min() - data.max()
    var_ratio = (var / range_data)[0]
    if var_ratio <= 0.1:
        var_label = "Very small variance"
    elif var_ratio <= 0.3:
        var_label = "Small variance"
    elif var_ratio <= 0.5:
        var_label = "Moderate variance"
    elif var_ratio <= 0.7:
        var_label = "Large variance"
    else:
        var_label = "Very large variance"
    return var_label

def mean_label_function(data):
    """
    This function takes in a dataset, calculates the mean, and returns a label based on the position of the mean relative to the range of the dataset.
    """
    median = data.median()[0]
    mean = data.mean()[0]
    range_data = data.max() - data.min()
    mean_ratio = ((mean - data.min()) / range_data)[0]
    if mean_ratio < 0.25:
        mean_label = "Mean skewed left"
    elif mean_ratio > 0.75:
        mean_label = "Mean skewed right"
    else:
        mean_label = "Mean balanced"
    if mean > median:
        mean_position = "Mean above median"
    elif mean < median:
        mean_position = "Mean below median"
    else:
        mean_position = "Mean equal to median"
    return mean_label, mean_position

def median_label_function(data):
    """
    This function takes in a dataset, calculates the median, and returns a label based on the position of the median relative to the range of the dataset.
    """
    median = data.median()[0]
    range_data = data.max() - data.min()
    median_ratio = ((median - data.min()) / range_data)[0]
    mean = data.mean()[0]
    mean_ratio = (mean - data.min()) / range_data
    if median_ratio < 0.25:
        median_label = "Median Skewed left"
    elif median_ratio > 0.75:
        median_label = "Median Skewed right"
    else:
        median_label = "Median balanced"
    if median > mean:
        median_position = "Median above mean"
    elif median < mean:
        median_position = "Median below mean"
    else:
        median_position = "Median equal to mean"
    return median_label, median_position
        
def kurtosis_label_function(data):
    """
    This function takes in a dataset, calculates the kurtosis, and returns a label based on the kurtosis value.
    """
    kurt = data.kurtosis()[0]
    if kurt < -2.0:
        kurtosis_label = 'Highly platykurtic (thin tails)'
    elif -2.0 <= kurt < -0.5:
        kurtosis_label = 'Moderately platykurtic (thin tails)'
    elif -0.5 <= kurt <= 0.5:
        kurtosis_label = 'Mesokurtic (medium tails)'
    elif 0.5 < kurt <= 2.0:
        kurtosis_label = 'Moderately leptokurtic (fat tails)'
    else:
        kurtosis_label = 'Highly leptokurtic (fat tails)'
    return kurtosis_label

def distinct_values_label_function(data):
    """
    This function takes in a dataset and returns a label based on the number of distinct values in the dataset.
    """
    num_distinct = len(data.drop_duplicates())
    num_total = len(data)
    distinct_ratio = num_distinct / num_total
    if distinct_ratio < 0.05:
        distinct_values_label = "Very low amount of distinct values"
    elif distinct_ratio < 0.1:
        distinct_values_label = "Low amount of distinct values"
    elif distinct_ratio < 0.5:
        distinct_values_label = "Moderate amount of distinct values"
    elif distinct_ratio < 0.9:
        distinct_values_label = "High amount of distinct values"
    else:
        distinct_values_label = "Very high amount of distinct values"
    return distinct_values_label

def adf_test(df, variable_loc, max_diffs=3):
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
    max_diff : int, optional
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
    col1, col2, col3 = st.columns([18,40,10])
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
            vertical_spacer(1) # newline vertical space
            h0 = st.markdown(r'❌$H_0$: the time series has a unit root, meaning it is **non-stationary**. It has some time dependent structure.')
            vertical_spacer(1)
            h1 = st.markdown(r'✅$H_1$: the time series does **not** have a unit root, meaning it is **stationary**. it does not have time-dependent structure.')
            vertical_spacer(1)
            result = f'**conclusion:**\
                      the null hypothesis can be :red[**rejected**] with a p-value of **`{p_value:.5f}`**, which is smaller than the 95% confidence interval (p-value = `0.05`).'
    else:
        # If the time series remains non-stationary after max_diffs differencings, return the non-stationary result
        for i in range(1, max_diffs+1):
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
                    h0 = st.markdown(f'❌$H_0$: the time series has a unit root, meaning it is :red[**non-stationary**]. it has some time dependent structure.')
                    vertical_spacer(1)
                    h1 = st.markdown(f'✅$H_1$: the time series does **not** have a unit root, meaning it is :green[**stationary**]. it does not have time-dependent structure.')
                    vertical_spacer(1)
                    result = f'**conclusion:**\
                              After *differencing* the time series **`{i}`** time(s), the null hypothesis can be :red[**rejected**] with a p-value of **`{p_value_str}`**, which is smaller than `0.05`.'
                break
            else:
               # If the time series remains non-stationary after max_diffs differencings, return the non-stationary result
                result = f'the {variable.name} time series is non-stationary even after **differencing** up to **{max_diffs}** times. last ADF p-value: {p_value:.5f}'
                
                with col2:
                    max_diffs = st.slider(':red[[Optional]] *Adjust maximum number of differencing:*', min_value=0, max_value=10, value=3, step=1, help='Adjust maximum number of differencing if Augmented Dickey-Fuller Test did not become stationary after differencing the data e.g. 3 times (default value)')
    return result

def ljung_box_test(df, variable_loc, lag, model_type="AutoReg"):
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
        st.info('**ForecastGenie message**: replaced your missing values with zero in a copy of original dataframe, in order to perform the ljung box test and can introduce bias into the results.')

    # Drop missing values
    variable = variable.dropna()

    if model_type == "AutoReg":
        # Fit AutoReg model to the data
        model = sm.tsa.AutoReg(variable, lags = lag, trend='c', old_names=False)
        res = model.fit() 
        
        # Perform Ljung-Box test on residuals for all lags including up to lag integer
        result_ljungbox = sm.stats.acorr_ljungbox(res.resid, lags=lag, return_df=True)
    else:
        raise ValueError("Invalid model type selected.")
        
    #test_statistic = result_ljungbox.iloc[0]['lb_stat']
    #p_value = result_ljungbox.iloc[0]['lb_pvalue']
    p_values = result_ljungbox['lb_pvalue']
    
    #white_noise = "True" if p_value > 0.05 else "False"
    
    alpha = 0.05 #significance level
    lag_with_autocorr = None
    
    # Check p-value for each lag
    for i, p_value in enumerate(p_values):
        lag_number = i + 1
        if p_value <= alpha:
            lag_with_autocorr = lag_number
            break
 
    # if p value is less than or equal to significance level reject zero hypothesis
    if lag_with_autocorr is not None: 
        st.markdown(f'❌ $H_0$: the residuals have **:green[no autocorrelation]** for one or more lags up to a maximum lag of **{lag}**.') # h0
        st.markdown(f'✅ $H_1$: the residuals **:red[have autocorrelation]** for one or more lags up to a maximum lag of **{lag}**.') #h1
    else: 
        st.markdown(f'✅ $H_0$: the residuals have **:green[no autocorrelation]** for one or more lags up to a maximum lag of **{lag}**.') # h0
        st.markdown(f'❌ $H_1$: the residuals **:red[have autocorrelation]** for one or more lags up to a maximum lag of **{lag}**.') #h1
   
    alpha = 0.05  # Significance level
    
    if lag_with_autocorr is None:
        st.markdown(f"**conclusion:** the null hypothesis **cannot** be rejected for all lags up to maximum lag of **{lag}**. the residuals show **no significant autocorrelation.**")
        vertical_spacer(2)
    else:
        st.markdown(f"**conclusion:** the null hypothesis can be **rejected** for at least one lag up to a maximum lag of **{lag}** with a p-value of **`{p_value:.2e}`**, which is smaller than the significance level of **`{alpha:}`**. this suggests presence of serial dependence in the time series.")
    
    # show dataframe with test results
    st.dataframe(result_ljungbox, use_container_width=True)
    
    return res, result_ljungbox
# =============================================================================
#     except:
#         pass
# =============================================================================

def ljung_box_plots(df, variable_loc, res, lag, result_ljungbox, my_chart_color):
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
    fig.update_traces(line_color = my_chart_color, 
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
    
    fig_pvalues = go.Figure(data=go.Scatter(x = df.index[0:lag], 
                                            y = p_values.iloc[0:lag], 
                                            mode = "markers", 
                                            name = 'p-value',
                                            marker = dict(symbol="circle-open", color = my_chart_color)))
    
    # Add a blue dotted line for the significance level
    fig_pvalues.add_trace(go.Scatter(x = [df.index[0], df.index[lag-1]],
                                     y = [0.05, 0.05],
                                     mode = "lines",
                                     line = dict(color = adjusted_color, dash="dot"), 
                                     name="α = 0.05"))
    
    # Position the legend at the top right inside the graph and update y-axis range and tick settings
    fig_pvalues.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='top'), xaxis_title="Lag", yaxis_title="P-value", yaxis=dict(range=[-0.1, 1], dtick=0.1), margin=dict(t=10), showlegend=True)
    
    # Show plotly chart
    st.plotly_chart(fig_pvalues, use_container_width=True)
    
#******************************************************************************
# GRAPH FUNCTIONS | PLOT FUNCTIONS
#******************************************************************************
def plot_missing_values_matrix(df):
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
                    color_continuous_scale='Viridis', # Viridis
                    title='')
    
    # Set Plotly configuration options
    fig.update_layout(width=400, height=400, margin=dict(l=50, r=50, t=0, b=50))
    fig.update_traces(showlegend=False)
    # Display Plotly Express figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def acf_pacf_info():
    background_colors = ["#456689", "#99c9f4"]
    
    col1, col2, col3 = st.columns([5,5,5])
    with col1:
        show_acf_info_btn = st.button(f'about acf plot', use_container_width=True, type='secondary')
    with col2:
        show_pacf_info_btn = st.button(f'about pacf plot', use_container_width=True, type='secondary')
    with col3:
        diff_acf_pacf_info_btn = st.button(f'difference acf/pacf', use_container_width=True, type='secondary')   
        
    if show_acf_info_btn == True:
        vertical_spacer(1)
        my_subheader('Autocorrelation Function (ACF)', my_background_colors = background_colors)
        col1, col2, col3 = st.columns([1,8,1]) 
        with col2:
           st.markdown('''
                        The **Autocorrelation Function (ACF)** plot is a statistical tool used to identify patterns of correlation between observations in a time series dataset. 
                        It is commonly used in time series analysis to determine the extent to which a given observation is related to its previous observations.  
                        \nThe **ACF** plot displays the correlation coefficient between the time series and its own lagged values (i.e., the correlation between the series at time $t$ and the series at times $t_{-1}$, $t_{-2}$, $t_{-3}$, etc.).  
                        The horizontal axis of the plot shows the lag or time difference between observations, while the vertical axis represents the correlation coefficient, ranging from -1 to 1.
                        ''')
        vertical_spacer(1)  
        my_subheader('How to interpret a ACF plot', my_background_colors = background_colors)
        col1, col2, col3 = st.columns([1,8,1]) 
        with col2:
            st.markdown('''Interpreting the **ACF** plot involves looking for significant peaks or spikes above the horizontal dashed lines (which represent the confidence interval) to determine if there is any correlation between the current observation and the lagged observations. 
                        If there is a significant peak at a particular lag value, it indicates that there is a strong correlation between the observation and its lagged values up to that point.
                        ''')
        vertical_spacer(1)                            
        my_subheader('Key Points:', my_background_colors = background_colors)  
        col1, col2, col3 = st.columns([1,8,1]) 
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
        my_subheader('Partial Autocorrelation Function (PACF)', my_background_colors = background_colors)
        col1, col2, col3 = st.columns([1,8,1])
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
        
        my_subheader('How to interpret a PACF plot', my_background_colors = background_colors)
        col1, col2, col3 = st.columns([1,8,1]) 
        col1, col2, col3 = st.columns([1,8,1])
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
        my_subheader('Key Points:', my_background_colors = background_colors)  
        col1, col2, col3 = st.columns([1,8,1])
        with col2:
            st.markdown('''                            
                        - **The first lag (lag 0) is always 1**, since an observation is perfectly correlated with itself.
                        - A significant spike at a particular lag indicates that there may be some **useful information** in that lagged value for predicting the current observation. 
                          This can be used to guide the selection of lag values in time series forecasting models.
                        - A sharp drop in the PACF plot after a certain lag suggests that the lags beyond that point **are not useful** for prediction, and can be safely ignored.
                        ''')
        vertical_spacer(1)                
        my_subheader('An analogy', my_background_colors = background_colors)
        col1, col2, col3 = st.columns([1,8,1])
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
        my_subheader('Differences explained between ACF and PACF',  my_background_colors = background_colors)
        col1, col2, col3 = st.columns([1,8,1]) 
        with col2:
            st.markdown('''
                        - The **ACF** plot measures the correlation between an observation and its lagged values.
                        - The **PACF** plot measures the correlation between an observation and its lagged values while controlling for the effects of intermediate observations.
                        - The **ACF** plot is useful for identifying the order of a moving average **(MA)** model, while the **PACF** plot is useful for identifying the order of an autoregressive **(AR)** model.  
                        ''')
        vertical_spacer(1)
        
def chart_title(title, subtitle, font, font_size):
    """
    Creates a dictionary containing the properties for a chart title and subtitle.

    Parameters:
    -----------
    title : str
        The main title of the chart.
    subtitle : str
        The subtitle of the chart.
    font : str
        The name of the font to be used for the title and subtitle.
    font_size : int
        The font size to be used for the title and subtitle.

    Returns:
    --------
    dict
        A dictionary containing the properties for the chart title and subtitle,
        including the text, font size, font family, and anchor position.
    """
    return {
            "text": title,
            "subtitle": subtitle,
            "fontSize": font_size,
            "font": font,
            "anchor": "middle"
            }

def altair_correlation_chart(total_features, importance_scores, pairwise_features_in_total_features, corr_threshold):
    """
    Creates an Altair chart object for visualizing pairwise feature importance scores.

    Args:
        total_features (list): List of all features.
        importance_scores (dict): Dictionary mapping feature names to importance scores.
        pairwise_features_in_total_features (list): List of pairs of features.
        corr_threshold (float): Threshold for displaying correlation scores.

    Returns:
        altair.Chart: Altair chart object for visualizing pairwise feature importance scores.
    """
    # Set title font style and size
    title_font = "Helvetica"
    title_font_size = 12
    num_features = len(total_features)
    num_cols = min(3, num_features)
    num_rows = math.ceil(num_features / num_cols)
    charts = []
    for i, (feature1, feature2) in enumerate(pairwise_features_in_total_features):
        if feature1 in total_features and feature2 in total_features:
            score1 = importance_scores[feature1]
            score2 = importance_scores[feature2]
            data = pd.DataFrame({'Feature': [feature1, feature2], 'Score': [score1, score2]})
            if score1 > score2:
                total_features.remove(feature2)
                chart = alt.Chart(data).mark_bar().encode(
                    x='Score:Q',
                    y=alt.Y('Feature:O', sort='-x'),
                    color=alt.condition(
                        alt.datum.Feature == feature2,
                        alt.value('#b8a5f9'), # red
                        alt.value('#6137f1')  # green
                    )
                ).properties(width=100, height=100, title=chart_title("Removing", feature2, title_font, title_font_size))
            elif score2 > score1:
                total_features.remove(feature1)
                chart = alt.Chart(data).mark_bar().encode(
                    x='Score:Q',
                    y=alt.Y('Feature:O', sort='-x'),
                    color=alt.condition(
                        alt.datum.Feature == feature1,
                        alt.value('#b8a5f9'), # red #DC143C
                        alt.value('#6137f1')   # green #00FF7F
                    )
                ).properties(width=100, height=100, title=chart_title("Removing", feature1, title_font, title_font_size))
            else:
                total_features.remove(feature1)
                chart = alt.Chart(data).mark_bar().encode(
                    x='Score:Q',
                    y=alt.Y('Feature:O', sort='-x'),
                    color=alt.condition(
                        alt.datum.Feature == feature1,
                        alt.value('#b8a5f9'), # red
                        alt.value('#6137f1')  # green
                    )
                ).properties(width=100, height=100, title=chart_title("Removing", feature1, title_font, title_font_size))
            charts.append(chart)
    # Combine all charts into a grid
    grid_charts = []
    for i in range(num_rows):
        row_charts = []
        for j in range(num_cols):
            idx = i*num_cols+j
            if idx < len(charts):
                row_charts.append(charts[idx])
        if row_charts:
            grid_charts.append(alt.hconcat(*row_charts))
    
    grid_chart = alt.vconcat(*grid_charts, spacing=10)
    
    # create a streamlit container with a title and caption
    my_text_paragraph("Removing Highly Correlated Features", my_font_size='26px')
    
    col1,col2,col3 = st.columns([5.5,4,5])
    with col2:
        st.caption(f'for pair-wise features >={corr_threshold*100:.0f}%')
        
    # show altair chart with pairwise correlation importance scores and in red lowest and green highest
    st.altair_chart(grid_chart, use_container_width=True)

def display_dataframe_graph(df, key=0, my_chart_color = '#217CD0'):
    """
    Displays a line chart of a Pandas DataFrame using Plotly Express.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to display.
    key : int or str, optional
        An optional identifier used to cache the output of the function when called with the same arguments,
        by default 0.

    Returns
    -------
    None
        The function displays the chart in the Streamlit app.

    """
    fig = px.line(df,
                  x=df.index,
                  y=df.columns,
                  #labels=dict(x="Date", y="y"),
                  title='',
                  )
    # Set Plotly configuration options
    fig.update_layout(width=800, height=400, xaxis=dict(title='Date'), yaxis=dict(title='', rangemode='tozero'), legend=dict(x=0.9, y=0.9))
    # set line color and width
    fig.update_traces(line=dict(color = my_chart_color, width= 2, dash = 'solid'))
    # Add the range slider to the layout
    fig.update_layout(
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
                y=1.2,
                yanchor='auto', #top
                font=dict(size=10),
            ),
            rangeslider=dict( #bgcolor='45B8AC',
                visible=True,
                range=[df.index.min(), df.index.max()]  # Set range of slider based on data
            ),
            type='date'
        )
    )

    # Display Plotly Express figure in Streamlit
    st.plotly_chart(fig, use_container_width=True, key=key)
    
def create_rfe_plot(df_ranking):
    """
    Create a scatter plot of feature rankings and selected features.

    Parameters:
        df_ranking (pandas.DataFrame): A DataFrame with feature rankings and selected features.

    Returns:
        None
    """
    # calculate number of top features chosen by user in slider
    num_top_features = (len(df_ranking[df_ranking['Selected'] == 'Yes']))
    
    # streamlit title rfe plot
    my_text_paragraph('Recursive Feature Elimination', my_font_size='26px')
    # streamlit subtitle rfe plot
    my_text_paragraph('with Cross-Validation (RFECV)', my_font_size='16px')
    
    rank_str = get_state("SELECT_PAGE_RFE", "num_features_rfe")
    my_text_paragraph(f'<b> TOP {num_top_features}</b>', my_font_size='16px', my_font_family='Segui UI')
    
    fig = px.scatter(df_ranking, x='Features', y='Ranking', color='Selected', hover_data=['Ranking'],   color_discrete_map={'Yes': '#4715EF', 'No': '#000000'})
    fig.update_layout(
        title={
            'text': '',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Features',
        yaxis_title='Ranking',
        legend_title='Selected',
        xaxis_tickangle=-45 # set the tickangle to x degrees
    )
    return fig  

def plot_scaling_before_after(X_unscaled_train, X_train, numerical_features):
    """
    Plots a figure showing the unscaled and scaled data for each numerical feature.

    Parameters:
    -----------
    X_unscaled_train : pandas.DataFrame
        The unscaled training data.
    X_train : pandas.DataFrame
        The scaled training data.
    numerical_features : list
        The list of numerical feature names.

    Returns:
    --------
    None.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    # create trace for unscaled data for each feature
    for feature in numerical_features:
        trace_unscaled = go.Scatter(
            x=X_unscaled_train.index,
            y=X_unscaled_train[feature],
            mode='lines',
            name=feature
        )
        fig.add_trace(trace_unscaled, row=1, col=1)
    # create trace for scaled data for each feature
    for feature in numerical_features:
        trace_scaled = go.Scatter(
            x=X_train.index,
            y=X_train[feature],
            mode='lines',
            name=f'{feature} (scaled)'
        )
        fig.add_trace(trace_scaled, row=2, col=1)
    # add selection event handler
    updatemenu = []
    buttons = []
    buttons.append(dict(label='All',
                        method='update',
                        args=[{'visible': [True] * len(fig.data)}]))   
    for feature in numerical_features:
        button = dict(
                        label=feature,
                        method="update",
                        args=[{"visible": [False] * len(fig.data)},
                              {"title": f"Feature: {feature}"}],
                    )
        button['args'][0]['visible'][numerical_features.index(feature)] = True
        button['args'][0]['visible'][numerical_features.index(feature) + len(numerical_features)] = True
        buttons.append(button)
    updatemenu.append(dict(buttons=buttons, 
                           direction="down", 
                           showactive=True, 
                           pad={"r": 5, "t": 5},
                           x=0.5, 
                           xanchor="center", 
                           y=1.2, 
                           yanchor="top"
                           )
                      )
    fig.update_layout(updatemenus=updatemenu, 
                      title="Feature: " + numerical_features[0])
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Unscaled", row=1, col=1, fixedrange=True)
    fig.update_yaxes(title_text="Scaled", row=2, col=1, fixedrange=True)
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.1,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12, family="Arial")
        ),
        updatemenus=updatemenu,
        title={
            'text': '',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    # show the figure in streamlit app
    st.plotly_chart(fig, use_container_width=True)

def plot_train_test_split(df, split_index):
    """
    Plot the train-test split of the given dataframe.
    
    Parameters:
    -----------
    df: pandas.DataFrame
        A pandas dataframe containing the time series data.
    split_index: int
        The index position that represents the split between the training and test set.
    
    Returns:
    --------
    fig: plotly.graph_objs._figure.Figure
        A plotly Figure object containing the train-test split plot.
    """
    try:
        train_linecolor = get_state('COLORS', 'chart_color')
        test_linecolor = adjusted_color = adjust_brightness(train_linecolor, 2)
    except:
        train_linecolor = '#4715ef'
        test_linecolor = '#efac15'
        
    
    # Get the absolute maximum value of the data
    max_value = df.iloc[:, 0].max()
    min_value = min(df.iloc[:, 0].min(), 0)
    fig = go.Figure()
    fig.add_trace(
                  go.Scatter(  
                              x=df.index[:split_index],
                              y=df.iloc[:split_index, 0],
                              mode='lines',
                              name='Train',
                              line=dict(color = train_linecolor)
                           )
                  )
    fig.add_trace(
                  go.Scatter(
                               x=df.index[split_index:],
                               y=df.iloc[split_index:, 0],
                               mode='lines',
                               name='Test',
                               line=dict(color = test_linecolor)
                           )
                  )
    fig.update_layout(
                      title='',
                      yaxis=dict(range=[min_value*1.1, max_value*1.1]), # Set y-axis range to include positive and negative values
                      shapes=[dict(type='line',
                                    x0=df.index[split_index],
                                    y0=-max_value*1.1, # Set y0 to -max_value*1.1
                                    x1=df.index[split_index],
                                    y1=max_value*1.1, # Set y1 to max_value*1.1
                                    line=dict(color='grey',
                                              dash='dash'))],
                      annotations=[dict(
                                        x=df.index[split_index],
                                        y=max_value*1.05,
                                        xref='x',
                                        yref='y',
                                        text='Train/Test<br>Split',
                                        showarrow=True,
                                        font=dict(color="grey", size=15),
                                        arrowhead=1,
                                        ax=0,
                                        ay=-40
                                       )
                                   ]
                      )
    split_date = df.index[split_index-1]
    fig.add_annotation(
                       x=split_date,
                       y=0.99*max_value,
                       text=str(split_date.date()),
                       showarrow=False,
                       font=dict(color="grey", size=16)
                       )
    return fig    

def correlation_heatmap(X, correlation_threshold=0.8):
    """
    Description:
        This function generates a correlation heatmap for a given pandas DataFrame X, 
        showing only features that have a correlation higher or equal to the given correlation_threshold.
    Parameters:
        X : pandas DataFrame object
            The input dataset for which the correlation heatmap is to be generated.
        correlation_threshold : float (default=0.8)
            The correlation threshold value that will determine which features are shown in the heatmap. 
            Only features with a correlation higher or equal to the given threshold will be displayed in the heatmap.
    Returns:
        None
    """
    # create a new dataframe with only features that have a correlation higher or equal to the threshold
    corr_matrix = X.corr()
    features_to_keep = corr_matrix[abs(corr_matrix) >= corr_threshold].stack().reset_index().iloc[:, [0, 1]]
    features_to_keep.columns = ['feature1', 'feature2']
    features_to_keep = features_to_keep[features_to_keep['feature1'] != features_to_keep['feature2']]
    X_corr = X[features_to_keep['feature1'].unique()]
    # compute correlation matrix
    corr_matrix = X_corr.corr()
    # create heatmap using plotly express
    fig = px.imshow(corr_matrix.values,
                    color_continuous_scale=[[0, '#FFFFFF'], [1, '#4715EF']], #"Blues",
                    zmin=-1,
                    zmax=1,
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    origin='lower')
    
    # add text annotations to heatmap cells
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                fig.add_annotation(x=i, y=j, text="{:.2f}".format(corr_matrix.iloc[i, j]), font=dict(color='white'))
    
    # add colorbar title
    fig.update_coloraxes(colorbar_title="Correlation")
    # set x and y axis labels to diagonal orientation
    fig.update_xaxes(tickangle=-45, showticklabels=True)
    fig.update_yaxes(tickangle=0, showticklabels=True)
    # adjust heatmap size and margins
    fig.update_layout(width=400,
                      height=400,
                      margin=dict(l=200, r=200, t=0, b=0))
    
    # show plotly figure in streamlit
    st.plotly_chart(fig, use_container_width=True)
    
#******************************************************************************
# DOCUMENTATION FUNCTIONS
#******************************************************************************
def model_documentation():
    ''' SHOW MODEL DOCUMENTATION
        - Naive Models
        - Linear Regression
        - SARIMAX
        - Prophet
    '''
    vertical_spacer(2)
    # =============================================================================
    # ADD MOON MENU BUTTONS    
    # =============================================================================
    # returns the lowercase with no spaces but underscore your button_names when clicked
    #clicked_moon_btn = create_moon_clickable_btns(button_names = ['Home', 'Naive Model', 'Linear Regression', 'SARIMAX', 'Prophet'])
    clicked_moon_btn = st.selectbox(label = 'Select Model', 
                                    options = ['-', 'Naive Model', 'Linear Regression', 'SARIMAX', 'Prophet'])
    
    with st.expander('', expanded=True):
        col1, col2, col3 = st.columns([2, 8, 2])
        if clicked_moon_btn == '-':
            st.image('./images/model_details_info.png')
            
        elif clicked_moon_btn == 'Naive Model':
            with col2:
                my_text_header('Naive Model')
                vertical_spacer(1)
                st.markdown('''
                <p style="text-align: justify;">
                The <strong>Naive Model</strong> is one of the simplest forecasting models in time series analysis. 
                It assumes that the value of a variable at any given time is equal to the value of the variable at the previous time period. 
                This means that this model is a special case of an <strong>A</strong>uto<strong>R</strong>egressive model of order 1, also known as <b> AR(1)</b>.
                </p>
                
                <p style="text-align: justify;">
                The Naive Model is useful as a baseline model to compare more complex forecasting models, such as ARIMA, exponential smoothing, or machine learning algorithms. 
                It is also useful when the underlying data generating process is highly unstable or unpredictable, and when there is no trend, seasonality, or other patterns to capture.
                The Naive Model can be expressed as a simple equation: 
                </p>
                         
                &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; $\hat{y}_{t} = y_{t-1}$
    
                <p style="text-align: justify;">
                where:
                </p>
    
                - $y_t$ is the value of the variable at time $t$
                <br> 
                - $y_{t-1}$ is the value of the variable at time $_{t-1}$
                
                <br>
                
                <p style="text-align: justify;">
                The Naive Model can be extended to incorporate seasonal effects, by introducing a lag period corresponding to the length of the seasonal cycle. 
                For example, if the time series has a weekly seasonality, the Naive Model with a lag of one week is equivalent to the model with a lag of seven days, and is given by:
                </p>
                
                &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$\hat{y}_{t} = y_{t-7}$
                
                <p style="text-align: justify;">
                where:
                </p>
                
                - $y_t$ is the value of the variable at time $t$
                <br>
                - $y_{t-7}$ is the value of the variable at time $_{t-7}$
    
                <br>
                <p style="text-align: justify;">
                In general, the lag value for the seasonal Naive Model should be determined based on the length of the seasonal cycle in the data, and can be estimated using visual inspection, autocorrelation analysis, or domain knowledge.
                </p>
                ''', unsafe_allow_html=True)
            
        elif clicked_moon_btn == 'Linear Regression':
            with col2:
                my_text_header('Linear Regression')
                vertical_spacer(1)
                st.markdown('''
                            <p style="text-align: justify;"> 
                            <strong> The Linear Regression Model </strong> is used to analyze the relationship between a dependent variable and one or more independent variables. 
                            It involves finding a line or curve that best fits the data and can be used to make predictions. 
                            The method assumes that the relationship between the variables is linear and that errors are uncorrelated.
                 
                            
                            To find this line, we use a technique called least squares regression, which involves finding the line that minimizes the sum of the squared differences between the predicted values and the actual values. 
                            The line is described by the equation:
                            </p>                           
                            
                            &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; $$Y = β₀ + β₁ X$$
                        
                            where:
                            - $Y$ is the dependent variable</li>
                            - $X$ is the independent variable</li>
                            - $β₀$ is the intercept &rarr; the value of $Y$ when $X = 0$
                            - $β₁$ is the slope &rarr; the change in $Y$ for a unit change in $X$
                            <br>
                            ''', unsafe_allow_html=True)
        elif clicked_moon_btn == 'SARIMAX':
            with col2:
                my_text_header('SARIMAX')
                vertical_spacer(1)
                st.markdown('''
                            <p style="text-align: justify;">
                            <strong>SARIMAX</strong>, or <b>S</b>easonal <b>A</b>utoregressive <b>I</b>ntegrated <b>M</b>oving <b>A</b>verage with e<b>X</b>ogenous variables, is a popular time series forecasting model.
                            The ARIMA model is a time series forecasting model that uses past values of a variable to predict future values. 
                            SARIMAX extends ARIMA by incorporating seasonal patterns and adding exogenous variables <b>X</b> that can impact the variable being forecasted. The model can be represented as follows:
                            <br>
                            <br>
                            &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <strong> SARIMA(p, d, q)(P, D, Q)s </strong>
                            
                            Non-Seasonal Terms    
                            - <b>p </b> &emsp;The order of the autoregressive (AR) term, which refers to the number of lagged observations of the dependent variable in the model. A higher value of p means the model is considering more past values of the dependent variable.
                            <br>
                            - <b>d</b> &emsp;The order of the differencing (I) term, which refers to the number of times the data needs to be differenced to make it stationary. Stationarity is a property of time series data where the statistical properties, such as the mean and variance, are constant over time.
                            <br>
                            - <b>q</b> &emsp;The order of the moving average (MA) term, which refers to the number of lagged forecast errors in the model. A higher value of q means the model is considering more past forecast errors.
                            <br>
                            
                            <br>Seasonal Terms:
                            - <b>P</b> &emsp;The seasonal order of the autoregressive term, which refers to the number of seasonal lags in the model.
                            <br>
                            - <b>D</b> &emsp;The seasonal order of differencing, which refers to the number of times the data needs to be differenced at the seasonal lag to make it stationary.
                            <br>
                            - <b>Q</b> &emsp;The seasonal order of the moving average term, which refers to the number of seasonal lags of the forecast errors in the model.
                            <br>
                            - <b>s:</b> &emsp;The length of the seasonal cycle, which is the number of time steps in each season. For example, if the data is monthly and the seasonality is yearly, s would be 12. The parameter s is used to determine the number of seasonal lags in the model.
                            <br>
                            <br>Exogenous Variables:
                            <br>
                            - <b>X</b>  &emsp; These are external factors that can impact the variable being forecasted. They are included in the model as additional inputs.  
                            </p>
                            ''', unsafe_allow_html=True)
        elif clicked_moon_btn == 'Prophet':
            with col2:
                my_text_header('Prophet')
                vertical_spacer(1)
                st.markdown('''
                            The Facebook <strong> Prophet </strong> model is a popular open-source library for time series forecasting developed by Facebook's Core Data Science team.
                            It is designed to handle time series data with strong seasonal effects and other external factors.
                            It uses a combination of historical data and user-defined inputs to generate forecasts for future time periods.  
                            <br>
                            <center><h6>Variables in the Prophet Model</h6></center>
                            The main variables in the Prophet model are:
                            <br> 
                            - <b>Trend</b>: This is the underlying pattern in the data that represents the long-term direction of the series. It can be linear or non-linear and is modeled using a piecewise linear function.
                            <br> 
                            - <b>Seasonality</b>: This is the periodic pattern in the data that repeats over fixed time intervals. It can be daily, weekly, monthly, or yearly, and is modeled using Fourier series.
                            <br> 
                            - <b>Holidays</b>: These are user-defined events or time periods that are known to affect the time series. The model includes them as additional regressors in the forecasting equation.
                            <br> 
                            - <b>Regressors</b>: These are additional time-varying features that can affect the time series, such as weather, economic indicators, or other external factors.
                            <br>                               
                            <br>
                            <center><h6>Math Behind the Prophet Model</h6></center>
                            The math behind the Prophet model involves fitting a Bayesian additive regression model to the time series data. The model is formulated as follows:
                            <br>
                            <br>
                            
                            &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; $$y_t = g_t + s_t + h_t + e_t$$
                            
                            where:
                            - $y_t$ is the observed value at time $t$
                            - $g_t$ is the trend component
                            - $s_t$ is the seasonality component
                            - $h_t$ is the holiday component
                            - $e_t$ is the error term. 
                            
                            <br>
                            The <b>trend</b> component is modeled using a piecewise linear function, while the <b>seasonality component</b> is modeled using a Fourier series. The <b>holiday component</b> and any additional regressors are included as additional terms in the regression equation.
                            <br>
                            <br>
                            The model is estimated using a Bayesian approach that incorporates prior information about the parameters and allows for uncertainty in the forecasts. The parameters are estimated using Markov Chain Monte Carlo (MCMC) sampling, which generates a large number of possible parameter values and uses them to estimate the posterior distribution of the parameters. The posterior distribution is then used to generate forecasts for future time periods.
                            <br>
                            <br>
                            Overall, the Prophet model is a powerful tool for time series forecasting that can handle complex data patterns and external factors. Its flexible modeling approach and Bayesian framework make it a popular choice for many data scientists and analysts.
                            <br>
                            <br>
                            ''', unsafe_allow_html=True)
        else:
            st.image('./images/train_info.png')   
            
#******************************************************************************
# OTHER FUNCTIONS
#******************************************************************************
def display_summary_statistics(df):
    """
    Calculate summary statistics for a given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame for which to calculate summary statistics.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the minimum, maximum, mean, median, standard deviation,
        and data type for each column of the input DataFrame.
    """
    summary = pd.DataFrame()
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            summary[col] = [df[col].min(), df[col].max(), '-', '-', '-', 'datetime']
        elif df[col].dtype != 'object':
            summary[col] = [df[col].min().round(2), df[col].max().round(2), 
                            df[col].mean().round(2), df[col].median().round(2), 
                            df[col].std().round(2), df[col].dtype]
        else:
            summary[col] = [np.nan, np.nan, np.nan, np.nan, np.nan, df[col].dtype]
    summary = summary.transpose()
    summary.columns = ['Min', 'Max', 'Mean', 'Median', 'Std', 'dtype']
    return summary

def generate_demo_data(seed=42):
    """
    Generate demo data with weekly, monthly, quarterly, and yearly patterns.

    Args:
        seed (int): Random seed for reproducibility.

    Returns:
        pandas.DataFrame: A DataFrame with two columns, 'date' and 'demo_data', where 'date'
        is a range of dates from 2018-01-01 to 2022-12-31, and 'demo_data' is a time series
        with weekly, monthly, quarterly, and yearly patterns plus random noise.
    """
    np.random.seed(seed)
    date_range = pd.date_range(start='1/1/2018', end='12/31/2022', freq='D')
    # generate weekly pattern
    weekly = 10 * np.sin(2 * np.pi * (date_range.dayofweek / 7))
    # generate monthly pattern
    monthly = 10 * np.sin(2 * np.pi * (date_range.month / 12))
    # generate quarterly pattern
    quarterly = 10 * np.sin(2 * np.pi * (date_range.quarter / 4))
    # generate yearly pattern
    yearly = 10 * np.sin(2 * np.pi * (date_range.year - 2018) / 4)
    # generate random variation
    noise = np.random.normal(loc=0, scale=2, size=len(date_range))   
    # generate time series data
    values = weekly + monthly + quarterly + yearly + noise + 10
    demo_data = pd.DataFrame({'date': date_range, 'demo_data': values})
    return demo_data

def forecast_wavelet_features(X, features_df_wavelet, future_dates, df_future_dates):
    """
    Forecast the wavelet features of a time series using historical data and a selected model.
    
    Parameters
    ----------
    X : pandas DataFrame
        The historical time series data, with each column representing a different feature.
    
    features_df_wavelet : pandas DataFrame
        The features to be forecasted using the discrete wavelet transform.
    
    future_dates : pandas DatetimeIndex
        The dates for which to make the wavelet feature forecasts.
    
    df_future_dates : pandas DataFrame
        The forecasted values for the wavelet features, with a column for each feature and a row for each future date.
    
    Returns
    -------
    pandas DataFrame
        The forecasted values for the wavelet features, combined with the original dataframe, with a column for each feature and a row for each future date.
    
    Raises
    ------
    StreamlitAPIException
        If an error occurs during the forecast.
    
    Notes
    -----
    This function uses the historical data and a selected model to forecast the wavelet features of a time series. The wavelet features are obtained using the discrete wavelet transform, and a separate model is fitted and used to forecast each feature. The resulting forecasts are combined with the original dataframe to produce the final forecast.
    
    """
    ########### START WAVELET CODE ###########  
    # if user selected checkbox for Discrete Wavelet Features run code run prediction for wavelet features
    if dwt_features_checkbox: 
        try:
            # only extract wavelet features
            X_wavelet_features = X.loc[:, features_df_wavelet.columns.intersection(X.columns)]
            X_future_wavelet = pd.DataFrame(index=future_dates, columns=X_wavelet_features.columns)
            # iterate over all wavelet feature column names
            for col in X_wavelet_features.columns:
                # define the length of the dataframe
                n = len(X_wavelet_features)
                # just use the date range as integers as independent feature (1 through n)
                X_wavelet = np.arange(1, n+1).reshape(-1, 1)
                # define target column e.g. each wavelet feature independently is forecasted into future to user defined date
                y_wavelet = X_wavelet_features[col]
                # define forecasting model user can chose for predicting the wavelet features for forecast date range
                if model_type_wavelet == 'Support Vector Regression':
                    model_wavelet = SVR()
                else:
                    model_wavelet = LinearRegression()  
                # fit the model on the historical data
                model_wavelet.fit(X_wavelet, y_wavelet)
                # predict future
                prediction_wavelet = model_wavelet.predict(np.arange(n+1, n+len(future_dates)+1).reshape(-1, 1))
                X_future_wavelet[col] = prediction_wavelet
            # reset the index and rename the datetimestamp column to 'date' - which is now a column that can be used to merge dataframes
            X_future_wavelet = X_future_wavelet.reset_index().rename(columns={'index': 'date'})
            # combine the independent features with the forecast dataframe
            df_future_dates = pd.merge(df_future_dates, X_future_wavelet, on='date', how='left' )
            return df_future_dates
        except:
            st.warning('Error: Discrete Wavelet Features are not created correctly, please remove from selection criteria')

def perform_train_test_split(df, my_insample_forecast_steps, scaler_choice=None, numerical_features=[]):
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
    X_train_numeric_scaled = pd.DataFrame() # TEST
    X_test_numeric_scaled = pd.DataFrame() # TEST
    X_numeric_scaled = pd.DataFrame() # TEST
    
    # Split the data into training and testing sets
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0:1]
# =============================================================================
#     # Find the index of the 'date' column
#     date_column_index = df.columns.get_loc('date')
#     # Get the date column + all columns 
#     # except the target feature which is assumed to be column after 'date' column
#     X = df.iloc[:, :date_column_index+1].join(df.iloc[:, date_column_index+2:])
#     y = df.iloc[:, date_column_index+1: date_column_index+2]
# =============================================================================

    X_train = X.iloc[:-my_insample_forecast_steps, :]
    X_test = X.iloc[-my_insample_forecast_steps:, :]
    y_train = y.iloc[:-my_insample_forecast_steps, :]
    y_test = y.iloc[-my_insample_forecast_steps:, :]
    
    # initialize variable
    scaler = ""
    
    # Scale the data if user selected a scaler choice in the normalization / standardization in streamlit sidebar
    if scaler_choice != "None":
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
                           raise ValueError("Invalid scaler choice. Please choose from: MinMaxScaler, RobustScaler, MaxAbsScaler, "
                                            "PowerTransformer, QuantileTransformer")

            scaler = scaler_choices[scaler_choice]
            
            # Fit the scaler on the training set and transform both the training and test sets
            X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
            X_train_numeric_scaled = pd.DataFrame(X_train_numeric_scaled, columns=X_train_numeric.columns, index=X_train_numeric.index)
            
            # note: you do not want to fit_transform on the test set else the distribution of the entire dataset is used and is data leakage
            X_test_numeric_scaled = scaler.transform(X_test_numeric)
            X_test_numeric_scaled = pd.DataFrame(X_test_numeric_scaled, columns=X_test_numeric.columns, index=X_test_numeric.index)
            
            # refit the scaler on the entire exogenous features e.g. X which is used for forecasting beyond train/test sets
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

def perform_train_test_split_standardization(X, y, X_train, X_test, y_train, y_test, my_insample_forecast_steps, scaler_choice=None, numerical_features=[]):
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
    # Check If the specified test-set size is greater than or equal to the total number of rows in the dataset.
    if my_insample_forecast_steps >= len(y):
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
                scaler=StandardScaler()
            else:
                raise ValueError("Invalid scaler choice. Please choose from: StandardScaler")
            # Fit the scaler on the training set and transform both the training and test sets
            X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
            X_train_numeric_scaled = pd.DataFrame(X_train_numeric_scaled, columns=X_train_numeric.columns, index=X_train_numeric.index)
            X_test_numeric_scaled = scaler.transform(X_test_numeric)
            X_test_numeric_scaled = pd.DataFrame(X_test_numeric_scaled, columns=X_test_numeric.columns, index=X_test_numeric.index)
            # refit the scaler on the entire exogenous features e.g. X which is used for forecasting beyond train/test sets
            X_numeric_scaled = scaler.fit_transform(X_numeric)
            # Convert the scaled array back to a DataFrame and set the column names
            X_numeric_scaled = pd.DataFrame(X_numeric_scaled, columns=X_numeric.columns, index=X_numeric.index)
    # Replace the original
    if scaler_choice != "None":
        X_train[numerical_features] = X_train_numeric_scaled[numerical_features]
        X_test[numerical_features] = X_test_numeric_scaled[numerical_features]
        X[numerical_features] = X_numeric_scaled[numerical_features]         
    # Return the training and testing sets as well as the scaler used (if any)
    return X, y, X_train, X_test, y_train, y_test

def train_test_split_slider(df):
    """
   Creates a slider for the user to choose the number of days or percentage for train/test split.

   Args:
       df (pd.DataFrame): Input DataFrame.

   Returns:
       tuple: A tuple containing the number of days and percentage for the in-sample forecast.
   """
    # update the session state of the form and make it persistent when switching pages
    def form_callback():
        st.session_state['percentage'] = st.session_state['percentage']
        st.session_state['steps'] = st.session_state['steps']
    
    with st.sidebar:
        with st.form('train/test split'):
            my_text_paragraph('Train/Test Split')
            col1, col2 = st.columns(2)
            with col1:
                split_type = st.radio(label = "*Select split type:*", 
                                      options = ("Steps", "Percentage"), 
                                      index = 1,
                                      help = """
                                             Set your preference for how you want to **split** the training data and test data:
                                             \n- as a `percentage` (between 1% and 99%)  \
                                             \n- in `steps` (for example number of days with daily data, number of weeks with weekly data, etc.)
                                             """)
               
                if split_type == "Steps":
                    with col2:
                        insample_forecast_steps = st.slider(label = '*Size of the test-set in steps:*', 
                                                            min_value=1, 
                                                            max_value=len(df)-1, 
                                                            step=1, 
                                                            key='steps')
                        
                        insample_forecast_perc =  st.session_state['percentage']
                else:
                    with col2:
                        insample_forecast_perc = st.slider('*Size of the test-set as percentage*', min_value=1, max_value=99, step=1, key='percentage')
                        insample_forecast_steps = round((insample_forecast_perc / 100) * len(df))
           
            # show submit button in streamlit centered in sidebar
            col1, col2, col3 = st.columns([4,4,4])
            with col2:       
                train_test_split_btn = st.form_submit_button("Submit", type="secondary", on_click=form_callback)        
    return insample_forecast_steps, insample_forecast_perc

def compute_importance_scores(X, y, estimator):
    """
    Compute feature importance scores using permutation importance.

    Parameters:
    -----------
    X : pandas.DataFrame
        The feature matrix.
    y : pandas.Series
        The target variable.
    estimator : object
        The model used to fit the data.

    Returns:
    --------
    importance_scores : dict
        A dictionary containing the importance scores for each feature.
    """
    # Compute feature importance scores using permutation importance
    # LinearRegression
    result = permutation_importance(estimator, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    # Create dictionary of importance scores
    importance_scores = {}
    for i in range(len(X.columns)):
        importance_scores[X.columns[i]] = result.importances_mean[i]
    return importance_scores

def get_feature_list(X):
    """
    Returns a comma-separated string of column names in a Pandas DataFrame.
    
    Parameters:
    X (Pandas DataFrame): the input DataFrame to get the column names from.
    
    Returns:
    A comma-separated string of column names in the input DataFrame.
    
    Example:
    >>> X = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    >>> get_feature_list(X)
    'col1, col2'
    """
    return ', '.join(X.columns)

def plot_forecast(df_actual, df_forecast, title=''):
    """
    Plot the actual and forecasted time series data on a line chart using Plotly.
    
    Args:
    - df_actual (pd.DataFrame): A Pandas DataFrame containing the actual time series data with DatetimeIndex as the index.
    - df_forecast (pd.DataFrame): A Pandas DataFrame containing the forecasted time series data with DatetimeIndex as the index and a 'forecast' column.
    - title (str, optional): The title of the chart. Default is an empty string.
    
    Returns:
    - None: Displays the chart in Streamlit using `st.plotly_chart`.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_actual.index, y=df_actual.iloc[:,0], name='Actual', mode='lines'))
    fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast['forecast'], name='Forecast', mode='lines', line=dict(dash='dot', color='#87CEEB'))) # dash styles: ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'] 
    # add buttons for showing actual data, forecast data, and both actual and forecast data
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                xanchor="center",
                yanchor="middle",
                x=0.5,
                y=1.15,
                buttons=list([
                    dict(label="Actual",
                         method="update",
                         args=[{"visible": [True, False]},
                               {}]),
                    dict(label="Forecast",
                         method="update",
                         args=[{"visible": [False, True]},
                               {}]),
                    dict(label="Actual + Forecast",
                         method="update",
                         args=[{"visible": [True, True]},
                               {}])
                ]),
            )
        ],
        title=title, 
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.99)
    )
    # show plot in streamlit
    st.plotly_chart(fig, use_container_width=True)

def create_date_features(df, year_dummies=True, month_dummies=True, day_dummies=True):
    '''
    This function creates dummy variables for year, month, and day of week from a date column in a pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): Input DataFrame containing a date column.
    year_dummies (bool): Flag to indicate if year dummy variables are needed. Default is True.
    month_dummies (bool): Flag to indicate if month dummy variables are needed. Default is True.
    day_dummies (bool): Flag to indicate if day of week dummy variables are needed. Default is True.

    Returns:
    pandas.DataFrame: A new DataFrame with added dummy variables.
    '''
    if year_dummies:
        df['year'] = df['date'].dt.year
        dum_year = pd.get_dummies(df['year'], columns=['year'], drop_first=True, prefix='year', prefix_sep='_')
        df = pd.concat([df, dum_year], axis=1)
    if month_dummies:
        month_dict = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
        df['month'] = df['date'].dt.month.apply(lambda x: month_dict.get(x))
        dum_month = pd.get_dummies(df['month'], columns=['month'], drop_first=True, prefix='', prefix_sep='')
        df = pd.concat([df, dum_month], axis=1)
    if day_dummies:
        week_dict = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
        df['day'] = df['date'].dt.weekday.apply(lambda x: week_dict.get(x))
        dum_day = pd.get_dummies(df['day'], columns=['day'], drop_first=True, prefix='', prefix_sep='')
        df = pd.concat([df, dum_day], axis=1)
    # Drop any duplicate columns based on their column names
    df = df.loc[:,~df.columns.duplicated()]
    return df

def create_calendar_holidays(df, slider=True):   
    """
    Create a calendar of holidays for a given DataFrame.
    
    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing a 'date' column.
    
    Returns:
    - pandas.DataFrame: The input DataFrame with additional columns 'is_holiday' (1 if the date is a holiday, 0 otherwise)
                        and 'holiday_desc' (description of the holiday, empty string if not a holiday).
    """
    try:
        # Note: create_calendar_holidays FUNCTION BUILD ON TOP OF HOLIDAY PACKAGE
        # some countries like Algeria have issues therefore if/else statement to catch it
        # Define variables
        start_date = df['date'].min()
        end_date = df['date'].max()
        
        # Available countries and their country codes
        country_data = [
                        ('Albania', 'AL'),
                        ('Algeria', 'DZ'),
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
        # retrieve index of default country e.g. 'United States of America'
        us_index = country_data.index(('United States of America', 'US'))

        # add slider if user on the page with menu_item = 'Engineer' otherwise do not show selectbox
        if slider == True:
            
            col1, col2, col3 = st.columns([1,3,1])
            with col2:
                
                with st.form('country_holiday'):
                    
                    selected_country_name = st.selectbox(label = "Select a country", 
                                                         options = [country[0] for country in country_data], 
                                                         key = key1_engineer_page_country, 
                                                         label_visibility = 'collapsed')
                    
                    col1, col2, col3 = st.columns([5,4,4])
                    with col2:
                       country_holiday_btn = st.form_submit_button('Apply', on_click = form_update, args = ("ENGINEER_PAGE_COUNTRY_HOLIDAY",))
                                              
                       # update country code as well in session state - which is used for prophet model holiday feature as well
                       set_state("ENGINEER_PAGE_COUNTRY_HOLIDAY", ("country_code",  dict(country_data).get(selected_country_name)))

        # else do not add selectbox in if function is called when user not on page
        else:
            selected_country_name = get_state("ENGINEER_PAGE_COUNTRY_HOLIDAY", "country_name")
            
        # create empty container for the calendar
        country_calendars = {}
        
        # iterate over all countries and try-except block for if holiday for country is not found in holiday python package
        for name, code in country_data:
            try:
                country_calendars[name] = getattr(holidays, code)()
            except AttributeError:
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    print(f"No holiday calendar found for country: {name}")
                continue
        
        # Retrieve the country code for the selected country
        selected_country_code = dict(country_data).get(get_state("ENGINEER_PAGE_COUNTRY_HOLIDAY", "country_name"))
        #st.write(selected_country_code, selected_country_name) # TEST
        
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

          #st.write(df_country_holidays) # TEST IF TWO COLUMNS ARE CREATED CORRECTLY FOR COUNTRY HOLIDAYS
          
          # merge dataframe of index with dates, is_holiday, holiday_desc with original df
          df = pd.merge(df, df_country_holidays, left_on='date', right_index=True, how='left')
          #st.write(df) # TEST IF MERGE WAS SUCCESFULL
          return df
          
        else:
            col1, col2, col3 = st.columns([1,2,1])
            with col2:  
                return st.error(f"⚠️No holiday calendar found for country: **{selected_country_name}**")
    except:
        st.error('Forecastgenie Error: the function create_calendar_holidays() could not execute correctly, please contact the administrator...')

def create_calendar_special_days(df, start_date_calendar=None,  end_date_calendar=None, special_calendar_days_checkbox=True):
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
    if start_date_calendar == None:
        start_date_calendar = df['date'].min()
    else:
        start_date_calendar = start_date_calendar
    if end_date_calendar == None:
        end_date_calendar = df['date'].max()
    else:
        end_date_calendar = end_date_calendar
    df_exogenous_vars = pd.DataFrame({'date': pd.date_range(start = start_date_calendar, end = end_date_calendar)})
    
    class UKEcommerceTradingCalendar(AbstractHolidayCalendar):
        rules = []
        # Seasonal trading events
        # only add Holiday if user checkmarked checkbox (e.g. equals True) 
        if get_state("ENGINEER_PAGE_VARS", "jan_sales"):
            rules.append(Holiday('January sale', month = 1, day = 1))
        if get_state("ENGINEER_PAGE_VARS", "val_day_lod"):
            rules.append(Holiday('Valentine\'s Day [last order date]', month = 2, day = 14, offset = BDay(-2)))
        if get_state("ENGINEER_PAGE_VARS", "val_day"):
            rules.append(Holiday('Valentine\'s Day', month = 2, day = 14))
        if get_state("ENGINEER_PAGE_VARS", "mother_day_lod"):
            rules.append(Holiday('Mother\'s Day [last order date]', month = 5, day = 1, offset = BDay(-2)))
        if get_state("ENGINEER_PAGE_VARS", "mother_day"):
            rules.append(Holiday('Mother\'s Day', month = 5, day = 1, offset = pd.DateOffset(weekday = SU(2))))
        if get_state("ENGINEER_PAGE_VARS", "father_day_lod"):
            rules.append(Holiday('Father\'s Day [last order date]', month = 6, day = 1, offset = BDay(-2)))
        if get_state("ENGINEER_PAGE_VARS", "father_day"):
            rules.append(Holiday('Father\'s Day', month = 6, day = 1, offset = pd.DateOffset(weekday = SU(3))))
        if get_state("ENGINEER_PAGE_VARS", "black_friday_lod"):
            rules.append(Holiday("Black Friday [sale starts]", month = 11, day = 1, offset = [pd.DateOffset(weekday = SA(4)), BDay(-5)]))
        if get_state("ENGINEER_PAGE_VARS", "black_friday"):
            rules.append(Holiday('Black Friday', month = 11, day = 1, offset = pd.DateOffset(weekday = FR(4))))
        if get_state("ENGINEER_PAGE_VARS", "cyber_monday"):
            rules.append(Holiday("Cyber Monday", month = 11, day = 1, offset = [pd.DateOffset(weekday = SA(4)), pd.DateOffset(2)]))
        if get_state("ENGINEER_PAGE_VARS", "christmas_day"):
            rules.append(Holiday('Christmas Day [last order date]', month = 12, day = 25, offset = BDay(-2)))
        if get_state("ENGINEER_PAGE_VARS", "boxing_day"):
            rules.append(Holiday('Boxing Day sale', month = 12, day = 26))       
    
    calendar = UKEcommerceTradingCalendar()
    start = df_exogenous_vars.date.min()
    end = df_exogenous_vars.date.max()
    events = calendar.holidays(start = start, end = end, return_name = True)
    events = events.reset_index(name = 'calendar_event_desc').rename(columns = {'index': 'date'})
    df_exogenous_vars = df_exogenous_vars.merge(events, on = 'date', how = 'left').fillna('')
    # source: https://splunktool.com/holiday-calendar-in-pandas-dataframe
    
    ###############################################
    # Create Pay Days 
    ###############################################
    class UKEcommerceTradingCalendar(AbstractHolidayCalendar):
        rules = []
        # Pay days(based on fourth Friday of the month)
        if get_state("ENGINEER_PAGE_VARS", "pay_days") == True:
            rules = [
                    Holiday('January Pay Day', month = 1, day = 31, offset = BDay(-1)),
                    Holiday('February Pay Day', month = 2, day = 28, offset = BDay(-1)),
                    Holiday('March Pay Day', month = 3, day = 31, offset = BDay(-1)),
                    Holiday('April Pay Day', month = 4, day = 30, offset = BDay(-1)),
                    Holiday('May Pay Day', month = 5, day = 31, offset = BDay(-1)),
                    Holiday('June Pay Day', month = 6, day = 30, offset = BDay(-1)),
                    Holiday('July Pay Day', month = 7, day = 31, offset = BDay(-1)),
                    Holiday('August Pay Day', month = 8, day = 31, offset = BDay(-1)),
                    Holiday('September Pay Day', month = 9, day = 30, offset = BDay(-1)),
                    Holiday('October Pay Day', month = 10, day = 31, offset = BDay(-1)),
                    Holiday('November Pay Day', month = 11, day = 30, offset = BDay(-1)),
                    Holiday('December Pay Day', month = 12, day = 31, offset = BDay(-1))
                    ]
    calendar = UKEcommerceTradingCalendar()
    start = df_exogenous_vars.date.min()
    end = df_exogenous_vars.date.max()
    events = calendar.holidays(start = start, end = end, return_name = True)
    events = events.reset_index(name = 'pay_day_desc').rename(columns = {'index': 'date'})
    df_exogenous_vars = df_exogenous_vars.merge(events, on = 'date', how = 'left').fillna('')
    df_exogenous_vars['pay_day'] = df_exogenous_vars['pay_day_desc'].apply(lambda x: 1 if len(x) > 1 else 0)
    df_exogenous_vars['calendar_event'] = df_exogenous_vars['calendar_event_desc'].apply(lambda x: 1 if len(x)>1 else 0)
    
    ###############################################################################
    # Reorder Columns to logical order e.g. value | description of value
    ###############################################################################
    df_exogenous_vars = df_exogenous_vars[['date', 'calendar_event', 'calendar_event_desc', 'pay_day','pay_day_desc']]
    
    ###############################################################################
    # combine exogenous vars with df
    ###############################################################################
    df_total_incl_exogenous = pd.merge(df, df_exogenous_vars, on='date', how='left' )
    df = df_total_incl_exogenous.copy(deep=True)
    return df 

def my_subheader_metric(string1, color1="#cfd7c2", metric=0, color2="#FF0000", my_style="#000000", my_size=5):
    metric_rounded = "{:.2%}".format(metric)
    metric_formatted = f"<span style='color:{color2}'>{metric_rounded}</span>"
    string1 = string1.replace(f"{metric}", f"<span style='color:{color1}'>{metric_rounded}</span>")
    st.markdown(f'<h{my_size} style="color:{my_style};"> <center> {string1} {metric_formatted} </center> </h{my_size}>', unsafe_allow_html=True)

def wait(seconds):
    start_time = time.time()
    with st.spinner(f"Please wait... {int(time.time() - start_time)} seconds passed"):
        time.sleep(seconds)     

def my_holiday_name_func(my_date):
    """
    This function takes a date as input and returns the name of the holiday that falls on that date.

    Parameters:
    -----------
    my_date : str
        The date for which the holiday name is to be returned. The date should be in the format 'YYYY-MM-DD'.

    Returns:
    --------
    str:
        The name of the holiday that falls on the given date. If there is no holiday on that date, an empty string is returned.

    Examples:
    ---------
    >>> my_holiday_name_func('2022-07-04')
    'Independence Day'
    >>> my_holiday_name_func('2022-12-25')
    'Christmas Day'
    >>> my_holiday_name_func('2022-09-05')
    'Labor Day'
    """    
    holiday_name = calendar().holidays(start = my_date, end = my_date, return_name = True)
    if len(holiday_name) < 1:
      holiday_name = ""
      return holiday_name
    else: 
     return holiday_name[0]

def my_metrics(my_df, model_name):
    """
    Calculates performance metrics for a given model.
    
    Args:
        my_df (pandas.DataFrame): DataFrame containing actual and predicted values for evaluation.
            Columns:
                - 'Actual': Actual target variable values.
                - 'Predicted': Predicted target variable values.
        model_name (str): Name of the model.
    
    Returns:
        mape (float): Mean Absolute Percentage Error (MAPE) for the model.
        rmse (float): Root Mean Squared Error (RMSE) for the model.
        r2 (float): R-squared (coefficient of determination) for the model.
        smape (float): Symmetric Mean Absolute Percentage Error (SMAPE) for the model.
    
    Note:
        - The function calculates MAPE, RMSE, R-squared, and SMAPE metrics based on the provided DataFrame.
        - MAPE is the average percentage difference between predicted and actual values.
        - RMSE is the square root of the mean squared error between predicted and actual values.
        - R-squared measures the proportion of the variance in the target variable explained by the model.
        - SMAPE is the symmetric mean absolute percentage error between predicted and actual values.
        - The function updates the 'metrics_dict' dictionary with the calculated metrics for the given model.
    
    Example:
        result = my_metrics(df_preds, 'MyModel')
        mape_value, rmse_value, r2_value, smape_value = result
    """
    # Check if not empty dataframe
    if not my_df.empty:
        # Calculate metrics
        mape = np.mean(np.abs((my_df['Actual'] - my_df['Predicted']) / my_df['Actual']))
        smape = (1 / len(my_df)) * np.sum(np.abs(my_df['Actual'] - my_df['Predicted']) / (np.abs(my_df['Actual']) + np.abs(my_df['Predicted'])))
        mse = mean_squared_error(my_df['Actual'], my_df['Predicted'])
        rmse = np.sqrt(mse)
        r2 = r2_score(my_df['Actual'], my_df['Predicted'])
      
    else:
        # Set metrics equal to none if df is empty
        mape = None
        smape = None
        mse = None
        rmse = None
        r2 = None
        
    # Add the results to the dictionary
    metrics_dict[model_name] = {'mape': mape, 'smape': smape, 'mse': mse, 'rmse': rmse, 'r2': r2}
    
    return metrics_dict

def display_my_metrics(my_df, model_name = "", my_subtitle = None):
    """
    Displays the evaluation metrics for a given model using the provided dataframe of predicted and actual values.
    
    Parameters:
    -----------
    my_df : pandas.DataFrame
        Dataframe containing columns 'Actual' and 'Predicted' for the actual and predicted values, respectively.
    model_name : str, default='Linear Regression'
        Name of the model being evaluated. This will be displayed as the title of the expander containing the metrics.
    
    Returns:
    --------
    None
    """
    # put all metrics and graph in expander for linear regression e.g. benchmark model
    #st.markdown(f'<h2 style="text-align:center">{model_name}</h2></p>', unsafe_allow_html=True)
    my_text_header(f'{model_name}')
    
    if my_subtitle is not None:
        my_text_paragraph(my_string = my_subtitle, my_text_align='center', my_font_weight=200, my_font_size='14px')
    
    vertical_spacer(2)
    
    # define vertical spacings
    col0, col1, col2, col3, col4, col5 = st.columns([2, 3, 3, 3, 3, 1])
    
    # Get the evaluation metrics for your model
    metrics_dict = my_metrics(my_df, model_name)
    
    # define the font size of the st.metric
    st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 18px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    
    # Show the evaluation metrics on Model Card
    with col1:  
        st.metric(label = ':red[MAPE]', 
                  value = "{:.2%}".format(metrics_dict[model_name]['mape']),
                  help = 'The `Mean Absolute Percentage Error` (MAPE) measures the average absolute percentage difference between the predicted values and the actual values. It provides a relative measure of the accuracy of a forecasting model. A lower MAPE indicates better accuracy.')
    with col2:
        st.metric(label = ':red[SMAPE]', 
                  value = "{:.2%}".format(metrics_dict[model_name]['smape']),
                  help= 'The `Symmetric Mean Absolute Percentage Error` (SMAPE) is similar to MAPE but addresses the issue of scale dependency. It calculates the average of the absolute percentage differences between the predicted values and the actual values, taking into account the magnitude of both values. SMAPE values range between 0% and 100%, where lower values indicate better accuracy.')
    with col3:
        st.metric(label = ':red[RMSE]', 
                  value = round(metrics_dict[model_name]['rmse'],2),
                  help = 'The `Root Mean Square Error` (RMSE) is a commonly used metric to evaluate the accuracy of a prediction model. It calculates the square root of the average of the squared differences between the predicted values and the actual values. RMSE is sensitive to outliers and penalizes larger errors.')
    with col4: 
        st.metric(label = ':green[R²]',  
                  value= round(metrics_dict[model_name]['r2'], 2),
                  help = '`R-squared` (R²) is a statistical measure that represents the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, where 1 indicates a perfect fit. R-squared measures how well the predicted values fit the actual values and is commonly used to assess the goodness of fit of a regression model.')

def forecast_naive_model1_insample(y_test, lag=None, custom_lag_value=None):
    """
    Implements Naive Model I for time series forecasting.

    Args:
        y_test: The target variable of the test set.
        lag: The lag value to use for shifting the target variable. Options are 'day', 'week', 'month', 'year', or 'custom'.
        custom_lag_value: The custom lag value to use if lag='custom'.

    Returns:
        df_preds (pandas.DataFrame): DataFrame containing actual and predicted values for evaluation.
            Columns:
                - 'Actual': Actual target variable values.
                - 'Predicted': Predicted target variable values.
                - 'Percentage_Diff': Percentage difference between predicted and actual values.
                - 'MAPE': Mean Absolute Percentage Error between predicted and actual values.

    Raises:
        ValueError: If an invalid value for 'lag' is provided, or if custom_lag_value is not provided when lag='custom'.

    Note:
        - The function applies a lag to the target variable based on the 'lag' argument.
        - It creates a DataFrame with actual and predicted values.
        - The index is set to the date portion of the datetime index.
        - Rows with missing values are dropped.
        - Percentage difference and MAPE (Mean Absolute Percentage Error) are calculated.

    Example:
        lag_value = 'day'
        custom_lag_value = 5
        result = forecast_naive_model1_insample(y_test, lag=lag_value, custom_lag_value=5)
    """

    if lag == 'day':
        y_pred = y_test.shift(1)
    elif lag == 'week':
        y_pred = y_test.shift(7)
    elif lag == 'month':
        y_pred = y_test.shift(30)
    elif lag == 'year':
        y_pred = y_test.shift(365)
    elif lag == 'custom':
        if custom_lag_value is None:
            raise ValueError('Custom lag value is required when lag is set to "custom".')
        y_pred = y_test.shift(custom_lag_value)
    else:
        raise ValueError('Invalid value for "lag". Must be "day", "week", "month", "year", or "custom".')

    # Create a DataFrame for insample predictions versus actual
    df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})

    # Set the index to just the date portion of the datetime index
    df_preds.index = df_preds.index.date

    # Drop rows with N/A values
    df_preds.dropna(inplace = True)

    # Calculate the percentage difference between actual and predicted values and add it as a new column
    df_preds['Error'] = (df_preds['Actual'] - df_preds['Predicted'])
    df_preds['Error (%)'] = (df_preds['Error'] / df_preds['Actual'])

    return df_preds

def forecast_naive_model2_insample(y_test, size_rolling_window, agg_method_rolling_window):
    """
    Implements Naive Model 2 for time series forecasting with a rolling window approach.
    
    Args:
        y_test: The target variable of the test set.
        size_rolling_window (int): The size of the rolling window.
        agg_method_rolling_window (str): The aggregation method to apply within the rolling window. Options are
                                          'mean', 'median', 'mode', or custom aggregation functions.
    
    Returns:
        df_preds (pandas.DataFrame): DataFrame containing actual and predicted values for evaluation.
            Columns:
                - 'Actual': Actual target variable values.
                - 'Predicted': Predicted target variable values.
                - 'Percentage_Diff': Percentage difference between predicted and actual values.
                - 'MAPE': Mean Absolute Percentage Error between predicted and actual values.
    
    Raises:
        ValueError: If an invalid value for 'agg_method_rolling_window' is provided.
    
    Note:
        - The function uses a rolling window approach to make predictions.
        - The rolling window size and aggregation method are specified by the arguments.
        - It creates a DataFrame with actual and predicted values.
        - The index is set to the date portion of the datetime index.
        - Rows with missing values are dropped.
        - Percentage difference and MAPE (Mean Absolute Percentage Error) are calculated.
    
    Example:
        size_window = 7
        agg_method = 'mean'
        result = forecast_naive_model2_insample(y_test, size_rolling_window=size_window, agg_method_rolling_window=agg_method)
    """
    agg_method_rolling_window = agg_method_rolling_window.lower()
    
    # Apply rolling window and aggregation
    if agg_method_rolling_window == 'mean':
        y_pred = y_test.rolling(size_rolling_window).mean()
    elif agg_method_rolling_window == 'median':
        y_pred = y_test.rolling(size_rolling_window).median()
    elif agg_method_rolling_window == 'mode':
        y_pred = y_test.rolling(size_rolling_window).apply(lambda x: x.value_counts().index[0])
    elif callable(agg_method_rolling_window):
        y_pred = y_test.rolling(size_rolling_window).apply(agg_method_rolling_window)
    else:
        raise ValueError('Invalid value for "agg_method_rolling_window". Must be "mean", "median", "mode", '
                         'or a valid custom aggregation function.')
            
    # Create dataframe for insample predictions versus actual
    df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
        
    # Set the index to just the date portion of the datetime index
    df_preds.index = df_preds.index.date
        
    # Drop rows with N/A values
    df_preds.dropna(inplace=True)
        
    # Calculate the percentage difference between actual and predicted values and add it as a new column
    df_preds['Error'] = (df_preds['Actual'] - df_preds['Predicted'])
    
    df_preds['Error (%)'] = (df_preds['Error'] / df_preds['Actual'])
    
    return df_preds

def forecast_naive_model3_insample(y_train, y_test, agg_method):
    """
    Implements Naive Model for time series forecasting without a rolling window approach.
    
    Args:
        X_train: The input features of the training set.
        y_train: The target variable of the training set.
        X_test: The input features of the test set.
        y_test: The target variable of the test set.
        agg_method (str): The aggregation method to apply to the entire time series. Options are 'mean', 'median', or 'mode'.
    
    Returns:
        df_preds (pandas.DataFrame): DataFrame containing actual and predicted values for evaluation.
            Columns:
                - 'Actual': Actual target variable values.
                - 'Predicted': Predicted target variable values.
                - 'Percentage_Diff': Percentage difference between predicted and actual values.
                - 'MAPE': Mean Absolute Percentage Error between predicted and actual values.
    
    Raises:
        ValueError: If an invalid value for 'agg_method' is provided.
    
    Note:
        - The function calculates the mean, median, or mode for the entire time series.
        - It creates a DataFrame with actual and predicted values.
        - The index is set to the date portion of the datetime index.
        - Rows with missing values are dropped.
        - Percentage difference and MAPE (Mean Absolute Percentage Error) are calculated.
    
    Example:
        agg_method = 'mean'
        result = forecast_naive_model_insample(X_train, y_train, X_test, y_test, agg_method)
    """
    agg_method = agg_method.lower()
    
    # Apply aggregation
    if agg_method == 'mean':
        my_metric = y_train.mean().values[0]
    elif agg_method == 'median':
        my_metric = y_train.median().values[0]
    elif agg_method == 'mode':
        my_metric = y_train.iloc[0].mode().iloc[0] # TEST IF CORRECTLY CALCULATED
    else:
        raise ValueError('Invalid value for "agg_method". Must be "mean", "median", or "mode".')
    
    # Create DataFrame with datetime index and metric values
    y_pred = pd.DataFrame(index=y_test.index, data=my_metric, columns=['Values'])
# =============================================================================
#     st.write(y_train, y_test) # TEST
#     st.write(y_pred) # TEST
# =============================================================================

    # Create dataframe for insample predictions versus actual
    df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
        
    #st.write(df_preds)
    
    # Set the index to just the date portion of the datetime index
    df_preds.index = df_preds.index.date
        
    # Drop rows with N/A values
    df_preds.dropna(inplace=True)
        
    # Calculate the percentage difference between actual and predicted values and add it as a new column
    df_preds['Error'] = (df_preds['Actual'] - df_preds['Predicted'])
    df_preds['Error (%)'] = (df_preds['Error'] / df_preds['Actual'])
    
    return df_preds
 
def evaluate_regression_model(model, X_train, y_train, X_test, y_test, **kwargs):
    """
    Evaluate a regression model on test data.

    Parameters:
        X_train (pd.DataFrame): Training input data.
        y_train (pd.DataFrame): Training output data.
        X_test (pd.DataFrame): Test input data.
        y_test (pd.DataFrame): Test output data.
        model_type (str): Type of regression model to use. Default is "linear".
        lag (str): Lag for seasonal naive model. Options are "day", "week", "month", and "year". Default is None.
        **kwargs: Optional keyword arguments to pass to the regression model.

    Returns:
        df_preds (pd.DataFrame): DataFrame of predicted and actual values on test data.
    """
    # Train the model using the training/test sets if selectbox equal to 'Yes' to include features (default option)
    # if not then only use the target feature and the date column converted to numeric        
    if get_state("TRAIN_PAGE", "include_feature_selection") == 'No':
        # create numeric date values e.g. 0,1,2,3 through N of your data (whereby X_test (e.g. first value = 101) continues counting from last value of X_train (e.g. last value = 100))
        X_train = pd.DataFrame(range(len(X_train)), index=X_train.index, columns=['date_numeric'])
        X_test = pd.DataFrame(range(len(X_train), len(X_train) + len(X_test)), index=X_test.index, columns=['date_numeric'])
    else:
        pass
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # =============================================================================
    # COEFFICIENTS TABLE   
    # =============================================================================
    # Extract the coefficient values and feature names
    coefficients = model.coef_
    #st.write(coefficients) # TEST
    
    intercept = model.intercept_
    #st.write(intercept) # TEST
    
    feature_names = X_train.columns.tolist()
    #st.write(feature_names) # TEST
    
    # Create a table to display the coefficients and intercept
    coefficients_table = pd.DataFrame({'Feature': ['Intercept'] + feature_names, 'Coefficient': np.insert(coefficients, 0, intercept)})
    #st.write(coefficients_table)
    
    # =============================================================================
    # LINEAR REGRESSION EQUATION
    # =============================================================================
    # Get the intercept and coefficients from the table
    intercept = coefficients_table.loc[0, 'Coefficient'].round(2)
    weights = coefficients_table.loc[1:, 'Coefficient'].round(2)
    feature_names = coefficients_table.loc[1:, 'Feature']
    
    # Reverse the weights and feature_names lists
    weights = weights[::-1]
    feature_names = feature_names[::-1]
    
    # Create the symbols for the features and coefficients
    x_symbols = sp.symbols(feature_names)
    coefficients_symbols = sp.symbols(['b0'] + [f'b{i}' for i in range(1, len(x_symbols) + 1)])
    
    # Create the custom equation string
    equation_parts = [f"{coeff} * {feat}" for coeff, feat in zip(weights, feature_names)]
    equation_str = f"y_hat = {intercept} + " + " + ".join(equation_parts)
    
    # Create a dictionary to substitute the coefficient symbols with their values
    subs_dict = {coefficients_symbols[i]: coeff for i, coeff in enumerate(weights)}
    subs_dict[coefficients_symbols[0]] = intercept
    
    # Substitute the coefficient symbols in the equation expression
    for symbol, value in subs_dict.items():
        equation_str = equation_str.replace(str(symbol), str(value))

    # Create dataframe for insample predictions versus actual
    df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
    
    # set the index to just the date portion of the datetime index
    df_preds.index = df_preds.index.date
    
    # Drop rows with N/A values
    df_preds.dropna(inplace=True)
    
    # Calculate the percentage difference between actual and predicted values and add it as a new column
    df_preds['Error'] = (df_preds['Actual'] - df_preds['Predicted'])
    df_preds['Error (%)'] = df_preds['Error'] / df_preds['Actual']
    
    return df_preds, coefficients_table, equation_str

def evaluate_sarimax_model(order, seasonal_order, exog_train, exog_test, endog_train, endog_test):
    """
    Train and evaluate SARIMAX model on test data.
    
    Parameters:
        order (tuple): Order of the SARIMAX model (p,d,q).
        seasonal_order (tuple): Seasonal order of the SARIMAX model (P,D,Q,s).
        exog_train (pd.DataFrame): Exogenous variables training data.
        exog_test (pd.DataFrame): Exogenous variables test data.
        endog_train (pd.DataFrame): Endogenous variable training data.
        endog_test (pd.DataFrame): Endogenous variable test data.
    
    Returns:
        preds_df (pd.DataFrame): DataFrame of predicted and actual values on test data.
    """
    # Step 0: check if user wants to include feature selection and if not create a date_numeric for X_train and X_test (exogenous variable)
    if get_state("TRAIN_PAGE", "include_feature_selection") == 'No':
        # create numeric date values e.g. 0,1,2,3 through N of your data (whereby X_test (e.g. first value = 101) continues counting from last value of X_train (e.g. last value = 100))
        exog_train = pd.DataFrame(range(len(exog_train)), index = exog_train.index, columns=['date_numeric'])
        exog_test = pd.DataFrame(range(len(exog_train), len(exog_train) + len(exog_test)), index = exog_test.index, columns=['date_numeric'])
    else:
        pass
    
    # Step 1: Define the Model
    model = sm.tsa.statespace.SARIMAX(endog = endog_train, 
                                      exog = exog_train, 
                                      order = order, 
                                      seasonal_order = seasonal_order)

    # Step 2: Fit the model e.g. Train the model
    results = model.fit()

    # Step 3: Generate predictions
    y_pred = results.predict(start = endog_test.index[0], 
                             end = endog_test.index[-1], 
                             exog = exog_test)

    # Step 4: Combine Actual values and Prediction in a single dataframe
    df_preds = pd.DataFrame({'Actual': endog_test.squeeze(), 
                             'Predicted': y_pred.squeeze()}, 
                             index = endog_test.index)

    # Step 5: Calculate Difference
    # Calculate the percentage difference between actual and predicted values and add it as a new column
    df_preds['Error'] = (df_preds['Actual'] - df_preds['Predicted'])
    
    df_preds['Error (%)'] = (df_preds['Error'] / df_preds['Actual'])

    return df_preds 

def create_streamlit_model_card(X_train, y_train, X_test, y_test, results_df, model, model_name):
    """
    Creates a Streamlit expander card displaying metrics, a graph, and a dataframe for a given model.
    
    Parameters:
    -----------
    X_train: numpy.ndarray or pandas.DataFrame
        The training set inputs.
    y_train: numpy.ndarray or pandas.Series
        The training set outputs.
    X_test: numpy.ndarray or pandas.DataFrame
        The test set inputs.
    y_test: numpy.ndarray or pandas.Series
        The test set outputs.
    model: estimator object
        The trained machine learning model to evaluate.
    model_name: str, default="your model name"
        The name of the machine learning model being evaluated. This will be displayed in the expander header.
    **kwargs: optional keyword arguments
        Additional keyword arguments to be passed to the evaluate_regression_model() function.
        
    Returns:
    --------
    None
    """
    # Evaluate the insample test-set performance linear regression model
    df_preds, coefficients_table, equation_str = evaluate_regression_model(model, X_train, y_train, X_test, y_test)

    with st.expander('📈'+ model_name, expanded=True):
        
        display_my_metrics(my_df = df_preds, 
                           model_name = model_name)
        
        # plot graph with actual versus insample predictions
        plot_actual_vs_predicted(df_preds, 
                                 my_conf_interval)
        
# =============================================================================
#         # =============================================================================
#         #  Show/Hide Button to download dataframe                   
#         # =============================================================================
#         # have button available for user and if clicked it expands with the dataframe
#         col1, col2, col3 = st.columns([100,50,95])
#         with col2:    
#             
#             # create empty placeholder for button show/hide
#             placeholder = st.empty()
#             
#             # create button (enabled to click e.g. disabled=false with unique key)
#             btn = placeholder.button('Show Details', disabled=False,  key = "show_linreg_model_btn")
#         
#         # if button is clicked run below code
#         if btn == True:                       
#             
#             # display button with text "click me again", with unique key
#             placeholder.button('Hide Details', disabled=False, key = "hide_linreg_model_btn")
#             
#             st.markdown('---')
# =============================================================================
                    
        # show the dataframe
        st.dataframe(df_preds.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Percentage_Diff': '{:.2%}', 'MAPE': '{:.2%}'}), 
                     use_container_width = True)
       

        # create download button for forecast results to .csv
        download_csv_button(df_preds, 
                            my_file = "insample_forecast_linear_regression_results.csv", 
                            help_message = f'Download your **{model_name}** model results to .CSV',
                            my_key = 'download_btn_linreg_df_preds')
                   
        st.markdown('---')
       
        my_text_paragraph('Coefficients Table', my_font_size='24px')
       
        vertical_spacer(1)
        
        st.dataframe(coefficients_table, use_container_width=True)
        
        st.markdown('---')
        
        my_text_paragraph('Regression Equation', my_font_size='24px')
        
        vertical_spacer(1)
       
        st.write(equation_str)
        # INTERCEPT ROUNDING SEEMS OFF? # TEST

    vertical_spacer(1)
    
    return df_preds
 
def preprocess_X_prophet(X):
    """
    Preprocess the X dataframe for Prophet modeling.

    Args:
        X (pd.DataFrame): Dataset with date index and independent features.

    Returns:
        pd.DataFrame: Preprocessed X dataframe with 'ds' column and regressor columns.
    """
    if isinstance(X.index, pd.DatetimeIndex):
        X = X.reset_index()

    X_prophet = X.copy()
    X_prophet = X_prophet.rename(columns={X_prophet.columns[0]: 'ds'})
    return X_prophet
      
def preprocess_data_prophet(y_data):
    """
    Preprocess the given data for Prophet model
    """
    # create a deep copy of dataframe e.g. completely new copy of the DataFrame is created with its own memory space
    # This means that any changes made to the new copy will not affect the original DataFrame.
    y_data_prophet = y_data.copy(deep=True)
    
    # Check if the index is already a datetime index and revert it to a column if needed
    if isinstance(y_data_prophet.index, pd.DatetimeIndex):
        y_data_prophet.reset_index(inplace=True)
        
    # REQUIREMENT PROPHET: CREATE DATAFRAME WITH DATE COLUMN 'DS' AND 'Y' column
    y_data_prophet = pd.DataFrame({"ds": y_data_prophet.iloc[:, 0], "y": y_data_prophet.iloc[:, 1]})
    return y_data_prophet

def predict_prophet(y_train, y_test, X, **kwargs):
    """
    Predict future values using Prophet model.

    Args:
        y_train (pd.DataFrame): Training dataset.
        y_test (pd.DataFrame): Test dataset.
        **kwargs: Keyword arguments used to adjust the Prophet model. 
            Allowed keyword arguments:
            - changepoint_prior_scale (float): Parameter for changepoint prior scale.
            - seasonality_mode (str): Parameter for seasonality mode.
            - seasonality_prior_scale (float): Parameter for seasonality prior scale.
            - holidays_prior_scale (float): Parameter for holidays prior scale.
            - yearly_seasonality (bool): Whether to include yearly seasonality.
            - weekly_seasonality (bool): Whether to include weekly seasonality.
            - daily_seasonality (bool): Whether to include daily seasonality.
            - interval_width (float): Width of the uncertainty interval.

    Returns:
        pd.DataFrame: A dataframe with the following columns:
            - Actual: The actual values of the test dataset.
            - Predicted: The predicted values of the test dataset.
            - Error: The difference between actual and predicted values.
            - Error (%): The percentage difference between actual and predicted values.
    """
    # if user set selectbox: 'include feature selection' -> 'Yes' then include the additional explanatory variables/features
    if get_state("TRAIN_PAGE", "include_feature_selection") == 'Yes':
        
        # Step 1: Preprocess Data for Prophet (required are at least 'ds' column and 'y' column)
        #######################
        X_train_prophet = preprocess_X_prophet(X_train)
        y_train_prophet = preprocess_data_prophet(y_train)
        joined_train_data = pd.merge(y_train_prophet, X_train_prophet, on = 'ds')
        #st.write('joined train data', joined_train_data) #TEST
        
        X_test_prophet = preprocess_X_prophet(X_test)
        y_test_prophet = preprocess_data_prophet(y_test)
        joined_test_data = pd.merge(y_test_prophet, X_test_prophet, on = 'ds')
        #st.write('joined test data', joined_test_data) # TEST
        
        # merge train and test data together in 1 dataframe
        merged_data = joined_train_data.append(joined_test_data, ignore_index = True)
        #st.write('merged data', merged_data) # TEST
        
        # Step 2: Define Model
        #######################
        # get the parameters from the settings either preset or adjusted by user and user pressed submit button
        m = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_mode = seasonality_mode,
                    seasonality_prior_scale = seasonality_prior_scale,
                    holidays_prior_scale = holidays_prior_scale,
                    yearly_seasonality = yearly_seasonality,
                    weekly_seasonality = weekly_seasonality,
                    daily_seasonality = daily_seasonality,
                    interval_width = interval_width)
        
        # Step 3: Add independent features/regressors to model
        #######################
        try:
            # ADD COUNTRY SPECIFIC HOLIDAYS IF AVAILABLE FOR SPECIFIC COUNTRY CODE AND USER HAS SELECTBOX SET TO TRUE FOR PROPHET HOLIDAYS
            if get_state("TRAIN_PAGE", "prophet_holidays") == True:        
                m.add_country_holidays(country_name = get_state("ENGINEER_PAGE_COUNTRY_HOLIDAY", "country_code"))
        except:
            st.warning('FORECASTGENIE WARNING: Could not add Prophet Holiday Features to Dataframe. Insample train/test will continue without these features and might lead to less accurate results.')
        
        # iterate over the names of features found from training dataset (X_train) and add them to prophet model as regressors
        for column in X_train.columns:
            m.add_regressor(column)
          
        # Step 4: Fit the model
        #######################
        # train the model on the data with set parameters
        #m.fit(y_train_prophet)
        m.fit(joined_train_data)
        
        # Step 5: Create current date range + future date range
        #######################
        # Predict on the test set
        future = m.make_future_dataframe(periods=len(y_test_prophet), freq='D')
        
        # step 6: Add regressors to future
        #######################
        for column in merged_data.columns:
            future[column] = merged_data[column]
            
        # step 7: forecast
        #######################
        forecast = m.predict(future)
        
    elif get_state("TRAIN_PAGE", "include_feature_selection") == 'No':
        
        # Step 1: Preprocess Data for Prophet (required are at least 'ds' column and 'y' column)
        #######################
        y_train_prophet = preprocess_data_prophet(y_train)
        y_test_prophet = preprocess_data_prophet(y_test)
        
        # Step 2: Define Model
        # Note: get the parameters from the settings either preset or adjusted by user and user pressed submit button
        #######################
        m = Prophet(changepoint_prior_scale = changepoint_prior_scale,
                    seasonality_mode = seasonality_mode,
                    seasonality_prior_scale = seasonality_prior_scale,
                    holidays_prior_scale = holidays_prior_scale,
                    yearly_seasonality = yearly_seasonality,
                    weekly_seasonality = weekly_seasonality,
                    daily_seasonality = daily_seasonality,
                    interval_width = interval_width)
        
        # Step 3: Add Prophet Holiday if User set to True
        #######################
        try:
            # ADD COUNTRY SPECIFIC HOLIDAYS IF AVAILABLE FOR SPECIFIC COUNTRY CODE AND USER HAS SELECTBOX SET TO TRUE FOR PROPHET HOLIDAYS
            if get_state("TRAIN_PAGE", "prophet_holidays") == True:        
                m.add_country_holidays(country_name = get_state("ENGINEER_PAGE_COUNTRY_HOLIDAY", "country_code"))
        except:
            st.warning('FORECASTGENIE WARNING: Could not add Prophet Holiday Features to Dataframe. Insample train/test will continue without these features and might lead to less accurate results.')
        
        # Step 4: Fit the model/Train the model on the data with set parameters
        #######################
        m.fit(y_train_prophet)
        
        # Step 5: Create (current date range + future date range_
        # Note: Prophet model needs entire date range (train+test) versus just test date range used in others models
        #######################
        future = m.make_future_dataframe(periods=len(y_test), freq='D')
        
        # step 6: Add additional regressors to future
        #######################
        # not performed when 'include feature selection' == False
    
        # step 7: forecast
        #######################
        forecast = m.predict(future)
                
    # slice the test-set of the forecast - exclude the forecast on the training set although prophet model does supply it
    # Prophet model provides it to check for overfitting the model, however to prevent user from thinking it trained on whole dataset clearer to provide forecast of test set only
    yhat_test = forecast['yhat'][-len(y_test):]
    preds_df_prophet = pd.DataFrame({'Actual': y_test_prophet['y'].values, 'Predicted': yhat_test.values}, index=y_test_prophet['ds'])
    
    # create column date and set the datetime index to date without the time i.e. 00:00:00
    preds_df_prophet['date'] = preds_df_prophet.index.strftime('%Y-%m-%d')
    
    # set the date column as index column
    preds_df_prophet = preds_df_prophet.set_index('date')
    
    # Calculate absolute error and add it as a new column
    preds_df_prophet['Error'] = preds_df_prophet['Predicted'] - preds_df_prophet['Actual']
    # Calculate percentage difference between actual and predicted values and add it as a new column
    preds_df_prophet['Error (%)'] = (preds_df_prophet['Error'] / preds_df_prophet['Actual'])

    return preds_df_prophet

def load_data():
    """
    This function loads data from a CSV or XLS file and returns a pandas DataFrame object.
    """
    try:
        df_raw = pd.read_csv(uploaded_file, parse_dates=['date'])
    except:
        df_raw = pd.read_excel(uploaded_file, parse_dates=['date'])
    return df_raw

def create_df_pred(y_test, y_pred, show_df_streamlit=True):
    try:
        df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
        # set the index to just the date portion of the datetime index
        df_preds.index = df_preds.index.date
        
        # Calculate percentage difference between actual and predicted values and add it as a new column
        df_preds['Error'] = df_preds['Actual'] - df_preds['Predicted']
        df_preds['Error (%)'] = df_preds['Error'] / df_preds['Actual']
        
        # show the predictions versus actual results
        my_df = df_preds.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Error': '{:.2f}', 'Error (%)': '{:.2f%}'})
        
        if show_df_streamlit == True:
            return st.write(my_df)
        else:
           return df_preds
    except:
        st.warning('error in create_df_pred function')

def convert_df(my_dataframe):
    return my_dataframe.to_csv(index=False).encode('utf-8')

def convert_df_with_index(my_dataframe):
    return my_dataframe.to_csv(index=True).encode('utf-8')

def download_csv_button(my_df, my_file="forecast_model.csv", help_message = 'Download dataframe to .CSV', set_index=False, my_key='define_unique_key'):
    """
    Create a download button for a pandas DataFrame in CSV format.
    
    Parameters:
    -----------
    my_df : pandas.DataFrame
        The DataFrame to be downloaded.
    my_file : str, optional (default="forecast_model.csv")
        The name of the downloaded file.
    help_message : str, optional (default="Download dataframe to .CSV")
        The text displayed when hovering over the download button.
    set_index : bool, optional (default=False)
        If True, sets the DataFrame index as the first column in the output file.
    
    Returns:
    --------
    None
    """
    # create download button for forecast results to .csv
    if set_index:
        csv = convert_df_with_index(my_df)
    else:
        csv = convert_df(my_df)
    col1, col2, col3 = st.columns([54,30,50])
    with col2: 
        st.download_button("🔲 Download", 
                           csv,
                           my_file,
                           "text/csv",
                           #key='', -> streamlit automatically assigns key if not defined
                           help = help_message,
                           key=my_key)

def plot_actual_vs_predicted(df_preds, my_conf_interval):
    """
    Plots the actual and predicted values from a dataframe, along with a shaded confidence interval.
    
    Parameters:
    -----------
    df_preds : pandas.DataFrame
        The dataframe containing the actual and predicted values to plot.
    my_conf_interval : float
        The level of confidence for the confidence interval to display, as a percentage (e.g. 95 for 95% confidence interval).
    
    Returns:
    --------
    None
    """
    # Define the color palette
    colors = ['#5276A7', '#ff7700']
    
    # set color shading of confidence interval
    my_fillcolor = 'rgba(222,235,247,0.5)'
    
    # Create the figure with easy on eyes colors
    fig = px.line(df_preds, x=df_preds.index, y=['Actual', 'Predicted'], color_discrete_sequence=colors)
    
    # Update the layout of the figure
    fig.update_layout(
                        legend_title = 'Legend',
                        font = dict(family='Arial', 
                                    size=12, 
                                    color='#707070'
                                    ),
                        yaxis = dict(
                                    gridcolor='#E1E1E1',
                                    range = [np.minimum(df_preds['Actual'].min(), df_preds['Predicted'].min()) - (df_preds['Predicted'].max() - df_preds['Predicted'].min()), 
                                    np.maximum(df_preds['Actual'].max(), df_preds['Predicted'].max()) + (df_preds['Predicted'].max() - df_preds['Predicted'].min())],
                                    zeroline=False, # remove the x-axis line at y=0
                                    ),
                        xaxis=dict(gridcolor='#E1E1E1'),
                        legend=dict(yanchor="bottom", y=0.0, xanchor="center", x=0.99,  bgcolor= 'rgba(0,0,0,0)' )   
                    )
    
    # Set the line colors
    for i, color in enumerate(colors):
        fig.data[i].line.color = color
        if fig.data[i].name == 'Predicted':
            fig.data[i].line.dash = 'dot' # dash styles options: ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
            
    # Compute the level of confidence 
    confidence = float(my_conf_interval/100)
    # level of significance (1 minus confidence interval)
    alpha = 1 - confidence
    n = len(df_preds)
    
    # Calculates the t-value based on the desired confidence level (alpha) and the sample size (n).
    t_value = stats.t.ppf(1 - alpha / 2, n - 2)
    
    # Calculates the standard error of the estimate, which measures the amount of variability in the data that is not explained by the model.
    std_error = np.sqrt(np.sum((df_preds['Actual'] - df_preds['Predicted']) ** 2) / (n - 2))
    
    # Calculates the margin of error for the confidence interval based on the t-value and the standard error.
    margin_error = t_value * std_error
    upper = df_preds['Predicted'] + margin_error
    lower = df_preds['Predicted'] - margin_error
    
    # Add shaded confidence interval to the plot
    fig.add_trace(go.Scatter(
        x=df_preds.index,
        y=upper,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        fillcolor=my_fillcolor,
        name='Upper Confidence Interval',
        showlegend=False,
        legendgroup='confidence intervals'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_preds.index,
        y=lower,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        fillcolor=my_fillcolor,
        name=f'{int(my_conf_interval)}% Confidence Interval',
        legendgroup='confidence intervals'
    ))
    
    # Render the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
# remove datatypes object - e.g. descriptive columns not used in modeling
def remove_object_columns(df, message_columns_removed = False):
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
        col1, col2, col3 = st.columns([295,800,400])
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

def copy_df_date_index(my_df, datetime_to_date=True, date_to_index=True):
    """
    Create a deep copy of a DataFrame and optionally convert the 'date' column to a date datatype and set it as the index.
    
    Parameters:
    -----------
    my_df : pandas.DataFrame
        The input DataFrame.
    datetime_to_date : bool, optional (default=True)
        Whether to convert the 'date' column to a date datatype.
    date_to_index : bool, optional (default=True)
        Whether to set the 'date' column as the index.
    
    Returns:
    --------
    pandas.DataFrame
        A new copy of the input DataFrame, with the 'date' column optionally converted to a date datatype and/or set as the index.
    
    Examples:
    ---------
    >>> df = pd.DataFrame({'date': ['2022-01-01 12:00:00', '2022-01-02 12:00:00', '2022-01-03 12:00:00'], 'value': [1, 2, 3]})
    >>> copy_df_date_index(df)
               value
    date            
    2022-01-01      1
    2022-01-02      2
    2022-01-03      3
    """
    # create a deep copy of dataframe e.g. completely new copy of the DataFrame is created with its own memory space
    # This means that any changes made to the new copy will not affect the original DataFrame.
    my_df_copy = my_df.copy(deep=True)
    if datetime_to_date == True:
        # convert the datetime to date (excl time 00:00:00)
        my_df_copy['date'] = pd.to_datetime(my_df['date']).dt.date
    if date_to_index == True:
        # set the index instead of 0,1,2... to date
        my_df_copy = my_df_copy.set_index('date')
    return my_df_copy

def resample_missing_dates(df, freq_dict, freq, original_freq):
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
    resampled_df = df.set_index('date').resample(freq_dict[freq]).asfreq()
    
    # Fill missing values with a specified method for non-daily data
    if freq_dict[freq] != 'D':
        # pad = forward fill
        resampled_df = resampled_df.fillna(method='pad')
    
    # Find skipped dates and insert them as rows with NaN values
    missing_dates = pd.date_range(start=resampled_df.index.min(), end=resampled_df.index.max(), freq=freq_dict[freq]).difference(resampled_df.index)
    new_df = resampled_df.reindex(resampled_df.index.union(missing_dates)).sort_index()
    
    # Display a message if there are skipped dates in the data
    if len(missing_dates) > 0:
        st.write("The skipped dates are:")
        st.write(missing_dates)
    
    # Reset the index and rename the columns
    return new_df.reset_index().rename(columns={'index': 'date'})

def my_fill_method(df, fill_method, custom_fill_value=None):
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
        df.iloc[:,1] = df.iloc[:, 1].bfill()
    elif fill_method == 'Forwardfill':
        df.iloc[:,1] = df.iloc[:, 1].ffill()
    elif fill_method == 'Mean':
        # rounding occurs to nearest decimal for filling in the average value of y
        df.iloc[:,1] = df.iloc[:, 1].fillna(df.iloc[:,1].mean().round(0))
    elif fill_method == 'Median':
        df.iloc[:,1]  = df.iloc[:, 1].fillna(df.iloc[:,1].median())
    elif fill_method == 'Mode':
        # if True, only apply to numeric columns (numeric_only=False)
        # Don’t consider counts of NaN/NaT (dropna=True)
        df.iloc[:,1] = df.iloc[:, 1].fillna(df.iloc[:,1].mode(dropna=True)[0])
    elif fill_method == 'Custom':
        df.iloc[:,1]  = df.iloc[:,1].fillna(custom_fill_value)
    return df

def infer_frequency(date_df_series):
    if len(date_df_series) >= 2:
        first_date = date_df_series.iloc[0]
        second_date = date_df_series.iloc[1]
        diff = second_date - first_date
        if diff.days == 1:
            my_freq = 'D'  # Daily frequency
            my_freq_name = 'Daily'
        elif diff.days == 7:
            my_freq = 'W'
            my_freq_name = 'Weekly'
        elif diff.days == 30:
            my_freq = 'M'
            my_freq_name = 'Monthly'
        elif diff.days >= 90 and diff.days < 92:      
            my_freq = 'Q'
            my_freq_name = 'Quarterly'
        elif diff.days == 365:
            my_freq = 'Y'
            my_freq_name = 'Yearly'
        else:
            my_freq = None
            my_freq_name = None
    return my_freq, my_freq_name

def determine_df_frequency(df, column_name='date'):
    try:
        # initialize variables
        my_freq = None
        my_freq_name = '-'
        # infer frequency with pandas function infer_freq that has outputs possible below
        freq = pd.infer_freq(df[column_name])
        # DAILY
        if freq in ['D', 'B', 'BS']:
            my_freq = 'D'
            my_freq_name = 'Daily'
        # WEEKLY
        elif freq in ['W']:
            my_freq = 'W'
            my_freq_name = 'Weekly'
        # MONTHLY
        elif freq in ['M', 'MS', 'BM', 'BMS']:
            my_freq = 'M'
            my_freq_name = 'Monthly'
        # QUARTERLY
        elif freq in ['Q', 'QS', 'BQS', 'Q-JAN', 'Q-FEB', 'Q-MAR', 'Q-APR', 'Q-MAY', 'Q-JUN', 'Q-JUL', 'Q-AUG', 'Q-SEP', 'Q-OCT', 'Q-NOV', 'Q-DEC', 'QS-JAN', 'QS-FEB', 'QS-MAR', 'QS-APR', 'QS-MAY', 'QS-JUN', 'QS-JUL', 'QS-AUG', 'QS-SEP', 'QS-OCT', 'QS-NOV', 'QS-DEC', 'BQ-JAN', 'BQ-FEB', 'BQ-MAR', 'BQ-APR', 'BQ-MAY', 'BQ-JUN', 'BQ-JUL', 'BQ-AUG', 'BQ-SEP', 'BQ-OCT', 'BQ-NOV', 'BQ-DEC', 'BQS-JAN', 'BQS-FEB', 'BQS-MAR', 'BQS-APR', 'BQS-MAY', 'BQS-JUN', 'BQS-JUL', 'BQS-AUG', 'BQS-SEP', 'BQS-OCT', 'BQS-NOV', 'BQS-DEC']:
            my_freq = 'Q'
            my_freq_name = 'Quarterly'
        # YEARLY
        elif freq in ['A', 'AS', 'Y', 'BYS', 'YS', 'A-JAN', 'A-FEB', 'A-MAR', 'A-APR', 'A-MAY', 'A-JUN', 'A-JUL', 'A-AUG', 'A-SEP', 'A-OCT', 'A-NOV', 'A-DEC', 'AS-JAN', 'AS-FEB', 'AS-MAR', 'AS-APR', 'AS-MAY', 'AS-JUN', 'AS-JUL', 'AS-AUG', 'AS-SEP', 'AS-OCT', 'AS-NOV', 'AS-DEC', 'BA-JAN', 'BA-FEB', 'BA-MAR', 'BA-APR', 'BA-MAY', 'BA-JUN', 'BA-JUL', 'BA-AUG', 'BA-SEP', 'BA-OCT', 'BA-NOV', 'BA-DEC', 'BAS-JAN', 'BAS-FEB', 'BAS-MAR', 'BAS-APR', 'BAS-MAY', 'BAS-JUN', 'BAS-JUL', 'BAS-AUG', 'BAS-SEP', 'BAS-OCT', 'BAS-NOV', 'BAS-DEC']:
            my_freq = 'Y'
            my_freq_name = 'Yearly'
        else:
            # check if missing dates creating gaps, because pandas infer_freq function does not work well then use custom function to determine frequency
            my_freq, my_freq_name = infer_frequency(df[column_name])
        return my_freq, my_freq_name
    except:
        print(f'Error in function "determine_df_frequency()": could not determine frequency of uploaded data.')
        
def adjust_brightness(hex_color, brightness_factor):
    """
    Adjusts the brightness of a given hex color.

    Args:
        hex_color (str): The hex color code (e.g., '#RRGGBB').
        brightness_factor (float): The factor by which to adjust the brightness. 
                                  Values less than 1.0 decrease brightness, 
                                  while values greater than 1.0 increase brightness.

    Returns:
        str: The adjusted hex color code.

    """
    # Convert hex color to RGB
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    # Adjust brightness
    adjusted_rgb_color = tuple(max(0, min(255, int(round(c * brightness_factor)))) for c in rgb_color)
    # Convert RGB back to hex color
    adjusted_hex_color = '#' + ''.join(format(c, '02x') for c in adjusted_rgb_color)
    return adjusted_hex_color

def plot_histogram(df, y_colname, fig, row, col, my_chart_color):
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
    bin_width = 3.5 * np.std(df[y_colname]) / (len(df[y_colname]) ** (1/3))
    num_bins = math.ceil((np.max(df[y_colname]) - np.min(df[y_colname])) / bin_width)

    mean = df[y_colname].mean()
    std = df[y_colname].std()

    x_vals = np.linspace(df[y_colname].min(), df[y_colname].max(), 100) # Generate x-values
    y_vals = (np.exp(-(x_vals - mean)**2 / (2 * std**2)) / (np.sqrt(2 * np.pi) * std) * len(df[y_colname]) * bin_width) # Calculate y-values

    # Define adjusted brightness color of normal distribution trace
    adjusted_color = adjust_brightness(my_chart_color, 2)

    # Check for NaN values in the column
    if df[y_colname].isnull().any():
        vertical_spacer(2)
        st.info('**ForecastGenie message**: replaced your missing values with zero in a copy of original dataframe, in order to plot the Histogram. No worries I kept the original dataframe in one piece.')
        # Handle missing values in copy of dataframe -> do not want to change original df
        df = df.copy(deep=True)
        df[y_colname].fillna(0, inplace=True)  # Replace NaN values with zero
    else:
        pass

    # Plot histogram based on frequency type
    freq_type = get_state("HIST", "histogram_freq_type")
    
    if freq_type == "Absolute":
        histogram_trace = px.histogram(df, x=y_colname, title='Histogram', nbins=num_bins) # Define Histogram Trace
        fig.add_trace(histogram_trace.data[0], row=row, col=col)  # Histogram
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line_color=adjusted_color, showlegend=False), row=row, col=col) # Normal Dist Curve
        fig.add_vline(x=mean, line_dash="dash", line_color=adjusted_color, row=row, col=col) # Mean line

    elif freq_type == "Relative":
        hist, bins = np.histogram(df[y_colname], bins=num_bins)
        rel_freq = hist / np.sum(hist)
        fig.add_trace(go.Bar(x=bins, y=rel_freq, name='Relative Frequency', showlegend=False), row=row, col=col)

    else:
        st.error('FORECASTGENIE ERROR: Could not execute the plot of the Histogram, please contact your administrator.')
       
def plot_overview(df, y):
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
    
    my_subplot_titles = all_subplot_titles[num_graph_start-1:]
    
    # set figure with 6 rows and 1 column with needed subplot titles and set the row_ieght
    fig = make_subplots(rows=len(my_subplot_titles), cols=1,
                        subplot_titles=my_subplot_titles,
                        # the row_heights parameter is set to [0.2] * 5 + [0.5]
                        # which means that the first five rows will have a height of 0.2 each
                        # and the last row will have a height of 0.5 for the histogram with larger height.
                        row_heights=[0.2] * (len(my_subplot_titles)-2) + [0.7] + [0.5]) 
    
    # define intervals for resampling
    df_weekly = df.resample('W', on='date').mean().reset_index()
    df_monthly =  df.resample('M', on='date').mean().reset_index()
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
        plot_histogram(df = df, y_colname = y_colname, fig = fig, row=7, col=1, my_chart_color = my_chart_color)                        

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
        plot_histogram(df = df, y_colname = y_colname, fig = fig, row=6, col=1, my_chart_color = my_chart_color)               
        
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
        plot_histogram(df = df, y_colname = y_colname, fig = fig, row=5, col=1, my_chart_color = my_chart_color)      
        
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
        plot_histogram(df = df, y_colname = y_colname, fig = fig, row=4, col=1, my_chart_color = my_chart_color)     
        
        
        # set color for graphs
        fig.update_traces(line_color=my_chart_color, row=1, col=1)
        fig.update_traces(line_color=my_chart_color, row=2, col=1)
        fig.update_traces(marker_color=my_chart_color, row=3, col=1)
        fig.update_traces(marker_color=my_chart_color, row=4, col=1)
    
    if num_graph_start == 5:
        # Yearly Pattern
        fig.add_trace(px.line(df_yearly, x='date', y=y_colname, title='Yearly Pattern').data[0], row=1, col=1)
        fig.add_trace(px.box(df, y=y_colname, title='Boxplot of {}'.format(y_colname)).data[0], row=2, col=1)
        plot_histogram(df = df, y_colname = y_colname, fig = fig, row=3, col=1, my_chart_color = my_chart_color)     
        
        # set color for graphs
        fig.update_traces(line_color=my_chart_color, row=1, col=1)
        fig.update_traces(marker_color=my_chart_color, row=2, col=1)
        fig.update_traces(marker_color=my_chart_color, row=3, col=1)
    
    # define height of graph
    my_height = len(my_subplot_titles)*266
    
    # set height dynamically e.g. 6 graphs maximum but less if frequency is not daily data and x 266 (height) per graph
    fig.update_layout(height = my_height)
    
    # Display in Streamlit app
    st.plotly_chart(fig, use_container_width=True)
    
def df_differencing(df, selection, my_chart_color):
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
        fig = px.line(df, x='date', y=df.columns[1], title='Original Series',    color_discrete_sequence=[my_chart_color])
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
    
    fig.update_layout(  title = ' ',
                        title_x=0.5, 
                        title_font=dict(size=14, family="Arial"),
                        yaxis=dict(title=df.columns[1], showticklabels=False, fixedrange=True),
                        margin=dict(l=0, r=20, t=0, b=0),
                        height=200
                     )
    return fig, df_select_diff

def calc_pacf(data, nlags, method):
    return pacf(data, nlags=nlags, method=method)

# Define function to plot PACF
def plot_pacf(data, nlags, method, my_chart_color):
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
    st.markdown('<p style="text-align:center; color: #707070">Partial Autocorrelation (PACF)</p>', unsafe_allow_html=True)
    # Drop NaN values if any
    data = data.dropna(axis=0)
    data = data.to_numpy()
    # Calculate PACF
    pacf_vals = calc_pacf(data, nlags, method)
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
                                        x0=0.5, #lag0 is y with itself so confidence interval starts from lag1 and I want to show a little over lag1 visually so 0.5
                                        y0=-conf95,
                                        x1=nlags+1,
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
                                        x0=0.5, #lag0 is y with itself so confidence interval starts from lag1 and I want to show a little over lag1 visually so 0.5
                                        y0=-conf99,
                                        x1=nlags+1,
                                        y1=conf99,
                                        fillcolor='rgba(68, 114, 196, 0.3)',
                                        line=dict(width=0),
                                        opacity=0.4
                                    )
        # if absolute value of lag is larger than confidence band 99% then color 'darkred'
        if abs(pacf_vals[i]) > conf99:
            trace.line.color = my_chart_color # 'darkred'
            trace.name += ' (>|99%|)'
        # else if absolute value of lag is larger than confidence band 95% then color 'lightcoral'
        elif abs(pacf_vals[i]) > conf95:
            trace.line.color = adjusted_color # 'lightcoral' 
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

def calc_acf(data, nlags):
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

def plot_acf(data, nlags, my_chart_color):
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
    acf_vals = calc_acf(data, nlags)
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
            trace.line.color = my_chart_color #'darkred'
            trace.name += ' (>|99%|)'
        elif abs(acf_vals[i]) > conf95:
            trace.line.color = adjusted_color # 'lightcoral'
            trace.name += ' (>|95%|)'
        traces.append(trace)
    # define the background shape and color for the 95% confidence band
    conf_interval_95_background = go.layout.Shape(
                                                    type='rect',
                                                    xref='x',
                                                    yref='y',
                                                    x0=0.5, #lag0 is y with itself so confidence interval starts from lag1 and I want to show a little over lag1 visually so 0.5
                                                    y0=-conf95,
                                                    x1=nlags+1,
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
                                                    x0=0.5, #lag0 is y with itself so confidence interval starts from lag1 and I want to show a little over lag1 visually so 0.5
                                                    y0=-conf99,
                                                    x1=nlags+1,
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
 
def outlier_form():
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
                                      label = '*Select outlier detection method:*',
                                      options = ('None', 'Isolation Forest', 'Z-score', 'IQR'),
                                      key = key1_outlier
                                      )
        
        # load when user selects "Isolation Forest" and presses 'Submit' detection algorithm parameters
        if outlier_method == 'Isolation Forest':
            col1, col2, col3 = st.columns([1,12,1])
            with col2:
                contamination = st.slider(
                                          label = 'Contamination:', 
                                          min_value = 0.01, 
                                          max_value = 0.5, 
                                          step = 0.01, 
                                          key = key2_outlier,
                                          help = '''**`Contamination`** determines the *proportion of samples in the dataset that are considered to be outliers*.
                                                 It represents the expected fraction of the contamination within the data, which means it should be set to a value close to the percentage of outliers present in the data.  
                                                 A **higher** value of **contamination** will result in a **higher** number of **outliers** being detected, while a **lower** value will result in a **lower** number of **outliers** being detected.
                                                 '''
                                          )
        # load when user selects "Z-Score" and presses 'Submit' detection algorithm parameters
        elif outlier_method == 'Z-score':
            col1, col2, col3 = st.columns([1,12,1])
            with col2:
                outlier_threshold = st.slider(
                                              label = 'Threshold:', 
                                              min_value = 1.0, 
                                              max_value = 10.0, 
                                              key = key3_outlier, 
                                              step=0.1, 
                                              help = 'Using a threshold of 3 for the z-score outlier detection means that any data point +3 standard deviations or -3 standard deviations away from the mean is considered an outlier'
                                              )
                
        # load when user selects "IQR" and presses 'Submit' detection algorithm parameters
        elif outlier_method == 'IQR':
            col1, col2, col3 = st.columns([1,12,1])
            with col2:
                q1 = st.slider(
                                label = 'Q1:', 
                                min_value = 0.0, 
                                max_value = 100.0, 
                                step = 1.0, 
                                key = key4_outlier,
                                help = 'Determines the value of the first quantile. If you have a Streamlit slider set at 25%, it represents the value below which e.g. 25% of the data points fall.'
                              )
                #value=75.0
                q3 = st.slider(
                                label = 'Q3:', 
                                min_value = 0.0, 
                                max_value = 100.0, 
                                step = 1.0, 
                                key = key5_outlier,
                                help = 'Determine the value of the third quantile. If you have a Streamlit slider set at 75%, it represents the value below which e.g. 75% of the data points fall.'
                              )
                              
                # value=1.5
                iqr_multiplier = st.slider(
                                           label = 'IQR multiplier:', 
                                           min_value = 1.0, 
                                           max_value = 5.0, 
                                           step = 0.1, 
                                           key = key6_outlier,
                                           help = '''**`IQR multiplier`** determines the value used to multiply the **Interquartile range** to detect outliers.   
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
            outlier_replacement_method = st.selectbox(label = '*Select outlier replacement method:*', 
                                                      options = ('Interpolation', 'Mean', 'Median'), 
                                                      key = key7_outlier,
                                                      help = '''**`Replacement method`** determines the actual value(s) to replace detected outlier(s) with.   
                                                        You can replace your outlier(s) with one of the following replacement methods:    
                                                        - *linear interpolation algorithm* **(default option)**  
                                                        - *mean*  
                                                        - *median*
                                                        ''')
           
        col1, col2, col3 = st.columns([4,4,4])
        with col2:
            st.form_submit_button(label = 'Submit', 
                                  on_click = form_update, 
                                  args=("CLEAN_PAGE",))
            
def handle_outliers(data, method, outlier_threshold, q1, q3, outlier_replacement_method='Median', contamination=0.01, random_state=10, iqr_multiplier=1.5):
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
    outliers = None
    if method == 'Isolation Forest':
        # detect and replace outlier values using Isolation Forest
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state)
        model.fit(data)
        outliers = model.predict(data) == -1
    elif method == 'Z-score':
        # detect and replace outlier values using Z-score method
        z_scores = np.abs(stats.zscore(data))
        # The first array contains the indices of the outliers in your data variable.
        # The second array contains the actual z-scores of these outliers.
        outliers = np.where(z_scores > outlier_threshold)[0]
        # create a boolean mask indicating whether each row is an outlier or not
        is_outlier = np.zeros(data.shape[0], dtype=bool)
        is_outlier[outliers] = True
        outliers =  is_outlier
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
        outliers = is_outlier
    # select the rows that are outliers and create a new dataframe
    if outlier_replacement_method == 'Mean':
        means = data.mean()
        for col in data.columns:
            data[col][outliers] = means[col]
    elif outlier_replacement_method == 'Median':
        medians = data.median()
        for col in data.columns:
            data[col][outliers] = medians[col]
    elif outlier_replacement_method == 'Interpolation':
        # iterate over each column present in the dataframe
        for col in data.columns:
            # replace outliers with NaNs
            data[col][outliers] = np.nan
            # interpolate missing values using linear method
            data[col] = data[col].interpolate(method='linear')
            if pd.isnull(data[col].iloc[0]):
                # Note: had an edge-case with quarterly data that had an outlier as first value and replaced it with NaN with interpolation and this imputes that NaN with second datapoint's value
                st.warning(f"⚠️ Note: The first value in column '{col}' is **NaN** and will be replaced during interpolation. This introduces some bias into the data.")
            # replace the first NaN value with the first non-NaN value in the column
            first_non_nan = data[col].dropna().iloc[0]
            data[col].fillna(first_non_nan, inplace=True)
    return data, outliers 

# =============================================================================
#  __      __     _____  _____          ____  _      ______  _____ 
#  \ \    / /\   |  __ \|_   _|   /\   |  _ \| |    |  ____|/ ____|
#   \ \  / /  \  | |__) | | |    /  \  | |_) | |    | |__  | (___  
#    \ \/ / /\ \ |  _  /  | |   / /\ \ |  _ <| |    |  __|  \___ \ 
#     \  / ____ \| | \ \ _| |_ / ____ \| |_) | |____| |____ ____) |
#      \/_/    \_\_|  \_\_____/_/    \_\____/|______|______|_____/ 
#                                                                  
# =============================================================================
def initiate_global_variables():
    # ================================ COLORS =====================================
    # store color scheme for app
    create_store("COLORS", [
        ("chart_color", "#4d5466"),
        ("chart_patterns", "#4d5466"),
        ("run", 0)
    ])
    
    # ================================ LOAD PAGE =======================================
    if "df_raw" not in st.session_state:
        st.session_state["df_raw"] = pd.DataFrame()
        st.session_state.df_raw = generate_demo_data()
        df_graph = st.session_state.df_raw.copy(deep=True)
        
        # set minimum date
        df_min = st.session_state.df_raw .iloc[:,0].min().date()
        
        # set maximum date
        df_max = st.session_state.df_raw .iloc[:,0].max().date()
        
    if "df_graph" not in st.session_state:
        df_graph = st.session_state.df_raw.copy(deep=True)
        
    if "my_data_choice" not in st.session_state:
        st.session_state['my_data_choice'] = "Demo Data"
    
    # create session state for if demo data (default) or user uploaded a file
    create_store("DATA_OPTION", [("upload_new_data", False)])
     
    # Save the Data Choice of User
    key1_load, key2_load, key3_load = create_store("LOAD_PAGE", [
                                                      ("my_data_choice", "Demo Data"), #key1_load,
                                                      ("user_data_uploaded", False),   #key2_load,  
                                                      ("uploaded_file_name", None)     #key3_load
                                                      ])  
                                                     
    # ================================ EXPLORE PAGE ====================================
    key1_explore, key2_explore, key3_explore, key4_explore, key5_explore = create_store("EXPLORE_PAGE", [
        ("lags_acf", min(30, int((len(st.session_state.df_raw)-1)))),    #key1_explore
        ("lags_pacf", min(30, int((len(st.session_state.df_raw)-2)/2))), #key2_explore
        ("default_pacf_method", "yw"),                                   #key3_explore
        ("order_of_differencing_series", "Original Series"),             #key4_explore
        ("run", 0)                                                       #key5_explore
    ])
    
    # HISTOGRAM PARAMETER FOR STORING RADIO_BUTTON USER PREFERENCE
    key_hist = create_store("HIST", [("histogram_freq_type", "Absolute"),  ("run", 0)])
    
    # ================================ CLEAN PAGE ======================================
    # set the random state
    random_state = 10
    
    # for missing values custom fill value can be set by user (initiate variable with None)
    custom_fill_value = None

    if 'freq_dict' not in st.session_state:
        st.session_state['freq_dict'] = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
    
    if 'freq' not in st.session_state:
        # assume frequency is daily data for now -> expand to automated frequency detection later 
        st.session_state['freq'] = 'Daily'
        
    # define keys and store them in memory with their associated values
    key1_missing, key2_missing, key3_missing, key4_missing = create_store("CLEAN_PAGE_MISSING", 
                                                                          [
                                                                            ("missing_fill_method", "Backfill"), #key1
                                                                            ("missing_custom_fill_value", "1"),  #key2
                                                                            ("data_frequency", 'Daily'),         #key3
                                                                            ("run", 0)                           #key4
                                                                          ]
                                                                         )
    
    key1_outlier, key2_outlier, key3_outlier, key4_outlier, key5_outlier, key6_outlier, key7_outlier, key8_outlier = create_store("CLEAN_PAGE", 
                                                                                                                                  [
                                                                                                                                    ("outlier_detection_method", "None"),            #key1
                                                                                                                                    ("outlier_isolationforest_contamination", 0.01), #key2
                                                                                                                                    ("outlier_zscore_threshold", 3.0),               #key3
                                                                                                                                    ("outlier_iqr_q1", 25.0),                        #key4
                                                                                                                                    ("outlier_iqr_q3", 75.0),                        #key5
                                                                                                                                    ("outlier_iqr_multiplier", 1.5),                 #key6
                                                                                                                                    ("outlier_replacement_method", "Interpolation"), #key7
                                                                                                                                    ("run", 0)                                       #key8
                                                                                                                                  ])
    # ================================ ENGINEER PAGE ===================================
    key1_engineer, key2_engineer, key3_engineer, key4_engineer, key5_engineer, key6_engineer, key7_engineer, key8_engineer = create_store("ENGINEER_PAGE",
                                                                                                              [
                                                                                                                ("calendar_dummies_checkbox", True),          #key1_engineer
                                                                                                                ("calendar_holidays_checkbox", True),         #key2_engineer
                                                                                                                ("special_calendar_days_checkbox", True),     #key3_engineer
                                                                                                                ("dwt_features_checkbox", False),             #key4_engineer
                                                                                                                ("wavelet_family_selectbox", "db4"),          #key5_engineer
                                                                                                                ("wavelet_level_decomposition_selectbox", 3), #key6_engineer
                                                                                                                ("wavelet_window_size_slider", 7),            #key7_engineer
                                                                                                                ("run", 0)                                    #key8_engineer
                                                                                                              ]
                                                                                                            )
    
    key1_engineer_var, key2_engineer_var, key3_engineer_var, key4_engineer_var, key5_engineer_var, key6_engineer_var, key7_engineer_var, key8_engineer_var, key9_engineer_var, key10_engineer_var, \
    key11_engineer_var, key12_engineer_var, key13_engineer_var, key14_engineer_var, key15_engineer_var, key16_engineer_var, key17_engineer_var = create_store("ENGINEER_PAGE_VARS",
                                                                                                              [("year_dummies_checkbox", True), #key1_engineer_var 
                                                                                                              ("month_dummies_checkbox", True), #key2_engineer_var 
                                                                                                              ("day_dummies_checkbox", True),   #key3_engineer_var 
                                                                                                              ("jan_sales", True),              #key4_engineer_var 
                                                                                                              ("val_day_lod", True),            #key5_engineer_var
                                                                                                              ("val_day", True),                #key6_engineer_var
                                                                                                              ("mother_day_lod", True),         #key7_engineer_var
                                                                                                              ("mother_day", True),             #key8_engineer_var
                                                                                                              ("father_day_lod", True),         #key9_engineer_var
                                                                                                              ("pay_days", True),               #key10_engineer_var
                                                                                                              ("father_day", True),             #key11_engineer_var
                                                                                                              ("black_friday_lod", True),       #key12_engineer_var
                                                                                                              ("black_friday", True),           #key13_engineer_var
                                                                                                              ("cyber_monday", True),           #key14_engineer_var
                                                                                                              ("christmas_day", True),          #key15_engineer_var
                                                                                                              ("boxing_day", True),             #key16_engineer_var
                                                                                                              ("run", 0)                        #key17_engineer_var
                                                                                                            ]
                                                                                                          )
    # set initial country holidays to country_code US and country_name "United States of America"
    key1_engineer_page_country, key2_engineer_page_country, key3_engineer_page_country = create_store("ENGINEER_PAGE_COUNTRY_HOLIDAY", [
                                                                      ("country_name", "United States of America"), #key1_engineer_page_country
                                                                      ("country_code", "US"),                       #key2_engineer_page_country
                                                                      ("run", 0)])                                  #key3_engineer_page_country
    
    # THIS SESSION STATE IS USED TO DEAL WITH RESETTING DEFAULT FEATURE SELECTION OR TO KEEP USER SELECTION ON SELECT PAGE
    create_store("ENGINEER_PAGE_FEATURES_BTN", [("engineering_form_submit_btn", False)])
    
    # ================================ PREPARE PAGE ====================================
    # define my_insample_forecast_steps used for train/test split
    if 'percentage' not in st.session_state:
        st.session_state['percentage'] = 20
        
    if 'steps' not in st.session_state:
        st.session_state['steps'] = 1
        
    # save user choice in session state of in sample test-size percentage
    if 'insample_forecast_perc' not in st.session_state:
        st.session_state['insample_forecast_perc'] = 20
        
    # save user choice in session state of in sample test-size in steps (e.g. days if daily frequency data)
    if 'insample_forecast_steps' not in st.session_state:
        st.session_state['insample_forecast_steps'] = 1
    
    # set default value for normalization method to 'None'
    if 'normalization_choice' not in st.session_state:
        st.session_state['normalization_choice'] = 'None'
    
    # set the normalization and standardization to default None which has index 0 
    key1_prepare_normalization, key2_prepare_standardization, key3_prepare = create_store("PREPARE", [
        ("normalization_choice", "None"),    #key1_prepare_normalization
        ("standardization_choice", "None"),  #key2_prepare_standardization
        ("run", 0)])                         #key3_prepare
    
    # ================================ SELECT PAGE ====================================
    # TO DEAL WITH EITHER FEATURE SELECTION METHODS FEATURES OR TO REVERT TO USER SELECTION
    create_store("SELECT_PAGE_BTN_CLICKED", [("rfe_btn", False),
                                            ("mifs_btn", False), 
                                            ("pca_btn", False),
                                            ("correlation_btn", False)])
    # FEATURE SELECTION BY USER
    key1_select_page_user_selection, key2_select_page_user_selection = create_store("SELECT_PAGE_USER_SELECTION", [
                                ("feature_selection_user", []), #key1_select_page_user_selection
                                ("run", 0)])                    #key2_select_page_user_selection
                                
    # PCA
    key1_select_page_pca, key2_select_page_pca = create_store("SELECT_PAGE_PCA", [
                                ("num_features_pca", 5), #key1_select_page_pca # default of PCA features set to minimum of either 5 or the number of independent features in the dataframe
                                ("run", 0)])             #key2_select_page_pca
                                
    # RFE
    key1_select_page_rfe, key2_select_page_rfe, key3_select_page_rfe, key4_select_page_rfe, key5_select_page_rfe, key6_select_page_rfe = create_store("SELECT_PAGE_RFE", [
                                ("num_features_rfe", 5),                #key1_select_page_rfe
                                ("estimator_rfe", "Linear Regression"), #key2_select_page_rfe
                                ("timeseriessplit_value_rfe", 5),       #key3_select_page_rfe
                                ("num_steps_rfe", 1),                   #key4_select_page_rfe
                                ("duplicate_ranks_rfe", True),          #key5_select_page_rfe
                                ("run", 0)])                            #key6_select_page_rfe
                                
    # MIFS
    key1_select_page_mifs, key2_select_page_mifs = create_store("SELECT_PAGE_MIFS", [
                                ("num_features_mifs", 5), #key1_select_page_mifs
                                ("run", 0)])              #key2_select_page_mifs
                                
    # CORR
    key1_select_page_corr, key2_select_page_corr, key3_select_page_corr = create_store("SELECT_PAGE_CORR", [
                                ("corr_threshold", 0.8),                      #key1_select_page_corr 
                                ("selected_corr_model", 'Linear Regression'), #key2_select_page_corr
                                ("run", 0)])                                  #key3_select_page_corr
                               

    # ================================ TRAIN PAGE ===================================  
    # TRAIN MENU TEST
    if 'train_models_btn' not in st.session_state:
        st.session_state['train_models_btn'] = False
     
# =============================================================================
#     if 'selected_model_info' not in st.session_state:
#         st.session_state['selected_model_info'] = '-'
# =============================================================================

    # set session states for the buttons when models are trained, 
    # to expand dataframe below graph
    create_store("TRAIN", [("naive_model_btn_show", False),
                           ("naive_model_btn_hide", False),
                           ("linreg_model_btn_show", False),
                           ("sarimax_model_btn_show", False),
                           ("prophet_model_btn_show", False)])
                          
    # Training models parameters
    key1_train, key2_train, key3_train, key4_train, key5_train, key6_train, key7_train, key8_train, key9_train, key10_train, \
    key11_train, key12_train, key13_train, key14_train, key15_train, key16_train, key17_train, key18_train, key19_train, key20_train, \
    key21_train, key22_train, key23_train, key24_train, key25_train, key26_train, key27_train, key28_train, key29_train, key30_train, \
    key31_train, key32_train = create_store("TRAIN_PAGE", [
                                                ("my_conf_interval", 80),               #key1_train
                                                ("naive_checkbox",  False),             #key2_train
                                                ("linreg_checkbox",  False),            #key3_train
                                                ("sarimax_checkbox", False),            #key4_train
                                                ("prophet_checkbox", False),            #key5_train
                                                ("selected_models", []),                #key6_train
                                                ("lag", 'Week'),                        #key7_train
                                                ("custom_lag_value", 5),                #key8_train
                                                ("p", 1),                               #key9_train
                                                ("d", 1),                               #key10_train
                                                ("q", 1),                               #key11_train
                                                ("P", 1),                               #key12_train
                                                ("D", 1),                               #key13_train
                                                ("Q", 1),                               #key14_train
                                                ("s", 7),                               #key15_train
                                                ("enforce_stationarity",  True),        #key16_train
                                                ("enforce_invertibility", True),        #key17_train
                                                ("horizon_option", 30),                 #key18_train
                                                ("changepoint_prior_scale", 0.01),      #key19_train
                                                ("seasonality_mode", "multiplicative"), #key20_train
                                                ("seasonality_prior_scale", 0.01),       #key21_train
                                                ("holidays_prior_scale", 0.01),          #key22_train
                                                ("yearly_seasonality", True),           #key23_train
                                                ("weekly_seasonality", True),           #key24_train
                                                ("daily_seasonality", True),            #key25_train
                                                ("results_df", pd.DataFrame(columns=['model_name', 'predicted', 'mape', 'smape', 'rmse', 'r2', 'features', 'model_settings'])), #key26_train
                                                ("prophet_holidays", True),             #key27_train
                                                ("include_feature_selection", 'Yes'),   #key28_train
                                                ("size_rolling_window_naive_model", 7), #key29_train
                                                ("method_rolling_window_naive_model", "Mean"), #key30_train
                                                ("method_baseline_naive_model", "Mean"),#key31_train
                                                ("run", 0)                              #key32_train
                                            ])                   
    
    # ================================ EVALUATE PAGE ===================================
    # create an empty dictionary to store the results of the models
    # that I call after I train the models to display on sidebar under hedaer "Evaluate Models"
    metrics_dict = {}
        
    # Initialize results_df in global scope that has model test evaluation results 
    results_df =  pd.DataFrame(columns=['model_name', 'predicted', 'mape', 'smape', 'rmse', 'r2', 'features', 'model_settings'])
    
    if 'results_df' not in st.session_state:
        st.session_state['results_df'] = pd.DataFrame(columns=['model_name', 'predicted', 'mape', 'smape', 'rmse', 'r2', 'features', 'model_settings'])  
   
    # save user's chosen metric in persistent session state - initiate default metric (MAPE)
    key1_evaluate, key2_evaluate = create_store("EVALUATE_PAGE", [
                                            ("selected_metric", 'Mean Absolute Percentage Error'), #key1_evaluate 
                                            ("run", 0)])                                           #key2_evaluate

    # ================================ TUNE PAGE ===================================
    #
    #
    
    # ================================ FORECAST PAGE ===================================
    #
    #

    # Logging
    print('ForecastGenie Print: Loaded Global Variables')
    
    return metrics_dict, random_state, results_df, custom_fill_value, \
    key1_load, key2_load, key3_load, \
    key1_explore, key2_explore, key3_explore, key4_explore, \
    key1_missing, key2_missing, key3_missing, \
    key1_outlier, key2_outlier, key3_outlier, key4_outlier, key5_outlier, key6_outlier, key7_outlier, \
    key1_engineer, key2_engineer, key3_engineer, key4_engineer, key5_engineer, key6_engineer, key7_engineer, \
    key1_engineer_page_country, key2_engineer_page_country, key3_engineer_page_country, \
    key1_prepare_normalization, key2_prepare_standardization, \
    key1_select_page_user_selection, \
    key1_select_page_pca, \
    key1_select_page_rfe, key2_select_page_rfe, key3_select_page_rfe, key4_select_page_rfe, key5_select_page_rfe, key6_select_page_rfe, \
    key1_select_page_mifs, \
    key1_select_page_corr, key2_select_page_corr, \
    key1_train, key2_train, key3_train, key4_train, key5_train, key6_train, key7_train, key8_train, key9_train, key10_train, \
    key11_train, key12_train, key13_train, key14_train, key15_train, key16_train, key17_train, key18_train, key19_train, key20_train, \
    key21_train, key22_train, key23_train, key24_train, key25_train, key26_train, key27_train, key28_train, key29_train, key30_train, key31_train, key32_train, \
    key1_evaluate, key2_evaluate

    
#///////////////////////////////////////////////////////////////////
# SHOW IN STREAMLIT DICTIONARY OF VARIABLES IN SESSION STATE DEBUG
#///////////////////////////////////////////////////////////////////
# show in streamlit the session state variables that are stored cache for the user session
st.write(st.session_state)
#///////////////////////////////////////////////////////////////////
        
# =============================================================================
#   _____ _   _ _____ _______ _____       _______ ______ 
#  |_   _| \ | |_   _|__   __|_   _|   /\|__   __|  ____|
#    | | |  \| | | |    | |    | |    /  \  | |  | |__   
#    | | | . ` | | |    | |    | |   / /\ \ | |  |  __|  
#   _| |_| |\  |_| |_   | |   _| |_ / ____ \| |  | |____ 
#  |_____|_| \_|_____|  |_|  |_____/_/    \_\_|  |______|
#                                                        
# =============================================================================
# INITIATE GLOBAL VARIABLES
metrics_dict, random_state, results_df, custom_fill_value, \
key1_load, key2_load, key3_load, \
key1_explore, key2_explore, key3_explore, key4_explore, \
key1_missing, key2_missing, key3_missing, \
key1_outlier, key2_outlier, key3_outlier, key4_outlier, key5_outlier, key6_outlier, key7_outlier, \
key1_engineer, key2_engineer, key3_engineer, key4_engineer, key5_engineer, key6_engineer, key7_engineer, \
key1_engineer_page_country, key2_engineer_page_country, key3_engineer_page_country, \
key1_prepare_normalization, key2_prepare_standardization, \
key1_select_page_user_selection, \
key1_select_page_pca, \
key1_select_page_rfe, key2_select_page_rfe, key3_select_page_rfe, key4_select_page_rfe, key5_select_page_rfe, key6_select_page_rfe, \
key1_select_page_mifs, \
key1_select_page_corr, key2_select_page_corr, \
key1_train, key2_train, key3_train, key4_train, key5_train, key6_train, key7_train, key8_train, key9_train, key10_train, \
key11_train, key12_train, key13_train, key14_train, key15_train, key16_train, key17_train, key18_train, key19_train, key20_train, \
key21_train, key22_train, key23_train, key24_train, key25_train, key26_train, key27_train, key28_train, key29_train, key30_train, key31_train, key32_train, \
key1_evaluate, key2_evaluate = initiate_global_variables()

# =============================================================================
#   _____ _____ ____  _   _  _____ 
#  |_   _/ ____/ __ \| \ | |/ ____|
#    | || |   | |  | |  \| | (___  
#    | || |   | |  | | . ` |\___ \ 
#   _| || |___| |__| | |\  |____) |
#  |_____\_____\____/|_| \_|_____/ 
#                                  
# =============================================================================
def load_icons():
    # svg image of open book used for text on docs page
    book_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-book" viewBox="0 0 16 16">
                   <path d="M1 2.828c.885-.37 2.154-.769 3.388-.893 1.33-.134 2.458.063 3.112.752v9.746c-.935-.53-2.12-.603-3.213-.493-1.18.12-2.37.461-3.287.811V2.828zm7.5-.141c.654-.689 1.782-.886 3.112-.752 1.234.124 2.503.523 3.388.893v9.923c-.918-.35-2.107-.692-3.287-.81-1.094-.111-2.278-.039-3.213.492V2.687zM8 1.783C7.015.936 5.587.81 4.287.94c-1.514.153-3.042.672-3.994 1.105A.5.5 0 0 0 0 2.5v11a.5.5 0 0 0 .707.455c.882-.4 2.303-.881 3.68-1.02 1.409-.142 2.59.087 3.223.877a.5.5 0 0 0 .78 0c.633-.79 1.814-1.019 3.222-.877 1.378.139 2.8.62 3.681 1.02A.5.5 0 0 0 16 13.5v-11a.5.5 0 0 0-.293-.455c-.952-.433-2.48-.952-3.994-1.105C10.413.809 8.985.936 8 1.783z"/>
                 </svg>
                """
    # svg image of heart used for text on about page
    balloon_heart_svg = """
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-balloon-heart" viewBox="0 0 16 16">
                          <path fill-rule="evenodd" d="m8 2.42-.717-.737c-1.13-1.161-3.243-.777-4.01.72-.35.685-.451 1.707.236 3.062C4.16 6.753 5.52 8.32 8 10.042c2.479-1.723 3.839-3.29 4.491-4.577.687-1.355.587-2.377.236-3.061-.767-1.498-2.88-1.882-4.01-.721L8 2.42Zm-.49 8.5c-10.78-7.44-3-13.155.359-10.063.045.041.089.084.132.129.043-.045.087-.088.132-.129 3.36-3.092 11.137 2.624.357 10.063l.235.468a.25.25 0 1 1-.448.224l-.008-.017c.008.11.02.202.037.29.054.27.161.488.419 1.003.288.578.235 1.15.076 1.629-.157.469-.422.867-.588 1.115l-.004.007a.25.25 0 1 1-.416-.278c.168-.252.4-.6.533-1.003.133-.396.163-.824-.049-1.246l-.013-.028c-.24-.48-.38-.758-.448-1.102a3.177 3.177 0 0 1-.052-.45l-.04.08a.25.25 0 1 1-.447-.224l.235-.468ZM6.013 2.06c-.649-.18-1.483.083-1.85.798-.131.258-.245.689-.08 1.335.063.244.414.198.487-.043.21-.697.627-1.447 1.359-1.692.217-.073.304-.337.084-.398Z"/>
                        </svg>
                        """
                        
    paint_bucket_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="#3b3b3b" class="bi bi-paint-bucket" viewBox="0 0 16 16">
      <path d="M6.192 2.78c-.458-.677-.927-1.248-1.35-1.643a2.972 2.972 0 0 0-.71-.515c-.217-.104-.56-.205-.882-.02-.367.213-.427.63-.43.896-.003.304.064.664.173 1.044.196.687.556 1.528 1.035 2.402L.752 8.22c-.277.277-.269.656-.218.918.055.283.187.593.36.903.348.627.92 1.361 1.626 2.068.707.707 1.441 1.278 2.068 1.626.31.173.62.305.903.36.262.05.64.059.918-.218l5.615-5.615c.118.257.092.512.05.939-.03.292-.068.665-.073 1.176v.123h.003a1 1 0 0 0 1.993 0H14v-.057a1.01 1.01 0 0 0-.004-.117c-.055-1.25-.7-2.738-1.86-3.494a4.322 4.322 0 0 0-.211-.434c-.349-.626-.92-1.36-1.627-2.067-.707-.707-1.441-1.279-2.068-1.627-.31-.172-.62-.304-.903-.36-.262-.05-.64-.058-.918.219l-.217.216zM4.16 1.867c.381.356.844.922 1.311 1.632l-.704.705c-.382-.727-.66-1.402-.813-1.938a3.283 3.283 0 0 1-.131-.673c.091.061.204.15.337.274zm.394 3.965c.54.852 1.107 1.567 1.607 2.033a.5.5 0 1 0 .682-.732c-.453-.422-1.017-1.136-1.564-2.027l1.088-1.088c.054.12.115.243.183.365.349.627.92 1.361 1.627 2.068.706.707 1.44 1.278 2.068 1.626.122.068.244.13.365.183l-4.861 4.862a.571.571 0 0 1-.068-.01c-.137-.027-.342-.104-.608-.252-.524-.292-1.186-.8-1.846-1.46-.66-.66-1.168-1.32-1.46-1.846-.147-.265-.225-.47-.251-.607a.573.573 0 0 1-.01-.068l3.048-3.047zm2.87-1.935a2.44 2.44 0 0 1-.241-.561c.135.033.324.11.562.241.524.292 1.186.8 1.846 1.46.45.45.83.901 1.118 1.31a3.497 3.497 0 0 0-1.066.091 11.27 11.27 0 0 1-.76-.694c-.66-.66-1.167-1.322-1.458-1.847z"/>
    </svg>
    """
    
    load_icon = """
     <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="#3b3b3b" class="bi bi-cloud-arrow-up" viewBox="0 0 16 16">
       <path fill-rule="evenodd" d="M7.646 5.146a.5.5 0 0 1 .708 0l2 2a.5.5 0 0 1-.708.708L8.5 6.707V10.5a.5.5 0 0 1-1 0V6.707L6.354 7.854a.5.5 0 1 1-.708-.708l2-2z"/>
       <path d="M4.406 3.342A5.53 5.53 0 0 1 8 2c2.69 0 4.923 2 5.166 4.579C14.758 6.804 16 8.137 16 9.773 16 11.569 14.502 13 12.687 13H3.781C1.708 13 0 11.366 0 9.318c0-1.763 1.266-3.223 2.942-3.593.143-.863.698-1.723 1.464-2.383zm.653.757c-.757.653-1.153 1.44-1.153 2.056v.448l-.445.049C2.064 6.805 1 7.952 1 9.318 1 10.785 2.23 12 3.781 12h8.906C13.98 12 15 10.988 15 9.773c0-1.216-1.02-2.228-2.313-2.228h-.5v-.5C12.188 4.825 10.328 3 8 3a4.53 4.53 0 0 0-2.941 1.1z"/>
     </svg>
    """
    
    explore_icon = """
    <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="#3b3b3b" class="bi bi-search" viewBox="0 0 16 16">
      <path d="M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0z"/>
    </svg>
    """
    
    clean_icon = """
    <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="#3b3b3b" class="bi bi-bezier2" viewBox="0 0 16 16">
      <path fill-rule="evenodd" d="M1 2.5A1.5 1.5 0 0 1 2.5 1h1A1.5 1.5 0 0 1 5 2.5h4.134a1 1 0 1 1 0 1h-2.01c.18.18.34.381.484.605.638.992.892 2.354.892 3.895 0 1.993.257 3.092.713 3.7.356.476.895.721 1.787.784A1.5 1.5 0 0 1 12.5 11h1a1.5 1.5 0 0 1 1.5 1.5v1a1.5 1.5 0 0 1-1.5 1.5h-1a1.5 1.5 0 0 1-1.5-1.5H6.866a1 1 0 1 1 0-1h1.711a2.839 2.839 0 0 1-.165-.2C7.743 11.407 7.5 10.007 7.5 8c0-1.46-.246-2.597-.733-3.355-.39-.605-.952-1-1.767-1.112A1.5 1.5 0 0 1 3.5 5h-1A1.5 1.5 0 0 1 1 3.5v-1zM2.5 2a.5.5 0 0 0-.5.5v1a.5.5 0 0 0 .5.5h1a.5.5 0 0 0 .5-.5v-1a.5.5 0 0 0-.5-.5h-1zm10 10a.5.5 0 0 0-.5.5v1a.5.5 0 0 0 .5.5h1a.5.5 0 0 0 .5-.5v-1a.5.5 0 0 0-.5-.5h-1z"/>
    </svg>
    """
    
    engineer_icon = """
    <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="#3b3b3b" class="bi bi-gear" viewBox="0 0 16 16">
      <path d="M8 4.754a3.246 3.246 0 1 0 0 6.492 3.246 3.246 0 0 0 0-6.492zM5.754 8a2.246 2.246 0 1 1 4.492 0 2.246 2.246 0 0 1-4.492 0z"/>
      <path d="M9.796 1.343c-.527-1.79-3.065-1.79-3.592 0l-.094.319a.873.873 0 0 1-1.255.52l-.292-.16c-1.64-.892-3.433.902-2.54 2.541l.159.292a.873.873 0 0 1-.52 1.255l-.319.094c-1.79.527-1.79 3.065 0 3.592l.319.094a.873.873 0 0 1 .52 1.255l-.16.292c-.892 1.64.901 3.434 2.541 2.54l.292-.159a.873.873 0 0 1 1.255.52l.094.319c.527 1.79 3.065 1.79 3.592 0l.094-.319a.873.873 0 0 1 1.255-.52l.292.16c1.64.893 3.434-.902 2.54-2.541l-.159-.292a.873.873 0 0 1 .52-1.255l.319-.094c1.79-.527 1.79-3.065 0-3.592l-.319-.094a.873.873 0 0 1-.52-1.255l.16-.292c.893-1.64-.902-3.433-2.541-2.54l-.292.159a.873.873 0 0 1-1.255-.52l-.094-.319zm-2.633.283c.246-.835 1.428-.835 1.674 0l.094.319a1.873 1.873 0 0 0 2.693 1.115l.291-.16c.764-.415 1.6.42 1.184 1.185l-.159.292a1.873 1.873 0 0 0 1.116 2.692l.318.094c.835.246.835 1.428 0 1.674l-.319.094a1.873 1.873 0 0 0-1.115 2.693l.16.291c.415.764-.42 1.6-1.185 1.184l-.291-.159a1.873 1.873 0 0 0-2.693 1.116l-.094.318c-.246.835-1.428.835-1.674 0l-.094-.319a1.873 1.873 0 0 0-2.692-1.115l-.292.16c-.764.415-1.6-.42-1.184-1.185l.159-.291A1.873 1.873 0 0 0 1.945 8.93l-.319-.094c-.835-.246-.835-1.428 0-1.674l.319-.094A1.873 1.873 0 0 0 3.06 4.377l-.16-.292c-.415-.764.42-1.6 1.185-1.184l.292.159a1.873 1.873 0 0 0 2.692-1.115l.094-.319z"/>
    </svg>
    """
    
    prepare_icon = """
    <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="#3b3b3b" class="bi bi-shuffle" viewBox="0 0 16 16">
      <path fill-rule="evenodd" d="M0 3.5A.5.5 0 0 1 .5 3H1c2.202 0 3.827 1.24 4.874 2.418.49.552.865 1.102 1.126 1.532.26-.43.636-.98 1.126-1.532C9.173 4.24 10.798 3 13 3v1c-1.798 0-3.173 1.01-4.126 2.082A9.624 9.624 0 0 0 7.556 8a9.624 9.624 0 0 0 1.317 1.918C9.828 10.99 11.204 12 13 12v1c-2.202 0-3.827-1.24-4.874-2.418A10.595 10.595 0 0 1 7 9.05c-.26.43-.636.98-1.126 1.532C4.827 11.76 3.202 13 1 13H.5a.5.5 0 0 1 0-1H1c1.798 0 3.173-1.01 4.126-2.082A9.624 9.624 0 0 0 6.444 8a9.624 9.624 0 0 0-1.317-1.918C4.172 5.01 2.796 4 1 4H.5a.5.5 0 0 1-.5-.5z"/>
      <path d="M13 5.466V1.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384l-2.36 1.966a.25.25 0 0 1-.41-.192zm0 9v-3.932a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384l-2.36 1.966a.25.25 0 0 1-.41-.192z"/>
    </svg>
    """
    
    select_icon = """
    <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="#3b3b3b" class="bi bi-sort-down" viewBox="0 0 16 16">
      <path d="M3.5 2.5a.5.5 0 0 0-1 0v8.793l-1.146-1.147a.5.5 0 0 0-.708.708l2 1.999.007.007a.497.497 0 0 0 .7-.006l2-2a.5.5 0 0 0-.707-.708L3.5 11.293V2.5zm3.5 1a.5.5 0 0 1 .5-.5h7a.5.5 0 0 1 0 1h-7a.5.5 0 0 1-.5-.5zM7.5 6a.5.5 0 0 0 0 1h5a.5.5 0 0 0 0-1h-5zm0 3a.5.5 0 0 0 0 1h3a.5.5 0 0 0 0-1h-3zm0 3a.5.5 0 0 0 0 1h1a.5.5 0 0 0 0-1h-1z"/>
    </svg>
    """
    
    train_icon = """
    <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="#3b3b3b" class="bi bi-cpu" viewBox="0 0 16 16">
      <path d="M5 0a.5.5 0 0 1 .5.5V2h1V.5a.5.5 0 0 1 1 0V2h1V.5a.5.5 0 0 1 1 0V2h1V.5a.5.5 0 0 1 1 0V2A2.5 2.5 0 0 1 14 4.5h1.5a.5.5 0 0 1 0 1H14v1h1.5a.5.5 0 0 1 0 1H14v1h1.5a.5.5 0 0 1 0 1H14v1h1.5a.5.5 0 0 1 0 1H14a2.5 2.5 0 0 1-2.5 2.5v1.5a.5.5 0 0 1-1 0V14h-1v1.5a.5.5 0 0 1-1 0V14h-1v1.5a.5.5 0 0 1-1 0V14h-1v1.5a.5.5 0 0 1-1 0V14A2.5 2.5 0 0 1 2 11.5H.5a.5.5 0 0 1 0-1H2v-1H.5a.5.5 0 0 1 0-1H2v-1H.5a.5.5 0 0 1 0-1H2v-1H.5a.5.5 0 0 1 0-1H2A2.5 2.5 0 0 1 4.5 2V.5A.5.5 0 0 1 5 0zm-.5 3A1.5 1.5 0 0 0 3 4.5v7A1.5 1.5 0 0 0 4.5 13h7a1.5 1.5 0 0 0 1.5-1.5v-7A1.5 1.5 0 0 0 11.5 3h-7zM5 6.5A1.5 1.5 0 0 1 6.5 5h3A1.5 1.5 0 0 1 11 6.5v3A1.5 1.5 0 0 1 9.5 11h-3A1.5 1.5 0 0 1 5 9.5v-3zM6.5 6a.5.5 0 0 0-.5.5v3a.5.5 0 0 0 .5.5h3a.5.5 0 0 0 .5-.5v-3a.5.5 0 0 0-.5-.5h-3z"/>
    </svg>
    """
    
    evaluate_icon = """
    <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="#3b3b3b" class="bi bi-clipboard-check" viewBox="0 0 16 16">
      <path fill-rule="evenodd" d="M10.854 7.146a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708 0l-1.5-1.5a.5.5 0 1 1 .708-.708L7.5 9.793l2.646-2.647a.5.5 0 0 1 .708 0z"/>
      <path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/>
      <path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3z"/>
    </svg>
    """
    
    tune_icon = """
    <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="#3b3b3b" class="bi bi-sliders" viewBox="0 0 16 16">
      <path fill-rule="evenodd" d="M11.5 2a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zM9.05 3a2.5 2.5 0 0 1 4.9 0H16v1h-2.05a2.5 2.5 0 0 1-4.9 0H0V3h9.05zM4.5 7a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zM2.05 8a2.5 2.5 0 0 1 4.9 0H16v1H6.95a2.5 2.5 0 0 1-4.9 0H0V8h2.05zm9.45 4a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zm-2.45 1a2.5 2.5 0 0 1 4.9 0H16v1h-2.05a2.5 2.5 0 0 1-4.9 0H0v-1h9.05z"/>
    </svg>
    """
    
    forecast_icon = """
    <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="3b3b3b" class="bi bi-graph-up-arrow" viewBox="0 0 16 16">
      <path fill-rule="evenodd" d="M0 0h1v15h15v1H0V0Zm10 3.5a.5.5 0 0 1 .5-.5h4a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-1 0V4.9l-3.613 4.417a.5.5 0 0 1-.74.037L7.06 6.767l-3.656 5.027a.5.5 0 0 1-.808-.588l4-5.5a.5.5 0 0 1 .758-.06l2.609 2.61L13.445 4H10.5a.5.5 0 0 1-.5-.5Z"/>
    </svg>
    """
    
    arrow_down_icon = """
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-arrow-down-circle" viewBox="0 0 16 16">
      <path fill-rule="evenodd" d="M1 8a7 7 0 1 0 14 0A7 7 0 0 0 1 8zm15 0A8 8 0 1 1 0 8a8 8 0 0 1 16 0zM8.5 4.5a.5.5 0 0 0-1 0v5.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V4.5z"/>
    </svg>
    """
    # show arrow which is a bootstrap icon from source: https://icons.getbootstrap.com/icons/arrow-right/
    arrow_right_icon = '''<svg xmlns="http://www.w3.org/2000/svg" width="25" height="20" fill="currentColor" class="bi bi-arrow-right" viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" d="M1 8a.5.5 0 0 1 .5-.5h11.793l-3.147-3.146a.5.5 0 0 1 .708-.708l4 4a.5.5 0 0 1 0 .708l-4 4a.5.5 0 0 1-.708-.708L13.293 8.5H1.5A.5.5 0 0 1 1 8z"/>'
                '</svg>'''
                        
    print('ForecastGenie Print: Loaded SVG ICONS')
    
    return book_icon, balloon_heart_svg, paint_bucket_icon, load_icon, explore_icon, clean_icon, engineer_icon, \
    prepare_icon, select_icon, train_icon, evaluate_icon, tune_icon, forecast_icon, arrow_down_icon, arrow_right_icon

# load svg icons which are used in moreover menu's and some headers
book_icon, balloon_heart_svg, paint_bucket_icon, \
load_icon, explore_icon, clean_icon, engineer_icon, prepare_icon, select_icon, \
train_icon, evaluate_icon, tune_icon, forecast_icon, arrow_down_icon, arrow_right_icon = load_icons()
    
# =============================================================================

# =============================================================================
#   _______ _____ _______ _      ______ 
#  |__   __|_   _|__   __| |    |  ____|
#     | |    | |    | |  | |    | |__   
#     | |    | |    | |  | |    |  __|  
#     | |   _| |_   | |  | |____| |____ 
#     |_|  |_____|  |_|  |______|______|
#                                       
# =============================================================================
with st.sidebar:
    st.image('./images/forecastgenie_logo.png')
    

# =============================================================================
#    _____ _____ _____  ______ ____          _____    __  __ ______ _   _ _    _ 
#   / ____|_   _|  __ \|  ____|  _ \   /\   |  __ \  |  \/  |  ____| \ | | |  | |
#  | (___   | | | |  | | |__  | |_) | /  \  | |__) | | \  / | |__  |  \| | |  | |
#   \___ \  | | | |  | |  __| |  _ < / /\ \ |  _  /  | |\/| |  __| | . ` | |  | |
#   ____) |_| |_| |__| | |____| |_) / ____ \| | \ \  | |  | | |____| |\  | |__| |
#  |_____/|_____|_____/|______|____/_/    \_\_|  \_\ |_|  |_|______|_| \_|\____/ 
#                                                                                
# =============================================================================  
# CREATE SIDEBAR MENU FOR HOME / ABOUT / FAQ PAGES FOR USER
with st.sidebar:
    sidebar_menu_item = option_menu(None, ["Home", "About", "FAQ", "Doc"], 
        icons = ["house", "file-person", "info-circle", "file-text"], 
        menu_icon = "cast", 
        default_index = 0, 
        orientation = "horizontal",
        styles = {
            "container": {
                "padding": "0px",
                "background": "rgba(255, 255, 255, 0.2)", #"linear-gradient(45deg, #0C8F74, #99D8FF)",
                "border-radius": "0px",
                #"box-shadow": "0px 0px 2px rgba(0, 0, 0, 0.2)"
            },
            "icon": {
                "color": "#FFFFFF",
                "font-size": "15px",
                "margin-right": "10px"
            }, 
            "nav-link": {   
                            "font-family": "Ysabeau SC",
                            "font-size": "14px",
                            "font-weight": "normal",
                            "letter-spacing": "0.5px",
                            "color": "#333333",
                            "text-align": "center",
                            "margin": "0px",
                            "padding": "10px",
                            "background-color": "transparent",
                            "opacity": "1",
                            "transition": "background-color 0.5s ease-out",
                        },
            "nav-link:hover": {
                "background-color": "rgba(255, 255, 255, 0.1)",
                "transition": "background-color 0.5s ease-out",
            },
            "nav-link-selected": {
                            "background": "linear-gradient(45deg, #000000, ##f5f5f5)",
                            "background": "-webkit-linear-gradient(45deg, #000000, ##f5f5f5)", 
                            "background": "-moz-linear-gradient(45deg, #000000, ##f5f5f5)", 
                "color": "grey",
                "opacity": "0.5",
                #"box-shadow": "0px 0px 2px rgba(0, 0, 0, 0.3)",
                #"border-radius": "15px",
                #"border": "0px solid #45B8AC"
            }
        })
    
# =============================================================================
#   __  __          _____ _   _   __  __ ______ _   _ _    _ 
#  |  \/  |   /\   |_   _| \ | | |  \/  |  ____| \ | | |  | |
#  | \  / |  /  \    | | |  \| | | \  / | |__  |  \| | |  | |
#  | |\/| | / /\ \   | | | . ` | | |\/| |  __| | . ` | |  | |
#  | |  | |/ ____ \ _| |_| |\  | | |  | | |____| |\  | |__| |
#  |_|  |_/_/    \_\_____|_| \_| |_|  |_|______|_| \_|\____/ 
#                                                            
# =============================================================================
# source for package: https://github.com/victoryhb/streamlit-option-menu
# CREATE MAIN MENU BAR FOR APP WITH BUTTONS OF DATA PIPELINE
# Horizontal menu with default on first tab e.g. default_index = 0
menu_item = option_menu(menu_title = None, 
                        options = ["Load", "Explore", "Clean", "Engineer", "Prepare", "Select", "Train", "Evaluate", "Tune", "Forecast", "Review"], 
                        icons = ['cloud-arrow-up', 'search', 'bi-bezier2', 'gear', 'shuffle', 'bi-sort-down', "cpu", 'clipboard-check', 'sliders', 'graph-up-arrow', 'bullseye'], 
                        menu_icon = "cast", 
                        default_index = 0, 
                        orientation = "horizontal", 
                        styles = {
                                    "container": {
                                        "padding": "0.5px",
                                        "background-color": "transparent",
                                        "border-radius": "10px",
                                        "border": "1px solid black",
                                        "margin": "0px 0px 0px 0px",
                                    },
                                    "icon": {
                                        "color": "#333333",
                                        "font-size": "20px",
                                    },
                                    "nav-link": {
                                        "font-size": "12px",
                                        "font-family": 'Ysabeau SC',
                                        "color": "F5F5F5",
                                        "text-align": "center",
                                        "margin": "0px",
                                        "padding": "8px",
                                        "background-color": "transparent",
                                        "opacity": "1",
                                        "transition": "background-color 0.3s ease-in-out",
                                    },
                                    "nav-link:hover": {
                                        "background-color": "#4715ef",
                                    },
                                    "nav-link-selected": {
                                        "background-color": "transparent",
                                        "color": "black",
                                        "opacity": "0.1",
                                        "box-shadow": "0px 4px 6px rgba(0, 0, 0, 0.0)",
                                        "border-radius": "0px",
                                    },
                                }
                        )

# =============================================================================
#            ____   ____  _    _ _______   _____        _____ ______ 
#      /\   |  _ \ / __ \| |  | |__   __| |  __ \ /\   / ____|  ____|
#     /  \  | |_) | |  | | |  | |  | |    | |__) /  \ | |  __| |__   
#    / /\ \ |  _ <| |  | | |  | |  | |    |  ___/ /\ \| | |_ |  __|  
#   / ____ \| |_) | |__| | |__| |  | |    | |  / ____ \ |__| | |____ 
#  /_/    \_\____/ \____/ \____/   |_|    |_| /_/    \_\_____|______|
#                                                                    
# =============================================================================
# ABOUT PAGE
if sidebar_menu_item == 'About':
    try:
        with st.sidebar:
            vertical_spacer(2)
            
            my_text_paragraph(f'{book_icon}')
            
            col1, col2, col3 = st.columns([2,14,2])
            with col2:
                selected_about_page = st.selectbox(
                                                     label = "*Select Page*:", 
                                                     options = ['-',
                                                                'What does it do?', 
                                                                'What do I need?', 
                                                                'Who is this for?', 
                                                                'Origin Story',
                                                                'Show your support',
                                                                'Version'], 
                                                    label_visibility='collapsed', 
                                                    index = 0,
                                                 )
            # show social media links    
            social_media_links(margin_before=30)
                
        # =============================================================================
        # Front Cover        
        # =============================================================================   
        if selected_about_page == '-':   
            with st.expander('', expanded=True):
                st.image('./images/about_page.png')
        
        # =============================================================================
        # What does it do?            
        # =============================================================================   
        if selected_about_page == 'What does it do?':   
            with st.expander('', expanded=True):
                col1, col2, col3 = st.columns([2,8,2])
                with col2:
                    my_text_header('What does it do?')
                    my_text_paragraph('🕵️‍♂️ <b> Analyze data:', add_border=True, border_color = "#F08A5D")
                    my_text_paragraph('Inspect seasonal patterns and distribution of the data', add_border=False)
                    my_text_paragraph('🧹 <b> Cleaning data: </b>', add_border=True, border_color = "#F08A5D")
                    my_text_paragraph('Automatic detection and replacing missing data points and remove outliers')
                    my_text_paragraph('🧰 <b> Feature Engineering: </b>', add_border=True,border_color = "#F08A5D")
                    my_text_paragraph(' Add holidays, calendar day/week/month/year and optional wavelet features')
                    my_text_paragraph('⚖️ <b> Normalization and Standardization </b>', add_border=True, border_color = "#F08A5D")
                    my_text_paragraph('Select from industry standard techniques')
                    my_text_paragraph('🍏 <b> Feature Selection: </b>', add_border=True, border_color = "#F08A5D")
                    my_text_paragraph('</b> Only keep relevant features based on feature selection techniques')
                    my_text_paragraph('🍻 <b> Correlation Analysis:</b> ', add_border=True, border_color = "#F08A5D")
                    my_text_paragraph('Automatically remove highly correlated features')
                    my_text_paragraph('🔢 <b> Train Models:</b>', add_border=True, border_color = "#F08A5D")
                    my_text_paragraph('Including Naive, Linear, SARIMAX and Prophet Models')
                    my_text_paragraph('🎯 <b> Evaluate Model Performance:', add_border=True, border_color = "#F08A5D")
                    my_text_paragraph('Benchmark models performance with evaluation metrics')
                    my_text_paragraph('🔮  <b> Forecast:', add_border=True, border_color = "#F08A5D")
                    my_text_paragraph('Forecast your variable of interest with ease by selecting your desired end-date from the calendar')
        
        # =============================================================================
        # What do I need?         
        # =============================================================================        
        if selected_about_page == 'What do I need?':   
            with st.expander('', expanded=True):  
                col1, col2, col3 = st.columns([2,8,2])
                with col2:
                    my_text_header('What do I need?')
                    my_text_paragraph('<b> File: </b>', add_border=True, border_color = "#0262ac")    
                    my_text_paragraph('''From the <i> Load </i> menu you can select the radio-button 'Upload Data' and drag-and-drop or left-click to browse the file from your local computer. 
                                      ''', my_text_align='justify')
   
                    my_text_paragraph('<b> File Requirements: </b>', add_border=True, border_color = "#0262ac")     
                    my_text_paragraph(''' - the 1st column should have a header with the following lowercase name: <b> 'date' </b> with dates in ascending order in format: <b>mm/dd/yyyy</b>.  For example 12/31/2024.
                                      <br>
                                          - the 2nd column should have a header as well with a name of your choosing. This column should contain the historical values of your variable of interest, which you are trying to forecast, based on historical data.
                                      ''', my_text_align='left')   
                    my_text_paragraph('<b> Supported File Formats: </b>', add_border=True, border_color = "#0262ac")                   
                    my_text_paragraph('Common file-formats are supported, namely: .csv, .xls .xlsx, xlsm and xlsb.', my_text_align='justify')                       
                    vertical_spacer(25)
        # =============================================================================
        # Who is this for?       
        # =============================================================================   
        if selected_about_page == 'Who is this for?':   
            with st.expander('', expanded=True):  
                col1, col2, col3 = st.columns([2,6,2])
                with col2:
                    my_text_header('Who is this for?')
                    my_text_paragraph("<b>Target Groups</b>", add_border=True, border_color="#bc62f6")
                    my_text_paragraph('ForecastGenie is designed to cater to a wide range of users, including <i>Business Analysts, Data Scientists, Statisticians, and Enthusiasts.</i> Whether you are a seasoned professional or an aspiring data enthusiast, this app aims to provide you with powerful data analysis and forecasting capabilities! If you are curious about the possibilities, some use cases are listed below.', my_text_align='justify')
                    
                    my_text_paragraph("<b>Business Analysts:</b>", add_border=True, border_color="#bc62f6")
                    my_text_paragraph('''
                                         - <b>Sales Forecasting:</b> Predict future sales volumes and trends to optimize inventory management, plan marketing campaigns, and make informed business decisions.<br>
                                         - <b>Demand Forecasting:</b> Estimate future demand for products or services to streamline supply chain operations, improve production planning, and optimize resource allocation.<br>
                                         - <b>Financial Forecasting:</b> Forecast revenues, expenses, and cash flows to facilitate budgeting, financial planning, and investment decisions.
                                     ''',  my_text_align='left')
                    my_text_paragraph("<b>Data Scientists:</b>", add_border=True, border_color="#bc62f6")
                    my_text_paragraph('''
                                         - <b>Time Series Analysis:</b> Analyze historical data patterns and trends to identify seasonality, trend components, and anomalies for various applications such as finance, energy, stock market, or weather forecasting.<br>
                                         - <b>Predictive Maintenance:</b> Forecast equipment failures or maintenance needs based on sensor data, enabling proactive maintenance scheduling and minimizing downtime.<br>
                                         - <b>Customer Churn Prediction:</b> Predict customer churn probabilities, allowing companies to take preventive measures, retain customers, and enhance customer satisfaction.''',
                                         my_text_align='left')
                    my_text_paragraph("<b>Statisticians:</b>", add_border=True, border_color="#bc62f6")
                    my_text_paragraph('''                                     
                                         - <b>Economic Forecasting:</b> Forecast macroeconomic indicators such as GDP, inflation rates, or employment levels to support economic policy-making, investment strategies, and financial planning.<br>
                                         - <b>Population Forecasting:</b> Predict future population sizes and demographic changes for urban planning, resource allocation, and infrastructure development.<br>
                                         - <b>Epidemiological Forecasting:</b> Forecast disease spread and outbreak patterns to aid in public health planning, resource allocation, and implementation of preventive measures.
                                      ''', my_text_align='left')
                    
                    my_text_paragraph("<b>Enthusiasts:</b>", add_border=True, border_color="#bc62f6")
                    my_text_paragraph('''- <b>Personal Budget Forecasting:</b> Forecast your personal income and expenses to better manage your finances, plan savings goals, and make informed spending decisions.<br>
                                         - <b>Weather Forecasting:</b> Generate short-term or long-term weather forecasts for personal planning, outdoor activities, or agricultural purposes.<br>
                                         - <b>Energy Consumption Forecasting:</b> Forecast your household or individual energy consumption patterns based on historical data, helping you optimize energy usage, identify potential savings opportunities, and make informed decisions about energy-efficient upgrades or renewable energy investments.''',
                                      my_text_align='left')
                    
        # =============================================================================
        # About ForecastGenie    
        # ============================================================================= 
        if selected_about_page == 'Origin Story':   
            with st.expander('', expanded=True):   
                col1, col2, col3 = st.columns([2,8,2])
                with col2:
                    my_text_header('Origin Story')
                    my_text_paragraph('ForecastGenie is a forecasting app created by Tony Hollaar with a background in data analytics with the goal of making accurate forecasting accessible to businesses of all sizes.')
                    my_text_paragraph('As the developer of the app, I saw a need for a tool that simplifies the process of forecasting, without sacrificing accuracy, therefore ForecastGenie was born.')
                    my_text_paragraph('With ForecastGenie, you can upload your data and quickly generate forecasts using state-of-the-art machine learning algorithms. The app also provides various evaluation metrics to help you benchmark your models and ensure their performance meets your needs.')
                    
                    # DISPLAY LOGO
                    col1, col2, col3 = st.columns([2,8,2])
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

                #stock_ticker(space_quotes, speed = 30)
                #st.markdown('---')
                streamlit_marquee(**{
                    # the marquee container background color
                    'background': "#f5f5f5",
                    # the marquee text size
                    'fontSize': '16px',
                    # the marquee text color
                    "color": "#000000",
                    # the marquee text content
                    'content': scroll_text,
                    # the marquee container width
                    'width': '800px',
                    # the marquee container line height
                    'lineHeight': "0px",
                    # the marquee duration
                    'animationDuration': '60s',
                })
                
        # =============================================================================
        # Show your support 
        # ============================================================================= 
        if selected_about_page == 'Show your support':  
            
            with st.expander('', expanded=True):    
                col1, col2, col3 = st.columns([2,4,2])
                with col2:
                    
                    st.markdown('<h1 style="text-align:center; font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif; font-weight: 200; font-size: 32px; line-height: 1.5;">Show your support {}</h1>'.format(balloon_heart_svg), unsafe_allow_html=True)
                    vertical_spacer(2)
                    my_text_paragraph('If you find this app useful, please consider supporting it by buying me a coffee. Your support helps me continue developing and maintaining this app. Thank you!')
                
                col1, col2, col3 = st.columns([2,2,2])
                with col2:
                    vertical_spacer(2)
                    my_bubbles('')
                    button(username="tonyhollaar", floating=False, width=221,  text = 'Buy me a coffee', bg_color = '#FFFFFF', font='Cookie', font_color='black', coffee_color='black')
                
                    vertical_spacer(14)

        # =============================================================================
        # Versioning 
        # ============================================================================= 
        if selected_about_page == 'Version':
            with st.expander('', expanded=True):  
                my_text_header('Version')
                st.caption(f'<h7><center> ForecastGenie version: `2.0` <br>  Release date: `06-30-2023`  </center></h7>', unsafe_allow_html=True)    
                vertical_spacer(30)              
    except:
        st.error('ForecastGenie Error: "About" in sidebar-menu did not load properly')
print('ForecastGenie Print: Loaded About Page')

# =============================================================================
#   ______      ____  
#  |  ____/\   / __ \ 
#  | |__ /  \ | |  | |
#  |  __/ /\ \| |  | |
#  | | / ____ \ |__| |
#  |_|/_/    \_\___\_\
#                     
# =============================================================================
if sidebar_menu_item == 'FAQ':
    with st.sidebar:
        vertical_spacer(2)
        my_text_paragraph(f'{book_icon}')
        
        col1, col2, col3 = st.columns([2,14,2])
        with col2:
            selected_faq_page = st.selectbox(
                                                label = "*Select Page*:", 
                                                options = ['-',
                                                            'What is ForecastGenie?', 
                                                            'What data can I use with ForecastGenie?', 
                                                            'What forecasting models does ForecastGenie offer?', 
                                                            'What metrics does ForecastGenie use?',
                                                            'Is ForecastGenie really free?',
                                                            'Is ForecastGenie suitable for non-technical users?',
                                                            'Other Questions?'], 
                                                label_visibility='collapsed', 
                                                index = 0,
                                             )
        social_media_links(margin_before=30)

    if selected_faq_page == '-':
        with st.expander('', expanded=True):
            st.image('./images/faq_page.png') # Tree Image     
    
    # FAQ - Questions and Answers
    # =============================================================================
    # What is ForecastGenie?    
    # =============================================================================
    if selected_faq_page == 'What is ForecastGenie?':
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([2,8,2])
            with col2:
                vertical_spacer(10)
                my_text_paragraph('<b>Question:</b> <i> What is ForecastGenie? </i>', 
                                  my_text_align='justify')
                my_text_paragraph('<b> Answer:</b> ForecastGenie is a free, open-source application that enables users to perform time-series forecasting on their data. The application offers a range of advanced features and models to help users generate accurate forecasts and gain insights into their data.', 
                                  my_text_align='justify')
                vertical_spacer(20)
                
    # =============================================================================
    # What data can I use with ForecastGenie?                    
    # =============================================================================
    if selected_faq_page == 'What data can I use with ForecastGenie?':
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([2,8,2])
            with col2:
                vertical_spacer(10)
                my_text_paragraph('<b>Question:</b><i> What data can I use with ForecastGenie?</i>', my_text_align='justify')
                my_text_paragraph('<b>Answer:</b> ForecastGenie accepts data in the form of common file types such as .CSV or .XLS. Hereby the first column should contain the dates and the second column containing the target variable of interest. The application can handle a wide range of time-series data, including financial data, sales data, weather data, and more.',
                                  my_text_align='justify')
                vertical_spacer(20)
                
    # =============================================================================
    # What forecasting models does ForecastGenie offer?                
    # =============================================================================
    if selected_faq_page == 'What forecasting models does ForecastGenie offer?':
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([2,8,2])
            with col2:
                vertical_spacer(10)
                my_text_paragraph('<b>Question:</b> <i>What forecasting models does ForecastGenie offer?</i>', my_text_align='justify')
                my_text_paragraph('<b>Answer:</b> ForecastGenie offers a range of models to suit different data and use cases, including Naive, SARIMAX, and Prophet. The application also includes hyper-parameter tuning, enabling users to optimize the performance of their models and achieve more accurate forecasts.',
                                  my_text_align='justify')
                vertical_spacer(20)
                
    # =============================================================================
    # What metrics does ForecastGenie use?             
    # =============================================================================            
    if selected_faq_page == 'What metrics does ForecastGenie use?':  
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([2,8,2])
            with col2:
                vertical_spacer(10)
                my_text_paragraph('<b>Question: </b><i>What metrics does ForecastGenie use?</i>', my_text_align='justify')
                my_text_paragraph('<b>Answer: </b> ForecastGenie uses a range of business-standard evaluation metrics to assess the accuracy of forecasting models, including Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and more. These metrics provide users with a reliable and objective measure of their model\'s performance.',
                                  my_text_align='justify')
                vertical_spacer(20)
    
    # =============================================================================
    # Is ForecastGenie really free?            
    # =============================================================================
    if selected_faq_page == 'Is ForecastGenie really free?':
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([2,8,2])
            with col2:
                vertical_spacer(10)
                my_text_paragraph('<b>Question: </b><i>Is ForecastGenie really free?</i>',  my_text_align='justify')
                my_text_paragraph('<b>Answer:</b> Yes! ForecastGenie is completely free and open-source.',
                                  my_text_align='justify')
                vertical_spacer(20)
                
    # =============================================================================
    # Is ForecastGenie suitable for non-technical users?         
    # =============================================================================
    if selected_faq_page == 'Is ForecastGenie suitable for non-technical users?':
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([2,8,2])
            with col2:
                vertical_spacer(10)           
                my_text_paragraph('<b>Question:</b><i> Is ForecastGenie suitable for non-technical users?</i>',  my_text_align='justify')
                my_text_paragraph('<b>Answer:</b> Yes! ForecastGenie is designed to be user-friendly and intuitive, even for users with little or no technical experience. The application includes automated data cleaning and feature engineering, making it easy to prepare your data for forecasting. Additionally, the user interface is simple and easy to navigate, with clear instructions and prompts throughout the process.',
                                  my_text_align='justify')
                vertical_spacer(20)
                
    # =============================================================================
    # Is ForecastGenie suitable for non-technical users?         
    # =============================================================================
    if selected_faq_page == 'Other Questions?':
        with st.expander('', expanded=True):    
            
            vertical_spacer(8)
            
            #### Flashcard front / back
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
# LOGGING
print('ForecastGenie Print: Loaded FAQ Page')

# =============================================================================
#   _____   ____   _____ 
#  |  __ \ / __ \ / ____|
#  | |  | | |  | | |     
#  | |  | | |  | | |     
#  | |__| | |__| | |____ 
#  |_____/ \____/ \_____|
#                        
# =============================================================================  
# Documentation
if sidebar_menu_item == 'Doc':
    with st.sidebar:
        vertical_spacer(2)
    
        my_text_paragraph(f'{book_icon}')
        col1, col2, col3 = st.columns([2,14,2])
        
        with col2:
            selected_doc_page = st.selectbox(
                                             label = "*Select Page*:", 
                                             options = ['-',
                                                        'Step 1: Load Dataset', 
                                                        'Step 2: Explore Dataset', 
                                                        'Step 3: Clean Dataset', 
                                                        'Step 4: Engineer Features',
                                                        'Step 5: Prepare Dataset',
                                                        'Step 6: Select Features',
                                                        'Step 7: Train Models',
                                                        'Step 8: Evaluate Models',
                                                        'Step 9: Tune Models',
                                                        'Step 10: Forecast'], 
                                            label_visibility='collapsed', 
                                            index = 0,
                                            )
        # show social media links    
        social_media_links(margin_before=30)
            
    with st.expander('', expanded=True):
        # DOC: FRONT PAGE
        ################################
        if selected_doc_page == '-':
            st.image('./images/documentation.png')
       
        # DOC: LOAD
        ################################
        if selected_doc_page == 'Step 1: Load Dataset':   
            
            my_text_header('<b> Step 1: </b> <br> Load Dataset')
            
            show_lottie_animation(url="./images/116206-rocket-fly-out-the-laptop.json", key="rocket_fly_out_of_laptop", height=200, width=200, speed = 1, loop=True, quality='high', col_sizes = [4,4,4])
            
            col1, col2, col3 = st.columns([2,8,2])    
            with col2:
               my_text_paragraph('''The <i> ForecastGenie </i> application provides users with a convenient way to upload data files from the sidebar menu. The application supports common file formats such as .csv and .xls. \
                                    To load a file, users can navigate to the sidebar menu and locate the <i> "Upload Data" </i> option. Upon clicking on this option, a file upload dialog box will appear, allowing users to select a file from their local system. \
                                    When uploading a file, it is important to ensure that the file meets specific requirements for proper data processing and forecasting. 
                                    <br> - The first column should have a header named <i> 'date' </i> and dates should be in the mm/dd/yyyy format (12/31/2023), representing the time series data. The dates should be sorted in chronological order, as this is crucial for accurate forecasting. 
                                    <br> - The second column should contain a header that includes the name of the variable of interest and below it's historical data, for which the user wishes to forecast.
                                    <br> Fasten your seatbelt, lean back, and savor the journey as your data blasts off into the realm of forecasting possibilities! Embrace the adventure and enjoy the ride.
                                    ''', my_text_align='justify')
            vertical_spacer(2)
      
        # DOC: EDA
        ################################      
        if selected_doc_page == 'Step 2: Explore Dataset':
            
            my_text_header('<b> Step 2: </b> <br> Explore Dataset')
            
            show_lottie_animation(url = "./images/58666-sputnik-mission-launch.json", 
                                  key = 'test', 
                                  width = 150, 
                                  height = 150,
                                  speed = 1, 
                                  col_sizes = [45, 40, 40], 
                                  margin_before = 1)
            
            col1, col2, col3 = st.columns([2,8,2])    
            with col2:
                my_text_paragraph('''
                                  The <i> Explore </i> tab in the app is designed to provide users with comprehensive tools for data exploration and analysis. It offers valuable insights into the dataset.
                                  <br> The <i> Quick Summary </i> section provides an overview of the dataset including the number of rows, start date, missing values, mean, minimum, standard deviation, maximum date, frequency, median, maximum, and mode.
                                  <br> The <i> Quick Insights </i>  section is designed to get the summarized observations that highlight important characteristics of the data. This includes indications of dataset size, presence or absence of missing values, balance between mean and median values, variability, symmetry, and the number of distinct values.
                                  <br> The <i> Patterns </i> section allows users to visually explore underlying patterns and relationships within the data. Users can select different histogram frequency types to visualize data distribution. The Ljung-Box test is available to assess the presence of white noise. The Augmented Dickey-Fuller (ADF) test can be used to determine stationarity, indicating whether the data exhibits time-dependent patterns. 
                                  Additionally, users can analyze autocorrelation using the ACF/PACF to understand how data point relates to previous data points.
                                  <br> Do not skip exploratory data analysis, because you don't want to end up finding yourself lost in a black hole!
                                  ''', my_text_align = 'justify')
                vertical_spacer(2)
        
        # DOC: Clean
        ################################ 
        if selected_doc_page == 'Step 3: Clean Dataset':                   
            
            my_text_header('<b> Step 3: </b> <br> Clean Dataset')
            
            show_lottie_animation(url="./images/88404-loading-bubbles.json", key="loading_bubbles", width=200, height=200, col_sizes=[2,2,2])
            
            col1, col2, col3 = st.columns([2,8,2])   
            with col2:
                my_text_paragraph('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam pretium nisl vel mauris congue, non feugiat neque lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Sed ullamcorper massa ut ligula sagittis tristique. Donec rutrum magna vitae felis finibus, vitae eleifend nibh commodo. Aliquam fringilla dui a tellus interdum vestibulum. Vestibulum pharetra, velit et cursus commodo, erat enim eleifend massa, ac pellentesque velit turpis nec ex. Fusce scelerisque, velit non lacinia iaculis, tortor neque viverra turpis, in consectetur diam dui a urna. Quisque in velit malesuada, scelerisque tortor vel, dictum massa. Quisque in malesuada libero.', my_text_align='justify')
        
        # DOC: Engineer
        ################################ 
        if selected_doc_page == 'Step 4: Engineer Features':
            
            my_text_header('<b> Step 4: </b> <br> Engineer Features')
            
            show_lottie_animation(url="./images/141844-shapes-changing-preloader.json", key='shapes_changing_preloader', width=200, height=200, col_sizes=[2,2,2])
            
            col1, col2, col3 = st.columns([2,8,2])   
            with col2:
                my_text_paragraph('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam pretium nisl vel mauris congue, non feugiat neque lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Sed ullamcorper massa ut ligula sagittis tristique. Donec rutrum magna vitae felis finibus, vitae eleifend nibh commodo. Aliquam fringilla dui a tellus interdum vestibulum. Vestibulum pharetra, velit et cursus commodo, erat enim eleifend massa, ac pellentesque velit turpis nec ex. Fusce scelerisque, velit non lacinia iaculis, tortor neque viverra turpis, in consectetur diam dui a urna. Quisque in velit malesuada, scelerisque tortor vel, dictum massa. Quisque in malesuada libero.', my_text_align='justify')
        
        # DOC: Prepare
        ################################    
        if selected_doc_page == 'Step 5: Prepare Dataset':
            
            my_text_header('<b> Step 5: </b> <br> Prepare Dataset')
            
            show_lottie_animation(url="./images/141560-loader-v25.json", key='prepare', width=200, height=200, col_sizes=[20,20,20])
            
            col1, col2, col3 = st.columns([2,8,2])   
            with col2:
                my_text_paragraph('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam pretium nisl vel mauris congue, non feugiat neque lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Sed ullamcorper massa ut ligula sagittis tristique. Donec rutrum magna vitae felis finibus, vitae eleifend nibh commodo. Aliquam fringilla dui a tellus interdum vestibulum. Vestibulum pharetra, velit et cursus commodo, erat enim eleifend massa, ac pellentesque velit turpis nec ex. Fusce scelerisque, velit non lacinia iaculis, tortor neque viverra turpis, in consectetur diam dui a urna. Quisque in velit malesuada, scelerisque tortor vel, dictum massa. Quisque in malesuada libero.', my_text_align='justify')
       
        # DOC: Select
        ################################   
        if selected_doc_page == 'Step 6: Select Features':
            
            my_text_header('<b> Step 6: </b> <br> Select Features')
            
            show_lottie_animation(url="./images/102149-square-loader.json", key='square_loader', width=200, height=200, col_sizes = [4,4,4])
            
            col1, col2, col3 = st.columns([2,8,2])   
            with col2:
                my_text_paragraph('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam pretium nisl vel mauris congue, non feugiat neque lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Sed ullamcorper massa ut ligula sagittis tristique. Donec rutrum magna vitae felis finibus, vitae eleifend nibh commodo. Aliquam fringilla dui a tellus interdum vestibulum. Vestibulum pharetra, velit et cursus commodo, erat enim eleifend massa, ac pellentesque velit turpis nec ex. Fusce scelerisque, velit non lacinia iaculis, tortor neque viverra turpis, in consectetur diam dui a urna. Quisque in velit malesuada, scelerisque tortor vel, dictum massa. Quisque in malesuada libero.', my_text_align='justify')
            
        # DOC: Train
        ################################   
        if selected_doc_page == 'Step 7: Train Models':
           
            my_text_header('<b> Step 7: </b> <br> Train Models')
            
            show_lottie_animation(url="./images/100037-rubiks-cube.json", key='rubiks_cube', width=200, height=200, col_sizes = [4,4,4])
            
            col1, col2, col3 = st.columns([2,8,2])   
            with col2:
                my_text_paragraph('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam pretium nisl vel mauris congue, non feugiat neque lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Sed ullamcorper massa ut ligula sagittis tristique. Donec rutrum magna vitae felis finibus, vitae eleifend nibh commodo. Aliquam fringilla dui a tellus interdum vestibulum. Vestibulum pharetra, velit et cursus commodo, erat enim eleifend massa, ac pellentesque velit turpis nec ex. Fusce scelerisque, velit non lacinia iaculis, tortor neque viverra turpis, in consectetur diam dui a urna. Quisque in velit malesuada, scelerisque tortor vel, dictum massa. Quisque in malesuada libero.', my_text_align='justify')
        
        # DOC: Evaluate
        ################################    
        if selected_doc_page == 'Step 8: Evaluate Models':
            
            my_text_header('<b> Step 8: </b> <br> Evaluate Models')
            
            show_lottie_animation(url="./images/70114-blue-stars.json", key='blue-stars', width=200, height=200, col_sizes = [4,4,4], speed = 1)
            
            col1, col2, col3 = st.columns([2,8,2])   
            with col2:
                my_text_paragraph('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam pretium nisl vel mauris congue, non feugiat neque lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Sed ullamcorper massa ut ligula sagittis tristique. Donec rutrum magna vitae felis finibus, vitae eleifend nibh commodo. Aliquam fringilla dui a tellus interdum vestibulum. Vestibulum pharetra, velit et cursus commodo, erat enim eleifend massa, ac pellentesque velit turpis nec ex. Fusce scelerisque, velit non lacinia iaculis, tortor neque viverra turpis, in consectetur diam dui a urna. Quisque in velit malesuada, scelerisque tortor vel, dictum massa. Quisque in malesuada libero.', my_text_align='justify')
            
        # DOC: Tune
        ################################    
        if selected_doc_page == 'Step 9: Tune Models':
            
            my_text_header('<b> Step 9: </b> <br> Tune Models')
            
            show_lottie_animation(url="./images/95733-loading-20.json", key='tune_sliders', width=150, height=150, col_sizes = [5,4,4], speed = 1)
            
            col1, col2, col3 = st.columns([2,8,2])   
            with col2:
                my_text_paragraph('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam pretium nisl vel mauris congue, non feugiat neque lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Sed ullamcorper massa ut ligula sagittis tristique. Donec rutrum magna vitae felis finibus, vitae eleifend nibh commodo. Aliquam fringilla dui a tellus interdum vestibulum. Vestibulum pharetra, velit et cursus commodo, erat enim eleifend massa, ac pellentesque velit turpis nec ex. Fusce scelerisque, velit non lacinia iaculis, tortor neque viverra turpis, in consectetur diam dui a urna. Quisque in velit malesuada, scelerisque tortor vel, dictum massa. Quisque in malesuada libero.', my_text_align='justify')
    
        # DOC: Forecast
        ################################   
        if selected_doc_page == 'Step 10: Forecast':
            
            my_text_header('<b> Step 10: </b> <br> Forecast')
            
            show_lottie_animation(url="./images/55298-data-forecast-loading.json", key='forecast', width=200, height=200, col_sizes = [4,4,4], speed = 1)
            
            col1, col2, col3 = st.columns([2,8,2])   
            with col2:
                my_text_paragraph('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam pretium nisl vel mauris congue, non feugiat neque lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Sed ullamcorper massa ut ligula sagittis tristique. Donec rutrum magna vitae felis finibus, vitae eleifend nibh commodo. Aliquam fringilla dui a tellus interdum vestibulum. Vestibulum pharetra, velit et cursus commodo, erat enim eleifend massa, ac pellentesque velit turpis nec ex. Fusce scelerisque, velit non lacinia iaculis, tortor neque viverra turpis, in consectetur diam dui a urna. Quisque in velit malesuada, scelerisque tortor vel, dictum massa. Quisque in malesuada libero.', my_text_align='justify')
# LOGGING
print('Forecastgenie Print: Loaded Documentation Page')

# =============================================================================
#   _      ____          _____  
#  | |    / __ \   /\   |  __ \ 
#  | |   | |  | | /  \  | |  | |
#  | |   | |  | |/ /\ \ | |  | |
#  | |___| |__| / ____ \| |__| |
#  |______\____/_/    \_\_____/ 
#                               
# =============================================================================          
def reset_session_states():
    """
    Resets the session states for variables by deleting all items in Session state, except for 'data_choice' and 'my_data_choice'.
    
    This function is useful when a new file is uploaded and you want to clear the existing session state.
    
    Returns:
        None
    
    Example:
        reset_session_states()
    """
    #st.write('TEST: I am going to delete session states for variables as new file is uploaded!') # TEST 
            
    # Delete all the items in Session state
    for key in st.session_state.keys():
        ############### NEW TEST ############################
        # FIX: review need of 2 session states
        if key == 'data_choice' or key == 'my_data_choice' or key == "__LOAD_PAGE-my_data_choice__" or key == "__LOAD_PAGE-user_data_uploaded__":
            pass
        else:
            del st.session_state[key]
            
    # reset session states to default -> initiate global variables
    metrics_dict, random_state, results_df, custom_fill_value, \
    key1_load, key2_load, key3_load, \
    key1_explore, key2_explore, key3_explore, key4_explore, \
    key1_missing, key2_missing, key3_missing, \
    key1_outlier, key2_outlier, key3_outlier, key4_outlier, key5_outlier, key6_outlier, key7_outlier, \
    key1_engineer, key2_engineer, key3_engineer, key4_engineer, key5_engineer, key6_engineer, key7_engineer, \
    key1_engineer_page_country, key2_engineer_page_country, key3_engineer_page_country, \
    key1_prepare_normalization, key2_prepare_standardization, \
    key1_select_page_user_selection, \
    key1_select_page_pca, \
    key1_select_page_rfe, key2_select_page_rfe, key3_select_page_rfe, key4_select_page_rfe, key5_select_page_rfe, key6_select_page_rfe, \
    key1_select_page_mifs, \
    key1_select_page_corr, key2_select_page_corr, \
    key1_train, key2_train, key3_train, key4_train, key5_train, key6_train, key7_train, key8_train, key9_train, key10_train, \
    key11_train, key12_train, key13_train, key14_train, key15_train, key16_train, key17_train, key18_train, key19_train, key20_train, \
    key21_train, key22_train, key23_train, key24_train, key25_train, key26_train, key27_train, key28_train, key29_train, key30_train, key31_train, key32_train, \
    key1_evaluate, key2_evaluate = initiate_global_variables()

def load_change():
    """
    Updates the in-memory value for a radio button representing the choice of data source.

    The function updates the value of the radio button based on the current selection. If the current selection is
    "Demo Data", the radio button value is changed to "Upload Data". If the current selection is "Upload Data", the
    radio button value is changed to "Demo Data".

    Raises:
        Exception: If an error occurs while updating the radio button value, it is caught and the radio button value is
                   set to "Demo Data" as a fallback.

    Note:
        This function assumes the use of a global state management system, such as Streamlit's `set_state()` function,
        to store and update the state of the radio button value.

    Example usage:
        load_change()
    """
    try:
        index_data_option = ("Upload Data" if data_option == "Demo Data" else "Demo Data")
        set_state("LOAD_PAGE", ("my_data_choice", index_data_option))
    except:
        set_state("LOAD_PAGE", ("my_data_choice", "Demo Data"))
        
if menu_item == 'Load' and sidebar_menu_item == 'Home':
    tab1_load, tab2_load = st.tabs(['Load', 'Dashboard'])
     
    with tab2_load:
        
        def calculate_relative_change(data, date_column, value_column, relative_change_type):
            """
            Calculate the relative change of a given value column based on the selected type of relative change.
        
            Args:
                data (pandas.DataFrame): Input DataFrame containing the date and value columns.
                date_column (str): Name of the date column.
                value_column (str): Name of the value column.
                relative_change_type (str): Type of relative change to calculate.
        
            Returns:
                pandas.Series: Relative change of the value column based on the selected type of relative change.
            """
            # Convert the date column to a pandas DateTimeIndex
            data = data.copy(deep=True)
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
        
        
        # Example usage with Streamlit
        # Assuming your data is stored in a pandas DataFrame called 'sales_data'
        # and has columns 'date' and 'value'
        
        # Retrieve the column name based on its relative position (position 1)
        value_column = st.session_state['df_raw'].iloc[:, 1].name
        
        # Call the function to calculate the relative change
        relative_change_type = st.selectbox('Select relative change type:', ['DoD%Δ', 'WoW%Δ', 'MoM%Δ', 'YoY%Δ'])
        
# =============================================================================
#         relative_change = calculate_relative_change(
#             data=st.session_state['df_raw'],
#             date_column='date',
#             value_column=value_column,
#             relative_change_type=relative_change_type
#         )
#         
#         # Filter out empty (NaN) values when comparing relative change
#         relative_change = relative_change.dropna()
#         
#         # Create the bar chart using Plotly Express
#         fig = px.bar(x=relative_change.index, y=relative_change.values)
#         
#         fig.update_layout(
#             xaxis_title='Day of the Week',
#             yaxis_title='Relative Change (%)',
#             xaxis=dict(
#                 rangeselector=dict(
#                     buttons=list([
#                         dict(count=7, label='1w', step='day', stepmode='backward'),
#                         dict(count=1, label='1m', step='month', stepmode='backward'),
#                         dict(count=6, label='6m', step='month', stepmode='backward'),
#                         dict(count=1, label='YTD', step='year', stepmode='todate'),
#                         dict(count=1, label='1y', step='year', stepmode='backward'),
#                         dict(step='all')
#                     ]),
#                     x=0.35,
#                     y=0.9,
#                     yanchor='auto',
#                     font=dict(size=10),
#                 ),
#                 rangeslider=dict(
#                     visible=True,
#                     range=[relative_change.index.min(), relative_change.index.max()]
#                 ),
#                 type='date'
#             ),
#             height=600,  # Adjust the height value as needed
#             title={
#                 'text': f'Relative Change by {relative_change_type}',
#                 'x': 0.5,  # Center the title horizontally
#                 'xanchor': 'center',  # Center the title horizontally
#                 'y': 0.9  # Adjust the vertical position of the title if needed
#             }
#         )
#         
#         # Add labels inside the bars
#         fig.update_traces(
#             texttemplate='%{y:.2f}%',
#             textposition='inside',
#             insidetextfont=dict(color='white')
#         )
#         
#         st.plotly_chart(fig)
# =============================================================================
    
    
    with tab1_load:
        # =============================================================================
        # SIDEBAR OF LOAD PAGE
        # =============================================================================
        with st.sidebar:
            my_title(f"{load_icon}", "#3b3b3b")
            with st.expander('', expanded = True):            
                
                col1, col2, col3 = st.columns(3)
                
                with col2:
                    # =============================================================================
                    #  STEP 1: SAVE USER DATA CHOICE (DEMO/UPLOAD) WITH RADIO BUTTON
                    # =============================================================================         
                    data_option = st.radio(
                                           label = "*Choose an option:*", 
                                           options = ["Demo Data", "Upload Data"], 
                                           index = 0 if get_state('LOAD_PAGE', "my_data_choice") == "Demo Data" else 1,
                                           on_change = load_change # run function
                                           )
                    
                    # UPDATE SESSION STATE WITH USER DATA CHOICE OF RADIO BUTTON
                    set_state("LOAD_PAGE", ("my_data_choice", data_option))
    
                    vertical_spacer(1)
                    
                st.image('./images/under_construction.png') #UNDER CONSTRUCTION
                # =============================================================================
                # STEP 2: IF UPLOADED FILE THEN SHOW FILE-UPLOADER WIDGET
                # =============================================================================
                # if user selects radio-button for 'Upload Data'
                if data_option == "Upload Data":
                   # show file-uploader widget from streamlit
                   uploaded_file = st.file_uploader(label = "Upload your file", 
                                                     type = ["csv", "xls", "xlsx", "xlsm", "xlsb"], 
                                                         accept_multiple_files = False, 
                                                     label_visibility = 'collapsed',
                                                     on_change = reset_session_states)
                   # if a file is uploaded by user
                   if uploaded_file != None:
                       # store the filename in session state
                       set_state("LOAD_PAGE", ("uploaded_file_name", uploaded_file.name))
                   # if prior a file was uploaded by user (note: st.file_uploader after page refresh does not keep the object in it)
                   if get_state('LOAD_PAGE', "user_data_uploaded") == True:
                       # retrieve filename from session state                    
                       file_name = get_state("LOAD_PAGE", "uploaded_file_name")
                       # show filename
                       my_text_paragraph(f'{file_name}', my_font_weight=100, my_font_size='14px')
                       
        # =============================================================================
        # MAIN PAGE OF LOAD PAGE
        # =============================================================================
        if get_state('LOAD_PAGE', "my_data_choice") == "Demo Data":
           
            # update session state that user uploaded a file
            set_state("LOAD_PAGE", ("user_data_uploaded", False))
            
            # GENERATE THE DEMO DATA WITH FUNCTION
            st.session_state['df_raw'] = generate_demo_data()
            
            df_raw = generate_demo_data()
            df_graph = df_raw.copy(deep=True)
            df_min = df_raw.iloc[:,0].min().date()
            df_max = df_raw.iloc[:,0].max().date()
            
            with st.expander('', expanded=True):
                col0, col1, col2, col3 = st.columns([10, 90, 8, 1])        
                with col2:
                    my_chart_color = st.color_picker(label = 'Color', 
                                                     value = get_state("COLORS", "chart_color"), 
                                                     #on_change = on_click_event,
                                                     label_visibility = 'collapsed')    
    
                    # save user color choice in session state
                    set_state("COLORS", ("chart_color", my_chart_color))
                
                with col1:
                   my_text_header('Demo Data')     
                
                #show_lottie_animation("./images/16938-asteroid.json", key='rocket_launch', speed=1, height=200, width=200, col_sizes=[4,4,4])
                     
                # create 3 columns for spacing
                col1, col2, col3 = st.columns([1,3,1])
                
                # short message about dataframe that has been loaded with shape (# rows, # columns)
                col2.markdown(f"<center>Your <b>dataframe</b> has <b><font color='#555555'>{st.session_state.df_raw.shape[0]}</b></font> \
                               rows and <b><font color='#555555'>{st.session_state.df_raw.shape[1]}</b></font> columns <br> with date range: \
                               <b><font color='#555555'>{df_min}</b></font> to <b><font color='#555555'>{df_max}</font></b>.</center>", unsafe_allow_html=True)
                    
                # create aa deep copy of dataframe which will be manipulated for graphs
                df_graph = copy_df_date_index(my_df=df_graph, datetime_to_date=True, date_to_index=True)
                            
                # Display Plotly Express figure in Streamlit
                display_dataframe_graph(df=df_graph, key=1, my_chart_color = my_chart_color)
    
                # try to use add-on package of streamlit dataframe_explorer
                try:
                    df_explore = dataframe_explorer(st.session_state['df_raw'])
                    st.dataframe(df_explore, use_container_width=True)
                # if add-on package does not work use the regular dataframe without index
                except:
                    st.dataframe(df_graph, use_container_width=True)
                
                # download csv button
                download_csv_button(df_graph, my_file="raw_data.csv", help_message='Download dataframe to .CSV', set_index=True)
                
                vertical_spacer(1)
        
            # 
            st.image('./images/load_page.png')
        
        # check if data is uploaded in file_uploader, if so -> use the function load_data to load file into a dataframe 
        if get_state('LOAD_PAGE', "my_data_choice") == "Upload Data" and uploaded_file is not None:
            # define dataframe from custom function to read from uploaded read_csv file
            st.session_state['df_raw'] = load_data()
            
            # update session state that user uploaded a file
            set_state("LOAD_PAGE", ("user_data_uploaded", True))
            
            # update session state that new file is uploaded / set to True -> use variable reset user selection of features list to empty list
            set_state("DATA_OPTION", ("upload_new_data", True))
    
        # for multi-page app if user previously uploaded data uploaded_file will return to None but then get the persistent session state stored
        if get_state('LOAD_PAGE', "my_data_choice") == "Upload Data" and ((uploaded_file is not None) or (get_state("LOAD_PAGE", "user_data_uploaded") == True)) and 'demo_data' not in st.session_state['df_raw'].columns:
            
            # define dataframe copy used for graphs
            df_graph = st.session_state['df_raw'].copy(deep=True)
            # set minimum date
            df_min = st.session_state['df_raw'].iloc[:,0].min().date()
            # set maximum date
            df_max = st.session_state['df_raw'].iloc[:,0].max().date()
            
            # let user select color for graph
            with st.expander('', expanded=True):
                col0, col1, col2, col3 = st.columns([15, 90, 8, 1])        
                with col2:
                    my_chart_color = st.color_picker(label = 'Color', 
                                                     value = get_state("COLORS", "chart_color"), 
                                                     #on_change = on_click_event,
                                                     label_visibility = 'collapsed')    
                    
                    set_state("COLORS", ("chart_color", my_chart_color))
                
                with col1:    
                    my_text_header('Uploaded Data')
                     
                # create 3 columns for spacing
                col1, col2, col3 = st.columns([1,3,1])
                
                # display df shape and date range min/max for user
                col2.markdown(f"<center>Your <b>dataframe</b> has <b><font color='#555555'>{st.session_state.df_raw.shape[0]}</b></font> \
                              rows and <b><font color='#555555'>{st.session_state.df_raw.shape[1]}</b></font> columns <br> with date range: \
                              <b><font color='#555555'>{df_min}</b></font> to <b><font color='#555555'>{df_max}</font></b>.</center>", unsafe_allow_html=True)
                    
                vertical_spacer(1)
                
                df_graph = copy_df_date_index(my_df=df_graph, datetime_to_date=True, date_to_index=True)            
                
                vertical_spacer(1)
                
                # display/plot graph of dataframe
                display_dataframe_graph(df=df_graph, key=2, my_chart_color = my_chart_color)
                
                try:
                    # show dataframe below graph
                    df_explore = dataframe_explorer(st.session_state['df_raw'])
                    st.dataframe(df_explore, use_container_width=True)
                except:
                    st.dataframe(df_graph, use_container_width=True)
                
                # download csv button
                download_csv_button(df_graph, my_file="raw_data.csv", help_message='Download dataframe to .CSV', set_index=True)
                
        elif get_state('LOAD_PAGE', "my_data_choice") == "Upload Data" and ((uploaded_file is None) or (get_state("LOAD_PAGE", "user_data_uploaded") == False)):
            # =============================================================================
            # Inform user what template to upload
            # =============================================================================
            with st.expander("", expanded=True):
                
                my_text_header("Instructions")
                
                vertical_spacer(2)
                
                col1, col2, col3 = st.columns([1,8,1])
                with col2:
                    my_text_paragraph('''👈 Please upload a file with your <b><span style="color:#000000">dates</span></b> and <b><span style="color:#000000">values</span></b> in below order:<br><br>
                             - first column named: <b><span style="color:#000000">date</span></b> in format: mm/dd/yyyy e.g. 12/31/2023<br>
                             - second column named:  <b><span style="color:#000000">&#60;insert variable name&#62;</span></b> e.g. revenue<br>
                             - supported frequencies: Daily/Weekly/Monthly/Quarterly/Yearly <br>
                             - supported file extensions: .CSV, .XLS, .XLSX, .XLSM, .XLSB
                             ''', my_font_weight=300, my_text_align='left')
                    
                    vertical_spacer(2)
                           
                # Display Doodle image of Load 
                st.image("./images/load.png", caption="", use_column_width=True)
                
# =============================================================================
#   ________   _______  _      ____  _____  ______ 
#  |  ____\ \ / /  __ \| |    / __ \|  __ \|  ____|
#  | |__   \ V /| |__) | |   | |  | | |__) | |__   
#  |  __|   > < |  ___/| |   | |  | |  _  /|  __|  
#  | |____ / . \| |    | |___| |__| | | \ \| |____ 
#  |______/_/ \_\_|    |______\____/|_|  \_\______|
#                                                  
# =============================================================================         
if menu_item == 'Explore' and sidebar_menu_item == 'Home':  
    # =============================================================================
    # Define Progress bar
    # =============================================================================
    progress_text = "Preparing your exploratory data analysis... Please wait!"
    my_bar = st.progress(0, text = progress_text)
    #with st.spinner('Loading your summary, insights, patterns, statistical tests, lag analysis...'):
    
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
        my_title(f'{explore_icon}', my_background_color="#3b3b3b", gradient_colors="#217CD0,#555555")
        
        # set color picker for user centered on page
        def update_color():
            # set session state for user chosen chart color
            set_state("COLORS", ("chart_patterns", my_chart_color))
            
        with st.form('ljung-box'):
             my_text_paragraph('White Noise')
         
             lag1_ljung_box = st.number_input(label = '*Enter maximum lag:*', 
                                              min_value = 1, 
                                              value = min(24, len(st.session_state.df_raw)-2), 
                                              max_value = len(st.session_state.df_raw)-2,
                                              key= 'lag1_ljung_box',
                                              help = 'the lag parameter in the Ljung-Box test determines the number of time periods over which the autocorrelation of residuals is evaluated to assess the presence of significant autocorrelation in the time series.')
             
             col1, col2, col3 = st.columns([5,4,4])
             with col2:
                # create button in sidebar for the white noise (Ljung-Box) test
                vertical_spacer(1)
                ljung_box_btn = st.form_submit_button("Submit", type="secondary")
                
        # Autocorrelation parameters form     
        with st.form('autocorrelation'):
            # Create sliders in sidebar for the parameters of PACF Plot
            my_text_paragraph('Autocorrelation ')
            col1, col2, col3 = st.columns([4,1,4])
        
            # If dataset is very small (less then 31 rows), then update the key1_explore and key2_explore
            if len(st.session_state['df_raw']) < 31:
                set_state("EXPLORE_PAGE", ("lags_acf", int((len(st.session_state['df_raw'])-1))))
                set_state("EXPLORE_PAGE", ("lags_pacf", int((len(st.session_state['df_raw'])-1)/2)))

            # create slider for number of lags for ACF Plot
            nlags_acf = st.slider(label = "*Lags ACF*", 
                                  min_value = 1, 
                                  max_value = (len(st.session_state['df_raw'])-1), 
                                  key = key1_explore)
            
            col1, col2, col3 = st.columns([4,1,4])
            with col1:
                # create slider for number of lags for PACF Plot
                nlags_pacf = st.slider(label = "*Lags PACF*", 
                                       min_value = 1, 
                                       max_value = int((len(st.session_state['df_raw'])-2)/2), 
                                       key = key2_explore)
            with col3:
                # create dropdown menu for method to calculate the PACF Plot
                method_pacf = st.selectbox(label = "*Method PACF*", 
                                           options = [ 'ols', 'ols-inefficient', 'ols-adjusted', 'yw', 'ywa', 'ld', 'ywadjusted', 'yw_adjusted', 'ywm', 'ywmle', 'yw_mle', 'lda', 'ldadjusted', 'ld_adjusted', 'ldb', 'ldbiased', 'ld_biased'], 
                                           key = key3_explore)
                
            # create dropdown menu to select if you want the original series or differenced series that will also impact the ACF & PACF Plots
            # next to having a preview in sidebar with the differenced series plotted if user selects 1st ,2nd or 3rd order difference
            selection = st.selectbox('*Apply Differencing [Optional]:*', 
                                     options = ['Original Series', 'First Order Difference', 'Second Order Difference', 'Third Order Difference'],
                                     key = key4_explore)
            
            col1, col2, col3 = st.columns([5,4,4])
            with col2:
                # create button in sidebar for the ACF and PACF Plot Parameters
                vertical_spacer(1)
                acf_pacf_btn = st.form_submit_button("Submit", type="secondary",  on_click=form_update, args=('EXPLORE_PAGE',))   
         
    ####################################################            
    # Explore MAIN PAGE (EDA)
    ####################################################
    with tab1:
        #my_title(f'{explore_icon} Exploratory Data Analysis ', my_background_color="#217CD0", gradient_colors="#217CD0,#555555")
        # create expandable card with data exploration information
        with st.expander('', expanded=True):
            col0, col1, col2, col3 = st.columns([18, 90, 8, 1])   
            with col2:
               	my_chart_color = st.color_picker(label = 'Color', 
               									 value = get_state("COLORS", "chart_patterns"), 
               									 label_visibility = 'collapsed',
                                                  help = 'Set the **`color`** of the charts and styling elements. It will revert back to the **default** color when switching pages.')

            #############################################################################
            # Quick Summary Results
            #############################################################################
            with col1:
                my_text_header('Quick Summary')
            
            ## show the flashcard with the quick summary results on page    
            # eda_quick_summary(my_chart_color)
            
            col1, col2, col3 = st.columns([7,120,1])
            with col2:
                vertical_spacer(1)
                
                create_flipcard_quick_summary(header_list = '',  
                                              paragraph_list_front = '', 
                                              paragraph_list_back = '', 
                                              font_family = 'Arial', 
                                              font_size_front ='16px', 
                                              font_size_back ='16px', 
                                              image_path_front_card = './images/quick_summary.png',
                                              my_chart_color = '#FFFFFF')
                
            # show button and if clicked, show dataframe
            col1, col2, col3 = st.columns([100,50,95])
            with col2:        
                placeholder = st.empty()
                # create button (enabled to click e.g. disabled=false with unique key)
                btn_summary_stats = placeholder.button('Show Details', disabled=False,  key = "summary_statistics_show_btn")
    
            # if button is clicked run below code
            if btn_summary_stats == True:
                # display button with text "click me again", with unique key
                placeholder.button('Hide Details', disabled=False, key = "summary_statistics_hide_btn")
               
                # Display summary statistics table
                summary_stats_df = display_summary_statistics(st.session_state.df_raw)
                
                st.dataframe(summary_stats_df, use_container_width=True)
                
                download_csv_button(summary_stats_df, my_file="summary_statistics.csv", help_message='Download your Summary Statistics Dataframe to .CSV', my_key = "summary_statistics_download_btn")      
            
            vertical_spacer(1)
            
            # Show Summary Statistics and statistical test results of dependent variable (y)
            summary_statistics_df = create_summary_df(data = st.session_state.df_raw.iloc[:,1])
        
    my_bar.progress(20, text = 'loading quick insights...')
    #######################################
    # Quick Insights Results
    #####################################
    with tab2:
        with st.expander('', expanded=True):   
            # old metrics card
            #eda_quick_insights(df=summary_statistics_df, my_string_column='Label', my_chart_color = my_chart_color)
            
            col1, col2, col3 = st.columns([7,120,1])
            with col2:  
                create_flipcard_quick_insights(1, header_list = [''],  
                                                paragraph_list_front = [''], 
                                                paragraph_list_back = [''], 
                                                font_family = 'Arial', 
                                                font_size_front ='12px', 
                                                font_size_back ='12px', 
                                                image_path_front_card = './images/futuristic_city_robot.png',
                                                df = summary_statistics_df, 
                                                my_string_column='Label', 
                                                my_chart_color = '#FFFFFF')
               
            # have button available for user and if clicked it expands with the dataframe
            col1, col2, col3 = st.columns([100,50,95])
            with col2:        
                placeholder = st.empty()
                
                # create button (enabled to click e.g. disabled=false with unique key)
                btn_insights = placeholder.button('Show Details', disabled=False, key = "insights_statistics_show_btn")
                
                vertical_spacer(1)
                
            # if button is clicked run below code
            if btn_insights == True:
                # display button with text "click me again", with unique key
                placeholder.button('Hide Details', disabled=False,  key = "insights_statistics_hide_btn")
                
                st.dataframe(summary_statistics_df, use_container_width=True)
               
                download_csv_button(summary_statistics_df, my_file="insights.csv", help_message='Download your Insights Dataframe to .CSV', my_key = "insights_download_btn")      
    
    my_bar.progress(40, text = 'loading patterns')
    with tab3:
        #############################################################################
        # Call function for plotting Graphs of Seasonal Patterns D/W/M/Q/Y in Plotly Charts
        #############################################################################
        with st.expander('', expanded=True):
            
            # Update layout
            my_text_header('Patterns')
            
            # show all graphs with patterns in streamlit
            plot_overview(df = st.session_state.df_raw, 
                          y = st.session_state.df_raw.columns[1])
      
            # radio button for user to select frequency of hist: absolute values or relative
            col1, col2, col3 = st.columns([10,10,7])
            with col2:
                # Add radio button for frequency type of histogram
                frequency_type = st.radio(label = "*histogram frequency type:*", 
                                          options = ("Absolute", "Relative"), 
                                          index = 1 if get_state("HIST", "histogram_freq_type") == "Relative" else 0,
                                          on_change = hist_change_freq,
                                          horizontal = True)
                
                vertical_spacer(3)
                
        st.image('./images/patterns_banner.png')
        
    my_bar.progress(60, text = 'loading statistical tests')
    with tab4:
        # =============================================================================
        # ################################## TEST #####################################
        # =============================================================================
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([7,120,1])
            with col2:  
                                
                # apply function flipcard
                create_flipcard_stats(image_path_front_card='./images/statistical_test.png', my_header = 'Test Results')
        
        ###################################################################
        # LJUNG-BOX STATISTICAL TEST FOR WHITE NOISE e.g. random residuals
        ###################################################################
        # Perform the Ljung-Box test on the residuals
        with st.expander('White Noise', expanded=False):
            
            my_text_header('White Noise')
            
            my_text_paragraph('Ljung-Box')
            
            col1, col2, col3 = st.columns([18,44,10])
            with col2:
                vertical_spacer(2)
                
                res, result_ljungbox = ljung_box_test(df = st.session_state.df_raw,
                                                      variable_loc = 1, 
                                                      lag = lag1_ljung_box,
                                                      model_type = "AutoReg")
            
            ljung_box_plots(df = st.session_state['df_raw'], 
                            variable_loc = 1,
                            lag = lag1_ljung_box,
                            res = res,
                            result_ljungbox = result_ljungbox,
                            my_chart_color = my_chart_color)

        ###################################################################  
        # AUGMENTED DICKEY-FULLER TEST
        ###################################################################
        # Show Augmented Dickey-Fuller Statistical Test Result with hypotheses
        with st.expander('Stationarity', expanded=False):
            
            my_text_header('Stationarity')
            my_text_paragraph('Augmented Dickey Fuller')
            
            # Augmented Dickey-Fuller (ADF) test results
            adf_result = adf_test(st.session_state['df_raw'], 1)        
            
            col1, col2, col3 = st.columns([18,40,10])
            col2.write(adf_result)
            
            vertical_spacer(2)
    
        ###################################################################  
        # SHAPIRO TEST
        ###################################################################
        def shapiro_test(df, variable_loc, alpha=0.05):
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
        
        with st.expander('Normality', expanded=False):
            my_text_header('Normality')
            my_text_paragraph('Shapiro')
            shapiro_result = shapiro_test(st.session_state['df_raw'], 1, alpha = 0.05)
            col1, col2, col3 = st.columns([18,40,10])
            col2.write(shapiro_result)
            vertical_spacer(2)
            
    my_bar.progress(80, text = 'loading autocorrelation')
    with tab5:    
        ###################################################################
        # AUTOCORRELATION PLOTS - Autocorrelation Plots (ACF & PACF) with optional Differencing applied
        ###################################################################
        with st.expander('Autocorrelation', expanded=True):     
            my_text_header('Autocorrelation')
            ############################## ACF & PACF ################################
            # Display the original or data differenced Plot based on the user's selection
            #my_text_paragraph(f'{selection}')
            st.markdown(f'<p style="text-align:center; color: #707070"> {selection} </p>', unsafe_allow_html=True)
            
            # get the differenced dataframe and original figure
            original_fig, df_select_diff = df_differencing(st.session_state.df_raw, selection, my_chart_color) 
            st.plotly_chart(original_fig, use_container_width=True)
        
            # set data equal to the second column e.g. expecting first column 'date' 
            data = df_select_diff
            # Plot ACF        
            plot_acf(data, nlags=nlags_acf, my_chart_color=my_chart_color)
            # Plot PACF
            plot_pacf(data, my_chart_color = my_chart_color, nlags=nlags_pacf, method=method_pacf)              
            # create 3 buttons, about ACF/PACF/Difference for more explanation on the ACF and PACF plots
            acf_pacf_info()
    
   
    # =============================================================================
    # PROGRESS BAR - LOAD COMPLETED    
    # =============================================================================
    my_bar.progress(100, text = '"All systems go!"')
    my_bar.empty()
# logging
print('ForecastGenie Print: Ran Explore')
                  
# =============================================================================
#    _____ _      ______          _   _ 
#   / ____| |    |  ____|   /\   | \ | |
#  | |    | |    | |__     /  \  |  \| |
#  | |    | |    |  __|   / /\ \ | . ` |
#  | |____| |____| |____ / ____ \| |\  |
#   \_____|______|______/_/    \_\_| \_|
#                                       
# =============================================================================
# CLEAN PAGE
if menu_item == 'Clean' and sidebar_menu_item=='Home':    
    # =============================================================================
    # Define Progress bar
    # =============================================================================
    progress_text = "Dusting off cleaning methods... Please wait!"
    my_bar = st.progress(0, text = progress_text)
        
    with st.sidebar:
        my_title(f"{clean_icon}", "#3b3b3b", gradient_colors="#440154, #2C2A6B, #FDE725")

        with st.form('data_cleaning'):
            
            my_text_paragraph('Handling Missing Data')      

            # get user input for filling method
            fill_method = st.selectbox(label = '*Select filling method for missing values:*', 
                                       options = ['Backfill', 'Forwardfill', 'Mean', 'Median', 'Mode', 'Custom'],
                                       key = key1_missing)
            
            if fill_method == 'Custom':
                custom_fill_value = st.text_input(label = '*Insert custom value to replace missing value(s) with:*', 
                                                  key = key2_missing, 
                                                  help = 'Please enter your **`custom value`** to impute missing values with, you can use a whole number or decimal point number')
            
            # Define a dictionary of possible frequencies and their corresponding offsets
            freq_dict = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}

            # infer and return the original data frequency e.g. 'M' and name e.g. 'Monthly'
            original_freq, original_freq_name = determine_df_frequency(st.session_state['df_raw'], 
                                                                       column_name='date')
            
            # determine the position of original frequency in the freq_dict to use as the index as chosen frequency in drop-down selectbox for user when loading page
            position = list(freq_dict.values()).index(original_freq)

            # Ask the user to select the frequency of the data
            # set the frequency that is inferred from the data itself with custom function to pre-selected e.g. index = ...
            freq = st.selectbox('*Select the frequency of the data:*', 
                                list(freq_dict.keys()), 
                                key = key3_missing)
            
            col1, col2, col3 = st.columns([4,4,4])
            with col2:       
                data_cleaning_btn = st.form_submit_button("Submit", type="secondary", on_click = form_update, args=("CLEAN_PAGE_MISSING",))

    # =========================================================================
    # 1. IMPUTE MISSING DATES WITH RESAMPLE METHOD
    # =========================================================================
    my_bar.progress(25, text = 'checking for missing dates...')
    
    # Apply function to resample missing dates based on user set frequency
    df_cleaned_dates = resample_missing_dates(df = st.session_state.df_raw, 
                                              freq_dict = freq_dict, 
                                              original_freq = original_freq, 
                                              freq = freq)
     
    
    # =========================================================================
    # 2. IMPUTE MISSING VALUES WITH FILL METHOD
    # =========================================================================
    my_bar.progress(50, text = 'checking for missing values...')
    
    df_clean = my_fill_method(df_cleaned_dates, fill_method, custom_fill_value)
    # Convert datetime column to date AND set date column as index column
    df_clean_show = copy_df_date_index(df_clean, datetime_to_date=True, date_to_index=True)
    #=========================================================================
    
    # create user tabs
    tab1_clean, tab2_clean = st.tabs(['missing data', 'outliers'])
    
    with tab1_clean:
    
        with st.expander('', expanded=True):
            my_text_header('Handling missing data')

            #show_lottie_animation(url = "./images/ufo.json", key='jumping_dots', width=300, height=300, speed = 1, col_sizes=[2,4,2])
            
            # check if there are no dates skipped for frequency e.g. daily data missing days in between dates
            missing_dates = pd.date_range(start = st.session_state.df_raw['date'].min(), 
                                          end = st.session_state.df_raw['date'].max()).difference(st.session_state.df_raw['date'])
            
            # check if there are no missing values (NaN) in dataframe
            missing_values = st.session_state.df_raw.iloc[:,1].isna().sum()
      
            # Plot missing values matrix with custom function
            plot_missing_values_matrix(df=df_cleaned_dates)
            
            # check if in continous time-series dataset no dates are missing in between
            if missing_dates.shape[0] == 0:
                st.success('Pweh 😅, no dates were skipped in your dataframe!')
            else:
                st.warning(f'💡 **{missing_dates.shape[0]}** dates were skipped in your dataframe, don\'t worry though! I will **fix** this by **imputing** the dates into your cleaned dataframe!')
            if missing_values != 0 and fill_method == 'Backfill':
                st.warning(f'💡 **{missing_values}** missing values are filled with the next available value in the dataset (i.e. backfill method), optionally you can change the *filling method* and press **\"Submit\"** from the sidebar menu.')
            elif missing_values != 0 and fill_method != 'Custom':
                st.warning(f'💡 **{missing_values}** missing values are replaced by the **{fill_method}**, optionally you can change the *filling method* and press **\"Submit\"** from the sidebar menu.')
            elif missing_values != 0 and fill_method == 'Custom':
                st.warning(f'💡 **{missing_values}** missing values are replaced by custom value **{custom_fill_value}**, optionally you can change the *filling method* and press **\"Submit\"** from the sidebar menu.')
    
            col1, col2, col3, col4, col5 = st.columns([2, 0.5, 2, 0.5, 2])
            with col1:
                df_graph = copy_df_date_index(my_df = st.session_state['df_raw'], 
                                              datetime_to_date = True, 
                                              date_to_index = True)
                
                highlighted_df = df_graph.style.highlight_null(color='yellow').format(precision = 2)
                
                my_subheader('Original DataFrame', my_style="#333333", my_size=6)
                
                st.dataframe(highlighted_df, use_container_width=True)
            
            with col2:
                st.markdown(arrow_right_icon, unsafe_allow_html=True)
            
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
                missing_df = copy_df_date_index(st.session_state.df_raw.loc[st.session_state.df_raw.iloc[:,1].isna(), st.session_state.df_raw.columns], datetime_to_date=True, date_to_index=True)
                st.write(missing_df)
            
            with col4:
                st.markdown(arrow_right_icon, unsafe_allow_html = True)
    
            with col5:
                my_subheader('Cleaned Dataframe', my_style="#333333", my_size=6)
                
                # Show the cleaned dataframe with if needed dates inserted if skipped to NaN and then the values inserted with impute method user selected backfill/forward fill/mean/median
                st.dataframe(df_clean_show, use_container_width=True)
            
            # Create and show download button in streamlit to user to download the dataframe with imputations performed to missing values
            download_csv_button(df_clean_show, my_file="df_imputed_missing_values.csv", set_index=True, help_message='Download cleaned dataframe to .CSV')
    
    my_bar.progress(75, text = 'loading outlier detection methods and cleaning methods')
    #########################################################
    # Handling Outliers
    #########################################################
    with st.sidebar:
        # call the function to get the outlier form in Streamlit 'Handling Outliers'
        outlier_form()
        
        # retrieve the parameters from default in-memory value or overwritten by user from streamlit widgets
        outlier_detection_method = get_state("CLEAN_PAGE", "outlier_detection_method")
        outlier_zscore_threshold = get_state("CLEAN_PAGE", "outlier_zscore_threshold")
        outlier_iqr_q1 = get_state("CLEAN_PAGE", "outlier_iqr_q1")
        outlier_iqr_q3 = get_state("CLEAN_PAGE", "outlier_iqr_q1")
        outlier_replacement_method = get_state("CLEAN_PAGE", "outlier_replacement_method")
        outlier_isolationforest_contamination = get_state("CLEAN_PAGE", "outlier_isolationforest_contamination")
        outlier_iqr_multiplier = get_state("CLEAN_PAGE", "outlier_iqr_multiplier")                    
                                                         
        # Create form and sliders for outlier detection and handling
        ##############################################################################       
        df_cleaned_outliers, outliers = handle_outliers(df_clean_show, 
                                                        outlier_detection_method,
                                                        outlier_zscore_threshold,
                                                        outlier_iqr_q1,
                                                        outlier_iqr_q3,
                                                        outlier_replacement_method,
                                                        outlier_isolationforest_contamination, 
                                                        random_state, 
                                                        outlier_iqr_multiplier)
        
        with tab2_clean:
            with st.expander('', expanded= True):
                # Set page subheader with custum function
                my_text_header('Handling outliers')
                my_text_paragraph('Outlier Plot')
                if outliers is not None and any(outliers):
                    
                    outliers_df = copy_df_date_index(df_clean[outliers], datetime_to_date=True, date_to_index=True).add_suffix('_outliers')
                    
                    df_cleaned_outliers = df_cleaned_outliers.add_suffix('_outliers_replaced')
                    
                    # inner join two dataframes of outliers and imputed values
                    outliers_df = outliers_df.join(df_cleaned_outliers, how='inner', rsuffix='_outliers_replaced')
                    
                    ## OUTLIER FIGURE CODE
                    fig_outliers = go.Figure()
                    fig_outliers.add_trace(
                                           go.Scatter(
                                                     x=df_clean['date'], 
                                                     y=df_clean.iloc[:,1], 
                                                     mode='markers', 
                                                     name='Before',
                                                     marker=dict(color='#440154'),  opacity = 0.5
                                                     )
                                           )
                    # add scatterplot
                    fig_outliers.add_trace(
                                           go.Scatter(
                                                      x = df_cleaned_outliers.index, 
                                                      y = df_cleaned_outliers.iloc[:,0], 
                                                      mode='markers', 
                                                      name='After',
                                                      marker=dict(color='#45B8AC'), opacity = 1
                                                     )
                                           )
                    
                    df_diff = df_cleaned_outliers.loc[outliers]
                    # add scatterplot
                    fig_outliers.add_trace(go.Scatter(
                                                      x=df_diff.index, 
                                                      y= df_diff.iloc[:,0], 
                                                      mode='markers', 
                                                      name='Outliers After',
                                                      marker=dict(color='#FFC300'),  opacity = 1
                                                     )
                                          )
        
                    # show the outlier plot
                    st.session_state['fig_outliers'] = fig_outliers
                    
                    st.plotly_chart(fig_outliers, use_container_width=True)
                    
                    #  show the dataframe of outliers
                    st.info(f'ℹ️ You replaced **{len(outliers_df)} outlier(s)** with their respective **{outlier_replacement_method}(s)** utilizing **{outlier_detection_method}**.')
                    
                    # Apply the color scheme to the dataframe, round values by 2 decimals and display it in streamlit using full size of expander window
                    st.dataframe(outliers_df.style.format("{:.2f}").apply(highlight_cols, axis=0), use_container_width=True)
                    
                    # add download button for user to be able to download outliers
                    download_csv_button(outliers_df, my_file="df_outliers.csv", set_index=True, help_message='Download outlier dataframe to .CSV', my_key='df_outliers')
               
                # if outliers are NOT found or None is selected as outlier detection method
                # ... run code... 
                # Show scatterplot data without outliers
                else:
                    vertical_spacer(1)
                    fig_no_outliers = go.Figure()
                    fig_no_outliers.add_trace(go.Scatter(x = df_clean['date'], 
                                             y = df_clean.iloc[:,1], 
                                             mode = 'markers', 
                                             name = 'Before',
                                  marker=dict(color = '#440154'),  opacity = 0.5))
                    
                    st.plotly_chart(fig_no_outliers, use_container_width=True)
                    my_text_paragraph(f'No <b> outlier detection </b> or <b> outlier replacement </b> method selected.', my_font_size='14px')
                    
    my_bar.progress(100, text = '100%')
    my_bar.empty()
else:
    ##################################################################################################################################
    #************************* PREPROCESSING DATA - CLEANING MISSING AND IF USER SELECTED IT ALSO IMPUTE OUTLIERS ********************
    ##################################################################################################################################
    # Retrieve the date frequency of the timeseries    
    freq_dict = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
    
    # Infer and return the original data frequency e.g. 'M' and name e.g. 'Monthly'
    original_freq, original_freq_name = determine_df_frequency(st.session_state['df_raw'], column_name='date')

    # =========================================================================
    # 1. IMPUTE MISSING DATES WITH RESAMPLE METHOD
    #******************************************************************
    # Apply function to resample missing dates based on user set frequency
    df_cleaned_dates = resample_missing_dates(df = st.session_state['df_raw'], 
                                              freq_dict = freq_dict, 
                                              original_freq = original_freq, 
                                              freq = st.session_state['freq'])

    # =========================================================================
    # 2. IMPUTE MISSING VALUES WITH FILL METHOD
    # =========================================================================
    df_clean = my_fill_method(df = df_cleaned_dates, 
                              fill_method = get_state('CLEAN_PAGE_MISSING', 'missing_fill_method'), 
                              custom_fill_value = get_state('CLEAN_PAGE_MISSING', 'missing_custom_fill_value'))
    
    df_clean_show = copy_df_date_index(df_clean, datetime_to_date=True, date_to_index=True)
    
    # =========================================================================
    # 3. IMPUTE OUTLIERS DETECTED WITH OUTLIER REPLACEMENT METHOD
    # =========================================================================
    df_cleaned_outliers, outliers = handle_outliers(data = df_clean_show, 
                                                    method = get_state('CLEAN_PAGE', 'outlier_detection_method'),
                                                    outlier_threshold = get_state('CLEAN_PAGE', 'outlier_zscore_threshold'),
                                                    q1 = get_state('CLEAN_PAGE', 'outlier_iqr_q1'),
                                                    q3 = get_state('CLEAN_PAGE', 'outlier_iqr_q3'),
                                                    outlier_replacement_method = get_state('CLEAN_PAGE', 'outlier_replacement_method'),
                                                    contamination = get_state('CLEAN_PAGE', 'outlier_isolationforest_contamination'), 
                                                    random_state = random_state,  # defined variable random_state top of script e.g. 10
                                                    iqr_multiplier = get_state('CLEAN_PAGE', 'outlier_iqr_multiplier'))
    
    # create a copy of the dataframe
    df_cleaned_outliers_with_index = df_cleaned_outliers.copy(deep=True)
    
    # reset the index again to have index instead of date column as index for further processing
    df_cleaned_outliers_with_index.reset_index(inplace=True)
    
    # convert 'date' column to datetime in DataFrame
    df_cleaned_outliers_with_index['date'] = pd.to_datetime(df_cleaned_outliers_with_index['date'])
    
    # =========================================================================
    # 4. SAVE TRANSFORMED (CLEANED) DATAFRAME TO SESSION STATE (IN-MEMORY)
    # =========================================================================
    # TEST replaced if statement with always updating the dataframe in session state e.g. if user changed from demo data to uploading data
    st.session_state['df_cleaned_outliers_with_index'] = df_cleaned_outliers_with_index
    
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
if menu_item == 'Engineer' and sidebar_menu_item == 'Home':
    # =============================================================================
    # Define Progress bar
    # =============================================================================
    progress_text = "Loading tabs... Please wait!"
    my_bar = st.progress(0, text = progress_text)
    
    tab1_engineer, tab2_engineer = st.tabs(['engineer', 'result'])
    
    with st.sidebar:
        my_title(f"{engineer_icon}", "#3b3b3b", gradient_colors="#1A2980, #FF6F61, #FEBD2E")
        
    with st.sidebar.form('feature engineering sidebar'):

        my_text_paragraph('Features')

        vertical_spacer(1)
        
        # show checkboxes in middle of sidebar to select all features or none
        col1, col2, col3 = st.columns([0.1,8,3])
        with col3:
            # create checkbox for all seasonal days e.g. dummy variables for day/month/year
            calendar_dummies_checkbox = st.checkbox(label = ' ', 
                                                    label_visibility='visible', 
                                                    key = key1_engineer,
                                                    help = 'Include independent features, namely create dummy variables for each `day` of the week, `month` and `year` whereby the leave-1-out principle is applied to not have `perfect multi-collinearity` i.e. the sum of the dummy variables for each observation will otherwise always be equal to one.')
            # create checkbox for country holidays
            calendar_holidays_checkbox = st.checkbox(label = ' ', 
                                                     label_visibility = 'visible', 
                                                     key = key2_engineer, 
                                                     help = 'Include **`official holidays`** of a specified country  \
                                                          \n**(default = USA)**')

            # create checkbox for all special calendar days
            special_calendar_days_checkbox = st.checkbox(label = ' ', 
                                                         label_visibility = 'visible', 
                                                         key = key3_engineer,
                                                         help = 'Include independent features including: **`pay-days`** and significant **`sales`** dates.')

            # create checkbox for Discrete Wavelet Transform features which automatically is checked
            dwt_features_checkbox = st.checkbox(label = ' ', 
                                                label_visibility = 'visible', 
                                                key = key4_engineer,
                                                help = 'In feature engineering, wavelet transform can be used to extract useful information from a time series by decomposing it into different frequency bands. This is done by applying a mathematical function called the wavelet function to the time series data. The resulting wavelet coefficients can then be used as features in machine learning models.')
        with col2:
            # show checkbox message if True/False for user options for Feature Selection
            st.write("*🌓 All Seasonal Periods*" if calendar_dummies_checkbox else "*🌓 No Seasonal Periods*")
            st.write("*⛱️ All Holiday Periods*" if calendar_holidays_checkbox else "*⛱️ No Holiday Periods*")
            st.write("*🎁 All Special Calendar Days*" if special_calendar_days_checkbox else "*🎁 No Special Calendar Days*")
            st.write("*🌊 All Wavelet Features*" if dwt_features_checkbox else "*🌊 No Wavelet Features*")
                
        # provide option for user in streamlit to adjust/set wavelet parameters
        with st.expander('🔽 Wavelet settings'):
            wavelet_family_selectbox = st.selectbox(label = '*Select Wavelet Family*', 
                                          options = ['db4', 'sym4', 'coif4'], 
                                          label_visibility = 'visible', 
                                          key = key5_engineer,
                                          help = ' A wavelet family is a set of wavelet functions that have different properties and characteristics.  \
                                          \n**`db4`** wavelet is commonly used for signals with *smooth variations* and *short-duration* pulses  \
                                          \n**`sym4`** wavelet is suited for signals with *sharp transitions* and *longer-duration* pulses.  \
                                          \n**`coif4`** wavelet, on the other hand, is often used for signals with *non-linear trends* and *abrupt* changes.  \
                                          \nIn general, the **`db4`** wavelet family is a good starting point, as it is a popular choice for a wide range of applications and has good overall performance.')
            
            # set standard level of decomposition to 3 
            wavelet_level_decomposition_selectbox = st.selectbox('*Select Level of Decomposition*', 
                                                       [1, 2, 3, 4, 5], 
                                                       label_visibility='visible', 
                                                       key = key6_engineer, 
                                                       help='The level of decomposition refers to the number of times the signal is decomposed recursively into its approximation coefficients and detail coefficients.  \
                                                             \nIn wavelet decomposition, the signal is first decomposed into two components: a approximation component and a detail component.\
                                                             The approximation component represents the coarsest level of detail in the signal, while the detail component represents the finer details.  \
                                                             \nAt each subsequent level of decomposition, the approximation component from the previous level is decomposed again into its own approximation and detail components.\
                                                             This process is repeated until the desired level of decomposition is reached.  \
                                                             \nEach level of decomposition captures different frequency bands and details in the signal, with higher levels of decomposition capturing finer and more subtle details.  \
                                                             However, higher levels of decomposition also require more computation and may introduce more noise or artifacts in the resulting representation of the signal.  \
                                                             \nThe choice of the level of decomposition depends on the specific application and the desired balance between accuracy and computational efficiency.')
            
            # add slider or text input to choose window size
            wavelet_window_size_slider = st.slider(label = '*Select Window Size (in days)*', 
                                                       label_visibility = 'visible',
                                                       min_value = 1, 
                                                       max_value = 30,
                                                       step = 1,
                                                       key = key7_engineer)
            
        col1, col2, col3 = st.columns([4,4,4])
        with col2:
            # add submit button to form, when user presses it it updates the selection criteria
            engineering_form_submit_btn = st.form_submit_button('Submit',  on_click=form_update, args=("ENGINEER_PAGE",))
            # if user presses the button on ENGINEER PAGE in sidebar
            if engineering_form_submit_btn:
                # update session state to True -> which is used to determine if default feature selection is chosen or user selection in SELECT PAGE
                set_state("ENGINEER_PAGE_FEATURES_BTN", ('engineering_form_submit_btn', True))
    
    with tab1_engineer:
        with st.expander("", expanded=True):
            
            show_lottie_animation(url="./images/aJ7Ra5vpQB.json", key="robot_engineering", width=350, height=350, speed=1, col_sizes= [1,3,1])
            
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
                year_dummies_checkbox = st.checkbox(label = 'Year', value = get_state("ENGINEER_PAGE_VARS", "year_dummies_checkbox"))
                set_state("ENGINEER_PAGE_VARS", ("year_dummies_checkbox", year_dummies_checkbox))
            with col2:
                month_dummies_checkbox = st.checkbox('Month', 
                                            value = get_state("ENGINEER_PAGE_VARS", "month_dummies_checkbox"))
                set_state("ENGINEER_PAGE_VARS", ("year_dummies_checkbox", month_dummies_checkbox))
            with col3:
                day_dummies_checkbox = st.checkbox('Day', 
                                          value = get_state("ENGINEER_PAGE_VARS", "day_dummies_checkbox"))
                set_state("ENGINEER_PAGE_VARS", ("year_dummies_checkbox", day_dummies_checkbox))
    
            ###############################################
            # create selectbox for country holidays
            ###############################################
            vertical_spacer(1)
            my_text_header('Holidays')
            my_text_paragraph('⛱️ Select country-specific holidays to include:')
            
            if calendar_holidays_checkbox:
                # apply function to create country specific holidays in columns is_holiday (boolean 1 if holiday otherwise 0) and holiday_desc for holiday_name
                df = create_calendar_holidays(df = st.session_state['df_cleaned_outliers_with_index'])
            
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
            
            col0, col1, col2, col3 = st.columns([6,12,12,1])
            with col1:               
                jan_sales = st.checkbox(label = 'January Sale', 
                                        value = get_state("ENGINEER_PAGE_VARS", "jan_sales"))
                
                set_state("ENGINEER_PAGE_VARS", ("jan_sales", jan_sales))
                                       
                val_day_lod = st.checkbox(label = "Valentine's Day [last order date]", 
                                          value = get_state("ENGINEER_PAGE_VARS", "val_day_lod"))
                
                set_state("ENGINEER_PAGE_VARS", ("val_day_lod", val_day_lod))
                
                val_day = st.checkbox(label = "Valentine's Day", 
                                      value = get_state("ENGINEER_PAGE_VARS", "val_day"))
                
                set_state("ENGINEER_PAGE_VARS", ("val_day", val_day))
                
                mother_day_lod = st.checkbox(label = "Mother's Day [last order date]", 
                                             value = get_state("ENGINEER_PAGE_VARS", "mother_day_lod"))
                
                set_state("ENGINEER_PAGE_VARS", ("mother_day_lod", mother_day_lod))
                
                mother_day = st.checkbox(label = "Mother's Day",
                                         value = get_state("ENGINEER_PAGE_VARS", "mother_day"))
                
                set_state("ENGINEER_PAGE_VARS", ("mother_day", mother_day))
                
                father_day_lod = st.checkbox(label = "Father's Day [last order date]", 
                                             value = get_state("ENGINEER_PAGE_VARS", "father_day_lod"))
                
                set_state("ENGINEER_PAGE_VARS", ("father_day_lod", father_day_lod))
                    
                pay_days = st.checkbox(label = 'Monthly Pay Days (4th Friday of month)', 
                                       value = get_state("ENGINEER_PAGE_VARS", "pay_days"))
                
                set_state("ENGINEER_PAGE_VARS", ("pay_days", pay_days))
           
            with col2:
                father_day = st.checkbox(label = "Father's Day", 
                                         value = get_state("ENGINEER_PAGE_VARS", "father_day"))
                
                set_state("ENGINEER_PAGE_VARS", ("father_day", father_day))
                
                black_friday_lod = st.checkbox(label = 'Black Friday [sale starts]', 
                                               value = get_state("ENGINEER_PAGE_VARS", "black_friday_lod"))
                
                set_state("ENGINEER_PAGE_VARS", ("black_friday_lod", black_friday_lod))
                
                black_friday = st.checkbox(label = 'Black Friday', 
                                          value = get_state("ENGINEER_PAGE_VARS", "black_friday"))
                
                set_state("ENGINEER_PAGE_VARS", ("black_friday", black_friday))
                
                cyber_monday = st.checkbox('Cyber Monday', 
                                           value = get_state("ENGINEER_PAGE_VARS", "cyber_monday"))
                
                set_state("ENGINEER_PAGE_VARS", ("cyber_monday", cyber_monday))
                
                christmas_day = st.checkbox(label = 'Christmas Day [last order date]', 
                                            value = get_state("ENGINEER_PAGE_VARS", "christmas_day"))
                
                set_state("ENGINEER_PAGE_VARS", ("christmas_day", christmas_day))
                
                boxing_day = st.checkbox(label = 'Boxing Day sale', 
                                         value = get_state("ENGINEER_PAGE_VARS", "boxing_day"))
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
                                      year_dummies = year_dummies_checkbox, 
                                      month_dummies = month_dummies_checkbox,
                                      day_dummies = day_dummies_checkbox)
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
                    data_in_window = df.iloc[i-window_size:i, 1].values
                    
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
                               [f'detail{i+1}_mean' for i in range(level)] + \
                               [f'detail{i+1}_std' for i in range(level)] + \
                               [f'detail{i+1}_max' for i in range(level)] + \
                               [f'detail{i+1}_min' for i in range(level)]
                
                # create a dataframe with the created features with discrete wavelet transform on target variable with timewindow set by user
                features_df_wavelet = pd.DataFrame(feature_vectors, columns=feature_cols, index=df.iloc[:,0].index[window_size:])
                
                # merge features dataframe with original data
                df = pd.merge(df, features_df_wavelet, left_index=True, right_index=True)
                #################################################################
                
                # PLOT WAVELET FEATURES
                #######################
                # create a dataframe again with the index set as the first column
                # assumption used: the 'date' column is the first column of the dataframe
                features_df_plot = pd.DataFrame(feature_vectors, columns=feature_cols, index=df.iloc[:,0])
                
                fig = px.line(features_df_plot, 
                              x=features_df_plot.index, 
                              y=['approx_mean'] + [f'detail{i+1}_mean' for i in range(level)],
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
    
    my_bar.progress(50, text = 'Combining Features into a presentable format...Please Wait!')
    #################################################################
    # ALL FEATURES COMBINED INTO A DATAFRAME
    #################################################################
    with tab2_engineer:
        with st.expander('', expanded=True):
            my_text_header('Engineered Features')
    
            # Retrieve number of features to show on page
            # -2 -> because of datetime index and target variable
            num_features_df = len(df.columns)-2
            my_text_paragraph(f'{num_features_df}')
            
            show_lottie_animation(url="./images/features_round.json", key="features_round", width=400, height=400)
            
            # Show dataframe in streamlit
            st.dataframe(copy_df_date_index(my_df = df, datetime_to_date = True, date_to_index = True), use_container_width = True)
            
            # add download button
            download_csv_button(df, my_file = "dataframe_incl_features.csv", help_message = "Download your dataset incl. features to .CSV")
    
    my_bar.progress(100, text = 'All systems go!')
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
        df = create_calendar_holidays(df = st.session_state['df_cleaned_outliers_with_index'], slider = False)
    
    # if feature engineering option is checked for special calendar days (e.g. black friday etc.), add features
    if special_calendar_days_checkbox:
        df = create_calendar_special_days(df)
    
    # if featue engineering option is checked for the year/month/day dummy variables, add features
    if calendar_dummies_checkbox:
        df = create_date_features(df, 
                                  year_dummies = get_state("ENGINEER_PAGE_VARS", "year_dummies_checkbox"), 
                                  month_dummies = get_state("ENGINEER_PAGE_VARS", "month_dummies_checkbox"), 
                                  day_dummies = get_state("ENGINEER_PAGE_VARS", "day_dummies_checkbox"))
    
    # =============================================================================
    #     # TEST OUTSIDE FUNCTION TO RUN WAVELET FUNCTION
    #     # df = <insert wavelet function code>
    #     if dwt_features_checkbox:
    #         df = 
    # =============================================================================    
    
    # Add the date column but only as numeric feature
    df['date_numeric'] = (df['date'] - df['date'].min()).dt.days
    # TIMESTAMP NUMERIC Unix gives same result
    #df['date_numeric'] = (df['date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    # =============================================================================
    # FIX DATA TYPES FOR ENGINEERED FEATURES
    # =============================================================================
    #columns_to_convert = {'holiday': 'uint8', 'calendar_event': 'uint8', 'pay_day': 'uint8', 'year': 'int32', 'is_holiday': 'uint8'}
    columns_to_convert = {'holiday': 'uint8', 'calendar_event': 'uint8', 'pay_day': 'uint8', 'year': 'str', 'is_holiday': 'uint8'}
    for column, data_type in columns_to_convert.items():
        if column in df:
            df[column] = df[column].astype(data_type)

    else:
        # cleaned df is unchanged e.g. no features added with feature engineering page
        #st.write('no features engineered') # TEST
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

if menu_item == 'Prepare' and sidebar_menu_item == 'Home':
    
    # define user tabs for the prepare page
    tab1_prepare, tab2_prepare, tab3_prepare, tab4_prepare = st.tabs(['preprocess', 'train/test split', 'normalization', 'standardization'])
    
    with st.sidebar:
        
        my_title(f'{prepare_icon}', "#3b3b3b", gradient_colors="#1A2980, #FF9F00, #FEBD2E")
       
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
                col1, col2, col3 = st.columns([130,60,120])
                with col2:        
                    # initiate placeholder
                    placeholder = st.empty()
                    
                    # create button (enabled to click e.g. disabled=false with unique key)
                    btn = placeholder.button('Show Data', disabled=False,  key = "preprocess_df_show_btn")
                
                # if button is clicked run below code
                if btn == True:
                    
                    # display button with text "click me again", with unique key
                    placeholder.button('Hide Data', disabled=False, key = "preprocess_df_hide_btn")
                    
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
            my_insample_forecast_steps, my_insample_forecast_perc = train_test_split_slider(df = st.session_state['df'])
            
            # update the session_state with train/test split chosen by user from sidebar slider
            # note: insample_forecast_steps is used in train/test split as variable
            st.session_state['insample_forecast_steps'] = my_insample_forecast_steps
            st.session_state['insample_forecast_perc'] = my_insample_forecast_perc
            
            # SHOW USER MESSAGE OF TRAIN/TEST SPLIT
            #######################################
            # SET VARIABLES
            length_df = len(st.session_state['df'])
            
            # format as new variables insample_forecast steps in days/as percentage e.g. the test set to predict for
            perc_test_set = "{:.2f}%".format((st.session_state['insample_forecast_steps']/length_df)*100)
            perc_train_set = "{:.2f}%".format(((length_df-st.session_state['insample_forecast_steps'])/length_df)*100)
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
            #st.warning(f"ℹ️ train/test split currently equals :green[**{perc_train_set}**] and :green[**{perc_test_set}**] ")
            
            my_text_paragraph('NOTE: a commonly used ratio is <b> 80:20 </b> split between the train- and test set.', my_font_size='12px', my_font_family='Arial') 
            
            vertical_spacer(2)
            
    ##############################
    # 5.2 Normalization
    ##############################
    with tab3_prepare:
        with st.sidebar:
            with st.form('normalization'):
                
                my_text_paragraph('Normalization')
                
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
                    normalization_choice = st.selectbox(label = "*Select normalization method:*", 
                                                        options = list(normalization_choices.keys()),
                                                        format_func = lambda x: f"{x} - {normalization_choices[x]}", 
                                                        key = key1_prepare_normalization,
                                                        help = '**`Normalization`** is a data pre-processing technique to transform the numerical data in a dataset to a standard scale or range.\
                                                                This process involves transforming the features of the dataset so that they have a common scale, which makes it easier for data scientists to analyze, compare, and draw meaningful insights from the data.')
                    # save user normalization choice in memory
                    st.session_state['normalization_choice'] = normalization_choice
                else:
                   # if no numerical features show user a message to inform
                   st.warning("No numerical features to normalize, you can try adding features!")
                   # set normalization_choice to None
                   normalization_choice = "None"
                
                # create form button centered on sidebar to submit user choice for normalization method
                col1, col2, col3 = st.columns([4,4,4])
                with col2:       
                    normalization_btn = st.form_submit_button("Submit", type="secondary", on_click = form_update, args=("PREPARE",))
            
            # apply function for normalizing the dataframe if user choice
            # IF user selected a normalization_choice other then "None" the X_train and X_test will be scaled
            X, y, X_train, X_test, y_train, y_test, scaler = perform_train_test_split(df = st.session_state['df'], 
                                                                                      my_insample_forecast_steps = st.session_state['insample_forecast_steps'], 
                                                                                      scaler_choice = st.session_state['normalization_choice'], 
                                                                                      numerical_features = numerical_features)
            
            
            
        # if user did not select normalization (yet) then show user message to select normalization method in sidebar
        if normalization_choice == "None":
            with st.expander('Normalization ',expanded=True):
                my_text_header('Normalization') 
                
                vertical_spacer(4)
                
                #show_lottie_animation(url="./images/monster-2.json", key="rocket_night_day", width=200, height=200, col_sizes=[6,6,6])
                st.image('./images/train_test_split.png')
                my_text_paragraph(f'Method: {normalization_choice}')
                
                vertical_spacer(12)
                
                st.info('👈 Please choose in the sidebar your normalization method for numerical columns. Note: columns with booleans will be excluded.')
    
        # else show user the dataframe with the features that were normalized
        else:
            with st.expander('Normalization',expanded=True):
               
                my_text_header('Normalized Features') 
                
                my_text_paragraph(f'Method: {normalization_choice}')
                
                # need original (unnormalized) X_train as well for figure in order to show before/after normalization
                X_unscaled_train = df.iloc[:, 1:].iloc[:-st.session_state['insample_forecast_steps'] , :]
                
                # with custom function create the normalization plot with numerical features i.e. before/after scaling
                plot_scaling_before_after(X_unscaled_train, X_train, numerical_features)
                
                st.success(f'🎉 Good job! **{len(numerical_features)}** numerical feature(s) are normalized with **{normalization_choice}**!')
                
                st.dataframe(X[numerical_features].assign(date=X.index.date).reset_index(drop=True).set_index('date'), use_container_width=True) # TEST
               
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
                    standardization_choice = st.selectbox(label = "*Select standardization method:*", 
                                                          options = list(standardization_choices.keys()), 
                                                          format_func = lambda x: f"{x} - {standardization_choices[x]}",
                                                          key = key2_prepare_standardization,
                                                          help = '**`Standardization`** is a preprocessing technique used to transform the numerical data to have zero mean and unit variance.\
                                                                  This is achieved by subtracting the mean from each value and then dividing by the standard deviation.\
                                                                  The resulting data will have a mean of zero and a standard deviation of one.\
                                                                  The distribution of the data is changed by centering and scaling the values, which can make the data more interpretable and easier to compare across different features' )
                else:
                   # if no numerical features show user a message to inform
                   st.warning("No numerical features to standardize, you can try adding features!")
                   
                   # set normalization_choice to None
                   standardization_choice = "None"
                
                # create form button centered on sidebar to submit user choice for standardization method   
                col1, col2, col3 = st.columns([4,4,4])
                with col2:       
                    standardization_btn = st.form_submit_button("Submit", type="secondary", on_click = form_update, args=("PREPARE",))
    
            # apply function for normalizing the dataframe if user choice
            # IF user selected a normalization_choice other then "None" the X_train and X_test will be scaled
            X, y, X_train, X_test, y_train, y_test = perform_train_test_split_standardization(X, y, X_train, X_test, y_train, y_test, 
                                                                                              st.session_state['insample_forecast_steps'], 
                                                                                              scaler_choice = standardization_choice, 
                                                                                              numerical_features = numerical_features)
            
        # if user did not select standardization (yet) then show user message to select normalization method in sidebar
        if standardization_choice == "None":
            # on page create expander
            with st.expander('Standardization ',expanded=True):
                
                my_text_header('Standardization') 
                my_text_paragraph(f'Method: {standardization_choice}')
                
                #show_lottie_animation(url="./images/2833-green-monster.json", key="green-monster", width=200, height=200, col_sizes=[6,6,6], speed=0.8)
                st.image('./images/standardization.png')
                st.info('👈 Please choose in the sidebar your Standardization method for numerical columns. Note: columns with booleans will be excluded.')
                
        # ELSE show user the dataframe with the features that were normalized
        else:
            with st.expander('Standardization',expanded=True):
                
                my_text_header('Standardization') 
                my_text_paragraph(f'Method: {standardization_choice}')
                
                # need original (unnormalized) X_train as well for figure in order to show before/after Standardization
                # TEST or do i need st.session_state['df'] instead of df? -> replaced df with st.session_state['df']
                X_unscaled_train = st.session_state['df'].iloc[:, 1:].iloc[:-st.session_state['insample_forecast_steps'], :]
                
                # with custom function create the Standardization plot with numerical features i.e. before/after scaling
                plot_scaling_before_after(X_unscaled_train, X_train, numerical_features)
               
                st.success(f'⚖️ Great, you balanced the scales! **{len(numerical_features)}** numerical feature(s) standardized with **{standardization_choice}**')
                
                st.dataframe(X[numerical_features], use_container_width=True)
                
                # create download button for user, to download the standardized features dataframe with dates as index i.e. first column
                download_csv_button(X[numerical_features], 
                                    my_file='standardized_features.csv', 
                                    help_message='Download standardized features to .CSV', 
                                    set_index=True, 
                                    my_key='standardization_download_btn')

else:
    # =============================================================================
    # - if user not in prepare screen then update the dataframe with preselected choices e.g. 80/20 split
    # - do not normalize and do not standardize
    # =============================================================================
    
    #################################
    # 1. REMOVE OBJECT DTYPE FEATURES
    #################################
    # remove dtype = object features from dataframe
    # these are/should be the descriptive columns like holiday description, day of week -> for which default dummy variables are created etc.
    st.session_state['df'] = remove_object_columns(st.session_state['df'], message_columns_removed = False)

    ##########################
    # 2. SET DATE AS INDEX COLUMN
    # 3. TRAIN/TEST SPLIT
    ##########################
    st.session_state['insample_forecast_steps'] = round((st.session_state['insample_forecast_perc'] / 100) * len(df))
    
    if 'date' in st.session_state['df']:
        #st.write('date is in the session state of dataframe df and will now be set as index') # TEST
        X, y, X_train, X_test, y_train, y_test, scaler = perform_train_test_split(st.session_state['df'].set_index('date'), 
                                                                                  st.session_state['insample_forecast_steps'], 
                                                                                  st.session_state['normalization_choice'], 
                                                                                  numerical_features=numerical_features)
    else:
        X, y, X_train, X_test, y_train, y_test, scaler = perform_train_test_split(st.session_state['df'], 
                                                                                  st.session_state['insample_forecast_steps'], 
                                                                                  st.session_state['normalization_choice'], 
                                                                                  numerical_features=numerical_features)
        
# =============================================================================
# Update Session States of dataset including: Target variable (y), independent features (X) for both train & test-set
# =============================================================================
st.session_state['X'] = X
st.session_state['y'] = y
st.session_state['X_train'] = X_train
st.session_state['X_test'] = X_test
st.session_state['y_train'] = y_train
st.session_state['y_test'] = y_test

# =============================================================================
# # If user changes engineered features to lower number than default e.g. 5, 
# # then check the number of features in X_train and also check if a user did not set slider to lower number than maximum of e.g. 3 features:
# =============================================================================
if st.session_state['X_train'].shape[1] < 5:
    st.write('number of features in X_train is smaller than default value of 5')
    
    # =============================================================================
    # PCA
    # =============================================================================
    # if user changed slider set it to that value
    if get_state("SELECT_PAGE_PCA", "num_features_pca") < st.session_state['X_train'].shape[1]:
        st.write('number of features is smaller than X_train for pca')
    else:    
        # PCA UPDATE SESSION STATE
        set_state("SELECT_PAGE_PCA", ("num_features_pca", min(5, st.session_state['X_train'].shape[1])))
    
    # =============================================================================
    # RFE   
    # =============================================================================
    if get_state("SELECT_PAGE_RFE", "num_features_rfe") < st.session_state['X_train'].shape[1]:    
        st.write('number of features is smaller than X_train for rfe')
    else:
        # RFE UPDATE SESSION STATE
        set_state("SELECT_PAGE_RFE", ("num_features_rfe", min(5, st.session_state['X_train'].shape[1])))
        # =============================================================================
        # set_state("SELECT_PAGE_RFE", ("estimator_rfe", "Linear Regression"))
        # set_state("SELECT_PAGE_RFE", ("timeseriessplit_value_rfe", 5))
        # set_state("SELECT_PAGE_RFE", ("num_steps_rfe", 1))
        # =============================================================================
    
    # =============================================================================
    # MIFS
    # =============================================================================
    if get_state("SELECT_PAGE_MIFS", "num_features_mifs") < st.session_state['X_train'].shape[1]:    
        st.write('number of features is smaller than X_train for mifs')
    else:
        # MIFS UPDATE SESSION STATE
        set_state("SELECT_PAGE_MIFS", ("num_features_mifs", min(5, st.session_state['X_train'].shape[1])))

# =============================================================================
# IF USER UPLOADS NEW FILE THEN RESET THE FEATURE SELECTION LIST TO EMPTY LIST
# =============================================================================
if get_state("DATA_OPTION", "upload_new_data") == True:
    # set equal to empty list again if user uploads new file
    set_state("SELECT_PAGE_USER_SELECTION", ("feature_selection_user", []))
    
# =============================================================================
#    _____ ______ _      ______ _____ _______ 
#   / ____|  ____| |    |  ____/ ____|__   __|
#  | (___ | |__  | |    | |__ | |       | |   
#   \___ \|  __| | |    |  __|| |       | |   
#   ____) | |____| |____| |___| |____   | |   
#  |_____/|______|______|______\_____|  |_|   
#                                             
# =============================================================================
# FEATURE SELECTION OF INDEPENDENT FEATURES / EXOGENOUS VARIABLES
if menu_item == 'Select' and sidebar_menu_item == 'Home':
    # =============================================================================
    # SHOW USER FORMS FOR PARAMETERS OF FEATURE SELECTION METHODS
    # =============================================================================
    with st.sidebar:
        num_features_rfe, duplicate_ranks_rfe, estimator_rfe, timeseriessplit_value_rfe, num_steps_rfe = form_feature_selection_method_rfe()
       
        num_features_pca = form_feature_selection_method_pca()
        
        num_features_mifs = form_feature_selection_method_mifs()
    
    # define user tabs to seperate SELECT PAGE sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["▫️info", "▫️rfe", "▫️pca", "▫️mi", "▫️pairwise correlation", "▫️feature selection"])
    
    # =============================================================================
    # SHOW INFORMATION CARD ABOUT FEATURE SELECTION METHODS
    # =============================================================================
    with tab1: 
        try:
            show_card_feature_selection_methods()
            
            st.image('./images/feature_info.png')
        except:
            pass
        
    # =============================================================================
    # RFE Feature Selection
    # =============================================================================
    with tab2:   
        try:
            with st.expander('', expanded=True):
                selected_cols_rfe, selected_features, rfecv, user_message = perform_rfe(X_train = st.session_state['X_train'], 
                                                                                        y_train = st.session_state['y_train'], 
                                                                                        duplicate_ranks_rfe = duplicate_ranks_rfe,
                                                                                        estimator_rfe = estimator_rfe, 
                                                                                        num_steps_rfe = num_steps_rfe, 
                                                                                        num_features_rfe = num_features_rfe, 
                                                                                        timeseriessplit_value_rfe = timeseriessplit_value_rfe)
                show_rfe_plot(rfecv, selected_features)
                # note the optimal features to user noted by RFE method
                st.write(user_message)
        except:
            selected_cols_rfe = []
            st.error('**ForecastGenie Error**: *Recursive Feature Elimination with Cross-Validation* could not execute. Need at least 2 features to be able to apply Feature Elimination. Please adjust your selection criteria.')

    # =============================================================================        
    # PCA Feature Selection
    # =============================================================================          
    with tab3: 
        try:
            with st.expander('', expanded=True):        
                sorted_features, pca, sorted_idx, selected_cols_pca = perform_pca(X_train = X_train, num_features_pca = num_features_pca)
                show_pca_plot(sorted_features = sorted_features, pca = pca, sorted_idx = sorted_idx, selected_cols_pca = selected_cols_pca)
        except:
            # IF PCA could not execute show user error
            selected_cols_pca = []
            st.error('**ForecastGenie Error**: *Principal Component Analysis* could not execute. Please adjust your selection criteria.')
    
    # =============================================================================
    # Mutual Information Feature Selection
    # =============================================================================
    with tab4:
        try:             
            with st.expander('', expanded=True):
                mutual_info, selected_features_mi, selected_cols_mifs = perform_mifs(X_train, y_train, num_features_mifs)
                show_mifs_plot(mutual_info = mutual_info, selected_features_mi = selected_features_mi, num_features_mifs = num_features_mifs)
        except: 
            selected_cols_mifs = []
            st.warning(':red[**ERROR**: Mutual Information Feature Selection could not execute...please adjust your selection criteria]')
    
    # =============================================================================
    # Removing Highly Correlated Independent Features
    # =============================================================================
    with tab5:
        try: 
            with st.sidebar:
                with st.form('correlation analysis'):
                    my_text_paragraph('Correlation Analysis')
                    
                    corr_threshold = st.slider("*Select Correlation Threshold*", 
                                               min_value = 0.0, 
                                               max_value = 1.0, 
                                               key = key1_select_page_corr,
                                               step = 0.05, 
                                               help = 'Set `Correlation Threshold` to determine which pair(s) of variables in the dataset are strongly correlated e.g. no correlation = 0, perfect correlation = 1')
    
                    # define models for computing importance scores of highly correlated independent features
                    models = {'Linear Regression': LinearRegression(), 'Random Forest Regressor': RandomForestRegressor(n_estimators=100)}
                    
                    # provide user option to select model (default = Linear Regression)
                    selected_corr_model = st.selectbox(label = '*Select **model** for computing **importance scores** for highly correlated feature pairs, to drop the **least important** feature of each pair which is highly correlated*:', 
                                                       options = list(models.keys()),
                                                       key = key2_select_page_corr)
                    
                    col1, col2, col3 = st.columns([4,4,4])
                    with col2:   
                        corr_btn = st.form_submit_button("Submit", type="secondary", on_click=form_update, args=('SELECT_PAGE_CORR',))
                        
                        # if button is pressed
                        if corr_btn:
                            set_state("SELECT_PAGE_BTN_CLICKED", ('correlation_btn', True))
                        
            with st.expander('', expanded=True):
                vertical_spacer(1)
                
                my_text_paragraph('Pairwise Correlation', my_font_size='26px')
                
                col1,col2,col3 = st.columns([5,3,5])
                with col2:
                    st.caption(f'with threshold >={corr_threshold*100:.0f}%')
                    
                ################################################################
                # PLOT HEATMAP WITH PAIRWISE CORRELATION OF INDEPENDENT FEATURES.
                ################################################################
                # Generate correlation heatmap for independent features based on threshold from slider set by user e.g. default to 0.8
                correlation_heatmap(X_train, correlation_threshold=corr_threshold)
                
                # Apply Function to analyse feature correlations and computes importance scores.             
                total_features, importance_scores, pairwise_features_in_total_features, df_pairwise = analyze_feature_correlations(selected_corr_model, 
                                                                                                                                   X_train, 
                                                                                                                                   y_train, 
                                                                                                                                   selected_cols_rfe, 
                                                                                                                                   selected_cols_pca, 
                                                                                                                                   selected_cols_mifs, 
                                                                                                                                   corr_threshold, 
                                                                                                                                   models)
                # =============================================================================
                # CHECK IF USER HAS SELECTED ANY FEATURES - IF ALL 3 SLIDERS ARE SET TO 0 FOR RFE, PCA, MIFS                                                                                                                               models)
                # =============================================================================
                if not total_features:   
                    st.info(f'**NOTE**: please adjust your selection criteria as the list of features is currently empty!')
                    st.stop()
                    
                # Display message with pairs in total_features
                if df_pairwise.empty:
                    st.info(f'There are no **pairwise combinations** in the selected features with a **correlation** larger than or equal to the user defined threshold of **{corr_threshold*100:.0f}%**')
                    vertical_spacer(1)
                else:
                    my_text_paragraph(f'The below pairwise combinations of features have a correlation >= {corr_threshold*100:.0f}% threshold:', my_font_size='16px')
                    vertical_spacer(1)
                    
                    # show dataframe with pairwise features
                    st.dataframe(df_pairwise, use_container_width=True)
                    
                    # download button for dataframe of pairwise correlation
                    download_csv_button(df_pairwise, my_file="pairwise_correlation.csv", help_message='Download the pairwise correlations to .CSV', set_index=False)
                    
                    # insert divider line
                    st.markdown('---')
                
                # SHOW CORRELATION CHART
                altair_correlation_chart(total_features, importance_scores, pairwise_features_in_total_features, corr_threshold)
                
                # Remove features with lowest importance scores from each pair
                # note: only remove one of highly correlated features based on importance score after plot is shown
                total_features_default = remove_lowest_importance_feature(total_features, importance_scores, pairwise_features_in_total_features)
                
        except:
            total_features_default = ['date_numeric']
            st.warning(':red[**ERROR**]: could not execute the correlation analysis...please adjust your selection criteria.')

    # =============================================================================
    # Top Features
    # =============================================================================
    with st.sidebar:        
        with st.form('top_features'):
            my_text_paragraph('Selected Features')
            
            #if rfe pca, mifs or correlation SUBMIT button is pressed, 
            # or if on ENGINEER PAGE button is pressed in sidebar (when e.g. features are removed or added)
            # set total_features equal again to recommended/calculated total_features from feature selection methods
            # if user changes one of the selection method parameters -> reset to feature selection method selection and remove user feature selection preference 
            if get_state("SELECT_PAGE_BTN_CLICKED", "rfe_btn") == True \
            or get_state("SELECT_PAGE_BTN_CLICKED", "pca_btn") == True \
            or get_state("SELECT_PAGE_BTN_CLICKED", "mifs_btn") == True \
            or get_state("SELECT_PAGE_BTN_CLICKED", "correlation_btn") == True \
            or get_state("ENGINEER_PAGE_FEATURES_BTN", "engineering_form_submit_btn") == True:
                
                st.write('option 1') #TEST1
                
                # reset state (back to False)
                set_state("SELECT_PAGE_BTN_CLICKED", ("rfe_btn", False)) 
                set_state("SELECT_PAGE_BTN_CLICKED", ("pca_btn", False))
                set_state("SELECT_PAGE_BTN_CLICKED", ("mifs_btn", False))
                set_state("SELECT_PAGE_BTN_CLICKED", ("correlation_btn", False))
                set_state("ENGINEER_PAGE_FEATURES_BTN", ("engineering_form_submit_btn", False))
                
                # deleting states set with fire-state package doesn't work need to use set_state to my knowledge
# =============================================================================
#                 # delete session states as it still otherwise evaluates to True
#                 del st.session_state["__SELECT_PAGE_BTN_CLICKED-rfe_btn__"]
#                 del st.session_state["__SELECT_PAGE_BTN_CLICKED-pca_btn__"]
#                 del st.session_state["__SELECT_PAGE_BTN_CLICKED-mifs_btn__"]
#                 del st.session_state["__SELECT_PAGE_BTN_CLICKED-correlation_btn__"]
# =============================================================================

# =============================================================================
#                 st.write('rfe_btn', get_state("SELECT_PAGE_BTN_CLICKED", "rfe_btn"))   # TEST1
#                 st.write('pca_btn', get_state("SELECT_PAGE_BTN_CLICKED", "pca_btn"))   # TEST1
#                 st.write('mifs_btn', get_state("SELECT_PAGE_BTN_CLICKED", "mifs_btn")) # TEST1
#                 st.write('correlation_btn', get_state("SELECT_PAGE_BTN_CLICKED", "correlation_btn")) # TEST1
# =============================================================================

                # set default feature selection
                total_features = total_features_default 
                
                # set features equal to default (based on 3 feature selection methods)
                set_state("SELECT_PAGE_USER_SELECTION", ("feature_selection_user", total_features))
            
            # if user selected features, use those! else use the recommended features by feature selection methods 
            elif "__SELECT_PAGE_USER_SELECTION-feature_selection_user__" in st.session_state and get_state("SELECT_PAGE_USER_SELECTION", "feature_selection_user") != []:
                st.write('option 2') # TEST
                total_features = get_state("SELECT_PAGE_USER_SELECTION", "feature_selection_user")
                
            else:
                st.write('option 3') # TEST
                # set feature selection to default based result of 3 feature selection methods)
                
                total_features = total_features_default 
                
                total_features = set_state("SELECT_PAGE_USER_SELECTION", ("feature_selection_user", total_features))

            
            feature_selection_user = st.multiselect(label = "favorite features", 
                                                    options = X_train.columns,
                                                    key = key1_select_page_user_selection,
                                                    label_visibility = "collapsed")

            col1, col2, col3 = st.columns([4,4,4])
            with col2:
                top_features_btn = st.form_submit_button("Submit", type="secondary",  on_click=form_update, args=("SELECT_PAGE_USER_SELECTION",))

    # IS THIS CODE REDUNDANT? I CAN DEFINE IT OUTSIDE OF SELECT PAGE WITH SESSION STATES               
    ######################################################################################################
    # Redefine dynamic user picked features for X, y, X_train, X_test, y_train, y_test
    ######################################################################################################
    # Apply columns user selected to X
    X = X.loc[:, feature_selection_user]
    y = y
    X_train = X[:(len(df)-st.session_state['insample_forecast_steps'])]
    X_test = X[(len(df)-st.session_state['insample_forecast_steps']):]

    # set endogenous variable train/test split
    y_train = y[:(len(df)-st.session_state['insample_forecast_steps'])]
    y_test = y[(len(df)-st.session_state['insample_forecast_steps']):]        
    
    with tab6:
        with st.expander('🥇 Top Features Selected', expanded=True):
            my_text_paragraph('Your Feature Selection', my_font_size='26px')
            
            
# =============================================================================
#             show_lottie_animation(url = "./images/astronaut_star_in_hand.json", 
#                                   key = 'lottie_animation_austronaut_star', 
#                                   width=200, 
#                                   height=200, 
#                                   col_sizes=[5,4,5], 
#                                   margin_before=1, 
#                                   margin_after=2)
# =============================================================================
            
            # TEST - show recommended features from 3 selection methods 
            #st.write(selected_cols_rfe, selected_cols_pca, selected_cols_mifs) # TEST1
            
            df_total_features = pd.DataFrame(feature_selection_user, columns = ['Top Features'])
            
            # Create the rating column and update it based on the presence in the lists
            df_total_features['rating'] = df_total_features['Top Features'].apply(lambda feature: '⭐' * (int(feature in selected_cols_rfe) + int(feature in selected_cols_pca) + int(feature in selected_cols_mifs)) if any(feature in lst for lst in [selected_cols_rfe, selected_cols_pca, selected_cols_mifs]) else '-')

            # Create boolean columns 'rfe', 'pca', and 'mi' based on the presence of features in each list
            df_total_features['rfe'] = df_total_features['Top Features'].apply(lambda feature: feature in selected_cols_rfe)
            df_total_features['pca'] = df_total_features['Top Features'].apply(lambda feature: feature in selected_cols_pca)
            df_total_features['mi'] = df_total_features['Top Features'].apply(lambda feature: feature in selected_cols_mifs)

            # =============================================================================      
            # THIS CODE RETRIEVES DIFFERENCE WHAT WAS RECOMMENDED BY FEATURE SELECTION METHODS
            # AND WHAT USER ADDS AS ADDITIONAL VARIABLES OUTSIDE OF RECOMMENDATION
            # Note: first time app starts the total_features is None
            # =============================================================================
            if total_features is not None:
                difference_lsts = list(set(feature_selection_user) - set(total_features_default))
            else:
                # if no features chosen by user then return empty list -> no cells need to be highlighted yellow 
                difference_lsts = []
            
            def highlight_cols_feature_selection(row):
                """
                A function that highlights the cells of a DataFrame based on their values.
                
                Args:
                row: A pandas Series representing a row of the DataFrame.
                
                Returns:
                list: A list of CSS styles to be applied to the row cells.
                """
                styles = ['background-color: #f0f6ff' if row['Top Features'] in difference_lsts else '' for _ in row]
                return styles
            
            # Apply the custom function to highlight rows that have features in difference_lsts, including the 'rating' column
            styled_df = df_total_features.style.apply(highlight_cols_feature_selection, axis=1)

            # show dataframe with top features + user selection in Streamlit
            st.dataframe(styled_df, use_container_width=True)
            
            ###
            
            # Display the dataframe with independent features (X) and values in Streamlit
            st.dataframe(X, use_container_width=True)
    
            # Create download button for forecast results to .CSV
            download_csv_button(X, my_file = "features_dataframe.csv", 
                                help_message = "Download your **features** to .CSV", 
                                my_key = 'features_df_download_btn')
            
        st.image('./images/feature_selection2.png')
    # Set session state variable 'upload_new_data' back to False
    set_state("DATA_OPTION", ("upload_new_data", False))
else:
    # =============================================================================
    # ELSE WHEN USER IS NOT ON SELECT PAGE:
    # 0. RUN THE FEATURE SELECTION TO SELECT TOP FEATURES
    # 1. REMOVE FROM HIGHLY CORRELATED INDEPENDENT FEATURES PAIRS THE ONE WITH LOWEST IMPORTANCE SCORES
    # =============================================================================
    
    # =============================================================================
    # Apply 3 Feature Selection Techniques
    # =============================================================================
    ################
    # 1. RFE
    ################   
    try:
        selected_cols_rfe, selected_features, rfecv, message = perform_rfe(X_train = st.session_state['X_train'], 
                                                                  y_train = st.session_state['y_train'], 
                                                                  estimator_rfe = get_state("SELECT_PAGE_RFE", "estimator_rfe"), 
                                                                  duplicate_ranks_rfe = get_state("SELECT_PAGE_RFE", "duplicate_ranks_rfe"),
                                                                  num_steps_rfe = get_state("SELECT_PAGE_RFE", "num_steps_rfe"),
                                                                  num_features_rfe = get_state("SELECT_PAGE_RFE", "num_features_rfe"), 
                                                                  timeseriessplit_value_rfe = get_state("SELECT_PAGE_RFE", "timeseriessplit_value_rfe"))
    except:
        selected_cols_rfe = []
        
    ################
    # 2. PCA
    ################
    try:
        sorted_features, pca, sorted_idx, selected_cols_pca = perform_pca(X_train = st.session_state['X_train'], 
                                                                          num_features_pca = get_state("SELECT_PAGE_PCA", "num_features_pca"))
    except:
        selected_cols_pca = []
    ################
    # 3. MIFS
    ################
    try:  
        mutual_info, selected_features_mi, selected_cols_mifs = perform_mifs(X_train = st.session_state['X_train'], 
                                                                             y_train = st.session_state['y_train'], 
                                                                             num_features_mifs = get_state("SELECT_PAGE_MIFS", "num_features_mifs"))
    except: 
        selected_cols_mifs = []

    # =============================================================================
    # Remove Highly Correlated Features >= threshold ( default = 80% pairwise-correlation)
    # remove one of each pair based on importance scores
    # =============================================================================            
    total_features, importance_scores, pairwise_features_in_total_features, df_pairwise = analyze_feature_correlations(selected_corr_model = get_state("SELECT_PAGE_CORR", "selected_corr_model"), 
                                                                                                                        X_train = X_train, 
                                                                                                                        y_train = y_train, 
                                                                                                                        selected_cols_rfe = selected_cols_rfe, 
                                                                                                                        selected_cols_pca = selected_cols_pca, 
                                                                                                                        selected_cols_mifs = selected_cols_mifs, 
                                                                                                                        models = {'Linear Regression': LinearRegression(), 
                                                                                                                                  'Random Forest Regressor': RandomForestRegressor(n_estimators=100)},
                                                                                                                        corr_threshold = get_state("SELECT_PAGE_CORR", "corr_threshold"))
    # NOTE: only remove one of highly correlated features based on importance score after plot is shown
    total_features_default = remove_lowest_importance_feature(total_features, importance_scores, pairwise_features_in_total_features)      
    
    # if user selected features, use those! else use the recommended features by feature selection methods 
    if "__SELECT_PAGE_USER_SELECTION-feature_selection_user__" in st.session_state and get_state("SELECT_PAGE_USER_SELECTION", "feature_selection_user") != []:
        total_features = get_state("SELECT_PAGE_USER_SELECTION", "feature_selection_user")
    else:
        total_features = total_features_default 
        
    X = X.loc[:, total_features]
    y = y
    X_train = X[:(len(df)-st.session_state['insample_forecast_steps'])]
    X_test = X[(len(df)-st.session_state['insample_forecast_steps']):]
    y_train = y[:(len(df)-st.session_state['insample_forecast_steps'])]
    y_test = y[(len(df)-st.session_state['insample_forecast_steps']):]    
    
# =============================================================================
#   _______ _____            _____ _   _ 
#  |__   __|  __ \     /\   |_   _| \ | |
#     | |  | |__) |   /  \    | | |  \| |
#     | |  |  _  /   / /\ \   | | | . ` |
#     | |  | | \ \  / ____ \ _| |_| |\  |
#     |_|  |_|  \_\/_/    \_\_____|_| \_|
#                                        
# =============================================================================
# TRAIN MODELS
if menu_item == 'Train' and sidebar_menu_item == 'Home':
    # =============================================================================
    # USER FORM: TRAIN MODELS (MODEL CHECKBOXES, PARAMETERS AND HYPERPARAMETERS)
    # =============================================================================
    with st.sidebar:
        my_title(f"{train_icon}", "#3b3b3b")
        with st.form('model_train_form'):
        
            vertical_spacer(1)
           
            my_text_paragraph('Settings')
            vertical_spacer(2)
            
            col1, col2, col3, col4, col5 = st.columns([1,7,1,7,1])
            with col2:    
                include_feature_selection = st.selectbox(label = '*include feature selection:*', 
                                                         options = ['Yes', 'No'],
                                                         key = key28_train,
                                                         help = '''if `feature selection` is enabled, the explanatory variables or features will be incorporated as additional inputs during the model training process. These features can be either the default recommendations based on the feature selection process (refer to the Select Page for more details), or if you have made changes to the selection, it will reflect the features you have chosen to include.  
                                                                   \nBy enabling feature selection, you allow the model to utilize these explanatory features with the goal to enhance its predictive capabilities to improve the overall performance of the model(s).''')
            with col4:
                # Generic Graph Settings
                my_conf_interval = st.slider(label = "*confidence interval (%)*", 
                                             min_value = 1, 
                                             max_value = 99,
                                             step = 1,
                                             key = key1_train,
                                             help = 'a **`confidence interval`** is a range of values around a sample statistic, such as a mean or proportion, which is likely to contain the true population parameter with a certain degree of confidence.\
                                                    The level of confidence is typically expressed as a percentage, such as 95%, and represents the probability that the true parameter lies within the interval.\
                                                    A wider interval will generally have a higher level of confidence, while a narrower interval will have a lower level of confidence.')
            vertical_spacer(2)
            
            my_text_paragraph('Model Selection')
            
            # *****************************************************************************
            # 1. NAIVE MODELS         
            # *****************************************************************************
            naive_model_checkbox = st.checkbox(label = 'Naive Models', 
                                               key = key2_train)
            
            # *****************************************************************************
            # NAIVE MODELS PARAMETERS        
            # *****************************************************************************  
            custom_lag_value = None  # initiate custom lag value
            
            with st.expander('◾'):
               
                # =============================================================================
                # 1.1 NAIVE MODEL I: LAG / CUSTOM LAG FOR NAIVE MODEL                
                # ============================================================================
                my_text_paragraph('Naive Model I: Lag')
                
                col1, col2, col3 = st.columns([1,2,1])
                with col2: 
                    lag = st.selectbox(label = '*select seasonal lag*', 
                                       options = ['Day', 'Week', 'Month', 'Year', 'Custom'],
                                       key = key7_train,
                                       label_visibility = 'visible')
                    
                    lag = lag.lower() # make lag name lowercase e.g. 'Week' becomes 'week'
                    
                    if lag == 'custom':
                        custom_lag_value = st.number_input(label = "*if seasonal **lag** set to Custom, please set lag value (in days):*", 
                                                           step = 1,
                                                           key = key8_train)
                        
                        custom_lag_value = custom_lag_value if custom_lag_value is not None else None
                        
                # =============================================================================
                # 1.2 NAIVE MODEL II: ROLLING WINDOW                    
                # =============================================================================
                vertical_spacer(1)
                my_text_paragraph('Naive Model II: Rolling Window')
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    
                    size_rolling_window_naive_model = st.number_input(label = '*select rolling window size:*',
                                                                 min_value=1,
                                                                 max_value = None,
                                                                 step = 1,
                                                                 format = '%i',
                                                                 key = key29_train)
                    
                    agg_method_rolling_window_naive_model = st.selectbox(label = '*select aggregation method:*', 
                                                                     options = ['Mean', 'Median', 'Mode'],
                                                                     key = key30_train)
                    
                # =============================================================================
                # 1.3 NAIVE MODEL III: Straight Line (Mean, Median, Mode)               
                # =============================================================================
                vertical_spacer(1)
                my_text_paragraph('Naive Model III: Constant Value')
                
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    agg_method_baseline_naive_model = st.selectbox(label = '*select aggregation method:*', 
                                                                   options = ['Mean', 'Median', 'Mode'],
                                                                   key = key31_train)
                vertical_spacer(1)
                
            # *****************************************************************************
            # 2. LINEAR REGRESSION MODEL        
            # *****************************************************************************
            linreg_checkbox = st.checkbox('Linear Regression', 
                                          key = key3_train)
            
            # *****************************************************************************
            # 3. SARIMAX MODEL        
            # *****************************************************************************
            sarimax_checkbox = st.checkbox('SARIMAX', 
                                           key = key4_train)
            
            # *****************************************************************************
            # 3.1 SARIMAX MODEL PARAMETERS         
            # *****************************************************************************
            with st.expander('◾', expanded=False):
                
                col1, col2, col3 = st.columns([5,1,5])
                with col1:
                    p = st.number_input(label = "Autoregressive Order (p):", 
                                        min_value = 0, 
                                        max_value = 10,
                                        key = key9_train)
                    
                    d = st.number_input(label = "Differencing (d):", 
                                        min_value = 0, 
                                        max_value = 10,
                                        key = key10_train)
                    
                    q = st.number_input(label = "Moving Average (q):", 
                                        min_value = 0, 
                                        max_value = 10,
                                        key = key11_train)   
                with col3:
                    P = st.number_input(label = "Seasonal Autoregressive Order (P):", 
                                        min_value = 0, 
                                        max_value = 10,
                                        key = key12_train)
                    
                    D = st.number_input(label = "Seasonal Differencing (D):", 
                                        min_value = 0, 
                                        max_value = 10,
                                        key = key13_train)
                    
                    Q = st.number_input(label = "Seasonal Moving Average (Q):", 
                                        #value=1, 
                                        min_value = 0, 
                                        max_value = 10,
                                        key = key14_train)
                    
                    s = st.number_input(label = "Seasonal Periodicity (s):", 
                                        min_value = 1, 
                                        key = key15_train,
                                        help = '`seasonal periodicity` i.e. **$s$** in **SARIMAX** refers to the **number of observations per season**.\
                                               \n\nfor example, if we have daily data with a weekly seasonal pattern, **s** would be 7 because there are 7 days in a week.\
                                               similarly, for monthly data with an annual seasonal pattern, **s** would be 12 because there are 12 months in a year.\
                                               here are some common values for **$s$**:\
                                               \n- **Daily data** with **weekly** seasonality: **$s=7$** \
                                               \n- **Monthly data** with **quarterly** seasonality: **$s=3$**\
                                               \n- **Monthly data** with **yearly** seasonality: **$s=12$**\
                                               \n- **Quarterly data** with **yearly** seasonality: **$s=4$**')
    
                col1, col2, col3 = st.columns([5,1,5])
                with col1:
                    # Add a selectbox for selecting enforce_stationarity
                    enforce_stationarity = st.selectbox(label = 'Enforce Stationarity', 
                                                        options = [True, False], 
                                                        key =  key16_train)
                with col3:
                    # Add a selectbox for selecting enforce_invertibility
                    enforce_invertibility = st.selectbox(label = 'Enforce Invertibility', 
                                                         options = [True, False], 
                                                         key = key17_train)
            
            # *****************************************************************************
            # 4 PROPHET MODEL
            # *****************************************************************************
            prophet_checkbox = st.checkbox('Prophet', 
                                           key = key5_train)    
            
            # *****************************************************************************
            # 4.1 PROPHET MODEL PARAMETERS
            # *****************************************************************************
            with st.expander('◾', expanded = False):
                # used to have int(st.slider) -> not needed? # TEST
                horizon_option = st.slider(label = 'Set Forecast Horizon (default = 30 Days):', 
                                           min_value = 1, 
                                           max_value = 365, 
                                           step = 1, 
                                           #value = 30,
                                           key = key18_train,
                                           help = 'The horizon for a Prophet model is typically set to the number of time periods that you want to forecast into the future. This is also known as the forecasting horizon or prediction horizon.')
                        
                changepoint_prior_scale = st.slider(label = "changepoint_prior_scale", 
                                                    min_value = 0.001, 
                                                    max_value = 1.0, 
                                                    #value = 0.05, 
                                                    step = 0.01, 
                                                    key = key19_train,
                                                    help = 'This is probably the most impactful parameter. It determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints. As described in this documentation, if it is too small, the trend will be underfit and variance that should have been modeled with trend changes will instead end up being handled with the noise term. If it is too large, the trend will overfit and in the most extreme case you can end up with the trend capturing yearly seasonality. The default of 0.05 works for many time series, but this could be tuned; a range of [0.001, 0.5] would likely be about right. Parameters like this (regularization penalties; this is effectively a lasso penalty) are often tuned on a log scale.')
                
                # used to be str(selectbox) -> not needed? # TEST
                seasonality_mode = st.selectbox(label = "seasonality_mode", 
                                                options = ["additive", "multiplicative"], 
                                                #index = 1,
                                                key = key20_train)
                
                seasonality_prior_scale = st.slider(label = "seasonality_prior_scale", 
                                                    min_value = 0.010, 
                                                    max_value = 10.0, 
                                                    #value = 1.0, 
                                                    step = 0.1,
                                                    key = key21_train)
                
                # retrieve from the session state the country code to display to user what country holidays are currently added to Prophet Model if set to TRUE
                country_holidays = st.selectbox(label = f'country_holidays ({get_state("ENGINEER_PAGE_COUNTRY_HOLIDAY", "country_code")})', 
                                                options = [True, False],
                                                key = key27_train,
                                                help = """Setting this to `True` adds Country Specific Holidays - whereby Prophet assigns each holiday a specific weight""")
                
                holidays_prior_scale = st.slider(label = "holidays_prior_scale", 
                                                 min_value = 0.010, 
                                                 max_value = 10.0, 
                                                 #value = 1.0, 
                                                 step = 0.1,
                                                 key = key22_train)
                
                yearly_seasonality = st.selectbox(label = "yearly_seasonality", 
                                                  options = [True, False], 
                                                  #index = 0,
                                                  key = key23_train)
                
                weekly_seasonality = st.selectbox(label = "weekly_seasonality", 
                                                  options = [True, False], 
                                                  #index = 0,
                                                  key = key24_train)
                
                daily_seasonality = st.selectbox(label = "daily_seasonality", 
                                                 options = [True, False], 
                                                 #index = 0,
                                                 key = key25_train)
    
                interval_width = int(my_conf_interval/100)
            
    
            col1, col2, col3 = st.columns([4,4,4])
            with col2: 
                # submit button of user form for training models and updating session states on click utilizing custom package fire-state
                train_models_btn = st.form_submit_button(label = "Submit", 
                                                         type = "secondary", 
                                                         on_click = form_update, 
                                                         args=('TRAIN_PAGE',)
                                                         )
                # update session_state 
                st.session_state['train_models_btn'] = train_models_btn

        # *****************************************************************************
        # ALL MODELS
        # *****************************************************************************
        # Define all models you want user to choose from (<model_name>, <actual model to be called when fitting the model>)
        models = [('Naive Models', None),
                  ('Linear Regression', LinearRegression(fit_intercept=True)), 
                  ('SARIMAX', SARIMAX(y_train)),
                  ('Prophet', Prophet())]

        # initiate empty list to hold user's checked model names and models from selectboxes
        selected_models = []
        
        # if user pressed submit button train the models else do not e.g. every time you go back to train page dont automatically kick off the training
        if st.session_state['train_models_btn']:
            
            for model_name, model in models:
    
                # Add model name and model if checkbox of model is checked into a list
                if model_name == 'Naive Models' and naive_model_checkbox == True:
                   selected_models.append((model_name, model))
                   
                # Add model name and model if checkbox of model is checked into a list
                if model_name == 'Linear Regression' and linreg_checkbox == True:
                    selected_models.append((model_name, model))
                    
                # add model name and model if checkbox of model is checked into a list
                if model_name == 'SARIMAX' and sarimax_checkbox == True:
                    selected_models.append((model_name, model))
                    
                # add model name and model if checkbox of model is checked into a list
                if model_name == 'Prophet' and prophet_checkbox == True:
                    selected_models.append((model_name, model))   
    
    # add the user selected trained models to the session state
    set_state("TRAIN_PAGE", ("selected_models", selected_models))
     
    # =============================================================================
    # CREATE TABS    
    # =============================================================================
    tab1, tab2, tab3, tab4 = st.tabs(['▫️info', '▫️inputs', '▫️outputs', '▫️model details'])

    with tab1:
        with st.expander('', expanded=True):
            col1, col2, col3 = st.columns([1,3,1])
            with col2:
                train_models_carousel(my_title= 'Select your models to train!')
                
        st.image('./images/train_page_info.png')
        
    with tab2:
            with st.expander('', expanded=True):
                # Show buttons with Training Data/Test Data 
                show_model_inputs()
                
    with tab3:
        # if no models are selected by user show message
        if selected_models == []:
            with st.expander('', expanded = True):
               my_text_header('Model Outputs')

               col1, col2, col3 = st.columns([7,120,1])
               with col2:
                   
                   create_flipcard_model_input(image_path_front_card = './images/train_info.png', 
                                               my_string_back_card = 'If you train your models, the output will show up here! You can use the checkboxes from the sidebar menu to select your model(s) and hit <i style="color:#9d625e;">\"Submit\"</i> to check out your test results!')
                       
               vertical_spacer(4)

        # if models are selected by user run code
        if selected_models:
            
            # iterate over all models and if user selected checkbox for model the model(s) is/are trained
            for model_name, model in selected_models:
                
                # =============================================================================
                # FOR RESULTS DATAFRAME - CHECK WHICH FEATURES ARE USED
                # IF NO FEATURES ARE USED THE DATE AS NUMERIC VALUES (INTEGERS 1,2,3 ... N are used)
                # =============================================================================
                # create a list of independent variables selected by user prior used 
                # for results dataframe when evaluating models which variables were included.
                if get_state("TRAIN_PAGE", "include_feature_selection") == "Yes":
                    features_str = get_feature_list(X)
                elif get_state("TRAIN_PAGE", "include_feature_selection") == "No":
                    features_str = 'date_numeric'

                # =============================================================================
                #  NAIVE MODELS                   
                # =============================================================================
                try:         
                    if model_name == "Naive Models":
                        with st.expander('📈' + model_name, expanded=True):
                            # =============================================================================
                            # Naive Model I
                            # =============================================================================
                            # set model name
                            model_name = 'Naive Model I'
                            
                            df_preds_naive_model1 = forecast_naive_model1_insample(y_test, 
                                                                                   lag = lag, 
                                                                                   custom_lag_value = custom_lag_value)
                            
                            # define for subtitle dictionary to tell user what lag frequency was chosen
                            lag_word = {'day': 'daily', 'week': 'weekly', 'month': 'monthly', 'year': 'yearly'}.get(lag)
                            
                            # Show metrics on model card
                            display_my_metrics(df_preds_naive_model1, 
                                               model_name = "Naive Model I",
                                               my_subtitle= f'custom lag: {custom_lag_value}' if custom_lag_value is not None else f'{lag_word} lag')
                           
                            # Plot graph with actual versus insample predictions
                            plot_actual_vs_predicted(df_preds_naive_model1, my_conf_interval)
                               
                            # Show the dataframe
                            st.dataframe(df_preds_naive_model1.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Error': '{:.2f}', 'Error (%)': '{:.2%}'}), use_container_width=True)
                            
                            # Create download button for forecast results to .csv
                            download_csv_button(df_preds_naive_model1, my_file="insample_forecast_naivemodel_results.csv", 
                                                 help_message="Download your Naive Model I results to .CSV",
                                                 my_key = 'naive_trained_modeli_download_btn', 
                                                 set_index = True)
                            
                            # =============================================================================
                            # SAVE TEST RESULTS FOR EVALUATION PAGE BY ADDING ROW TO RESULTS_DF                
                            # =============================================================================
                            metrics_dict = my_metrics(df_preds_naive_model1, model_name = "Naive Model I")
                                                        
                            # Retrieve the feature for Naive Model for results_df 'features' column
                            lag_int = {'day': '1', 'week': '7', 'month': '30', 'year': '-365'}.get(lag)
                          
                            naive_model1_feature_str = f"t-{custom_lag_value}" if custom_lag_value else f"yt-{lag_int}"
                            
                            # Create new row with test result details
                            new_row = {'model_name': model_name,
                                       'mape': '{:.2%}'.format(metrics_dict[model_name]['mape']),
                                       'smape': '{:.2%}'.format(metrics_dict[model_name]['smape']),
                                       'rmse': round(metrics_dict[model_name]['rmse'], 2),
                                       'r2': round(metrics_dict[model_name]['r2'], 2),
                                       'features': naive_model1_feature_str,
                                       'model_settings': f"lag: {lag}",
                                       'predicted':  np.ravel(df_preds_naive_model1['Predicted'])
                                      }
                            
                            # Turn new row into a dataframe
                            new_row_df = pd.DataFrame([new_row])
                            
                            # Concatenate new row DataFrame with results_df
                            results_df = pd.concat([results_df, new_row_df], ignore_index=True)
                            
                            # update session state with latest results with new row added
                            set_state("TRAIN_PAGE", ("results_df", results_df))
                            
                            # =============================================================================
                            # Naive Model II
                            # =============================================================================
                            # set model name
                            model_name = 'Naive Model II'
                            
                            st.markdown('---')
                            
                            df_preds_naive_model2 = forecast_naive_model2_insample(y_test, 
                                                                                   size_rolling_window_naive_model, 
                                                                                   agg_method_rolling_window_naive_model)
                            # Show metrics on model card
                            display_my_metrics(df_preds_naive_model2, 
                                               model_name = 'Naive Model II', 
                                               my_subtitle = f'{agg_method_rolling_window_naive_model.lower()} rolling window of {size_rolling_window_naive_model}')
                           
                            # Plot graph with actual versus insample predictions
                            plot_actual_vs_predicted(df_preds_naive_model2, my_conf_interval)
                               
                            # Show the dataframe
                            st.dataframe(df_preds_naive_model2.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Error': '{:.2f}', 'Error (%)': '{:.2%}'}), use_container_width=True)
                          
                            # Create download button for forecast results to .csv
                            download_csv_button(df_preds_naive_model2, my_file="insample_forecast_naivemodel_results.csv", 
                                                help_message="Download your Naive Model II results to .CSV",
                                                my_key = 'naive_trained_modelii_download_btn',
                                                set_index = True)
                            
                            # =============================================================================
                            # SAVE TEST RESULTS FOR EVALUATION PAGE BY ADDING ROW TO RESULTS_DF                
                            # =============================================================================
                            metrics_dict = my_metrics(df_preds_naive_model2, model_name = model_name)
                            
                            # Get a scientific notation for the rolling window (Note: Latex not supported in Streamlit st.dataframe() so not currently possible)
                            if agg_method_rolling_window_naive_model == 'Mean':      
                                naive_model2_feature_str = f"(1/{size_rolling_window_naive_model}) * ∑yt-1, ..., yt-{size_rolling_window_naive_model}"
                            elif agg_method_rolling_window_naive_model == 'Median':
                                naive_model2_feature_str = f"median(yt-1, ..., yt-{size_rolling_window_naive_model})"
                            elif agg_method_rolling_window_naive_model == 'Mode':
                                naive_model2_feature_str = f"mode(yt-1, ..., yt-{size_rolling_window_naive_model})"
                            else:
                                naive_model2_feature_str = '-'
                                
                            # Create new row with test result details
                            new_row = {'model_name': 'Naive Model II',
                                       'mape': '{:.2%}'.format(metrics_dict[model_name]['mape']),
                                       'smape': '{:.2%}'.format(metrics_dict[model_name]['smape']),
                                       'rmse': round(metrics_dict[model_name]['rmse'], 2),
                                       'r2': round(metrics_dict[model_name]['r2'], 2),
                                       'features': naive_model2_feature_str,
                                       'model_settings': f"rolling window: {size_rolling_window_naive_model}, aggregation method: {agg_method_rolling_window_naive_model}",
                                       'predicted':  np.ravel(df_preds_naive_model2['Predicted'])}
                                      
                            # Turn new row into a dataframe
                            new_row_df = pd.DataFrame([new_row])
                            
                            # Concatenate new row DataFrame with results_df
                            results_df = pd.concat([results_df, new_row_df], ignore_index=True)
                            
                            # Update session state with latest results with new row added
                            set_state("TRAIN_PAGE", ("results_df", results_df))
                            
                            # =============================================================================
                            # Naive Model III
                            # =============================================================================    
                            # set model name
                            model_name = 'Naive Model III'
                            
                            st.markdown('---')
                            
                            # retrieve dataframe with insample prediction results from train/test of naive model III
                            df_preds_naive_model3 = forecast_naive_model3_insample(y_train, 
                                                                                   y_test, 
                                                                                   agg_method = agg_method_baseline_naive_model)
                            # Show metrics on model card
                            display_my_metrics(df_preds_naive_model3, 
                                               model_name = 'Naive Model III', 
                                               my_subtitle = f'{agg_method_baseline_naive_model.lower()}')
                           
                            # Plot graph with actual versus insample predictions
                            plot_actual_vs_predicted(df_preds_naive_model3,
                                                     my_conf_interval)
                               
                            # Show the dataframe
                            st.dataframe(df_preds_naive_model3.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Error': '{:.2f}', 'Error (%)': '{:.2%}'}), use_container_width=True)
                            
                            # Create download button for forecast results to .csv
                            download_csv_button(df_preds_naive_model3, my_file="insample_forecast_naivemodeliii_results.csv", 
                                                help_message="Download your Naive Model III results to .CSV",
                                                my_key = 'naive_trained_modeliii_download_btn',
                                                set_index = True)
                            
                            # =============================================================================
                            # SAVE TEST RESULTS FOR EVALUATION PAGE BY ADDING ROW TO RESULTS_DF                
                            # =============================================================================
                            metrics_dict = my_metrics(df_preds_naive_model3, model_name = "Naive Model III")
                            
                            # Get a scientific notation for the rolling window (Note: Latex not supported in Streamlit st.dataframe() so not currently possible)
                            if agg_method_baseline_naive_model == 'Mean':      
                                naive_model3_feature_str = f"(1/{len(y_train)}) * ∑yt, ..., yt-{len(y_train)}"
                            elif agg_method_baseline_naive_model == 'Median':
                                naive_model3_feature_str = f"median(yt, ..., yt-{len(y_train)})"
                            elif agg_method_baseline_naive_model == 'Mode':
                                naive_model3_feature_str = f"mode(yt, ..., yt-{len(y_train)})"
                            else:
                                naive_model3_feature_str = '-'
    
                            # Create new row with test result details
                            new_row = {'model_name': model_name,
                                       'mape': '{:.2%}'.format(metrics_dict[model_name]['mape']),
                                       'smape': '{:.2%}'.format(metrics_dict[model_name]['smape']),
                                       'rmse': round(metrics_dict[model_name]['rmse'], 2),
                                       'r2': round(metrics_dict[model_name]['r2'], 2),
                                       'features': naive_model3_feature_str,
                                       'model_settings': f"aggregation method: {agg_method_baseline_naive_model.lower()}",
                                       'predicted':  np.ravel(df_preds_naive_model3['Predicted'])}
                                      
                            # Turn new row into a dataframe
                            new_row_df = pd.DataFrame([new_row])
                            
                            # Concatenate new row DataFrame with results_df
                            results_df = pd.concat([results_df, new_row_df], ignore_index=True)
                            
                            # Update session state with latest results with new row added
                            set_state("TRAIN_PAGE", ("results_df", results_df))
                except:
                    st.warning(f'Naive Model(s) failed to train, please check parameters...!')
                try:
                    # =============================================================================
                    # LINEAR REGRESSION MODEL
                    # =============================================================================
                    if model_name == "Linear Regression":        
                        
                         # create card with model insample prediction with linear regression model
                         df_preds_linreg = create_streamlit_model_card(X_train = X_train, 
                                                                       y_train = y_train, 
                                                                       X_test = X_test, 
                                                                       y_test = y_test, 
                                                                       results_df = results_df, 
                                                                       model = model, 
                                                                       model_name = model_name)
                         
                        # =============================================================================
                        # SAVE TEST RESULTS FOR EVALUATION PAGE BY ADDING ROW TO RESULTS_DF                
                        # =============================================================================
                         # Define new row of train/test results
                         new_row = {'model_name': 'Linear Regression',
                                    'mape': '{:.2%}'.format(metrics_dict[model_name]['mape']),
                                    'smape': '{:.2%}'.format(metrics_dict[model_name]['smape']),
                                    'rmse': round(metrics_dict[model_name]['rmse'], 2),
                                    'r2': round(metrics_dict[model_name]['r2'], 2),
                                    'features': features_str,
                                    'model_settings': '-',
                                    'predicted':  np.ravel(df_preds_linreg['Predicted']),
                                    }
                         
                         # Turn new row into a dataframe
                         new_row_df = pd.DataFrame([new_row])
                        
                         # Concatenate new row DataFrame with results_df
                         results_df = pd.concat([results_df, new_row_df], ignore_index=True)
                        
                         # Update session state with latest results with new row added
                         set_state("TRAIN_PAGE", ("results_df", results_df))
                except:
                    st.warning(f'Linear Regression failed to train, please contact your administrator!')

                # =============================================================================
                # SARIMAX MODEL
                # =============================================================================
                try:
                    if model_name == "SARIMAX":
                        with st.expander('📈' + model_name, expanded=True):
                            with st.spinner('This model might require some time to train... you can grab a coffee ☕ or tea 🍵'):   
                                
                                # =============================================================================
                                # PREPROCESS                                
                                # =============================================================================
                                # Assume DF is your DataFrame with boolean columns - needed for SARIMAX model that does not handle boolean, but integers (int) datatype instead
                                bool_cols = X_train.select_dtypes(include=bool).columns
                                X_train.loc[:, bool_cols] = X_train.loc[:, bool_cols].astype(int)
                                bool_cols = X_test.select_dtypes(include=bool).columns
                                X_test.loc[:, bool_cols] = X_test.loc[:, bool_cols].astype(int)
                                
                                # Parameters have standard value but can be changed by user
                                preds_df_sarimax = evaluate_sarimax_model(order=(p,d,q), 
                                                                          seasonal_order = (P,D,Q,s), 
                                                                          exog_train = X_train, 
                                                                          exog_test = X_test, 
                                                                          endog_train = y_train, 
                                                                          endog_test = y_test)
                                
                                # Show metrics on Model Card
                                display_my_metrics(preds_df_sarimax, 
                                                   model_name = "SARIMAX",
                                                   my_subtitle = f'(p, d, q)(P, D, Q) = ({p}, {d}, {q})({P}, {D}, {Q})')
                                                   
                                
                                # Plot graph with actual versus insample predictions on Model Card
                                plot_actual_vs_predicted(preds_df_sarimax, my_conf_interval)
                                
                                # Show the dataframe on Model Card
                                st.dataframe(preds_df_sarimax.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Error': '{:.2f}', 'Error (%)': '{:.2%}'}), use_container_width=True)
                               
                                # Create download button for forecast results to .csv
                                download_csv_button(preds_df_sarimax, my_file="insample_forecast_sarimax_results.csv", help_message="Download your **SARIMAX** model results to .CSV")
                                
                                # =============================================================================
                                # SAVE TEST RESULTS FOR EVALUATION PAGE BY ADDING ROW TO RESULTS_DF                
                                # =============================================================================
                                # Define metrics for sarimax model
                                metrics_dict = my_metrics(preds_df_sarimax, model_name=model_name)
                                
                                # Create new row
                                new_row = {'model_name': 'SARIMAX', 
                                           'mape': '{:.2%}'.format(metrics_dict[model_name]['mape']),
                                           'smape': '{:.2%}'.format(metrics_dict[model_name]['smape']),
                                           'rmse': round(metrics_dict[model_name]['rmse'], 2),
                                           'r2': round(metrics_dict[model_name]['r2'], 2),
                                           'features': features_str,
                                           'model_settings': f'({p},{d},{q})({P},{D},{Q},{s})',
                                           'predicted':  np.ravel(preds_df_sarimax['Predicted'])}
                                
                                # turn new row into a dataframe
                                new_row_df = pd.DataFrame([new_row])
                                
                                # Concatenate new row DataFrame with results_df
                                results_df = pd.concat([results_df, new_row_df], ignore_index=True)

                                # Update session state with latest results with new row added
                                set_state("TRAIN_PAGE", ("results_df", results_df))
                except:
                    st.warning(f'SARIMAX failed to train, please contact administrator!')     
                    
                # =============================================================================
                # PROPHET MODEL                
                # =============================================================================
                if model_name == "Prophet":
                    with st.expander('📈' + model_name, expanded=True):

                        # Use custom function that creates in-sample prediction and return a dataframe with 'Actual', 'Predicted', 'Percentage_Diff', 'MAPE' 
                        preds_df_prophet = predict_prophet(y_train,
                                                           y_test, 
                                                           changepoint_prior_scale = changepoint_prior_scale,
                                                           seasonality_mode = seasonality_mode,
                                                           seasonality_prior_scale = seasonality_prior_scale,
                                                           holidays_prior_scale = holidays_prior_scale,
                                                           yearly_seasonality = yearly_seasonality,
                                                           weekly_seasonality = weekly_seasonality,
                                                           daily_seasonality = daily_seasonality,
                                                           interval_width = interval_width,
                                                           X = X) #TO EXTRACT FEUATURE(S) AND HOLIDAYS BASED ON USER FEATURE SELECTOIN TRUE/FALSE

                        display_my_metrics(preds_df_prophet, "Prophet")
                        
                        # Plot graph with actual versus insample predictions
                        plot_actual_vs_predicted(preds_df_prophet, my_conf_interval)

                        # Show the dataframe
                        st.dataframe(preds_df_prophet.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Error': '{:.2f}', 'Error (%)': '{:.2%}'}), use_container_width=True)

                        # Create download button for forecast results to .csv
                        download_csv_button(preds_df_prophet, 
                                            my_file = "insample_forecast_prophet_results.csv", 
                                            help_message = "Download your **Prophet** model results to .CSV",
                                            my_key = 'download_btn_prophet_df_preds')

                        # =============================================================================
                        # SAVE TEST RESULTS FOR EVALUATION PAGE BY ADDING ROW TO RESULTS_DF                
                        # =============================================================================
                        # define metrics for prophet model
                        metrics_dict = my_metrics(preds_df_prophet, model_name = model_name)
                        
                        # display evaluation results on sidebar of streamlit_model_card
                        new_row = {'model_name': 'Prophet', 
                                   'mape': '{:.2%}'.format(metrics_dict[model_name]['mape']),
                                   'smape': '{:.2%}'.format(metrics_dict[model_name]['smape']),
                                   'rmse': round(metrics_dict[model_name]['rmse'], 2),
                                   'r2': round(metrics_dict[model_name]['r2'], 2),
                                   'features': features_str,
                                   'predicted':  np.ravel(preds_df_prophet['Predicted']),
                                   'model_settings': f' changepoint_prior_scale: {changepoint_prior_scale}, seasonality_prior_scale: {seasonality_prior_scale}, country_holidays: {country_holidays}, holidays_prior_scale: {holidays_prior_scale}, yearly_seasonality: {yearly_seasonality}, weekly_seasonality: {weekly_seasonality}, daily_seasonality: {daily_seasonality}, interval_width: {interval_width}'}

                        # turn new row into a dataframe
                        new_row_df = pd.DataFrame([new_row])

                        # Concatenate new row DataFrame with results_df
                        results_df = pd.concat([results_df, new_row_df], ignore_index=True)

                        # update session state with latest results with new row added
                        set_state("TRAIN_PAGE", ("results_df", results_df))

            # show friendly user reminder message they can compare results on the evaluation page
            st.markdown(f'<h2 style="text-align:center; font-family: Ysabeau SC, sans-serif; font-size: 18px ; color: black; border: 1px solid #d7d8d8; padding: 10px; border-radius: 5px;">💡 Vist the Evaluation Page for a comparison of your test results! </h2>', unsafe_allow_html=True)

    with tab4:
        # SHOW MODEL DETAILED DOCUMENTATION
        #model_documentation(st.session_state['selected_model_info'])
        model_documentation()
     
# =============================================================================
#   ________      __     _     _    _      _______ ______ 
#  |  ____\ \    / /\   | |   | |  | |  /\|__   __|  ____|
#  | |__   \ \  / /  \  | |   | |  | | /  \  | |  | |__   
#  |  __|   \ \/ / /\ \ | |   | |  | |/ /\ \ | |  |  __|  
#  | |____   \  / ____ \| |___| |__| / ____ \| |  | |____ 
#  |______|   \/_/    \_\______\____/_/    \_\_|  |______|
#                                                         
# =============================================================================
#EVALUATE
if menu_item == 'Evaluate' and sidebar_menu_item == 'Home':
    # =============================================================================
    # SIDEBAR OF EVALUATE PAGE
    # =============================================================================
    with st.sidebar:
        
        my_title(f"{evaluate_icon}", "#000000")
        
        with st.form('Plot Metric'):
            
            my_text_paragraph('Plot')
            
            col1, col2, col3 = st.columns([2,8,2])
            with col2:
                
                # define list of metrics user can choose from in drop-down selectbox                
                metrics_list = ['Mean Absolute Percentage Error', 'Symmetric Mean Absolute Percentage Error', 'Root Mean Square Error', 'R-squared']
                
                selected_metric = st.selectbox(label = "*Select evaluation metric to sort by:*", 
                                               options = metrics_list,
                                               key = key1_evaluate,
                                               help = '''Choose which evaluation metric is used to sort the top score of each model by:  
                                                \n- `Mean Absolute Percentage Error` (MAPE):  
                                                \nThe Mean Absolute Percentage Error measures the average absolute percentage difference between the predicted values and the actual values. It provides a relative measure of the accuracy of a forecasting model. A lower MAPE indicates better accuracy.
                                                \n- `Symmetric Mean Absolute Percentage Error` (SMAPE):
                                                \n The Symmetric Mean Absolute Percentage Error (SMAPE) is similar to MAPE but addresses the issue of scale dependency. It calculates the average of the absolute percentage differences between the predicted values and the actual values, taking into account the magnitude of both values. SMAPE values range between 0% and 100%, where lower values indicate better accuracy.
                                                \n- `Root Mean Square Error` (RMSE):
                                                \n The Root Mean Square Error is a commonly used metric to evaluate the accuracy of a prediction model. It calculates the square root of the average of the squared differences between the predicted values and the actual values. RMSE is sensitive to outliers and penalizes larger errors.
                                                \n- `R-squared` (R2):
                                                \nR-squared is a statistical measure that represents the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, where 1 indicates a perfect fit. R-squared measures how well the predicted values fit the actual values and is commonly used to assess the goodness of fit of a regression model.
                                                ''')
                                                
            col1, col2, col3 = st.columns([5,4,4])
            with col2:
                metric_btn = st.form_submit_button('Submit', type="secondary", on_click=form_update, args=('EVALUATE_PAGE',))

        with st.expander('', expanded=True):
            my_text_paragraph('Latest Test Results')
            
            if 'results_df' not in st.session_state:
                st.session_state['results_df'] = results_df  
            else:
                # if user just reloads the page - do not add the same results to the dataframe (e.g. avoid adding duplicates)
                # by checking if the values of the dataframes are the same
                if get_state("TRAIN_PAGE", "results_df").equals(st.session_state['results_df']):
                    pass
                else:
                    # add new test results to results dataframe
                    st.session_state['results_df'] = pd.concat([st.session_state['results_df'], 
                                                                get_state("TRAIN_PAGE", "results_df")], 
                                                                ignore_index=True)
            # =============================================================================
            # 1. Show Last Test Run Results in a Dataframe                    
            # =============================================================================
            st.dataframe(data = get_state("TRAIN_PAGE", "results_df"),
                         column_config = {"predicted": st.column_config.LineChartColumn("predicted", width = "small", help = "insample prediction", y_min = None, y_max = None)},
                         hide_index = True)
           
            my_text_paragraph('Top 3 Test Results')
            
            # Convert the 'mape' column to floats, removes duplicates based on the 'model_name' and 'mape' columns, sorts the unique DataFrame by ascending 'mape' values, selects the top 3 rows, and displays the resulting DataFrame in Streamlit.
            test_df = st.session_state['results_df'].assign(mape = st.session_state.results_df['mape'].str.rstrip('%').astype(float)).drop_duplicates(subset=['model_name', 'mape']).sort_values(by='mape', ascending=True).iloc[:3]
            
            test_df['mape'] = test_df['mape'].astype(str) + '%'
            
            # =============================================================================
            # 2. Show Top 3 test Results in Dataframe
            # =============================================================================
            st.dataframe(data = test_df,  
                         column_config = {"predicted": st.column_config.LineChartColumn("predicted", width = "small", help = "insample prediction", y_min = None, y_max = None)}, 
                         use_container_width=True,
                         hide_index = True)
    
    # =============================================================================
    # MAIN PAGE OF EVALUATE PAGE
    # =============================================================================    
    with st.expander('', expanded=True):
        
        col0, col1, col2, col3 = st.columns([10, 90, 8, 5]) 
        with col1:
           
            my_text_header('Model Performance')
           
            my_text_paragraph(f'{selected_metric}')
            
        # Plot for each model the expirement run with the lowest error (MAPE or RMSE) or highest r2 based on user's chosen metric
        plot_model_performance(st.session_state['results_df'], my_metric = selected_metric)
        
        with col2:
            # Clear session state results_df button to remove prior test results and start fresh
            clear_results_df = st.button(label = '🗑️', on_click = clear_test_results, help = 'Clear Results') # on click run function to clear results
            
        # =============================================================================
        # 3. Show Historical Test Runs Dataframe
        # =============================================================================
        # see for line-chart: https://docs.streamlit.io/library/api-reference/data/st.column_config/st.column_config.linechartcolumn
        st.dataframe(data = st.session_state['results_df'], 
                     column_config = {"predicted": st.column_config.LineChartColumn("predicted", width = "small", help = "insample prediction", y_min = None, y_max = None)}, 
                     hide_index=False, 
                     use_container_width=True)
        
        # Download Button for test results to .csv
        download_csv_button(my_df = st.session_state['results_df'], 
                            my_file = "Modeling Test Results.csv", 
                            help_message = "Download your Modeling Test Results to .CSV")
        
    st.image('./images/evaluation_page.png')

def hyperparameter_tuning_form():
    my_title(f'{tune_icon}', "#88466D")         
     
    with st.form("hyper_parameter_tuning"):
        
        # create option for user to early stop the hyper-parameter tuning process based on time
        max_wait_time = st.time_input(label = 'Set the maximum time allowed for tuning (in minutes)', 
                          value = datetime.time(0, 5),
                          step = 60) # seconds /  You can also pass a datetime.timedelta object.
        
        # create a list of the selected models by the user in the training section of the streamlit app
        model_lst = [model_name for model_name, model in selected_models]
        
        # SELECT MODEL(S): let the user select the models multi-selectbox for hyper-parameter tuning
        selected_model_names = st.multiselect('Select Models', 
                                              options = model_lst,
                                              help='The `Selected Models` are tuned using the chosen hyperparameter search algorithm. A set of hyperparameters is defined, and the models are trained on the training dataset. These models are then evaluated using the selected evaluation metric on a validation dataset for a range of different hyperparameter combinations.')
        
        # SELECT EVALUATION METRIC: let user set evaluation metric for the hyper-parameter tuning
        metric = st.selectbox(label = 'Select Evaluation Metric To Optimize', 
                              options = ['AIC', 'BIC', 'RMSE'], 
                              label_visibility = 'visible', 
                              help = '**`AIC`** (**Akaike Information Criterion**): A measure of the quality of a statistical model, taking into account the goodness of fit and the complexity of the model. A lower AIC indicates a better model fit. \
                              \n**`BIC`** (**Bayesian Information Criterion**): Similar to AIC, but places a stronger penalty on models with many parameters. A lower BIC indicates a better model fit.  \
                              \n**`RMSE`** (**Root Mean Squared Error**): A measure of the differences between predicted and observed values in a regression model. It is the square root of the mean of the squared differences between predicted and observed values. A lower RMSE indicates a better model fit.')
        
        search_algorithm = st.selectbox(label = 'Select Search Algorithm',
                                        options = ['Grid Search', 'Random Search', 'Bayesian Optimization'],
                                        index = 2,
                                        help = '''1. `Grid Search:` algorithm exhaustively searches through all possible combinations of hyperparameter values within a predefined range. It evaluates the model performance for each combination and selects the best set of hyperparameters based on a specified evaluation metric.  
                                                  \n2. `Random Search:` search randomly samples hyperparameters from predefined distributions. It performs a specified number of iterations and evaluates the model performance for each sampled set of hyperparameters. Random search can be more efficient than grid search when the search space is large.
                                                  \n3. `Bayesian Optimization:` is an iterative approach to hyperparameter tuning that uses Bayesian inference. Unlike grid search and random search, Bayesian optimization considers past observations to make informed decisions. It intelligently explores the hyperparameter space by selecting promising points based on the model's predictions and uncertainty estimates. It involves defining the search space, choosing an acquisition function, building a surrogate model, iteratively evaluating and updating, and terminating the optimization. It is particularly effective when the search space is large or evaluation is computationally expensive.
                                                ''')
        # set the number of combinations to search for Bayesian Optimization algorithm                                    
        if search_algorithm == 'Bayesian Optimization':                   
            trial_runs = st.number_input(label = 'Set number of trial runs',
                                         value = 10,
                                         step = 1, 
                                         min_value = 1,
                                         help = '''`Trial Runs` sets the number of maximum different combinations of hyperparameters to be tested by the search algorithm. A larger number will increase the hyperparameter search space, potentially increasing the likelihood of finding better-performing hyperparameter combinations. However, keep in mind that a larger n_trials value can also increase the overall computational time required for the optimization process.
                                                ''')
        else:
            # set default value
            trial_runs = 10
        
        # =============================================================================
        # SARIMAX HYPER-PARAMETER GRID TO SELECT BY USER
        # =============================================================================
        my_text_paragraph('Set Model(s) Search Space(s)', my_text_align='left', my_font_family = 'Ysabeau SC', my_font_weight=200, my_font_size='14px')
        
        with st.expander('◾ Naive Models Parameters'):
            # =============================================================================
            # Naive Model I: Lag      
            # =============================================================================
            my_text_paragraph('Naive Model I: Lag')
        
            col1, col2, col3 = st.columns([1,12,1])
            with col2: 
                lag_options = st.multiselect(label = '*select seasonal lag*', 
                                             options = ['Day', 'Week', 'Month', 'Year'],
                                             default = ['Day', 'Week', 'Month', 'Year'])
                
            vertical_spacer(1)
            
            # =============================================================================
            # Naive Model II: Rolling Window           
            # =============================================================================
            my_text_paragraph('Naive Model II: Rolling Window')
            
            col1, col2, col3 = st.columns([1,12,1])
            with col2:
                rolling_window_range = st.slider(label = 'Select rolling window size:',
                                                                  min_value = 2,
                                                                  max_value = 365,
                                                                  value = (2, 365),
                                                                  step = 1)
                
                rolling_window_options = st.multiselect(label = '*select aggregation method:*', 
                                                                               options = ['Mean', 'Median', 'Mode'],
                                                                               default = ['Mean', 'Median', 'Mode'],
                                                                               key = 'tune_options_naivemodelii')
                
            # =============================================================================
            # Naive Model III                    
            # =============================================================================
            vertical_spacer(1)
            my_text_paragraph('Naive Model III: Constant Value')
            
            col1, col2, col3 = st.columns([1,12,1])
            with col2:
                agg_options = st.multiselect(label = '*select aggregation method:*', 
                                                                         options = ['Mean', 'Median', 'Mode'],
                                                                         default = ['Mean', 'Median', 'Mode'],
                                                                         key = 'tune_options_naivemodeliii')
            vertical_spacer(1)
        
        with st.expander('◾ SARIMAX Hyperparameters'):
            
            col1, col2, col3 = st.columns([5,1,5])
            with col1:
                p_max = st.number_input(label = "*Max Autoregressive Order (p):*", 
                                        value=1, 
                                        min_value=0, 
                                        max_value=10)
                
                d_max = st.number_input(label = "*Max Differencing (d):*", 
                                        value=1, 
                                        min_value=0, 
                                        max_value=10)
                
                q_max = st.number_input(label = "*Max Moving Average (q):*", 
                                        value=1, 
                                        min_value=0, 
                                        max_value=10)   
                
                trend = st.multiselect(label = 'trend',
                                       options = ['n', 'c', 't', 'ct'],
                                       default = ['n'],
                                       help = '''
                                              Options for the 'trend' parameter:
                                              - `n`: No trend component is included in the model.
                                              - `c`: A constant (i.e., a horizontal line) is included in the model.
                                              - `t`: A linear trend component with a time-dependent slope.
                                              - `ct`: A combination of `c` and `t`, including both a constant and a linear time-dependent trend.
                                              ''' )
            with col3:
                P_max = st.number_input(label = "*Max Seasonal Autoregressive Order (P):*", 
                                        value=1, 
                                        min_value=0, 
                                        max_value=10)
                
                D_max = st.number_input(label = "*Max Seasonal Differencing (D):*", 
                                        value=1, 
                                        min_value=0, 
                                        max_value=10)
                
                Q_max = st.number_input(label = "*Max Seasonal Moving Average (Q):*", 
                                        value=1, 
                                        min_value=0, 
                                        max_value=10)
                
                s = st.number_input(label = "*Set Seasonal Periodicity (s):*", 
                                    value = 7, 
                                    min_value = 1)
                

            
            st.write('Parameters')
            col1, col2, col3 = st.columns([5,1,5])
            
            with col1:
                
                enforce_stationarity = st.selectbox('*Enforce Stationarity*', 
                                                       options = [True, False], 
                                                       index = 0,
                                                       help='Whether or not to transform the AR parameters to enforce stationarity in the autoregressive component of the model.')
            with col3:     
                
                enforce_invertibility = st.selectbox('*Enforce Invertibility*', 
                                                       options = [True, False], 
                                                       index = 0,
                                                       help='Whether or not to transform the MA parameters to enforce invertibility in the moving average component of the model.')
        
        # =============================================================================
        # PROPHET HYPER-PARAMETER GRID TO SELECT BY USER
        # =============================================================================
        with st.expander('◾ Prophet Hyperparameters'):
            
            vertical_spacer(1)
            
            col1, col2 = st.columns([5,1])
            with col1:
                # usually set to forecast horizon e.g. 30 days
                horizon_option = st.slider(label = 'Set Forecast Horizon (default = 30 Days):', 
                                           min_value = 1, 
                                           max_value = 365, 
                                           step = 1, value=30, 
                                           help='The horizon for a Prophet model is typically set to the number of time periods that you want to forecast into the future. This is also known as the forecasting horizon or prediction horizon.')
                
                changepoint_prior_scale_options = st.multiselect(label = '*Changepoint Prior Scale*', 
                                                                 options =  [0.001, 0.01, 0.05, 0.1, 1], 
                                                                 default = [0.001, 0.01, 0.1, 1], 
                                                                 help = 'This is probably the most impactful parameter. It determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints. As described in this documentation, if it is too small, the trend will be underfit and variance that should have been modeled with trend changes will instead end up being handled with the noise term. If it is too large, the trend will overfit and in the most extreme case you can end up with the trend capturing yearly seasonality. The default of 0.05 works for many time series, but this could be tuned; a range of [0.001, 0.5] would likely be about right. Parameters like this (regularization penalties; this is effectively a lasso penalty) are often tuned on a log scale.')
                
                seasonality_mode_options = st.multiselect(label = '*Seasonality Modes*', 
                                                          options = [ 'additive', 'multiplicative'], 
                                                          default = [ 'additive', 'multiplicative'])
                
                seasonality_prior_scale_options = st.multiselect(label = '*Seasonality Prior Scales*', 
                                                                 options = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0], 
                                                                 default = [0.01, 10.0])
                
                holidays_prior_scale_options = st.multiselect(label = '*Holidays Prior Scales*', 
                                                              options = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0], 
                                                              default = [0.01, 10.0])
                
                yearly_seasonality_options = st.multiselect(label = '*Yearly Seasonality*', 
                                                            options = [True, False], 
                                                            default = [True])
                
                weekly_seasonality_options = st.multiselect(label = '*Weekly Seasonality*', 
                                                            options = [True, False], 
                                                            default = [True])
                
                daily_seasonality_options = st.multiselect(label = '*Daily Seasonality*', 
                                                           options = [True, False], 
                                                           default = [True])
                
        # create vertical spacing columns
        col1, col2, col3 = st.columns([4,4,4])
        with col2: 
            # create submit button for the hyper-parameter tuning
            hp_tuning_btn = st.form_submit_button("Submit", type="secondary")
             
    return max_wait_time, selected_model_names, metric, search_algorithm, trial_runs, trend, \
           lag_options, rolling_window_range, rolling_window_options, agg_options, \
           p_max, d_max, q_max, P_max, D_max, Q_max, s,  enforce_stationarity, enforce_invertibility, \
           horizon_option, changepoint_prior_scale_options, seasonality_mode_options, seasonality_prior_scale_options, holidays_prior_scale_options, yearly_seasonality_options, weekly_seasonality_options, daily_seasonality_options, \
           hp_tuning_btn

# =============================================================================
#   _______ _    _ _   _ ______ 
#  |__   __| |  | | \ | |  ____|
#     | |  | |  | |  \| | |__   
#     | |  | |  | | . ` |  __|  
#     | |  | |__| | |\  | |____ 
#     |_|   \____/|_| \_|______|
#                               
# =============================================================================
# 9. Hyper-parameter tuning
if menu_item == 'Tune' and sidebar_menu_item == 'Home':

    # =============================================================================
    # Initiate Variables Required
    # =============================================================================
    selected_models = [('Naive Model', None),
                      ('Linear Regression', LinearRegression(fit_intercept=True)), 
                      ('SARIMAX', SARIMAX(y_train)),
                      ('Prophet', Prophet())] # Define tuples of (model name, model)

    # =============================================================================
    # CREATE USER FORM FOR HYPERPARAMETER TUNING
    # =============================================================================
    with st.sidebar:
        # return parameters set for hyper-parameter tuning either default or overwritten by user options
        max_wait_time, selected_model_names, metric, search_algorithm, trial_runs, trend, \
        lag_options, rolling_window_range, rolling_window_options, agg_options, \
        p_max, d_max, q_max, P_max, D_max, Q_max, s, enforce_stationarity, enforce_invertibility, \
        horizon_option, changepoint_prior_scale_options, seasonality_mode_options, seasonality_prior_scale_options, holidays_prior_scale_options, yearly_seasonality_options, weekly_seasonality_options, daily_seasonality_options, \
        hp_tuning_btn = hyperparameter_tuning_form()
        
    # =============================================================================
    # CREATE DATAFRAMES TO STORE HYPERPARAMETER TUNING RESULTS                    
    # =============================================================================
    sarimax_tuning_results = pd.DataFrame()
    prophet_tuning_results = pd.DataFrame()
    
    # if user clicks the hyper-parameter tuning button start hyperparameter tuning code below for selected models
    if hp_tuning_btn == True and selected_model_names:

        # iterate over user selected models from multi-selectbox when user pressed SUBMIT button for hyperparameter tuning
        for model_name in selected_model_names:
            
            # Set start time when grid-search is kicked-off to define total time it takes as computationaly intensive
            start_time = time.time()

            if model_name == 'Naive Model':
                with st.expander('⚙️ Naive Models', expanded = True):
                    #########################################################################################################
                    # INITIATE VARIABLES                    
                    #########################################################################################################
                    progress_bar = st.progress(0)
                    max_wait_time_minutes = int(str(max_wait_time.minute))
                    max_wait_time_seconds = max_wait_time_minutes * 60
                    total_options = len(lag_options) + (len(list(range(rolling_window_range[0], rolling_window_range[1] + 1)))*len(rolling_window_options)) + len(agg_options)  # Store total number of combinations in the parameter grid for progress bar
                    
                    #########################################################################################################
                    # Run Naive Models: I (LAG), II (ROLLING WINDOW), III (CONSTANT: MEAN/MEDIAN/MODE)
                    #########################################################################################################
                    run_naive_model_1(lag_options = lag_options, validation_set = y_test)
                    run_naive_model_2(rolling_window_range = rolling_window_range, rolling_window_options = rolling_window_options, validation_set = y_test)
                    run_naive_model_3(agg_options = agg_options, train_set = y_train, validation_set = y_test)
                    
            if model_name == "SARIMAX":
                if search_algorithm == 'Grid Search':

                    start_time = time.time() # set start time when grid-search is kicked-off to define total time it takes as computationaly intensive
                    progress_bar = st.progress(0) # initiate progress bar for SARIMAX Grid Search runtime
                    
                    # =============================================================================
                    # Define Parameter Grid for Grid Search             
                    # =============================================================================
                    # check if trend has at least 1 option (not an empty list) / e.g. when user clears all options in sidebar - set default to 'n'
                    if trend == []:
                        trend = ['n']
                    
                    param_grid = {'order': [(p, d, q) for p, d, q in itertools.product(range(p_max+1), range(d_max+1), range(q_max+1))],
                                  'seasonal_order': [(p, d, q, s) for p, d, q in itertools.product(range(P_max+1), range(D_max+1), range(Q_max+1))],
                                  'trend': trend}
                    
                    # store total number of combinations in the parameter grid for progress bar
                    total_combinations = len(param_grid['order']) * len(param_grid['seasonal_order']) * len(param_grid['trend'])
                    
                    # Convert max_wait_time to an integer
                    max_wait_time_minutes = int(str(max_wait_time.minute))
                    
                    # Convert the maximum waiting time from minutes to seconds
                    max_wait_time_seconds = max_wait_time_minutes * 60
                    
                    # iterate over grid of all possible combinations of hyperparameters
                    for i, (param, param_seasonal, trend) in enumerate(itertools.product(param_grid['order'], param_grid['seasonal_order'], param_grid['trend']), 0):
                        
                        # Check if the maximum waiting time has been exceeded
                        elapsed_time_seconds = time.time() - start_time
                        
                        if elapsed_time_seconds > max_wait_time_seconds:
                            st.warning("Maximum waiting time exceeded. The grid search has been stopped.")
                            # exit the loop once maximum time is exceeded defined by user or default = 5 minutes
                            break
                        
                        # Update the progress bar
                        progress_percentage = i / total_combinations * 100
                        progress_bar.progress(value = (i / total_combinations), 
                                              text = f'''Please wait up to {max_wait_time_minutes} minute(s) while hyperparameters of {model_name} model are being tuned!  
                                                         \n{progress_percentage:.2f}% of total combinations within the search space reviewed ({i} out of {total_combinations} combinations).''')
                        
                        try:
                            # Create a SARIMAX model with the current parameter values
                            model = SARIMAX(y_train,
                                            order = param,
                                            seasonal_order = param_seasonal,
                                            exog = X_train, 
                                            enforce_stationarity = enforce_stationarity,
                                            enforce_invertibility = enforce_invertibility)
                            
                            # Train/Fit the model to the data
                            mdl = model.fit(disp=0)
                            
                            # Append a new row to the dataframe with the parameter values and AIC score
                            sarimax_tuning_results = sarimax_tuning_results.append({'SARIMAX (p,d,q)x(P,D,Q,s)': f'{param} x {param_seasonal}', 
                                                                                    'order': param, 
                                                                                    'seasonal_order': param_seasonal, 
                                                                                    'trend': trend,
                                                                                    'AIC':  "{:.2f}".format(mdl.aic),
                                                                                    'BIC':  "{:.2f}".format(mdl.bic),
                                                                                    'RMSE': "{:.2f}".format(math.sqrt(mdl.mse))
                                                                                    }, ignore_index=True)
                            
                        # If the model fails to fit, skip it and continue to the next model
                        except:
                            continue
                        
                    # set the end of runtime
                    end_time_sarimax = time.time()
                    
                    # clear progress bar in streamlit for user as process is completed
                    progress_bar.empty()
                    
                    # add a rank column and order dataframe ascending based on the chosen metric
                    gridsearch_sarimax_results_df = rank_dataframe(df = sarimax_tuning_results, metric = metric)
                    
                    # ============================================================================
                    # Show Results in Streamlit
                    # =============================================================================    
                    with st.expander('⚙️ SARIMAX', expanded = True):

                        # Display the result dataframe
                        st.dataframe(gridsearch_sarimax_results_df, hide_index=True, use_container_width = True)
                        st.write(f'🏆 **SARIMAX** hyperparameters with the lowest {metric} of **{gridsearch_sarimax_results_df.iloc[0, 5]}** found in **{end_time_sarimax - start_time:.2f}** seconds is:')
                        st.write(f'- **`(p,d,q)(P,D,Q,s)`**: {gridsearch_sarimax_results_df.iloc[0,1]}')
                        st.markdown('---')
                        
                        # =============================================================================
                        # Hyperparameter Importance                    
                        # =============================================================================
                        # Step 1: Perform the grid search and store the evaluation metric values for each combination of hyperparameters.
                        columns = ['p', 'd', 'q', 'P', 'D', 'Q', 's', 'trend', 'AIC', 'BIC', 'RMSE']
                        df_gridsearch = pd.DataFrame(columns=columns)
                        
                        for _, row in sarimax_tuning_results.iterrows():
                            order = tuple(map(int, str(row['order']).strip('()').split(',')))
                            seasonal_order = tuple(map(int, str(row['seasonal_order']).strip('()').split(',')))
                            aic = float(row['AIC'])
                            bic = float(row['BIC'])
                            rmse = float(row['RMSE'])
                            trend = row['trend']
                            
                            df_gridsearch = df_gridsearch.append({'p': order[0], 'd': order[1], 'q': order[2],
                                                      'P': seasonal_order[0], 'D': seasonal_order[1], 'Q': seasonal_order[2], 's': seasonal_order[3],
                                                      'trend': trend, 'AIC': aic, 'BIC': bic, 'RMSE': rmse}, ignore_index=True)
                        
                        # =============================================================================
                        # Step 2: Calculate the baseline metric as the minimum value of the evaluation metric.
                        # =============================================================================
                        baseline_metric = df_gridsearch[metric].astype(float).min()
                        #st.write(baseline_metric) # TEST
                        
                        # =============================================================================
                        # Step 3: Define / Train Model on hyperparameters (X) and target metric (y)
                        # =============================================================================
                        # Prepare the data
                        df_gridsearch['trend'] = df_gridsearch['trend'].replace({None: 'None'})                        
                        df_gridsearch['trend'] = df_gridsearch['trend'].map({'n': 0, 'c': 1, 't': 2, 'ct': 3, 'None':4})
                        
                        X = df_gridsearch[['p', 'd', 'q', 'P', 'D', 'Q', 'trend']]  # Features (hyperparameters)
                        y = df_gridsearch[metric]  # Target (evaluation metric values)
                        
                        # Define Model for Importance Score testing
                        rf_model = RandomForestRegressor()
                        
                        # Define X and y (X = parameters, y = evaluation metric scores)
                        rf_model.fit(X, y)
                        
                        # =============================================================================
                        # Step 4: Calculate the relative importance of each hyperparameter
                        # =============================================================================
                        feature_importances = rf_model.feature_importances_
                        importance_scores = {param: score for param, score in zip(X.columns, feature_importances)}
                        
                        # Sort the feature importances in descending order
                        sorted_feature_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=False)
                                                
                        # =============================================================================
                        # STEP 5: Plot results
                        # =============================================================================
                        create_hyperparameter_importance_plot(sorted_feature_importance, plot_type='bar_chart', grid_search=True) 
                        
                if search_algorithm == 'Bayesian Optimization':
                    # =============================================================================
                    # Step 0: Define variables in order to retrieve time it takes to run hyperparameter tuning
                    # =============================================================================
                    # set start time when grid-search is kicked-off to define total time it takes as computationaly intensive
                    start_time = time.time()

                    # Convert max_wait_time to an integer
                    max_wait_time_minutes = int(str(max_wait_time.minute))

                    # Convert the maximum waiting time from minutes to seconds
                    max_wait_time_seconds = max_wait_time_minutes * 60
                    
                    # =============================================================================
                    # Step 1: Define Dataframe to save results in
                    # =============================================================================
                    # Create a dataframe to store the tuning results
                    optuna_results = pd.DataFrame(columns=['SARIMAX (p,d,q)x(P,D,Q,s)', 'order', 'seasonal_order', 'trend', 'AIC', 'BIC', 'RMSE'])
                    
                    # =============================================================================
                    # Step 2: Define progress bar for user to visually see progress
                    # =============================================================================
                    # Set progress bar for Optuna runtime
                    progress_bar = st.progress(0)
    
                    # =============================================================================
                    # Step 3: Define parameter grid
                    # =============================================================================
                    pdq =  [(p, d, q) for p, d, q in itertools.product(range(p_max+1), range(d_max+1), range(q_max+1))]
                    pdqs = [(p, d, q, s) for p, d, q in itertools.product(range(P_max+1), range(D_max+1), range(Q_max+1))]
                    
                    # =============================================================================
                    # Step 4: Define the objective function (REQUIREMENT FOR OPTUNA PACKAGE)
                    # =============================================================================
                    def objective_sarima(trial, trend=trend):
                        # define parameters of grid for optuna package with syntax: suggest_categorical()
                        p = trial.suggest_int('p', 0, p_max)
                        d = trial.suggest_int('d', 0, d_max)
                        q = trial.suggest_int('q', 0, q_max)
                        P = trial.suggest_int('P', 0, P_max)
                        D = trial.suggest_int('D', 0, D_max)
                        Q = trial.suggest_int('Q', 0, Q_max)
                        trend = trial.suggest_categorical('trend', trend)
                        
                        order = (p, d, q)
                        seasonal_order = (P, D, Q, s)
                        
                        try:
                            # Your SARIMAX model code here
                            model = SARIMAX(y_train, 
                                            order = order, 
                                            seasonal_order = seasonal_order, 
                                            trend = trend, 
                                            #initialization='approximate_diffuse' # Initialization method for the initial state. If a string, must be one of {‘diffuse’, ‘approximate_diffuse’, ‘stationary’, ‘known’}.
                                            )

                        except ValueError as e:
                            # Handle the specific exception for overlapping moving average terms
                            if 'moving average lag(s)' in str(e):
                                st.warning(f'Skipping parameter combination: {order}, {seasonal_order}, {trend} due to invalid model.')
                            else:
                                raise e
                                                
                        # Train the model
                        mdl = model.fit(disp=0)

                        # Note that we set best_metric to -np.inf instead of np.inf since we want to maximize the R2 metric. 
                        if metric in ['AIC', 'BIC', 'RMSE']:
                            # Check if the current model has a lower AIC than the previous models
                            if metric == 'AIC':
                                my_metric = mdl.aic
                            elif metric == 'BIC':
                                my_metric = mdl.bic
                            elif metric == 'RMSE':
                                my_metric = math.sqrt(mdl.mse)

                        # Append the result to the dataframe
                        optuna_results.loc[len(optuna_results)] = [f'{order}x{seasonal_order}', order, seasonal_order, trend, mdl.aic, mdl.bic, math.sqrt(mdl.mse)]

                        # Update the progress bar
                        progress_percentage = len(optuna_results) / trial_runs * 100
                        progress_bar.progress(value=len(optuna_results) / trial_runs, text=f'Please Wait! Tuning {model_name} hyperparameters... {progress_percentage:.2f}% completed ({len(optuna_results)} of {trial_runs} total combinations).')
                        return my_metric

                    # =============================================================================
                    # Step 5: Create the study (hyperparameter tuning experiment)
                    # =============================================================================                                
                    study = optuna.create_study(direction="minimize")
                    
                    # =============================================================================
                    # Step 6: Optimize the study with the objective function (run experiment)
                    # =============================================================================
                    study.optimize(objective_sarima, 
                                   n_trials = trial_runs,
                                   timeout = max_wait_time_seconds)

                    # =============================================================================
                    # Step 8: Clear progress bar and stop time when optimization has completed                    
                    # =============================================================================
                    # Clear the progress bar in Streamlit
                    progress_bar.empty()
                    
                    # set the end of runtime
                    end_time_sarimax = time.time()
                    
                    # =============================================================================
                    # Step 9: Rank the results dataframe on the evaluation metric
                    # =============================================================================                                        
                    bayesian_sarimax_results_df = rank_dataframe(df = optuna_results, metric = metric)
                    
                    # round to 2 decimals
                    bayesian_sarimax_results_df[['AIC', 'BIC', 'RMSE']] = bayesian_sarimax_results_df[['AIC', 'BIC', 'RMSE']].round(2)
                    
                    # =============================================================================
                    # Step 10: Show Results in Streamlit
                    # =============================================================================    
                    with st.expander('⚙️ SARIMAX', expanded = True):
                    
                        # Display the result dataframe
                        st.dataframe(bayesian_sarimax_results_df, hide_index=True, use_container_width = True)
                        
                        # user messages about the results
                        st.write(f'🏆 **SARIMAX** hyperparameters with the lowest {metric} of **{bayesian_sarimax_results_df.iloc[0, 5]}** found in **{end_time_sarimax - start_time:.2f}** seconds is:')
                        st.write(f'- **`(p,d,q)(P,D,Q,s)`**: {bayesian_sarimax_results_df.iloc[0,1]}')
                        st.markdown('---')
                        
                        # =============================================================================
                        # Plot Hyperparameter Importance                        
                        # =============================================================================
                        # Optuna Parameter Importance Plot
                        create_hyperparameter_importance_plot(study, plot_type='param_importances', grid_search=False)
                    
            if model_name == "Prophet":
                horizon_int = horizon_option
                horizon_str = f'{horizon_int} days'  # construct horizon parameter string
                
                # define cutoffs
                cutoff_date = X_train.index[-(horizon_int+1)].strftime('%Y-%m-%d')
                
                # define the parameter grid - user can select options with multi-select box in streamlit app
                param_grid = {'changepoint_prior_scale': changepoint_prior_scale_options,
                              'seasonality_prior_scale': seasonality_prior_scale_options,
                              'changepoint_prior_scale': changepoint_prior_scale_options,
                              'seasonality_mode': seasonality_mode_options,
                              'seasonality_prior_scale': seasonality_prior_scale_options,
                              'holidays_prior_scale': holidays_prior_scale_options,
                              'yearly_seasonality': yearly_seasonality_options,
                              'weekly_seasonality': weekly_seasonality_options,
                              'daily_seasonality': daily_seasonality_options}
                
                # Generate all combinations of parameters
                all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
                rmses = [] # Store the RMSEs for each params here
                aics = []  # Store the AICs for each params here
                bics = []  # Store the BICs for each params here
                
                # for simplicity set only a single cutoff for train/test split defined by user in the streamlit app
                #cutoffs = pd.to_datetime(['2021-06-01', '2021-12-31']) # add list of dates 
                cutoffs = pd.to_datetime([cutoff_date])
                
                # preprocess the data (y_train/y_test) for prophet model with datestamp (DS) and target (y) column
                y_train_prophet = preprocess_data_prophet(y_train)
                y_test_prophet = preprocess_data_prophet(y_test)
                
                # Use cross validation to evaluate all parameters
                for params in all_params:
                    m = Prophet(**params)  # Fit model with given params
                    # train the model on the data with set parameters
                    m.fit(y_train_prophet)

                    # other examples of forecast horizon settings:
                    #df_cv2 = cross_validation(m, cutoffs=cutoffs, horizon='100 days')
                    #df_cv2 = cross_validation(m, cutoffs=cutoffs, horizon='365 days')
                    df_cv = cross_validation(m, cutoffs=cutoffs, horizon=horizon_str, parallel=False)
                   
                    # rolling_window = 1 computes performance metrics using all the forecasted data to get a single performance metric number.
                    df_p = performance_metrics(df_cv, rolling_window=1)
                    
                    rmses.append(df_p['rmse'].values[0])
                    
                    # Get residuals to compute AIC and BIC
                    df_cv['residuals'] = df_cv['y'] - df_cv['yhat']
                    
                    residuals = df_cv['residuals'].values
                    
                    # Compute AIC and BIC
                    nobs = len(residuals)
                    k = len(params)
                    loglik = -0.5 * nobs * (1 + np.log(2*np.pi) + np.log(np.sum(residuals**2)/nobs))
                    aic = -2 * loglik + 2 * k
                    bic = 2 * loglik + k * np.log(nobs)
                    
                    # Add AIC score to list
                    aics.append(aic)
                    
                    # Add BIC score to list
                    bics.append(bic)
                
                # create dataframe with parameter combinations
                prophet_tuning_results = pd.DataFrame(all_params)
                # add RMSE scores to dataframe
                prophet_tuning_results['rmse'] = rmses
                # add AIC scores to dataframe
                prophet_tuning_results['aic'] = aics
                # add BIC scores to dataframe
                prophet_tuning_results['bic'] = bics
                # set the end of runtime
                end_time_prophet = time.time()        

        if 'Prophet' in selected_model_names:
            with st.expander('⚙️ Prophet', expanded=True):  
                
                if metric.lower() in prophet_tuning_results.columns:
                    
                    prophet_tuning_results['Rank'] = prophet_tuning_results[metric.lower()].rank(method='min', ascending=True).astype(int)
                    
                    # sort values by rank
                    prophet_tuning_results = prophet_tuning_results.sort_values('Rank', ascending=True)
                    
                    # show user dataframe of gridsearch results ordered by ranking
                    st.dataframe(prophet_tuning_results.set_index('Rank'), use_container_width=True)
                    
                    # provide button for user to download the hyperparameter tuning results
                    download_csv_button(prophet_tuning_results, my_file='Prophet_Hyperparameter_Gridsearch.csv', help_message='Download your Hyperparameter tuning results to .CSV')
                
                #st.success(f"ℹ️ Prophet search for your optimal hyper-parameters finished in **{end_time_prophet - start_time:.2f}** seconds")
                
                if not prophet_tuning_results.empty:
                    st.markdown(f'🏆 **Prophet** set of parameters with the lowest {metric} of **{"{:.2f}".format(prophet_tuning_results.loc[0,metric.lower()])}** found in **{end_time_prophet - start_time:.2f}** seconds are:')
                    st.write('\n'.join([f'- **`{param}`**: {prophet_tuning_results.loc[0, param]}' for param in prophet_tuning_results.columns[:6]]))
    else:
            
        with st.expander('', expanded=True):
            st.image('./images/tune_models.png')
            my_text_paragraph('Please select at least one model in the sidebar and press \"Submit\"!', my_font_size = '16px')
                   
# =============================================================================
#   ______ ____  _____  ______ _____           _____ _______ 
#  |  ____/ __ \|  __ \|  ____/ ____|   /\    / ____|__   __|
#  | |__ | |  | | |__) | |__ | |       /  \  | (___    | |   
#  |  __|| |  | |  _  /|  __|| |      / /\ \  \___ \   | |   
#  | |   | |__| | | \ \| |___| |____ / ____ \ ____) |  | |   
#  |_|    \____/|_|  \_\______\_____/_/    \_\_____/   |_|   
#                                                            
# =============================================================================
if menu_item == 'Forecast':
    
    # =============================================================================
    # DEFINE VARIABLES NEEDED FOR FORECAST
    # =============================================================================
    min_date = df['date'].min()
    max_date = df['date'].max()
    max_value_calendar=None
    
    # define maximum value in dataset for year, month day
    year = max_date.year
    month = max_date.month
    day = max_date.day
    end_date_calendar = df['date'].max()
    
    # end date dataframe + 1 day into future is start date of forecast
    start_date_forecast = end_date_calendar + timedelta(days=1)

    # =============================================================================
    # FORECAST SIDEBAR    
    # =============================================================================
    with st.sidebar:
        
        my_title(f'{forecast_icon}', "#48466D")                  
        
        with st.form("📆 "):
            if dwt_features_checkbox:
                # wavelet model choice forecast
                my_subheader('Select Model for Discrete Wavelet Feature(s) Forecast Estimates')
            
                model_type_wavelet = st.selectbox('Select a model', ['Support Vector Regression', 'Linear'], label_visibility='collapsed') 
            
            # define all models in list as we retrain models on entire dataset anyway
            selected_models_forecast_lst = ['Linear Regression', 'SARIMAX', 'Prophet']
            
            # SELECT MODEL(S) for Forecasting
            selected_model_names = st.multiselect('*Select Forecasting Models*', selected_models_forecast_lst, default=selected_models_forecast_lst)  
            
            # create column spacers
            col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 7, 7, 1, 6, 7, 1])
            
            with col2: 
                st.markdown(f'<h5 style="color: #48466D; background-color: #F0F2F6; padding: 12px; border-radius: 5px;"><center> End Date:</center></h5>', unsafe_allow_html=True)
            with col3:
                for model_name in selected_model_names:
                    # if model is linear regression max out the time horizon to maximum possible
                    if model_name == "Linear Regression":
                        
                        # max value is it depends on the length of the input data and the forecasting method used. Linear regression can only forecast as much as the input data if you're using it for time series forecasting.
                        max_value_calendar = end_date_calendar + timedelta(days=len(df))
                        
                    if model_name == "SARIMAX":
                        
                        max_value_calendar = None
                        
                # create user input box for entering date in a streamlit calendar widget
                end_date_forecast = st.date_input(label = "input forecast date", 
                                                  value = start_date_forecast,
                                                  min_value = start_date_forecast, 
                                                  max_value = max_value_calendar, 
                                                  label_visibility = 'collapsed')   
            with col5: 
                # set text information for dropdown frequency
                st.markdown(f'<h5 style="color: #48466D; background-color: #F0F2F6; padding: 12px; border-radius: 5px;"><center> Frequency:</center></h5>', unsafe_allow_html=True)
           
            with col6:
                # Define a dictionary of possible frequencies and their corresponding offsets
                forecast_freq_dict = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
                
                # Ask the user to select the frequency of the data
                forecast_freq = st.selectbox('Select the frequency of the data', list(forecast_freq_dict.keys()), label_visibility='collapsed')
                
                # get the value of the key e.g. D, W, M, Q or Y
                forecast_freq_letter = forecast_freq_dict[forecast_freq]
          
            vertical_spacer(1)
            
            # create vertical spacing columns
            col1, col2, col3 = st.columns([4,4,4])
            with col2:
                # create submit button for the forecast
                forecast_btn = st.form_submit_button("Submit", type="secondary")    
    
    # =============================================================================
    # RUN FORECAST - when user clicks the forecast button then run below
    # =============================================================================
    if forecast_btn:
        # create a date range for your forecast
        future_dates = pd.date_range(start=start_date_forecast, end=end_date_forecast, freq=forecast_freq_letter)
        
        # first create all dates in dataframe with 'date' column
        df_future_dates = future_dates.to_frame(index=False, name='date')
        
        # add the special calendar days
        df_future_dates = create_calendar_special_days(df_future_dates)
        
        # add the year/month/day dummy variables
        df_future_dates = create_date_features(df, year_dummies=year_dummies_checkbox, month_dummies=month_dummies_checkbox, day_dummies=day_dummies_checkbox)
        
        # if user wants discrete wavelet features add them
        if dwt_features_checkbox:
            df_future_dates = forecast_wavelet_features(X, features_df_wavelet, future_dates, df_future_dates)
        
        ##############################################
        # DEFINE X future
        ##############################################
        # select only features user selected from df e.g. slice df    
        X_future = df_future_dates.loc[:, ['date'] + [col for col in feature_selection_user if col in df_future_dates.columns]]
        # set the 'date' column as the index again
        X_future = copy_df_date_index(X_future, datetime_to_date=False, date_to_index=True)
        # iterate over each model name and model in list of lists
        for model_name in selected_model_names:

# =============================================================================
#             def add_prediction_interval(model, X_future, alpha, df):
#                 # calculate the prediction interval for the forecast data
#                 y_forecast = model.predict(X_future)
#                 mse = np.mean((model.predict(model.X) - model.y) ** 2)
#                 n = len(model.X)
#                 dof = n - 2
#                 t_value = stats.t.ppf(1 - alpha / 2, dof)
#                 y_std_err = np.sqrt(mse * (1 + 1 / n + (X_future - np.mean(model.X)) ** 2 / ((n - 1) * np.var(model.X))))
#                 lower_pi = y_forecast - t_value * y_std_err
#                 upper_pi = y_forecast + t_value * y_std_err
#                 # create a dataframe with the prediction interval and add it to the existing dataframe
#                 df['lower_pi'] = lower_pi
#                 df['upper_pi'] = upper_pi
#                 return df
# =============================================================================

            if model_name == "Linear Regression":                
                model = LinearRegression()
                # train the model on all data (X) for which we have data in forecast that user feature selected
                # e.g. if X originaly had August, but the forecast does not have August
                model.fit(X.loc[:, [col for col in feature_selection_user if col in df_future_dates.columns]], y)
                # forecast (y_hat with dtype numpy array)
                y_forecast = model.predict(X_future) 
                # convert numpy array y_forecast to a dataframe
                df_forecast_lr = pd.DataFrame(y_forecast, columns = ['forecast']).round(0)
                # create a dataframe with the DatetimeIndex as the index
                df_future_dates_only = future_dates.to_frame(index=False, name='date')
                # combine dataframe of date with y_forecast
                df_forecast_lr = copy_df_date_index(df_future_dates_only.join(df_forecast_lr), datetime_to_date=False, date_to_index=True)
# =============================================================================
#                 # Add the prediction interval to the forecast dataframe
#                 alpha = 0.05  # Level of significance for the prediction interval
#                 df_forecast_lr = add_prediction_interval(model, X_future, alpha, df_forecast_lr)
# =============================================================================

                # create forecast model score card in streamlit
                with st.expander('ℹ️' + model_name + ' Forecast', expanded=True):   
                    my_header(f'{model_name}')
                    # Create the forecast plot
                    plot_forecast(y, df_forecast_lr, title='')
                    # set note that maximum chosen date can only be up to length of input data with Linear Regression Model
                    #st.caption('Note: Linear Regression Model maximum end date depends on length of input data')
                    # show dataframe / output of forecast in streamlit linear regression
                    st.dataframe(df_forecast_lr, use_container_width=True)
                    download_csv_button(df_forecast_lr, my_file="forecast_linear_regression_results.csv") 
            
            if model_name == "SARIMAX":
                
                # =============================================================================
                # Define Model Parameters
                # =============================================================================
                order = (p,d,q)
                seasonal_order = (P,D,Q,s)
                
                # Define model on all data (X)
                # Assume df is your DataFrame with boolean columns - needed for SARIMAX model that does not handle boolean, but int instead
                # X bool to int dtypes
                bool_cols = X.select_dtypes(include=bool).columns
                X.loc[:, bool_cols] = X.loc[:, bool_cols].astype(int)
                
                # X_future bool to int dtypes
                bool_cols = X_future.select_dtypes(include=bool).columns
                X_future.loc[:, bool_cols] = X_future.loc[:, bool_cols].astype(int)
                
                # define model SARIMAX
                model = SARIMAX(endog = y, 
                                order=(p, d, q),  
                                seasonal_order=(P, D, Q, s), 
                                exog=X.loc[:, [col for col in feature_selection_user if col in df_future_dates.columns]], 
                                enforce_invertibility = enforce_invertibility, 
                                enforce_stationarity = enforce_stationarity
                                ).fit()
           
                # Forecast future values
                my_forecast_steps = (end_date_forecast-start_date_forecast.date()).days
                
                #y_forecast = model_fit.predict(start=start_date_forecast, end=(start_date_forecast + timedelta(days=len(X_future)-1)), exog=X_future)
                forecast_values = model.get_forecast(steps = my_forecast_steps, exog = X_future.iloc[:my_forecast_steps,:])

                # set the start date of forecasted value e.g. +7 days for new date
                start_date = max_date + timedelta(days=1)
                # create pandas series before appending to the forecast dataframe
                date_series = pd.date_range(start=start_date, end=None, periods= my_forecast_steps, freq=forecast_freq_letter)
                
                # create dataframe
                df_forecast =  pd.DataFrame()
                
                # add date series to forecasting pandas dataframe
                df_forecast['date'] = date_series.to_frame(index = False)
                
                # convert forecast to integers (e.g. round it)
                df_forecast[('forecast')] = forecast_values.predicted_mean.values.astype(int).round(0)        
                
                # set 'date' as the index of the dataframe
                df_forecast_sarimax = copy_df_date_index(df_forecast)
                
                with st.expander('ℹ️ ' + model_name + ' Forecast', expanded=True):   
                    my_header(f'{model_name}')
                    # Create the forecast plot
                    plot_forecast(y, df_forecast_sarimax, title='')
                    # set note that maximum chosen date can only be up to length of input data with Linear Regression Model
                    #st.caption('Note: Linear Regression Model maximum end date depends on length of input data')
                    # show dataframe / output of forecast in streamlit linear regression
                    st.dataframe(df_forecast_sarimax, use_container_width=True)
            
            if model_name == "Prophet": # NOTE: Currently no X features included in this Prophet model
                forecast_prophet = pd.DataFrame()
                
                # prep data for specificalmodelly prophet model requirements, data should have ds column and y column
                y_prophet = preprocess_data_prophet(y)
                
                # define 
                m = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                            seasonality_mode=seasonality_mode,
                            seasonality_prior_scale=seasonality_prior_scale,
                            holidays_prior_scale=holidays_prior_scale,
                            yearly_seasonality=yearly_seasonality,
                            weekly_seasonality=weekly_seasonality,
                            daily_seasonality=daily_seasonality,
                            interval_width=interval_width)
                
                # train the model on the entire dataset with set parameters
                m.fit(y_prophet)
                
                
                future = m.make_future_dataframe(periods=len(future_dates), freq='D')
               
                # Predict on the test set
                forecast_prophet = m.predict(future)
                
                # change name of yhat to forecast in df
                forecast_prophet['forecast'] = forecast_prophet['yhat'].round(0)
                #forecast_prophet['date'] = forecast_prophet['ds']
                forecast_prophet['date'] = forecast_prophet['ds']
                forecast_prophet = forecast_prophet[['date', 'forecast']]
                # set the date column as index column
                forecast_prophet = forecast_prophet.set_index('date')
                # cut off the insample forecast
                forecast_prophet= forecast_prophet[len(y):]
                with st.expander('ℹ️ ' + model_name + ' Forecast', expanded=True):
                    my_header(f'{model_name}')
                    plot_forecast(y, forecast_prophet, title='')
                    # show dataframe / output of forecast in streamlit
                    st.dataframe(forecast_prophet, use_container_width=True)
                    download_csv_button(forecast_prophet, my_file="forecast_prophet_results.csv")