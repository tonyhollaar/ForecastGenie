# -*- coding: utf-8 -*-
"""
Streamlit App to Forecast Website Traffic
Created on Mon Mar  6 17:02:34 2023
@author: tholl
"""
# Import required packages
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import math
import plotly.express as px
from sklearn.linear_model import LinearRegression
from pmdarima.arima import auto_arima
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense
from sklearn.feature_selection import mutual_info_classif
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import(
                                    AbstractHolidayCalendar, Holiday, DateOffset, \
                                    SU, MO, TU, WE, TH, FR, SA, \
                                    next_monday, nearest_workday, sunday_to_monday,
                                    EasterMonday, GoodFriday, Easter
                                  )
#########################################################################
# set/modify standard page configuration
#########################################################################
st.set_page_config(page_title="ForecastGenie", 
                   layout="centered", # "centered" or "wide"
                   page_icon="🌀", 
                   initial_sidebar_state="auto") # "auto" or "expanded" or "collapsed"
###############################################################################
# FUNCTIONS
###############################################################################
@st.cache_data
def convert_df(my_dataframe):
   return my_dataframe.to_csv(index=False).encode('utf-8')

# create a title with custom background color and header size
def my_title(my_string, my_background_color="#45B8AC"):
    st.markdown(f'<h3 style="color:#FFFFFF; background-color:{my_background_color}; padding:5px; border-radius: 5px;"> <center> {my_string} </center> </h3>', unsafe_allow_html=True)

def my_header(my_string, my_style="#8c8c8c"):
    #st.markdown(f'<h2 style="text-align:center"> {my_string} </h2>', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color:{my_style};"> <center> {my_string} </center> </h2>', unsafe_allow_html=True)

def my_subheader(my_string, my_style="#217CD0", my_size=5):
    st.markdown(f'<h{my_size} style="color:{my_style};"> <center> {my_string} </center> </h{my_size}>', unsafe_allow_html=True)

def my_subheader_metric(string1, color1="#cfd7c2", metric=0, color2="#FF0000", my_style="#000000", my_size=5):
    metric_rounded = "{:.2%}".format(metric)
    metric_formatted = f"<span style='color:{color2}'>{metric_rounded}</span>"
    string1 = string1.replace(f"{metric}", f"<span style='color:{color1}'>{metric_rounded}</span>")
    st.markdown(f'<h{my_size} style="color:{my_style};"> <center> {string1} {metric_formatted} </center> </h{my_size}>', unsafe_allow_html=True)

def wait(seconds):
    start_time = time.time()
    with st.spinner(f"Please wait... {int(time.time() - start_time)} seconds passed"):
        time.sleep(seconds)
        
@st.cache_data
def my_holiday_name_func(my_date):
  # function to get the holiday name from package calendar
  holiday_name = cal.holidays(start = my_date, end = my_date, return_name = True)
  if len(holiday_name) < 1:
    holiday_name = ""
    return holiday_name
  else: 
   return holiday_name[0]  

@st.cache_data
def evaluate_linear_regression_model(X_train, y_train, X_test, y_test):
    # Create linear regression object
    regr = LinearRegression(fit_intercept=True)
    # Train the model using the training sets
    # note: first fit X then Y in that order
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mse, rmse, r2

@st.cache_data
def load_data():
    df = pd.read_csv(uploaded_file, parse_dates=['date'])
    return df

@st.cache_data   
def create_df_pred(y_test, y_pred, show_df_streamlit=True):
    try:
        df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
        # set the index to just the date portion of the datetime index
        df_preds.index = df_preds.index.date
        # Calculate percentage difference between actual and predicted values and add it as a new column
        df_preds = df_preds.assign(Percentage_Diff = ((df_preds['Predicted'] - df_preds['Actual']) / df_preds['Actual']))
        # Calculate MAPE and add it as a new column
        df_preds = df_preds.assign(MAPE = abs(df_preds['Percentage_Diff']))
        # show the predictions versus actual results
        my_df = df_preds.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Percentage_Diff': '{:.2%}'})
        if show_df_streamlit == True:
            return st.write(my_df)
        else:
           return df_preds
    except:
        st.warning('error in create_df_pred function')
   
@st.cache_data
def my_metrics(my_df):
   mape = my_df['MAPE'].mean()
   mse = mean_squared_error(my_df['Actual'], my_df['Predicted'])
   rmse = np.sqrt(mse)
   r2 = r2_score(my_df['Actual'], my_df['Predicted'])
   return mape, mse, rmse, r2
  
def my_download_button(my_df, my_file="forecast_model.csv"):
     # create download button for forecast results to .csv
     csv = convert_df(my_df)
     col1, col2, col3 = st.columns([2,2,2])
     with col2: 
         st.download_button(":arrow_down: Download",
                            csv,
                            my_file,
                            "text/csv",
                            key='download-csv_arimax',
                            help = "Download your forecast to .CSV file")
@st.cache_data   
def my_plot(X_test, preds):
    # LINEPLOT WITH PLOTLY
    # Plot the predictions versus actual results
    # Combine X_test, y_test and y_pred into a single DataFrame
    df_X_test = X_test.copy()
    df_X_test['Actual'] = y_test
    df_X_test['Insample Prediction'] = preds
    
    # Define the color palette
    colors = ['#5276A7', '#60B49F']
    # Create the figure with easy on eyes colors
    fig = px.line(df_X_test, x=X_test.index, y=['Actual', 'Insample Prediction'],
                  color_discrete_sequence=colors)
    
    # Update the layout of the figure
    fig.update_layout(
        title='',
        xaxis_title='',
        yaxis_title='',
        legend_title='Legend',
        font=dict(family='Arial', size=12, color='#707070'),
        yaxis=dict(gridcolor='#E1E1E1', range=[0, max(df_X_test['Actual'])]),
        xaxis=dict(gridcolor='#E1E1E1'))
    # Set the line colors
    for i in range(len(colors)):
        fig.data[i].line.color = colors[i]
    fig.update_layout(legend=dict(yanchor="bottom",
                                  y=0.0,
                                  xanchor="center",
                                  x=0.99))
    
    # Render the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)     

def train_my_models(y_train, X_train, X_test, y_test, selected_models):
    # title of model on model-card
    st.markdown('<h2 style="text-align:center"> ARIMAX </h2></p>', unsafe_allow_html=True)
    # create vertical spacing columns    
    col0, col1, col2, col3, col4 = st.columns([2, 3, 3, 3, 1])                    
    # calculate metrics
    arimax_model = auto_arima(y_train, X=X_train, seasonal=True, m=12, suppress_warnings=True, 
                              stepwise=True, 
                              param_distributions=param_subset['ARIMAX'], 
                              n_iter=10)

    # Display the evaluation metrics
    preds = arimax_model.predict(n_periods = len(X_test), X = X_test)
    my_df = create_df_pred(y_test, preds, False)  
    # apply custom function my_metrics to dataframe
    mape, mse, rmse, r2 = my_metrics(my_df)
    with col1:
        st.metric(':red[**MAPE:**]', delta=None, value = "{:.2%}".format(mape))
    with col2:
        st.metric(':red[**RMSE:**]', delta = None, value = round(rmse,2))
    with col3: 
        st.metric(':green[**R-squared:**]',  delta=None, value= round(r2, 2))
    
    # call function to plot    
    # my_plot(X_test, preds)  
    return my_df, preds

@st.cache_data   
# remove datatypes object - e.g. descriptive columns not used in modeling
def remove_object_columns(df):
    # Get a list of column names with object datatype
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    # Remove the columns that are not needed
    st.write('the following columns are removed from a copy of the dataframe due to their datatype (object)')
    obj_cols_to_remove = []
    for col in obj_cols:
        obj_cols_to_remove.append(col)
    st.write(obj_cols_to_remove)
    df = df.drop(columns=obj_cols_to_remove)
    return df

###############################################################################
# I want the dataframe with model name and metrics in sidebar to 
# update when additional model(s) are trained
###############################################################################
# Define a function to update the dataframe
def update_dataframe(df, model_name, mape):
    # Check if model already exists in dataframe
    if model_name in df["Model"].tolist():
        # Replace the row
        df.loc[df['Model'] == model_name, 'MAPE'] = "{:.2f}%".format(mape*100)
    else:
        # Add new row
        new_row = {"Model": model_name, "MAPE": "{:.2f}%".format(mape*100)}
        df = df.append(new_row, ignore_index=True)
    return df

# Define a function to initialize the dataframe
def initialize_dataframe():
    df = pd.DataFrame(columns=["Model", "MAPE"])
    return df

# Initialize the dataframe using SessionState
session_state = st.session_state.setdefault("state", {})
if "df" not in session_state:
    session_state["df"] = initialize_dataframe() 

###############################################################################
# Create Left-Sidebar Streamlit App
###############################################################################
with st.sidebar:   
        st.markdown(f'<h1 style="color:#45B8AC;"> <center> ForecastGenie™️ </center> </h1>', unsafe_allow_html=True)         
# set the title of page
st.title(":blue[]")
st.markdown(f'<h1 style="color:#45B8AC;"> <center> ForecastGenie™️ </center> </h1>', unsafe_allow_html=True)
# add vertical spacing
st.write("")

with st.sidebar.expander(':information_source: About', expanded=False):
    st.write('''Hi :wave: **Welcome** to the ForecastGenie app created with Streamlit :smiley:
            
                \n**What does it do?**  
                - Forecast your Time Series Dataset - for example predict your future daily website `visits`!  
                \n**What do you need?**  
                - Upload your .csv file that contains your **date** (X) column and your target variable of interest (Y)  
            ''')
    st.markdown('---')
    
    col1, col2, col3 = st.columns([1,3,2])
    with col2:
        st.image('./images/logo_dark.png', caption="Developed by")  
        # added spaces to align website link with logo in horizontal center
        st.markdown(f'<h5 style="color:#217CD0;"><center><a href="https://www.tonyhollaar.com/">www.tonyhollaar.com</a></center></h5>', unsafe_allow_html=True)
    st.markdown('---')

###############################################################################
# 1. Create Button to Load Dataset (.CSV format)
###############################################################################
#st.markdown(f'<h1> <center>  1. Load Dataset 🚀 </center> </h1>', unsafe_allow_html=True)
my_title("1. Load Dataset 🚀 ", "#2CB8A1")
with st.sidebar:
    my_title("Load Dataset 🚀 ", "#2CB8A1")
    uploaded_file = st.file_uploader("upload your .CSV file", label_visibility="collapsed")

# if nothing is uploaded yet by user run below code
if uploaded_file is None:
    # let user upload a file
    # inform user what template to upload
    with st.expander("", expanded=True):
        my_header("Instructions")
        st.info('''👈 **Please upload a .CSV file with:**  
                 - first column named: **$date$** with format: **$mm-dd-yyyy$**  
                 - second column the target variable: $y$''')

# if user uploaded csv file run below code
# wrap all code inside this related to data analysis / modeling
if uploaded_file is not None:   
    # only keep date and total_traffic
    df = load_data()
    df_graph = df.copy(deep=True)
    df_total = df.copy(deep=True)
   
    # if loaded dataframe succesfully run below
    # set minimum date
    df_min = df.iloc[:,0].min().date()
    # set maximum date
    df_max = df.iloc[:,0].max().date()
    # create dataframe from custom built function
    df = load_data()
    ## show message to user data loaded
    st.info('''🗨️ **Great Work!** your data is loaded, lets take a look :eyes: shall we...''')
    
    # create expandable card with data exploration information
    with st.expander(':arrow_down: EDA', expanded=True):
        # set header
        my_header("Exploratory Data Analysis")
        # create 3 columns for spacing
        col1, col2, col3 = st.columns([1,3,1])
        # display df shape and date range min/max for user
        col2.markdown(f"<center>Your <b>dataframe</b> has <b><font color='#0F52BA'>{df.shape[0]}</b></font> \
                      rows and <b><font color='#0F52BA'>{df.shape[1]}</b></font> columns <br> with date range: \
                      <b><font color='#FF5733'>{df_min}</b></font> to <b><font color='#FF5733'>{df_max}</font></b>.</center>", 
                      unsafe_allow_html=True)
        # add a vertical linespace
        st.write("")
        # set two column spacers
        col1, col2 = st.columns([2,3])
        with col1:    
            # show dataframe
            # set the date as the index of the pandas dataframe
            show_df = df.copy(deep=True)
            # convert the datetime to date (excl time 00:00:00)
            show_df['date'] = pd.to_datetime(show_df['date']).dt.date
            # set the index instead of 0,1,2... to date
            show_df = show_df.set_index('date')
            # display the dataframe in streamlit
            st.dataframe(show_df, use_container_width=True)
        with col2:
            # set caption
            st.caption('')
            ## create graph
            df_graph = df_graph.set_index('date')
            ## display/plot graph of dataframe
            st.line_chart(df_graph)
            
    ###############################################################################
    # 2. Feature Engineering
    ###############################################################################
    my_title("2. Feature Engineering 🧰", "#FF6F61")
    with st.sidebar:
        my_title("Feature Engineering 🧰", "#FF6F61")  
        st.info(''' Select your explanatory variables''')
        # show checkbox in middle of sidebar to select all features or none
        col1, col2, col3 = st.columns([0.1,8,3])
        with col3:
            select_all_days = st.checkbox('Select All Special Days:', value=True, label_visibility='collapsed' )
            select_all_seasonal = st.checkbox('Select All Sesaonal:', value=True, label_visibility='collapsed' )
        with col2:
            if select_all_days == True:
               st.write("All Special Days")
            else:
                st.write("No Special Days") 
            if select_all_seasonal == True:
                st.write("All Seasonal Days")
            else:
                st.write("No Seasonal Days") 
                
    with st.expander("📌", expanded=True):
        my_header('Special Calendar Days')
        start_date_calendar = df['date'].min()
        end_date_calendar = df['date'].max()
        st.markdown('---')
        df_exogenous_vars = pd.DataFrame({'date': pd.date_range(start = start_date_calendar, 
                                                                end = end_date_calendar)})
        cal = calendar()
        holidays = cal.holidays(start=start_date_calendar, end=end_date_calendar)
        # holiday = true/ otherwise false
        df_exogenous_vars['holiday'] = (df_exogenous_vars['date'].isin(holidays)).apply(lambda x: 1 if x==True else 0)
        
        # create column for holiday description
        df_exogenous_vars['holiday_desc'] = df_exogenous_vars['date'].apply(lambda x: my_holiday_name_func(x))
        my_subheader("🎁 Pick your special days to include: ")
        st.write("")
        col0, col1, col2, col3 = st.columns([6,12,12,1])
        
        ###############################################
        # create checkboxes for special days
        ###############################################
        with col1:
            jan_sales = st.checkbox('January Sale', value=select_all_days)
            val_day_lod = st.checkbox('Valentine\'s Day [last order date]', value=select_all_days)
            val_day = st.checkbox('Valentine\'s Day', value=select_all_days)
            mother_day_lod = st.checkbox('Mother\'s Day [last order date]', value=select_all_days)
            mother_day = st.checkbox('Mother\'s Day', value=select_all_days)
            father_day_lod = st.checkbox('Father\'s Day [last order date]', value=select_all_days)
            pay_days = st.checkbox('Monthly Pay Days (4th Friday of month)', value=select_all_days)
        with col2:
            father_day = st.checkbox('Father\'s Day', value=select_all_days)
            black_friday_lod = st.checkbox('Black Friday [sale starts]', value=select_all_days)
            black_friday = st.checkbox('Black Friday', value=select_all_days)
            cyber_monday = st.checkbox('Cyber Monday', value=select_all_days)
            christmas_day = st.checkbox('Christmas Day [last order date]', value=select_all_days)
            boxing_day = st.checkbox('Boxing Day sale', value=select_all_days)
        # source: https://practicaldatascience.co.uk/data-science/how-to-create-an-ecommerce-trading-calendar-using-pandas
        class UKEcommerceTradingCalendar(AbstractHolidayCalendar):
            rules = []
            # Seasonal trading events
            # only add Holiday if user checkmarked checkbox (e.g. equals True) 
            if jan_sales:
                rules.append(Holiday('January sale', month = 1, day = 1))
            if val_day_lod:
                rules.append(Holiday('Valentine\'s Day [last order date]', month = 2, day = 14, offset = BDay(-2)))
            if val_day:
                rules.append(Holiday('Valentine\'s Day', month = 2, day = 14))
            if mother_day_lod:
                rules.append(Holiday('Mother\'s Day [last order date]', month = 5, day = 1, offset = BDay(-2)))
            if mother_day:
                rules.append(Holiday('Mother\'s Day', month = 5, day = 1, offset = pd.DateOffset(weekday = SU(2))))
            if father_day_lod:
                rules.append(Holiday('Father\'s Day [last order date]', month = 6, day = 1, offset = BDay(-2)))
            if father_day:
                rules.append(Holiday('Father\'s Day', month = 6, day = 1, offset = pd.DateOffset(weekday = SU(3))))
            if black_friday_lod:
                rules.append(Holiday("Black Friday [sale starts]", month = 11, day = 1, offset = [pd.DateOffset(weekday = SA(4)), BDay(-5)]))
            if black_friday:
                rules.append(Holiday('Black Friday', month = 11, day = 1, offset = pd.DateOffset(weekday = FR(4))))
            if cyber_monday:
                rules.append(Holiday("Cyber Monday", month = 11, day = 1, offset = [pd.DateOffset(weekday = SA(4)), pd.DateOffset(2)]))
            if christmas_day:
                rules.append(Holiday('Christmas Day [last order date]', month = 12, day = 25, offset = BDay(-2)))
            if boxing_day:
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
            if pay_days == True:
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
        # add boolean
        df_exogenous_vars['calendar_event'] = df_exogenous_vars['calendar_event_desc'].apply(lambda x: 1 if len(x)>1 else 0)
        
        ###############################################################################
        # Reorder Columns to logical order e.g. value | description of value
        ###############################################################################
        df_exogenous_vars = df_exogenous_vars[['date', 'holiday', 'holiday_desc', 'calendar_event', 'calendar_event_desc', 'pay_day','pay_day_desc']]
        
        ###############################################################################
        # combine exogenous vars with df_total
        ###############################################################################
        df_total_incl_exogenous = pd.merge(df_total, df_exogenous_vars, on='date', how='left' )
        df = df_total_incl_exogenous.copy(deep=True)
        
        ##############################
        # Add Day/Month/Year Features
        ##############################
        # create checkboxes for user to checkmark if to include features
        st.markdown('---')
        my_subheader('🌓 Pick your seasonal days to include: ')
        # vertical space / newline between header and checkboxes
        st.write("")
        # create columns for aligning in middle the checkboxes
        col0, col1, col2, col3, col4 = st.columns([2, 2, 2, 2, 1])
        with col1:
            year_dummies = st.checkbox('Year', value=select_all_seasonal)
        with col2:
            month_dummies = st.checkbox('Month', value=select_all_seasonal)
        with col3:
            day_dummies = st.checkbox('Day', value=select_all_seasonal)
        st.info(':information_source: **Note**: to prevent perfect multi-collinearity, leave-one-out is applied e.g. one year/month/day')
        ##########
        ## YEAR
        ##########
        if year_dummies:
            df['year'] = df['date'].dt.year
            dum_year =  pd.get_dummies(df['year'], columns = ['year'], drop_first=True, prefix='year', prefix_sep='_')
            df = pd.concat([df, dum_year], axis=1)   
        ##########
        ## MONTH
        ##########
        if month_dummies:
            month_dict = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
            df['month'] = df['date'].dt.month.apply(lambda x:month_dict.get(x))
            dum_month =  pd.get_dummies(df['month'], columns = ['month'], drop_first=True, prefix='', prefix_sep='')
            df = pd.concat([df, dum_month], axis=1)                            
        ##########
        # DAY
        ##########
        # date.weekday() - Return the day of the week as an integer, where Monday is 0 and Sunday is 6
        # convert 0 - 6 to weekday names
        if day_dummies:
            week_dict = {0:'Monday', 1:'Tuesday',2:'Wednesday', 3:'Thursday', 4: 'Friday', 5:'Saturday', 6:'Sunday'}
            df['day'] = df['date'].dt.weekday.apply(lambda x: week_dict.get(x))
            # get dummies
            dum_day = pd.get_dummies(df['day'], columns = ['day'], drop_first=True, prefix='', prefix_sep='')
            df = pd.concat([df, dum_day], axis=1)    
        st.markdown('---')
        # SHOW DATAFRAME
        st.write('**re-order dataframe TODO**')
        st.dataframe(df)     

    ###############################################################################
    # 3. Prepare Data (split into training/test)
    ###############################################################################
    my_title('3. Prepare Data 🧪', "#FFB347")
    with st.sidebar:
        my_title('Prepare Data 🧪', "#FFB347")
    # 5.1) set date as index/ create local_df
    # create copy of dataframe not altering original
    local_df = df.copy(deep=True)
    # set the date as the index of the pandas dataframe
    local_df.index = pd.to_datetime(local_df['date'])
    local_df.drop(columns='date', inplace=True)

    
    with st.expander(':information_source: I removed the following descriptive columns automatically from analysis'):
        local_df = remove_object_columns(local_df)
    
    # USER TO SET the insample forecast days 
    with st.expander("", expanded=True):
        st.write("")
        my_subheader('How many days you want to use as your test-set*?')  
        st.caption(f'<h6> <center> *Note: A commonly used ratio is 80:20 split between train and test </center> <h6>', unsafe_allow_html=True)
    
        #### create sliders for user insample test-size (/train-size automatically as well)
        def update(change):
            if change == 'steps':
                st.session_state.steps = st.session_state.steps
            else:
                st.session_state.steps = math.floor((st.session_state.perc/100)*len(local_df))
        my_max_value = len(df)-1
        my_insample_forecast_steps = st.slider('**days**', 
                                               min_value = 1, 
                                               max_value = my_max_value, 
                                               value = int(len(df)*0.2),
                                               step = 1,
                                               key='steps',
                                               on_change=update,
                                               args=('steps',))
        
        with st.sidebar:
            st.info('Select in-sample test-size')
            my_insample_forecast_perc = st.slider('**percentage**', 
                                                  min_value = 1, 
                                                  max_value = int(my_max_value/len(df)*100), 
                                                  value = int(my_insample_forecast_steps/len(df)*100),
                                                  step = 1,
                                                  key='perc',
                                                  on_change=update,
                                                  args=('perc',))
            
        perc_test_set = "{:.2f}%".format((my_insample_forecast_steps/len(df))*100)
        perc_train_set = "{:.2f}%".format(((len(df)-my_insample_forecast_steps)/len(df))*100)
        st.info(f"the sample-sizes of the train/test split equals :green[**{perc_train_set}**] and :green[**{perc_test_set}**] ")
    
    X = local_df.iloc[:, 1:]
    y = local_df.iloc[:, 0:1]
    X_train = local_df.iloc[:, 1:][:(len(df)-my_insample_forecast_steps)]
    X_test = local_df.iloc[:, 1:][(len(df)-my_insample_forecast_steps):]
    # set endogenous variable train/test split
    y_train = local_df.iloc[:, 0:1][:(len(df)-my_insample_forecast_steps)]
    y_test = local_df.iloc[:, 0:1][(len(df)-my_insample_forecast_steps):]
    
    ###############################################################################
    # 4. Feature Selection
    ###############################################################################
    my_title('4. Feature Selection 🍏🍐🍋', "#CBB4D4")
    with st.sidebar:
        my_title('Feature Selection 🍏🍐🍋', "#CBB4D4")
    st.info('Let\'s review your top features to use in analysis ')
    st.info('''    Recursive Feature Elimination (RFE): This method involves recursively removing features and building a model on the remaining features. It then ranks the features based on their importance and eliminates the least important feature.
    Principal Component Analysis (PCA): This method transforms the original set of features into a smaller set of features, called principal components, that capture most of the variability in the data.
    Mutual Information: This method measures the dependence between two variables, such as the target variable and each feature. It selects the features that have the highest mutual information with the target variable.
    Lasso Regression: This method performs both feature selection and regularization by adding a penalty term to the objective function that encourages sparsity in the coefficients.''')
    try:
        with st.expander('RFECV'):
                my_subheader('Recursive Feature Elimination with Cross-Validation')
                # Scale the features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Set up the time series cross validation
                st.info(':information_source: The n_splits parameter in TimeSeriesSplit determines the number of splits to be made in the data. In other words, how many times the data is split into training and testing sets. It ensures that the testing set contains only data points that are more recent than the training set. The default value of n_splits is 5')
                timeseriessplit = st.slider('timeseries split', min_value=2, max_value=5, value=5)
                tscv = TimeSeriesSplit(n_splits=timeseriessplit)
                
                # Set up the linear regression model
                lr = LinearRegression()
                
                # Set up the recursive feature elimination with cross validation
                rfecv = RFECV(estimator=lr, step=1, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                
                # Fit the feature selection model
                rfecv.fit(X_scaled, y)
                
                # Print the selected features
                col1, col2, col3, col4 = st.columns([1,2,2,1])
                selected_features = X.columns[rfecv.support_]
                with col2:
                    st.write(':blue[**Selected features:**]', selected_features)
                
                # Print the feature rankings
                with col3: 
                    feature_rankings = pd.Series(rfecv.ranking_, index=X.columns).rename('Ranking')
                    st.write(':blue[**Feature rankings:**]')
                    st.write(feature_rankings.sort_values())
                    
                # Get the feature ranking
                feature_rankings = pd.Series(rfecv.ranking_, index=X.columns).rename('Ranking')
                
                # Create a dataframe with feature rankings and selected features
                df_ranking = pd.DataFrame({'Features': feature_rankings.index, 'Ranking': feature_rankings})
                
                # Highlight selected features
                df_ranking['Selected'] = np.where(df_ranking['Features'].isin(selected_features), 'Yes', 'No')
                
                # Create the plot
                fig = px.scatter(df_ranking, x='Features', y='Ranking', color='Selected', hover_data=['Ranking'])
                fig.update_layout(
                    title={
                        'text': 'Recursive Feature Elimination with Cross-Validation (RFECV)',
                        'x': 0.5,
                        'y': 0.95,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    xaxis_title='Features',
                    yaxis_title='Ranking',
                    legend_title='Selected'
                )
                
                # Show the plot
                st.plotly_chart(fig)
        # PCA feature selection
        pca = PCA(n_components=5)
        pca.fit(X)
        X_pca = pca.transform(X)
        selected_features_pca = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
        feature_names = X.columns
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pca.explained_variance_ratio_, y=feature_names[pca.components_.argmax(axis=1)], 
                             orientation='h', text=np.round(pca.explained_variance_ratio_ * 100, 2), textposition='auto'))
        fig.update_layout(title='PCA Feature Selection', xaxis_title='Explained Variance Ratio', yaxis_title='Feature Name')
        # Display plot in Streamlit
        st.plotly_chart(fig)
        # show user info about how to interpret the graph
        st.info('''When you fit a PCA model, it calculates the amount of variance that is captured by each principal component. \
                The variance ratio is the fraction of the total variance in the data that is explained by each principal component. \
                The sum of the variance ratios of all the principal components equals 1. 
                The variance ratio is expressed as a percentage by multiplying it by 100, so it can be easily interpreted. \
                For example, a variance ratio of 0.75 means that 75% of the total variance in the data is captured by the corresponding principal component.''')
    except:
        st.write(':red[Error: Recursive Feature Elimination with Cross-Validation could not execute...please adjust your selection criteria]')
    
    # Add slider to select number of top features
    st.sidebar.info('Select number of top features:')
    num_features = st.sidebar.slider("**value**", min_value=1, max_value=len(X.columns), value=len(X.columns), step=1, label_visibility="collapsed")
    
    # Mutual information feature selection
    mutual_info = mutual_info_classif(X, y, random_state=42)
    selected_features_mi = X.columns[np.argsort(mutual_info)[::-1]][:num_features]
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mutual_info[np.argsort(mutual_info)[::-1]][:num_features],
                         y=selected_features_mi, 
                         orientation='h',
                         text=[f'{val:.2f}' for val in mutual_info[np.argsort(mutual_info)[::-1]][:num_features]],
                         textposition='inside'))
    fig.update_layout(title=f'Top {num_features} Mutual Information Feature Selection')
    
    # Display plot in Streamlit
    st.plotly_chart(fig)
    
    #feature_selection_user = st.multiselect("favorite features", list(selected_features_mi))
    ##############################################################
    # SELECT YOUR FAVORITE FEATURES TO INCLUDE IN MODELING
    ##############################################################
    selected_cols = ["total_traffic"] + list(selected_features_mi)
    st.info(f'the columns you selected are: {selected_cols}')
    df_filtered = local_df[selected_cols].copy(deep=True)
    st.write('df')
    st.write(df)
    st.write(df.dtypes)

    # CHANGE DATAFRAME TO ONLY FILTERED COLUMNS
    def convert_uint8_to_bool(df):
        uint8_cols = df.select_dtypes(include="uint8").columns
        bool_cols = {col: bool for col in uint8_cols}
        return df.astype(bool_cols)
    df_filtered = convert_uint8_to_bool(df_filtered)
    st.write('df filtered')
    st.write(df_filtered)
    st.write(df_filtered.dtypes)
    
    X = df_filtered.iloc[:, 1:]
    y = df_filtered.iloc[:, 0:1]
    X_train = df_filtered.iloc[:, 1:][:(len(df_filtered)-my_insample_forecast_steps)]
    X_test = df_filtered.iloc[:, 1:][(len(df_filtered)-my_insample_forecast_steps):]
    # set endogenous variable train/test split
    y_train = df_filtered.iloc[:, 0:1][:(len(df_filtered)-my_insample_forecast_steps)]
    y_test = df_filtered.iloc[:, 0:1][(len(df_filtered)-my_insample_forecast_steps):]
    
    ###############################################################################
    # 5. Evaluate Model Performance
    ###############################################################################
    my_title("5. Evaluate Model Performance 🔎", "#0072B2")
    with st.sidebar:
        my_title("Evaluate Model Performance 🔎", "#0072B2")
    
    #***********************************
    # Benchmark Linear Regression Model
    #***********************************
    # Evaluate the insample test-set performance linear regression model
    y_pred, mse, rmse, r2 = evaluate_linear_regression_model(X_train, y_train, X_test, y_test)
    # Create dataframe for insample predictions versus actual
    df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
    # set the index to just the date portion of the datetime index
    df_preds.index = df_preds.index.date
    # Calculate percentage difference between actual and predicted values and add it as a new column
    df_preds = df_preds.assign(Percentage_Diff = ((df_preds['Predicted'] - df_preds['Actual']) / df_preds['Actual']))
    
    # Calculate MAPE and add it as a new column
    df_preds = df_preds.assign(MAPE = abs(df_preds['Percentage_Diff']))
    # Calculate overall MAPE
    mape = df_preds['MAPE'].mean()
    
    # put all metrics and graph in expander for linear regression e.g. benchmark model
    with st.expander(':information_source: LRM ', expanded=True):
        st.markdown('<h2 style="text-align:center"> Linear Regression </h2></p>', unsafe_allow_html=True)
        # show metrics
        col0, col1, col2, col3, col4 = st.columns([2, 3, 3, 3, 1])
        # Display the evaluation metrics
        with col1: 
            st.metric(':red[**MAPE:**]', delta=None, value = "{:.2%}".format(mape))
        with col2:
            st.metric(':red[**RMSE:**]', delta = None, value = round(rmse,2))
        with col3: 
            st.metric(':green[**R-squared:**]',  delta=None, value= round(r2, 2))
        
        # show model linear regression result metrics MAPE in sidebar
        model_name = "Linear Regression"
        #mape = mape
        session_state["df"] = update_dataframe(session_state["df"], model_name, mape)
        # Display the updated table in Streamlit
        with st.sidebar:
            # Display the updated table in Streamlit
            st.dataframe(session_state["df"])

        ################################################################
        # Create the line chart with Plotly
        ################################################################
        # Plot the predictions versus actual results
        # Combine X_test, y_test and y_pred into a single DataFrame
        
        df_X_test = X_test.copy()
        df_X_test['Actual'] = y_test
        df_X_test['Insample Prediction'] = y_pred
        #???? has independent variables next to actual and insample columns added
        # check to optimize
        #st.write(df_X_test)
        #????
        # Define the color palette
        colors = ['#5276A7', '#60B49F']
        # Create the figure with easy on eyes colors
        fig = px.line(df_X_test, x=X_test.index, y=['Actual', 'Insample Prediction'],
                      color_discrete_sequence=colors)
        # Update the layout of the figure
        fig.update_layout(
            title='',
            xaxis_title='',
            yaxis_title='',
            legend_title='Legend',
            font=dict(family='Arial', size=12, color='#707070'),
            yaxis=dict(gridcolor='#E1E1E1'),
            xaxis=dict(gridcolor='#E1E1E1'))
        # Set the line colors
        for i in range(len(colors)):
            fig.data[i].line.color = colors[i]
        fig.update_layout(yaxis=dict(gridcolor='#E1E1E1', range=[0, max(df_X_test['Actual'])]),
                          xaxis=dict(gridcolor='#E1E1E1'),
                          legend=dict(yanchor="bottom",
                                      y=0.0,
                                      xanchor="center",
                                      x=0.99))
        # Render the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)     
        # show the dataframe
        st.dataframe(df_preds.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Percentage_Diff': '{:.2%}', 'MAPE': '{:.2%}'}), use_container_width=True)
        
        #####################################################################
        # DOWNLOAD BUTTON
        #####################################################################
        # create download button for forecast results to .csv
        csv = convert_df(df_preds)
        col1, col2, col3 = st.columns([2,2,2])
        with col2: 
            st.download_button(":arrow_down: Download",
                               csv,
                               "forecast_linear_regression_model.csv",
                               "text/csv",
                               key='download-csv_lrm',
                               help = "Download your in-sample forecast to .CSV file")

if uploaded_file is not None:    
    #col1, col2, col3, col4 = st.columns([1,5,5,1])
    with st.form('model_train_form'):
        # set title
        my_header('''Add Models to Evaluation''')
        # set vertical spacers
        col1, col2, col3 = st.columns([4,2,4])
        # define all models you want user to choose from
        models = [('ARIMAX', None)]
        # create a checkbox in column 2 
        with col2:
            selected_models = [model_name for model_name, _ in models if st.checkbox(model_name)]
        param_grid = {
                        'ARIMAX': {
                            'order': [(1, 1, 1), (2, 1, 1)],
                            'seasonal_order': [(1, 0, 0, 12), (0, 1, 0, 12)]
                        },
                        'Prophet': {
                            'changepoint_prior_scale': [0.001, 0.01, 0.1],
                            'seasonality_prior_scale': [0.01, 0.1, 1.0]
                        }
                    }
        
        # create a subset of hyperparameters to search over
        param_subset = {model_name: {k: param_grid[model_name][k][:2] for k in param_grid[model_name]} for model_name in selected_models}

        st.info(':information_source: This will require time to train... Click the **"Submit"** button and grab coffee ☕ or tea 🍵')
        with col2: 
            #train_models_btn = st.button('Train selected models', type="primary")
            train_models_btn = st.form_submit_button("Submit")
            # the code block to train the selected models will only be executed if both the button has been clicked and the list of selected models is not empty.
            selected_models = st.session_state.get("selected_models", [])
    if train_models_btn==False:
        pass
    elif train_models_btn==True:
        # APPLY MY CUSTOM FUNCTION
        with st.expander(':information_source: ARIMAX', expanded=True):
            my_df, preds = train_my_models(y_train, X_train, X_test, y_test, selected_models)
            st.dataframe(my_df.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Percentage_Diff': '{:.2%}', 'MAPE': '{:.2%}'}), use_container_width=True)
            my_download_button(my_df, my_file="forecast_arima_model.csv")
            # Update the dataframe with the new results
            mape, mse, rmse, r2 = my_metrics(my_df)

        # show model ARIMAX result metrics MAPE in sidebar
        #if st.button('test'):
        with st.sidebar:
            model_name = "ARIMAX"
            #mape = mape
            session_state["df"] = update_dataframe(session_state["df"], model_name, mape)
        #else:
        #    pass  
# =============================================================================
#         with st.sidebar:
#             element.dataframe(df_metrics)
# =============================================================================
    else:
        # if button to train models is not clicked -> do nothing
        pass
else:
    # if no .csv is uploaded -> do nothing
    pass


##############################################################################
# 6. Forecast
##############################################################################
if uploaded_file is not None:
    my_title('6. Forecast 🔮', "#48466D")   
    with st.sidebar:
        my_title('Forecast 🔮', "#48466D")                    
        with st.expander("📆 ", expanded=True):
            #my_header("Set Date Range Forecast 📆")
            # set start date
            start_date_calendar = df['date'].min()
            end_date_calendar = df['date'].max()
            # set end date 
            max_date = df['date'].max()
            # define maximum value in dataset for year, month day
            year = max_date.year
            month = max_date.month
            day = max_date.day
            start_date_calendar = df['date'].min()
            end_date_calendar = df['date'].max()
            
            
            col1, col2 = st.columns(2)
            with col1: 
                st.write(" ")
                st.write(" ")
                st.markdown(f'<h4 style="color: #48466D; background-color: #F0F2F6; padding: 12px; border-radius: 5px;"><center> Select End Date:</center></h4>', unsafe_allow_html=True)
            with col2:
                end_date_calendar = st.date_input("", (datetime.date(year, month, day) + datetime.timedelta(days=90)))