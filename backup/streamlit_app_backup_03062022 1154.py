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
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# source: https://splunktool.com/holiday-calendar-in-pandas-dataframe
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import(
   AbstractHolidayCalendar, Holiday, DateOffset, \
   SU, MO, TU, WE, TH, FR, SA, \
   next_monday, nearest_workday, sunday_to_monday,
   EasterMonday, GoodFriday, Easter)
    
###############################################################################
# Create Left-Sidebar Streamlit App
###############################################################################
with st.sidebar:
    st.markdown('''## **About**  ''')
    st.markdown('''
                Author: Tony Hollaar  
                Date: 03-31-2023  
                Website: www.tonyhollaar.com''')

# set the title of page
st.title(":blue[Streamlit - Forecast Website Traffic]")
st.write('''Hi :wave: **Welcome** to my **Streamlit** app to forecast
            website traffic e.g. future `visits`!        
        ''')

###############################################################################
# 1. Create Button to Load Dataset (.CSV format)
###############################################################################
st.markdown('''## :blue[1. Load Dataset :rocket:]''')
# let user upload a file
uploaded_file = st.file_uploader("")
# if nothing is uploaded yet by user run below code
if uploaded_file is None:
    # inform user what template to upload
    st.info('''☝️ **Please upload a .CSV file with:**  
             - first column `date` in format **<mm-dd-yyy>**  
             - second column `traffic`''')
             
# if user uploaded csv file run below code
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['date'])
    # only keep date and total_traffic
    df_graph = df[['date', 'total_traffic']].copy(deep=True)
    df_total = df[['date', 'total_traffic']].copy(deep=True)
    # if loaded dataframe succesfully run below
    ## set variables
    df_min = df['date'].min()
    df_max = df['date'].max()
    
    ## set titles
    st.markdown('''---''')
    ## show information to user
    st.write(f'The below ***dataframe*** :card_index_dividers: has :green[**{df.shape[0]}**] rows and :green[**{df.shape[1]}**] columns with date range: **`{df_min}`** to **`{df_max}`**.')
    st.write(df)
    ## create graph
    st.caption(f'Website Traffic between {df_min} and {df_max} :chart_with_upwards_trend:')
    df_graph = df_graph.set_index('date')
    st.line_chart(df_graph)
        
    ###############################################################################
    # 2. Feature Engineering
    ###############################################################################
    st.markdown(''' ## :blue[2. Feature Engineering :toolbox:]''')
    st.markdown('''---''')
    st.markdown(''' ### ***Set Date Range Forecast*** :calendar:''')
    
    # set start date
    start_date_calendar = df_min
    # set end date 
    end_date_calendar = st.date_input(":blue[**Please select your forecast end date**]",
                                      datetime.date(2024, 1, 1))
    #st.write('forecast date', end_date_calendar)
    st.markdown('''---''')
    st.markdown('### ***Special Calendar Days***')
    df_exogenous_vars = pd.DataFrame({'date': pd.date_range(start = start_date_calendar, 
                                                            end = end_date_calendar)})
    cal = calendar()
    holidays = cal.holidays(start=start_date_calendar, end=end_date_calendar)
    # holiday = true/ otherwise false
    df_exogenous_vars['holiday'] = (df_exogenous_vars['date'].isin(holidays)).apply(lambda x: 1 if x==True else 0)
    
    # function to get the holiday name from package calendar
    def my_holiday_name_func(my_date):
      holiday_name = cal.holidays(start = my_date, end = my_date, return_name = True)
      if len(holiday_name) < 1:
        holiday_name = ""
        return holiday_name
      else: 
       return holiday_name[0]
    
    # create column for holiday description
    df_exogenous_vars['holiday_desc'] = df_exogenous_vars['date'].apply(lambda x: my_holiday_name_func(x))     
    st.markdown(':blue[**Pick your seasonal days to include: :gift:**]')
    # create two columns
    col1, col2 = st.columns(2)
    ###############################################
    # create checkboxes for special days
    ###############################################
    with col1:
        jan_sales = st.checkbox('January Sale', value=True)
        val_day_lod = st.checkbox('Valentine\'s Day [last order date]', value=True)
        val_day = st.checkbox('Valentine\'s Day', value=True)
        mother_day_lod = st.checkbox('Mother\'s Day [last order date]', value=True)
        mother_day = st.checkbox('Mother\'s Day', value=True)
        father_day_lod = st.checkbox('Father\'s Day [last order date]', value=True)
    with col2:
        father_day = st.checkbox('Father\'s Day', value=True)
        black_friday_lod = st.checkbox('Black Friday [sale starts]', value=True)
        black_friday = st.checkbox('Black Friday', value=True)
        cyber_monday = st.checkbox('Cyber Monday', value=True)
        christmas_day = st.checkbox('Christmas Day [last order date]', value=True)
        boxing_day = st.checkbox('Boxing Day sale', value=True)
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
    st.markdown('''---''')
    st.markdown(':blue[**Pick your other special days to include:**]')
    pay_days = st.checkbox('Monthly Pay Days (4th Friday of month)', value=True)
    
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
    # Reorder Columns
    ###############################################################################
    df_exogenous_vars = df_exogenous_vars[['date', 'holiday', 'holiday_desc', 'calendar_event', 'calendar_event_desc', 'pay_day','pay_day_desc']]
    
    ###############################################################################
    #combine exogenous vars with df_total
    ###############################################################################
    df_total_incl_exogenous = pd.merge(df_total, df_exogenous_vars, on='date', how='left' )
    df = df_total_incl_exogenous.copy(deep=True)
    st.write(df)
    
    ###############################################################################
    # 3. Prepare Data (split into training / test)
    ###############################################################################
    st.markdown(''' ## :blue[3. Prepare Data (split into Train/Test):test_tube:]''')
    # take partial dataset
    df = df[(df['date'] >= '2021-04-01')].copy(deep=True)
    print('start date range df:', df['date'].min())
    print('end date range df:  ', df['date'].max())
    
    