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
import plotly.express as px
from sklearn.linear_model import LinearRegression
from pmdarima.arima import auto_arima
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error, r2_score
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import(
   AbstractHolidayCalendar, Holiday, DateOffset, \
   SU, MO, TU, WE, TH, FR, SA, \
   next_monday, nearest_workday, sunday_to_monday,
   EasterMonday, GoodFriday, Easter)
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense

###############################################################################
# Create Left-Sidebar Streamlit App
###############################################################################
with st.sidebar:
    st.markdown('''## :blue[1. Load Dataset :rocket:]''')
    # let user upload a file
    uploaded_file = st.file_uploader("")
    # if nothing is uploaded yet by user run below code
    if uploaded_file is None:
        # inform user what template to upload
        st.info('''☝️ **Please upload a .CSV file with:**  
                 - first column `date` in format **<mm-dd-yyy>**  
                 - second column `y`''')

# set the title of page
st.title(":blue[Streamlit - Forecast TimeSeries Data]")
with st.expander(':information_source: About', expanded=True):
    st.info('''Hi :wave: **Welcome** to my **Streamlit** app :smiley:
            
                \n**What does it do?**  
                - Forecast your Time Series Dataset - for example predict your future daily website `visits`!  
                \n**What do you need?**  
                - A .csv file that contains your **Date** (X) column and your target variable of interest (Y)
            ''')
    # to get image centered, create 3 equally spaced columns
    col1, col2, col3 = st.columns(3)
    # in middle column show image
    with col2:
        st.image('./images/logo_dark.png', caption="Created by Tony Hollaar")    
        #st.markdown("<h2 style='text-align: center; color: black;'>About</h2>", unsafe_allow_html=True)
        st.markdown('<p style="text-align:center"> <a href="https://www.tonyhollaar.com/" >www.tonyhollaar.com </a></p>', unsafe_allow_html=True) 
###############################################################################
# 1. Create Button to Load Dataset (.CSV format)
###############################################################################
# if user uploaded csv file run below code
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['date'])
    # only keep date and total_traffic
    df_graph = df[['date', 'total_traffic']].copy(deep=True)
    df_total = df[['date', 'total_traffic']].copy(deep=True)
    # if loaded dataframe succesfully run below
    ## set variables
    df_min = df['date'].min().date()
    df_max = df['date'].max().date()
    ## show information to user
    st.markdown(f'The ***dataframe*** :card_index_dividers: has :green[**{df.shape[0]}**] rows and :green[**{df.shape[1]}**] columns with date range: **`{df_min}`** to **`{df_max}`**.')
    ## BUTTON
    # create empty placeholder
    with st.expander(':arrow_down:', expanded=True ):
        col1, col2 = st.columns([2,3])
        with col1:    
            # show dataframe
            st.dataframe(df)
        with col2:
            st.caption(f'Website Traffic between {df_min} and {df_max} :chart_with_upwards_trend:')
            ## create graph
            df_graph = df_graph.set_index('date')
            ## display graph
            st.line_chart(df_graph)
            
    ###############################################################################
    # 2. Feature Engineering
    ###############################################################################
    st.markdown(''' ## :blue[2. Feature Engineering :toolbox:]''')
    st.markdown(''' ### ***Set Date Range Forecast*** :calendar:''')
    
    # set start date
    start_date_calendar = df['date'].min()
    # set end date 
    max_date = df['date'].max()
    # define maximum value in dataset for year, month day
    year = max_date.year
    month = max_date.month
    day = max_date.day
    # create variable to store user chosen end date to forecast future
    end_date_calendar = st.date_input(":blue[**Please select your forecast end date.**:arrow_down: Default value below set to +90 days]",
                                      (datetime.date(year, month, day) + datetime.timedelta(days=90)))
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

    st.markdown(':blue[**Pick your special days to include: :gift:**]')
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
        pay_days = st.checkbox('Monthly Pay Days (4th Friday of month)', value=True)
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
    
    ##############################
    # Add Day/Month/Year Features
    ##############################
    # create checkboxes for user to checkmark if to include features
    st.markdown(':blue[**Pick your seasonal days to include:**]')
    year_dummies = st.checkbox('Year Dummy Variables', value=True)
    month_dummies = st.checkbox('Month Dummy Variables', value=True)
    day_dummies = st.checkbox('Day Dummy Variables', value=True)
    ## YEAR
    if year_dummies:
        df['year'] = df['date'].dt.year
        dum_year =  pd.get_dummies(df['year'], columns = ['year'], drop_first=False, prefix='year', prefix_sep='_')
        df = pd.concat([df, dum_year], axis=1)   
    ## MONTH
    if month_dummies:
        month_dict = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
        df['month'] = df['date'].dt.month.apply(lambda x:month_dict.get(x))
        dum_month =  pd.get_dummies(df['month'], columns = ['month'], drop_first=False, prefix='', prefix_sep='')
        df = pd.concat([df, dum_month], axis=1)                            
    # DAY
    # date.weekday() - Return the day of the week as an integer, where Monday is 0 and Sunday is 6
    # convert 0 - 6 to weekday names
    if day_dummies:
        week_dict = {0:'Monday', 1:'Tuesday',2:'Wednesday', 3:'Thursday', 4: 'Friday', 5:'Saturday', 6:'Sunday'}
        df['day'] = df['date'].dt.weekday.apply(lambda x: week_dict.get(x))
        # get dummies
        dum_day = pd.get_dummies(df['day'], columns = ['day'], drop_first=False, prefix='', prefix_sep='')
        df = pd.concat([df, dum_day], axis=1)    
    
    # SHOW DATAFRAME
    st.dataframe(df)
    
    ###############################################################################
    # 3. Prepare Data (split into training / test)
    ###############################################################################
    st.markdown(''' ## :blue[3. Prepare Data (split into Train/Test):test_tube:]''')
    # 5.1) set date as index/ create local_df
    # create copy of dataframe not altering original
    local_df = df.copy(deep=True)
    # set the date as the index of the pandas dataframe
    local_df.index = pd.to_datetime(local_df['date'])
    local_df.drop(columns='date', inplace=True)
    local_df = local_df[['total_traffic', 'holiday', 'calendar_event', 'pay_day']]
    # USER TO SET the insample forecast days 
    my_insample_forecast_steps = st.slider(':blue[Select how many days of your current data you want to use as test-set]', 
                                           min_value = 1, 
                                           max_value = 180, 
                                           value = 7,
                                           step = 1)
    
    my_insample_forecast_steps = my_insample_forecast_steps 
    perc_test_set = "{:.2f}%".format((my_insample_forecast_steps/len(df))*100)
    st.write(f"the size of the  test set equals: :green[**{perc_test_set}**] of the total dataset.")
    # test without total traffic spent
    X_train = local_df.iloc[:, 1:][:(len(df)-my_insample_forecast_steps)]
    X_test = local_df.iloc[:, 1:][(len(df)-my_insample_forecast_steps):]
    #st.write(f'exog_train: `{len(X_train)}` samples')
    #st.write(f'exog_test: `{len(X_test)}` samples')
    #st.write('total length:',len(X_train)+len(X_test))
    
    # set endogenous variable train/test split
    y_train = local_df.iloc[:, 0:1][:(len(df)-my_insample_forecast_steps)]
    y_test = local_df.iloc[:, 0:1][(len(df)-my_insample_forecast_steps):]
    #st.write('y_train:',len(y_train))
    #st.write('y_test:', len(y_test))
    #st.write('total length:', len(y_train)+len(y_test))
    
    ###############################################################################
    # 5. Forecast
    ###############################################################################
    st.markdown(''' ## :blue[5. Evaluate Model Performance Metric :test_tube:]''')
    
    ###############################################################################
    # Benchmark Linear Regression Model
    ###############################################################################

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
    
    # Evaluate the linear regression model
    y_pred, mse, rmse, r2 = evaluate_linear_regression_model(X_train, y_train, X_test, y_test)
    # Create dataframe for insample predictions versus actual
    
    df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
    # set the index to just the date portion of the datetime index
    df_preds.index = df_preds.index.date
    # Calculate percentage difference between actual and predicted values and add it as a new column
    df_preds = df_preds.assign(Percentage_Diff = ((df_preds['Predicted'] - df_preds['Actual']) / df_preds['Actual']))
    
    # Display the evaluation metrics
    st.write('Root Mean Squared Error:', round(rmse,2))
    st.write('R-squared:', round(r2, 2))
        
    # show the predictions versus actual results
    st.write(df_preds.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Percentage_Diff': '{:.2%}'}))
    
    # Plot the predictions versus actual results
    # source: https://sengul-krdrl.medium.com/developing-prediction-app-with-streamlit-2b8e4f8abfef
# =============================================================================
#     st.caption('Linear Regression Insample prediction versus actual')
#     fig = plt.figure(figsize=(20,10))
#     plt.plot(X_test.index, y_test, label = 'Actual Traffic')
#     plt.plot(X_test.index, y_pred, label = 'Insample Prediction Traffic')
#     plt.legend()
#     st.pyplot(fig)
# =============================================================================

    # Combine X_test, y_test and y_pred into a single DataFrame
    df = X_test.copy()
    df['Actual Traffic'] = y_test
    df['Insample Prediction Traffic'] = y_pred

    ################################
    # Create the line chart with Plotly
    
    # Define the color palette
    colors = ['#5276A7', '#60B49F']
    # Create the figure with easy on eyes colors
    fig = px.line(df, x=X_test.index, y=['Actual Traffic', 'Insample Prediction Traffic'],
                  color_discrete_sequence=colors)
    # Update the layout of the figure
    fig.update_layout(
        title='Linear Regression Insample prediction versus actual',
        xaxis_title=None,
        yaxis_title=None,
        legend_title='Legend',
        font=dict(family='Arial', size=12, color='#707070'),
        #plot_bgcolor='#F2F2F2',
        #paper_bgcolor='#F2F2F2',
        yaxis=dict(gridcolor='#E1E1E1'),
        xaxis=dict(gridcolor='#E1E1E1'))
    # Set the line colors
    for i in range(len(colors)):
        fig.data[i].line.color = colors[i]
    # Render the chart in Streamlit
    st.plotly_chart(fig)
      
    ##################################
    # TESTING CHATGPT RESPONSE
    ##################################  
    ### ADD MY OWN FUNCTION
    def create_df_pred(y_test, y_pred, show_df_streamlit=True):
        df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
        # set the index to just the date portion of the datetime index
        df_preds.index = df_preds.index.date
        # Calculate percentage difference between actual and predicted values and add it as a new column
        df_preds = df_preds.assign(Percentage_Diff = ((df_preds['Predicted'] - df_preds['Actual']) / df_preds['Actual']))
        # show the predictions versus actual results
        my_df = st.write(df_preds.style.format({'Actual': '{:.2f}', 'Predicted': '{:.2f}', 'Percentage_Diff': '{:.2%}'}))
        # Calculate MAPE and add it as a new column
        df_preds = df_preds.assign(MAPE = abs(df_preds['Percentage_Diff']))
        if show_df_streamlit == True:
            return my_df
        else:
           return df_preds
    
    # define the models to be evaluated
    models = [#('Linear Regression', LinearRegression()),
              ('ARIMAX', None),
              #('LSTM', None)
              ]
    
    # define the hyperparameters to be tuned for linear regression and ARIMAX models
    # define the hyperparameters to be tuned for linear regression and ARIMAX models
    param_grid = {#'Linear Regression': [{'normalize': [True, False]}],
                  'ARIMAX': [{'order': 
                              [(1, 1, 0), 
                              (2, 1, 0), 
                              (3, 1, 0)], 
                              'seasonal_order': 
                              [(0, 0, 0, 0), 
                              (1, 0, 0, 12)]}]}
    # generate all possible combinations of hyperparameters
    params = list(ParameterGrid(param_grid))
   
    # evaluate each model using k-fold cross-validation
    for name, model in models:
        st.write(name)
        if name == 'ARIMAX':
            # find the best ARIMA model using grid search
            arimax_model = auto_arima(y_train, X = X_train, seasonal=True, m = 12, suppress_warnings=True)
            preds = arimax_model.predict(n_periods = len(X_test), X = X_test)
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            my_df = create_df_pred(y_test, preds, False)
            # Calculate overall MAPE
            mape = my_df['MAPE'].mean()
            st.write('RMSE ARIMAX', rmse)
            # format to two decimals and multiply by 100 to get percentage
            st.write('MAPE ARIMAX:', ":green[{:.2f}%]".format(mape * 100))
            # plot the prediction
            st.line_chart(my_df.iloc[:, :2])
