# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 20:14:19 2023

@author: tholl
"""

 ###############################################################################
 # Reorder Columns
 ###############################################################################
 df_exogenous_vars = df_exogenous_vars[['date', 'holiday', 'holiday_desc', 'calendar_event', 'calendar_event_desc', 'pay_day','pay_day_desc']]
 
 #combine exogenous vars with df_total
 df_total_incl_exogenous = pd.concat([df_total, df_exogenous_vars], join='inner' )
 df = df_total_incl_exogenous.copy(deep=True)
 
 # check datatypes
 df.dtypes
 
 # Add Day/Month/Year Features
 df['year'] = df['year'].dt.year
 dum_year =  pd.get_dummies(df['year'], columns = ['year'], drop_first=False, prefix='year', prefix_sep='_')
 df = pd.concat([df, dum_year], axis=1)    
 
 # Add Month Dummy
 month_dict = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
 df['month'] = df['date'].dt.month.apply(lambda x:month_dict.get(x))
 dum_month =  pd.get_dummies(df['month'], columns = ['month'], drop_first=False, prefix='', prefix_sep='')
 df = pd.concat([df, dum_month], axis=1)                            
 
 # Add Weekday Dummy
 # date.weekday() - Return the day of the week as an integer, where Monday is 0 and Sunday is 6
 # convert 0 - 6 to weekday names
 week_dict = {0:'Monday', 1:'Tuesday',2:'Wednesday', 3:'Thursday', 4: 'Friday', 5:'Saturday', 6:'Sunday'}
 df['day'] = df['date'].dt.weekday.apply(lambda x: week_dict.get(x))
 
 # get dummies
 dum_day = pd.get_dummies(df['day'], columns = ['day'], drop_first=False, prefix='', prefix_sep='')
 df = pd.concat([df, dum_day], axis=1)
 
 ## check dataframe and dtypes
 #df.head()
 #df.dtypes