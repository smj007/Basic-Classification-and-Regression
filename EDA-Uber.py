# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 06:00:54 2020

@author: saimi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE= 'uber-raw-data-sep14.csv/uber-raw-data-sep14.csv'
uber_data = pd.read_csv(DATA_FILE)
uber_data.head()


uber_data['Date/Time'] = pd.to_datetime(uber_data['Date/Time'], format="%m/%d/%Y %H:%M:%S")

uber_data['DayOfWeekNum'] = uber_data['Date/Time'].dt.dayofweek
uber_data['DayOfWeek'] = uber_data['Date/Time'].dt.day_name(locale = 'English')
uber_data['MonthDayNum'] = uber_data['Date/Time'].dt.day
uber_data['HourOfDay'] = uber_data['Date/Time'].dt.hour
uber_data['MinOfDay'] = uber_data['Date/Time'].dt.minute
uber_data.head()


weekday = uber_data.pivot_table(index = 'DayOfWeek',
                                values = 'Base',
                                aggfunc = 'count')


weekday.head()

weekdayAverage = weekday/30
weekdayAverage.head()

weekdayAverage.plot.bar(figsize = (20,10))
plt.ylabel('Average per day')
plt.title('Average rides per day vs Day of week')

hours = uber_data.pivot_table(index = 'HourOfDay',
                              values = 'Base',
                              aggfunc = 'count')
hours.head()

hourAverage = hours /30
hourAverage.plot.bar(figsize = (20,10))
plt.ylabel('Rides per hour')
plt.title('Rides vs Hour of the day')


minute = uber_data.pivot_table(index = 'MinOfDay',
                              values = 'Base',
                              aggfunc = 'count')
hours.head(50)

minAverage = minute/30

minAverage.plot.bar(figsize = (20,10))
plt.ylabel('Rides per min')
plt.title('Rides vs Min of the day')

minute.max()
minute.min()

print(hours)

print(minute)

print(weekday)
print(hourAverage)

print(minAverage)






