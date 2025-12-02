import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
from prophet.plot import plot_plotly, plot_components_plotly

# Load in dataframe
df24 = pd.read_csv('data/CO_2024_MA.csv')
df23 = pd.read_csv('data/CO_2023_MA.csv')
df22 = pd.read_csv('data/CO_2022_MA.csv')
df21 = pd.read_csv('data/CO_2021_MA.csv')
df19 = pd.read_csv('data/CO_2019_MA.csv')
df18 = pd.read_csv('data/CO_2018_MA.csv')
df17 = pd.read_csv('data/CO_2017_MA.csv')
df16 = pd.read_csv('data/CO_2016_MA.csv')
df_all = pd.concat([df24, df23, df22, df21, df19, df18, df17, df16], ignore_index=True)

# Rename Daily Max 8-hour CO Concentration
df_all.rename(columns={'Daily Max 8-hour CO Concentration': 'CO'}, inplace=True)

# Convert dates to datetime
df_all['Date'] = pd.to_datetime(df_all['Date'])

# Create column for seasons
def get_season(month):
    '''Returns the season for a given date'''
    if 3 <= month <= 5:
        return 'Spring'
    if 6 <= month <= 8:
        return 'Summer'
    if 9 <= month <= 11:
        return 'Fall'
    return 'Winter'
df_all['Season'] = df_all['Date'].dt.month.apply(get_season)

# Create column that signifies if it is a weekend
df_all['is_weekend'] = df_all['Date'].dt.dayofweek >= 5

# Create column that signifies if it is a federal holiday
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=df_all['Date'].min(), end=df_all['Date'].max())
df_all['is_holiday'] = df_all['Date'].isin(holidays)

# Determine which holidays create long weekends
holidays = pd.Series(holidays)

# Identify holidays that fall near or on weekends
weekend_expanding_holidays = holidays[(holidays.dt.dayofweek == 4) |
    (holidays.dt.dayofweek == 5) |
    (holidays.dt.dayofweek == 6) |
    (holidays.dt.dayofweek == 0)]

# Define long weekend days
df_all['is_long_weekend'] = (df_all['Date'].isin(weekend_expanding_holidays) |
    df_all['Date'].isin(weekend_expanding_holidays - pd.Timedelta(days=1)) |
    df_all['Date'].isin(weekend_expanding_holidays + pd.Timedelta(days=1)))

# Keep relevant columns only
df = df_all[['Date', 'Site ID', 'CO', 'Season', 'is_weekend',
          'is_holiday', 'is_long_weekend']]

#Rename Site ID
df.rename(columns={'Site ID': 'ID'}, inplace=True)

# Day of week and month for modeling
df['dayofweek'] = df['Date'].dt.dayofweek
df['month'] = df['Date'].dt.month
df['dayofyear'] = df['Date'].dt.dayofyear

# Exploratory Data Analysis
print(df.head())
print(df.info())

# Visualize distribution of CO levels
sns.histplot(df['CO'], bins=30, kde=True)
plt.title('Distribution of CO Levels')
plt.xlabel('CO Level')
plt.ylabel('Frequency')
plt.show()

# Visualize CO levels by Season
sns.boxplot(x='Season', y='CO', data=df)
plt.title('CO Levels by Season')
plt.xlabel('Season')
plt.ylabel('CO Level')
plt.show()

# Visualize CO levels on Weekends vs Weekdays
sns.boxplot(x='is_weekend', y='CO', data=df)
plt.title('CO Levels: Weekends vs Weekdays')
plt.xlabel('Is Weekend')
plt.ylabel('CO Level')
plt.show()

# Visualize CO levels on Long Weekends vs Regular Days
sns.boxplot(x='is_long_weekend', y='CO', data=df)
plt.title('CO Levels: Long Weekends vs Regular Days')
plt.xlabel('Is Long Weekend')
plt.ylabel('CO Level')
plt.show()

# Normalization and Cleaning
df = df[df['CO'] > 0].copy()
df['log_CO'] = np.log(df['CO'])

# Visualize distribution of log_CO levels
sns.histplot(df['log_CO'], bins=30, kde=True)
plt.title('Distribution of log_CO Levels')
plt.xlabel('log_CO Level')
plt.ylabel('Frequency')
plt.show()

# OLS Regression with Fixed Effects
model_fe = smf.ols("log_CO ~ is_weekend + is_holiday + is_long_weekend + C(ID) + C(month)",
    data=df).fit(cov_type='cluster', cov_kwds={'groups': df['ID']})
print(model_fe.summary2())

# Random Forest Prediction
df_rf = df.copy()
df_rf['Season'] = df_rf['Season'].astype('category').cat.codes
X = df_rf[['is_weekend','is_holiday','is_long_weekend','Season','month','dayofweek','dayofyear']]
y = df_rf['log_CO']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
rmse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"Random Forest RMSE: {rmse:.4f}")
print(f"Random Forest R²: {r2:.4f}")

# Feature importance
importances = rf.feature_importances_
for f, imp in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True):
    print(f"{f}: {imp:.3f}")

# Prophet model
df_prophet = df[['Date', 'log_CO']].rename(columns={'Date': 'ds', 'log_CO': 'y'})
holidays = pd.DataFrame({'holiday': 'us_federal_holiday',
    'ds': cal.holidays(start=df_prophet['ds'].min(), end=df_prophet['ds'].max()),
    'lower_window': 0,
    'upper_window': 1})
m = Prophet(holidays = holidays, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
m.fit(df_prophet)

# Make predictions
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Forecast plot
fig1 = m.plot(forecast)
plt.title("CO Forecast (log_CO)")
plt.show()

# Plot components
fig2 = m.plot_components(forecast)
plt.show()