# AdEase - Time Series Forecasting Models
Time series forecasting methods such as ARIMA, SARIMA, SARIMAX, and Facebook Prophet to model trend seasonality and external factors. The project includes exploratory data analysis, performance evaluation, and visualizations of the wikipedia page views for Ad optimization.

### **About AdEase**
Ad Ease is an ads and marketing based company helping businesses elicit maximum clicks @ minimum cost. AdEase is an ad infrastructure to help businesses promote themselves easily, effectively, and economically. The interplay of 3 AI modules - Design, Dispense, and Decipher, come together to make it this an end-to-end 3 step process digital advertising solution for all.
<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTSGGqB27Li3IFoh_JIwopSTnnJmbzPz_a6lQ&s" alt="adease_img" width="400"/>

## Business Problem Statement:
You are working in the Data Science team of Ad ease trying to understand the per page view report for different wikipedia pages for 550 days, and forecasting the number of views so that you can predict and optimize the ad placement for your clients. Your clients belong to different regions and need data on how their ads will perform on pages in different languages.

You are provided with the data of 145k wikipedia pages and daily view count for each of them.

**Concepts Used:**

- Exploratory data analysis
- Time Series forecasting- ARIMA, SARIMA, SARIMAX
- Time Series forecasting- Prophet
- Hyperparameter tuning - Auto Arima, Grid Search

### Dataset

Datalink: https://drive.google.com/drive/folders/1mdgQscjqnCtdg7LGItomyK0abN6lcHBb


**Data Dictionary:**
There are two csv files given
1. train_1.csv: In the csv file, each row corresponds to a particular article and each column corresponds to a particular date. The values are the number of visits on that date.

  The page name contains data in this format:

  SPECIFIC NAME _ LANGUAGE.wikipedia.org _ ACCESS TYPE _ ACCESS ORIGIN

  having information about the page name, the main domain, the device type used to access the page, and also the request origin(spider or browser agent)

2. Exog_Campaign_eng: This file contains data for the dates which had a campaign or significant event that could affect the views for that day. The data is just for pages in English.

  There’s 1 for dates with campaigns and 0 for remaining dates. It is to be treated as an exogenous variable for models when training and forecasting data for pages in English 

### Data Visualization Insights:
 - There are total seven different languages and few unkown wikipedia pages view data, among which english wiki pages are prominent.
 - Most of the wiki pages are accessible on all platforms, while most of the data gathering is done by agents.
 - The null values/ no views on wiki pages reduces as time go by, trend reducing on a linear scale, meaning more engagement in future.
 - English - average daily views: 3767.33,  dominates the platform with the **highest traffic**, likely due to its global reach and the vast number of English-speaking users worldwide, has high-potential for global advertising.
 - Global average (excluding English) daily views across all other languages is ~860.17.

  * **Spanish (`es`)**: Average daily views: **1262.72**
    → Spanish comes second, indicating a **strong user base**

**Stationary Time Series** 
- **First-order** differencing (d=1) to make time series stationary, confirmed by statistical tests and visual inspection (ADF test, residuals plot).
- When applying SARIMA or SARIMAX, using seasonal differencing (D=1) with the  seasonal period (7 for weekly patterns) improved the model’s performance by capturing recurring patterns over time.

### Model Performances:
1. **ARIMA (AutoRegressive Integrated Moving Average)**

`model = SARIMAX(train, order=(4,1,3))`
  * **RMSE**: 682.68
  * **MAPE**: 0.0897

2. **SARIMA (Seasonal ARIMA)**

`model = SARIMAX(train, order=(0,1,0), seasonal_order=(3,0,1,7)`
  * **MAPE**: Approx. 0.06

3. **SARIMAX (SARIMA with eXogenous variables)**

`model = SARIMAX(train, order=(4,1,3), seasonal_order=(3,0,2,7), exog=exog_df.loc[train.index])`
  * **RMSE**: 304.71
  * **MAPE**: 0.0456

4. **Facebook Prophet**

`model = Prophet()`
  * **RMSE** : 544.49
  * **MAPE**: 0.0672 
