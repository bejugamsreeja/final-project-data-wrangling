#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


# 1. Import the dataset using Pandas from above mentioned url.


# In[4]:


df=pd.read_csv("https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv")


# In[5]:


df


# In[6]:


# 2. High Level Data Understanding:
# a. Find no. of rows & columns in the dataset
# b. Data types of columns.
# c. Info & describe of data in dataframe.


# In[7]:


# a.
num_rows, num_columns = df.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")


# In[8]:


# b.
column_data_types = df.dtypes
print("Data types of columns:")
print(column_data_types)


# In[9]:


# c.
df.info()
df_description = df.describe()
print(df_description)


# In[10]:


# 3. Low Level Data Understanding :
# a. Find count of unique values in location column.
# b. Find which continent has maximum frequency using values counts.
# c. Find maximum & mean value in 'total_cases'.
# d. Find 25%,50% & 75% quartile value in 'total_deaths'.
# e. Find which continent has maximum 'human_development_index'.
# f. Find which continent has minimum 'gdp_per_capita'


# In[11]:


# a.
unique_location_count = df['location'].nunique()
print("Count of unique values in the 'location' column :", unique_location_count)


# In[12]:


# b.
continent_frequency = df['continent'].value_counts()
max_continent = continent_frequency.idxmax()
print(f"The continent with the maximum frequency is: {max_continent}")


# In[13]:


# c.
max_total_cases = df['total_cases'].max()
mean_total_cases = df['total_cases'].mean()
print(f"Maximum value in 'total_cases': {max_total_cases}")
print(f"Mean value in 'total_cases': {mean_total_cases}")


# In[14]:


# d.
quartiles_total_deaths = df['total_deaths'].quantile([0.25, 0.5, 0.75])
print("25th percentile (Q1):", quartiles_total_deaths[0.25])
print("50th percentile (Q2 or median):", quartiles_total_deaths[0.5])
print("75th percentile (Q3):", quartiles_total_deaths[0.75])


# In[15]:


# e.
maximum_HDI_continent = df.groupby('continent')['human_development_index'].idxmax()
continent_with_max_HDI = df.loc[maximum_HDI_continent]['continent'].values[0]
print(f"The continent with the maximum 'human_development_index' is: {continent_with_max_HDI}")


# In[16]:


# f.
min_gdp_continent = df.groupby('continent')['gdp_per_capita'].idxmin()
continent_with_min_gdp = df.loc[min_gdp_continent]['continent'].values[0]
print(f"The continent with the minimum 'gdp_per_capita' is: {continent_with_min_gdp}")


# In[17]:


# 4. Filter the dataframe with only this columns ['continent','location','date','total_cases','total_deaths','gdp_per_capita','human_development_index'] and update the data frame.
specific_columns = ['continent', 'location', 'date', 'total_cases', 'total_deaths', 'gdp_per_capita', 'human_development_index']
filtered_df = df[specific_columns]
filtered_df


# In[18]:


# 5. Data Cleaning
# a. Remove all duplicates observations
# b. Find missing values in all columns
# c. Remove all observations where continent column value is missing....... Tip : using subset parameter in dropna
# d. Fill all missing values with 0


# In[19]:


# a.
removing_duplicates = df.drop_duplicates()
removing_duplicates


# In[20]:


# b.
missing_values = df.isnull().sum()

print("Missing values in each column:")
missing_values


# In[21]:


# c.
df_cleaned = df.dropna(subset=['continent'])
df_cleaned


# In[22]:


# d.
df_filled = df.fillna(0)
df_filled


# In[23]:


# 6. Date time format :
# a. Convert date column in datetime format using pandas to_datetime
# b. Create new column month after extracting month data from date column.


# In[24]:


# a.
df['date'] = pd.to_datetime(df['date'])
df


# In[25]:


# b.
df['month'] = df['date'].dt.month
df


# In[26]:


# 7. Data Aggregation:
# a. Find max value in all columns using groupby function on 'continent' column....... Tip: use reset_index() after applying groupby
# b. Store the result in a new dataframe named 'df_groupby'.......(Use df_groupby dataframe for all further analysis)


# In[27]:


# a.
max_values = df.groupby('continent').max().reset_index()
max_values


# In[28]:


# b.
df_groupby = df.groupby('continent').max().reset_index()
df_groupby


# In[29]:


# 8. Feature Engineering :
# Create a new feature 'total_deaths_to_total_cases' by ratio of 'total_deaths' column to 'total_cases'
df['total_deaths_to_total_cases'] = df['total_deaths'] / df['total_cases']
df


# In[30]:


# 9. Data Visualization :
# a. Perform Univariate analysis on 'gdp_per_capita' column by plotting histogram using seaborn dist plot.
# b. Plot a scatter plot of 'total_cases' & 'gdp_per_capita'
# c. Plot Pairplot on df_groupby dataset.
# d. Plot a bar plot of 'continent' column with 'total_cases' ....Tip : using kind='bar' in seaborn catplot


# In[31]:


# a.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


gdp_per_capita_column = df['gdp_per_capita']


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(gdp_per_capita_column, kde=True, bins=30, color='skyblue')
plt.title('Histogram of GDP per Capita')
plt.xlabel('GDP per Capita')
plt.ylabel('Frequency')
plt.show()


# In[32]:


# b.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=df, x='total_cases', y='gdp_per_capita', color='blue', alpha=0.5)
plt.title('Scatter Plot of Total Cases vs GDP per Capita')
plt.xlabel('Total Cases')
plt.ylabel('GDP per Capita')
plt.show()


# In[1]:


# c.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df_groupby)
plt.suptitle("Pairplot of df_groupby Dataset", y=1.02)
plt.show()


# In[ ]:


# d.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(12, 6))
sns.catplot(x='continent', y='total_cases', kind='bar', data=df, height=6, aspect=2)
plt.title('Bar Plot of Total Cases by Continent')
plt.xlabel('Continent')
plt.ylabel('Total Cases')
plt.show()


# In[ ]:


# 10. Save the df_groupby dataframe in your local drive using pandas to_csv function .
df.to_csv('grouped_data.csv')

