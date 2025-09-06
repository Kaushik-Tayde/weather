

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("weather.csv")

print("First 5 records:")
print(df.head())


df['Date'] = pd.to_datetime(df['Date'])


print("\nMissing Values:\n", df.isnull().sum())


df['Temperature'] = df['Temperature'].fillna(df['Temperature'].mean())
df['Humidity'] = df['Humidity'].fillna(df['Humidity'].median())
df['WindSpeed'] = df['WindSpeed'].fillna(df['WindSpeed'].mean())
df['Rainfall'] = df['Rainfall'].fillna(0)  # Assume missing rainfall as 0

df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year


def season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else:
        return 'Autumn'

df['Season'] = df['Month'].apply(season)

print("\nBasic Statistics:\n", df.describe())

print("\nAverage Temperature by Month:")
print(df.groupby('Month')['Temperature'].mean())

print("\nTotal Rainfall by Year:")
print(df.groupby('Year')['Rainfall'].sum())

plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Temperature'], color='blue')
plt.title("Daily Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x='Month', y='Temperature', data=df, ci=None, palette="coolwarm")
plt.title("Average Monthly Temperature")
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(df['Rainfall'], bins=20, kde=True, color='skyblue')
plt.title("Rainfall Distribution")
plt.xlabel("Rainfall (mm)")
plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(df[['Temperature','Humidity','WindSpeed','Rainfall']].corr(),
            annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Weather Parameters")
plt.show()


plt.figure(figsize=(8,5))
sns.boxplot(x='Season', y='Temperature', data=df, palette="Set2")
plt.title("Season-wise Temperature Distribution")
plt.show()

df.to_csv("weather_analysis_output.csv", index=False)
print("\nAnalysis complete! Results saved to 'weather_analysis_output.csv'")

