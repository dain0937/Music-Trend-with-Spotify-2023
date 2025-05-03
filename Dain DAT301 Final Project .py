#!/usr/bin/env python
# coding: utf-8

# # Project 2
# ## Dain Lee
# ### Feb. 25th. 2024

# ## Background and Problem Setting Introduction
# 
# The dataset *Spotify-2023* offers an extensive compilation of the most prominent songs of 2023, providing insights into the ever-evolving landscape of music consumption and popularity trends on streaming platforms. With detailed attributes including track names, artist information, streaming statistics, and platform presence, it serves as a valuable resource for exploring the dynamics of the modern music industry.
# 
# ### Key Explorations
# 
# - <u>**Popularity Trends**</u>: Uncover patterns in song popularity over time and identify factors contributing to a song's success on streaming platforms.
# 
# - <u>**Artist and Genre Dynamics**</u>: Investigate the relationships between artists, genres, and audience preferences to understand musical collaboration and genre evolution.
# 
# - <u>**Audio Feature Analysis**</u>: Explore the audio features of songs, including beats per minute, danceability, and energy level, to understand musical composition and style preferences among listeners.

# ## Importing the Dataset and Libraries

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


spotify_2023_df = pd.read_csv('spotify-2023.csv')


# ## Data Cleaning and Standadrdizationm

# In[9]:


# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(spotify_2023_df.head())

# Check for missing values
print("Missing values:\n", spotify_2023_df.isnull().sum())

# Remove duplicates
spotify_2023_df = spotify_2023_df.drop_duplicates()

# Standardize column names (convert to lowercase and replace spaces with underscores)
spotify_2023_df.columns = spotify_2023_df.columns.str.strip().str.lower().str.replace(' ', '_')

# Check the standardized column names
print("Standardized column names:\n", spotify_2023_df.columns)

# Save the cleaned dataset to a new CSV file
spotify_2023_df.to_csv('cleaned_spotify_2023.csv', index=False)


#     It performs essential data preprocessing tasks on a DataFrame named spotify_2023_df. It first displays the initial rows of the dataset and then checks for any missing values, printing their sum for each column. Duplicate rows are removed from the dataset to ensure data integrity. Subsequently, the code standardizes the column names by converting them to lowercase and replacing spaces with underscores for consistency. It then confirms the standardized column names. Finally, the cleaned dataset is saved to a new CSV file named cleaned_spotify_2023.csv, excluding the index column. These operations ensure that the dataset is properly formatted and ready for subsequent analysis or modeling tasks.

# # Popularity Trends
# ## Top 10 Songs for Spotify 2023

# In[13]:


# Load the dataset into spotify_df
spotify_df = pd.read_csv('spotify-2023.csv')

# Convert 'streams' column to numeric
spotify_df['streams'] = pd.to_numeric(spotify_df['streams'], errors='coerce')

# Remove rows with NaN values in 'streams' column
spotify_df = spotify_df.dropna(subset=['streams'])

# Sort the dataset based on the popularity metric (total streams)
sorted_df = spotify_df.sort_values(by='streams', ascending=False)

# Print the top 10 songs based on total streams
top_10_df = sorted_df[['track_name', 'artist(s)_name', 'streams']].head(10)
top_10_df['Rank'] = range(1, 11)
print("Top 10 Songs Based on Total Streams:")
print(top_10_df[['Rank', 'track_name', 'artist(s)_name', 'streams']])

# Create a bar chart
plt.figure(figsize=(12, 8))
plt.bar(top_10_df['track_name'], top_10_df['streams'], color='thistle')

# Add labels and title
plt.xlabel('Songs')
plt.ylabel('Stream Counts')
plt.title('Top 10 Songs Based on Stream Counts')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show plot
plt.tight_layout()
plt.show()


#     This part analyzes the popularity trends of Spotify songs in 2023, focusing on the top 10 most streamed tracks. Initially, the code converts the 'streams' column to numeric values, handling any errors with coercion. It then removes rows with NaN values in the 'streams' column and sorts the dataset based on total streams. The top 10 songs, along with their respective artists and stream counts, are displayed and ranked. Additionally, the code generates a bar chart visualizing the stream counts of the top 10 songs, enhancing the understanding of their popularity. Notable tracks include "Blinding Lights" by The Weeknd and "Shape of You" by Ed Sheeran, illustrating the prevailing trends in Spotify streaming for the given year.

# ## 2023 Top 10 Songs' Released Date

# In[14]:


# Select the released_year and released_month columns for the top 10 songs
top_10_released_dates = spotify_df[['track_name', 'released_year', 'released_month']].head(10)

# Print the top 10 songs' released dates
print("Top 10 Songs' Released Dates:")
print(top_10_released_dates)


#     The code extracts and displays the release dates (year and month) of the top 10 songs from the Spotify dataset for 2023. It selects the 'track_name', 'released_year', and 'released_month' columns and prints them. The output shows a variety of tracks released throughout the year, such as "Seven (feat. Latto)" in July 2023 and "LALA" in March 2023, offering insights into the temporal distribution of popular songs.

# ## Factors Influencing Popularity

# In[21]:


# Correlation Matrix
corr_matrix = spotify_df.corr()

# Plotting Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Song Attributes and Popularity Metrics')
plt.show()


#     The correlation heatmap shows the relationships between song attributes and popularity metrics. Positive correlations suggest that certain attributes may contribute to a song's success, while negative correlations indicate factors that may have a lesser impact. 

# # Artists and Genre Dynamics
# ## Pie Chart of Top Artists

# In[19]:


# Group by 'artist(s)_name' and calculate the total streams for each artist
artist_streams = spotify_df.groupby('artist(s)_name')['streams'].sum()

# Sort the artists by total streams in descending order
sorted_artists = artist_streams.sort_values(ascending=False)

# Extract the top 10 artists
top_10_artists = sorted_artists.head(10)

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(top_10_artists, labels=top_10_artists.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
plt.title('Top Artists Based on Total Streams')
plt.show()


#     The visualization presents a snapshot of the most popular artists based on streaming data. Artists like The Weeknd, Taylor Swift, Ed Sheeran, and Harry Styles stand out, each representing a significant portion of the total streams. This insight underscores their relative popularity within the dataset, offering valuable perspective on listener preferences and trends in music consumption.

# ## Total Streams for Top Artists

# In[21]:


# Plotting the scatter plot for total streams of top artists
plt.figure(figsize=(10, 6))
plt.scatter(top_10_artists.values, top_10_artists.index, color='salmon', alpha=0.7)
plt.xlabel('Total Streams')
plt.ylabel('Artist')
plt.title('Total Streams for Top Artists (Scatter Plot)')
plt.tight_layout()
plt.show()


#     A scatter plot illustrating the total streams for top artists, specifically The Weeknd, Taylor Swift, and Ed Sheeran in order. The visualization portrays their streaming performance, displaying total streams on the x-axis and the respective artists on the y-axis.

# # Audio Feature Analysis
# ## Tempo, Danceability, and Energy Level Trend in 2023

# In[22]:


# Load the dataset containing audio features
spotify_audio_df = pd.read_csv('spotify-2023.csv')

# Visualize the distribution of audio features
plt.figure(figsize=(12, 8))

# Histograms for individual audio features
plt.subplot(3, 1, 1)
sns.histplot(spotify_audio_df['bpm'], bins=20, color='thistle', kde=True)
plt.xlabel('Beats Per Minute (BPM)')
plt.title('Distribution of Tempo')

plt.subplot(3, 1, 2)
sns.histplot(spotify_audio_df['danceability_%'], bins=20, color='salmon', kde=True)
plt.xlabel('Danceability (%)')
plt.title('Distribution of Danceability')

plt.subplot(3, 1, 3)
sns.histplot(spotify_audio_df['energy_%'], bins=20, color='lightsteelblue', kde=True)
plt.xlabel('Energy Level (%)')
plt.title('Distribution of Energy')

plt.tight_layout()
plt.show()

# Scatter plot matrix to visualize relationships between audio features
sns.pairplot(spotify_audio_df[['bpm', 'danceability_%', 'energy_%']])
plt.suptitle('Scatter Plot Matrix of Audio Features', y=1.02)
plt.show()


#     The code analyzes audio features focusing on tempo, danceability, and energy level trends in 2023. and visualizes the distribution of these features using histograms. Tempo peaks around 100 and 120 BPM with a slight right-skew, danceability peaks around 70% and 80% with a left-skewed distribution, and energy level peaks around 65% with a left-skewed distribution. Additionally, a scatter plot matrix illustrates the relationships between these features, providing insights into their correlations and patterns.

# ## Audio Feature Trends Over 2023: Monthly Analysis

# In[20]:


# Convert 'released_month' to datetime
spotify_df['released_month'] = pd.to_datetime(spotify_df['released_month'], format='%m')

# Group by released month and calculate the average values of audio features
monthly_avg_features = spotify_df.groupby('released_month').mean()

# Plotting
plt.figure(figsize=(12, 8))

# Loop through each audio and plot its average value over time
for feature in ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%']:
    plt.plot(monthly_avg_features.index.month_name(), monthly_avg_features[feature], label=feature)

# Add labels and title
plt.xlabel('Month')
plt.ylabel('Average Value')
plt.title('Average Audio Feature Values Over Time (2023)')
plt.legend()
plt.xticks(rotation=45, ha='right')

# Show plot
plt.tight_layout()
plt.show()


#     Throughout 2023, the analysis reveals a consistent pattern in audio feature trends. Danceability and energy exhibit peaks early in the year, followed by a slight decline. Valence maintains a steady level of popularity, while acousticness and instrumentalness remain relatively low. This pattern suggests a preference for energetic and danceable tracks, with consistent levels of positivity, while acoustic and instrumental elements take a backseat in popularity.

# # Conclusion

# - <u>**Insights into Popularity Trends**</u> : 
# * Tracks like "Blinding Lights" by The Weeknd and "Shape of You" by Ed Sheeran dominate with high stream counts, reflecting widespread popularity.
# * Diverse release dates of top songs showcase ongoing appeal across seasons.
# * Positive correlations between attributes like tempo, danceability, and energy level underscore their significance in driving listener engagement.
# 
# - <u>**Understanding Artist and Genre Dynamics**</u> : 
# * Artists like The Weeknd, Taylor Swift, Ed Sheeran, and Harry Styles emerge as frontrunners, representing significant portions of total streams.
# * Snapshot of their relative popularity within the dataset offers valuable perspective on listener preferences and trends in music consumption.
# 
# - <u>**Exploration of Audio Features**</u>: 
# * Analysis of tempo, danceability, and energy level trends provides insights into listener preferences.
# * Tempo peaks around 100 and 120 BPM, danceability at 70% to 80%, and energy levels around 65%, informing discussions on musical trends and audience tastes.
