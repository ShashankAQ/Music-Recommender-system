import matplotlib
import sns
import streamlit
import streamlit as st
import pandas as pd
import numpy as np
from markdown_it.rules_core import inline
from matplotlib import pyplot as plt
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="c5628d3138c345f5862fcebc9ebecf4a", client_secret="a7a535d78b304399af308b20bc9f0a62"))
from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Add missing definition for 'song_cluster_pipeline'
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=20, verbose=False))
])
users = {
    "user1": "password1",
    "user2": "password2",
    # Add more users as needed
}

def authenticate(username, password):
    """Dummy authentication function."""
    return users.get(username) == password

# Load data
data = pd.read_csv('spotifynew.csv', encoding='ISO-8859-1')
df = pd.read_csv("spotifynew.csv", encoding="latin-1")


import matplotlib
import os
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
def musicanalysis():
    import streamlit as st
    st.title("Music Analysis")
    import pandas as pd
    data = pd.read_csv('spotifynew.csv', encoding='ISO-8859-1')
    df = pd.read_csv("spotifynew.csv", encoding="latin-1")
    pd.set_option('display.max_columns', None)
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    pd.set_option('display.max_columns', None)

    import warnings

    warnings.filterwarnings("ignore")
    categorical_columns = ['track_name', 'artist(s)_name', 'key', 'mode']
    for col in df.columns:
        if col not in categorical_columns and col not in ['streams']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    categorical_columns = ['track_name', 'artist(s)_name', 'key', 'mode']
    for col in df.columns:
        if col not in categorical_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Top 10 artists with most songs in the dataset
    top_artists = data['artist(s)_name'].value_counts().head(10)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_artists.values, y=top_artists.index, palette='viridis')
    plt.xlabel('Number of Songs')
    plt.ylabel('Artist(s) Name')
    top_artists = data['artist(s)_name'].value_counts().head(10)

    st.subheader('Top 10 Artists with Most Songs')
    st.bar_chart(top_artists)



    import pandas as pd
    top_artists_2023 = data.groupby('artist(s)_name')['streams'].sum().reset_index()

    # Convert the 'streams' column to numeric
    top_artists_2023['streams'] = pd.to_numeric(top_artists_2023['streams'], errors='coerce')

    # Handle NaN or non-numeric values in the 'streams' column
    top_artists_2023['streams'] = top_artists_2023['streams'] / 1e8  # Convert to 100 million

    # Set a global display format for floats
    pd.options.display.float_format = '{:,.1f} hundred million'.format

    # Sort the DataFrame after handling NaN or non-numeric values
    top_artists_2023 = top_artists_2023.sort_values(by='streams', ascending=False).head(10)

    st.subheader("Top Ten Artists of 2023")
    st.table(top_artists_2023[['artist(s)_name']])
    df['key'] = pd.Categorical(df['key']).codes
    df['mode'] = pd.Categorical(df['mode']).codes

    # Reset the global display format
    pd.options.display.float_format = None


    # Reset the global display format


    # Histogram: Streams Distribution
    fig_streams_dist = px.histogram(df, x='streams', nbins=50, marginal='box', title='Streams Distribution')
    fig_streams_dist.update_layout(xaxis_title='Streams', yaxis_title='Count')
    st.subheader("Streams Distribution")
    st.plotly_chart(fig_streams_dist)
    # Top 10 songs with most streams on Spotify
    top_spotify_streams = data[['track_name', 'artist(s)_name', 'streams']].sort_values(by='streams',
                                                                                        ascending=False).head(10)
    st.subheader("Top 10 Songs with Most Streams on Spotify")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot

    sns.barplot(x=top_spotify_streams['streams'], y=top_spotify_streams['track_name'], palette='viridis', ax=ax)

    # Customize the plot
    ax.set_xlabel('Streams (in billions)')
    ax.set_ylabel('Track Name')
    ax.set_title('Top 10 Songs with Most Streams on Spotify')
    ax.tick_params(axis='x', rotation=45)

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Top 10 songs with highest presence in Apple Music playlists
    top_apple_playlists = data[['track_name', 'artist(s)_name', 'in_apple_playlists']].sort_values(
        by='in_apple_playlists', ascending=False).head(10)

    st.subheader("Highest Presence in Apple Music Playlists")
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(x=top_apple_playlists['in_apple_playlists'], y=top_apple_playlists['track_name'], palette='viridis',
                ax=ax)
    ax.set_xlabel('Number of Apple Music Playlists')  # Fix: Use set_xlabel instead of set.xlabel
    ax.set_ylabel('Track Name')  # Fix: Use set_ylabel instead of set.ylabel
    ax.set_title('Top 10 Songs with Highest Presence in Apple Music Playlists')

    # Fix: Use xaxis.set_tick_params and yaxis.set_tick_params
    ax.xaxis.set_tick_params(rotation=45)
    ax.yaxis.set_tick_params(rotation=45)
    st.pyplot(fig)

    st.subheader("Distribution of Danceability")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(data['danceability_%'], bins=20, kde=True, color='purple', ax=ax)
    ax.set_xlabel('Danceability (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Danceability')

    st.pyplot(fig)

    st.subheader("Audio Features Comparison - Parallel Coordinates")
    fig_parallel = px.parallel_coordinates(df, dimensions=['danceability_%', 'valence_%', 'energy_%',
                                                           'acousticness_%', 'instrumentalness_%', 'liveness_%'],
                                           color='in_spotify_charts',
                                           title='Audio Features Comparison',
                                           width=1200)
    st.plotly_chart(fig_parallel)
    import pandas as pd
    import numpy as np
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import streamlit as st

    # Assuming 'data' is your DataFrame

    # Select columns for cross-platform metrics
    cross_platform_columns = [
        'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists',
        'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts'
    ]

    # Clean the data by removing commas and converting to numeric
    data[cross_platform_columns] = data[cross_platform_columns].replace(',', '', regex=True).astype(float)

    # Calculate the correlation matrix
    correlation_matrix = data[cross_platform_columns].corr()

    st.subheader('Correlation Heatmap: Cross-Platform Metrics')

    # Use Seaborn to create a heatmap directly in Streamlit
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5, ax=ax)
    st.pyplot(fig)
    import pandas as pd
    import plotly.express as px

    # Assuming 'df' is your DataFrame

    # Convert columns to numeric (if not already)
    df['energy_%'] = pd.to_numeric(df['energy_%'], errors='coerce')
    df['valence_%'] = pd.to_numeric(df['valence_%'], errors='coerce')
    df['danceability_%'] = pd.to_numeric(df['danceability_%'], errors='coerce')
    df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

    # Drop rows with NaN values in the selected columns
    df = df.dropna(subset=['energy_%', 'valence_%', 'danceability_%', 'streams'])

    # Convert categorical columns to codes
    df['key'] = pd.Categorical(df['key']).codes
    df['mode'] = pd.Categorical(df['mode']).codes
    st.subheader('Danceability vs. Valence vs. Energy (3D Scatter Plot)')

    # Create 3D scatter plot with organized axis labels
    fig_3d_scatter = px.scatter_3d(df, x='danceability_%', y='valence_%', z='energy_%', color='streams',
                                   size='streams', hover_name='track_name',
                                   title='Danceability vs. Valence vs. Energy')
    st.plotly_chart(fig_3d_scatter)
    audio_features_columns = ['danceability_%', 'energy_%', 'valence_%']
    popularity_column = 'streams'

    # Create scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, feature in zip(axes, audio_features_columns):
        sns.scatterplot(data=df, x=feature, y=popularity_column, color='blue', alpha=0.5, ax=ax)
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Number of Streams')
        ax.set_title(f'{feature.replace("_", " ").title()} vs. Number of Streams')


    plt.tight_layout()
    st.pyplot(fig)

    import plotly.express as px
    st.subheader('Popularity vs. Danceability')

    # Create scatter plot for Popularity vs. Danceability
    fig1 = px.scatter(data, x='danceability_%', y='streams', title='Popularity vs. Danceability')
    fig1.update_layout(xaxis_title='Danceability (%)', yaxis_title='Number of Streams')
    st.plotly_chart(fig1)
    import plotly.express as px
    st.subheader('Valence Distribution by Year')

    # Create box plot for Valence distribution by Year
    fig2 = px.box(data, x='released_year', y='valence_%', title='Valence Distribution by Year')
    fig2.update_layout(xaxis_title='Year', yaxis_title='Valence (%)')
    st.plotly_chart(fig2)
    import plotly.express as px

    # Create an interactive scatter plot with dropdown menu for multiple audio features on Spotify
    fig = px.scatter(data, x='danceability_%', y='streams', color='streams',
                     title='Audio Features vs. Spotify Streams',
                     labels={'danceability_%': 'Danceability (%)', 'streams': 'Number of Streams'},
                     hover_name='track_name', template='plotly_dark')

    # Add dropdown menu for audio features
    fig.update_layout(
        updatemenus=[
            {
                'buttons': [
                    {'method': 'relayout', 'label': feature.replace('_', ' ').title(),
                     'args': [{'xaxis.title.text': feature.replace('_', ' ').title()}]}
                    for feature in data.select_dtypes(include=['float']).columns
                ],
                'direction': 'down',
                'showactive': True,
                'x': 0.15,
                'xanchor': 'left',
                'y': 1.15,
                'yanchor': 'top'
            }
        ]
    )

    # Show the interactive plot
    fig.show()


    import plotly.express as px

    # Create an interactive scatter plot with dropdown menu for Spotify
    fig = px.scatter(data, x='danceability_%', y='streams', color='streams',
                     title='Danceability vs. Spotify Streams',
                     labels={'danceability_%': 'Danceability (%)', 'streams': 'Number of Streams'},
                     hover_name='track_name', template='plotly_dark')

    # Add dropdown menu for artists
    fig.update_layout(
        updatemenus=[
            {
                'buttons': [
                    {'method': 'relayout', 'label': artist,
                     'args': [{'yaxis.title.text': f'Number of Streams for {artist}'}]}
                    for artist in data['artist(s)_name'].unique()
                ],
                'direction': 'down',
                'showactive': True,
                'x': 0.15,
                'xanchor': 'left',
                'y': 1.15,
                'yanchor': 'top'
            }
        ]
    )

    # Show the interactive plot
    fig.show()
    import streamlit as st
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load the data
    df_genre = pd.read_csv('SpotifyFeatures.csv')

    # Set the title for the Streamlit app
    st.title("Duration of the Songs in Different Genres")

    # Create a bar plot
    st.set_option('deprecation.showPyplotGlobalUse', False)  # To avoid warning
    plt.figure(figsize=(10, 6))
    sns.color_palette("rocket", as_cmap=True)
    sns.barplot(y='genre', x='duration_ms', data=df_genre)
    plt.xlabel("Duration in ms")
    plt.ylabel('Genres')

    # Display the plot in Streamlit
    st.pyplot(plt)

import matplotlib
import os
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist


import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
Spotify_features = pd.read_csv('SpotifyFeatures.csv')
Spotify_features['genre'] = Spotify_features['genre'].str.lower()
Spotify_features['artist_name'] = Spotify_features['artist_name'].str.lower()
Spotify_features_lower = Spotify_features.copy()

data1 = pd.read_csv("data.csv")
genre_data = pd.read_csv('data_by_genres.csv')
year_data = pd.read_csv('data_by_year.csv')



def find_genre_songs(genre, Spotify_features, n_songs=10):
    import streamlit as st
    import pandas as pd

    Spotify_features = pd.read_csv('SpotifyFeatures.csv')
    Spotify_features['genre'] = Spotify_features['genre'].str.lower()
    Spotify_features['artist_name'] = Spotify_features['artist_name'].str.lower()
    Spotify_features_lower = Spotify_features.copy()
    genre_songs = Spotify_features[Spotify_features['genre'] == genre]

    if len(genre_songs) < n_songs:
        print(f"Not enough songs in the '{genre}' genre. Showing all available songs:")
        n_songs = len(genre_songs)

    rec_songs = genre_songs.sample(n=n_songs)
    uris = rec_songs['track_id'].tolist()

    # Specify the number of columns in the grid
    num_columns = 2

    # Calculate the number of rows needed
    num_rows = (len(uris) + num_columns - 1) // num_columns

    # Create a container to hold the iframes
    st.write('<div style="display: flex; flex-wrap: wrap;">', unsafe_allow_html=True)

    # Iterate through each URI and create an iframe
    for row in range(num_rows):
        st.write('<div style="display: flex; flex-direction: row;">', unsafe_allow_html=True)
        for col in range(num_columns):
            index = row * num_columns + col
            if index < len(uris):
                uri = uris[index]
                iframe_code = f'<iframe src="https://open.spotify.com/embed/track/{uri}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>'
                st.write(f'<div style="margin: 10px;">{iframe_code}</div>', unsafe_allow_html=True)
        st.write('</div>', unsafe_allow_html=True)

    # Close the container
    st.write('</div>', unsafe_allow_html=True)


def find_artist_songs(artist, Spotify_features, n_songs=10):
    import streamlit as st
    import pandas as pd

    Spotify_features = pd.read_csv('SpotifyFeatures.csv')
    Spotify_features['artist_name'] = Spotify_features['artist_name'].str.lower()

    artist_songs = Spotify_features[Spotify_features['artist_name'] == artist]
    rec_songs = artist_songs.sample(n=n_songs)
    uris = rec_songs['track_id'].tolist()

    # Specify the number of columns in the grid
    num_columns = 2

    # Calculate the number of rows needed
    num_rows = (len(uris) + num_columns - 1) // num_columns

    # Create a container to hold the iframes
    st.markdown('<div style="display: flex; flex-wrap: wrap;">', unsafe_allow_html=True)

    # Iterate through each URI and create an iframe
    for row in range(num_rows):
        st.markdown('<div style="display: flex; flex-direction: row;">', unsafe_allow_html=True)
        for col in range(num_columns):
            index = row * num_columns + col
            if index < len(uris):
                uri = uris[index]
                iframe_code = f'<iframe src="https://open.spotify.com/embed/track/{uri}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>'
                st.markdown(f'<div style="margin: 10px;">{iframe_code}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Close the container
    st.markdown('</div>', unsafe_allow_html=True)


def recommend_songs(song_list, spotify_data, n_songs=10):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Create a pipeline without specifying n_jobs in KMeans initialization
    cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])

    # Select numeric columns for clustering
    X = genre_data.select_dtypes(np.number)

    # Fit the pipeline
    cluster_pipeline.fit(X)

    # Predict clusters
    genre_data['cluster'] = cluster_pipeline.named_steps['kmeans'].predict(X)

    from sklearn.manifold import TSNE

    tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
    genre_embedding = tsne_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
    projection['genres'] = genre_data['genres']
    projection['cluster'] = genre_data['cluster']

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import numpy as np

    # Create a pipeline for clustering
    song_cluster_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=20, verbose=False))
    ])

    # Select numeric columns for clustering
    X = data1.select_dtypes(np.number)

    # Fit the pipeline
    song_cluster_pipeline.fit(X)

    # Predict cluster labels
    song_cluster_labels = song_cluster_pipeline.named_steps['kmeans'].predict(X)

    # Add cluster labels to the DataFrame
    data1['cluster_label'] = song_cluster_labels

    def find_song(name):
        song_data = defaultdict()
        results = sp.search(q=f'track:{name}', limit=1)
        if results['tracks']['items'] == []:
            return None

        results = results['tracks']['items'][0]
        track_id = results['id']
        audio_features = sp.audio_features(track_id)[0]

        song_data['name'] = [name]
        song_data['year'] = [results['album']['release_date'][:4]]  # Extracting year from release_date
        song_data['explicit'] = [int(results['explicit'])]
        song_data['duration_ms'] = [results['duration_ms']]
        song_data['popularity'] = [results['popularity']]

        for key, value in audio_features.items():
            song_data[key] = value

        return pd.DataFrame(song_data)

    from collections import defaultdict
    from sklearn.metrics import euclidean_distances
    from scipy.spatial.distance import cdist
    import difflib

    spotify_data = data1
    number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
                   'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

    def get_song_data(song, spotify_data):
        try:
            # Check if 'year' key is present in the song dictionary
            if 'year' in song:
                song_data = spotify_data[(spotify_data['name'] == song['name'])
                                         & (spotify_data['year'] == song['year'])].iloc[0]
            else:
                song_data = spotify_data[spotify_data['name'] == song['name']].iloc[0]
            return song_data
        except IndexError:
            # Assuming find_song is a function that searches for the song by name only
            return find_song(song['name'])

    def get_mean_vector(song_list, spotify_data):
        song_vectors = []

        for song in song_list:
            song_data = get_song_data(song, spotify_data)
            if song_data is None:
                print('Warning: {} does not exist in Spotify or in the database'.format(song['name']))
                continue
            song_vector = song_data[number_cols].astype(float).values
            song_vectors.append(song_vector)

        if not song_vectors:
            print('Warning: No valid song vectors found')
            return None

        # Convert the list of arrays to a 2D NumPy array
        song_matrix = np.vstack(song_vectors)

        # Calculate the mean along the rows (axis=0)
        return np.nanmean(song_matrix, axis=0)

    def flatten_dict_list(dict_list):
        flattened_dict = defaultdict()
        for key in dict_list[0].keys():
            flattened_dict[key] = []

        for dictionary in dict_list:
            for key, value in dictionary.items():
                flattened_dict[key].append(value)

        return flattened_dict
    from sklearn.decomposition import PCA

    pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
    song_embedding = pca_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
    projection['title'] = data1['name']
    projection['cluster'] = data1['cluster_label']
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]

        # Convert recommendations to a table
    table = rec_songs[metadata_cols].to_markdown(index=False)

    # Display the iframes in a grid
    uris = rec_songs['id'].tolist()

    # Specify the number of columns in the grid
    num_columns = 2

    # Calculate the number of rows needed
    num_rows = (len(uris) + num_columns - 1) // num_columns

    # Create a container to hold the iframes
    st.markdown('<div style="display: flex; flex-wrap: wrap;">', unsafe_allow_html=True)

    # Iterate through each URI and create an iframe
    for row in range(num_rows):
        st.markdown('<div style="display: flex; flex-direction: row;">', unsafe_allow_html=True)
        for col in range(num_columns):
            index = row * num_columns + col
            if index < len(uris):
                uri = uris[index]
                iframe_code = f'<iframe src="https://open.spotify.com/embed/track/{uri}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>'
                st.markdown(f'<div style="margin: 10px;">{iframe_code}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Close the container
    st.markdown('</div>', unsafe_allow_html=True)


def user_interaction():
        import streamlit as st
        import pandas as pd

        st.title("Music Recommender System")

        # Load necessary dataframes
        Spotify_features = pd.read_csv('SpotifyFeatures.csv')  # Adjust the file name and path accordingly
        spotify_data = pd.read_csv('data.csv')  # Adjust the file name and path accordingly

        user_choice = st.radio("Select search method:",
                               ["Search by Genre", "Search by Artist", "Search for Similar Songs"])

        if user_choice == "Search by Genre":
            genre = st.text_input("Enter the genre:").lower()
            if st.button("Find Genre Songs"):
                find_genre_songs(genre, Spotify_features)
        elif user_choice == "Search by Artist":
            artist = st.text_input("Enter the artist name:").lower()
            if st.button("Find Artist Songs"):
                find_artist_songs(artist, Spotify_features)
        elif user_choice == "Search for Similar Songs":
            song_name = st.text_input("Enter the name of the song:").lower()
            if st.button("Recommend Similar Songs"):
                recommend_songs([{'name': song_name}], spotify_data)




def main():
    import streamlit as st
    import streamlit_authenticator as stauth
    from ok import sign_up, fetch_users

    st.set_page_config(page_title='Streamlit', page_icon='🐍', initial_sidebar_state='collapsed')
    # Add a background style using CSS
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"]{
    background-image:url("https://static.vecteezy.com/system/resources/previews/011/426/067/original/aesthetic-white-and-green-background-with-space-for-text-modern-background-design-with-liquid-shape-white-background-with-green-liquid-shapes-free-vector.jpg");
    background-size:cover;
    }
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)
    title_style = f"""
                <style>
                    .title-widget {{
                        color: black;
                    }}
                </style>
            """
    st.markdown(title_style, unsafe_allow_html=True)
    st.title("Analyzing and Recommendations of Music Trends")

    try:
        users = fetch_users()
        emails = []
        usernames = []
        passwords = []

        for user in users:
            emails.append(user['key'])
            usernames.append(user['username'])
            passwords.append(user['password'])

        credentials = {'usernames': {}}
        for index in range(len(emails)):
            credentials['usernames'][usernames[index]] = {'name': emails[index], 'password': passwords[index]}

        Authenticator = stauth.Authenticate(credentials, cookie_name='Streamlit', key='abcdef', cookie_expiry_days=4)

        email, authentication_status, username = Authenticator.login(':green[Login]', 'main')

        info, info1 = st.columns(2)

        if not authentication_status:
            sign_up()

        if authentication_status and username:
            if username in usernames:
                # let User see app
                st.sidebar.subheader(f'Welcome {username}')
                Authenticator.logout('Log Out', 'sidebar')

                # Add a diagonal line
                st.markdown('<style> .diag { border: 5px solid #fff; height: 200px; width: 200px; transform: rotate(45deg); margin: auto; }</style>', unsafe_allow_html=True)
                st.markdown('<div class="diag"></div>', unsafe_allow_html=True)

                if st.button("Music Analysis", key="music_analysis_button", help="Click to analyze music trends"):
                    musicanalysis()



            else:
                with info:
                    st.warning('Username does not exist, Please Sign up')

    except:
        st.success('Refresh Page')

if __name__ == "__main__":
    main()
    user_interaction()