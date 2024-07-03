# Music-Recommender-system
Music Recommender System! This innovative platform is designed to enhance your music discovery experience by offering personalized song recommendations and in-depth music analytics. Whether you're in the mood for something new or want to explore more from your favorite artists, our system has you covered,A simple machine learning based project used for music analysis and a music recommender system with a simple ui built using streamlit 

Find Genre Songs:

Search for songs within a specific genre. Perfect for exploring music that fits your mood or preferences.
Find Artist Songs:

Input an artist's name to receive a curated list of their popular songs. Dive deeper into their catalog with ease.
Music Analysis:

Generate visual analytics for a deeper understanding of musical trends and patterns.
Analyze various aspects like:
Popularity Trends: Track the most streamed and popular songs.
Instrumentalness: Determine the instrumental nature of the music.
Acousticness: Assess the acoustic quality of tracks.
Energy Levels: Measure the intensity and excitement of songs.
Danceability: See how suitable songs are for dancing.
Valence: Analyze the emotional tone and positivity of the music.
View trend charts and streaming statistics for a comprehensive overview.
Personalized Song Recommendations:
Receive tailored song recommendations based on your input of a song or artist. Discover new music that matches your taste.
Using K-means clustering, the system sums up and analyzes multiple input values (like genre, artist, and song attributes) to recommend songs with similar mean values, providing a personalized playlist that matches your preferences.

PERFORMING K MEANS CLUSTERING 
Certian Musical Factors such as Instrumentalness,Liveliness,Danceability,popularity etc etc.
![Screenshot 2024-07-03 225102](https://github.com/ShashankAQ/Music-Recommender-system/assets/139737140/f0113fa1-7d3b-4973-9337-9014dc624d27)


K-Means Clustering mainly for the recommend_songs fucntion


![newplot](https://github.com/ShashankAQ/Music-Recommender-system/assets/139737140/51f86b60-fab4-47d0-8c0a-941ec1e1465f)


Basic UI:
Consists of a Login page:

![Screenshot 2023-12-11 225427](https://github.com/ShashankAQ/Music-Recommender-system/assets/139737140/6caec0eb-5e4c-4b53-98d8-f68d61d584db)
Ive have also added a ok.py file from which the login page use this link for reference in the ok.py file replace the deta key with your deta key 
Reference Video:https://youtu.be/eCbH2nPL9sU?si=wJo_OsAVfePYZCXt
after you have provided the deta key in the ok.py file just run the main.py file and paste the command i mentioned  below

Comprises of two main Buttons the Music analysis button and the Music recommender Button 

![Screenshot 2023-12-12 111948](https://github.com/ShashankAQ/Music-Recommender-system/assets/139737140/7d8cfc27-10e8-44f1-9a0b-5cc9e006864f)


After you Click the Music analysis Button 

![Screenshot 2023-12-11 233903](https://github.com/ShashankAQ/Music-Recommender-system/assets/139737140/770638fd-674d-4988-961a-e487a6ed8929)


![Screenshot 2023-12-12 112308](https://github.com/ShashankAQ/Music-Recommender-system/assets/139737140/845880d1-b956-4003-8629-7cfe3f30d88d)


NOW for the music recommender part you either choose search by Genre,Artist,Similar Songs here the similar songs take some times because it compares the song which you have selected by taking the k means of that song and comparing  the mean value of other songs and which song is the closest mean it selects those songs like the above graph mentioned while the artist and Genre just matches the artist name in the dataset and return the song name of either the genre or artist and while either searching for the artist or genre make sure the spelling of the artist and genre be correct based on the dataset SpotifyNew dataset


![Screenshot 2023-12-12 112829](https://github.com/ShashankAQ/Music-Recommender-system/assets/139737140/34b5a3f2-6354-4cfd-b61e-300d6dd59b90)



![Screenshot 2023-12-12 112817](https://github.com/ShashankAQ/Music-Recommender-system/assets/139737140/36cb26fe-88b0-4de8-972e-3f5fb5675812)

make sure once the code executes:run this command streamlit run main.py #replace main.py with whatever you have named the file 

If any doubts feel free to reach me out on:Shashankashok94@gmail.com

