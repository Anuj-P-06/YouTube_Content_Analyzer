import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from textblob import TextBlob
import re
import nltk

# Ensure that necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to summarize text
def summarize_text(text, max_length=80000):  # Increased max_length to 80,000
    summarization_pipeline = pipeline("summarization")
    summary = summarization_pipeline(text, max_length=max_length, min_length=100, do_sample=False)
    return summary[0]['summary_text']

# Function to extract keywords
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]
    keywords = [word for word in words if word not in stop_words and len(word) > 1]

    counter = CountVectorizer().fit_transform([' '.join(keywords)])
    vocabulary = CountVectorizer().fit([' '.join(keywords)]).vocabulary_
    top_keywords = sorted(vocabulary, key=vocabulary.get, reverse=True)[:5]

    return top_keywords

# Function to perform topic modeling
def topic_modeling(text):
    vectorizer = CountVectorizer(max_df=2, min_df=0.95, stop_words='english')
    tf = vectorizer.fit_transform([text])
    lda_model = LatentDirichletAllocation(n_components=5, max_iter=5, learning_method='online', random_state=42)
    lda_model.fit(tf)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-6:-1]])
    return topics

# Function to extract YouTube video ID from URL
def extract_video_id(url):
    video_id = None
    patterns = [
        r'v=([^&]+)',  # Pattern for URLs with 'v=' parameter
        r'youtu.be/([^?]+)',  # Pattern for shortened URLs
        r'youtube.com/embed/([^?]+)'  # Pattern for embed URLs
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            break
    return video_id

# Main Streamlit app
def main():
    st.title("YouTube Video Summarizer")

    # Sidebar
    st.sidebar.title("App Features and Description")
    st.sidebar.subheader("What does this app do?")
    st.sidebar.write("""
    - Extracts the transcript of a YouTube video.
    - Summarizes the video content.
    - Extracts keywords from the transcript.
    - Performs topic modeling.
    - Conducts sentiment analysis to show how positive or negative the content is.
    """)
    
    st.sidebar.subheader("Sentiment Analysis Explained")
    st.sidebar.write("""
    - **Polarity**: Measures how positive or negative the content is. Ranges from -1 to 1. 
        - Negative polarity (< 0) indicates negative sentiment.
        - Positive polarity (> 0) indicates positive sentiment.
        - Neutral polarity (0) indicates neutral sentiment.
    - **Subjectivity**: Measures how subjective or objective the content is. Ranges from 0 to 1.
        - A subjectivity score closer to 1 indicates personal opinions or beliefs.
        - A score closer to 0 indicates factual information.
    """)

    # User input for YouTube video URL
    video_url = st.text_input("Enter YouTube Video URL:", "")

    # User customization options
    max_summary_length = st.slider("Max Summary Length:", 1000, 80000, 50000)  # Increased max length to 80,000

    if st.button("Summarize"):
        try:
            # Extract video ID from URL
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL. Please enter a valid URL.")
                return

            # Get transcript of the video
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            if not transcript:
                st.error("Transcript not available for this video.")
                return

            video_text = ' '.join([line['text'] for line in transcript])

            # Summarize the transcript
            summary = summarize_text(video_text, max_length=max_summary_length)

            # Extract keywords from the transcript
            keywords = extract_keywords(video_text)

            # Perform topic modeling
            topics = topic_modeling(video_text)

            # Perform sentiment analysis
            sentiment = TextBlob(video_text).sentiment

            # Display summarized text, keywords, topics, and sentiment
            st.subheader("Video Summary:")
            st.write(summary)

            st.subheader("Keywords:")
            st.write(keywords)

            st.subheader("Topics:")
            for idx, topic in enumerate(topics):
                st.write(f"Topic {idx+1}: {', '.join(topic)}")

            st.subheader("Sentiment Analysis:")
            st.write(f"Polarity: {sentiment.polarity}")
            st.write(f"Subjectivity: {sentiment.subjectivity}")

        except TranscriptsDisabled:
            st.error("Transcripts are disabled for this video.")
        except NoTranscriptFound:
            st.error("No transcript found for this video.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
