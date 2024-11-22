# YouTube_Content_Analyzer
## Table of Contents
- [Project Overview](#project-overview)
- [Reasons for Making This Project](#reasons-for-making-this-project)
- [Tools Used](#tools-used)
- [Tasks Performed](#tasks-performed)
- [Results and Findings](#results-and-findings)

## Project Overview:
The YouTube Video Summarizer is a web application built using Streamlit that allows users to extract and analyze content from YouTube videos. The app performs several tasks such as extracting video transcripts, summarizing the content, extracting keywords, performing topic modeling, and analyzing the sentiment of the video transcript. Users can input a YouTube video URL and get detailed insights, including a summary, key topics, keywords, and sentiment analysis.

## Reasons for Making This Project:
- **Content Summarization**: To provide concise summaries of long YouTube videos.
- **Keyword Extraction**: To help identify the most important keywords or phrases in a video.
- **Topic Modeling**: To uncover the key topics discussed in the video.
- **Sentiment Analysis**: To determine the overall sentiment (positive or negative) of the video content.
- **User Engagement**: To offer a tool that enhances the experience of consuming  videos by providing summarized insights.

## Tools Used:
- **Streamlit**: For building the interactive web application.
- **YouTube Transcript API**: For extracting video transcripts.
- **NLTK**: For text tokenization, stopword removal, and lemmatization.
- **TextBlob**: For sentiment analysis.
- **Latent Dirichlet Allocation (LDA)**: For topic modeling.
- **Transformers (Hugging Face)**: For summarization using pre-trained models.

## Tasks Performed:

### 1) Importing Libraries:
- The necessary libraries were imported to handle text extraction, processing, and analysis, including `nltk`, `textblob`, `transformers`, and `sklearn`.

### 2) User Input:
- Users input a YouTube video URL into the app interface.

### 3) Video Processing:
- **Transcript Extraction**: The app extracts the transcript of the YouTube video using the YouTube Transcript API.
- **Text Summarization**: The transcript is passed through a summarization pipeline to generate a concise summary of the video content.

### 4) Text Analysis:
- **Keyword Extraction**: The most important keywords from the transcript are identified by tokenizing the text, lemmatizing the words, and removing stopwords.
- **Topic Modeling**: Latent Dirichlet Allocation (LDA) is used to identify the main topics discussed in the video.
- **Sentiment Analysis**: The sentiment (polarity and subjectivity) of the transcript is analyzed using TextBlob.

### 5) Displaying Results:
- **Summary**: The summarized version of the video transcript is displayed.
- **Keywords**: The top 5 keywords identified from the transcript are shown.
- **Topics**: The main topics from the video transcript are listed.
- **Sentiment**: The polarity and subjectivity of the transcript are displayed to indicate the sentiment of the video.

## Results and Findings:
- The app successfully extracts and summarizes the content of YouTube videos.
- Sentiment analysis shows how positive or negative the video content is, providing insights into the emotional tone of the video.
- Keyword extraction helps identify the most discussed topics in the video.
- Topic modeling uncovers the main themes present in the video, helping users understand the key subjects.
- The app is fully interactive and user-friendly, offering a seamless experience for summarizing and analyzing YouTube videos.

## Example Use Case:
1. Input a YouTube video URL.
2. Click "Summarize" to get the transcript summary, keywords, topics, and sentiment analysis.
3. Review the summarized insights, which help to quickly understand the content and mood of the video without watching it entirely.

