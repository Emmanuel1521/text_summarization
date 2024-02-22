import nltk
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from bs4 import BeautifulSoup
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging

def fetch_article(url):
    # Make a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from the HTML content
        article_text = ' '.join([p.text for p in soup.find_all('p')])
        
        return article_text
    else:
        print("Failed to fetch article from URL:", url)
        return None

def summarize_extractive(text, num_sentences=3):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    
    # Calculate word frequency
    word_freq = FreqDist(filtered_words)
    
    # Calculate sentence scores based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        sentence_words = word_tokenize(sentence.lower())
        sentence_score = sum([word_freq[word] for word in sentence_words if word in word_freq])
        sentence_scores[sentence] = sentence_score
    
    # Get the top sentences with highest scores
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    
    # Return the summarized text
    summarized_text = ' '.join(top_sentences)
    return summarized_text

def summarize_abstractive(text):
    # Suppressing transformers logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Example usage
url = "https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2021.669042/full?fbclid"
article = fetch_article(url)
if article:
    print("Extractive Summary:")
    extractive_summary = summarize_extractive(article, num_sentences=5)
    print(extractive_summary)

    print("\nAbstractive Summary:")
    abstractive_summary = summarize_abstractive(article)
    print(abstractive_summary)
