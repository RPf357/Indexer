import os
import time
import json
from collections import defaultdict, Counter
import nltk
from nltk.stem import PorterStemmer

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Ensure NLTK resources are downloaded (do this one time)
nltk.download('punkt')

# Function to load stopwords
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().split())
    return stopwords

# Function to preprocess and tokenize text
def preprocess_and_tokenize(text, stopwords):
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stopwords]
    return filtered_tokens

# Build the Forward Index
def build_forward_index(documents):
    forward_index = {}
    for doc_id, tokens in documents.items():
        word_freq = Counter(tokens)
        forward_index[doc_id] = word_freq
    return forward_index

# Build the Inverted Index
def build_inverted_index(forward_index):
    inverted_index = defaultdict(dict)
    for doc_id, word_freqs in forward_index.items():
        for word, freq in word_freqs.items():
            inverted_index[word][doc_id] = freq
    return inverted_index

# Save index to file
def save_index(index, filename, sort_by_key=True):
    with open(filename, 'w', encoding='utf-8') as file:
        for key in sorted(index) if sort_by_key else index:
            entries = "; ".join([f"{doc}: {freq}" for doc, freq in sorted(index[key].items())])
            file.write(f"{key}: {entries}\n")

# Measure the file size in MB
def get_file_size(filename):
    size_bytes = os.path.getsize(filename)
    return size_bytes / (1024 * 1024)  # Convert bytes to megabytes

# Read and process documents from a directory
def read_and_process_documents(directory, stopwords):
    documents = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
            documents[filename] = preprocess_and_tokenize(text, stopwords)
    return documents

def query_inverted_index(inverted_index, query):
    # Tokenize the input query string into words
    tokens = nltk.word_tokenize(query.lower())
    results = {}
    for token in tokens:
        # Stem each token
        stemmed_token = stemmer.stem(token)
        # Check if the stemmed token is in the inverted index
        if stemmed_token in inverted_index:
            results[stemmed_token] = inverted_index[stemmed_token]
        else:
            results[stemmed_token] = "No entries found."
    return results

# Main function
def main():
    documents_folder = 'ft911'  # Adjust path as necessary
    stopwords_file = 'stopwordlist.txt'  # Ensure this file is in the correct location

    print("Loading stopwords...")
    stopwords = load_stopwords(stopwords_file)

    start_time = time.time()

    print("Processing documents...")
    documents = read_and_process_documents(documents_folder, stopwords)
    
    print("Building forward index...")
    forward_index = build_forward_index(documents)
    print("Building inverted index...")
    inverted_index = build_inverted_index(forward_index)
    
    print("Saving forward index...")
    save_index(forward_index, 'forward_index.txt')
    print("Saving inverted index...")
    save_index(inverted_index, 'inverted_index.txt')

    indexing_time = time.time() - start_time
    print(f"Indexing Time: {indexing_time:.2f} seconds")
    
    forward_index_size = get_file_size('forward_index.txt')
    inverted_index_size = get_file_size('inverted_index.txt')
    print(f"Forward Index Size: {forward_index_size:.2f} MB")
    print(f"Inverted Index Size: {inverted_index_size:.2f} MB")
    
    # User interface for querying the inverted index
    while True:
        query = input("Enter a term or sentence to search (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        results = query_inverted_index(inverted_index, query)
        print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()



