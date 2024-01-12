#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# load the dataset and store it in a Dataframe 
csv_file = 'C:\\Users\\vonta\\OneDrive\\Desktop\\Dissertation&Industrial Expertise\\ChatGPT_Reviews.csv'
df = pd.read_csv(csv_file)

# Checking for duplicates
duplicates = df.duplicated()
print("Duplicate Rows except first occurrence:")
print(df[duplicates])

# Checking for null values
null_values = df.isnull().sum()
print("\nNull Values in each column:")
print(null_values)

# Checking for rows with null values
rows_with_null = df[df.isnull().any(axis=1)]
print("\nRows with Null Values:")
print(rows_with_null)

#counting number of labels count
class_counts = df['labels'].value_counts()

# Plotting the bar graph
plt.bar(class_counts.index, class_counts.values, color=['orange', 'green', 'blue'])
plt.xlabel('Sentiment polarity')
plt.ylabel('Count of tweets')
plt.title('Distribution of Sentiment polarity')
plt.xticks(class_counts.index, ['bad', 'good', 'neutral'])
plt.show()

# Random Undersampling technique
X = df.drop('labels', axis=1)
y = df['labels']

target_count = y.value_counts().min() # This code Determine the class with the minimum samples

rus = RandomUnderSampler(sampling_strategy={label: target_count for label in y.unique()}, random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

df_resampled = pd.DataFrame({'tweets': X_resampled['tweets'], 'labels': y_resampled})

df_resampled.to_csv('undersampled_data.csv', index=False)

#counting number of labels count
class_counts = df_resampled['labels'].value_counts()

# Plotting the bar graph
plt.bar(class_counts.index, class_counts.values, color=['orange', 'green', 'blue'])
plt.xlabel('Sentiment polarity')
plt.ylabel('Count of tweets')
plt.title('Distribution of Sentiment polarity')
plt.xticks(class_counts.index, ['bad', 'good', 'neutral'])
plt.show()



# In[2]:


import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

nltk.download('stopwords')
nltk.download('punkt')

# Load the CSV file
csv_file = 'C:\\Users\\vonta\\OneDrive\\Desktop\\Dissertation&Industrial Expertise\\undersampled_data.csv'  # Replace with your file path
df = pd.read_csv(csv_file)

# Display the first few rows to understand the data
print(df.head())

# Convert label column to numeric values
df['labels'] = df['labels'].map({'good': 1, 'bad': 0, 'neutral': 2})

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@(\w+)', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Apply data preprocessing to the 'text' column in the DataFrame
df['preprocessed_tweets'] = df['tweets'].apply(preprocess_text)

# Tokenization and word embedding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['preprocessed_tweets'])
sequences = tokenizer.texts_to_sequences(df['preprocessed_tweets'])
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, df['labels'], test_size=0.2, random_state=42
)

# Word2Vec embedding
word2vec_model = Word2Vec(sentences=[text.split() for text in df['preprocessed_tweets']], vector_size=100, window=5, min_count=1, workers=4)
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, 100))

for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

# CNN Model
embedding_dim = 100
input_length = max_sequence_length

cnn_model = Sequential()
cnn_model.add(Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, input_length=input_length, weights=[embedding_matrix], trainable=False))
cnn_model.add(Conv1D(256, 5, activation='relu'))
cnn_model.add(MaxPooling1D())
cnn_model.add(Bidirectional(LSTM(128, return_sequences=True)))
cnn_model.add(GlobalMaxPooling1D())
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(3, activation='softmax'))

# Compile the CNN model
cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping for CNN model
cnn_early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Model checkpointing for CNN model
cnn_model_checkpoint = ModelCheckpoint('best_cnn_model.h5', save_best_only=True)

# Train the CNN model with early stopping and model checkpointing
cnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[cnn_early_stopping, cnn_model_checkpoint])

# RNN LSTM Model
rnn_lstm_model = Sequential()
rnn_lstm_model.add(Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, input_length=input_length, weights=[embedding_matrix], trainable=False))
rnn_lstm_model.add(Bidirectional(LSTM(128, return_sequences=True)))
rnn_lstm_model.add(Dropout(0.5))
rnn_lstm_model.add(Bidirectional(LSTM(64)))
rnn_lstm_model.add(Dropout(0.5))
rnn_lstm_model.add(Dense(3, activation='softmax'))

# Compile the RNN LSTM model
rnn_lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping for RNN LSTM model
rnn_lstm_early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Model checkpointing for RNN LSTM model
rnn_lstm_model_checkpoint = ModelCheckpoint('best_rnn_lstm_model.h5', save_best_only=True)

# Train the RNN LSTM model with early stopping and model checkpointing
rnn_lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[rnn_lstm_early_stopping, rnn_lstm_model_checkpoint])

# Evaluate the CNN model on the test set
cnn_y_pred = cnn_model.predict(X_test)
cnn_y_pred_classes = np.argmax(cnn_y_pred, axis=1)

# Evaluate the RNN LSTM model on the test set
rnn_lstm_y_pred = rnn_lstm_model.predict(X_test)
rnn_lstm_y_pred_classes = np.argmax(rnn_lstm_y_pred, axis=1)

# Compute and print accuracy for CNN model
cnn_accuracy = accuracy_score(y_test, cnn_y_pred_classes)
print(f'CNN Test Accuracy: {cnn_accuracy:.4f}')

# Compute and print accuracy for RNN LSTM model
rnn_lstm_accuracy = accuracy_score(y_test, rnn_lstm_y_pred_classes)
print(f'RNN LSTM Test Accuracy: {rnn_lstm_accuracy:.4f}')

# Compute and print classification report for CNN model
cnn_classification_rep = classification_report(y_test, cnn_y_pred_classes, target_names=['bad', 'good', 'neutral'])
print('CNN Classification Report:\n', cnn_classification_rep)

# Compute and print classification report for RNN LSTM model
rnn_lstm_classification_rep = classification_report(y_test, rnn_lstm_y_pred_classes, target_names=['bad', 'good', 'neutral'])
print('RNN LSTM Classification Report:\n', rnn_lstm_classification_rep)


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot trend charts for metrics
def plot_metric_trend(epochs, train_metric, val_metric, metric_name, model_name):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_metric, label=f'Training {metric_name}')
    plt.plot(epochs, val_metric, label=f'Validation {metric_name}')
    plt.title(f'{metric_name} Trend - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()

# Train and evaluate CNN model
cnn_history = cnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[cnn_early_stopping, cnn_model_checkpoint])

# Train and evaluate RNN LSTM model
rnn_lstm_history = rnn_lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[rnn_lstm_early_stopping, rnn_lstm_model_checkpoint])

# Plot accuracy trend for CNN model
plot_metric_trend(range(1, len(cnn_history.history['accuracy']) + 1),
                  cnn_history.history['accuracy'],
                  cnn_history.history['val_accuracy'],
                  'Accuracy', 'CNN Model')

# Plot accuracy trend for RNN LSTM model
plot_metric_trend(range(1, len(rnn_lstm_history.history['accuracy']) + 1),
                  rnn_lstm_history.history['accuracy'],
                  rnn_lstm_history.history['val_accuracy'],
                  'Accuracy', 'RNN LSTM Model')


# In[4]:


import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Function to plot a confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Function to plot bar charts for accuracy, precision, recall, and F1 score
def plot_metrics_bar_charts(metrics_cnn, metrics_rnn_lstm):
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, metrics_cnn, width, label='CNN', color='orange')
    ax.bar([i + width for i in range(len(labels))], metrics_rnn_lstm, width, label='RNN LSTM', color='green')

    ax.set_title('Comparison of Metrics between CNN and RNN LSTM')
    ax.set_ylabel('Value')
    ax.legend()

    plt.show()

# Function to calculate metrics and plot confusion matrix for both models
def evaluate_models(cnn_model, rnn_lstm_model, X_test, y_test):
    # Evaluate CNN model
    cnn_y_pred = np.argmax(cnn_model.predict(X_test), axis=1)
    cnn_metrics = calculate_metrics(y_test, cnn_y_pred, 'CNN')

    # Evaluate RNN LSTM model
    rnn_lstm_y_pred = np.argmax(rnn_lstm_model.predict(X_test), axis=1)
    rnn_lstm_metrics = calculate_metrics(y_test, rnn_lstm_y_pred, 'RNN LSTM')

    # Plot bar charts for metrics comparison
    plot_metrics_bar_charts(cnn_metrics, rnn_lstm_metrics)

# Function to calculate metrics and print confusion matrix for a model
def calculate_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    # Print metrics
    print(f'{model_name} Test Accuracy: {accuracy:.4f}')
    print(f'{model_name} Precision: {precision:.4f}')
    print(f'{model_name} Recall: {recall:.4f}')
    print(f'{model_name} F1 Score: {f1:.4f}')

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, labels=['bad', 'good', 'neutral'], model_name=model_name)

    return [accuracy, precision, recall, f1]

# Evaluate both models and compare metrics
evaluate_models(cnn_model, rnn_lstm_model, X_test, y_test)


# In[ ]:




