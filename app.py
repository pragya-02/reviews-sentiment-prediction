import streamlit as st
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Set the page configuration
st.set_page_config(page_title="Movie Review Sentiment Prediction", page_icon=":clapper:", layout="centered")

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the trained model
model = load_model("sentiment_prediction_model_1.h5")

# Define a function to predict sentiment
def predict_sentiment(review):
    # Tokenize and pad the review
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    # Predict the sentiment
    prediction = model.predict(padded_sequence)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment

# Streamlit UI
st.title("Movie Review Sentiment Prediction")
st.write("Enter a movie review to predict its sentiment (positive/negative).")

# Text area for user input
user_review = st.text_area("Enter your review here:")

# Button for prediction
if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.write("Please enter a review.")
    else:
        sentiment = predict_sentiment(user_review)
        st.write(f"The sentiment of the review is: **{sentiment}**")
