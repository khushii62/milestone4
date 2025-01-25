import pandas as pd
import cohere
import speech_recognition as sr
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import gspread
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import os
from dotenv import load_dotenv 
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
# Load environment variables from .env file
load_dotenv()

# Access variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GOOGLE_SHEET_KEY = os.getenv("GOOGLE_SHEET_KEY")
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")
PRODUCT_DATA_PATH = os.getenv("PRODUCT_DATA_PATH")
OBJECTION_QUESTIONS_PATH = os.getenv("OBJECTION_QUESTIONS_PATH")

# Initialize Cohere and Elasticsearch clients
co = cohere.Client(COHERE_API_KEY)
es = Elasticsearch("http://localhost:9200")
productdata = pd.read_csv(PRODUCT_DATA_PATH).fillna('')
objection_questions = pd.read_csv(OBJECTION_QUESTIONS_PATH).fillna('')
GOOGLE_SHEET="Speech analysis"
index_name = "productdata" 

# Google Sheets setup

def authenticate_google_sheets():
    try:
        creds = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_key(GOOGLE_SHEET_KEY).sheet1
        print(f"Authenticated and connected to sheet: {sheet.title}")
        return sheet
    except Exception as e:
        print(f"Error in Google Sheets authentication: {e}")
        return None

# Function to save data to Google Sheets
def save_to_google_sheets(sheet, data):
    try:
        print(f"Attempting to append data: {data}")
        if isinstance(data, list):
            sheet.append_row(data)
            print("Data successfully appended to Google Sheets!")
            st.success("Data saved to Google Sheets!")
        else:
            st.error("Data format is incorrect. Expected a list.")
            print("Error: Data is not a list.")
    except Exception as e:
        print(f"Error saving data to Google Sheets: {e}")
        st.error(f"Error saving data to Google Sheets: {e}")
# Fetch call data from Google Sheets
def fetch_call_data(sheet_id, sheet_range="Sheet1!A1:E"):
    """
    Fetches data from the specified Google Sheet and returns a pandas DataFrame.

    :param sheet_id: The ID of the Google Sheet to fetch data from.
    :param sheet_range: The range in A1 notation to fetch data from.
    :return: pandas DataFrame with the sheet data.
    """
    try:
        # Authenticate and get credentials
        creds = authenticate_google_sheets()
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()

        # Fetch the data
        result = sheet.values().get(
            spreadsheetId=sheet_id,
            range=sheet_range
        ).execute()

        # Get the rows
        rows = result.get("values", [])

        # If rows exist, convert to DataFrame
        if rows:
            # Use the first row as column headers
            headers = rows[0]
            data = rows[1:]

            # Create DataFrame
            df = pd.DataFrame(data, columns=headers)

            return df
        else:
            # Return an empty DataFrame if no data
            return pd.DataFrame()

    except Exception as e:
        print(f"Error fetching data from Google Sheets: {e}")
        # Return an empty DataFrame in case of error
        return pd.DataFrame()
    
# Sentiment analysis function
def analyze_sentiment(text):
    try:
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(text)
        if sentiment_scores['compound'] > 0.05:
            return "Positive"
        elif sentiment_scores['compound'] < -0.05:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        st.error(f"Error in sentiment analysis: {e}")
        return "Error analyzing sentiment"

# Generate call summary using Cohere
def generate_call_summary(call_script, context):
    try:
        combined_input = call_script + "\n\nRelevant Context:\n" + "\n".join(context)

        response = co.generate(
            model="command-xlarge-nightly",
            prompt=f"Call Script:\n{combined_input}\n\nGenerate a concise summary and key points:",
            max_tokens=300,
            temperature=0.5
        )

        generated_text = response.generations[0].text
        sections = generated_text.split("Key Points:")
        summary = sections[0].strip()
        key_points = sections[1].strip().split("\n") if len(sections) > 1 else []

        return {
            "Key Points": key_points,
            "Summary": summary
        }
    except Exception as e:
        st.error(f"Error in generating call summary: {e}")
        return {"Key Points": [], "Summary": "Error generating summary."}

# Handle objection or question
def handle_objection(query):
    try:
        response = co.generate(
            model="command-xlarge",
            prompt=f"Handle the following objection or question: {query}",
            max_tokens=100,
            temperature=0.7
        )
        return response.generations[0].text.strip()
    except Exception as e:
        st.error(f"Error in objection handling: {e}")
        return "Sorry, I couldn't generate a response."

# Elasticsearch helper functions
def generate_embedding(text):
    try:
        response = co.embed(texts=[text], model="embed-english-v2.0")
        return response.embeddings[0]
    except Exception as e:
        st.error(f"Error generating embedding for text: {e}")
        return None

#generate embbeding using cohere api for product recommendation
def generate_embedding(text):
    try:
        response = co.embed(texts=[text], model="embed-english-v2.0")
        return response.embeddings[0]
    except Exception as e:
        print(f"Error generating embedding for text: {text}\n{e}")
        return None
    
#search product
def search_product(query):
#def search_product(query, productdata):
    try:
        query_embedding = generate_embedding(query)
        if query_embedding is None:
            st.write("Failed to generate embedding for query.")
            return

        # Search for the top product
        search_query = {
            "size": 5,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            }
        }

        response = es.search(index=index_name, body=search_query)
        hits = response['hits']['hits']

        if hits:
            top_product = hits[0]["_source"]
            st.write(f"**Top Product Recommendation:**\nName: {top_product['name']} | Price: {top_product['price']} | Category: {top_product['category']} | Description: {top_product['description']}")

            category = top_product['category']
            similar_products_query = {
                "size": 4,
                "query": {
                    "bool": {
                        "must": [{"match": {"category": category}}],
                        "must_not": [{"term": {"name": top_product['name']}}]  # Exclude the top product
                    }
                }
            }
            similar_products_response = es.search(index=index_name, body=similar_products_query)
            similar_hits = similar_products_response['hits']['hits']

            st.write(f"Similar products in the {category} category:")
            for hit in similar_hits:
                source = hit["_source"]
                st.write(f"Name: {source['name']} | Price: {source['price']} | Description: {source['description']}")
        else:
            st.warning("No matching products found.")
    except Exception as e:
        st.error(f"Error during product search: {e}")
# Real-time speech recognition and processing
def real_time_analysis():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    st.info("Say 'stop' to end the process.")
    sheet = authenticate_google_sheets()  # Authenticate and get the Google Sheet
    if not sheet:
        st.error("Failed to access Google Sheets.") 
        return
    try:
        while True:
            with mic as source:
                st.write("Listening...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

            try:
                st.write("Recognizing...")
                text = recognizer.recognize_google(audio)
                st.write(f"**Recognized Text:** {text}")

                if 'stop' in text.lower():
                    st.write("Stopping real-time analysis...")
                    break

                # Sentiment analysis
                sentiment = analyze_sentiment(text)
                st.write(f"**Sentiment:** {sentiment}")
                
                # Context retrieval (placeholder)
                context = []  # Replace with context retrieval logic

                # Call summary
                result = generate_call_summary(text, context)
                st.write("**Key Points:**")
                for point in result["Key Points"]:
                    st.write(f"- {point}")
                st.write("\n**Summary:**")
                st.write(result["Summary"])

                # Objection handling
                objection_response = handle_objection(text)
                st.write(f"**Objection Response:** {objection_response}")

                # Product recommendation
                st.write("**Product Recommendation:**")
                search_product(text)
                # Save data to Google Sheets
                row = [
                    text,  # Recognized text
                    sentiment,  # Sentiment analysis
                    result["Summary"],  # Call summary
                    ", ".join(result["Key Points"]),  # Key points as a single string
                    objection_response  # Objection response
                ]
                sheet.append_row(row)  # Append the row to the Google Sheet
                st.success("Data saved to Google Sheets!")
            except sr.UnknownValueError:
                st.error("Speech Recognition could not understand the audio.")
            except sr.RequestError as e:
                st.error(f"Error with the Speech Recognition service: {e}")
            except Exception as e:
                st.error(f"Error during processing: {e}")

    except Exception as e:
        st.error(f"Error in real-time analysis: {e}")

#Initialize session state to store analysis history
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# Navigation bar
menu = ["Dashboard", "Real-Time Analysis"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Dashboard":
    # Display Dashboard content
    st.markdown("# 🧠 AI Dashboard")
    st.markdown("Welcome to your intelligent assistant! Choose a feature from the options below.")

    # Fetch and display data from Google Sheets
    sheet = authenticate_google_sheets()
    if sheet:
        try:
            data = sheet.get_all_records(expected_headers=["recognized text", "Sentiment", "product detial", "object handling", "key points", "Summary"])  # Simplified to avoid header mismatch issues

            st.subheader("Real-Time Analysis History")
            if data:
                st.write("Data from Google Sheets:")
                df = pd.DataFrame(data)
                st.write(df)
            else:
                st.info("No analysis history available yet.")
        except Exception as e:
            st.error(f"Error fetching data from Google Sheets: {e}")
    else:
        st.error("Unable to authenticate Google Sheets or fetch data.")

    # Sentiment chart data (example)
    sentiment_counts = {
        "Neutral": 500,
        "Positive": 150,
        "Negative": 100,
        "Very Positive": 50,
        "Very Negative": 20,
    }

    # Filter out "Very Positive" and "Very Negative" sentiments
    filtered_sentiment_counts = {k: v for k, v in sentiment_counts.items() if k in ["Neutral", "Positive", "Negative"]}

    # Function to plot sentiment counts as a bar chart
    def plot_sentiment_bar_chart(sentiment_data):
        fig, ax = plt.subplots()
        ax.bar(sentiment_data.keys(), sentiment_data.values(), color=["blue", "green", "red"])
        ax.set_title("Sentiment Counts")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Calls")
        ax.set_xticklabels(sentiment_data.keys(), rotation=45)
        st.pyplot(fig)

    # Add sentiment chart to the dashboard
    if st.button("Show Sentiment Chart"):
        plot_sentiment_bar_chart(filtered_sentiment_counts)

elif choice == "Real-Time Analysis":
    # Real-Time Analysis content
    st.title("Speech-to-Text and Sentiment Analysis")

    # Tabs for navigation
    tab1, tab2, tab3 = st.tabs(["Real-Time Analysis", "Analyze Call Script", "Product Recommendations"])
    
    with tab1:
        st.subheader("Real-Time Analysis")
        if st.button("Start Real-Time Analysis"):
            with st.spinner("Processing..."):
                # Example analysis result
                result = real_time_analysis()  # Replace with actual function logic
                timestamp = "2025-01-24 10:00:00"  # Use `datetime.now()` in actual implementation

                # Generate call summary and key points
                summary_data = generate_call_summary(result, ["Example context data relevant to the query."])

                # Save the result to session state
                st.session_state.analysis_history.append({
                    "timestamp": timestamp,
                    "summary": summary_data['Summary'],
                    "key_points": summary_data['Key Points']
                })
            st.success("Real-Time Analysis Completed!")

    with tab2:
        st.subheader("Analyze Call Script")
        user_query = st.text_area("Enter text for analysis:")
        if st.button("Analyze"):
            if user_query.strip():
                # Sentiment analysis
                sentiment = analyze_sentiment(user_query)

                # Call summary (context retrieval simplified)
                context = ["Example context data relevant to the query."]  # Placeholder for Elasticsearch retrieval
                summary_data = generate_call_summary(user_query, context)

                # Display results
                st.subheader("Results")
                st.metric(label="Sentiment Score", value="Positive", delta="+0.8")
                st.metric(label="Processed Text", value="1200 words")
                st.write(f"**Sentiment:** {sentiment}")
                st.write(f"**Summary:** {summary_data['Summary']}")

                st.subheader("Key Points")
                for point in summary_data['Key Points']:
                    st.write(f"- {point}")
            else:
                st.error("Please enter some text to analyze!")

    with tab3:
        st.subheader("Product Recommendations")
        product_query = st.text_input("Enter a product-related query:")
        product_data = "productdata"

        if st.button("Recommend Products"):
            if product_query.strip():
                search_product(product_query, product_data)
            else:
                st.error("Please enter a query to get product recommendations!")


