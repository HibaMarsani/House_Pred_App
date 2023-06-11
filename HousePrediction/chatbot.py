import numpy as np

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

model_path = 'model.pkl'
class ChatBot:
    def __init__(self, model_path):
        print("modelllllllllllllll555")
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        with open('model.pkl','rb') as file:
            model = pickle.load(file)
            print("modelllllllllllllll")
        return model

    def predict(self, values):
        values = np.array(values).reshape(1, -1)
        prediction = self.load_model(model_path).predict(values)
        return prediction[0]

    def process_text(self, text):
        # Tokenize text
        tokens = word_tokenize(text)

        # Remove stop words
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token.lower() not in stop_words]

        # Perform stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

        return tokens

    def chat(self ,inputuser,feature_values):
        print("Welcome to House Prediction Chatbot!")
        print(
            "You can ask me questions about your preferences to find your dream house and I will do my best to assist you.")
        print("Enter 'quit' to exit the chat.")

        while True:
            user_input = inputuser
            user_input = user_input.lower()

            if user_input == 'quit':
                print("Chatbot: Goodbye!")
                break

            # Process user input
            tokens = self.process_text(user_input)

            # Check for keywords and respond accordingly

            if 'hello' in tokens or 'hi' in tokens:
                return(" Hi there!")
            elif 'how are you' in user_input:
                return(" I'm doing well, thank you!")
            elif 'thank you' in tokens or 'thanks' in tokens:
                return(" You're welcome!")
            elif 'bye' in tokens or 'goodbye' in tokens:
                return(" Goodbye!")
                break



            elif 'predict' in tokens:
                if self.model is None:
                    return " Model not trained. Please train the model first."
                else:
                    if len(feature_values) < 5:
                        # Ask for the next feature value
                        feature = ['LotArea', 'OverallQual', 'TotalBsmtSF', 'GarageArea', 'GrLivArea'][
                            len(feature_values)]
                        return f"Chatbot: Please enter the value of {feature}:"
                    else:
                        try:
                            values = [float(value) for value in feature_values.values()]
                            print("values array :",values)
                            prediction = self.predict(values)
                            return "The predicted value is: " + str(prediction)
                        except ValueError:
                            return " Invalid input. Please enter numerical values for the features."

            elif 'process' in tokens and 'text' in tokens:
                text = input(" Please enter the text for processing: ")
                processed_text = self.process_text(text)
                print(" The processed text is: ")
                print(processed_text)

            else:
                return("I'm sorry, I don't understand. Can you please rephrase your question?")


