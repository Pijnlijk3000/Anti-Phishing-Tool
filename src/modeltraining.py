import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import time
import pickle

class PhishingDetector:
    def __init__(self):
        self.model = None
        self.tfidf = None
        self.best_params = None
        self.feature_importance = None

    # Function to preprocess text data, removing unwanted characters and numbers, and converting to lowercase    
    def preprocess_text(self, text, unwanted_chars):
        text = text.lower()
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(r'\b(?:mon|tue|wed|thu|fri|sat|sun)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', '', text, flags=re.IGNORECASE)
        for char in unwanted_chars:
            text = re.sub(rf'\b{char}\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Function to load and preprocess data, removing duplicates and null values. Also, downsampling the data to balance the classes, and removing unwanted characters
    def load_and_preprocess_data(self, data_path, sample_size=10000):
        print("Loading and preprocessing data...")
        
        data_path = 'data/phishing_email.csv'

        df = pd.read_csv(data_path)
        df = df.dropna()
        df = df.drop_duplicates()
        
        print("\nInitial data shape:", df.shape)
        print("\nLabel distribution:")
        print(df['label'].value_counts())
        
        phishing = df[df['label'] == 1]
        non_phishing = df[df['label'] == 0]
        
        # I use a sample size of 10000 for each class to balance the classes, and because 80000 rows of data is too large for my liking
        phishing_sample = phishing.sample(n=sample_size, random_state=42)
        non_phishing_sample = non_phishing.sample(n=sample_size, random_state=42)
        
        # Combine the samples and shuffle the data
        df_sample = pd.concat([phishing_sample, non_phishing_sample]).reset_index(drop=True)
        
        # remove unwanted characters. enron, hpl, nom, forwarded are common words in the dataset that are not useful for the model
        unwanted_chars = ['enron', 'hpl', 'nom', 'forwarded']
        # lambda function to apply the preprocess_text function to the text_combined column
        df_sample['text_combined'] = df_sample['text_combined'].apply(lambda x: self.preprocess_text(x, unwanted_chars))
        
        return df_sample
    
    def prepare_data_for_model(self, df_sample, test_size=0.3):
        print("\nPreparing data for modeling...")
        
        X = df_sample['text_combined']
        y = df_sample['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        self.tfidf = TfidfVectorizer(max_features=5000)
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        X_test_tfidf = self.tfidf.transform(X_test)
        
        print(f"Training set shape: {X_train_tfidf.shape}")
        print(f"Test set shape: {X_test_tfidf.shape}")
        
        return X_train_tfidf, X_test_tfidf, y_train, y_test

    def train(self, X_train_tfidf, y_train):
        print("\nTraining model...")
        start_time = time.time()
        
        param_grid = {'C': [0.1, 1.0, 10.0],'penalty': ['l1', 'l2'],'solver': ['liblinear'],'max_iter': [1000]}
        
        grid_search = GridSearchCV(LogisticRegression(),param_grid,cv=5,scoring='f1',n_jobs=-1)
        
        grid_search.fit(X_train_tfidf, y_train)
        
        self.model = grid_search.best_estimator_
        self.save_model()
        self.best_params = grid_search.best_params_
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best parameters: {self.best_params}")
        
        return self
    
    def save_model(self):
        model_pkl_filename = 'src/models/phishingdetector.pkl'
        with open (model_pkl_filename, 'wb') as file:
            pickle.dump(self, file)
    
    def predict(self, X_test_tfidf):
        return self.model.predict(X_test_tfidf)
    
    def predict_proba(self, X_test_tfidf):
        return self.model.predict_proba(X_test_tfidf)
    
    def evaluate(self, X_test_tfidf, y_test):
        print("\nEvaluating model...")
        
        y_pred = self.predict(X_test_tfidf)
        
        metrics = {'accuracy': accuracy_score(y_test, y_pred),'precision': precision_score(y_test, y_pred),'recall': recall_score(y_test, y_pred),'f1': f1_score(y_test, y_pred)}
        
        feature_importance = pd.DataFrame({'feature': self.tfidf.get_feature_names_out(),'importance': abs(self.model.coef_[0])})
        self.feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    # Function to predict if a new email is phishing or not
    def predict_new_email(self, email_text):
        processed_text = self.preprocess_text(email_text, ['enron', 'hpl', 'nom', 'forwarded'])
        text_tfidf = self.tfidf.transform([processed_text])
        prediction = self.predict(text_tfidf)[0]
        probability = self.predict_proba(text_tfidf)[0][1]
        
        return {'is_phishing': bool(prediction),'confidence': probability}

def main():
    detector = PhishingDetector()
    data_path = 'data/phishing_email.csv'
    df_sample = detector.load_and_preprocess_data(data_path)
    # Split the data into training and test sets, and prepare the data for the model, X_test_tfidf and y_test are not used but are required for the return values of the function
    X_train_tfidf, X_test_tfidf, y_train, y_test = detector.prepare_data_for_model(df_sample)
    detector.train(X_train_tfidf, y_train)
    
    # Small test suite to check the model with example emails
    print("\nTesting model with example emails:")
    example_emails = [
        "Dear user, your account needs immediate verification. Click here to verify.",
        "Meeting scheduled for tomorrow at 2 PM in the conference room.",
        "URGENT: Your bank account has been suspended. Click this link to restore access."
    ]
    
    # Print the results of the test suite
    for i, email in enumerate(example_emails, 1):
        result = detector.predict_new_email(email)
        print(f"\nExample {i}:")
        print(f"Email: {email}")
        print(f"Prediction: {'Phishing' if result['is_phishing'] else 'Not Phishing'}")
        print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    main()