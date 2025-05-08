from modeltraining import PhishingDetector
import pickle
import sys

def load_model(model_path='src/models/phishingdetector.pkl'):
    try:
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

def check_email(email_text):
    # Load the trained model
    detector = load_model()
    
    try:
        # Make prediction
        result = detector.predict_new_email(email_text)
        
        # Print results
        print("\nEmail Analysis Results:")
        print("-" * 20)
        print(f"Prediction: {'Phishing' if result['is_phishing'] else 'Not Phishing'}")
        print(f"Confidence: {result['confidence']:.2%}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        sys.exit(1)

while True:
    user_input = input("Enter the email text (to exit, type 'quit'):\n")
    if user_input.lower() == 'quit':
        exit()
    else:
        check_email(user_input)