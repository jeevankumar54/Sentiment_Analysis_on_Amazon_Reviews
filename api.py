from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
import pandas as pd
import pickle
import base64
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

app = Flask(__name__)
STOPWORDS = set(stopwords.words("english"))

def check_dependencies():
    """Check if all required modules are available"""
    required_modules = ['xgboost', 'sklearn', 'pandas', 'numpy', 'matplotlib', 'nltk']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing required modules: {', '.join(missing_modules)}")
        print("Please install them using: pip install " + " ".join(missing_modules))
        return False
    return True

def load_models():
    """Load trained models with better error handling"""
    model_files = {
        'predictor': 'Models/model_xgb.pkl',
        'scaler': 'Models/scaler.pkl',
        'cv': 'Models/countVectorizer.pkl'
    }
    
    # Check if Models directory exists
    if not os.path.exists('Models'):
        print("❌ Models directory not found!")
        print("Please create the Models directory and train your models first.")
        print("Run: python sentiment_analysis_on_amazon_reviews.py")
        return None, None, None
    
    models = {}
    for name, file_path in model_files.items():
        if not os.path.exists(file_path):
            print(f"❌ Model file not found: {file_path}")
            print("Please train your models first by running:")
            print("python sentiment_analysis_on_amazon_reviews.py")
            return None, None, None
        
        try:
            print(f"Loading {name} from {file_path}...")
            with open(file_path, "rb") as f:
                models[name] = pickle.load(f)
            print(f"✅ {name} loaded successfully")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
            return None, None, None
    
    print("🎉 All models loaded successfully!")
    return models['predictor'], models['scaler'], models['cv']

# Check dependencies first
if not check_dependencies():
    print("Exiting due to missing dependencies...")
    sys.exit(1)

# Load models
print("Loading trained models...")
predictor, scaler, cv = load_models()

if not all([predictor, scaler, cv]):
    print("Failed to load models. Please check the error messages above.")
    sys.exit(1)

def preprocess_text(text):
    """Preprocess text for sentiment prediction"""
    stemmer = PorterStemmer()
    # Remove non-alphabetic characters
    text = re.sub("[^a-zA-Z]", " ", text)
    # Convert to lowercase and split
    words = text.lower().split()
    # Stem words and remove stopwords
    words = [stemmer.stem(word) for word in words if word not in STOPWORDS]
    return " ".join(words)

def predict_sentiment(text):
    """Predict sentiment for a single text"""
    try:
        # Preprocess text
        processed = preprocess_text(text)
        # Vectorize
        vectorized = cv.transform([processed]).toarray()
        # Scale
        scaled = scaler.transform(vectorized)
        # Predict
        prediction = predictor.predict(scaled)[0]
        return "Positive" if prediction == 1 else "Negative"
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error"

@app.route("/")
def home():
    """Home page"""
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint"""
    try:
        # Handle file upload (CSV)
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Read CSV
            try:
                data = pd.read_csv(file)
            except Exception as e:
                return jsonify({"error": f"Error reading CSV: {str(e)}"}), 400
            
            # Check if required column exists
            if 'Sentence' not in data.columns:
                return jsonify({"error": "CSV must have a 'Sentence' column"}), 400
            
            # Predict sentiments
            print(f"Processing {len(data)} sentences...")
            data["Predicted"] = data["Sentence"].apply(lambda x: predict_sentiment(str(x)))
            
            # Generate pie chart
            plt.figure(figsize=(8, 6))
            sentiment_counts = data["Predicted"].value_counts()
            colors = ['#ff9999', '#66b3ff']
            plt.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                   autopct="%1.1f%%", colors=colors, startangle=90)
            plt.title("Sentiment Analysis Results")
            
            # Save plot to base64 string
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format="png", bbox_inches='tight', dpi=300)
            plt.close()
            img_buffer.seek(0)
            graph_data = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
            
            # Prepare CSV response
            output = BytesIO()
            data.to_csv(output, index=False)
            output.seek(0)
            
            response = send_file(output, mimetype="text/csv", 
                               as_attachment=True, download_name="predictions.csv")
            response.headers["X-Graph"] = graph_data
            print("✅ CSV processing completed successfully")
            return response
        
        # Handle single text prediction
        elif request.is_json and "text" in request.json:
            text = request.json["text"]
            if not text.strip():
                return jsonify({"error": "Text cannot be empty"}), 400
            
            result = predict_sentiment(text)
            return jsonify({"result": result})
        
        else:
            return jsonify({"error": "No valid input provided"}), 400
            
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route("/health")
def health():
    """Health check endpoint"""
    models_loaded = all([predictor, scaler, cv])
    return jsonify({
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded
    })

if __name__ == "__main__":
    print("🚀 Starting Flask application...")
    print("🌐 Access the application at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)