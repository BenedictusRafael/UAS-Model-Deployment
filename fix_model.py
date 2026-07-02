import pickle
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

print("="*50)
print("FIXING MODEL FOR STREAMLIT CLOUD")
print("="*50)

# 1. Load model yang ada
print("\n1. Loading model")
try:
    model = joblib.load('model_compressed.pkl')
    print("Model loaded with joblib")
except:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded with pickle")

# 2. Load preprocessor
print("\n2. Loading preprocessor")
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
print("Preprocessor loaded")

# 3. Load encoder
print("\n3. Loading encoder")
encoder = pickle.load(open('encoder.pkl', 'rb'))
print("Encoder loaded")

# 4. Save dengan versi yang kompatibel
print("\n4. Saving with compatible versions...")
joblib.dump(model, 'model_cloud.pkl', compress=3)
with open('preprocessor_cloud.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
with open('encoder_cloud.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("\nFiles created:")
print("model_cloud.pkl")
print("preprocessor_cloud.pkl")
print("encoder_cloud.pkl")
print("\nUpload these files to GitHub and update app.py")

