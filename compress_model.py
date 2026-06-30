import joblib
import pickle

print("Compressing model...")

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    print("model.pkl loaded successfully!")

# Kompres dengan joblib
joblib.dump(model, 'model_compressed.pkl', compress=3)
print("model_compressed.pkl created successfully!")
print(f"Size: {__import__('os').path.getsize('model_compressed.pkl') / 1024 / 1024:.2f} MB")


