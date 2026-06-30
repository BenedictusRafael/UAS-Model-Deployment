import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import pickle


# KONFIGURASI FITUR
NUMERIC_FEATURES = [
    'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 
    'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
    'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
    'Changed_Credit_Limit', 'Num_Credit_Inquiries',
    'Outstanding_Debt', 'Credit_Utilization_Ratio',
    'Total_EMI_per_month', 'Amount_invested_monthly',
    'Monthly_Balance', 'Debt_to_Income_Ratio', 
    'Min_Payment_Ratio', 'Credit_History_Years'
]

CATEGORICAL_FEATURES = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Category']


# CLASS DATA PREPROCESSOR
class DataPreprocessor:
    def __init__(self):
        self.numeric_features = NUMERIC_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES
        self.preprocessor = None
    
    def fit(self, X, y=None):
        """Membangun pipeline preprocessing"""
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                 self.categorical_features)
            ]
        )
        self.preprocessor.fit(X)
        return self
    
    def transform(self, X):
        """Transformasi data"""
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X, y=None):
        """Fit dan transform sekaligus"""
        self.fit(X)
        return self.transform(X)
    
    def get_feature_names(self):
        """Mendapatkan nama fitur setelah transformasi"""
        cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out(
            self.categorical_features
        )
        return list(self.numeric_features) + list(cat_features)


# CLASS MODEL TRAINER
class ModelTrainer:
    def __init__(self, model_name, params=None):
        self.model_name = model_name
        self.params = params or {}
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Membagi data training dan testing"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _get_model(self):
        """Membuat instance model berdasarkan nama"""
        if self.model_name == 'RandomForest':
            return RandomForestClassifier(**self.params, random_state=42)
        elif self.model_name == 'LogisticRegression':
            return LogisticRegression(**self.params, max_iter=1000, random_state=42)
        elif self.model_name == 'XGBoost':
            return XGBClassifier(**self.params, random_state=42, eval_metric='mlogloss')
        else:
            raise ValueError(f"Model {self.model_name} tidak dikenal")
    
    def train(self, X_train, y_train):
        """Training model dengan MLflow tracking"""
        # Akhiri run sebelumnya jika masih aktif
        mlflow.end_run()
        
        # Pastikan tidak ada NaN
        if np.any(pd.isnull(X_train)):
            print(f"Warning: Data masih mengandung NaN! Mengisi dengan 0...")
            X_train = np.nan_to_num(X_train)
        
        with mlflow.start_run(run_name=self.model_name):
            # Log parameter
            mlflow.log_params(self.params)
            
            # Train model
            self.model = self._get_model()
            self.model.fit(X_train, y_train)
            
            # Log model ke MLflow
            mlflow.sklearn.log_model(self.model, self.model_name)
            
            print(f" Model {self.model_name} berhasil dilatih")
            return self.model
    
    def evaluate(self, X_test=None, y_test=None):
        """Evaluasi model dan log metrics ke MLflow"""
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        
        # Pastikan tidak ada NaN
        if np.any(pd.isnull(X_test)):
            print(f"Warning: Data test mengandung NaN! Mengisi dengan 0...")
            X_test = np.nan_to_num(X_test)
        
        if self.model is None:
            raise ValueError("Model belum dilatih!")
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics ke MLflow
        mlflow.log_metrics({
            'accuracy': accuracy,
            'f1_weighted': f1_weighted
        })
        
        # Print hasil
        print(f"\n Evaluasi {self.model_name}:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score (weighted): {f1_weighted:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'y_pred': y_pred,
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def save_model(self, filepath):
        """Menyimpan model ke file .pkl"""
        if self.model is None:
            raise ValueError("Model belum dilatih!")
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f" Model disimpan ke {filepath}")


# CLASS MODEL EVALUATOR
class ModelEvaluator:
    def __init__(self):
        self.results = []
    
    def compare_models(self, models, X_test, y_test):
        """Membandingkan beberapa model"""
        print("\n" + "="*60)
        print("PERBANDINGAN MODEL")
        print("="*60)
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            self.results.append({
                'Model': name,
                'Accuracy': accuracy,
                'F1-Score (weighted)': f1
            })
            
            print(f"\n{name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
        
        # DataFrame hasil
        df_results = pd.DataFrame(self.results)
        print("\n" + "-"*40)
        print(df_results.to_string(index=False))
        
        # Tentukan model terbaik
        best_model = df_results.loc[df_results['F1-Score (weighted)'].idxmax()]
        print(f"\n Model Terbaik: {best_model['Model']} (F1-Score: {best_model['F1-Score (weighted)']:.4f})")
        
        return df_results, best_model


# MAIN EXECUTION
if __name__ == "__main__":
    # Force end any active run
    mlflow.end_run()
    
    # Setup MLflow
    mlflow.set_experiment("Credit_Score_Prediction")
    
    # Load data
    print("Load data")
    df = pd.read_csv('data_A.csv', index_col=0)
    

    # DATA CLEANING
    print("Cleaning data...")

    # Ganti nilai missing
    missing_values = ['______', 'NM', '_', '!@9#%8', '__10000__', '?', '-', 'na', 'N/A', 'NULL']
    df.replace(missing_values, np.nan, inplace=True)

    # Bersihkan kolom Age
    def clean_age(age):
        try:
            age_int = int(age)
            if age_int < 0 or age_int > 120:
                return np.nan
            return age_int
        except (ValueError, TypeError):
            return np.nan

    df['Age'] = df['Age'].apply(clean_age)

    # Bersihkan kolom Annual_Income
    def clean_income(income):
        if isinstance(income, str):
            income = income.replace('_', '')
        try:
            income_float = float(income)
            if income_float < 0:
                return np.nan
            return income_float
        except (ValueError, TypeError):
            return np.nan

    df['Annual_Income'] = df['Annual_Income'].apply(clean_income)

    # Bersihkan kolom Num_of_Loan
    def clean_loan(loan):
        if isinstance(loan, str):
            loan = loan.replace('_', '')
        try:
            loan_int = int(loan)
            if loan_int < 0:
                return np.nan
            return loan_int
        except (ValueError, TypeError):
            return np.nan

    df['Num_of_Loan'] = df['Num_of_Loan'].apply(clean_loan)

    # Konversi kolom numerik
    numeric_cols_all = [
        'Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Amount_invested_monthly',
        'Monthly_Balance', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Interest_Rate', 'Delay_from_due_date',
        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month'
    ]

    for col in numeric_cols_all:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Imputasi missing value - numeric
    for col in numeric_cols_all:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Imputasi missing value - categorical
    categorical_cols_all = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount']
    for col in categorical_cols_all:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')


    # FEATURE ENGINEERING
    # 1. Debt to Income Ratio
    df['Debt_to_Income_Ratio'] = df['Outstanding_Debt'] / df['Annual_Income']
    
    # 2. Min Payment Ratio
    df['Min_Payment_Ratio'] = df['Total_EMI_per_month'] / df['Monthly_Inhand_Salary']
    
    # 3. Credit History Years
    def extract_credit_years(age_str):
        if pd.isna(age_str):
            return np.nan
        try:
            parts = str(age_str).split(' Years')
            years = int(parts[0].strip())
            return years
        except:
            return np.nan

    df['Credit_History_Years'] = df['Credit_History_Age'].apply(extract_credit_years)

    # 4. Payment Category (dari Payment_Behaviour)
    def categorize_payment(behaviour):
        if pd.isna(behaviour):
            return 'Unknown'
        if 'Low' in str(behaviour):
            return 'Low'
        elif 'Medium' in str(behaviour):
            return 'Medium'
        elif 'High' in str(behaviour):
            return 'High'
        else:
            return 'Other'
    
    df['Payment_Category'] = df['Payment_Behaviour'].apply(categorize_payment)

    # 5. Encode target
    le = LabelEncoder()
    df['Credit_Score_Encoded'] = le.fit_transform(df['Credit_Score'])

    print(" Data cleaning selesai!")
    

    # PREPROCESSING
    print("Preprocessing data")
    preprocessor = DataPreprocessor()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['Credit_Score_Encoded']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Transform data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    

    # CEK DAN PERBAIKI NAN
    # Cek NaN pada data training
    if np.any(pd.isnull(X_train_transformed)):
        print("Data training mengandung NaN! Mengisi dengan 0...")
        X_train_transformed = np.nan_to_num(X_train_transformed)
    
    if np.any(pd.isnull(X_test_transformed)):
        print("Data test mengandung NaN! Mengisi dengan 0...")
        X_test_transformed = np.nan_to_num(X_test_transformed)
    
    # Simpan preprocessor
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print(" Preprocessor disimpan ke preprocessor.pkl")
    

    # TRAINING BEBERAPA MODEL
    models_to_train = {
        'RandomForest': {
            'params': {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 5}
        },
        'LogisticRegression': {
            'params': {'C': 1.0}
        },
        'XGBoost': {
            'params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
        }
    }
    
    trained_models = {}
    
    for name, config in models_to_train.items():
        print(f"\n Training {name}")
        trainer = ModelTrainer(name, config['params'])
        trainer.train(X_train_transformed, y_train)
        trainer.evaluate(X_test_transformed, y_test)
        trained_models[name] = trainer.model
    

    # PERBANDINGAN MODEL
    evaluator = ModelEvaluator()
    df_results, best_model = evaluator.compare_models(trained_models, X_test_transformed, y_test)
    
    
    # SIMPAN MODEL TERBAIK
    best_model_name = best_model['Model']
    best_model_obj = trained_models[best_model_name]
    
    # Simpan ke .pkl
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model_obj, f)
    print(f"\n Model terbaik ({best_model_name}) disimpan ke model.pkl")
    
    # Simpan encoder target
    le = LabelEncoder()
    le.fit(df['Credit_Score'])
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print(" Encoder disimpan ke encoder.pkl")
    
    print("\n Pipeline training selesai!")

