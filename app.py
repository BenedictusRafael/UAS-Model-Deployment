import streamlit as st
import pandas as pd
import pickle
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# DEFINISIKAN ULANG CLASS DATAPREPROCESSOR
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

class DataPreprocessor:
    def __init__(self):
        self.numeric_features = NUMERIC_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES
        self.preprocessor = None
    
    def fit(self, X, y=None):
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
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


# LOAD MODEL DAN PREPROCESSOR
@st.cache_resource
def load_models():
    # Coba load model_compressed.pkl dengan joblib
    try:
        model = joblib.load('model_compressed.pkl')
        print("Model loaded with joblib")
    except:
        # Jika gagal, coba model_fixed.pkl dengan pickle
        try:
            with open('model_fixed.pkl', 'rb') as f:
                model = pickle.load(f)
            print("Model loaded with pickle (fixed)")
        except:
            # Terakhir, coba model.pkl dengan pickle
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            print("Model loaded with pickle (original)")
    
    preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
    encoder = pickle.load(open('encoder.pkl', 'rb'))
    return model, preprocessor, encoder

model, preprocessor, encoder = load_models()


# TEST CASES
TEST_CASES = {
    "Good": {
        'Age': 35, 'Annual_Income': 150000000, 'Monthly_Inhand_Salary': 12000000,
        'Num_Bank_Accounts': 2, 'Num_Credit_Card': 2, 'Interest_Rate': 10.0,
        'Num_of_Loan': 1, 'Delay_from_due_date': 2, 'Num_of_Delayed_Payment': 0,
        'Changed_Credit_Limit': 15.0, 'Num_Credit_Inquiries': 1,
        'Outstanding_Debt': 500000, 'Credit_Utilization_Ratio': 20.0,
        'Total_EMI_per_month': 50000, 'Amount_invested_monthly': 1000000,
        'Monthly_Balance': 5000000, 'Credit_History_Years': 10,
        'Occupation': 'Manager', 'Credit_Mix': 'Good', 
        'Payment_of_Min_Amount': 'Yes', 'Payment_Category': 'High'
    },
    "Poor": {
        'Age': 40, 'Annual_Income': 25000000, 'Monthly_Inhand_Salary': 2000000,
        'Num_Bank_Accounts': 5, 'Num_Credit_Card': 5, 'Interest_Rate': 25.0,
        'Num_of_Loan': 5, 'Delay_from_due_date': 30, 'Num_of_Delayed_Payment': 15,
        'Changed_Credit_Limit': -10.0, 'Num_Credit_Inquiries': 10,
        'Outstanding_Debt': 5000000, 'Credit_Utilization_Ratio': 45.0,
        'Total_EMI_per_month': 300000, 'Amount_invested_monthly': 100000,
        'Monthly_Balance': 200000, 'Credit_History_Years': 2,
        'Occupation': 'Musician', 'Credit_Mix': 'Bad', 
        'Payment_of_Min_Amount': 'No', 'Payment_Category': 'Low'
    },
    "Standard": {
        'Age': 28, 'Annual_Income': 60000000, 'Monthly_Inhand_Salary': 5000000,
        'Num_Bank_Accounts': 3, 'Num_Credit_Card': 3, 'Interest_Rate': 15.0,
        'Num_of_Loan': 2, 'Delay_from_due_date': 10, 'Num_of_Delayed_Payment': 3,
        'Changed_Credit_Limit': 5.0, 'Num_Credit_Inquiries': 3,
        'Outstanding_Debt': 1500000, 'Credit_Utilization_Ratio': 30.0,
        'Total_EMI_per_month': 120000, 'Amount_invested_monthly': 500000,
        'Monthly_Balance': 1500000, 'Credit_History_Years': 5,
        'Occupation': 'Engineer', 'Credit_Mix': 'Standard', 
        'Payment_of_Min_Amount': 'Yes', 'Payment_Category': 'Medium'
    }
}


# UI STREAMLIT
st.set_page_config(page_title="Credit Score Predictor", layout="wide")

st.title("Credit Score Prediction App")
st.markdown("""
Masukkan data nasabah di bawah ini untuk memprediksi **Credit Score** (Good, Poor, Standard).
""")

with st.expander("Tentang Aplikasi"):
    st.write("""
    Aplikasi ini menggunakan model **Random Forest** yang dilatih dengan data nasabah.
    Model memprediksi Credit Score berdasarkan berbagai fitur seperti:
    - Demografi (Umur, Pekerjaan)
    - Keuangan (Pendapatan, Hutang, Aset)
    - Riwayat Kredit (Jumlah pinjaman, keterlambatan pembayaran)
    """)


# SIDEBAR - TEST CASES
with st.sidebar:
    st.subheader("Test Cases")
    st.write("Klik tombol di bawah untuk mengisi form dengan data test case:")
    
    col_test1, col_test2, col_test3 = st.columns(3)
    
    if 'test_case' not in st.session_state:
        st.session_state['test_case'] = None
    
    with col_test1:
        if st.button("Good"):
            st.session_state['test_case'] = 'Good'
            st.rerun()
    
    with col_test2:
        if st.button("Standard"):
            st.session_state['test_case'] = 'Standard'
            st.rerun()
    
    with col_test3:
        if st.button("Poor"):
            st.session_state['test_case'] = 'Poor'
            st.rerun()
    
    if st.session_state['test_case']:
        test_data = TEST_CASES[st.session_state['test_case']]
        st.success(f"Test Case '{st.session_state['test_case']}' loaded!")
        with st.expander("Lihat Data Test Case"):
            st.json(test_data)


# FORM INPUT
st.subheader("Form Data Nasabah")

test_data = TEST_CASES.get(st.session_state['test_case'], {})

col1, col2 = st.columns(2)

with col1:
    st.markdown("Data Pribadi")
    age = st.number_input("Umur", min_value=18.0, max_value=100.0, 
                          value=float(test_data.get('Age', 30)), step=1.0)
    
    occupation = st.selectbox("Pekerjaan", [
        'Accountant', 'Developer', 'Doctor', 'Engineer', 'Entrepreneur',
        'Journalist', 'Manager', 'Musician', 'Scientist', 'Teacher'
    ], index=0 if 'Occupation' not in test_data else [
        'Accountant', 'Developer', 'Doctor', 'Engineer', 'Entrepreneur',
        'Journalist', 'Manager', 'Musician', 'Scientist', 'Teacher'
    ].index(test_data.get('Occupation', 'Accountant')))
    
    st.markdown("Keuangan")
    annual_income = st.number_input("Pendapatan Tahunan", min_value=0.0, 
                                    value=float(test_data.get('Annual_Income', 50000000)), 
                                    step=1000000.0, format="%.0f")
    monthly_inhand_salary = st.number_input("Gaji Bulanan (Take Home Pay)", min_value=0.0, 
                                            value=float(test_data.get('Monthly_Inhand_Salary', 5000000.0)), 
                                            step=100000.0)
    num_bank_accounts = st.number_input("Jumlah Rekening Bank", min_value=0, max_value=20, 
                                         value=int(test_data.get('Num_Bank_Accounts', 2)), step=1)
    num_credit_card = st.number_input("Jumlah Kartu Kredit", min_value=0, max_value=20, 
                                       value=int(test_data.get('Num_Credit_Card', 2)), step=1)

with col2:
    st.markdown("Riwayat Kredit")
    credit_mix = st.selectbox("Campuran Kredit", ['Good', 'Standard', 'Bad'],
                              index=['Good', 'Standard', 'Bad'].index(test_data.get('Credit_Mix', 'Standard')))
    payment_of_min_amount = st.selectbox("Pembayaran Minimal", ['Yes', 'No', 'NM'],
                                         index=['Yes', 'No', 'NM'].index(test_data.get('Payment_of_Min_Amount', 'Yes')))
    payment_category = st.selectbox("Kategori Pembayaran", ['Low', 'Medium', 'High'],
                                    index=['Low', 'Medium', 'High'].index(test_data.get('Payment_Category', 'Medium')))
    
    st.markdown("Detail Pinjaman")
    interest_rate = st.number_input("Suku Bunga (%)", min_value=0.0, max_value=100.0, 
                                    value=float(test_data.get('Interest_Rate', 15.0)), step=0.5)
    num_of_loan = st.number_input("Jumlah Pinjaman", min_value=0, max_value=50, 
                                  value=int(test_data.get('Num_of_Loan', 3)), step=1)
    num_of_delayed_payment = st.number_input("Jumlah Pembayaran Terlambat", min_value=0, max_value=100, 
                                              value=int(test_data.get('Num_of_Delayed_Payment', 5)), step=1)
    delay_from_due_date = st.number_input("Rata-rata Keterlambatan (hari)", min_value=0, max_value=60, 
                                           value=int(test_data.get('Delay_from_due_date', 10)), step=1)
    changed_credit_limit = st.number_input("Perubahan Batas Kredit (%)", min_value=-50.0, max_value=50.0, 
                                            value=float(test_data.get('Changed_Credit_Limit', 5.0)), step=0.5)
    num_credit_inquiries = st.number_input("Jumlah Pertanyaan Kredit", min_value=0, max_value=50, 
                                            value=int(test_data.get('Num_Credit_Inquiries', 3)), step=1)
    
    st.markdown("Utang & Investasi")
    outstanding_debt = st.number_input("Total Hutang", min_value=0.0, 
                                        value=float(test_data.get('Outstanding_Debt', 1000000.0)), 
                                        step=100000.0)
    credit_utilization_ratio = st.number_input("Rasio Penggunaan Kredit (%)", min_value=0.0, max_value=100.0, 
                                                value=float(test_data.get('Credit_Utilization_Ratio', 30.0)), 
                                                step=0.5)
    total_emi_per_month = st.number_input("Total EMI per Bulan", min_value=0.0, 
                                           value=float(test_data.get('Total_EMI_per_month', 100000.0)), 
                                           step=10000.0)
    amount_invested_monthly = st.number_input("Investasi Bulanan", min_value=0.0, 
                                               value=float(test_data.get('Amount_invested_monthly', 500000.0)), 
                                               step=50000.0)
    monthly_balance = st.number_input("Saldo Bulanan", min_value=0.0, 
                                       value=float(test_data.get('Monthly_Balance', 1000000.0)), 
                                       step=100000.0)
    credit_history_years = st.number_input("Lama Riwayat Kredit (tahun)", min_value=0, max_value=50, 
                                            value=int(test_data.get('Credit_History_Years', 5)), step=1)



# PREDIKSI
if st.button("Prediksi Credit Score", type="primary"):
    errors = []
    
    if annual_income <= 0:
        errors.append("Pendapatan Tahunan harus lebih dari 0")
    
    if monthly_inhand_salary <= 0:
        errors.append("Gaji Bulanan harus lebih dari 0")
    
    if outstanding_debt < 0:
        errors.append("Total Hutang tidak boleh negatif")
    
    if total_emi_per_month < 0:
        errors.append("Total EMI tidak boleh negatif")
    
    if errors:
        for error in errors:
            st.error(f"{error}")
    else:
        with st.spinner("Memproses data..."):
            input_dict = {
                'Age': age,
                'Annual_Income': annual_income,
                'Monthly_Inhand_Salary': monthly_inhand_salary,
                'Num_Bank_Accounts': num_bank_accounts,
                'Num_Credit_Card': num_credit_card,
                'Interest_Rate': interest_rate,
                'Num_of_Loan': num_of_loan,
                'Delay_from_due_date': delay_from_due_date,
                'Num_of_Delayed_Payment': num_of_delayed_payment,
                'Changed_Credit_Limit': changed_credit_limit,
                'Num_Credit_Inquiries': num_credit_inquiries,
                'Outstanding_Debt': outstanding_debt,
                'Credit_Utilization_Ratio': credit_utilization_ratio,
                'Total_EMI_per_month': total_emi_per_month,
                'Amount_invested_monthly': amount_invested_monthly,
                'Monthly_Balance': monthly_balance,
                'Debt_to_Income_Ratio': outstanding_debt / annual_income if annual_income > 0 else 0,
                'Min_Payment_Ratio': total_emi_per_month / monthly_inhand_salary if monthly_inhand_salary > 0 else 0,
                'Credit_History_Years': credit_history_years,
                'Occupation': occupation,
                'Credit_Mix': credit_mix,
                'Payment_of_Min_Amount': payment_of_min_amount,
                'Payment_Category': payment_category
            }
            
            input_df = pd.DataFrame([input_dict])
            
            feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
            input_df = input_df[feature_cols]
            
            input_processed = preprocessor.transform(input_df)
            
            prediction = model.predict(input_processed)[0]
            probabilities = model.predict_proba(input_processed)[0]
            
            predicted_label = encoder.inverse_transform([prediction])[0]
            
            color_map = {
                'Good': 'green',
                'Standard': 'orange',
                'Poor': 'red'
            }
            
            st.subheader("Hasil Prediksi")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.markdown(f"""
                <div style="
                    background-color: {color_map.get(predicted_label, 'gray')}20;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid {color_map.get(predicted_label, 'gray')};
                ">
                    <h2 style="color: {color_map.get(predicted_label, 'gray')};">
                        {predicted_label}
                    </h2>
                    <p style="font-size: 14px; color: #666;">
                        Credit Score nasabah diprediksi sebagai <b>{predicted_label}</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_result2:
                st.markdown("#### Probabilitas per Kelas")
                prob_df = pd.DataFrame({
                    'Kelas': encoder.classes_,
                    'Probabilitas': probabilities
                })
                st.dataframe(prob_df, hide_index=True, use_container_width=True)
                st.bar_chart(prob_df.set_index('Kelas'))
            
            st.subheader("Rekomendasi")
            
            recommendations = {
                'Good': """
                Nasabah memiliki profil kredit yang sangat baik.
                Direkomendasikan untuk mendapatkan penawaran produk premium.
                Limit kredit dapat dinaikkan.
                """,
                'Standard': """
                Nasabah memiliki profil kredit standar.
                Perlu monitoring rutin terhadap perilaku pembayaran.
                Dapat diberikan edukasi keuangan untuk meningkatkan skor.
                """,
                'Poor': """
                Nasabah memiliki profil kredit yang buruk.
                Perlu dilakukan evaluasi mendalam sebelum memberikan pinjaman.
                Disarankan program restrukturisasi utang.
                Monitoring ketat terhadap pembayaran.
                """
            }
            
            st.markdown(recommendations.get(predicted_label, ""))
            
            with st.expander("Detail Data yang Dimasukkan"):
                st.dataframe(input_df, use_container_width=True)


# FOOTER
st.markdown("---")
st.caption("Credit Score Prediction App")



