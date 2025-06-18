from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# FastAPI uygulama nesnesini oluştur
app = FastAPI()

# Kayıtlı model ve scaler'ı uygulama başlarken yükle
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# API'ye gönderilecek veri formatını tanımlayan Pydantic modeli
# Python'da standart olan küçük harf ve alt çizgili isimleri kullanıyoruz,
# 'alias' ile de modelin beklediği gerçek sütun adını belirtiyoruz.
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    PaperlessBilling: int
    MonthlyCharges: float
    TotalCharges: float
    multiple_lines_no_phone_service: bool = Field(alias='MultipleLines_No_phone_service')
    multiple_lines_yes: bool = Field(alias='MultipleLines_Yes')
    internet_service_fiber_optic: bool = Field(alias='InternetService_Fiber_optic')
    internet_service_no: bool = Field(alias='InternetService_No')
    online_security_no_internet_service: bool = Field(alias='OnlineSecurity_No_internet_service')
    online_security_yes: bool = Field(alias='OnlineSecurity_Yes')
    online_backup_no_internet_service: bool = Field(alias='OnlineBackup_No_internet_service')
    online_backup_yes: bool = Field(alias='OnlineBackup_Yes')
    device_protection_no_internet_service: bool = Field(alias='DeviceProtection_No_internet_service')
    device_protection_yes: bool = Field(alias='DeviceProtection_Yes')
    tech_support_no_internet_service: bool = Field(alias='TechSupport_No_internet_service')
    tech_support_yes: bool = Field(alias='TechSupport_Yes')
    streaming_tv_no_internet_service: bool = Field(alias='StreamingTV_No_internet_service')
    streaming_tv_yes: bool = Field(alias='StreamingTV_Yes')
    streaming_movies_no_internet_service: bool = Field(alias='StreamingMovies_No_internet_service')
    streaming_movies_yes: bool = Field(alias='StreamingMovies_Yes')
    contract_one_year: bool = Field(alias='Contract_One_year')
    contract_two_year: bool = Field(alias='Contract_Two_year')
    payment_method_credit_card_automatic: bool = Field(alias='PaymentMethod_Credit_card_(automatic)')
    payment_method_electronic_check: bool = Field(alias='PaymentMethod_Electronic_check')
    payment_method_mailed_check: bool = Field(alias='PaymentMethod_Mailed_check')

# Ana sayfa endpoint'i
@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API!"}

# Tahmin yapacak olan endpoint
@app.post("/predict")
def predict_churn(customer_data: CustomerData):
    # Gelen Pydantic verisini, 'alias'ları kullanarak bir dictionary'ye çevir
    data_dict = customer_data.dict(by_alias=True)
    
    # Bu dictionary'yi bir Pandas DataFrame'e dönüştür
    df = pd.DataFrame([data_dict])
    
    # DataFrame'i scaler ile ölçeklendir
    df_scaled = scaler.transform(df)
    
    # Model ile tahmini yap
    prediction = model.predict(df_scaled)
    probability = model.predict_proba(df_scaled)
    
    # Sonucu hazırla
    churn_status = "Will Churn" if prediction[0] == 1 else "Will Not Churn"
    churn_probability = probability[0][1]

    return {
        "prediction_status": churn_status,
        "churn_probability_percent": f"{churn_probability:.2%}"
    }