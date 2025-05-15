import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

model=pk.load(open('model.pkl','rb'))
st.header("Car Price Prediction Model")


df=pd.read_csv("Cardetails.csv")


# Araba Markasını almak sadece
def brand_car(car_name):
    car_name=car_name.split(' ')[0]
    return car_name.strip(' ')


df["name"]=df["name"].apply(brand_car)


name=st.selectbox("Select Car Brand",df["name"].unique())
year=st.slider("Car Manufactured Year",1994,2024)
km_driven=st.slider("No of kms Driven",10,250000)
fuel=st.selectbox("Fuel Type",df["fuel"].unique())
seller_type=st.selectbox("Seller Type",df["seller_type"].unique())
transmission=st.selectbox("Transmission type",df["transmission"].unique())
owner=st.selectbox("Which owner are you?",df["owner"].unique())
mileage=st.slider("Car Mileage",10,45)
engine=st.slider("Engine CC",650,4000)
max_power=st.slider("Max Power",10,400)


if st.button("Predict"):
  input_data_model = pd.DataFrame(
    [[name, year,km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power]],
    columns=["name", "year", "km_driven","fuel","seller_type","transmission","owner","mileage", "engine", "max_power"]
  )
 
 
  input_data_model["transmission"]=input_data_model["transmission"].replace(['Manual',"Automatic"],[1,2])
  input_data_model["fuel"] = input_data_model["fuel"].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4]).astype(int)
  input_data_model["seller_type"]=input_data_model["seller_type"].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3]).astype(int)
  input_data_model["owner"]=input_data_model["owner"].replace(['First Owner', 'Second Owner', 'Third Owner','Fourth & Above Owner', 'Test Drive Car'],
      [1,2,3,4,5]).astype(int)
  brand_map = {
    'Maruti':1,'Skoda':2,'Honda':3,'Hyundai':4,'Toyota':5,'Ford':6,'Renault':7,'Mahindra':8,
    'Tata':9,'Chevrolet':10,'Fiat':11, 'Datsun':12, 'Jeep':13,'Mercedes-Benz':14,'Mitsubishi':15,'Audi':16,
    'Volkswagen':17,'BMW':18,'Nissan':19,'Lexus':20,'Jaguar':21,'Land':22, 'MG':23, 'Volvo':24,
    'Daewoo':25,'Kia':26, 'Force':27, 'Ambassador':28, 'Ashok':29, 'Isuzu':30, 'Opel':31, 'Peugeot':32,
    
  }

  input_data_model["name"] = input_data_model["name"].replace(brand_map)
  

  car_price=model.predict(input_data_model)
  st.markdown("Car price is going to be "+ str(car_price))
  
