# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:05:58 2024

@author: n.nteits
"""

import numpy as np
import random
from faker import Faker
import pandas as pd



# Initialize Faker
faker = Faker()
Faker.seed(123)
random.seed(123)
# Number of records
num_patients = 100  # Number of patients
num_measurements_per_patient = 5  # Longitudinal measurements per patient
num_events_per_patient = 5  # Events per patient

# 1. Generate Patients Table
def generate_patients(num_patients):
    patients = []
    for _ in range(num_patients):
        patient_id = _
        gender = random.choice(["Male", "Female"])  # Randomly assign gender
        if gender == "Male":
           first_name = faker.first_name_male()
        else:
           first_name = faker.first_name_female()
        patients.append({
            "PatientID": patient_id,
            "FirstName": first_name,
            "LastName": faker.last_name(),
            "BirthDate": faker.date_of_birth(minimum_age=30, maximum_age=80),
            "Gender": gender,
            "Ethnicity": random.choice(["Caucasian", "Asian", "African American", "Hispanic", "Other"]),
            "BaselineDiagnosisDate": faker.date_between(start_date="-10y", end_date="today"),
            "PrimaryDiagnosis": random.choice(["Hypertension", "Coronary Artery Disease", "Arrhythmia"]),
            "SmokingStatus": random.choice(["Never", "Current", "Former"]),
            "AlcoholUse": random.choice(["Yes", "No", "Occasional"]),
            "BaselineBMI": round(random.uniform(18.5, 35.0), 1),
           
        })
    return pd.DataFrame(patients)



# 2. Generate Clinical Measurements Table
def generate_clinical_measurements(patients, num_measurements):
    measurements = []
    for patient in patients.itertuples():
        #Alter generated values according to Sex
        
                  
        if patient.Gender=='Male' :
            for _ in range(num_measurements):
                measurements.append({
                    "MeasurementID": _ ,
                    "PatientID": patient.PatientID,
                    "MeasurementDate": faker.date_between(start_date=patient.BaselineDiagnosisDate, end_date='today'),
                    "SystolicBP": random.randint(110, 140),
                    "DiastolicBP": random.randint(60, 90),
                    "TotalCholesterol": random.randint(160, 240),
                    "LDL": random.randint(100, 160),
                    "HDL": random.randint(40, 60),
                    "Triglycerides": random.randint(100, 200),
                    "HeartRate": random.randint(60, 80),
                    "CRP": round(random.uniform(1.0, 4.0), 2),
                    "HbA1c": round(random.uniform(5.0, 6.0), 1),
                    "Weight": round(random.uniform(60, 100), 1)
                    })
        else :   
            for _ in range(num_measurements):
                measurements.append({
                    "MeasurementID": _ ,
                    "PatientID": patient.PatientID,
                    "MeasurementDate": faker.date_between(start_date=patient.BaselineDiagnosisDate, end_date='today'),
                    "SystolicBP": random.randint(105, 135),
                    "DiastolicBP": random.randint(65, 95),
                    "TotalCholesterol": random.randint(155, 220),
                    "LDL": random.randint(85, 120),
                    "HDL": random.randint(55, 75),
                    "Triglycerides": random.randint(80, 180),
                    "HeartRate": random.randint(70, 90),
                    "CRP": round(random.uniform(1.5, 4.5), 2),
                    "HbA1c": round(random.uniform(4.5, 5.5), 1),
                    "Weight": round(random.uniform(50, 90), 1)
       
                                                    })
    return pd.DataFrame(measurements)




# 3. Generate Events and Treatments Table
def generate_events_and_treatments(patients, num_events):
    events = []
    for patient in patients.itertuples():
        for _ in range(num_events):
            #Initialise diagnosis date to draw new instance for the 1st measurement
            if _==0:
                s_date=patient.BaselineDiagnosisDate
                
            Eventtp=random.choice(["Heart Attack", "Stroke", "Hospitalization","Death"])
            event_date=faker.date_between(start_date=s_date, end_date="today")
            events.append({
                "EventID": _,
                "PatientID": patient.PatientID,
                "EventDate": event_date,
                "EventType": Eventtp,
               # "TreatmentType": random.choice(["Angioplasty", "Medication Adjustment", "Bypass Surgery"]),
               # "MedicationPrescribed": random.choice(["Statins", "Beta Blockers", "ACE Inhibitors", "None"]),
               # "Adherence": random.choice(["Yes", "No", "Partial"]),
                "Notes": faker.text(max_nb_chars=50)
            })
            #Set the new starting date as the previous drawn date in order to make the drawn dates ascending 
            s_date=event_date
            #In case of death event go to the next patient
            if Eventtp=="Death":
                break
    return pd.DataFrame(events)

def induce_noise(clinical_measurements_df):
    #Inject noise 
    for p in range(len(clinical_measurements_df)) :      
        for j in clinical_measurements_df.columns[3:]:
            if random.randint(0, 19)<2: 
                clinical_measurements_df.loc[p,j]*=10
            elif random.randint(0, 19)>18 :
                clinical_measurements_df.loc[p,j]+=round(np.random.randn()*5,0)
    return clinical_measurements_df

def censor_rows(measurements) :
    #Drop 20% of rows randomly to induce missingness
    for p in range(len(measurements)):
        if random.randint(0, 9)<2: 
            measurements.drop(index=p,inplace=True)
    measurements.reset_index(drop=True, inplace=True)        
    return measurements            
            
def censor_values(measurements) :
    #Turn 10 % of the values to na in order to create missing data points
    for p in range(len(measurements)):
        for j in measurements.columns[3:]:
            if random.randint(0, 19)<2: 
                measurements.loc[p,j]=np.nan
                 
    return measurements               





# Generate Data
#Patients
patients_df = generate_patients(num_patients)
#Event data
events_and_treatments_df = generate_events_and_treatments(patients_df, num_events_per_patient)
#Longitudinal measurements
clinical_measurements_df = generate_clinical_measurements(patients_df, num_measurements_per_patient)



#Find death dates of patients in order to drop measurements with date later than their death date
death_date=events_and_treatments_df[events_and_treatments_df['EventType']=='Death'][['PatientID','EventDate']]
death_date.reset_index(drop=True, inplace=True)
#Merge death_date to measurements dataframe
merged_df = clinical_measurements_df.merge(
    death_date,
    on='PatientID',
    suffixes=('', '_death'),how='left'
)
#Create drop indicator
merged_df['Drop'] = (merged_df['MeasurementDate'] > merged_df['EventDate'])
#Drop reset index
clinical_measurements_final_df = merged_df[~merged_df['Drop']].drop(columns=['Drop', 'EventDate'])
clinical_measurements_final_df.reset_index(drop=True, inplace=True)



#induce noise
clinical_measurements_final_df=induce_noise(clinical_measurements_final_df)
#create censored longitudinal data by dropping rows 
clinical_measurements_final_df=censor_rows(clinical_measurements_final_df)
#create missing measurements by randomly setting to na values in the measurement dataframe
clinical_measurements_final_df=censor_values(clinical_measurements_final_df)

clinical_measurements_final_df.to_csv("clinical_measurements.csv", index=False) 
patients_df.to_csv("patients_df.csv", index=False) 
events_and_treatments_df.to_csv("events_and_treatments.csv", index=False) 
