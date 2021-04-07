# Stroke Prediction
This a machine learning project used for predicting stroke.

# Status
**Alpha**. model still overfitted.

# Dataset
[https://www.kaggle.com/joshuaswords/predicting-a-stroke-shap-lime-explainer-eli5]()

# Dataset Decription
## Context

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.
This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.

## Attribute Information

1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient

## Acknowledgements

(Confidential Source) - Use only for educational purposes
If you use this dataset in your research, please credit the author.

## Last Result

```bash

Train set
~~~~~~~~~
[[3181  134]
 [  20 3247]]
accuracy	:0.9766028562746886
precision	:0.9603667553978112
recall	    :0.9938781756963575
f1		    :0.9768351383874849

Validation set
~~~~~~~~~
[[317  25]
 [  5 385]]
accuracy	:0.9590163934426229
precision	:0.9390243902439024
recall     	:0.9871794871794872
f1		    :0.9625

Test set
~~~~~~~~~
[[1114   90]
 [  59   15]]
accuracy	:0.8834115805946792
precision	:0.14285714285714285
recall	    :0.20270270270270271
f1		    :0.1675977653631285
```