import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score


train_features = pd.read_csv('training_set_features.csv')
test_features = pd.read_csv('test_set_features.csv')
train_labels = pd.read_csv('training_set_labels.csv')
submission_format = pd.read_csv('submission_format.csv')


feature_cols = [
    'xyz_concern', 'xyz_knowledge', 'behavioral_antiviral_meds', 'behavioral_avoidance',
    'behavioral_face_mask', 'behavioral_wash_hands', 'behavioral_large_gatherings',
    'behavioral_outside_home', 'behavioral_touch_face', 'doctor_recc_xyz', 'doctor_recc_seasonal',
    'chronic_med_condition', 'child_under_6_months', 'health_worker', 'health_insurance',
    'opinion_xyz_vacc_effective', 'opinion_xyz_risk', 'opinion_xyz_sick_from_vacc',
    'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc',
    'age_group', 'education', 'race', 'sex', 'income_poverty', 'marital_status', 'rent_or_own',
    'employment_status', 'hhs_geo_region', 'census_msa', 'household_adults', 'household_children',
    'employment_industry', 'employment_occupation'
]

for col in feature_cols:
    if col not in train_features.columns:
        raise ValueError(f"Column {col} is not present in the training features dataset")
    if col not in test_features.columns:
        raise ValueError(f"Column {col} is not present in the test features dataset")

X = train_features[feature_cols]
y = train_labels[['xyz_vaccine', 'seasonal_vaccine']]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = MultiOutputClassifier(RandomForestClassifier(random_state=42))

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])

clf.fit(X_train, y_train)

y_pred_proba = clf.predict_proba(X_val)

roc_auc_xyz = roc_auc_score(y_val['xyz_vaccine'], y_pred_proba[0][:, 1])
roc_auc_seasonal = roc_auc_score(y_val['seasonal_vaccine'], y_pred_proba[1][:, 1])
mean_roc_auc = np.mean([roc_auc_xyz, roc_auc_seasonal])

print(f'Mean ROC AUC Score: {mean_roc_auc}')

X_test_final = test_features[feature_cols]
y_test_pred_proba = clf.predict_proba(X_test_final)

submission = pd.DataFrame({
    'respondent_id': test_features['respondent_id'],
    'xyz_vaccine': y_test_pred_proba[0][:, 1],
    'seasonal_vaccine': y_test_pred_proba[1][:, 1]
})

submission.to_csv('submission.csv', index=False)
