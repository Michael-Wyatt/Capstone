# Capstone.py
# Michael Wyatt
# Data Science and Analytics Bootcamp
# Capstone Project - Predicting COVID-19 ICU Admissions

# ----------------------
# ---IMPORT LIBRARIES---
# ----------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report

# ---------------
# ---FUNCTIONS---
# ---------------

def wait_for_enter_key():
    # Function to wait for the user to press the enter key to continue.
    print("Press the enter key to continue...")
    input()

def introductionScreen(title, description):
    # Function to display an introduction screen for the following visualization
    plt.figure(figsize=(12, 6))
    plt.text(0.5, 0.7, title, ha='center', va='center', fontsize=24, fontweight='bold')
    plt.text(0.5, 0.5, description, ha='center', va='center', fontsize=14)
    plt.gca().set_facecolor('white')
    plt.axis('off')
    plt.show()

# -----------------
# ---DATA IMPORT---
# -----------------

# Load the CSV file
filePath = 'C:/Capstone/Kaggle_Sirio_Libanes_ICU_Prediction.csv'
df = pd.read_csv(filePath)
print("CSV file import completed successfully.")

# ----------------
# ---DATA CLEAN---
# ----------------

# Remove the Patient Visit Identifier column
df = df.drop(['PATIENT_VISIT_IDENTIFIER'], axis=1)

# Confirm data
# print(df.describe())

# ------------------------
# ---SHOWING THE TABLES---
# ------------------------

# -----------------------------
# ---Count of ICU Admissions---
# -----------------------------


title = "Count of ICU Admissions"
description = (
    "In this first visualization, we will compare the number of ER visits to the number of ICU admissions.\n"
    "This will show how many patients are being sent to ICU with COVID-19 complications.\n"    
)

introductionScreen(title, description)

# Visualization 1: Countplot of ICU Admissions
plt.figure(figsize=(12, 8))
sns.countplot(x='ICU', data=df)
plt.title('Count of ICU Admissions')
plt.show()

# -------------------------------------
# ---Distribution of Age Percentiles---
# -------------------------------------

title = "Distribution of Age Percentiles"
description = (
    "In this visualization, we will compare the age percentiles of patients.\n"
    "Note that this statistic does not show actual or specific ages of patients.\n"
    "It represents the age percentile ranking of patients within the dataset,\n"
    "indicating where the age falls relative to others.\n"
    "For example, a value in the 20th percentile means the patient is younger than 80 percent of others,\n"
    "not the exact age of the patient."
)

introductionScreen(title, description)

df['AGE_PERCENTIL'].value_counts()

# Define the order of the categories including "Above 90th"
age_order = ['10th', '20th', '30th', '40th', '50th', '60th', '70th', '80th', '90th', 'Above 90th']

# Convert AGE_PERCENTIL to an ordered categorical type
df['AGE_PERCENTIL'] = pd.Categorical(df['AGE_PERCENTIL'], categories=age_order, ordered=True)

# Plot the histogram
plt.figure(figsize=(12, 8))
sns.histplot(data=df, x='AGE_PERCENTIL')
plt.title('Distribution of Age Percentiles')
# plt.xlabel('Age Percentile')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()

# -------------------------------------------------------------------------------------------
# ---Pairplot for selected variables to visualize pairwise relationships and distributions---
# -------------------------------------------------------------------------------------------

title = "Pairplot of Age, Blood Pressure, and Heart Rate"
description = (
    "This pairplot visualizes the pairwise relationships and distributions between variables\n"
    "like age (above 65), systolic blood pressure, and heart rate, categorized by ICU admission status.\n"
    "The plot helps to identify patterns or correlations between these health indicators and ICU outcomes,\n"
    "with different colors representing whether a patient was admitted to the ICU."
)

introductionScreen(title, description)

# Create a pairplot for selected variables to visualize pairwise relationships and distributions
sns.pairplot(df[['AGE_ABOVE65', 'BLOODPRESSURE_SISTOLIC_MEDIAN', 'HEART_RATE_MEDIAN', 'ICU']], hue='ICU')
# Display the plot
plt.show()


# ---------------------------------------------------------
# ---Violin plot of Age and Gender within ICU admissions---
# ---------------------------------------------------------

title = "Violin plot of Age and Gender within ICU admissions"
description = (
    "This violin plot compares the age distribution of patients aged above 65 across gender and ICU admission status.\n"
    "The split sections of the violin show the distribution for those admitted to the ICU\n"
    "and those not admitted, allowing for a clear comparison of age patterns between genders within each group.\n"
    "\nNote that the genders are not identified or unspecified in the dataset."
)

introductionScreen(title, description)

# Set figure size for better visibility
plt.figure(figsize=(8, 5))
# Create a violin plot to compare distributions of age across gender within ICU admissions
sns.violinplot(x='GENDER', y='AGE_ABOVE65', hue='ICU', data=df, split=True)
# Title of the plot
plt.title('Age Distribution by Gender and ICU Admission')
# Label the x-axis
plt.xlabel('Gender')
# Label the y-axis
plt.ylabel('Age Above 65')
# Display the plot
plt.show()

# -----------------------------------------
# ---Countplots of each Disease Grouping---
# -----------------------------------------

title = "Countplots of each Disease Grouping"
description = (
    "This set of countplots displays the frequency of occurrences for each Disease Grouping category,\n"
    "with bars color-coded by ICU admission status.\n"
    "Each subplot represents one of the six disease groupings,\n"
    "allowing for a clear comparison of how each disease grouping relates to ICU admissions."
)

introductionScreen(title, description)

# List of disease grouping columns to visualize
disease_groupings = ['DISEASE GROUPING 1', 'DISEASE GROUPING 2', 'DISEASE GROUPING 3', 
                     'DISEASE GROUPING 4', 'DISEASE GROUPING 5', 'DISEASE GROUPING 6']

# Set up a grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

# Flatten the axes array for easy indexing
axes = axes.flatten()

# Loop over each disease grouping and create a countplot
for i, grouping in enumerate(disease_groupings):
    sns.countplot(x=grouping, hue='ICU', data=df, ax=axes[i])
    axes[i].set_title(f'{grouping} by ICU Admission', fontsize=14)
    axes[i].set_xlabel(grouping, fontsize=12)
    axes[i].set_ylabel('Count', fontsize=12)
    
    # Add data labels
    for container in axes[i].containers:
        axes[i].bar_label(container, label_type='edge')

    # Rotate x-axis labels properly without warning
    axes[i].tick_params(axis='x', rotation=45)

    # Remove individual legends
    axes[i].get_legend().remove()

# Set consistent y-axis limits
for ax in axes:
    ax.set_ylim(0, df['ICU'].value_counts().max())

# Add overall title
fig.suptitle('Frequency of Disease Groupings by ICU Admission', fontsize=16, y=1.02)

# Add a single legend for the entire figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize=12)

# Adjust layout for better visibility
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# -----------------------------------------------------------------
# ---Bar Plot of Median Systolic Blood Pressure by ICU Admission---
# -----------------------------------------------------------------

title = "Median Systolic Blood Pressure by ICU Admission"
description = (
    "This bar plot shows the median systolic blood pressure for patients categorized by ICU admission status.\n"
    "The x-axis labels indicate whether patients were admitted or not admitted to the ICU,\n"
    "allowing for a comparison of blood pressure between the two groups."
)

introductionScreen(title, description)

# Set figure size for better visibility
plt.figure(figsize=(10, 6))
# Create a bar plot to compare median vital signs for ICU admitted vs non-admitted patients
sns.barplot(x='ICU', y='BLOODPRESSURE_SISTOLIC_MEDIAN', data=df)
# Title of the plot
plt.title('Median Systolic Blood Pressure by ICU Admission')
# Label the x-axis
plt.xlabel('ICU Admission')
# Label the y-axis
plt.ylabel('Median Systolic Blood Pressure')
# Set the custom labels for the x-axis
plt.xticks([0, 1], ['Not Admitted', 'Admitted'])
# Display the plot
plt.show()

# -------------------------------------------------------------
# ---lmplot of trends between Glucose Median and Glucose Max---
# -------------------------------------------------------------

title = "lmplot of trends between Glucose Median and Glucose Max"
description = (
    "This  visualization is a scatter plot with a regression line showing the relationship between glucose median and glucose max\n"
    "with data points colored by ICU admission status. Different markers represent ICU-admitted and non-admitted patients,\n"
    "allowing for a comparison of glucose variability between the two groups."
)

introductionScreen(title, description)

# Create an lmplot to see trends between glucose median and glucose max, colored by ICU status
sns.lmplot(x='GLUCOSE_MEDIAN', y='GLUCOSE_MAX', hue='ICU', data=df, markers=["o", "s"], palette="Set1")
# Add a title
plt.title('Trends Between Glucose Median and Glucose Max by ICU Admission')
# Display the plot
plt.show()

# -------------------------------------------------------------------------------------------------------
# ---Distribution plot for Creatinine Median levels showing different distributions for ICU admissions---
# -------------------------------------------------------------------------------------------------------

title = "Distribution plot for Creatinine Median levels"
description = (
    "This visualization is a kernel density estimate (KDE) plot that shows\n"
    "the distribution of creatinine median levels for patients,\n"
    "with separate curves for those who were admitted to the ICU and those who were not."
)

introductionScreen(title, description)

# Set figure size for better visibility
plt.figure(figsize=(10, 6))
# Create a distribution plot for Creatinine Median levels, showing different distributions for ICU admissions
sns.displot(df, x='CREATININ_MEDIAN', kind='kde', hue='ICU', fill=True, palette="Set2", rug=True)
# Calculate and annotate the medians
median_ICU = df[df['ICU'] == 1]['CREATININ_MEDIAN'].median()
median_non_ICU = df[df['ICU'] == 0]['CREATININ_MEDIAN'].median()
# Add vertical lines for medians
plt.axvline(median_ICU, color='blue', linestyle='--', label='ICU Median')
plt.axvline(median_non_ICU, color='green', linestyle='--', label='Non-ICU Median')
# Improve the title and labels
plt.title('Distribution of Creatinine Median Levels by ICU Admission', fontsize=16)
plt.xlabel('Creatinine Median (mg/dL)', fontsize=12)
plt.ylabel('Density', fontsize=12)
# Customize the legend to show 'Not Admitted' and 'Admitted'
plt.legend(title='ICU Admission', labels=['Not Admitted', 'Admitted'])
# Display the plot
plt.show()

# ------------------------------------------------------------------------------------------
# ---FacetGrid to show histograms of heart rate medians separated by ICU admission status---
# ------------------------------------------------------------------------------------------

title = "FacetGrid of histograms of Heart Rate Medians"
description = (
    "This visualization displays two histograms, one for patients who were admitted to the ICU\n"
    "and another for those not admitted.\n"
    "Each histogram shows the distribution of median heart rate values for the respective group."
)

introductionScreen(title, description)

# Create a FacetGrid to show histograms of heart rate medians separated by ICU admission status
g = sns.FacetGrid(df, col='ICU', height=4, aspect=1.5)
# Map a histogram to the grids
g.map_dataframe(sns.histplot, 'HEART_RATE_MEDIAN')
# Define a custom title mapping for the ICU status
icu_labels = {0: 'Not admitted to ICU', 1: 'Admitted to ICU'}
# Set custom titles based on ICU status
g.set_titles("{col_name}", size=12)
for ax, title in zip(g.axes.flat, g.col_names):
    ax.set_title(icu_labels[int(title)])
# Display the plots
plt.show()

# --------------------------------------------------------------------------------------
# ---Scatter Plot of Relationship Between Systolic BP Median and Max by ICU Admission---
# --------------------------------------------------------------------------------------

title = "Relationship Between Systolic BP Median and Max by ICU Admission"
description = (
    "This scatter plot visualizes the relationship between the median systolic blood pressure\n"
    "and maximum systolic blood pressure for patients, with data points color-coded by ICU admission status.\n"
    "Patients admitted to the ICU will be shown in orange, while non-ICU patients will be in blue.\n"
    "This plot helps identify patterns or trends in how systolic blood pressure varies between median and maximum values\n"
    "and whether these trends differ based on ICU admission."
)

introductionScreen(title, description)

# Use relplot to create a scatter plot with hue for ICU admission
sns.relplot(
    data=df,
    x='BLOODPRESSURE_SISTOLIC_MEDIAN',
    y='BLOODPRESSURE_SISTOLIC_MAX',
    hue='ICU',  # Color by ICU status
    kind='scatter',
    height=5,  # Specify the height of the plot
    aspect=1.5  # Specify the aspect ratio of the plot
)
plt.title('Relationship Between Systolic BP Median and Max by ICU Admission')
plt.show()

# --------------------------------------------------------------------------------------
# ---Scatter Plot of Relationship Between Systolic BP Median and Max by ICU Admission---
# --------------------------------------------------------------------------------------

title = "Relationship Between Systolic BP Median and Max by ICU Admission"
description = (
    "This  grid of scatter plots shows the relationship between Median Heart Rate and Median Respiratory Rate for patients.\n"
    "The grid is split by ICU Admission Status and Gender."
)

introductionScreen(title, description)

# Set up a FacetGrid to plot multiple vital signs
g = sns.FacetGrid(df, col="ICU", row="GENDER", height=4, aspect=1.5)
g.map_dataframe(sns.scatterplot, x='HEART_RATE_MEDIAN', y='RESPIRATORY_RATE_MEDIAN')
g.set_axis_labels('Median Heart Rate (bpm)', 'Median Respiratory Rate (breaths per min)')
g.set_titles(col_template='ICU Admission: {col_name}', row_template='Gender: {row_name}')
g.add_legend()
plt.show()

# ----------------------
# ---MACHINE LEARNING---
# ----------------------

# Clean up the data:
# Example to identify columns containing string ranges
for column in df.columns:
    if df[column].dtype == 'object':  # Assuming object type denotes potentially problematic columns
        if any('-' in str(x) for x in df[column].unique()):
            print(f"Column '{column}' contains range values.")

# Separate features and target
X = df.drop('ICU', axis=1)
y = df['ICU']

# Identify non-numeric columns (all columns that are not of numeric type)
non_numeric_columns = X.select_dtypes(exclude=['int', 'float']).columns.tolist()

# Create a column transformer with OneHotEncoder for non-numeric columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), non_numeric_columns)
    ],
    remainder='passthrough'  # Pass through other columns as is
)

# Create a pipeline that includes preprocessing and the classifier
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Fit the model on the training data
clf.fit(X_train, y_train)

# Predict probabilities for the test data
y_probs = clf.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

# Predict class labels for the test data
y_pred = clf.predict(X_test)

# Compute AUROC
# The AUROC (Area Under the Receiver Operating Characteristic Curve) is computed to evaluate the model’s ability to distinguish between positive and negative classes.
roc_auc = roc_auc_score(y_test, y_probs)
print(f"Area Under the ROC Curve (AUROC): {roc_auc:.4f}")

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Display the classification report
print(classification_report(y_test, y_pred))

# Compute ROC curve points
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# ---------------
# ---ROC Curve---
# ---------------

title = "ROC Curve: Predicting ICU Admission Using Random Forest Classifier"
description = (
    "This ROC curve shows the trade-off between the true positive rate (sensitivity) and the false positive rate,\n"
    "with the diagonal line representing a random classifier. The closer the curve is to the top-left corner, the better the model."
)

introductionScreen(title, description)

# Plot the ROC curve using Seaborn
plt.figure(figsize=(8, 6))
sns.lineplot(x=fpr, y=tpr, estimator=None, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# -------------------------------------
# ---ROC Curve with cross-validation---
# -------------------------------------

title = "ROC Curve: Predicting ICU Admission Using Cross-Validation with StratifiedKFold"
description = (
    "This ROC curve also shows the trade-off between the true positive rate (sensitivity) and the false positive rate,\n"
    "with the diagonal line representing a random classifier. The closer the curve is to the top-left corner, the better the model."
    "However, the previous ROC curve was generated using a train-test split, and this one uses cross-validation with StratifiedKFold."
)

# Separate features and target
X = df.drop('ICU', axis=1)
y = df['ICU']
# Identify non-numeric columns
non_numeric_columns = X.select_dtypes(exclude=['int', 'float']).columns.tolist()
# Create a column transformer with OneHotEncoder for non-numeric columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), non_numeric_columns)
    ],
    remainder='passthrough'  # Pass through other columns as is
)
# Create a pipeline that includes preprocessing and the classifier
classifier = RandomForestClassifier(random_state=42)
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])
# Define the cross-validation strategy
cv = StratifiedKFold(n_splits=3)
# Use cross_val_predict to generate cross-validated estimates of each class's probability
y_probs = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
# Predict class labels for cross-validation
y_pred = cross_val_predict(clf, X, y, cv=cv)
# Compute AUROC
roc_auc = roc_auc_score(y, y_probs)
print(f"Area Under the ROC Curve (AUROC): {roc_auc:.4f}")
# Compute accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.4f}")
# Display the classification report
print(classification_report(y, y_pred))
# Compute ROC curve points
fpr, tpr, thresholds = roc_curve(y, y_probs)

introductionScreen(title, description)

# Plot the ROC curve using Seaborn
plt.figure(figsize=(8, 6))
sns.lineplot(x=fpr, y=tpr, estimator=None, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# -------------------------------
# ---Top 20 Important Features---
# -------------------------------

title = "ROC Curve: Predicting ICU Admission Using Cross-Validation with StratifiedKFold"
description = (
    "This barplot shows the top 20 features that are most important in predicting the ICU admission based on the Random Forest model."
)


num_features = 20
# Fit the model
clf.fit(X, y)
# Extract feature importances from the classifier
feature_importances = clf.named_steps['classifier'].feature_importances_
# Get feature names from the ColumnTransformer
feature_names = np.concatenate(
    [preprocessor.named_transformers_['cat'].get_feature_names_out(non_numeric_columns),
     X.select_dtypes(include=['int', 'float']).columns])
# Map feature importances to their corresponding feature names
feature_importance_dict = dict(zip(feature_names, feature_importances))
# Sort features by importance and take the top 15
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:num_features]
# Split the tuples into lists
sorted_feature_names, sorted_importances = zip(*sorted_features)

introductionScreen(title, description)

# Plotting
plt.figure(figsize=(10, 8))
sns.barplot(x=sorted_importances, y=sorted_feature_names)
plt.title(f'Top {num_features} Feature Importances in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

'''
Summary:
This program trains a Random Forest classifier to predict ICU admission using both numeric and categorical features.
We preprocess the data using a ColumnTransformer to apply OneHotEncoding to categorical variables and pass through numeric features.
After training the model, we extract feature importances to identify which features contribute the most to the model’s predictions.
We retrieve and map the feature names, including encoded categorical features, and sort them by importance.
Finally, we visualize the top 20 most important features using a bar plot, which helps in understanding the key drivers of the Random Forest model's decisions.

~Michael Wyatt 9/7/2024
'''