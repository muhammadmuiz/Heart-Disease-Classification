#!/usr/bin/env python
# coding: utf-8

# # Predicting heart disease using Machine Learning
# 
# This notebook looks into using various python based machine learning and data science libraries in an attempt to build a machine
# learning model capable of predicting whether someone has heart disease based on their medical attributes
# 
# We're going to take the following approach
# 1. Problem defination
# 2. Data
# 3. Evaluation
# 4. Features
# 5. Modelling
# 6. Experimentation
# 
# #### 1. Problem Defination
# 
# In a statement,
# > Given clinical parameters about a patient, can we predict whether or not they have heart disease?
# 
# #### 2. Data
# 
# The original data come from Cleavland data from the uci machine learning repository
# https://archive.ics.uci.edu/ml/datasets/heart+disease.
# There is also a version of it available in kaggle https://www.kaggle.com/ronitf/heart-disease-uci
# 
# #### 3. Evaluation
# 
# > If we can reach 90% of the accuracy at predicting whether the patient has heart disease or not during the proof of concept, we will persue the project.
# 
# #### 4. Features
# 
# **Create data dictionary**
# 
# * age
# * sex
# * chest pain type (4 values)
# * resting blood pressure
# * serum cholestoral in mg/dl
# * fasting blood sugar > 120 mg/dl
# * resting electrocardiographic results (values 0,1,2)
# * maximum heart rate achieved
# * exercise induced angina
# * oldpeak = ST depression induced by exercise relative to rest
# * the slope of the peak exercise ST segment
# * number of major vessels (0-3) colored by flourosopy
# * thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# 

# ## Preparing the tools
# We're going to use Pandas, Numpy and matplotlib for data analysis and manipulation

# In[1]:


# Import all the tools we need

# Import EDA(Exploratory data analysis) tool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# We want to see our plots inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Models from scikit learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluation
from sklearn.model_selection import train_test_split # model_selection is blueprint for analyzing data, so we can apply best model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve


# ## Load Data

# In[2]:


df = pd.read_csv("data/heart-disease.csv")
df.shape


# ### Data exploration (Exploratory data analysis(EDA))
# 1. What questions are you trying to solve?
# 2. What kind of data we have and how do we treat different types?
# 3. What's missing with the data and how you deals with it?
# 4. What are the outliers and why you should care about them?
# 5. How can you add, change or remove features to get more out of your data

# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df["target"].value_counts()


# In[6]:


df["target"].value_counts().plot(kind = "bar", color = ["salmon", "lightblue"]);


# In[7]:


df.info()


# In[8]:


df.describe()


# ## Heart disease frequency according to sex

# In[9]:


df.sex.value_counts()


# In[10]:


pd.crosstab(df.target, df.sex)


# In[11]:


pd.crosstab(df.target, df.sex).plot(kind = 'bar',
                                    color = ['salmon', 'lightblue'],
                                    figsize = (10,6))
plt.title('Frequency of Heart disease by age')
plt.xlabel('Number of people')
plt.ylabel('0 = No Disease, 1 = Disease')
plt.legend(['Female', 'Male'])
plt.xticks(rotation=0)
plt.show()


# ## Age vs (thalach)Max Heart disease rate

# In[12]:


# Creating figure
plt.figure(figsize=(10, 6))

# Scatter with positive example
plt.scatter(df.age[df.target==1],
           df.thalach[df.target==1],
           c = "salmon")

# Scatter with negative example
plt.scatter(df.age[df.target==0],
           df.thalach[df.target==0],
           c = 'lightblue')

plt.title('Age & thalach effects on heart disease')
plt.xlabel('Age')
plt.ylabel('Thalach')
plt.legend(['Heart Disease', 'No Disease'])
plt.show()


# In[13]:


# Checking the distribution of the age column
df.age.plot.hist(color = 'salmon');


# ## Heart Disease frequency per chest pain type(cp)

# CP - chest pain type<br>
# <b>Typical Angina</b>: Chest pain related decrease blood supply to the heart<br> 
# <b>Atypical Angina</b>: Chest pain not related to heart<br>
# <b>Non-angina pain</b>: Typical esophageal sperms (non heart related)<br>
# <b>Asymptomatic</b>: Chest pain not showing sign of disease<br>

# In[14]:


pd.crosstab(df.cp, df.target)


# In[15]:


# Make crosstab more visually to understand a bit better

pd.crosstab(df.cp, df.target).plot(kind = 'bar',
                                  figsize = (10, 6),
                                  color = ["salmon", "lightblue"]
                                  )

plt.title("Frequency per chest pain")
plt.xlabel("Chest pain Type")
plt.ylabel("Amount")
plt.xticks(rotation = 0)
plt.legend(['heart disease', 'No disease'])
plt.show()


# A <b>correlation matrix</b> is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. A correlation matrix is used to summarize data, as an input into a more advanced analysis, and as a diagnostic for advanced analyses.
# 
# 

# In[16]:


# Set a correlation matrix
df.corr()


# In[17]:


# Make corr matrix visually using seaborn's heatmap
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                annot=True,
                linewidth=0.5,
                fmt = ".2f",
                cmap='YlGnBu');


# ## Modelling

# In[18]:


df.head()


# In[19]:


# Split data into X and y
X = df.drop("target", axis=1)
y = df["target"]


# In[20]:


X.head()


# In[21]:


y[:10]


# In[22]:


# Splitting the data into train and test sets

np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2
                                                    )


# In[23]:


X_train.head()    


# In[24]:


y_train, len(y_train)


# Now we've got our data split into training and test, its a time to build machine learning model.<br>
# We will train it (Find patterns) on the training sets.<br>
# And we will test it (use the pattern) on the test data.

# ** We are going to try three machine learning models here (For Classification) **
# 1. Logistic Regression
# 2. K-Nearest Neighbours Classifier
# 3. Random Forest Classifier

# In[25]:


# Put a dictionary

models = {
            "Logistic Regression": LogisticRegression(),
            "KNN": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier()
}

# Create a function to fit and score models

def fit_and_score(models, X_train, X_test, y_train, y_test):
    '''
    Fits and evaluate ml models
    models: a dict of different scikit learn models
    X_train: training data(no labels)
    X_test: testing data(no labels)
    y_train: training labels
    y_test: test labels
    '''
    
    # Set random seed
    np.random.seed(42)
    
    # Making empty dict to keep model scores
    model_scores = {}
    
    # Looping through all 3 models
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores


# In[26]:


model_scores = fit_and_score(models = models,
                            X_train = X_train,
                            X_test = X_test,
                            y_train = y_train,
                            y_test = y_test)
model_scores


# In[27]:


# Compare the models score
model_compare = pd.DataFrame(model_scores, index = ["accuracy"])
model_compare.T.plot.bar();


# Now we have got the baseline model and we know a model's first predictions aren't always what we should based our
# next steps off.
# What should do?
# 
# Let's look at the following:
# * Hyperparameter tuning
# * Feature importance
# * Confusion matrix
# * Cross validation
# * Precision
# * Recall
# * F1 score
# * Classification report
# * ROC curve
# * Area under curve
# 
# ## Hyperparameter tuning

# In[28]:


# Lets start tuning KNN model's first
train_scores = []
test_scores = []

# Create a list of different values for n neighbor
neighbors = range(1, 21)

# Setup KNN instance
knn = KNeighborsClassifier()

# Loop through different n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors = i)
    
    # Fit algo
    knn.fit(X_train, y_train)
    
    # Update training score
    train_scores.append(knn.score(X_train, y_train))
    
    # Update test score
    test_scores.append(knn.score(X_test, y_test))    


# In[29]:


train_scores


# In[30]:


test_scores


# In[31]:


# Create results visually
plt.plot(neighbors, train_scores, label="Train scores")
plt.plot(neighbors, test_scores, label = "Test score")
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.xticks(range(1, 21))
plt.legend()
print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")
plt.show()


# ## Hyperparameter tuning with RandomizedSearchCV
# 
# We are going to tune:
# * Logistic Regression
# * RandomForestClassifier()
# 
# ... Using RandomizedSearchCV

# In[32]:


# Create a hyperparameter grid for logistic Regression

log_reg_grid = {
                "C": np.logspace(-4,4, 20),
                "solver": ["liblinear"]
               }

# Create a hyperparameter grid for RandomForestClassifier

rf_grid = {
                "n_estimators" : np.arange(10, 1000, 50),
                "max_depth": [None, 3, 5, 10],
                "min_samples_split" : np.arange(2, 20, 2),
                "min_samples_leaf" : np.arange(1, 20, 2)
            } 


#  Now we have got hyperparameter grids setup for each of our models, let's tune them using RandomizedSearchCV

# In[33]:


# Tune logistic regression
np.random.seed(42)

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                               param_distributions = log_reg_grid,
                               cv = 5,
                               n_iter = 20,
                               verbose = True)
rs_log_reg.fit(X_train, y_train)


# In[34]:


rs_log_reg.best_params_


# In[35]:


rs_log_reg.score(X_test, y_test)


# Now we have tuned LogisticRegression(), let's do the same for RandomForestClassifer()...

# In[36]:


# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search model for RandomForestClassifier()
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                          param_distributions = rf_grid,
                          cv = 5,
                          n_iter = 20,
                          verbose = True)

# Fit for RandomForestClassifier()
rs_rf.fit(X_train, y_train)


# In[37]:


# Find the best hyperparameter
rs_rf.best_params_


# In[38]:


rs_rf.score(X_test, y_test)


# 1. by hand
# 2. RandomizedSearchCV
# 3. GridSearchCV

# ## Hyperparameter Tuning with GridSearchCV
# 
# Since our LogisticRegression model provides the best scores so far, we will try and improve them again using GridSearchCV

# In[39]:


# Different hyperparameters for our LogisticRegerssion mode
log_reg_grid = {
                    "C": np.logspace(-4, 4, 30),
                    "solver": ["liblinear"]
                }

# Setup grid hyperparameter search for logisticRegression
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid = log_reg_grid,
                          cv = 5,
                          verbose = True)

# Fit grid hyperparameter search model
gs_log_reg.fit(X_train, y_train)


# In[40]:


gs_log_reg.best_params_


# In[41]:


gs_log_reg.score(X_test, y_test)


# In[42]:


model_scores


# In[43]:


# Not much improvement but it performs overall so well


# ## Evaluating our tuned machine learning classification, beyond accuracy
# 
# * ROC curve and AUC curve
# * confusion matrix
# * Classification report
# * Precision
# * Recall
# * F1 score
# 
# and it would be great to use cross-validation where possible.<br>
# To make comparison and evaluate our trained model, we need to first make predictions

# In[44]:


# Make predictions with our tuned model 

y_preds = gs_log_reg.predict(X_test)


# In[45]:


y_preds


# In[46]:


y_test


# In[47]:


# Plotting ROC and calculating AUC matrix
plot_roc_curve(gs_log_reg, X_test, y_test);


# <b>Confusion matrix</b> is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.

# In[48]:


# Confusion matrix

print(confusion_matrix(y_test, y_preds))


# In[49]:


sns.set(font_scale = 1.2)

def plot_conf_matrix(y_test, y_preds):
    '''
    Plots a nice looking confusion matrix using seaborn heatmap()
    '''
    fig, ax = plt.subplots(figsize = (3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                    annot = True, 
                    cbar = False)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    
    '''
    bottom, top = ax.getylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    '''
    
plot_conf_matrix(y_test, y_preds)


# Now we have got the ROC curve , AUC curve metric and a confusion matrix, let's get classification report as well as cross validated precision, recall and f1

# In[50]:


print(classification_report(y_test, y_preds))


# ## Calculate evaluation metrics using cross-validation 
# 
# We are going to calculate precision, recall and f1-score of our model using cross validation and to do so we will be using cross_val_score() .

# In[51]:


# Check the best hyperparameters
gs_log_reg.best_params_


# In[52]:


# Create a new classifier with best parameters
clf =LogisticRegression(C = 0.20433597178569418,
                       solver = "liblinear")


# #### Cross Validated Accuracy

# In[53]:


cv_acc = cross_val_score(clf,
                        X,
                        y,
                        cv = 5,
                        scoring = 'accuracy')


# In[54]:


cv_acc


# In[55]:


cv_acc = np.mean(cv_acc)
cv_acc


# #### Cross validated precision

# In[56]:


cv_precision = cross_val_score(clf,
                              X,
                              y,
                              cv = 5,
                              scoring = 'precision')
cv_precision = np.mean(cv_precision)
cv_precision


# #### Cross validated Recall

# In[57]:


cv_recall = cross_val_score(clf,
                              X,
                              y,
                              cv = 5,
                              scoring = 'recall')
cv_recall = np.mean(cv_recall)
cv_recall


# #### Cross validated f1-score

# In[58]:


cv_f1 = cross_val_score(clf,
                              X,
                              y,
                              cv = 5,
                              scoring = 'f1')
cv_f1 = np.mean(cv_f1)
cv_f1


# In[59]:


# Visualize cross validated metrics
cv_metrics = pd.DataFrame(
                            {
                                "Accuracy": cv_acc,
                                "Precision": cv_precision,
                                "Recall": cv_recall,
                                "F1": cv_f1
                            }
                                ,index = [0]
                            
                         )
cv_metrics.T.plot.bar(title = "Cross validated Classification metrics", 
                     legend = False);


# ### Feature Importance
# 
#     Feature importance is asking that, "which feature contributes the most to the outcomes of the model and how did they work?"
#     
#     Finding feature importance is different for each machine learning model. One way to find the importance is to search for "(Model name) feature importance"
#     
#     Let's find the feature importance for our LogisticRegressionModel

# In[60]:


# Fit the instance for Logistic Regression
gs_log_reg.best_params_


# In[61]:


clf = LogisticRegression(C = 0.20433597178569418,
                        solver = "liblinear")
clf.fit(X_train, y_train);


# In[62]:


# Check coef_
clf.coef_


# In[63]:


# Match coef's of features to columns
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict


# In[64]:


# Visualize the feature importance
feature_df = pd.DataFrame(feature_dict, index = [0])
feature_df.T.plot.bar(title="Feature importance", legend = False);


# In[65]:


pd.crosstab(df['slope'], df['target'])


# In[66]:


pd.crosstab(df['sex'], df['target'])


# slope - the slope of the peak excercise ST segment
# 
# * 0: Upsloping -) Better heart rate with excercise(Uncommon)
# * 1: Flatslopping -) minimal change (typical healthy rate)
# * 2: Downslopings -) signs of unhealthy rate

# ## Experimentation
# if you  have not hit the evaluation metric yet.. ask yourself..
#    * Could you collect more data?
#    * Could you try a better model? Like CatBoost or XGBoost
#    * Could you improve the current models?
#    * If your model is good enough(you have hit the evaluation metric) how would your export it and share with others?
