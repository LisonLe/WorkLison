#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd

df1 = pd.read_csv('C:\\Users\\lison\Documents\\Mini projet DIA1\\gender_submission.csv')
df2 = pd.read_csv('C:\\Users\\lison\\Documents\\Mini projet DIA1\\test.csv')

colonne_id = 'PassengerId'

df_combiné = pd.merge(df1, df2, on=colonne_id, how='inner') 

chemin_fichier_combiné = 'C:\\Users\\lison\\Documents\\Mini projet DIA1\\Combinaison1.csv'
df_combiné.to_csv(chemin_fichier_combiné, index=False)  
df_combiné.head()


# In[13]:


final=pd.read_csv('C:\\Users\\lison\\Documents\\Mini projet DIA1\\last.csv',sep=';')
final.head()


# In[14]:


final['Sex']=final['Sex'].map({'male':0, 'female':1})
final.head()


# In[15]:


df_one_hot = pd.get_dummies(final['Embarked'])

df_one_hot.columns = ['S', 'C', 'Q']

df_final = pd.concat([final, df_one_hot], axis=1)

df_final.head()


# In[16]:


df_final.drop(['Embarked','PassengerId'],axis=1)


# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

features = ['Age', 'Sex', 'Pclass','SibSp','Parch','S','C','Q']
target = 'Survived'  

df_final['Age'].fillna(df_final['Age'].mean(), inplace=True)

X = df_final[features]
y = df_final[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

logreg = LogisticRegression(max_iter=1000)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the logistic regression model: {accuracy}")

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]} 
grid_search = GridSearchCV(logreg, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_val_pred = best_model.predict(X_val)
accuracy_val = accuracy_score(y_val, y_val_pred)
print(f"Accuracy on the validation set: {accuracy_val}")

y_test_pred = best_model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
print(f"Accuracy on the test set: {accuracy_test}")
print(best_model)


# In[18]:


from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

confusion = confusion_matrix(y_test, logreg.predict(X_test))
disp = plot_confusion_matrix(logreg, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
disp.ax_.set_title('Confusion Matrix')

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

plt.figure(figsize=(8, 6))
feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
feature_importance.nlargest().plot(kind='barh')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.show()


# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def predict_survival():
    print("Veuillez entrer les caractéristiques du passager :")
    
    pclass = int(input("Classe du passager (1, 2, 3) : "))
    sex = int(input("Sexe (0 pour homme, 1 pour femme) : "))
    age = int(input("Age : "))
    sibsp = int(input("Nombre de frères et sœurs : "))
    parch = int(input("Nombre de parents/enfants à bord : "))
    embarked_s = int(input("Embarqué à Southampton (1 pour oui, 0 pour non) : "))
    embarked_c = int(input("Embarqué à Cherbourg (1 pour oui, 0 pour non)"))
    embarked_q = int(input("Embarqué à Queenstown (1 pour oui, 0 pour non) : "))

    input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, embarked_s, embarked_c, embarked_q]],
                              columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked_S', 'Embarked_C', 'Embarked_Q'])
    
    prediction = best_model.predict(input_data)

    print("\nPrédiction de survie : " + ("Survécu" if prediction[0] == 1 else "N'a pas survécu"))

predict_survival()


# In[ ]:




