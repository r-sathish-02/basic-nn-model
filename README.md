# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Sathish R
### Register Number: 212222230138
```python

from google.colab import auth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import gspread
import pandas as pd
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds,_=default()
gc=gspread.authorize(creds)
worksheet = gc.open('gh').sheet1
data = worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'x':'int'})
dataset1 = dataset1.astype({'y':'int'})
dataset1.head()
dataset1.describe()
dataset1.info()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler=MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
model=Sequential([Dense(units=4,activation='relu',input_shape=[1]),
                  Dense(units=6,activation='relu'),
                  Dense(units=4,activation='relu'),
                  Dense(units=1)])
model.compile(optimizer='rmsprop',loss='mse')
model.fit(X_train1,y_train,epochs=2000)
loss=pd.DataFrame(model.history.history)
loss.plot()
xtestscaled=Scaler.transform(X_test)
model.evaluate(xtestscaled,y_test)
p=[[5]]
pscale= Scaler.transform(p)
model.predict(pscale)


```
## Output:

### Dataset Information:
##### df.head()

![Screenshot 2024-09-06 021415](https://github.com/user-attachments/assets/366c977e-7654-4f26-9b2d-3eb849cc2f51)


##### df.info()

![Screenshot 2024-09-06 021457](https://github.com/user-attachments/assets/086a8d69-755b-4c25-86f3-9dbbfa9e87bd)


##### df.describe()
![image](https://github.com/user-attachments/assets/942c57a2-062c-4d68-a8ed-1db0bc5bb81e)





##### Training Loss Vs Iteration Plot:
![image](https://github.com/user-attachments/assets/e647fc3a-8b67-42fb-b69f-05747bd82bda)



##### Test Data Root Mean Squared Error:
[![image](https://github.com/user-attachments/assets/875f5cf9-5440-4a6a-8183-dfb1d994fb6d)
](https://private-user-images.githubusercontent.com/118343978/364413442-875f5cf9-5440-4a6a-8183-dfb1d994fb6d.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjU1NjkzNDMsIm5iZiI6MTcyNTU2OTA0MywicGF0aCI6Ii8xMTgzNDM5NzgvMzY0NDEzNDQyLTg3NWY1Y2Y5LTU0NDAtNGE2YS04MTgzLWRmYjFkOTk0ZmI2ZC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwOTA1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDkwNVQyMDQ0MDNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0wZjRkYTNjOGFiNDYxZDhjYjI4MTAzODNlMGQ3Yjg5MGNmNjE4NWQ3ZWJiODBjZjdhOTk1NzRiOGQyNWU2ZmY1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.VI218y5m5wCQVXx3YfCXjhCEs8GxwCpyUwOkPNYwnN0)



##### New Sample Data Prediction:
![image](https://github.com/user-attachments/assets/aa66c377-341a-4044-98c6-997dfb5358be)




### Result:
Thus a neural network regression model for the given dataset is developed and the prediction for the given input is obtained accurately.
