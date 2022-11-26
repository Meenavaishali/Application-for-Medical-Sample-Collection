import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

specimen_data=pd.read_excel("D:/PROJECT 69/New model building/final_data1.xlsx")

specimen_data.dtypes

# specimen_data=specimen_data.rename(columns={"Cut-off Schedule":"Cut_off_Schedule","Cut-off time_HH_MM":"Cut_off_time_HH_MM"})

specimen_data.head()
specimen_data.columns

#Droping unnecessary columns
specimen_data=specimen_data.drop(["Patient_ID","Test_Booking_Date","Latitudes_and_Longitudes_Patient","Latitudes_and_Longitudes_Agent","Latitudes_and_Longitudes_Diagnostic_Center","Sample_Collection_Date","Test_Booking_Time_HH_MM","patient_location","Agent_Arrival_Time_range_HH_MM"],axis=1)



# #splitting latitude and longitude of Patient location
# specimen_data1=specimen_data["Latitudes_and_Longitudes_Patient"].str.split(',', expand=True)
# #renaming the column names
# specimen_data1.columns="Latitude_patient","Longitude_patient"
# #concating with patient latitude and longitude
# specimen_data=pd.concat([specimen_data,specimen_data1],axis=1)



# #splitting latitude and longitude of Agent location
# specimen_data2=specimen_data["Latitudes_and_Longitudes_Agent"].str.split(',', expand=True)
# #renaming the column names
# specimen_data2.columns="Latitude_Agent","Longitude_Agent"
# #concating with Agent latitude and longitude
# specimen_data=pd.concat([specimen_data,specimen_data2],axis=1)



# #splitting latitude and longitude of Diagnostic Center location
# specimen_data3=specimen_data["Latitudes_and_Longitudes_Diagnostic_Center"].str.split(',', expand=True)
# #renaming the column names
# specimen_data3.columns="Latitude_Diagnostic_Center","Longitude_Diagnostic_Center"
# #concating with Diagnostic Center latitude and longitude
# specimen_data=pd.concat([specimen_data,specimen_data3],axis=1)



specimen_data.pincode=specimen_data.pincode.astype("int64")
# specimen_data.Latitude_patient=specimen_data.Latitude_patient.astype("float64")
# specimen_data.Longitude_patient=specimen_data.Longitude_patient.astype("float64")
# specimen_data.Latitude_Agent=specimen_data.Latitude_Agent.astype("float64")
# specimen_data.Longitude_Agent=specimen_data.Longitude_Agent.astype("float64")
# specimen_data.Latitude_Diagnostic_Center=specimen_data.Latitude_Diagnostic_Center.astype("float64")
# specimen_data.Longitude_Diagnostic_Center=specimen_data.Longitude_Diagnostic_Center.astype("float64")

#EDA
# Measures of Central Tendency / First moment business decision
specimen_data.mean()
specimen_data.median()
specimen_data.mode()

# Measures of Dispersion / Second moment business decision
specimen_data.std()
specimen_data.var()

# Third moment business decision
specimen_data.skew()

# Fourth moment business decision
specimen_data.kurt()

from sklearn.preprocessing import LabelEncoder


le=LabelEncoder()


specimen_data["Diagnostic_Centers"]=le.fit_transform(specimen_data["Diagnostic_Centers"])
specimen_data["Time_slot"]=le.fit_transform(specimen_data["Time_slot"])
specimen_data["Availabilty_time_Patient"]=le.fit_transform(specimen_data["Availabilty_time_Patient"])
specimen_data["Gender"]=le.fit_transform(specimen_data["Gender"])
specimen_data["Test_name"]=le.fit_transform(specimen_data["Test_name"])
specimen_data["Sample"]=le.fit_transform(specimen_data["Sample"])
specimen_data["Way_Of_Storage_Of_Sample"]=le.fit_transform(specimen_data["Way_Of_Storage_Of_Sample"])
# specimen_data["Agent_Arrival_Time_range_HH_MM"]=le.fit_transform(specimen_data["Agent_Arrival_Time_range_HH_MM"])


#determining correlation between different variables
sns.heatmap(specimen_data.corr(),cmap='coolwarm',cbar=True)

#Input and Output Variables
x=specimen_data.loc[:,specimen_data.columns!="Exact_Arrival_Time_MM"]
y=specimen_data["Exact_Arrival_Time_MM"]



#splitting data
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(x,y,test_size = 0.25,random_state=0)




#fitting the model 
from sklearn.linear_model import LinearRegression

reg_model = LinearRegression()  #creating object of the class LinearRegression
reg_model.fit(X_train,y_train)

test_prediction=reg_model.predict(X_test)



#saving the model
pickle.dump(reg_model,open("regression_model.pkl","wb"))


#load the model from disk
model=pickle.load(open('regression_model.pkl','rb'))


from sklearn.metrics import mean_squared_error,r2_score

mean_squared_error(y_test,test_prediction)

r2_score(y_test,test_prediction)

mean_squared_error(y_test,test_prediction)**(1/2)

specimen_data.shape

#checking for the results
list_value=np.array((specimen_data.loc[0,specimen_data.columns!="Exact_Arrival_Time_MM"])).reshape(1,-1)
list_value

print(reg_model.predict(list_value))


