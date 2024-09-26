## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
Developed by : Goutham.K
Reg No : 212223110019
```

```python
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/7b8596ff-44c6-47fd-869f-c87884a1bbee)
```python
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/user-attachments/assets/2c499e18-4ddb-48d3-83be-d8ec30232b82)
```python
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/user-attachments/assets/50da8dcc-5f90-46ed-8374-997cecc491e2)
```python
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![image](https://github.com/user-attachments/assets/37a26cce-6e0e-4924-be87-08212b91893b)


```python
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))

```

![image](https://github.com/user-attachments/assets/68518e7b-c627-453c-a9db-6516bf9e2e99)

```python
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/user-attachments/assets/279db00e-0e67-4928-8147-f488f53e5c3c)
```python
pd.get_dummies(df2,columns=["nom_0"])
```

![image](https://github.com/user-attachments/assets/1ff2d9bc-8d57-4609-9480-66b44eab81b7)
```python
pip install --upgrade category_encoders
```

![image](https://github.com/user-attachments/assets/88b9adf9-69b2-4cae-98f4-15cf4eb2b20c)

```python
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/user-attachments/assets/6be026a8-0c5c-4b3c-a28d-0c2c5e6bb130)

```python
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```

![image](https://github.com/user-attachments/assets/e7aa6cc5-b07f-44d2-a188-02c9dd0dd93f)
```python
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/a6c2ed31-8c12-4a72-8fb8-86dbe00dbba0)
```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/a6dec60b-b008-4c73-8778-e67d8f101b5f)
```python
df.skew()
```
![image](https://github.com/user-attachments/assets/b70e6dd1-b61f-4387-9c91-e22635b32f89)
```python
df["Highly Positive Skew"]=np.log(df["Highly Positive Skew"])
df["Moderate Positive Skew"]=np.reciprocal(df["Moderate Positive Skew"] )
df["Highly Positive Skew"]=np.sqrt(df["Highly Positive Skew"])
df["Highly Positive Skew"]=np.square(df["Highly Positive Skew"])
```
```python
df.skew()
```
![image](https://github.com/user-attachments/assets/a3e45da3-8650-425b-b816-d9b329aeaefb)
```python
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df ["Moderate Negative Skew_ yeojohnson"], parameters=stats.yeojohnson(df ["Moderate Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/4b02e04b-1d4a-49a4-9dae-f9ab5cc5c635)
```python
df.skew()
```
![image](https://github.com/user-attachments/assets/72043f33-344d-4265-9a35-8f7905ed83aa)
```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/1bbffcee-48d5-4e0c-abac-355a2d1acc79)

```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/f05787f2-575f-4928-941b-77a41d309c3f)
```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/36a1b2ce-429b-44de-a643-7afab7855921)
```python
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/52ad574d-9bcf-4195-8be7-a2d066e6e08a)
```python
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/30d7c416-cc40-495c-b87c-6c12149dc245)
```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/c0125f74-161a-4b30-aeab-8e6d3d562b4d)

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
