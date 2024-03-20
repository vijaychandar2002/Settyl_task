import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Load the data
url = "https://gist.githubusercontent.com/farhaan-settyl/ecf9c1e7ab7374f18e4400b7a3d2a161/raw/f94652f217eeca83e36dab9d08727caf79ebdecf/dataset.json"
response = requests.get(url)
data = response.json()
df = pd.json_normalize(data)

# Check for missing values
print(df.isnull().sum())

# Explore the distribution of categories
print(df['internalStatus'].value_counts())

# Text preprocessing on 'externalStatus'
df['externalStatus'] = df['externalStatus'].str.lower().str.replace('[^\w\s]', '')

# One-hot encoding of 'externalStatus'
vec = CountVectorizer()
externalStatus_encoded = vec.fit_transform(df['externalStatus'])
externalStatus_encoded_df = pd.DataFrame(externalStatus_encoded.toarray(), columns=vec.get_feature_names_out())

# Label encoding of 'internalStatus'
df['internalStatus'] = df['internalStatus'].astype('category')
df['internalStatus_cat'] = df['internalStatus'].cat.codes

# Combine the encoded features with the rest of the dataframe
df = pd.concat([df, externalStatus_encoded_df], axis=1)

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(df.drop('internalStatus_cat', axis=1), df['internalStatus_cat'], test_size=0.2, random_state=42)

# Print the first few rows of the DataFrame to see the changes
print(df.head())


# Output:
# externalStatus    0
# internalStatus    0
# dtype: int64
# Loaded on Vessel            331
# Departure                   287
# Gate Out                    146
# Gate In                     143
# Arrival                      62
# Empty Return                 47
# Empty Container Released     47
# Unloaded on Vessel           37
# On Rail                      25
# Off Rail                     25
# Outbound Terminal            24
# Port Out                     15
# Port In                      14
# In-transit                   10
# Inbound Terminal              9
# Name: internalStatus, dtype: int64
#                                       externalStatus    internalStatus  \
# 0                                           port out          Port Out   
# 1                                        terminal in  Inbound Terminal   
# 2                                            port in           Port In   
# 3  vessel departure from first pol vessel name  t...         Departure   
# 4  vessel arrival at final pod vessel name  tian ...           Arrival   

#    internalStatus_cat  001s  008e  027e  115n  175w  224n  226e  ...  ts  \
# 0                  13     0     0     0     0     0     0     0  ...   0   
# 1                   7     0     0     0     0     0     0     0  ...   0   
# 2                  12     0     0     0     0     0     0     0  ...   0   
# 3                   1     0     0     0     0     0     0     0  ...   0   
# 4                   0     0     0     0     0     0     0     0  ...   0   

#    tucapel  unloaded  unloading  update  usjax  vayenga  vessel  volga  ym  
# 0        0         0          0       0      0        0       0      0   0  
# 1        0         0          0       0      0        0       0      0   0  
# 2        0         0          0       0      0        0       0      0   0  
# 3        0         0          0       0      0        0       2      0   0  
# 4        0         0          0       0      0        0       2      0   0  

# [5 rows x 149 columns]
# <ipython-input-4-444f61e1767b>:19: FutureWarning: The default value of regex will change from True to False in a future version.
#   df['externalStatus'] = df['externalStatus'].str.lower().str.replace('[^\w\s]', '')
