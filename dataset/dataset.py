import pandas as pd
from tqdm import tqdm

train_data = pd.read_csv('train.csv')

# removing unnecessary columns
train = train_data.drop(labels='PRODUCT_ID', axis=1)

print("\nCombining TITLE, DESCRIPTION, and BULLET_POINTS")
print("This process can take upto ten minutes...")
# adding an empty column for final string
train['FINAL_STRING'] = ''

# combining dataset
for i in tqdm(range(0, len(train)), desc="Processing", unit="items"):
  text_features = [str(train.iloc[i,0]), str(train.iloc[i,1]), str(train.iloc[i,2])]
  combined_text = " ".join([text for text in text_features if text is not None and text != "nan"])
  train.iloc[i,5] = combined_text
#   if((i+1)%1000==0):
#       print(f"Currently on line {i+1}/{len(train)}")
  
# dropping the columns
train = train.drop(labels='TITLE', axis=1)
train = train.drop(labels='DESCRIPTION', axis=1)
train = train.drop(labels='BULLET_POINTS', axis=1)
train = train.reindex(columns=['PRODUCT_TYPE_ID', 'FINAL_STRING', 'PRODUCT_LENGTH'])

# checking for the null values
print(f"\n{train.isnull().any()}")
print(f"Final dataset looks like.... \n{train.head()}")

# saving the dataset
train.to_csv('train_final.csv')