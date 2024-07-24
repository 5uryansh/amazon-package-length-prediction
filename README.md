# amazon-package-length-prediction
This repository contains code for predicting product lengths based on their textual descriptions using natural language processing (NLP) and machine learning techniques.

### Step 1
Download the dataset named 'train.csv' from https://zenodo.org/records/12802196 and save it inside dataset folder.

### Step 2
```
pip install -r requirements.txt
```

### Step 3
```
cd dataset
```

### Step 4
This step includes:
* Load the dataset and remove any missing values.
* Convert all textual descriptions to lowercase for uniformity.
* Split the data into training and testing sets.
* Remove outliers to improve model performance.
* TF-IDF Vectorization:
* Convert product descriptions into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).
```
python dataset.py
```
Above code will generate a file named 'train_final.csv' in dataset folder.

### Step 5
```
cd ../src
```

### Step 6
```
python main.py
```
You can also pass the values for `learning_rate`, `epochs`, `batch_size`, and `max_grad_norm` in above code for example:
```
python main.py --learning_rate 0.0001 --epochs 40 --batch_size 512
```
