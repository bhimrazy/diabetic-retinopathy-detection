import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data/train"
CSV_PATH = "data/trainLabels.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Load the CSV file into a pandas DataFrame and add the image path
df = pd.read_csv(CSV_PATH, names=['image_path', 'label'], converters={'image_path': lambda x: f"{DATA_DIR}/{x}.jpeg"})

#  drop row where image does not exist
df = df[df['image_path'].apply(lambda x: os.path.exists(x))]

# split the data into train and validation sets such that the class distribution is the same in both sets
df_train, df_val = train_test_split(df, test_size=TEST_SIZE, stratify=df['label'], random_state=RANDOM_STATE)

# Save the train and validation sets to CSV files
df_train.to_csv("data/train.csv", index=False)
df_val.to_csv("data/val.csv", index=False)
