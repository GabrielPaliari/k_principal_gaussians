from KPrincipalGaussiansClassifier import KPrincipalGaussiansClassifier, MetaParams
import pandas as pd
import numpy as np

def load_data(file_path): 
    print("loading data")
    all_data = pd.read_csv(file_path)
    train_data = all_data.to_numpy()[:, 1:]
    labels = all_data.to_numpy()[:, 0]
    print(f"train shape: {train_data.shape}")
    print(f"labels shape: {labels.shape}")
    
    return train_data, labels

train_data_file_path = 'C:/src/data_analysis/data/digit-recognizer/train.csv'
train_data, labels = load_data(train_data_file_path)
meta_params = MetaParams(num_classes=10, n_init=1, tol=1e-3)

k_prin_gauss_model = KPrincipalGaussiansClassifier(meta_params=meta_params)
k_prin_gauss_model.fit(train_data=train_data, labels=labels)
num_samples = 1
validation_data = train_data[:num_samples,:]
labels_v = labels[:num_samples]

predictions = k_prin_gauss_model.predict(validation_data)

import time
start_time = time.time()
print(len(predictions[labels_v == predictions]) / 1)
end_time = time.time()

print(f"Time taken to run the function: {end_time - start_time}")