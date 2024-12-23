from KPrincipalGaussiansClassifier import KPrincipalGaussiansClassifier, MetaParams
import pandas as pd

variance_perc_threshold=0.95
model_name = f"mahalanobis_class-specific_{ variance_perc_threshold * 100 }"
print("model name: ", model_name)

def load_data(file_path): 
    print("loading data")
    all_data = pd.read_csv(file_path)
    train_data = all_data.to_numpy()[:, 1:]
    labels = all_data.to_numpy()[:, 0]
    print(f"train shape: {train_data.shape}")
    print(f"labels shape: {labels.shape}")
    
    return train_data, labels

class_components_weights = [1.2,1,1,1,1,1,1,1.2,1.1,1.15]
train_data_file_path = 'C:/src/data_analysis/data/digit-recognizer/train.csv'
train_data, labels = load_data(train_data_file_path)
meta_params = MetaParams(num_classes=10,class_components_weights=class_components_weights, n_init=1,tol=1e-4,variance_perc_threshold=0.95,k_near_centroids=5) 

k_prin_gauss_model = KPrincipalGaussiansClassifier(meta_params=meta_params)
k_prin_gauss_model.fit(train_data=train_data, labels=labels)
