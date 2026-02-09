import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn import preprocessing
from warnings import simplefilter
from imblearn.under_sampling import RandomUnderSampler
import glob

simplefilter(action='ignore',category=FutureWarning)

start_time=time.time()

main_dataset=pd.read_csv("c_data2.csv")
main_dataset.columns = main_dataset.columns.str.strip()
print(main_dataset.columns)

attack_types=["Bot","DDOS","DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris",
              "FTP-Patator","Heartbleed", "Infiltration", "PortScan", "SSH-Patator", 
              "Web Attack - Brute Force","Web Attack - Sql Injection", "Web Attack - XSS"]

benign_type="BENIGN"


b_data = main_dataset[main_dataset["Label"] == benign_type]
for it in attack_types:
    
    a_data=main_dataset[main_dataset["Label"] == it]
    
    c_data=pd.concat([a_data,b_data],axis=0)
    
    c_data=c_data.sample(frac=1,random_state=42)
    
    op_file=f"{it}_vs_BENIGN.csv"
    c_data.to_csv(op_file,index=False)
    print(f"Saved{op_file}")
    
end_time=time.time()
exec_time=end_time-start_time
    
print(f"Execution time: {exec_time:.2f}seconds")
    
b_data = main_dataset[main_dataset["Label"] == benign_type]

fnames=['Bot_vs_BENIGN.csv', 'DDoS_vs_BENIGN.csv', 'DoS GoldenEye_vs_BENIGN.csv',
    'DoS Hulk_vs_BENIGN.csv', 'DoS Slowhttptest_vs_BENIGN.csv',
    'DoS slowloris_vs_BENIGN.csv', 'FTP-Patator_vs_BENIGN.csv',
    'Heartbleed_vs_BENIGN.csv', 'Infiltration_vs_BENIGN.csv',
    'PortScan_vs_BENIGN.csv', 'SSH-Patator_vs_BENIGN.csv',
    'Web Attack - Brute Force_vs_BENIGN.csv',
    'Web Attack - Sql Injection_vs_BENIGN.csv', 'Web Attack - XSS_vs_BENIGN.csv']

for fname in fnames:
    data=pd.read_csv(fname)
    
    num_benign=(data['Label'] == 'BENIGN').sum()
    num_attack=(data['Label']!='BENIGN').sum()
    
    print(f"File: {fname}")
    print(f"Number of Benign instances: {num_benign}")
    print(f"Number of Attack instances: {num_attack}")
    print("Shape of the dataset: ",data.shape)
    print("-----------------------------")
    
