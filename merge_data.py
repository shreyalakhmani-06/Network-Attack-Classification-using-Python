import pandas as pd
import glob
import os

data_path = "dataset/*.csv"   # folder where CSVs are kept

csv_files = glob.glob(data_path)
print("CSV files found:", csv_files)

df_list = []

for file in csv_files:
    df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.strip()  #  VERY IMPORTANT
    df_list.append(df)

final_df = pd.concat(df_list, ignore_index=True)

print("Merged dataset shape:", final_df.shape)
print("Label distribution:\n", final_df["Label"].value_counts())

final_df.to_csv("c_data.csv", index=False)
print(" c_data.csv saved successfully")
