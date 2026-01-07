import pandas as pd

# Load dataset (nama file SESUAI folder)
df = pd.read_csv("Diabetes_Final_Data_V2.csv")

print("===== 5 DATA TERATAS =====")
print(df.head())

print("\n===== NAMA KOLOM =====")
print(df.columns)

print("\n===== INFO DATA =====")
print(df.info())
