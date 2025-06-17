import pandas as pd

# Check LUNG dataset
df_lung = pd.read_csv("lung_train.csv")
print("🫁 Lung Dataset:")
print(df_lung.iloc[:, -1].value_counts())

# Check HEART dataset
df_heart = pd.read_csv("heart_train.csv")
print("\n❤️ Heart Dataset:")
print(df_heart.iloc[:, -1].value_counts())

# Check KIDNEY dataset
df_kidney = pd.read_csv("kidney_train.csv")
print("\n🩺 Kidney Dataset:")
print(df_kidney.iloc[:, -1].value_counts())
