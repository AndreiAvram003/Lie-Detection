import pandas as pd
import os

# Căi către fișiere
csv1 = r"C:\Users\Andrei\Desktop\Faculta\Lie Detection\ai_model\preprocessed_data.csv"
csv2 = r"C:\Users\Andrei\Desktop\Faculta\Lie Detection\ai_model\preprocessed_data_from_png.csv"
output_csv = r"C:\Users\Andrei\Desktop\Faculta\Lie Detection\ai_model\final_data.csv"

# Încarcă ambele CSV-uri
df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)

# Păstrează doar coloanele necesare
df1_filtered = df1[['id', 'class', 'label_numeric']]
df2_filtered = df2[['id', 'class', 'label_numeric']]

# Elimină ".png" din id pentru df2
df2_filtered['id'] = df2_filtered['id'].str.replace('.png', '', regex=False)

# Concatenează cele două seturi
final_df = pd.concat([df1_filtered, df2_filtered], ignore_index=True)

# Salvează CSV final
final_df.to_csv(output_csv, index=False)
print(f"✅ CSV final salvat în: {output_csv}")
