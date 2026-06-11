import pandas as pd

df1 = pd.read_csv("model_comparison_results_merged.csv")
df2 = pd.read_csv("model_comparison_results_2.csv")

merged = pd.concat([df1, df2], ignore_index=True)

output = "model_comparison_results_merged.csv"
merged.to_csv(output, index=False)

print(f"File 1: {len(df1)} rows")
print(f"File 2: {len(df2)} rows")
print(f"Merged: {len(merged)} rows -> {output}")
print(f"Models: {merged['model'].unique().tolist()}")
