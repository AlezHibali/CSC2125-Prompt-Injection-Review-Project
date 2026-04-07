import pandas as pd

df = pd.read_csv("./prompt_injection_data_v3.csv")

df["label"] = df["label"].astype(str).map({
    "0": "safe",
    "1": "unsafe"
})

df.to_csv("updated_file.csv", index=False)