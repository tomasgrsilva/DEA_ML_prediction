from common.feature_engineering.df_eng import build_model_dataframe
from eda_rep import run_eda

df = build_model_dataframe(task="r", use_halogen_positions=True)  # or task="c"
df = df.drop(columns=["result_id", "molecule_id"])
run_eda(df, target_col="peak", task_type="regression")  # or "energy_range", "classification"
#print(df.to_string())