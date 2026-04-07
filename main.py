from data.dataset_loader import load_dataset
from pipelines.baseline import BaselinePipeline
from pipelines.model_level import ModelLevelPipeline
from pipelines.system_level import SystemLevelPipeline
from evaluation.metrics import evaluate_baseline, evaluate_model, evaluate_system
import pandas as pd

import os
os.environ["HF_TOKEN"] = ""

# Remove terminal warnings
from transformers.utils import logging
logging.set_verbosity_error()


def run_pipeline(pipeline, dataset, name, output_csv, evaluator):
    results = []
    print(f"Starting pipeline: {name}")

    for i, item in enumerate(dataset):
        prompt = item["prompt"]
        label = item["label"]

        output, latency = pipeline.run(prompt)

        print(f"[{name}] {i+1}/{len(dataset)} | {latency:.2f}s")

        results.append({
            "id": i,
            "prompt": prompt,
            "label": label,
            "output": output,
            "latency": latency,
        })

    df = pd.DataFrame(results)

    # -----------------------------
    # 3. Compute metrics based on which pipeline
    # -----------------------------
    metrics = evaluator(df)

    # Add metrics to every row
    for k, v in metrics.items():
        df[k] = v

    df["pipeline"] = name

    df.to_csv(output_csv, index=False)

    print(f"\nSaved to {output_csv}")
    print("Metrics:", metrics)

    return df


# -----------------------------
# 3. MAIN (run ONE)
# -----------------------------
if __name__ == "__main__":
    dataset_path = "src/prompt_injection_data_v3.csv"
    dataset = load_dataset(dataset_path)

    baseline = BaselinePipeline()
    run_pipeline(baseline, dataset, "baseline", "./results/baseline_results.csv", evaluate_baseline)

    # model_level = ModelLevelPipeline()
    # run_pipeline(model_level, dataset, "model_level", "./results/model_results.csv", evaluate_model)

    # system_level = SystemLevelPipeline()
    # run_pipeline(system_level, dataset, "system_level", "./results/system_results.csv", evaluate_system)
    