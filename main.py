from data.dataset_loader import load_dataset
from pipelines.baseline import BaselinePipeline
from pipelines.model_level import ModelLevelPipeline
from pipelines.system_level import SystemLevelPipeline
from evaluation.metrics import evaluate
import pandas as pd

import os
# os.environ["HF_TOKEN"] = ""


def run_pipeline(pipeline, dataset, name):
    results = []

    for item in dataset:
        output, latency = pipeline.run(item['prompt'])

        results.append({
            "prompt": item['prompt'],
            "label": item['label'],
            "output": output,
            "latency": latency
        })
    
    return results
    # TODO: uncomment this when successfully run

    # metrics = evaluate(results)
    # metrics['pipeline'] = name

    # return metrics


if __name__ == "__main__":
    dataset_path = "src/dataset_safe_250.csv"
    dataset = load_dataset(dataset_path)

    baseline = BaselinePipeline()
    # model_level = ModelLevelPipeline()
    # system_level = SystemLevelPipeline()

    results = []

    results.append(run_pipeline(baseline, dataset, "baseline"))
    # results.append(run_pipeline(model_level, dataset, "model_level"))
    # results.append(run_pipeline(system_level, dataset, "system_level"))

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
