import json

with open("research/kaggle_notebooks/flexonafft_biohub-lineage-forge-precision-tracking/biohub-lineage-forge-precision-tracking.ipynb") as f:
    nb = json.load(f)

updated_count = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "BIOHUB_DET_THRESHOLD" in source and "0.965" in source:
            source = source.replace('os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"', 'os.environ["BIOHUB_DET_THRESHOLD"] = "0.96"')
            source = source.replace('"BIOHUB_DET_THRESHOLD": 0.965', '"BIOHUB_DET_THRESHOLD": 0.96')
            cell["source"] = [source]
            updated_count += 1
            print(f"Successfully updated cell {i}!")

if updated_count == 0:
    raise RuntimeError("Target string not found or cells not updated!")

with open("scripts/kaggle_kernels/biohub_0_946_edge_feature_tta_tuned/kernel.ipynb", "w") as f:
    json.dump(nb, f)
print(f"Saved modified notebook successfully with {updated_count} cells updated.")
