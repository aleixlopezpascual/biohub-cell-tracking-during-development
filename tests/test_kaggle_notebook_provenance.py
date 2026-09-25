import json
import re
from pathlib import Path

NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "kaggle_kernels"
    / "biohub_0_948_momentum_deepcenter_tta"
    / "biohub-0-948-momentum-deepcenter-tta.ipynb"
)

def get_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

def test_cell_0_unscored():
    notebook = get_notebook()
    cell_0 = "".join(notebook["cells"][0]["source"]).lower()
    assert "0.939" not in cell_0, "Cell 0 should not claim 0.939 score"
    assert "unscored" in cell_0 or "no verified score" in cell_0, (
        "Cell 0 must label candidate as unscored/no verified score"
    )

def test_cell_1_no_stale_claims():
    notebook = get_notebook()
    cell_1 = "".join(notebook["cells"][1]["source"])
    assert "0.913" not in cell_1, "Cell 1 should have no stale claims of a 0.913 baseline"
    assert "0.200" not in cell_1, "Cell 1 should have no stale claims of a 0.200 reverse-time weight"
    assert "EMA motion relinking" in cell_1, (
        "Cell 1 must accurately describe EMA motion relinking as the candidate delta"
    )

def test_cell_6_experiment_metadata():
    notebook = get_notebook()
    cell_6 = "".join(notebook["cells"][6]["source"])
    cell_6_lower = cell_6.lower()
    assert "biohub-0-948-momentum-deepcenter-tta" in cell_6, (
        "Cell 6 must identify experiment biohub-0-948-momentum-deepcenter-tta"
    )
    assert "biohub-0-946-edge-feature-tta-tuned" in cell_6, (
        "Cell 6 must identify parent biohub-0-946-edge-feature-tta-tuned"
    )
    assert "unscored" in cell_6_lower or "no verified score" in cell_6_lower, (
        "Cell 6 must record no verified Kaggle score"
    )
    assert "BIOHUB_" in cell_6, "Cell 6 must capture actual BIOHUB_* runtime config"
    assert "_EXPECTED_NUMERIC = {" not in cell_6, (
        "Cell 6 must not use copied hard-coded configuration"
    )

def test_quality_gate_requires_receipt():
    notebook = get_notebook()
    sources = ["".join(c["source"]) for c in notebook["cells"]]
    combined_source = "\n".join(sources)
    assert re.search(
        r"required_receipt.*?bidirectional_blend_union13_receipt\.json",
        combined_source,
    ), \
        "Quality gate must require bidirectional_blend_union13_receipt.json"
    assert re.search(r"required_condition.*?promote=true", combined_source), \
        "Quality gate must require condition promote=true"
    assert re.search(
        r"execute_push_submit.*?FORBIDDEN_UNTIL_REQUIRED_CONDITION",
        combined_source,
    ), \
        "execute_push_submit must be FORBIDDEN_UNTIL_REQUIRED_CONDITION"
    assert re.search(r"validated_receipt_sha256.*?None", combined_source), \
        "validated_receipt_sha256 must be None"
