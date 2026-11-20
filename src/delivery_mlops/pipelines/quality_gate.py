
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]

champion_path = (
    ROOT / "artifacts/quality_monitoring/champion_evaluation.json"
)
candidate_path = (
    ROOT / "artifacts/quality_monitoring/candidate_evaluation.json"
)

output_path = (
    ROOT / "reports/quality_gate/promotion_decision.json"
)

with open(champion_path, "r") as f:
    champion = json.load(f)

with open(candidate_path, "r") as f:
    candidate = json.load(f)

# Both models must be evaluated on the same holdout size.
same_records = (
    champion["evaluation_records"]
    == candidate["evaluation_records"]
)

# Project-defined quality criteria.
mae_improvement = (
    (champion["MAE"] - candidate["MAE"])
    / champion["MAE"]
) * 100

rmse_degradation = (
    (candidate["RMSE"] - champion["RMSE"])
    / champion["RMSE"]
) * 100

r2_difference = candidate["R2"] - champion["R2"]

checks = {
    "same_evaluation_records": same_records,
    "mae_improved_by_at_least_1_percent": mae_improvement >= 1.0,
    "rmse_not_worse_by_more_than_2_percent": rmse_degradation <= 2.0,
    "r2_not_lower_by_more_than_0.01": r2_difference >= -0.01,
}

eligible = all(checks.values())

decision = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "champion_model": champion["model_name"],
    "candidate_model": candidate["model_name"],
    "evaluation_records": candidate["evaluation_records"],
    "champion_metrics": {
        "MAE": champion["MAE"],
        "RMSE": champion["RMSE"],
        "R2": champion["R2"],
    },
    "candidate_metrics": {
        "MAE": candidate["MAE"],
        "RMSE": candidate["RMSE"],
        "R2": candidate["R2"],
    },
    "mae_improvement_percent": mae_improvement,
    "rmse_change_percent": rmse_degradation,
    "r2_difference": r2_difference,
    "quality_checks": checks,
    "promotion_eligible": eligible,
    "decision": (
        "CANDIDATE_ELIGIBLE_FOR_PROMOTION"
        if eligible
        else "RETAIN_EXISTING_CHAMPION"
    ),
    "champion_overwritten": False,
}

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    json.dump(decision, f, indent=4)

print("\n========== QUALITY GATE RESULTS ==========")

for check, passed in checks.items():
    print(f"{check}: {'PASS' if passed else 'FAIL'}")

print(f"\nMAE improvement: {mae_improvement:.2f}%")
print(f"RMSE change: {rmse_degradation:.2f}%")
print(f"R2 difference: {r2_difference:.4f}")

print("\n========== FINAL DECISION ==========")

if eligible:
    print("CANDIDATE IS ELIGIBLE FOR PROMOTION!")
    print("All quality criteria passed.")
else:
    print("RETAIN EXISTING CHAMPION.")
    print("One or more quality criteria failed.")

print("\nChampion model has NOT been overwritten.")
print(f"Decision report saved: {output_path}")
