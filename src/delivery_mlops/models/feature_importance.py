
from pathlib import Path
import joblib
import pandas as pd


def analyze_feature_importance():

    root = Path(__file__).resolve().parents[3]

    model_path = root / "models" / "xgboost_model.joblib"
    output_dir = root / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading XGBoost model...")

    model = joblib.load(model_path)

    # Extract fitted preprocessing and XGBoost components
    preprocessor = model.named_steps["preprocessor"]
    xgb_model = model.named_steps["regressor"]

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    importance_values = xgb_model.feature_importances_

    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance_values
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    # Display top 20 individual features
    print("\nTOP 20 IMPORTANT FEATURES")
    print("=" * 60)

    print(
        importance_df.head(20).to_string(index=False)
    )

    # Save individual feature importance
    importance_df.to_csv(
        output_dir / "xgboost_feature_importance.csv",
        index=False
    )

    # Aggregate importance by original feature
    def get_original_feature(name):

        name = name.replace("categorical__", "")
        name = name.replace("numerical__", "")

        for col in ["region_id", "aoi_type", "city"]:
            if name == col or name.startswith(col + "_"):
                return col

        return name

    importance_df["original_feature"] = (
        importance_df["feature"].apply(get_original_feature)
    )

    grouped_importance = (
        importance_df.groupby("original_feature")["importance"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    print("\nIMPORTANCE BY ORIGINAL FEATURE")
    print("=" * 60)

    print(grouped_importance.to_string(index=False))

    # Save grouped importance
    grouped_importance.to_csv(
        output_dir / "xgboost_grouped_importance.csv",
        index=False
    )

    print("\nBoth feature importance reports saved successfully!")


if __name__ == "__main__":
    analyze_feature_importance()
