import matplotlib.pyplot as plt
import xgboost as xgb
import os

# Load the model
model = os.path.join(os.path.dirname(__file__),
                     '../..', 'model_-0.0011.json')

best_model = xgb.XGBRegressor()
best_model.load_model(model)


def save_feature_importance_plot(importance_type, filename, title, figsize=(15, 10), rotation=45):
    ax = xgb.plot_importance(best_model,
                             importance_type=importance_type,
                             )
    ax.figure.set_size_inches(figsize)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    ax.tick_params(axis='x', labelsize=10, rotation=rotation)
    ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()  # Automatically adjust layout to avoid overlapping
    plt.savefig(filename, dpi=300)  # Save with high resolution
    plt.close()


# Plot and save feature importance by 'weight'
save_feature_importance_plot(
    importance_type='weight',
    filename='feature_importance_weight.png',
    title="Feature Importance by Weight"
)

# Plot and save feature importance by 'gain'
save_feature_importance_plot(
    importance_type='gain',
    filename='feature_importance_gain.png',
    title="Feature Importance by Gain"
)

# Plot and save feature importance by 'cover'
save_feature_importance_plot(
    importance_type='cover',
    filename='feature_importance_cover.png',
    title="Feature Importance by Cover"
)
