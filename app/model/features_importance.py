import matplotlib.pyplot as plt
import xgboost as xgb
import os

model = os.path.join(os.path.dirname(__file__),
                     '../..', 'model_-0.034003352103285714_17-11.json')

best_model = xgb.XGBRegressor()
best_model.load_model(model)
# 'weight' is the number of times a feature is used
figsize = (50, 40)  # Width, Height in inches

# 'weight' is the number of times a feature is used
plt.figure(figsize=figsize)
xgb.plot_importance(best_model, importance_type='weight')
plt.title("Feature Importance by Weight")
plt.savefig('feature_importance_weight.png', dpi=300)  # High resolution
plt.close()

# 'gain' is the average gain of each feature
plt.figure(figsize=figsize)
xgb.plot_importance(best_model, importance_type='gain')
plt.title("Feature Importance by Gain")
plt.savefig('feature_importance_gain.png', dpi=300)  # High resolution
plt.close()

# 'cover' is the average coverage of each feature
plt.figure(figsize=figsize)
xgb.plot_importance(best_model, importance_type='cover')
plt.title("Feature Importance by Cover")
plt.savefig('feature_importance_cover.png', dpi=300)  # High resolution
plt.close()
