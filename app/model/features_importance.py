import matplotlib.pyplot as plt
import xgboost as xgb
import os

model = os.path.join(os.path.dirname(__file__),
                     '..', 'best_model_5.671712954134959_old.json')

best_model = xgb.XGBRegressor()
best_model.load_model(model)
# 'weight' is the number of times a feature is used
xgb.plot_importance(best_model, importance_type='weight')
plt.title("Feature Importance by Weight")
plt.savefig('feature_importance_weight.png')
plt.close()

# 'gain' is the average gain of each feature
xgb.plot_importance(best_model, importance_type='gain')
plt.show()
plt.title("Feature Importance by Gain")
plt.savefig('feature_importance_gain.png')
plt.close()

# 'cover' is the average coverage of each feature
xgb.plot_importance(best_model, importance_type='cover',)
plt.show()
plt.title("Feature Importance by Cover")
plt.savefig('feature_importance_cover.png')
plt.close()
