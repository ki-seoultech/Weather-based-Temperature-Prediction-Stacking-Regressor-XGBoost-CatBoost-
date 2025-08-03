# ============================================================
# Script: train_models.py
# Project: Weather-based Temperature Prediction
# Description:
#   - Loads preprocessed data
#   - Splits data into train/validation
#   - Defines and trains XGBoost, CatBoost models
#   - Combines models using Ridge Stacking
#   - Evaluates performance and saves trained model
# Author: Hyunbin Ki
# ============================================================

# 10. 학습용 데이터 구성
drop_cols = ['id', 'date', 'target']
features = [col for col in train.columns if col not in drop_cols]
X = train[features]
y = train['target']
X_test = test[features]

mask = (y >= -20) & (y <= 20)
X_cleaned = X[mask]
y_cleaned = y[mask]

# target 클리핑 (남은 것들도 너무 과한 영향 방지)
y_cleaned = np.clip(y_cleaned, -10, 10)


# 11. 학습/검증 분할
X_train, X_val, y_train, y_val = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

dtrain = DMatrix(X_train, label=y_train)
dval = DMatrix(X_val, label=y_val)

# 12. XGBoost 학습
#그리드 서치로 최적의 하이퍼파라미터값 결정완료.
#참고용으로!
'''
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'seed': 42
}
evals = [(dtrain, 'train'), (dval, 'valid')]
xgb_model = xgb_train(params, dtrain, num_boost_round=1000,
                      evals=evals, early_stopping_rounds=50, verbose_eval=100)

# 중요도 추출
xgb_importance = xgb_model.get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'feature': list(xgb_importance.keys()),
    'importance': list(xgb_importance.values())
}).sort_values(by='importance', ascending=False).head(30)  # 상위 30개
'''
'''
# 시각화
plt.figure(figsize=(10, 8))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel("Gain")
plt.title("Top 30 Feature Importances from XGBoost")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
#vapor 시각화
'''

'''
# 시각화
plt.figure(figsize=(10, 6))
plt.barh(vapor_importance_df['feature'], vapor_importance_df['gain'])
plt.xlabel('Gain')
plt.title('XGBoost Gain for Vapor-related Features')
plt.tight_layout()
plt.show()
# 1. 중요도 추출 (gain 기준)
xgb_importance = xgb_model.get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'feature': list(xgb_importance.keys()),
    'gain': list(xgb_importance.values())
})
'''


'''
# 4. 시각화
plt.figure(figsize=(10, 6))
plt.barh(dew_importance_df['feature'], dew_importance_df['gain'])
plt.xlabel("Gain")
plt.title("Gain of Dew Point Related Features (XGBoost)")
plt.tight_layout()
plt.show()
# 1. gain 정보를 포함한 DataFrame 생성
xgb_importance = xgb_model.get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'feature': list(xgb_importance.keys()),
    'gain': list(xgb_importance.values())
})
'''


'''
# 4. 시각화
plt.figure(figsize=(10, 6))
plt.barh(humidity_importance['feature'], humidity_importance['gain'])
plt.xlabel("Gain")
plt.title("XGBoost Gain: Humidity Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
'''
'''
# 5. 정렬된 상위 feature 확인
comparison_df_sorted = comparison_df.sort_values(by='|correlation|', ascending=False)
print(comparison_df_sorted[['correlation', 'gain']].head(30))
'''

#13-1. data scaling
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
X_val_scaled = scaler.transform(imputer.transform(X_val))
X_test_scaled = scaler.transform(imputer.transform(X_test))


# 13-2. 최적화된 모델 정의

xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.04539959788704079,
    max_depth=6,
    subsample=0.9026957782384011,
    colsample_bytree=0.5552962436558909,
    reg_alpha=0.8206921855009772,
    reg_lambda=0.7415963140774737,
    random_state=42
)
#xgb.fit(X_train_scaled, y_train)


cat = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.09982049649669206,
    depth=8,
    l2_leaf_reg=3.0653347605901677,
    random_strength=1.980696940191291,
    bagging_temperature=0.4021996337809488,
    random_state=42,
    verbose=0
)

# 13-3. Stacking
stacking_model= StackingRegressor(
    estimators=[
        ('xgb', xgb),
        ('cat', cat)
    ],
    final_estimator=Ridge(alpha=1.0),
    passthrough=True,
    n_jobs=-1

)




# 13-4. 학습

stacking_model.fit(X_train_scaled, y_train)
val_pred = stacking_model.predict(X_val_scaled)

# 최적 클리핑 범위 (Optuna 결과)
best_min = -9.960395198668593
best_max = 11.365031978405499

# 🔧 클리핑 적용
val_pred_clipped = np.clip(val_pred, best_min, best_max)

# 평가
rmse = np.sqrt(mean_squared_error(y_val, val_pred_clipped))
r2 = r2_score(y_val, val_pred_clipped)

print("📌 Clipped Validation RMSE:", rmse)
print("📌 Clipped Validation R²:", r2)

