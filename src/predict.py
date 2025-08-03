# ============================================================
# Script: predict.py
# Project: Weather-based Temperature Prediction
# Description:
#   - Loads trained stacking model
#   - Generates predictions for test stations
#   - Applies clipping and saves submission CSV
# Author: Hyunbin Ki
# ============================================================

# 13-6. Test 예측 + 클리핑 + 제출 저장
test_pred = stacking_model.predict(X_test_scaled)
test_pred = np.clip(test_pred, best_min, best_max)

submission = pd.DataFrame({
    'id': range(len(test_pred)),
    'target': test_pred
})
submission.to_csv('submission.csv', index=False)

print("\n최적 클리핑 적용된 제출 파일 'submission.csv' 저장 완료!")


