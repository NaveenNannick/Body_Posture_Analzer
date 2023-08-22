from sklearn.metrics import accuracy_score # Accuracy metrics
import pickle
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))
fit_models['rf'].predict(X_test)
y_test
with open('pose_analyzer.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)