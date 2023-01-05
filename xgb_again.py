from sklearn.datasets import make_regression

from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error

X, y = make_regression(n_samples=100, n_features=1, noise=20)

# 切分训练集、测试集
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=1)

# 调用XGBoost模型，使用训练集数据进行训练（拟合）
my_model = XGBRegressor(
    max_depth=30,
    learning_rate=0.01,
    n_estimators=5,
    silent=True,
    objective='reg:linear',
    booster='gblinear',
    n_jobs=50,
    nthread=None,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=1,
    colsample_bytree=1,
    colsample_bylevel=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    random_state=0,
    seed=None,
    missing=None,
    importance_type='gain')

my_model.fit(train_X, train_y)

# 使用模型对测试集数据进行预测
predictions = my_model.predict(test_X)

# 对模型的预测结果进行评判（平均绝对误差）
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))