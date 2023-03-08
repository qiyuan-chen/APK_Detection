import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib

# 读入数据集
data = pd.read_csv('dataset.csv')

# 划分数据集为训练数据和预测数据
train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

# 提取特征和标签
features = train_data.drop('Class', axis=1)
labels = train_data['Class']

# 训练决策树模型
dt = DecisionTreeClassifier()
dt.fit(features, labels)

# 在验证集上评估模型性能
val_features = val_data.drop('Class', axis=1)
val_labels = val_data['Class']
val_pred = dt.predict(val_features)
print(classification_report(val_labels, val_pred))

# 保存模型
joblib.dump(dt, 'model.pkl')
