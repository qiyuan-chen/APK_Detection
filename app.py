# 导入 Flask 框架、渲染模板、请求模块
from flask import Flask, render_template, request

# 导入 Pandas 和 NumPy 库
import pandas as pd
import numpy as np

# 导入预训练模型
import joblib
model = joblib.load('model.pkl')

# 定义预测函数
def predict(data):
    return model.predict(data)

# 创建 Flask 应用程序
app = Flask(__name__)

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')

# 文件上传路由
@app.route('/', methods=['POST'])
def upload_file():
    # 获取上传的文件
    file = request.files['file']
    if file:
        # 读取 CSV 文件数据
        df = pd.read_csv(file)

        # 获取需要预测的数据
        data = df.drop(['Class'], axis=1)

        # 进行预测
        pred = predict(data)

        # 将预测结果添加到数据中
        df['prediction'] = pred

        # 将结果转换成 Pandas 数据框
        result = pd.DataFrame(pred, columns=['预测值'])

        # 创建字典来存储数字到字符串的映射
        mapping = {1: '广告软件', 2: '银行恶意软件', 3: '短信恶意软件', 4: '风险软件', 5: '良性'}

        # 使用 replace() 方法将数字替换为字符串，并将结果存储在新的“类别”列中
        result['类别'] = result['预测值'].replace(mapping)

        # 将结果呈现为 HTML 表格
        return render_template('result.html', tables=[result.to_html(classes='data')], titles=result.columns.values)

    # 如果没有选择文件，则返回首页，并显示错误消息
    else:
        return render_template('index.html', error='No file selected')

# 运行应用程序
if __name__ == '__main__':
    app.run(debug=True)
