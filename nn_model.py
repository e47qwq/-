"""
神经网络模型模块
使用深度学习神经网络进行城市宜居度预测

主要功能：
1. 构建全连接神经网络回归模型（预测连续宜居度评分）
2. 训练模型并使用早停防止过拟合
3. 评估模型性能（R²、RMSE）
4. 提供宜居度预测接口
5. 模型保存和加载（避免重复训练）
"""
import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# 路径配置
MODEL_FILE_PATH = os.path.join(os.path.dirname(__file__), 'livability_nn_model.h5')
SCALER_FILE_PATH = os.path.join(os.path.dirname(__file__), 'livability_scaler.pkl')
DATA_PATH = 'cleaned_air_quality.csv'


def train_neural_network():
    """
    训练神经网络回归模型
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件不存在：{DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding='utf-8')

    features = ['PM2.5', 'PM10', 'So2', 'No2', 'Co', 'O3', '年份', '月份']
    X = df[features].values
    y = df['宜居度评分'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 搭建网络
    model = Sequential([
        Dense(64, activation='relu', input_shape=(8,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )

    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=8,
        validation_split=0.15,
        callbacks=[early_stopping],
        verbose=0
    )

    y_pred = model.predict(X_test_scaled, verbose=0).flatten()
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        'model': model,
        'scaler': scaler,
        'r2': round(r2, 4),
        'rmse': round(rmse, 2),
        'features': features,
        'raw_data': df
    }


def predict_livability_nn(model_info, input_data):
    """单样本宜居度预测接口"""
    if len(input_data) != 8:
        raise ValueError(f"输入必须包含8个特征，当前输入{len(input_data)}个")

    input_array = np.array([input_data])
    scaled_data = model_info['scaler'].transform(input_array)
    predict_score = model_info['model'].predict(scaled_data, verbose=0)[0][0]

    return {
        '预测宜居度评分': round(float(predict_score), 2),
        '特征说明': model_info['features']
    }


def save_model(model_info):
    try:
        model_info['model'].save(MODEL_FILE_PATH)
        joblib.dump(model_info['scaler'], SCALER_FILE_PATH)
        print(f"✅ 模型保存完成")
        return True
    except Exception as e:
        print(f"❌ 保存模型失败：{str(e)}")
        return False


def load_saved_model():
    try:
        if not os.path.exists(MODEL_FILE_PATH) or not os.path.exists(SCALER_FILE_PATH):
            print("⚠️ 无已保存模型，即将重新训练")
            return None

        # TF 2.21.0 专属修复：关闭自动编译
        model = load_model(MODEL_FILE_PATH, compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        scaler = joblib.load(SCALER_FILE_PATH)
        
        # 计算评估指标
        df = pd.read_csv(DATA_PATH, encoding='utf-8')
        features = ['PM2.5', 'PM10', 'So2', 'No2', 'Co', 'O3', '年份', '月份']
        X = df[features].values
        y = df['宜居度评分'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled, verbose=0).flatten()
        
        r2 = round(r2_score(y_test, y_pred), 4)
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
        
        print("✅ 成功加载本地模型")
        return {
            'model': model,
            'scaler': scaler,
            'features': features,
            'r2': r2,
            'rmse': rmse,
            'raw_data': df
        }
    except Exception as e:
        print(f"❌ 加载模型失败：{str(e)}")
        return None


def get_or_train_model():
    model_info = load_saved_model()
    if model_info:
        return model_info

    print("🔄 开始基于本地数据集训练模型...")
    new_model = train_neural_network()
    save_model(new_model)
    print(f"\n📊 模型评估指标：")
    print(f"  R² 决定系数：{new_model['r2']}")
    print(f"  RMSE均方根误差：{new_model['rmse']}")
    return new_model


if __name__ == '__main__':
    # 获取模型
    model = get_or_train_model()
    df = model['raw_data']

    # 从自己的数据集中随机抽取1条样本做预测测试
    random_idx = random.randint(0, len(df)-1)
    sample_row = df.iloc[random_idx]

    # 提取该样本的特征与真实评分
    test_features = [
        sample_row['PM2.5'], sample_row['PM10'], sample_row['So2'],
        sample_row['No2'], sample_row['Co'], sample_row['O3'],
        sample_row['年份'], sample_row['月份']
    ]
    real_score = sample_row['宜居度评分']

    # 模型预测
    pred_result = predict_livability_nn(model, test_features)

    # 输出对比结果
    print("\n🔍 抽取数据集样本测试结果：")
    print(f"样本真实宜居度评分：{real_score:.2f}")
    print(f"模型预测宜居度评分：{pred_result['预测宜居度评分']}")