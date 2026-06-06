"""
随机森林模型模块
使用随机森林算法进行城市宜居度预测

主要功能：
1. 训练随机森林分类器
2. 评估模型性能（准确率、分类报告）
3. 分析特征重要性
4. 提供宜居度预测接口
5. 模型保存和加载（避免重复训练）
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import joblib

# 模型保存路径（保存在random_forest文件夹内）
RF_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'rf_total_model.pkl')

def train_random_forest():
    """
    训练随机森林分类模型
    
    返回：
        dict: 包含模型、标准化器、评估指标和特征重要性的字典
    """
    # 加载预处理后的城市统计数据
    df = pd.read_csv('cleaned_air_quality.csv', encoding='utf-8')
    
    # 选择特征列
    features = ['PM2.5', 'PM10', 'So2', 'No2', 'Co', 'O3', '年份', '月份']
    
    # 特征矩阵X和目标变量y
    X = df[features]
    y = df['宜居度评分'].values
    
    # 划分训练集和测试集（80%训练，20%测试）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化特征（随机森林对特征尺度不敏感，但标准化有助于统一量纲）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 初始化随机森林分类器
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # 训练模型
    rf.fit(X_train_scaled, y_train)
    
    # 在测试集上进行预测
    y_pred = rf.predict(X_test_scaled)

    # 回归评估指标
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # 删除分类用的混淆矩阵
    conf_matrix = None


    # 计算特征重要性并排序
    feature_importances = pd.DataFrame({
        '特征': features,
        '重要性': rf.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    # 打包模型信息
    model_info = {
        'model': rf,
        'scaler': scaler,
        'r2': r2,  # 回归分数
        'rmse': rmse,  # 误差
        'feature_importances': feature_importances,
        'features': features
    }
    
    return model_info


def predict_livability(model_info, input_data):
    """
    使用训练好的随机森林模型预测宜居度评分

    参数：
        model_info (dict): 包含模型和相关信息的字典
        input_data (list): 输入的污染物数据列表，顺序需与训练时一致

    返回：
        dict: 包含预测分数结果
    """
    # 将输入数据转换为DataFrame
    df = pd.DataFrame([input_data])

    # 使用训练时的标准化器进行标准化
    scaled_data = model_info['scaler'].transform(df)

    # 进行预测，得到连续评分
    prediction = model_info['model'].predict(scaled_data)

    # 返回保留两位小数的预测分数
    return {
        '预测宜居度评分': round(prediction[0], 2)
    }


def save_model(model_info):
    """
    单文件保存模型+标准化器
    """
    try:
        # 一次性保存所有内容（模型、标准化器、指标）
        joblib.dump(model_info, RF_MODEL_PATH)
        print(f"✅ 随机森林单文件保存至: {RF_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌ 保存模型失败: {str(e)}")
        return False

def load_saved_model():
    """
    加载单文件模型（包含评估指标）
    """
    try:
        if not os.path.exists(RF_MODEL_PATH):
            print("⚠️ 模型文件不存在，需要先训练模型")
            return None
        
        # 加载单文件（包含模型、标准化器和评估指标）
        model_info = joblib.load(RF_MODEL_PATH)
        print("✅ 已加载随机森林单文件模型")
        
        # 确保返回完整的模型信息
        return {
            'model': model_info['model'],
            'scaler': model_info['scaler'],
            'r2': model_info.get('r2', None),
            'rmse': model_info.get('rmse', None),
            'feature_importances': model_info.get('feature_importances', None),
            'features': model_info.get('features', ['PM2.5', 'PM10', 'So2', 'No2', 'Co', 'O3', '年份', '月份'])
        }
    except Exception as e:
        print(f"❌ 加载模型失败: {str(e)}")
        return None

def get_or_train_model():
    """
    获取模型（优先从文件加载，如果不存在则训练并保存）
    
    返回：
        dict: 包含模型、标准化器和评估指标的字典
    """
    # 尝试加载已保存的模型
    model_info = load_saved_model()
    
    if model_info is not None:
        print("📦 使用已保存的随机森林模型（无需重新训练）")
        # 如果加载的模型缺少评估指标，打印警告
        if model_info['r2'] is None:
            print("⚠️ 已保存的模型不包含评估指标")
        return model_info
    else:
        print("🔄 开始训练新的随机森林模型...")
        # 训练模型
        new_model_info = train_random_forest()
        # 保存模型
        save_model(new_model_info)
        return new_model_info

# 如果直接运行此脚本，训练模型并输出结果
if __name__ == '__main__':
    # 使用新的模型加载机制
    model_info = get_or_train_model()
    
    # 如果模型是从文件加载的，需要添加额外的评估信息
    if 'accuracy' not in model_info:
        print("⚠️ 加载的模型不包含评估信息（需要重新训练以获取完整信息）")
    else:
        print(f"随机森林模型 R²分数: {model_info['r2']:.4f}")
        print(f"随机森林模型 RMSE误差: {model_info['rmse']:.2f}")
        print("\n特征重要性:")
        print(model_info['feature_importances'])