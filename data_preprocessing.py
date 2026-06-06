import pandas as pd
import numpy as np

def load_data(file_path):
    """加载CSV数据"""
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"原始数据行数: {len(df)}")
    print(f"数据列名: {df.columns.tolist()}")
    return df

def clean_data(df):
    """数据清洗"""
    # 去除列名中的空格
    df.columns = df.columns.str.strip()
    
    # 检查缺失值
    print("\n缺失值统计:")
    print(df.isnull().sum())
    
    # 去除缺失值
    df = df.dropna()
    print(f"\n去除缺失值后行数: {len(df)}")
    
    # 质量等级映射为宜居度评分
    # 优: 100, 良: 80, 轻度污染: 60, 中度污染: 40, 重度污染: 20, 严重污染: 0
    quality_mapping = {
        '优': 100,
        '良': 80,
        '轻度污染': 60,
        '中度污染': 40,
        '重度污染': 20,
        '严重污染': 0
    }
    df['宜居度评分'] = df['质量等级'].map(quality_mapping)
    
    # 提取日期特征
    df['日期'] = pd.to_datetime(df['日期'])
    df['年份'] = df['日期'].dt.year
    df['月份'] = df['日期'].dt.month
    df['季节'] = df['月份'].apply(get_season)
    
    # 异常值处理 - 使用IQR方法
    numeric_cols = ['AQI指数', 'PM2.5', 'PM10', 'So2', 'No2', 'Co', 'O3']
    for col in numeric_cols:
        df = remove_outliers(df, col)
    
    print(f"\n处理异常值后行数: {len(df)}")
    
    return df

def get_season(month):
    """根据月份返回季节"""
    if month in [3, 4, 5]:
        return '春季'
    elif month in [6, 7, 8]:
        return '夏季'
    elif month in [9, 10, 11]:
        return '秋季'
    else:
        return '冬季'

def remove_outliers(df, column):
    """使用IQR方法去除异常值"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def get_city_stats(df):
    """计算各城市统计信息"""
    city_stats = df.groupby('城市').agg({
        'AQI指数': ['mean', 'median', 'min', 'max'],
        'PM2.5': ['mean', 'median'],
        '宜居度评分': ['mean', 'median']
    }).round(2)
    return city_stats

def main():
    # 加载数据
    file_path = '全国主要城市空气质量.csv'
    df = load_data(file_path)
    
    # 数据清洗
    df_clean = clean_data(df)
    
    # 保存清洗后的数据
    df_clean.to_csv('cleaned_air_quality.csv', index=False, encoding='utf-8')
    print("\n清洗后的数据已保存到 cleaned_air_quality.csv")
    
    # 计算城市统计信息
    city_stats = get_city_stats(df_clean)
    city_stats.to_csv('city_stats.csv', encoding='utf-8')
    print("城市统计信息已保存到 city_stats.csv")
    
    return df_clean

if __name__ == '__main__':
    df = main()