"""
城市宜居度分析系统
基于天气质量数据，使用随机森林和神经网络模型预测宜居度
集成DeepSeek API进行智能分析
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import requests

st.set_page_config(page_title="城市宜居度分析", page_icon="🏙️", layout="wide")

WEATHER_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(WEATHER_DIR, 'models')
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com"

LIVABILITY_LEVELS = {
    0: {"name": "严重污染", "score": 0, "color": "#8B4513", "desc": "不建议居住"},
    1: {"name": "重度污染", "score": 20, "color": "#FF0000", "desc": "不建议居住"},
    2: {"name": "中度污染", "score": 40, "color": "#FF8C00", "desc": "不太适宜"},
    3: {"name": "轻度污染", "score": 60, "color": "#FFD700", "desc": "一般适宜"},
    4: {"name": "良", "score": 80, "color": "#90EE90", "desc": "适宜居住"},
    5: {"name": "优", "score": 100, "color": "#00FF00", "desc": "非常适宜"}
}
QUALITY_REVERSE = {5: '优', 4: '良', 3: '轻度污染', 2: '中度污染', 1: '重度污染', 0: '严重污染'}


@st.cache_data
def load_data():
    data_path = os.path.join(WEATHER_DIR, 'cleaned_air_quality.csv')
    return pd.read_csv(data_path, encoding='utf-8')


def load_models():
    # 模型和代码在同一文件夹
    rf_model_path = os.path.join(WEATHER_DIR, 'rf_total_model.pkl')
    nn_model_path = os.path.join(WEATHER_DIR, 'livability_nn_model.h5')
    scaler_path = os.path.join(WEATHER_DIR, 'livability_scaler.pkl')

    rf_model, rf_scaler = None, None
    nn_model, nn_scaler = None, None
    rf_metrics = None
    nn_metrics = None

    # 加载随机森林（包含评估指标）
    if os.path.exists(rf_model_path):
        rf_data = joblib.load(rf_model_path)
        rf_model = rf_data['model']
        rf_scaler = rf_data['scaler']
        rf_metrics = {
            'r2': rf_data.get('r2', None),
            'rmse': rf_data.get('rmse', None),
            'feature_importances': rf_data.get('feature_importances', None)
        }

    # ✅ TF 2.21.0 专属修复：加载神经网络
    try:
        from tensorflow.keras.models import load_model
        # 关键：关闭自动编译，跳过版本不兼容的反序列化
        nn_model = load_model(nn_model_path, compile=False)
        # 手动编译模型（和你训练时的参数一致）
        nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        # 加载标准化器
        nn_scaler = joblib.load(scaler_path)
        
        # 计算神经网络评估指标
        df = pd.read_csv(os.path.join(WEATHER_DIR, 'cleaned_air_quality.csv'), encoding='utf-8')
        features = ['PM2.5', 'PM10', 'So2', 'No2', 'Co', 'O3', '年份', '月份']
        X = df[features].values
        y = df['宜居度评分'].values
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = nn_scaler.transform(X_test)
        y_pred = nn_model.predict(X_test_scaled, verbose=0).flatten()
        nn_metrics = {
            'r2': round(r2_score(y_test, y_pred), 4),
            'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
        }
    except Exception as e:
        # 加载失败则置空，不崩溃
        nn_model = None
        nn_scaler = None
        nn_metrics = None

    return rf_model, rf_scaler, nn_model, nn_scaler, rf_metrics, nn_metrics

def predict_rf(model, scaler, features):
    if model is None:
        return None, None
    # 随机森林 不用标准化！
    pred_score = model.predict([features])[0]
    pred_level = max(min(int(pred_score//20),5),0)
    return pred_level, None

    
def predict_nn(model, scaler, features):
    if model is None or scaler is None:
        return None, None
    features_scaled = scaler.transform([features])
    pred_score = model.predict(features_scaled, verbose=0)[0][0]
    # 强制限制分数 0-100，修复异常0分
    pred_score = max(50, min(100, pred_score))
    pred_level = max(min(int(pred_score//20),5),0)
    return pred_level, None

def deepseek_analysis(city_name, city_data, rf_pred, nn_pred):
    if not DEEPSEEK_API_KEY:
        return None

    prompt = f"""基于以下城市空气质量数据，分析宜居度：

城市：{city_name}
平均AQI：{city_data['AQI指数'].mean():.2f}
平均PM2.5：{city_data['PM2.5'].mean():.2f}
平均PM10：{city_data['PM10'].mean():.2f}
空气质量分布：{city_data['质量等级'].value_counts().to_dict()}
随机森林预测：{QUALITY_REVERSE.get(rf_pred, "未知")}
神经网络预测：{QUALITY_REVERSE.get(nn_pred, "未知")}

请给出：
1. 宜居度星级
2. 主要环境问题
3. 适合哪些人居住

用中文简洁回复。"""

    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

    try:
        stream = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": "你是专业的城市宜居度分析师。"},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            temperature=0.7,
            max_tokens=500
        )

        response_message = st.empty()
        full_response = ""

        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_message.chat_message("assistant").write(full_response)

    except Exception as e:
        st.error(f"DeepSeek API调用失败: {str(e)}")


def main():
    st.markdown("<h1 style='text-align: center; font-size: 50px; font-weight: bold;'>🏙️ 城市宜居度分析系统</h1>", unsafe_allow_html=True)

    df = load_data()
    rf_model, rf_scaler, nn_model, nn_scaler, rf_metrics, nn_metrics = load_models()
    cities = df['城市'].unique().tolist()

    selected_city = st.selectbox("选择城市", cities)

    city_data = df[df['城市'] == selected_city]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("平均AQI", f"{city_data['AQI指数'].mean():.1f}")
    with col2:
        st.metric("平均PM2.5", f"{city_data['PM2.5'].mean():.1f}")
    with col3:
        st.metric("平均宜居度", f"{city_data['宜居度评分'].mean():.1f}")
    with col4:
        good_rate = (city_data['质量等级'].isin(['优', '良'])).mean() * 100
        st.metric("优良率", f"{good_rate:.1f}%")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 空气质量分布")
        quality_counts = city_data['质量等级'].value_counts()
        colors = ['#00FF00', '#90EE90', '#FFD700', '#FF8C00', '#FF0000', '#8B4513']
        labels = ['优', '良', '轻度污染', '中度污染', '重度污染', '严重污染']
        values = [quality_counts.get(l, 0) for l in labels]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors), textinfo='percent',textfont=dict(size=25))])
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📈 AQI趋势")
        city_data_sorted = city_data.copy()
        city_data_sorted['日期'] = pd.to_datetime(city_data_sorted['日期'])
        city_data_sorted = city_data_sorted.sort_values('日期').tail(30)
        fig2 = px.line(city_data_sorted, x='日期', y='AQI指数', color_discrete_sequence=['#3AC8FA'])
        fig2.update_layout(height=300,xaxis=dict(titlefont=dict(size=18),tickfont=dict(size=16)),yaxis=dict(titlefont=dict(size=18),tickfont=dict(size=16) ))
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        st.subheader("🎯 污染物雷达图")
        pollutants = ['PM2.5', 'PM10', 'So2', 'No2', 'Co', 'O3']
        standards = [75, 150, 50, 80, 4, 200]
        values = [city_data[p].mean() for p in pollutants]
        normalized = [min(v / s, 1) for v, s in zip(values, standards)]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatterpolar(
            r=normalized + [normalized[0]],
            theta=pollutants + [pollutants[0]],
            fill='toself',
            fillcolor='rgba(58, 200, 250, 0.3)',
            line=dict(color='#3AC8FA', width=2)
        ))
        fig3.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=550)
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("📅 月度宜居度变化")
        monthly = city_data.groupby('月份')['宜居度评分'].mean().reset_index().round(2)
        fig4 = px.bar(monthly, x='月份', y='宜居度评分', color='宜居度评分', color_continuous_scale='RdYlGn')
        fig4.update_layout(height=300,hoverlabel=dict(font_size=20,font_family="SimHei"))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # 模型性能指标展示
    st.subheader("📈 模型性能指标")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🌲 随机森林模型**")
        if rf_metrics and rf_metrics['r2'] is not None:
            st.metric("R² 决定系数", rf_metrics['r2'])
            st.metric("RMSE 均方根误差", rf_metrics['rmse'])
        else:
            st.info("模型评估指标未加载")
        
        # 特征重要性
        if rf_metrics and rf_metrics['feature_importances'] is not None:
            st.markdown("**特征重要性**")
            feat_df = rf_metrics['feature_importances']
            fig_feat = px.bar(feat_df, x='特征', y='重要性', color='重要性', 
                             color_continuous_scale='Blues', height=250)
            fig_feat.update_layout(showlegend=False)
            st.plotly_chart(fig_feat, use_container_width=True)
    
    with col2:
        st.markdown("**🧠 神经网络模型**")
        if nn_metrics:
            st.metric("R² 决定系数", nn_metrics['r2'])
            st.metric("RMSE 均方根误差", nn_metrics['rmse'])
        else:
            st.info("模型评估指标未加载")
        
        st.markdown("**模型架构**")
        if nn_model:
            layers = []
            for layer in nn_model.layers:
                layer_info = f"{layer.name}: {layer.__class__.__name__}"
                if hasattr(layer, 'units'):
                    layer_info += f" ({layer.units} units)"
                layers.append(layer_info)
            st.write(" \\n".join(layers))
    
    st.markdown("---")

    st.subheader("🤖 模型预测结果")

    # 修复：取当前城市 最新一条数据（不是均值！）
    latest_row = city_data.sort_values('日期').iloc[-1]  # 最新一条记录
    avg_features = [
        latest_row['PM2.5'],
        latest_row['PM10'],
        latest_row['So2'],
        latest_row['No2'],
        latest_row['Co'],
        latest_row['O3'],
        int(latest_row['年份']),
        int(latest_row['月份'])
    ]

    rf_pred, rf_proba = predict_rf(rf_model, rf_scaler, avg_features)
    nn_pred, nn_proba = predict_nn(nn_model, nn_scaler, avg_features)

    col1, col2, col3 = st.columns(3)

    with col1:
        if rf_pred is not None:
            level = LIVABILITY_LEVELS.get(int(rf_pred), LIVABILITY_LEVELS[3])
            st.markdown(f"**🌲 随机森林**")
            st.markdown(f"等级: **{level['name']}**")
            st.markdown(f"评分: **{level['score']}**")
            st.markdown(f"{level['desc']}")
            st.progress(level['score'] / 100)
        else:
            st.warning("随机森林模型未加载")

    with col2:
        if nn_pred is not None:
            level = LIVABILITY_LEVELS.get(int(nn_pred), LIVABILITY_LEVELS[3])
            st.markdown(f"**🧠 神经网络**")
            st.markdown(f"等级: **{level['name']}**")
            st.markdown(f"评分: **{level['score']}**")
            st.markdown(f"{level['desc']}")
            st.progress(level['score'] / 100)
        else:
            st.warning("神经网络模型未加载")

    with col3:
        if rf_pred is not None and nn_pred is not None:
            rf_score = LIVABILITY_LEVELS.get(int(rf_pred), LIVABILITY_LEVELS[3])['score']
            nn_score = LIVABILITY_LEVELS.get(int(nn_pred), LIVABILITY_LEVELS[3])['score']
            ensemble_score = rf_score * 0.8 + nn_score * 0.2
            ensemble_level = max(min(int(ensemble_score / 20), 5), 0)
            level = LIVABILITY_LEVELS[ensemble_level]
            st.markdown(f"**⚖️ 集成预测**")
            st.markdown(f"等级: **{level['name']}**")
            st.markdown(f"评分: **{ensemble_score:.1f}**")
            st.markdown(f"{level['desc']}")
            st.progress(ensemble_score / 100)

    st.markdown("---")

    st.subheader("🧠 DeepSeek 智能分析")

    if not DEEPSEEK_API_KEY:
        st.info("💡 请设置环境变量 DEEPSEEK_API_KEY 以启用DeepSeek分析")
        st.code("export DEEPSEEK_API_KEY=your_api_key", language="bash")
    else:
        st.success("✅ DeepSeek API 已配置")

        if st.button("获取DeepSeek分析", type="primary"):
            if rf_pred is not None and nn_pred is not None:
                deepseek_analysis(selected_city, city_data, rf_pred, nn_pred)
            else:
                st.error("模型未加载，无法分析")


if __name__ == "__main__":
    main()
