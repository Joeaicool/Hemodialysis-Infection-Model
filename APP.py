import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

# =========================
# 1. 页面配置 (Page Config)
# =========================
st.set_page_config(
    page_title="AI Clinical Risk Predictor | Precision Medicine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 2. 高级医学期刊风格 CSS
# =========================
st.markdown("""
<style>
    /* 全局背景色 */
    .main { background-color: #F8F9FA; }
    
    /* 顶部标题卡片 */
    .title-box {
        background: linear-gradient(135deg, #0A2540 0%, #1750A1 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .title-box h1 { margin: 0; font-size: 2.2rem; font-weight: 700; font-family: 'Helvetica Neue', sans-serif;}
    .title-box p { margin-top: 10px; font-size: 1.1rem; opacity: 0.9; }
    
    /* 临床说明卡片 */
    .clinical-note {
        background-color: #EBF4FA;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 6px solid #1750A1;
        margin-bottom: 2rem;
        color: #0A2540;
        font-size: 1rem;
    }
    
    /* 内容卡片 */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        margin-bottom: 1.5rem;
    }
    .card-title {
        color: #0A2540;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 0.5rem;
    }
    
    /* 页脚 */
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        color: #64748B;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 3. 页面头与图片
# =========================
st.markdown(
    """
    <div class="title-box">
        <h1>Advanced Machine Learning Model for Clinical Risk Prediction</h1>
        <p>Explainable Artificial Intelligence (XAI) for Personalized Patient Assessment</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="clinical-note">
    <b>Objective:</b> This tool leverages an advanced machine learning algorithm to predict individual clinical risk based on routine biomarker panels. <br>
    <b>Note for Patients & Clinicians:</b> This is a supplementary decision-support tool. Final medical decisions should always be made by a qualified healthcare professional in conjunction with comprehensive clinical evaluations.
    </div>
    """,
    unsafe_allow_html=True
)

# 插入横幅图片
if os.path.exists("PIC1.png"):
    st.image("PIC1.png", use_column_width=True)

# =========================
# 4. 数据与模型加载
# =========================
MODEL_FILE = "model.pkl"
DATA_FILE = "Final_Cleaned_Data.xlsx"
FEATURES = ['CysC', 'ADA', 'MONO_pct', 'TP', 'MYO', 'HCT']

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_data():
    return pd.read_excel(DATA_FILE)

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"⚠️ Error loading files. Please ensure `{MODEL_FILE}` and `{DATA_FILE}` are in the same directory. \nDetails: {e}")
    st.stop()

# 提取特征数据范围用于初始化输入框
X_f = df.drop(columns=['status', 'ID'], errors='ignore')

# =========================
# 5. 用户输入面板 (UI设计对医生极其友好)
# =========================
st.markdown('<div class="card"><div class="card-title">📝 Step 1: Patient Biomarker Input</div>', unsafe_allow_html=True)

# 使用 3 列布局让界面更紧凑美观
col1, col2, col3 = st.columns(3)
columns = [col1, col2, col3, col1, col2, col3]

input_vals = []
for idx, f in enumerate(FEATURES):
    with columns[idx]:
        if pd.api.types.is_numeric_dtype(X_f[f]):
            min_val = float(X_f[f].min())
            max_val = float(X_f[f].max())
            median_val = float(X_f[f].median())
            # 增加一点帮助文本提升专业感
            v = st.number_input(
                f"{f}",
                min_value=min_val,
                max_value=max_val,
                value=median_val,
                help=f"Reference Range in dataset: {min_val:.2f} - {max_val:.2f}"
            )
        else:
            opts = X_f[f].unique().tolist()
            v = st.selectbox(f"{f}", opts)
        input_vals.append(v)
        
st.markdown('</div>', unsafe_allow_html=True)

# 构建输入DataFrame
X_in = pd.DataFrame([input_vals], columns=FEATURES)

# =========================
# 6. 预测与可视化 (核心结果区)
# =========================
# 居中显示一个漂亮的大按钮
_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    predict_btn = st.button("🚀 Run Personalized Risk Assessment", type="primary", use_container_width=True)

if predict_btn:
    st.markdown('<div class="card"><div class="card-title">📊 Step 2: Prediction Results & Interpretation</div>', unsafe_allow_html=True)
    
    # 模型预测
    prob_pos = model.predict_proba(X_in)[0][1] * 100
    pred_class = model.predict(X_in)[0]

    res_c1, res_c2 = st.columns([1.2, 1])

    with res_c1:
        st.markdown('#### 🩺 Clinical Conclusion')
        if prob_pos >= 50:
            st.error("### ⚠️ High Risk Detected")
            st.write(f"The model indicates a **higher likelihood** of positive clinical outcome/progression.")
        else:
            st.success("### ✅ Low Risk Detected")
            st.write(f"The model indicates a **lower likelihood** of positive clinical outcome/progression.")
            
        st.info(f"**Calculated Probability:** **{prob_pos:.2f}%**")
        st.write("*Interpretation: A probability closer to 100% indicates higher risk. Please review the SHAP explainability plots below to understand which specific biomarkers are driving this patient's risk profile.*")

    with res_c2:
        # Plotly 风险表盘 (顶级刊物最喜欢的直观方式)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pos,
            number={"suffix": "%", "font": {"size": 40, "color": "#0A2540"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "#1750A1"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 30], "color": "rgba(40, 167, 69, 0.2)"},  # 绿
                    {"range": [30, 70], "color": "rgba(255, 193, 7, 0.2)"}, # 黄
                    {"range": [70, 100], "color": "rgba(220, 53, 69, 0.2)"},# 红
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": prob_pos
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # 7. SHAP 机器学习可解释性 (XAI)
    # =========================
    st.markdown('<div class="card"><div class="card-title">🔍 Step 3: AI Explainability (SHAP Analysis)</div>', unsafe_allow_html=True)
    st.write("The plots below unpack the 'black box' of the AI, showing exactly how each biomarker pushes the patient's risk higher (Red) or lower (Blue) compared to the baseline.")

    try:
        with st.spinner('Calculating SHAP values for personalized explainability...'):
            # 兼容处理 SHAP explainer
            # 使用样本背景数据以加快计算速度且避免全量数据过大
            bg_data = shap.sample(X_f[FEATURES], min(100, len(X_f)))
            
            try:
                explainer = shap.Explainer(model, bg_data)
                sv_in = explainer(X_in)
                sv_values = sv_in.values[0]
                base_val = sv_in.base_values[0]
            except:
                # 备用方案 (针对无法直接使用Explainer的模型如特定版本的随机森林/MLP)
                explainer = shap.KernelExplainer(model.predict_proba, bg_data)
                shap_values_raw = explainer.shap_values(X_in)
                if isinstance(shap_values_raw, list): # 多分类或二分类返回list
                    sv_values = shap_values_raw[1][0]
                    base_val = explainer.expected_value[1]
                else: # 某些情况返回3D array
                    sv_values = shap_values_raw[0, :, 1] if shap_values_raw.ndim == 3 else shap_values_raw[0]
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                
                # 手动构建 Explanation 对象给 waterfall 画图用
                sv_in = shap.Explanation(values=sv_values, base_values=base_val, data=X_in.iloc[0].values, feature_names=FEATURES)

            p1, p2 = st.columns(2)

            with p1:
                st.markdown("**SHAP Waterfall Plot**")
                fig_wf, ax_wf = plt.subplots(figsize=(6, 5), dpi=150)
                shap.plots.waterfall(sv_in if isinstance(sv_in, shap.Explanation) else sv_in[0], max_display=10, show=False)
                st.pyplot(fig_wf, use_container_width=True)
                plt.close(fig_wf)

            with p2:
                st.markdown("**Feature Contribution Ranking**")
                # 计算贡献百分比
                abs_sv = np.abs(sv_values)
                total = abs_sv.sum() if abs_sv.sum() != 0 else 1.0
                pct = abs_sv / total * 100

                contrib_df = pd.DataFrame({
                    "Biomarker": FEATURES,
                    "Patient Value": X_in.iloc[0].values,
                    "Effect": ["⬆️ Increased Risk" if v > 0 else "⬇️ Decreased Risk" for v in sv_values],
                    "Contribution Impact": pct
                }).sort_values("Contribution Impact", ascending=False)

                st.dataframe(
                    contrib_df.style.format({
                        "Patient Value": "{:.2f}",
                        "Contribution Impact": "{:.1f}%"
                    }).background_gradient(subset=['Contribution Impact'], cmap='Blues'),
                    use_container_width=True,
                    hide_index=True
                )
    except Exception as e:
        st.warning(f"⚠️ Could not generate SHAP explanation. Model architecture might require specific SHAP explainer settings. Details: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# 8. 专业页脚
# =========================
st.markdown(
    """
    <div class="footer">
        <b>Author:</b> Sheng Liang, M.D., Ph.D.<br>
        <b>Affiliation:</b> Hengzhou City People's Hospital, Hengzhou, Guangxi, China<br>
        <i>Powered by Streamlit | Developed for clinical research and precision medicine.</i>
    </div>
    """,
    unsafe_allow_html=True
)
