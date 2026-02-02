#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cardiovascular Disease Risk Assessment Web Application
"""

import os
os.environ['SCIPY_ARRAY_API'] = '1'
os.environ['SKLEARN_ARRAY_API_DISPATCH'] = '0'

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import uuid
from datetime import datetime
import io

warnings.filterwarnings('ignore')

# 设置matplotlib
import matplotlib
matplotlib.use('Agg')
plt.ioff()

# ============================================================================
# Flask应用初始化
# ============================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ============================================================================
# 配置参数
# ============================================================================

DIAGNOSIS_MODEL_PATH = './CatBoost_diagnosis.pkl'
PROGNOSIS_MODEL_PATH = './CoxPH_prognosis.pkl'
SHAP_EXPLAINER_PATH = './shap_explainer.pkl'

DIAGNOSIS_THRESHOLD = 0.46
PROGNOSIS_THRESHOLD = 1.944

# ============================================================================
# 加载模型
# ============================================================================

print("Loading models...")
diagnosis_model = joblib.load(DIAGNOSIS_MODEL_PATH)
prognosis_model = joblib.load(PROGNOSIS_MODEL_PATH)
explainer = joblib.load(SHAP_EXPLAINER_PATH)
print("Models loaded successfully!")

# ============================================================================
# 辅助函数
# ============================================================================

def create_input_data(data):
    """构建输入数据框"""
    return pd.DataFrame([{
        'age': float(data['age']),
        'basic_gender': int(data['basic_gender']),
        'basic_height': float(data['basic_height']),
        'basic_weight': float(data['basic_weight']),
        'basic_education': int(data['basic_education']),
        'basic_marital_status': int(data['basic_marital_status']),
        'disease_hypertension': int(data['disease_hypertension']),
        'disease_diabetes': int(data['disease_diabetes']),
        'disease_cancer': int(data['disease_cancer']),
        'disease_lung': int(data['disease_lung']),
        'disease_arthritis': int(data['disease_arthritis']),
        'grip_left': float(data['grip_left']),
        'grip_right': float(data['grip_right']),
        'self_rated_health': int(data['self_rated_health']),
        'smokev': int(data['smokev']),
        'smoken': int(data['smoken']),
        'work': int(data['work']),
        'family_size': int(data['family_size'])
    }])

def generate_shap_plot(X, diagnosis_proba):
    """生成SHAP力图"""
    # 计算SHAP值
    shap_values_original = explainer(X)
    
    # 获取SHAP值和基线值
    original_shap_values = shap_values_original[0].values
    original_base_value = shap_values_original[0].base_values
    
    # 调整SHAP值
    if np.sum(np.abs(original_shap_values)) > 0:
        adjustment_factor = (diagnosis_proba - original_base_value) / np.sum(original_shap_values)
        adjusted_shap_values = original_shap_values * adjustment_factor
    else:
        adjusted_shap_values = original_shap_values
        original_base_value = diagnosis_proba
    
    # 创建SHAP Explanation对象
    custom_explanation = shap.Explanation(
        values=adjusted_shap_values,
        base_values=original_base_value,
        data=shap_values_original[0].data,
        feature_names=X.columns.tolist()
    )
    
    # 绘制力图
    fig = plt.figure(figsize=(12, 6))
    shap.force_plot(
        custom_explanation,
        matplotlib=True,
        text_rotation=45,
        show=False
    )
    
    # 移除f(x)文本
    ax = plt.gca()
    for text in ax.texts:
        if 'f(x)' in text.get_text():
            text.set_visible(False)
    
    plt.tight_layout()
    
    # 保存为临时文件
    shap_filename = f'shap_{uuid.uuid4().hex}.pdf'
    shap_path = os.path.join('static', 'temp', shap_filename)
    os.makedirs(os.path.join('static', 'temp'), exist_ok=True)
    
    plt.savefig(shap_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return shap_filename

# ============================================================================
# 路由
# ============================================================================

@app.route('/')
def index():
    """主页 - 显示输入表单"""
    return render_template('index.html')

@app.route('/predict_diagnosis', methods=['POST'])
def predict_diagnosis():
    """处理诊断预测请求"""
    try:
        # 获取表单数据
        data = request.json
        
        # 构建输入数据框
        X = create_input_data(data)
        
        # ====================================================================
        # 诊断预测
        # ====================================================================
        
        diagnosis_proba = diagnosis_model.predict_proba(X)[:, 1][0]
        diagnosis_risk = "high" if diagnosis_proba >= DIAGNOSIS_THRESHOLD else "low"
        diagnosis_image = "诊断高风险1.png" if diagnosis_risk == "high" else "诊断低风险1.png"
        
        # ====================================================================
        # 生成SHAP力图
        # ====================================================================
        
        shap_filename = generate_shap_plot(X, diagnosis_proba)
        
        # ====================================================================
        # 返回结果
        # ====================================================================
        
        result = {
            'success': True,
            'diagnosis': {
                'score': round(diagnosis_proba, 4),
                'risk': diagnosis_risk,
                'threshold': DIAGNOSIS_THRESHOLD,
                'image': f'/static/{diagnosis_image}',
                'shap_pdf': f'/static/temp/{shap_filename}',
                'shap_filename': shap_filename
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/predict_prognosis', methods=['POST'])
def predict_prognosis():
    """处理预后预测请求"""
    try:
        # 获取表单数据
        data = request.json
        
        # 构建输入数据框
        X = create_input_data(data)
        
        # ====================================================================
        # 预后预测
        # ====================================================================
        
        prognosis_risk_score = prognosis_model.predict_partial_hazard(X).values[0]
        prognosis_risk = "high" if prognosis_risk_score >= PROGNOSIS_THRESHOLD else "low"
        prognosis_image = "预后图1高风险.png" if prognosis_risk == "high" else "预后图1低风险.png"
        
        # ====================================================================
        # 返回结果
        # ====================================================================
        
        result = {
            'success': True,
            'prognosis': {
                'score': round(prognosis_risk_score, 4),
                'risk': prognosis_risk,
                'threshold': PROGNOSIS_THRESHOLD,
                'image': f'/static/{prognosis_image}',
                'image2': '/static/预后图2.png'
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/download_shap/<filename>')
def download_shap(filename):
    """下载SHAP PDF文件"""
    try:
        return send_file(
            os.path.join('static', 'temp', filename),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='shap_force_plot.pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404

# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    # 确保必要的目录存在
    os.makedirs('static/temp', exist_ok=True)
    
    print("\n" + "="*80)
    print("Cardiovascular Disease Risk Assessment System")
    print("="*80)
    print("\nServer starting...")
    print("Access the application at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)