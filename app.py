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
import pickle

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
# 加载模型 - 使用兼容的加载方式
# ============================================================================

print("Loading models...")

# 修复：使用pickle加载CatBoost模型
try:
    # 尝试用joblib加载，如果失败则用pickle
    diagnosis_model = joblib.load(DIAGNOSIS_MODEL_PATH)
except Exception as e:
    print(f"Joblib加载失败，尝试使用pickle加载: {e}")
    with open(DIAGNOSIS_MODEL_PATH, 'rb') as f:
        diagnosis_model = pickle.load(f)

# 对于其他模型，继续使用joblib
prognosis_model = joblib.load(PROGNOSIS_MODEL_PATH)

# 尝试不同方式加载SHAP解释器
try:
    explainer = joblib.load(SHAP_EXPLAINER_PATH)
except Exception as e:
    print(f"Joblib加载SHAP解释器失败，尝试使用pickle加载: {e}")
    with open(SHAP_EXPLAINER_PATH, 'rb') as f:
        explainer = pickle.load(f)

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
    try:
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
    except Exception as e:
        print(f"生成SHAP图时出错: {e}")
        # 返回一个占位符PDF
        shap_filename = f'shap_placeholder_{uuid.uuid4().hex}.pdf'
        shap_path = os.path.join('static', 'temp', shap_filename)
        os.makedirs(os.path.join('static', 'temp'), exist_ok=True)
        
        # 创建简单的占位符PDF
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(shap_path) as pdf:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'SHAP Plot Could Not Be Generated\n\nPlease check the model and data format.',
                   ha='center', va='center', fontsize=14, wrap=True)
            ax.set_title('SHAP Force Plot - Placeholder')
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
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
        
        # 使用predict_proba获取概率
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
        print(f"诊断预测错误: {e}")
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
        
        # 使用CoxPH模型的predict_partial_hazard方法
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
        print(f"预后预测错误: {e}")
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
# 清理临时文件的任务（可选）
# ============================================================================

import atexit
import glob
import time

def cleanup_temp_files():
    """清理临时文件"""
    temp_dir = 'static/temp'
    if os.path.exists(temp_dir):
        for file in glob.glob(os.path.join(temp_dir, 'shap_*.pdf')):
            try:
                # 删除创建时间超过1小时的文件
                if time.time() - os.path.getctime(file) > 3600:
                    os.remove(file)
            except:
                pass

# 注册清理函数
atexit.register(cleanup_temp_files)

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