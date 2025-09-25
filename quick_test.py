#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import json
import logging
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_msl_dataset(data_base_path):
    """加载MSL数据集"""
    try:
        train_path = os.path.join(data_base_path, "MSL/MSL_train.npy")
        test_path = os.path.join(data_base_path, "MSL/MSL_test.npy")
        test_label_path = os.path.join(data_base_path, "MSL/MSL_test_label.npy")
        
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        test_labels = np.load(test_label_path)
        
        logger.info("MSL数据集形状: 训练%s, 测试%s, 标签%s" % (str(train_data.shape), str(test_data.shape), str(test_labels.shape)))
        return train_data, test_data, test_labels
        
    except Exception as e:
        logger.error("加载MSL数据集失败: %s" % str(e))
        return None, None, None

def preprocess_data(train_data, test_data):
    """数据预处理"""
    try:
        train_data = np.nan_to_num(train_data, nan=0.0)
        test_data = np.nan_to_num(test_data, nan=0.0)
        
        scaler = StandardScaler()
        train_data_scaled = scaler.fit_transform(train_data)
        test_data_scaled = scaler.transform(test_data)
        
        return train_data_scaled, test_data_scaled
        
    except Exception as e:
        logger.error("数据预处理失败: %s" % str(e))
        return train_data, test_data

def test_isolation_forest(train_data, test_data, test_labels):
    """测试IsolationForest"""
    try:
        logger.info("运行IsolationForest...")
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(train_data)
        scores = model.score_samples(test_data)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        auroc = roc_auc_score(test_labels, scores)
        f1 = f1_score(test_labels, (scores > 0.5).astype(int))
        logger.info("IsolationForest: AUROC %.4f, F1 %.4f" % (auroc, f1))
        return {'auroc': auroc, 'f1': f1}
    except Exception as e:
        logger.error("IsolationForest失败: %s" % str(e))
        return {'auroc': 0.5, 'f1': 0.0}

def test_simple_multi_agent(train_data, test_data, test_labels):
    """测试简单多智能体方法"""
    try:
        logger.info("运行简单多智能体方法...")
        
        # 创建多个IsolationForest作为不同的智能体
        agents = []
        for i in range(3):
            agent = IsolationForest(contamination=0.1, random_state=42+i)
            agent.fit(train_data)
            agents.append(agent)
        
        # 获取每个智能体的预测分数
        all_scores = []
        for i, agent in enumerate(agents):
            scores = agent.score_samples(test_data)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            all_scores.append(scores)
            logger.info("智能体 %d 预测完成，平均分数: %.4f" % (i, np.mean(scores)))
        
        # 简单平均融合
        final_scores = np.mean(all_scores, axis=0)
        
        auroc = roc_auc_score(test_labels, final_scores)
        f1 = f1_score(test_labels, (final_scores > 0.5).astype(int))
        
        logger.info("SimpleMultiAgent: AUROC %.4f, F1 %.4f" % (auroc, f1))
        return {'auroc': auroc, 'f1': f1}
        
    except Exception as e:
        logger.error("简单多智能体方法失败: %s" % str(e))
        return {'auroc': 0.5, 'f1': 0.0}

def main():
    """主函数"""
    data_base_path = "/home/jiangh/GinData/workspace/6.17cxz/data/jiangh@10.103.16.22"
    
    logger.info("🚀 开始快速测试")
    logger.info("=" * 50)
    
    # 加载MSL数据集
    train_data, test_data, test_labels = load_msl_dataset(data_base_path)
    if train_data is None:
        logger.error("无法加载数据集，退出")
        return
    
    # 数据预处理
    train_data, test_data = preprocess_data(train_data, test_data)
    
    # 测试IsolationForest
    if_result = test_isolation_forest(train_data, test_data, test_labels)
    
    # 测试简单多智能体方法
    ma_result = test_simple_multi_agent(train_data, test_data, test_labels)
    
    # 保存结果
    results = {
        'IsolationForest': if_result,
        'SimpleMultiAgent': ma_result
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs/quick_test", exist_ok=True)
    
    results_file = "outputs/quick_test/quick_test_results_%s.json" % timestamp
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("结果已保存: %s" % results_file)
    
    # 打印结果摘要
    logger.info("=" * 50)
    logger.info("📊 快速测试结果摘要")
    logger.info("=" * 50)
    logger.info("IsolationForest: AUROC %.4f, F1 %.4f" % (if_result['auroc'], if_result['f1']))
    logger.info("SimpleMultiAgent: AUROC %.4f, F1 %.4f" % (ma_result['auroc'], ma_result['f1']))
    
    if ma_result['auroc'] > if_result['auroc']:
        logger.info("🎉 多智能体方法优于单智能体方法！")
    else:
        logger.info("⚠️ 多智能体方法需要优化")
    
    logger.info("🎉 快速测试完成！")

if __name__ == "__main__":
    main()
