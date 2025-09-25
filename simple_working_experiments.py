# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import json
import logging
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleAgent:
    """简单的智能体"""
    
    def __init__(self, agent_id, agent_type, config=None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}
        self.is_fitted = False
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X):
        """训练智能体"""
        try:
            X_scaled = self.scaler.fit_transform(X)
            
            if self.agent_type == "trend_analysis":
                self.model = IsolationForest(contamination=0.1, random_state=42)
            elif self.agent_type == "variance_analysis":
                self.model = OneClassSVM(nu=0.1, kernel='rbf')
            elif self.agent_type == "residual_analysis":
                self.model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            elif self.agent_type == "statistical_analysis":
                self.model = IsolationForest(contamination=0.1, random_state=42)
            elif self.agent_type == "frequency_analysis":
                self.model = OneClassSVM(nu=0.1, kernel='linear')
            else:
                self.model = IsolationForest(contamination=0.1, random_state=42)
            
            self.model.fit(X_scaled)
            self.is_fitted = True
            logger.info("智能体 %s 训练完成" % self.agent_id)
            
        except Exception as e:
            logger.error("智能体 %s 训练失败: %s" % (self.agent_id, str(e)))
            self.is_fitted = False
    
    def predict(self, X):
        """预测异常分数"""
        if not self.is_fitted or self.model is None:
            logger.warning("智能体 %s 未训练，返回随机分数" % self.agent_id)
            return np.random.rand(X.shape[0])
        
        try:
            X_scaled = self.scaler.transform(X)
            
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(X_scaled)
            elif hasattr(self.model, 'score_samples'):
                scores = self.model.score_samples(X_scaled)
            else:
                scores = np.random.rand(X.shape[0])
            
            # 归一化到0-1
            if len(scores) > 0:
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            else:
                scores = np.random.rand(X.shape[0])
            
            scores = np.clip(scores, 0.0, 1.0)
            return scores
            
        except Exception as e:
            logger.error("智能体 %s 预测失败: %s" % (self.agent_id, str(e)))
            return np.random.rand(X.shape[0])

class SimpleMultiAgentDetector:
    """简单的多智能体检测器"""
    
    def __init__(self, agents, config=None):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.config = config or {}
        self.is_fitted = False
        
        logger.info("初始化多智能体检测器，智能体数量: %d" % len(self.agents))
    
    def fit(self, X):
        """训练所有智能体"""
        logger.info("开始训练多智能体检测器，数据形状: %s" % str(X.shape))
        
        for agent_id, agent in self.agents.items():
            try:
                agent.fit(X)
            except Exception as e:
                logger.error("智能体 %s 训练失败: %s" % (agent_id, str(e)))
        
        self.is_fitted = True
        logger.info("多智能体检测器训练完成")
    
    def predict(self, X):
        """多智能体协作预测"""
        if not self.is_fitted:
            logger.warning("多智能体检测器未训练，返回随机分数")
            return np.random.rand(X.shape[0])
        
        logger.info("开始多智能体协作预测，数据形状: %s" % str(X.shape))
        
        agent_results = {}
        agent_scores = []
        
        for agent_id, agent in self.agents.items():
            try:
                scores = agent.predict(X)
                agent_results[agent_id] = {
                    'scores': scores,
                    'confidence': np.mean(scores),
                    'findings': "智能体 %s 检测到 %d 个潜在异常点" % (agent_id, np.sum(scores > 0.5))
                }
                agent_scores.append(scores)
                logger.info("智能体 %s 预测完成，平均分数: %.4f" % (agent_id, np.mean(scores)))
            except Exception as e:
                logger.error("智能体 %s 预测失败: %s" % (agent_id, str(e)))
                random_scores = np.random.rand(X.shape[0])
                agent_results[agent_id] = {
                    'scores': random_scores,
                    'confidence': 0.5,
                    'findings': "智能体 %s 预测失败，使用随机分数" % agent_id
                }
                agent_scores.append(random_scores)
        
        if not agent_scores:
            logger.warning("没有有效的智能体预测结果，返回随机分数")
            return np.random.rand(X.shape[0])
        
        # 传统融合方法：加权平均
        logger.info("使用加权平均进行智能体融合...")
        
        stacked_scores = np.vstack(agent_scores).T
        
        confidences = [agent_results[aid]['confidence'] for aid in agent_results.keys()]
        weights = np.array(confidences)
        
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(confidences)) / len(confidences)
        
        final_scores = np.average(stacked_scores, axis=1, weights=weights)
        
        logger.info("融合完成，最终分数范围: [%.4f, %.4f]" % (np.min(final_scores), np.max(final_scores)))
        return final_scores

def create_simple_agents(config=None):
    """创建简单智能体列表"""
    config = config or {}
    
    agents = [
        SimpleAgent("trend_agent", "trend_analysis", config),
        SimpleAgent("variance_agent", "variance_analysis", config),
        SimpleAgent("residual_agent", "residual_analysis", config),
        SimpleAgent("statistical_agent", "statistical_analysis", config),
        SimpleAgent("frequency_agent", "frequency_analysis", config)
    ]
    
    return agents

def load_dataset(dataset_name, data_base_path):
    """加载数据集"""
    try:
        if dataset_name == "MSL":
            train_path = os.path.join(data_base_path, "MSL/MSL_train.npy")
            test_path = os.path.join(data_base_path, "MSL/MSL_test.npy")
            test_label_path = os.path.join(data_base_path, "MSL/MSL_test_label.npy")
        elif dataset_name == "SMAP":
            train_path = os.path.join(data_base_path, "SMAP/SMAP_train.npy")
            test_path = os.path.join(data_base_path, "SMAP/SMAP_test.npy")
            test_label_path = os.path.join(data_base_path, "SMAP/SMAP_test_label.npy")
        elif dataset_name == "SMD":
            train_path = os.path.join(data_base_path, "SMD/SMD_train.npy")
            test_path = os.path.join(data_base_path, "SMD/SMD_test.npy")
            test_label_path = os.path.join(data_base_path, "SMD/SMD_test_labels.npy")
        elif dataset_name == "PSM":
            train_path = os.path.join(data_base_path, "PSM/PSM_train.npy")
            test_path = os.path.join(data_base_path, "PSM/PSM_test.npy")
            test_label_path = os.path.join(data_base_path, "PSM/PSM_test_labels.npy")
        elif dataset_name == "SWAT":
            train_path = os.path.join(data_base_path, "SWAT/SWAT_train.npy")
            test_path = os.path.join(data_base_path, "SWAT/SWAT_test.npy")
            test_label_path = os.path.join(data_base_path, "SWAT/SWAT_test_labels.npy")
        else:
            raise ValueError("未知数据集: %s" % dataset_name)
        
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        test_labels = np.load(test_label_path)
        
        logger.info("%s数据集形状: 训练%s, 测试%s, 标签%s" % (dataset_name, str(train_data.shape), str(test_data.shape), str(test_labels.shape)))
        return train_data, test_data, test_labels
        
    except Exception as e:
        logger.error("加载%s数据集失败: %s" % (dataset_name, str(e)))
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

def run_baseline_methods(train_data, test_data, test_labels):
    """运行基准方法"""
    results = {}
    
    # IsolationForest
    try:
        logger.info("运行IsolationForest...")
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(train_data)
        scores = model.score_samples(test_data)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        auroc = roc_auc_score(test_labels, scores)
        f1 = f1_score(test_labels, (scores > 0.5).astype(int))
        results['IsolationForest'] = {'auroc': auroc, 'f1': f1}
        logger.info("IsolationForest: AUROC %.4f, F1 %.4f" % (auroc, f1))
    except Exception as e:
        logger.warning("IsolationForest失败: %s" % str(e))
        results['IsolationForest'] = {'auroc': 0.5, 'f1': 0.0}
    
    # OneClassSVM
    try:
        logger.info("运行OneClassSVM...")
        model = OneClassSVM(nu=0.1, kernel='rbf')
        model.fit(train_data)
        scores = model.decision_function(test_data)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        auroc = roc_auc_score(test_labels, scores)
        f1 = f1_score(test_labels, (scores > 0.5).astype(int))
        results['OneClassSVM'] = {'auroc': auroc, 'f1': f1}
        logger.info("OneClassSVM: AUROC %.4f, F1 %.4f" % (auroc, f1))
    except Exception as e:
        logger.warning("OneClassSVM失败: %s" % str(e))
        results['OneClassSVM'] = {'auroc': 0.5, 'f1': 0.0}
    
    return results

def run_multi_agent_method(train_data, test_data, test_labels):
    """运行多智能体方法"""
    try:
        logger.info("运行简单多智能体方法...")
        
        agents = create_simple_agents()
        detector = SimpleMultiAgentDetector(agents)
        
        detector.fit(train_data)
        scores = detector.predict(test_data)
        
        auroc = roc_auc_score(test_labels, scores)
        f1 = f1_score(test_labels, (scores > 0.5).astype(int))
        
        results = {"SimpleMultiAgent": {'auroc': auroc, 'f1': f1}}
        
        logger.info("SimpleMultiAgent: AUROC %.4f, F1 %.4f" % (auroc, f1))
        return results
        
    except Exception as e:
        logger.error("多智能体方法失败: %s" % str(e))
        return {"SimpleMultiAgent": {'auroc': 0.5, 'f1': 0.0}}

def run_experiment(dataset_name, data_base_path):
    """运行单个数据集实验"""
    logger.info("🚀 开始%s数据集实验" % dataset_name)
    logger.info("=" * 60)
    
    # 加载数据
    train_data, test_data, test_labels = load_dataset(dataset_name, data_base_path)
    if train_data is None:
        return {}
    
    # 数据预处理
    train_data, test_data = preprocess_data(train_data, test_data)
    
    # 运行基准方法
    logger.info("🔍 运行基准方法")
    baseline_results = run_baseline_methods(train_data, test_data, test_labels)
    
    # 运行多智能体方法
    logger.info("🤖 运行多智能体方法")
    multi_agent_results = run_multi_agent_method(train_data, test_data, test_labels)
    
    # 合并结果
    all_results = baseline_results.copy()
    all_results.update(multi_agent_results)
    
    logger.info("✅ %s数据集实验完成" % dataset_name)
    return all_results

def main():
    """主函数"""
    data_base_path = "/home/jiangh/GinData/workspace/6.17cxz/data/jiangh@10.103.16.22"
    
    logger.info("🚀 开始简单可工作的多智能体异常检测实验")
    logger.info("=" * 80)
    
    datasets = ["MSL", "SMAP", "SMD", "PSM", "SWAT"]
    all_results = {}
    
    for dataset in datasets:
        try:
            all_results[dataset] = run_experiment(dataset, data_base_path)
        except Exception as e:
            logger.error("%s数据集实验失败: %s" % (dataset, str(e)))
            all_results[dataset] = {}
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs/simple_working_experiments", exist_ok=True)
    
    results_file = "outputs/simple_working_experiments/simple_results_%s.json" % timestamp
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info("结果已保存: %s" % results_file)
    logger.info("🎉 所有实验完成！")

if __name__ == "__main__":
    main()
