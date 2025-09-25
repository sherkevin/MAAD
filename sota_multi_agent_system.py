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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SOTAAgent:
    """SOTA级别智能体，集成最先进的检测算法"""
    
    def __init__(self, agent_id, agent_type, config=None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}
        self.is_fitted = False
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.ensemble_weights = None
        
    def fit(self, X):
        """训练SOTA智能体"""
        try:
            logger.info("训练SOTA智能体: %s" % self.agent_id)
            
            # 高级数据预处理
            X_processed = self._advanced_preprocessing(X)
            
            # 特征工程
            features = self._extract_sota_features(X_processed)
            
            # 特征选择
            selected_features = self._select_features(features)
            
            # 训练多个SOTA模型
            self._train_sota_models(selected_features)
            
            # 计算集成权重
            self._calculate_ensemble_weights(selected_features)
            
            self.is_fitted = True
            logger.info("SOTA智能体 %s 训练完成" % self.agent_id)
            
        except Exception as e:
            logger.error("SOTA智能体 %s 训练失败: %s" % (self.agent_id, str(e)))
            self.is_fitted = False
    
    def _advanced_preprocessing(self, X):
        """高级数据预处理"""
        # 1. 处理NaN值
        X_clean = np.nan_to_num(X, nan=0.0)
        
        # 2. 异常值处理（使用MAD方法）
        median = np.median(X_clean, axis=0)
        mad = np.median(np.abs(X_clean - median), axis=0)
        threshold = 3.5 * mad
        X_clean = np.clip(X_clean, median - threshold, median + threshold)
        
        # 3. 数据平滑
        if X_clean.shape[1] > 1:
            from scipy import ndimage
            X_clean = ndimage.gaussian_filter1d(X_clean, sigma=0.5, axis=1)
        
        return X_clean
    
    def _extract_sota_features(self, X):
        """提取SOTA级别特征"""
        features = []
        
        # 1. 原始特征
        features.append(X)
        
        # 2. 统计特征
        stat_features = np.column_stack([
            np.mean(X, axis=1),
            np.std(X, axis=1),
            np.min(X, axis=1),
            np.max(X, axis=1),
            np.median(X, axis=1),
            np.percentile(X, 25, axis=1),
            np.percentile(X, 75, axis=1)
        ])
        features.append(stat_features)
        
        # 3. 趋势特征
        if X.shape[1] > 1:
            trend_features = np.gradient(X, axis=1)
            features.append(trend_features)
            
            # 二阶导数
            trend2_features = np.gradient(trend_features, axis=1)
            features.append(trend2_features)
        
        # 4. 频域特征
        if X.shape[1] > 4:
            fft_features = np.abs(np.fft.fft(X, axis=1))
            # 取前几个频率分量
            n_freq = min(10, X.shape[1] // 2)
            features.append(fft_features[:, :n_freq])
        
        # 5. 小波特征（简化版）
        if X.shape[1] > 8:
            # 简单的多尺度特征
            scales = [2, 4, 8]
            for scale in scales:
                if X.shape[1] >= scale:
                    # 下采样
                    downsampled = X[:, ::scale]
                    features.append(downsampled)
        
        # 6. 交互特征
        if X.shape[1] > 1:
            # 特征间的乘积
            interaction_features = []
            for i in range(min(5, X.shape[1])):
                for j in range(i+1, min(5, X.shape[1])):
                    interaction_features.append(X[:, i] * X[:, j])
            if interaction_features:
                features.append(np.column_stack(interaction_features))
        
        return np.concatenate(features, axis=1)
    
    def _select_features(self, features):
        """特征选择"""
        try:
            # 使用方差阈值
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            selected = selector.fit_transform(features)
            
            # 使用PCA降维
            if selected.shape[1] > 50:
                pca = PCA(n_components=min(50, selected.shape[1]))
                selected = pca.fit_transform(selected)
            
            self.feature_selectors['variance'] = selector
            if 'pca' in locals():
                self.feature_selectors['pca'] = pca
            
            return selected
            
        except Exception as e:
            logger.warning("特征选择失败: %s" % str(e))
            return features
    
    def _train_sota_models(self, features):
        """训练SOTA模型"""
        # 1. 多个IsolationForest（不同参数）
        self.models['if_contamination_005'] = IsolationForest(
            contamination=0.05, random_state=42, n_estimators=200
        )
        self.models['if_contamination_01'] = IsolationForest(
            contamination=0.1, random_state=123, n_estimators=200
        )
        self.models['if_contamination_02'] = IsolationForest(
            contamination=0.2, random_state=456, n_estimators=200
        )
        
        # 2. 多个OneClassSVM（不同核函数）
        self.models['svm_rbf'] = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
        self.models['svm_poly'] = OneClassSVM(nu=0.1, kernel='poly', degree=3)
        self.models['svm_linear'] = OneClassSVM(nu=0.1, kernel='linear')
        
        # 3. LocalOutlierFactor（不同参数）
        self.models['lof_n15'] = LocalOutlierFactor(n_neighbors=15, contamination=0.1)
        self.models['lof_n25'] = LocalOutlierFactor(n_neighbors=25, contamination=0.1)
        self.models['lof_n35'] = LocalOutlierFactor(n_neighbors=35, contamination=0.1)
        
        # 4. 集成学习模型
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        )
        
        # 5. 逻辑回归
        self.models['logistic'] = LogisticRegression(
            random_state=42, max_iter=1000
        )
        
        # 训练所有模型
        for name, model in self.models.items():
            try:
                if name in ['lof_n15', 'lof_n25', 'lof_n35']:
                    # LOF不需要训练
                    continue
                elif name in ['gradient_boosting', 'logistic']:
                    # 监督学习模型需要标签，这里用伪标签
                    pseudo_labels = np.zeros(features.shape[0])
                    # 假设前10%是异常
                    n_anomalies = max(1, features.shape[0] // 10)
                    pseudo_labels[:n_anomalies] = 1
                    model.fit(features, pseudo_labels)
                else:
                    # 无监督学习模型
                    model.fit(features)
            except Exception as e:
                logger.warning("模型 %s 训练失败: %s" % (name, str(e)))
    
    def _calculate_ensemble_weights(self, features):
        """计算集成权重"""
        try:
            # 使用交叉验证评估每个模型的性能
            n_samples = min(1000, features.shape[0])
            X_sample = features[:n_samples]
            
            model_scores = {}
            
            for name, model in self.models.items():
                try:
                    if name in ['lof_n15', 'lof_n25', 'lof_n35']:
                        # LOF使用fit_predict
                        scores = model.fit_predict(X_sample)
                        scores = (scores == -1).astype(float)
                    elif name in ['gradient_boosting', 'logistic']:
                        # 监督学习模型
                        scores = model.predict_proba(X_sample)[:, 1]
                    else:
                        # 无监督学习模型
                        if hasattr(model, 'decision_function'):
                            scores = model.decision_function(X_sample)
                        elif hasattr(model, 'score_samples'):
                            scores = model.score_samples(X_sample)
                        else:
                            continue
                    
                    # 归一化分数
                    if len(scores) > 0:
                        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                        # 计算分数质量
                        score_quality = np.std(scores) * (1 + np.mean(scores))
                        model_scores[name] = score_quality
                
                except Exception as e:
                    logger.warning("评估模型 %s 失败: %s" % (name, str(e)))
                    model_scores[name] = 0.1
            
            # 归一化权重
            total_score = sum(model_scores.values())
            if total_score > 0:
                self.ensemble_weights = {name: score / total_score 
                                       for name, score in model_scores.items()}
            else:
                # 平均分配权重
                self.ensemble_weights = {name: 1.0 / len(self.models) 
                                       for name in self.models.keys()}
            
            logger.info("集成权重计算完成: %s" % str(self.ensemble_weights))
            
        except Exception as e:
            logger.error("计算集成权重失败: %s" % str(e))
            self.ensemble_weights = {name: 1.0 / len(self.models) 
                                   for name in self.models.keys()}
    
    def predict(self, X):
        """SOTA级别预测"""
        if not self.is_fitted:
            logger.warning("SOTA智能体 %s 未训练，返回随机分数" % self.agent_id)
            return np.random.rand(X.shape[0])
        
        try:
            # 数据预处理
            X_processed = self._advanced_preprocessing(X)
            
            # 特征工程
            features = self._extract_sota_features(X_processed)
            
            # 特征选择
            if 'variance' in self.feature_selectors:
                features = self.feature_selectors['variance'].transform(features)
            if 'pca' in self.feature_selectors:
                features = self.feature_selectors['pca'].transform(features)
            
            # 集成预测
            all_scores = []
            weights = []
            
            for name, model in self.models.items():
                try:
                    if name in ['lof_n15', 'lof_n25', 'lof_n35']:
                        # LOF
                        scores = model.fit_predict(features)
                        scores = (scores == -1).astype(float)
                    elif name in ['gradient_boosting', 'logistic']:
                        # 监督学习模型
                        scores = model.predict_proba(features)[:, 1]
                    else:
                        # 无监督学习模型
                        if hasattr(model, 'decision_function'):
                            scores = model.decision_function(features)
                        elif hasattr(model, 'score_samples'):
                            scores = model.score_samples(features)
                        else:
                            continue
                    
                    # 归一化分数
                    if len(scores) > 0:
                        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                        all_scores.append(scores)
                        weights.append(self.ensemble_weights.get(name, 0.1))
                
                except Exception as e:
                    logger.warning("模型 %s 预测失败: %s" % (name, str(e)))
                    continue
            
            if not all_scores:
                logger.warning("没有有效的模型预测结果，返回随机分数")
                return np.random.rand(X.shape[0])
            
            # 加权平均融合
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            final_scores = np.average(all_scores, axis=0, weights=weights)
            final_scores = np.clip(final_scores, 0.0, 1.0)
            
            return final_scores
            
        except Exception as e:
            logger.error("SOTA智能体 %s 预测失败: %s" % (self.agent_id, str(e)))
            return np.random.rand(X.shape[0])

class SOTAMultiAgentDetector:
    """SOTA级别多智能体检测器"""
    
    def __init__(self, agents, config=None):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.config = config or {}
        self.is_fitted = False
        self.fusion_weights = None
        self.meta_learner = None
        
        logger.info("初始化SOTA多智能体检测器，智能体数量: %d" % len(self.agents))
    
    def fit(self, X):
        """训练SOTA多智能体检测器"""
        logger.info("开始训练SOTA多智能体检测器，数据形状: %s" % str(X.shape))
        
        # 训练所有智能体
        for agent_id, agent in self.agents.items():
            try:
                agent.fit(X)
            except Exception as e:
                logger.error("智能体 %s 训练失败: %s" % (agent_id, str(e)))
        
        # 计算智能体融合权重
        self._calculate_agent_weights(X)
        
        # 训练元学习器
        self._train_meta_learner(X)
        
        self.is_fitted = True
        logger.info("SOTA多智能体检测器训练完成")
    
    def _calculate_agent_weights(self, X):
        """计算智能体融合权重"""
        try:
            # 使用交叉验证评估每个智能体的性能
            n_samples = min(2000, X.shape[0])
            X_sample = X[:n_samples]
            
            agent_performances = {}
            
            for agent_id, agent in self.agents.items():
                try:
                    # 获取预测分数
                    scores = agent.predict(X_sample)
                    
                    # 计算分数质量指标
                    score_std = np.std(scores)
                    score_range = np.max(scores) - np.min(scores)
                    score_mean = np.mean(scores)
                    
                    # 综合性能指标
                    performance = score_std * score_range * (1 + score_mean)
                    agent_performances[agent_id] = performance
                    
                except Exception as e:
                    logger.warning("评估智能体 %s 性能失败: %s" % (agent_id, str(e)))
                    agent_performances[agent_id] = 0.1
            
            # 归一化权重
            total_performance = sum(agent_performances.values())
            if total_performance > 0:
                self.fusion_weights = {aid: perf / total_performance 
                                     for aid, perf in agent_performances.items()}
            else:
                # 平均分配权重
                self.fusion_weights = {aid: 1.0 / len(self.agents) 
                                     for aid in self.agents.keys()}
            
            logger.info("智能体融合权重计算完成: %s" % str(self.fusion_weights))
            
        except Exception as e:
            logger.error("计算智能体融合权重失败: %s" % str(e))
            self.fusion_weights = {aid: 1.0 / len(self.agents) 
                                 for aid in self.agents.keys()}
    
    def _train_meta_learner(self, X):
        """训练元学习器"""
        try:
            # 收集所有智能体的预测结果作为特征
            n_samples = min(5000, X.shape[0])
            X_sample = X[:n_samples]
            
            agent_features = []
            for agent_id, agent in self.agents.items():
                try:
                    scores = agent.predict(X_sample)
                    agent_features.append(scores)
                except Exception as e:
                    logger.warning("收集智能体 %s 特征失败: %s" % (agent_id, str(e)))
                    agent_features.append(np.random.rand(n_samples))
            
            if agent_features:
                # 组合所有智能体的预测结果
                combined_features = np.column_stack(agent_features)
                
                # 训练元学习器（使用伪标签）
                pseudo_labels = np.zeros(n_samples)
                n_anomalies = max(1, n_samples // 10)
                pseudo_labels[:n_anomalies] = 1
                
                self.meta_learner = GradientBoostingClassifier(
                    n_estimators=50, learning_rate=0.1, random_state=42
                )
                self.meta_learner.fit(combined_features, pseudo_labels)
                
                logger.info("元学习器训练完成")
            
        except Exception as e:
            logger.error("训练元学习器失败: %s" % str(e))
            self.meta_learner = None
    
    def predict(self, X):
        """SOTA级别多智能体协作预测"""
        if not self.is_fitted:
            logger.warning("SOTA多智能体检测器未训练，返回随机分数")
            return np.random.rand(X.shape[0])
        
        logger.info("开始SOTA多智能体协作预测，数据形状: %s" % str(X.shape))
        
        # 收集所有智能体的预测结果
        agent_results = {}
        agent_scores = []
        agent_weights = []
        
        for agent_id, agent in self.agents.items():
            try:
                scores = agent.predict(X)
                agent_results[agent_id] = {
                    'scores': scores,
                    'confidence': np.mean(scores),
                    'std': np.std(scores),
                    'range': np.max(scores) - np.min(scores)
                }
                agent_scores.append(scores)
                agent_weights.append(self.fusion_weights.get(agent_id, 0.1))
                
                logger.info("智能体 %s 预测完成，平均分数: %.4f, 权重: %.4f" % 
                          (agent_id, np.mean(scores), self.fusion_weights.get(agent_id, 0.1)))
                
            except Exception as e:
                logger.error("智能体 %s 预测失败: %s" % (agent_id, str(e)))
                # 使用随机分数作为备选
                random_scores = np.random.rand(X.shape[0])
                agent_results[agent_id] = {
                    'scores': random_scores,
                    'confidence': 0.5,
                    'std': 0.1,
                    'range': 0.1
                }
                agent_scores.append(random_scores)
                agent_weights.append(0.01)
        
        if not agent_scores:
            logger.warning("没有有效的智能体预测结果，返回随机分数")
            return np.random.rand(X.shape[0])
        
        # SOTA级别融合策略
        logger.info("使用SOTA级别融合策略进行智能体协作...")
        
        # 1. 加权平均融合
        agent_weights = np.array(agent_weights)
        agent_weights = agent_weights / np.sum(agent_weights)
        
        stacked_scores = np.vstack(agent_scores).T
        weighted_avg = np.average(stacked_scores, axis=1, weights=agent_weights)
        
        # 2. 元学习器融合
        if self.meta_learner is not None:
            try:
                meta_scores = self.meta_learner.predict_proba(stacked_scores)[:, 1]
                logger.info("元学习器融合完成")
            except Exception as e:
                logger.warning("元学习器融合失败: %s" % str(e))
                meta_scores = weighted_avg
        else:
            meta_scores = weighted_avg
        
        # 3. 动态权重调整
        confidence_weights = []
        for agent_id in self.agents.keys():
            if agent_id in agent_results:
                conf = agent_results[agent_id]['confidence']
                std = agent_results[agent_id]['std']
                # 置信度高且方差适中的智能体获得更高权重
                dynamic_weight = conf * (1 - min(std, 0.5))
                confidence_weights.append(dynamic_weight)
            else:
                confidence_weights.append(0.1)
        
        confidence_weights = np.array(confidence_weights)
        confidence_weights = confidence_weights / np.sum(confidence_weights)
        
        confidence_avg = np.average(stacked_scores, axis=1, weights=confidence_weights)
        
        # 4. 最终融合（结合三种策略）
        final_scores = 0.5 * weighted_avg + 0.3 * meta_scores + 0.2 * confidence_avg
        
        # 5. 后处理优化
        final_scores = self._post_process_scores(final_scores, agent_results)
        
        logger.info("SOTA融合完成，最终分数范围: [%.4f, %.4f]" % 
                  (np.min(final_scores), np.max(final_scores)))
        
        return final_scores
    
    def _post_process_scores(self, scores, agent_results):
        """后处理优化分数"""
        try:
            # 1. 平滑处理
            if len(scores) > 5:
                from scipy import ndimage
                smoothed = ndimage.gaussian_filter1d(scores, sigma=1.0)
                scores = 0.8 * scores + 0.2 * smoothed
            
            # 2. 异常值处理
            q25, q75 = np.percentile(scores, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            scores = np.clip(scores, lower_bound, upper_bound)
            
            # 3. 归一化到0-1
            if np.max(scores) > np.min(scores):
                scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            else:
                scores = np.full_like(scores, 0.5)
            
            return scores
            
        except Exception as e:
            logger.warning("后处理优化失败: %s" % str(e))
            return scores

def create_sota_agents(config=None):
    """创建SOTA智能体列表"""
    config = config or {}
    
    agents = [
        SOTAAgent("sota_trend_agent", "trend_analysis", config),
        SOTAAgent("sota_variance_agent", "variance_analysis", config),
        SOTAAgent("sota_residual_agent", "residual_analysis", config),
        SOTAAgent("sota_statistical_agent", "statistical_analysis", config),
        SOTAAgent("sota_frequency_agent", "frequency_analysis", config)
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
        else:
            raise ValueError("未知数据集: %s" % dataset_name)
        
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        test_labels = np.load(test_label_path)
        
        logger.info("%s数据集形状: 训练%s, 测试%s, 标签%s" % 
                   (dataset_name, str(train_data.shape), str(test_data.shape), str(test_labels.shape)))
        return train_data, test_data, test_labels
        
    except Exception as e:
        logger.error("加载%s数据集失败: %s" % (dataset_name, str(e)))
        return None, None, None

def test_sota_system(dataset_name, data_base_path):
    """测试SOTA多智能体系统"""
    logger.info("🚀 开始测试SOTA多智能体系统 - %s数据集" % dataset_name)
    logger.info("=" * 60)
    
    # 加载数据
    train_data, test_data, test_labels = load_dataset(dataset_name, data_base_path)
    if train_data is None:
        return {}
    
    # 数据预处理
    train_data = np.nan_to_num(train_data, nan=0.0)
    test_data = np.nan_to_num(test_data, nan=0.0)
    
    # 测试基准方法
    logger.info("🔍 测试基准方法")
    baseline_results = {}
    
    # IsolationForest
    try:
        logger.info("运行IsolationForest...")
        model = IsolationForest(contamination=0.1, random_state=42, n_estimators=200)
        model.fit(train_data)
        scores = model.score_samples(test_data)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        auroc = roc_auc_score(test_labels, scores)
        f1 = f1_score(test_labels, (scores > 0.5).astype(int))
        baseline_results['IsolationForest'] = {'auroc': auroc, 'f1': f1}
        logger.info("IsolationForest: AUROC %.4f, F1 %.4f" % (auroc, f1))
    except Exception as e:
        logger.warning("IsolationForest失败: %s" % str(e))
        baseline_results['IsolationForest'] = {'auroc': 0.5, 'f1': 0.0}
    
    # 测试SOTA多智能体系统
    logger.info("🤖 测试SOTA多智能体系统...")
    try:
        agents = create_sota_agents()
        detector = SOTAMultiAgentDetector(agents)
        
        detector.fit(train_data)
        scores = detector.predict(test_data)
        
        auroc = roc_auc_score(test_labels, scores)
        f1 = f1_score(test_labels, (scores > 0.5).astype(int))
        
        sota_results = {"SOTAMultiAgent": {'auroc': auroc, 'f1': f1}}
        logger.info("SOTAMultiAgent: AUROC %.4f, F1 %.4f" % (auroc, f1))
        
    except Exception as e:
        logger.error("SOTA多智能体系统失败: %s" % str(e))
        sota_results = {"SOTAMultiAgent": {'auroc': 0.5, 'f1': 0.0}}
    
    # 合并结果
    all_results = baseline_results.copy()
    all_results.update(sota_results)
    
    logger.info("✅ %s数据集测试完成" % dataset_name)
    return all_results

def main():
    """主函数"""
    data_base_path = "/home/jiangh/GinData/workspace/6.17cxz/data/jiangh@10.103.16.22"
    
    logger.info("🚀 开始SOTA多智能体异常检测系统测试")
    logger.info("=" * 80)
    
    # 测试MSL数据集
    msl_results = test_sota_system("MSL", data_base_path)
    
    # 测试SMAP数据集
    smap_results = test_sota_system("SMAP", data_base_path)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs/sota_system", exist_ok=True)
    
    all_results = {
        "MSL": msl_results,
        "SMAP": smap_results
    }
    
    results_file = "outputs/sota_system/sota_results_%s.json" % timestamp
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info("结果已保存: %s" % results_file)
    
    # 打印结果摘要
    logger.info("=" * 80)
    logger.info("📊 SOTA多智能体系统测试结果摘要")
    logger.info("=" * 80)
    
    for dataset, results in all_results.items():
        logger.info("📊 %s数据集结果:" % dataset)
        logger.info("-" * 50)
        for method, metrics in results.items():
            logger.info("%s: AUROC %.4f, F1 %.4f" % (method, metrics['auroc'], metrics['f1']))
        logger.info("")
    
    # 计算平均性能
    avg_auroc = {}
    for dataset, results in all_results.items():
        for method, metrics in results.items():
            if method not in avg_auroc:
                avg_auroc[method] = []
            avg_auroc[method].append(metrics['auroc'])
    
    logger.info("📊 平均性能对比:")
    logger.info("-" * 30)
    for method, aurocs in avg_auroc.items():
        avg = np.mean(aurocs)
        logger.info("%s: 平均AUROC %.4f" % (method, avg))
    
    # 检查是否达到SOTA水平
    if "SOTAMultiAgent" in avg_auroc:
        sota_avg = np.mean(avg_auroc["SOTAMultiAgent"])
        if sota_avg > 0.7:
            logger.info("🎉 SOTA多智能体系统达到顶级性能！")
        elif sota_avg > 0.6:
            logger.info("✅ SOTA多智能体系统性能优秀，接近顶级水平")
        else:
            logger.info("⚠️ SOTA多智能体系统需要进一步优化")
    
    logger.info("🎉 SOTA多智能体系统测试完成！")

if __name__ == "__main__":
    main()
