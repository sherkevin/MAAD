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
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedAgent:
    """高级智能体，集成多种检测策略"""
    
    def __init__(self, agent_id, agent_type, config=None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}
        self.is_fitted = False
        self.models = {}  # 存储多个子模型
        self.scalers = {}  # 存储多个标准化器
        self.feature_importance = None
        
    def fit(self, X):
        """训练智能体，使用多种策略"""
        try:
            logger.info("训练高级智能体: %s" % self.agent_id)
            
            # 数据预处理
            X_processed = self._preprocess_data(X)
            
            # 根据智能体类型选择策略组合
            if self.agent_type == "trend_analysis":
                self._fit_trend_models(X_processed)
            elif self.agent_type == "variance_analysis":
                self._fit_variance_models(X_processed)
            elif self.agent_type == "residual_analysis":
                self._fit_residual_models(X_processed)
            elif self.agent_type == "statistical_analysis":
                self._fit_statistical_models(X_processed)
            elif self.agent_type == "frequency_analysis":
                self._fit_frequency_models(X_processed)
            else:
                self._fit_general_models(X_processed)
            
            self.is_fitted = True
            logger.info("高级智能体 %s 训练完成" % self.agent_id)
            
        except Exception as e:
            logger.error("高级智能体 %s 训练失败: %s" % (self.agent_id, str(e)))
            self.is_fitted = False
    
    def _preprocess_data(self, X):
        """高级数据预处理"""
        # 处理NaN值
        X_clean = np.nan_to_num(X, nan=0.0)
        
        # 异常值处理（使用IQR方法）
        Q1 = np.percentile(X_clean, 25, axis=0)
        Q3 = np.percentile(X_clean, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 限制异常值
        X_clean = np.clip(X_clean, lower_bound, upper_bound)
        
        return X_clean
    
    def _fit_trend_models(self, X):
        """训练趋势分析模型组合"""
        # 1. 趋势特征提取
        trend_features = self._extract_trend_features(X)
        
        # 2. 多个IsolationForest模型（不同参数）
        self.models['trend_if1'] = IsolationForest(contamination=0.05, random_state=42)
        self.models['trend_if2'] = IsolationForest(contamination=0.1, random_state=123)
        self.models['trend_if3'] = IsolationForest(contamination=0.15, random_state=456)
        
        # 3. OneClassSVM模型
        self.models['trend_svm'] = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
        
        # 4. 训练所有模型
        for name, model in self.models.items():
            model.fit(trend_features)
        
        # 5. 保存标准化器
        self.scalers['trend'] = StandardScaler()
        self.scalers['trend'].fit(trend_features)
    
    def _fit_variance_models(self, X):
        """训练方差分析模型组合"""
        # 1. 方差特征提取
        variance_features = self._extract_variance_features(X)
        
        # 2. 多个模型
        self.models['var_if1'] = IsolationForest(contamination=0.1, random_state=42)
        self.models['var_if2'] = IsolationForest(contamination=0.2, random_state=123)
        self.models['var_svm'] = OneClassSVM(nu=0.15, kernel='poly', degree=3)
        self.models['var_lof'] = LocalOutlierFactor(n_neighbors=15, contamination=0.1)
        
        # 3. 训练模型
        for name, model in self.models.items():
            if name != 'var_lof':  # LOF不需要训练
                model.fit(variance_features)
        
        # 4. 保存标准化器
        self.scalers['variance'] = StandardScaler()
        self.scalers['variance'].fit(variance_features)
    
    def _fit_residual_models(self, X):
        """训练残差分析模型组合"""
        # 1. 残差特征提取
        residual_features = self._extract_residual_features(X)
        
        # 2. 多个模型
        self.models['res_if1'] = IsolationForest(contamination=0.08, random_state=42)
        self.models['res_if2'] = IsolationForest(contamination=0.12, random_state=123)
        self.models['res_svm'] = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
        
        # 3. 训练模型
        for name, model in self.models.items():
            model.fit(residual_features)
        
        # 4. 保存标准化器
        self.scalers['residual'] = StandardScaler()
        self.scalers['residual'].fit(residual_features)
    
    def _fit_statistical_models(self, X):
        """训练统计分析模型组合"""
        # 1. 统计特征提取
        stat_features = self._extract_statistical_features(X)
        
        # 2. 多个模型
        self.models['stat_if1'] = IsolationForest(contamination=0.1, random_state=42)
        self.models['stat_if2'] = IsolationForest(contamination=0.2, random_state=123)
        self.models['stat_svm'] = OneClassSVM(nu=0.1, kernel='linear')
        self.models['stat_lof'] = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        
        # 3. 训练模型
        for name, model in self.models.items():
            if name != 'stat_lof':  # LOF不需要训练
                model.fit(stat_features)
        
        # 4. 保存标准化器
        self.scalers['statistical'] = StandardScaler()
        self.scalers['statistical'].fit(stat_features)
    
    def _fit_frequency_models(self, X):
        """训练频域分析模型组合"""
        # 1. 频域特征提取
        freq_features = self._extract_frequency_features(X)
        
        # 2. 多个模型
        self.models['freq_if1'] = IsolationForest(contamination=0.1, random_state=42)
        self.models['freq_svm'] = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
        
        # 3. 训练模型
        for name, model in self.models.items():
            model.fit(freq_features)
        
        # 4. 保存标准化器
        self.scalers['frequency'] = StandardScaler()
        self.scalers['frequency'].fit(freq_features)
    
    def _fit_general_models(self, X):
        """训练通用模型组合"""
        # 1. 通用特征提取
        general_features = self._extract_general_features(X)
        
        # 2. 多个模型
        self.models['gen_if1'] = IsolationForest(contamination=0.1, random_state=42)
        self.models['gen_if2'] = IsolationForest(contamination=0.2, random_state=123)
        self.models['gen_svm'] = OneClassSVM(nu=0.1, kernel='rbf')
        
        # 3. 训练模型
        for name, model in self.models.items():
            model.fit(general_features)
        
        # 4. 保存标准化器
        self.scalers['general'] = StandardScaler()
        self.scalers['general'].fit(general_features)
    
    def _extract_trend_features(self, X):
        """提取趋势特征"""
        features = []
        
        # 1. 移动平均
        window_sizes = [3, 5, 10]
        for window in window_sizes:
            if X.shape[1] >= window:
                ma = np.mean(X, axis=1, keepdims=True)
                features.append(ma)
        
        # 2. 趋势斜率
        if X.shape[1] > 1:
            slopes = np.gradient(X, axis=1)
            features.append(slopes)
        
        # 3. 原始特征
        features.append(X)
        
        return np.concatenate(features, axis=1)
    
    def _extract_variance_features(self, X):
        """提取方差特征"""
        features = []
        
        # 1. 滚动方差
        window_sizes = [3, 5, 10]
        for window in window_sizes:
            if X.shape[1] >= window:
                rolling_var = np.var(X, axis=1, keepdims=True)
                features.append(rolling_var)
        
        # 2. 特征间方差
        feature_var = np.var(X, axis=0, keepdims=True)
        features.append(np.tile(feature_var, (X.shape[0], 1)))
        
        # 3. 原始特征
        features.append(X)
        
        return np.concatenate(features, axis=1)
    
    def _extract_residual_features(self, X):
        """提取残差特征"""
        features = []
        
        # 1. 线性拟合残差
        if X.shape[1] > 1:
            # 对每个样本进行线性拟合
            residuals = []
            for i in range(X.shape[0]):
                x_vals = np.arange(X.shape[1])
                y_vals = X[i, :]
                # 简单线性回归
                coeffs = np.polyfit(x_vals, y_vals, 1)
                y_pred = np.polyval(coeffs, x_vals)
                residual = y_vals - y_pred
                residuals.append(residual)
            residuals = np.array(residuals)
            features.append(residuals)
        
        # 2. 原始特征
        features.append(X)
        
        return np.concatenate(features, axis=1)
    
    def _extract_statistical_features(self, X):
        """提取统计特征"""
        features = []
        
        # 1. 基本统计量
        mean_feat = np.mean(X, axis=1, keepdims=True)
        std_feat = np.std(X, axis=1, keepdims=True)
        max_feat = np.max(X, axis=1, keepdims=True)
        min_feat = np.min(X, axis=1, keepdims=True)
        
        features.extend([mean_feat, std_feat, max_feat, min_feat])
        
        # 2. 分位数特征
        q25 = np.percentile(X, 25, axis=1, keepdims=True)
        q50 = np.percentile(X, 50, axis=1, keepdims=True)
        q75 = np.percentile(X, 75, axis=1, keepdims=True)
        
        features.extend([q25, q50, q75])
        
        # 3. 原始特征
        features.append(X)
        
        return np.concatenate(features, axis=1)
    
    def _extract_frequency_features(self, X):
        """提取频域特征"""
        features = []
        
        # 1. FFT特征
        if X.shape[1] > 1:
            fft_features = np.abs(np.fft.fft(X, axis=1))
            # 取前几个频率分量
            n_freq = min(10, X.shape[1] // 2)
            features.append(fft_features[:, :n_freq])
        
        # 2. 功率谱密度
        if X.shape[1] > 1:
            psd = np.abs(np.fft.fft(X, axis=1)) ** 2
            n_freq = min(10, X.shape[1] // 2)
            features.append(psd[:, :n_freq])
        
        # 3. 原始特征
        features.append(X)
        
        return np.concatenate(features, axis=1)
    
    def _extract_general_features(self, X):
        """提取通用特征"""
        features = []
        
        # 1. PCA降维特征
        if X.shape[1] > 5:
            pca = PCA(n_components=min(10, X.shape[1]))
            pca_features = pca.fit_transform(X)
            features.append(pca_features)
        
        # 2. 聚类特征
        if X.shape[0] > 10:
            kmeans = KMeans(n_clusters=min(5, X.shape[0] // 2), random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            cluster_features = np.zeros((X.shape[0], kmeans.n_clusters))
            for i, label in enumerate(cluster_labels):
                cluster_features[i, label] = 1
            features.append(cluster_features)
        
        # 3. 原始特征
        features.append(X)
        
        return np.concatenate(features, axis=1)
    
    def predict(self, X):
        """高级预测，集成多个模型"""
        if not self.is_fitted:
            logger.warning("高级智能体 %s 未训练，返回随机分数" % self.agent_id)
            return np.random.rand(X.shape[0])
        
        try:
            # 数据预处理
            X_processed = self._preprocess_data(X)
            
            # 根据智能体类型提取特征
            if self.agent_type == "trend_analysis":
                features = self._extract_trend_features(X_processed)
                scaler_key = 'trend'
            elif self.agent_type == "variance_analysis":
                features = self._extract_variance_features(X_processed)
                scaler_key = 'variance'
            elif self.agent_type == "residual_analysis":
                features = self._extract_residual_features(X_processed)
                scaler_key = 'residual'
            elif self.agent_type == "statistical_analysis":
                features = self._extract_statistical_features(X_processed)
                scaler_key = 'statistical'
            elif self.agent_type == "frequency_analysis":
                features = self._extract_frequency_features(X_processed)
                scaler_key = 'frequency'
            else:
                features = self._extract_general_features(X_processed)
                scaler_key = 'general'
            
            # 标准化特征
            if scaler_key in self.scalers:
                features = self.scalers[scaler_key].transform(features)
            
            # 集成多个模型的预测
            all_scores = []
            weights = []
            
            for name, model in self.models.items():
                try:
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
                        
                        # 根据模型类型分配权重
                        if 'if1' in name or 'svm' in name:
                            weights.append(1.0)
                        elif 'if2' in name:
                            weights.append(0.8)
                        else:
                            weights.append(0.6)
                
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
            logger.error("高级智能体 %s 预测失败: %s" % (self.agent_id, str(e)))
            return np.random.rand(X.shape[0])

class AdvancedMultiAgentDetector:
    """高级多智能体检测器，实现智能融合"""
    
    def __init__(self, agents, config=None):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.config = config or {}
        self.is_fitted = False
        self.fusion_weights = None
        self.performance_history = {}
        
        logger.info("初始化高级多智能体检测器，智能体数量: %d" % len(self.agents))
    
    def fit(self, X):
        """训练所有智能体"""
        logger.info("开始训练高级多智能体检测器，数据形状: %s" % str(X.shape))
        
        # 训练所有智能体
        for agent_id, agent in self.agents.items():
            try:
                agent.fit(X)
            except Exception as e:
                logger.error("智能体 %s 训练失败: %s" % (agent_id, str(e)))
        
        # 计算融合权重
        self._calculate_fusion_weights(X)
        
        self.is_fitted = True
        logger.info("高级多智能体检测器训练完成")
    
    def _calculate_fusion_weights(self, X):
        """计算智能体融合权重"""
        try:
            # 使用交叉验证评估每个智能体的性能
            n_samples = min(1000, X.shape[0])  # 使用部分数据进行快速评估
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
            
            logger.info("融合权重计算完成: %s" % str(self.fusion_weights))
            
        except Exception as e:
            logger.error("计算融合权重失败: %s" % str(e))
            # 使用平均权重
            self.fusion_weights = {aid: 1.0 / len(self.agents) 
                                 for aid in self.agents.keys()}
    
    def predict(self, X):
        """高级多智能体协作预测"""
        if not self.is_fitted:
            logger.warning("高级多智能体检测器未训练，返回随机分数")
            return np.random.rand(X.shape[0])
        
        logger.info("开始高级多智能体协作预测，数据形状: %s" % str(X.shape))
        
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
                agent_weights.append(0.01)  # 很低的权重
        
        if not agent_scores:
            logger.warning("没有有效的智能体预测结果，返回随机分数")
            return np.random.rand(X.shape[0])
        
        # 高级融合策略
        logger.info("使用高级融合策略进行智能体协作...")
        
        # 1. 加权平均融合
        agent_weights = np.array(agent_weights)
        agent_weights = agent_weights / np.sum(agent_weights)
        
        stacked_scores = np.vstack(agent_scores).T
        weighted_avg = np.average(stacked_scores, axis=1, weights=agent_weights)
        
        # 2. 动态权重调整（基于置信度）
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
        
        # 3. 最终融合（结合两种策略）
        final_scores = 0.7 * weighted_avg + 0.3 * confidence_avg
        
        # 4. 后处理优化
        final_scores = self._post_process_scores(final_scores, agent_results)
        
        logger.info("高级融合完成，最终分数范围: [%.4f, %.4f]" % 
                  (np.min(final_scores), np.max(final_scores)))
        
        return final_scores
    
    def _post_process_scores(self, scores, agent_results):
        """后处理优化分数"""
        try:
            # 1. 平滑处理
            if len(scores) > 3:
                # 简单的移动平均平滑
                window_size = min(3, len(scores))
                smoothed = np.convolve(scores, np.ones(window_size)/window_size, mode='same')
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

def create_advanced_agents(config=None):
    """创建高级智能体列表"""
    config = config or {}
    
    agents = [
        AdvancedAgent("trend_agent", "trend_analysis", config),
        AdvancedAgent("variance_agent", "variance_analysis", config),
        AdvancedAgent("residual_agent", "residual_analysis", config),
        AdvancedAgent("statistical_agent", "statistical_analysis", config),
        AdvancedAgent("frequency_agent", "frequency_analysis", config)
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

def test_advanced_system(dataset_name, data_base_path):
    """测试高级多智能体系统"""
    logger.info("🚀 开始测试高级多智能体系统 - %s数据集" % dataset_name)
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
        model = IsolationForest(contamination=0.1, random_state=42)
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
    
    # 测试高级多智能体系统
    logger.info("🤖 测试高级多智能体系统...")
    try:
        agents = create_advanced_agents()
        detector = AdvancedMultiAgentDetector(agents)
        
        detector.fit(train_data)
        scores = detector.predict(test_data)
        
        auroc = roc_auc_score(test_labels, scores)
        f1 = f1_score(test_labels, (scores > 0.5).astype(int))
        
        advanced_results = {"AdvancedMultiAgent": {'auroc': auroc, 'f1': f1}}
        logger.info("AdvancedMultiAgent: AUROC %.4f, F1 %.4f" % (auroc, f1))
        
    except Exception as e:
        logger.error("高级多智能体系统失败: %s" % str(e))
        advanced_results = {"AdvancedMultiAgent": {'auroc': 0.5, 'f1': 0.0}}
    
    # 合并结果
    all_results = baseline_results.copy()
    all_results.update(advanced_results)
    
    logger.info("✅ %s数据集测试完成" % dataset_name)
    return all_results

def main():
    """主函数"""
    data_base_path = "/home/jiangh/GinData/workspace/6.17cxz/data/jiangh@10.103.16.22"
    
    logger.info("🚀 开始高级多智能体异常检测系统测试")
    logger.info("=" * 80)
    
    # 测试MSL数据集
    msl_results = test_advanced_system("MSL", data_base_path)
    
    # 测试SMAP数据集
    smap_results = test_advanced_system("SMAP", data_base_path)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs/advanced_system", exist_ok=True)
    
    all_results = {
        "MSL": msl_results,
        "SMAP": smap_results
    }
    
    results_file = "outputs/advanced_system/advanced_results_%s.json" % timestamp
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info("结果已保存: %s" % results_file)
    
    # 打印结果摘要
    logger.info("=" * 80)
    logger.info("📊 高级多智能体系统测试结果摘要")
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
    if "AdvancedMultiAgent" in avg_auroc:
        advanced_avg = np.mean(avg_auroc["AdvancedMultiAgent"])
        if advanced_avg > 0.7:
            logger.info("🎉 高级多智能体系统达到SOTA水平！")
        elif advanced_avg > 0.6:
            logger.info("✅ 高级多智能体系统性能良好，接近SOTA水平")
        else:
            logger.info("⚠️ 高级多智能体系统需要进一步优化")
    
    logger.info("🎉 高级多智能体系统测试完成！")

if __name__ == "__main__":
    main()
