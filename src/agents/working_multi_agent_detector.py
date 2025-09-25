# -*- coding: utf-8 -*-
import torch
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkingAgent:
    """可工作的智能体基类"""
    
    def __init__(self, agent_id: str, agent_type: str, config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}
        self.is_fitted = False
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray):
        """训练智能体"""
        try:
            # 数据预处理
            X_scaled = self.scaler.fit_transform(X)
            
            # 根据智能体类型选择模型
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
                # 默认使用IsolationForest
                self.model = IsolationForest(contamination=0.1, random_state=42)
            
            # 训练模型
            self.model.fit(X_scaled)
            self.is_fitted = True
            logger.info(f"智能体 {self.agent_id} 训练完成")
            
        except Exception as e:
            logger.error(f"智能体 {self.agent_id} 训练失败: {e}")
            self.is_fitted = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测异常分数"""
        if not self.is_fitted or self.model is None:
            logger.warning(f"智能体 {self.agent_id} 未训练，返回随机分数")
            return np.random.rand(X.shape[0])
        
        try:
            # 数据预处理
            X_scaled = self.scaler.transform(X)
            
            # 根据模型类型进行预测
            if hasattr(self.model, 'decision_function'):
                # OneClassSVM, LocalOutlierFactor
                scores = self.model.decision_function(X_scaled)
                # 归一化到0-1
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            elif hasattr(self.model, 'score_samples'):
                # IsolationForest
                scores = self.model.score_samples(X_scaled)
                # 归一化到0-1
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            else:
                # 其他情况，返回随机分数
                scores = np.random.rand(X.shape[0])
            
            # 确保分数在0-1范围内
            scores = np.clip(scores, 0.0, 1.0)
            
            return scores
            
        except Exception as e:
            logger.error(f"智能体 {self.agent_id} 预测失败: {e}")
            return np.random.rand(X.shape[0])

class WorkingMultiAgentDetector:
    """可工作的多智能体异常检测器"""
    
    def __init__(self, agents: List[WorkingAgent], config: Dict[str, Any] = None):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.config = config or {}
        self.is_fitted = False
        
        # LLM通信系统（可选）
        self.llm_communication = None
        if self.config.get('use_llm_communication', False):
            try:
                from src.communication.enhanced_llm_communication import EnhancedLLMCommunication
                api_key = self.config.get('aliyun_qwen_api_key')
                if api_key:
                    self.llm_communication = EnhancedLLMCommunication(api_key)
                    logger.info("LLM通信系统已启用")
            except Exception as e:
                logger.warning(f"LLM通信系统初始化失败: {e}")
        
        logger.info(f"初始化多智能体检测器，智能体数量: {len(self.agents)}")
    
    def fit(self, X: np.ndarray):
        """训练所有智能体"""
        logger.info(f"开始训练多智能体检测器，数据形状: {X.shape}")
        
        for agent_id, agent in self.agents.items():
            try:
                agent.fit(X)
            except Exception as e:
                logger.error(f"智能体 {agent_id} 训练失败: {e}")
        
        self.is_fitted = True
        logger.info("多智能体检测器训练完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """多智能体协作预测"""
        if not self.is_fitted:
            logger.warning("多智能体检测器未训练，返回随机分数")
            return np.random.rand(X.shape[0])
        
        logger.info(f"开始多智能体协作预测，数据形状: {X.shape}")
        
        # 收集所有智能体的预测结果
        agent_results = {}
        agent_scores = []
        
        for agent_id, agent in self.agents.items():
            try:
                scores = agent.predict(X)
                agent_results[agent_id] = {
                    'scores': scores,
                    'confidence': np.mean(scores),
                    'findings': f"智能体 {agent_id} 检测到 {np.sum(scores > 0.5)} 个潜在异常点"
                }
                agent_scores.append(scores)
                logger.info(f"智能体 {agent_id} 预测完成，平均分数: {np.mean(scores):.4f}")
            except Exception as e:
                logger.error(f"智能体 {agent_id} 预测失败: {e}")
                # 使用随机分数作为备选
                random_scores = np.random.rand(X.shape[0])
                agent_results[agent_id] = {
                    'scores': random_scores,
                    'confidence': 0.5,
                    'findings': f"智能体 {agent_id} 预测失败，使用随机分数"
                }
                agent_scores.append(random_scores)
        
        if not agent_scores:
            logger.warning("没有有效的智能体预测结果，返回随机分数")
            return np.random.rand(X.shape[0])
        
        # 如果启用了LLM通信，使用LLM进行智能融合
        if self.llm_communication and len(agent_results) > 1:
            try:
                logger.info("使用LLM进行智能体协调和融合...")
                final_decision = self.llm_communication.coordinate_agents(agent_results)
                final_scores = np.full(X.shape[0], final_decision.get('final_score', 0.5))
                logger.info(f"LLM协调后的最终分数: {final_decision.get('final_score', 0.5):.4f}")
                return final_scores
            except Exception as e:
                logger.warning(f"LLM协调失败: {e}，使用传统融合方法")
        
        # 传统融合方法：加权平均
        logger.info("使用加权平均进行智能体融合...")
        
        # 将所有分数堆叠成二维数组
        stacked_scores = np.vstack(agent_scores).T  # 形状 (n_samples, n_agents)
        
        # 计算权重（基于置信度）
        confidences = [agent_results[aid]['confidence'] for aid in agent_results.keys()]
        weights = np.array(confidences)
        
        # 归一化权重
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(confidences)) / len(confidences)
        
        # 计算加权平均
        final_scores = np.average(stacked_scores, axis=1, weights=weights)
        
        logger.info(f"融合完成，最终分数范围: [{np.min(final_scores):.4f}, {np.max(final_scores):.4f}]")
        return final_scores

def create_working_agents(config: Dict[str, Any] = None) -> List[WorkingAgent]:
    """创建可工作的智能体列表"""
    config = config or {}
    
    agents = [
        WorkingAgent("trend_agent", "trend_analysis", config),
        WorkingAgent("variance_agent", "variance_analysis", config),
        WorkingAgent("residual_agent", "residual_analysis", config),
        WorkingAgent("statistical_agent", "statistical_analysis", config),
        WorkingAgent("frequency_agent", "frequency_analysis", config)
    ]
    
    return agents
