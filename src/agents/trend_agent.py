import torch
import torch.nn as nn
from typing import Dict, Any
from .base_agent import BaseAgent

class TrendAgent(BaseAgent):
    """趋势分析智能体"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("trend", "trend_analysis", config)
        
        # 趋势分析器
        self.trend_analyzer = TrendAnalyzer(config.get('trend_analyzer', {}))
        
        # 物理约束
        self.physics_constraints = PhysicsConstraints(config.get('physics', {}))
        
        # 特征提取器
        self.feature_extractor = self._build_feature_extractor()
        
        # 异常检测器
        self.anomaly_detector = self._build_anomaly_detector()
    
    def _build_feature_extractor(self):
        """构建特征提取器"""
        # 简单的趋势特征提取器
        return TrendFeatureExtractor(self.config.get('feature_extractor', {}))
    
    def _build_anomaly_detector(self):
        """构建异常检测器"""
        return TrendAnomalyDetector(self.config.get('anomaly_detector', {}))
    
    def process_data(self, data: torch.Tensor) -> Dict[str, Any]:
        """处理趋势数据"""
        try:
            self.state.status = "active"
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            # 1. 趋势分析
            trend_features = self.trend_analyzer.extract(data)
            
            # 2. 物理约束应用
            constrained_features = self.physics_constraints.apply(trend_features)
            
            # 3. 特征提取
            extracted_features = self.feature_extractor.extract(constrained_features)
            
            # 4. 异常检测
            anomaly_score = self.anomaly_detector.detect(extracted_features)
            
            # 5. 置信度计算
            confidence = self._calculate_confidence(extracted_features, anomaly_score)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                processing_time = start_time.elapsed_time(end_time) / 1000.0  # 转换为秒
            else:
                processing_time = 0.0
            
            # 更新状态
            self.state.processing_time = processing_time
            self.state.confidence_score = confidence
            self.state.memory_usage = self.get_memory_usage()
            self.state.status = "idle"
            
            return {
                'agent_id': self.agent_id,
                'features': extracted_features,
                'anomaly_score': anomaly_score,
                'confidence': confidence,
                'processing_time': processing_time
            }
            
        except Exception as e:
            self.state.status = "error"
            self.state.error_count += 1
            raise e
    
    def _calculate_confidence(self, features: torch.Tensor, anomaly_score: float) -> float:
        """计算置信度"""
        # 基于特征质量和异常分数的置信度计算
        feature_quality = torch.std(features).item() if features.numel() > 0 else 0.0
        confidence = min(1.0, max(0.0, 1.0 - abs(anomaly_score - 0.5) * 2))
        return confidence * (1.0 + feature_quality)

class TrendAnalyzer:
    """趋势分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smoothing_factor = config.get('smoothing_factor', 0.1)
    
    def extract(self, data: torch.Tensor) -> torch.Tensor:
        """提取趋势特征"""
        # 简单的趋势分析：计算移动平均
        if data.dim() > 2:
            # 对于多维数据，在最后一个维度上计算趋势
            trend = torch.mean(data, dim=-1, keepdim=True)
        else:
            trend = data
        return trend

class PhysicsConstraints:
    """物理约束"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smoothness_weight = config.get('smoothness_weight', 0.1)
    
    def apply(self, features: torch.Tensor) -> torch.Tensor:
        """应用物理约束"""
        # 简单的平滑约束
        if features.dim() > 1:
            # 应用平滑滤波
            kernel_size = min(3, features.shape[-1])
            if kernel_size > 1:
                smooth_features = torch.nn.functional.avg_pool1d(
                    features.unsqueeze(0), 
                    kernel_size=kernel_size, 
                    stride=1, 
                    padding=kernel_size//2
                ).squeeze(0)
                return smooth_features
        return features

class TrendFeatureExtractor:
    """趋势特征提取器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def extract(self, data: torch.Tensor) -> torch.Tensor:
        """提取趋势特征"""
        # 计算趋势相关的统计特征
        if data.dim() > 1:
            # 计算均值、方差、最大值、最小值
            mean_feat = torch.mean(data, dim=-1, keepdim=True)
            std_feat = torch.std(data, dim=-1, keepdim=True)
            max_feat = torch.max(data, dim=-1, keepdim=True)[0]
            min_feat = torch.min(data, dim=-1, keepdim=True)[0]
            features = torch.cat([mean_feat, std_feat, max_feat, min_feat], dim=-1)
        else:
            features = data.unsqueeze(-1)
        return features

class TrendAnomalyDetector:
    """趋势异常检测器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.threshold = config.get('threshold', 0.5)
        self.memory_bank = []
    
    def detect(self, features: torch.Tensor) -> float:
        """检测异常"""
        # 简单的基于阈值的异常检测
        if len(self.memory_bank) < 10:
            # 如果记忆库太小，返回中性分数
            self.memory_bank.append(features.detach().cpu())
            return 0.5
        
        # 计算与历史特征的相似度
        current_feature = features.detach().cpu()
        similarities = []
        for hist_feature in self.memory_bank[-10:]:  # 使用最近10个特征
            if hist_feature.shape == current_feature.shape:
                similarity = torch.cosine_similarity(
                    current_feature.flatten(), 
                    hist_feature.flatten(), 
                    dim=0
                ).item()
                similarities.append(similarity)
        
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            # 相似度越低，异常分数越高
            anomaly_score = 1.0 - avg_similarity
        else:
            anomaly_score = 0.5
        
        # 更新记忆库
        self.memory_bank.append(current_feature)
        if len(self.memory_bank) > 100:  # 限制记忆库大小
            self.memory_bank = self.memory_bank[-50:]
        
        return min(1.0, max(0.0, anomaly_score))
