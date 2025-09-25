"""
差分隐私机制实现
基于差分隐私理论的多智能体联邦学习隐私保护
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import math
from dataclasses import dataclass
from enum import Enum
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrivacyMechanism(Enum):
    """隐私机制类型"""
    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    MECHANISM_SELECTION = "mechanism_selection"

@dataclass
class PrivacyBudget:
    """隐私预算管理"""
    epsilon: float  # 隐私参数ε
    delta: float    # 失败概率δ
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0
    
    def can_spend(self, epsilon_cost: float, delta_cost: float = 0.0) -> bool:
        """检查是否可以消费隐私预算"""
        return (self.consumed_epsilon + epsilon_cost <= self.epsilon and 
                self.consumed_delta + delta_cost <= self.delta)
    
    def spend(self, epsilon_cost: float, delta_cost: float = 0.0) -> bool:
        """消费隐私预算"""
        if self.can_spend(epsilon_cost, delta_cost):
            self.consumed_epsilon += epsilon_cost
            self.consumed_delta += delta_cost
            return True
        return False
    
    def get_remaining_epsilon(self) -> float:
        """获取剩余ε预算"""
        return max(0.0, self.epsilon - self.consumed_epsilon)
    
    def get_remaining_delta(self) -> float:
        """获取剩余δ预算"""
        return max(0.0, self.delta - self.consumed_delta)

class DifferentialPrivacy:
    """差分隐私实现"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.privacy_budget = PrivacyBudget(
            epsilon=config.get('epsilon', 1.0),
            delta=config.get('delta', 1e-5)
        )
        self.mechanism = PrivacyMechanism(config.get('mechanism', 'gaussian'))
        self.sensitivity = config.get('sensitivity', 1.0)
        self.noise_scale = self._calculate_noise_scale()
        
        # 隐私监控
        self.privacy_monitor = PrivacyMonitor(config.get('privacy_monitor', {}))
        
        # 噪声生成器
        self.noise_generator = NoiseGenerator(config.get('noise_generator', {}))
    
    def _calculate_noise_scale(self) -> float:
        """计算噪声尺度"""
        if self.mechanism == PrivacyMechanism.LAPLACE:
            return self.sensitivity / self.privacy_budget.epsilon
        elif self.mechanism == PrivacyMechanism.GAUSSIAN:
            # 高斯机制的噪声尺度计算
            sigma = math.sqrt(2 * math.log(1.25 / self.privacy_budget.delta)) * self.sensitivity / self.privacy_budget.epsilon
            return sigma
        else:
            return self.sensitivity / self.privacy_budget.epsilon
    
    def add_noise(self, data: torch.Tensor, sensitivity: Optional[float] = None) -> torch.Tensor:
        """添加差分隐私噪声"""
        try:
            if sensitivity is None:
                sensitivity = self.sensitivity
            
            # 检查隐私预算
            epsilon_cost = self._calculate_epsilon_cost(data, sensitivity)
            if not self.privacy_budget.can_spend(epsilon_cost):
                logger.warning("隐私预算不足，使用备用策略")
                return self._apply_fallback_noise(data)
            
            # 生成噪声
            noise = self.noise_generator.generate_noise(
                data.shape, 
                self.noise_scale, 
                self.mechanism
            )
            
            # 添加噪声
            noisy_data = data + noise
            
            # 消费隐私预算
            self.privacy_budget.spend(epsilon_cost)
            
            # 更新隐私监控
            self.privacy_monitor.record_noise_addition(data, noisy_data, epsilon_cost)
            
            return noisy_data
            
        except Exception as e:
            logger.error(f"添加噪声失败: {e}")
            return self._apply_fallback_noise(data)
    
    def _calculate_epsilon_cost(self, data: torch.Tensor, sensitivity: float) -> float:
        """计算ε成本"""
        # 简化的ε成本计算
        data_norm = torch.norm(data).item()
        return (sensitivity * data_norm) / (self.privacy_budget.epsilon + 1e-8)
    
    def _apply_fallback_noise(self, data: torch.Tensor) -> torch.Tensor:
        """应用备用噪声策略"""
        # 使用较小的噪声作为备用策略
        fallback_scale = self.noise_scale * 0.1
        noise = self.noise_generator.generate_noise(
            data.shape, 
            fallback_scale, 
            PrivacyMechanism.GAUSSIAN
        )
        return data + noise
    
    def protect_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """保护梯度"""
        protected_gradients = []
        
        for grad in gradients:
            if grad is not None:
                protected_grad = self.add_noise(grad)
                protected_gradients.append(protected_grad)
            else:
                protected_gradients.append(None)
        
        return protected_gradients
    
    def protect_model_parameters(self, model_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """保护模型参数"""
        protected_params = {}
        
        for name, param in model_params.items():
            protected_params[name] = self.add_noise(param)
        
        return protected_params
    
    def get_privacy_budget_status(self) -> Dict[str, Any]:
        """获取隐私预算状态"""
        return {
            'total_epsilon': self.privacy_budget.epsilon,
            'total_delta': self.privacy_budget.delta,
            'consumed_epsilon': self.privacy_budget.consumed_epsilon,
            'consumed_delta': self.privacy_budget.consumed_delta,
            'remaining_epsilon': self.privacy_budget.get_remaining_epsilon(),
            'remaining_delta': self.privacy_budget.get_remaining_delta(),
            'privacy_ratio': self.privacy_budget.consumed_epsilon / self.privacy_budget.epsilon
        }
    
    def reset_privacy_budget(self):
        """重置隐私预算"""
        self.privacy_budget.consumed_epsilon = 0.0
        self.privacy_budget.consumed_delta = 0.0
        self.privacy_monitor.reset()

class NoiseGenerator:
    """噪声生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.random_seed = config.get('random_seed', None)
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
    
    def generate_noise(self, shape: Tuple[int, ...], scale: float, 
                      mechanism: PrivacyMechanism) -> torch.Tensor:
        """生成噪声"""
        if mechanism == PrivacyMechanism.LAPLACE:
            return self._generate_laplace_noise(shape, scale)
        elif mechanism == PrivacyMechanism.GAUSSIAN:
            return self._generate_gaussian_noise(shape, scale)
        elif mechanism == PrivacyMechanism.EXPONENTIAL:
            return self._generate_exponential_noise(shape, scale)
        else:
            return self._generate_gaussian_noise(shape, scale)
    
    def _generate_laplace_noise(self, shape: Tuple[int, ...], scale: float) -> torch.Tensor:
        """生成拉普拉斯噪声"""
        # 拉普拉斯分布: L(0, scale)
        uniform = torch.rand(shape) - 0.5
        laplace = scale * torch.sign(uniform) * torch.log(1 - 2 * torch.abs(uniform))
        return laplace
    
    def _generate_gaussian_noise(self, shape: Tuple[int, ...], scale: float) -> torch.Tensor:
        """生成高斯噪声"""
        # 高斯分布: N(0, scale^2)
        return torch.normal(0, scale, shape)
    
    def _generate_exponential_noise(self, shape: Tuple[int, ...], scale: float) -> torch.Tensor:
        """生成指数噪声"""
        # 指数分布: Exp(1/scale)
        uniform = torch.rand(shape)
        exponential = -scale * torch.log(1 - uniform)
        return exponential

class PrivacyMonitor:
    """隐私监控器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.noise_records = []
        self.privacy_loss_records = []
        self.max_records = config.get('max_records', 1000)
    
    def record_noise_addition(self, original_data: torch.Tensor, 
                            noisy_data: torch.Tensor, epsilon_cost: float):
        """记录噪声添加"""
        record = {
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None,
            'original_norm': torch.norm(original_data).item(),
            'noisy_norm': torch.norm(noisy_data).item(),
            'noise_norm': torch.norm(noisy_data - original_data).item(),
            'epsilon_cost': epsilon_cost,
            'privacy_loss': self._calculate_privacy_loss(original_data, noisy_data)
        }
        
        self.noise_records.append(record)
        
        # 限制记录数量
        if len(self.noise_records) > self.max_records:
            self.noise_records = self.noise_records[-self.max_records//2:]
    
    def _calculate_privacy_loss(self, original_data: torch.Tensor, 
                              noisy_data: torch.Tensor) -> float:
        """计算隐私损失"""
        # 简化的隐私损失计算
        data_diff = torch.norm(noisy_data - original_data).item()
        return data_diff / (torch.norm(original_data).item() + 1e-8)
    
    def get_privacy_statistics(self) -> Dict[str, Any]:
        """获取隐私统计信息"""
        if not self.noise_records:
            return {'error': 'No privacy records available'}
        
        total_epsilon_cost = sum(record['epsilon_cost'] for record in self.noise_records)
        avg_privacy_loss = sum(record['privacy_loss'] for record in self.noise_records) / len(self.noise_records)
        avg_noise_norm = sum(record['noise_norm'] for record in self.noise_records) / len(self.noise_records)
        
        return {
            'total_noise_additions': len(self.noise_records),
            'total_epsilon_cost': total_epsilon_cost,
            'average_privacy_loss': avg_privacy_loss,
            'average_noise_norm': avg_noise_norm,
            'privacy_efficiency': 1.0 / (avg_privacy_loss + 1e-8)
        }
    
    def reset(self):
        """重置监控器"""
        self.noise_records.clear()
        self.privacy_loss_records.clear()

class PrivacyAnalyzer:
    """隐私分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def analyze_privacy_guarantee(self, epsilon: float, delta: float, 
                                 sensitivity: float) -> Dict[str, Any]:
        """分析隐私保证"""
        # 计算隐私保证强度
        privacy_strength = self._calculate_privacy_strength(epsilon, delta)
        
        # 计算攻击难度
        attack_difficulty = self._calculate_attack_difficulty(epsilon, delta, sensitivity)
        
        # 计算隐私风险
        privacy_risk = self._calculate_privacy_risk(epsilon, delta, sensitivity)
        
        return {
            'privacy_strength': privacy_strength,
            'attack_difficulty': attack_difficulty,
            'privacy_risk': privacy_risk,
            'recommendation': self._get_privacy_recommendation(privacy_strength, privacy_risk)
        }
    
    def _calculate_privacy_strength(self, epsilon: float, delta: float) -> float:
        """计算隐私强度"""
        # 基于ε和δ计算隐私强度
        if epsilon <= 0.1 and delta <= 1e-6:
            return 1.0  # 强隐私
        elif epsilon <= 1.0 and delta <= 1e-5:
            return 0.8  # 中等隐私
        elif epsilon <= 10.0 and delta <= 1e-4:
            return 0.6  # 弱隐私
        else:
            return 0.3  # 很弱隐私
    
    def _calculate_attack_difficulty(self, epsilon: float, delta: float, 
                                   sensitivity: float) -> float:
        """计算攻击难度"""
        # 基于隐私参数计算攻击难度
        difficulty = 1.0 / (epsilon + 1e-8)
        difficulty *= (1.0 / (delta + 1e-8))
        difficulty *= sensitivity
        
        return min(1.0, difficulty / 1000.0)  # 归一化到[0,1]
    
    def _calculate_privacy_risk(self, epsilon: float, delta: float, 
                              sensitivity: float) -> float:
        """计算隐私风险"""
        # 隐私风险与ε和δ成正比
        risk = epsilon * delta * sensitivity
        return min(1.0, risk)
    
    def _get_privacy_recommendation(self, privacy_strength: float, 
                                  privacy_risk: float) -> str:
        """获取隐私建议"""
        if privacy_strength > 0.8 and privacy_risk < 0.2:
            return "隐私保护良好，可以继续使用"
        elif privacy_strength > 0.6 and privacy_risk < 0.4:
            return "隐私保护中等，建议监控使用"
        elif privacy_strength > 0.4 and privacy_risk < 0.6:
            return "隐私保护较弱，建议加强保护"
        else:
            return "隐私保护不足，建议重新配置参数"
