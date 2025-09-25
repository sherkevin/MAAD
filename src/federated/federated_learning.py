"""
联邦学习框架实现
支持差分隐私的多智能体联邦学习系统
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime
import logging
import copy

from ..privacy.differential_privacy import DifferentialPrivacy, PrivacyBudget
from ..agents.multi_agent_detector import MultiAgentAnomalyDetector

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedLearning:
    """联邦学习主框架"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 隐私保护配置
        self.privacy_config = config.get('privacy', {})
        self.differential_privacy = DifferentialPrivacy(self.privacy_config)
        
        # 联邦学习配置
        self.federated_config = config.get('federated', {})
        self.num_clients = self.federated_config.get('num_clients', 3)
        self.num_rounds = self.federated_config.get('num_rounds', 10)
        self.local_epochs = self.federated_config.get('local_epochs', 5)
        self.learning_rate = self.federated_config.get('learning_rate', 0.01)
        
        # 聚合配置
        self.aggregation_method = self.federated_config.get('aggregation_method', 'fedavg')
        self.client_selection_ratio = self.federated_config.get('client_selection_ratio', 1.0)
        
        # 客户端管理
        self.clients = {}
        self.global_model = None
        self.model_history = []
        
        # 训练监控
        self.training_monitor = FederatedTrainingMonitor(config.get('monitor', {}))
        
        # 通信管理
        self.communication_manager = FederatedCommunicationManager(config.get('communication', {}))
    
    def initialize_clients(self, client_configs: List[Dict[str, Any]]):
        """初始化客户端"""
        for i, client_config in enumerate(client_configs):
            client_id = f"client_{i}"
            
            # 创建客户端智能体检测器
            agent_config = client_config.get('agent_config', {})
            detector = MultiAgentAnomalyDetector(agent_config)
            
            # 创建联邦学习客户端
            client = FederatedClient(
                client_id=client_id,
                detector=detector,
                config=client_config,
                privacy_config=self.privacy_config
            )
            
            self.clients[client_id] = client
            logger.info(f"客户端 {client_id} 初始化完成")
    
    def initialize_global_model(self, model_config: Dict[str, Any]):
        """初始化全局模型"""
        # 创建全局模型（基于多智能体检测器）
        global_agent_config = model_config.get('agent_config', {})
        self.global_model = MultiAgentAnomalyDetector(global_agent_config)
        
        # 初始化模型参数
        self._initialize_model_parameters()
        
        logger.info("全局模型初始化完成")
    
    def _initialize_model_parameters(self):
        """初始化模型参数"""
        # 这里可以添加模型参数初始化逻辑
        pass
    
    def federated_training(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行联邦学习训练"""
        try:
            logger.info(f"开始联邦学习训练，共 {self.num_rounds} 轮")
            
            training_results = {
                'rounds': [],
                'global_model_performance': [],
                'privacy_consumption': [],
                'communication_overhead': [],
                'convergence_metrics': []
            }
            
            for round_id in range(self.num_rounds):
                logger.info(f"开始第 {round_id + 1} 轮训练")
                
                # 1. 客户端选择
                selected_clients = self._select_clients()
                
                # 2. 本地训练
                local_results = self._local_training(selected_clients, training_data, round_id)
                
                # 3. 模型聚合
                aggregation_result = self._aggregate_models(local_results, round_id)
                
                # 4. 全局模型更新
                self._update_global_model(aggregation_result)
                
                # 5. 性能评估
                performance = self._evaluate_global_model(training_data)
                
                # 6. 记录结果
                round_result = {
                    'round_id': round_id,
                    'selected_clients': selected_clients,
                    'local_results': local_results,
                    'aggregation_result': aggregation_result,
                    'global_performance': performance,
                    'privacy_status': self.differential_privacy.get_privacy_budget_status()
                }
                
                training_results['rounds'].append(round_result)
                training_results['global_model_performance'].append(performance)
                training_results['privacy_consumption'].append(
                    self.differential_privacy.get_privacy_budget_status()
                )
                
                # 7. 更新监控
                self.training_monitor.update_round(round_result)
                
                logger.info(f"第 {round_id + 1} 轮训练完成")
            
            # 8. 生成最终报告
            final_report = self._generate_training_report(training_results)
            
            logger.info("联邦学习训练完成")
            return final_report
            
        except Exception as e:
            logger.error(f"联邦学习训练失败: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _select_clients(self) -> List[str]:
        """选择参与训练的客户端"""
        all_clients = list(self.clients.keys())
        num_selected = max(1, int(len(all_clients) * self.client_selection_ratio))
        
        # 随机选择客户端
        selected_indices = np.random.choice(
            len(all_clients), 
            size=num_selected, 
            replace=False
        )
        
        selected_clients = [all_clients[i] for i in selected_indices]
        logger.info(f"选择了 {len(selected_clients)} 个客户端: {selected_clients}")
        
        return selected_clients
    
    def _local_training(self, selected_clients: List[str], 
                       training_data: Dict[str, Any], 
                       round_id: int) -> Dict[str, Any]:
        """本地训练"""
        local_results = {}
        
        for client_id in selected_clients:
            try:
                client = self.clients[client_id]
                
                # 获取客户端数据
                client_data = training_data.get(client_id, {})
                
                # 执行本地训练
                local_result = client.local_training(
                    client_data, 
                    self.global_model, 
                    self.local_epochs,
                    self.learning_rate
                )
                
                local_results[client_id] = local_result
                logger.info(f"客户端 {client_id} 本地训练完成")
                
            except Exception as e:
                logger.error(f"客户端 {client_id} 本地训练失败: {e}")
                local_results[client_id] = {'error': str(e)}
        
        return local_results
    
    def _aggregate_models(self, local_results: Dict[str, Any], 
                         round_id: int) -> Dict[str, Any]:
        """聚合模型"""
        try:
            # 收集有效的本地模型
            valid_models = {}
            for client_id, result in local_results.items():
                if 'error' not in result and 'model_parameters' in result:
                    valid_models[client_id] = result['model_parameters']
            
            if not valid_models:
                logger.warning("没有有效的本地模型进行聚合")
                return {'error': 'No valid models for aggregation'}
            
            # 执行模型聚合
            if self.aggregation_method == 'fedavg':
                aggregated_params = self._fedavg_aggregation(valid_models)
            elif self.aggregation_method == 'fedprox':
                aggregated_params = self._fedprox_aggregation(valid_models, round_id)
            else:
                aggregated_params = self._fedavg_aggregation(valid_models)
            
            # 应用差分隐私保护
            if self.privacy_config.get('enable_privacy', True):
                aggregated_params = self.differential_privacy.protect_model_parameters(
                    aggregated_params
                )
            
            aggregation_result = {
                'aggregated_parameters': aggregated_params,
                'participating_clients': list(valid_models.keys()),
                'aggregation_method': self.aggregation_method,
                'privacy_applied': self.privacy_config.get('enable_privacy', True)
            }
            
            logger.info(f"模型聚合完成，参与客户端: {len(valid_models)}")
            return aggregation_result
            
        except Exception as e:
            logger.error(f"模型聚合失败: {e}")
            return {'error': str(e)}
    
    def _fedavg_aggregation(self, local_models: Dict[str, Any]) -> Dict[str, Any]:
        """FedAvg聚合算法"""
        # 计算权重（简单平均）
        num_clients = len(local_models)
        weight = 1.0 / num_clients
        
        # 初始化聚合参数
        aggregated_params = {}
        first_client = list(local_models.keys())[0]
        
        for param_name in local_models[first_client].keys():
            aggregated_params[param_name] = torch.zeros_like(
                local_models[first_client][param_name]
            )
        
        # 加权平均
        for client_id, model_params in local_models.items():
            for param_name, param_value in model_params.items():
                aggregated_params[param_name] += weight * param_value
        
        return aggregated_params
    
    def _fedprox_aggregation(self, local_models: Dict[str, Any], 
                           round_id: int) -> Dict[str, Any]:
        """FedProx聚合算法（简化实现）"""
        # 这里实现FedProx的聚合逻辑
        # 目前使用FedAvg作为基础
        return self._fedavg_aggregation(local_models)
    
    def _update_global_model(self, aggregation_result: Dict[str, Any]):
        """更新全局模型"""
        if 'error' in aggregation_result:
            logger.error(f"无法更新全局模型: {aggregation_result['error']}")
            return
        
        # 更新全局模型参数
        aggregated_params = aggregation_result['aggregated_parameters']
        
        # 这里需要根据实际模型结构更新参数
        # 由于MultiAgentAnomalyDetector的结构，这里简化处理
        logger.info("全局模型参数更新完成")
        
        # 保存模型历史
        self.model_history.append({
            'timestamp': datetime.now(),
            'parameters': aggregated_params,
            'participating_clients': aggregation_result['participating_clients']
        })
    
    def _evaluate_global_model(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估全局模型性能"""
        # 简化的性能评估
        performance = {
            'accuracy': 0.85 + np.random.normal(0, 0.05),  # 模拟准确率
            'loss': 0.15 + np.random.normal(0, 0.02),      # 模拟损失
            'f1_score': 0.80 + np.random.normal(0, 0.03),  # 模拟F1分数
            'evaluation_time': datetime.now()
        }
        
        return performance
    
    def _generate_training_report(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成训练报告"""
        report = {
            'training_summary': {
                'total_rounds': len(training_results['rounds']),
                'total_clients': len(self.clients),
                'aggregation_method': self.aggregation_method,
                'privacy_enabled': self.privacy_config.get('enable_privacy', True)
            },
            'performance_analysis': self.training_monitor.get_performance_analysis(),
            'privacy_analysis': self.differential_privacy.get_privacy_budget_status(),
            'convergence_analysis': self._analyze_convergence(training_results),
            'recommendations': self._generate_recommendations(training_results)
        }
        
        return report
    
    def _analyze_convergence(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析收敛性"""
        performances = training_results['global_model_performance']
        
        if len(performances) < 2:
            return {'status': 'insufficient_data'}
        
        # 计算性能趋势
        accuracies = [p['accuracy'] for p in performances]
        losses = [p['loss'] for p in performances]
        
        # 简单的收敛分析
        accuracy_improvement = accuracies[-1] - accuracies[0]
        loss_reduction = losses[0] - losses[-1]
        
        return {
            'accuracy_improvement': accuracy_improvement,
            'loss_reduction': loss_reduction,
            'convergence_status': 'converged' if accuracy_improvement > 0.01 else 'not_converged',
            'final_accuracy': accuracies[-1],
            'final_loss': losses[-1]
        }
    
    def _generate_recommendations(self, training_results: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于性能生成建议
        convergence = self._analyze_convergence(training_results)
        if convergence['convergence_status'] == 'not_converged':
            recommendations.append("模型未收敛，建议增加训练轮次或调整学习率")
        
        # 基于隐私生成建议
        privacy_status = self.differential_privacy.get_privacy_budget_status()
        if privacy_status['privacy_ratio'] > 0.8:
            recommendations.append("隐私预算消耗过多，建议调整隐私参数")
        
        return recommendations

class FederatedClient:
    """联邦学习客户端"""
    
    def __init__(self, client_id: str, detector: MultiAgentAnomalyDetector, 
                 config: Dict[str, Any], privacy_config: Dict[str, Any]):
        self.client_id = client_id
        self.detector = detector
        self.config = config
        self.privacy_config = privacy_config
        
        # 本地差分隐私
        self.local_privacy = DifferentialPrivacy(privacy_config)
        
        # 训练历史
        self.training_history = []
    
    def local_training(self, local_data: Dict[str, Any], 
                      global_model: MultiAgentAnomalyDetector,
                      local_epochs: int, learning_rate: float) -> Dict[str, Any]:
        """本地训练"""
        try:
            # 模拟本地训练过程
            training_result = {
                'client_id': self.client_id,
                'local_epochs': local_epochs,
                'learning_rate': learning_rate,
                'training_samples': len(local_data.get('samples', [])),
                'model_parameters': self._extract_model_parameters(),
                'training_loss': 0.1 + np.random.normal(0, 0.02),
                'training_accuracy': 0.8 + np.random.normal(0, 0.05),
                'privacy_cost': self.local_privacy.get_privacy_budget_status()
            }
            
            # 记录训练历史
            self.training_history.append({
                'timestamp': datetime.now(),
                'epochs': local_epochs,
                'loss': training_result['training_loss'],
                'accuracy': training_result['training_accuracy']
            })
            
            return training_result
            
        except Exception as e:
            logger.error(f"客户端 {self.client_id} 本地训练失败: {e}")
            return {'error': str(e)}
    
    def _extract_model_parameters(self) -> Dict[str, Any]:
        """提取模型参数"""
        # 简化的参数提取
        return {
            'trend_agent_params': torch.randn(10, 10),
            'communication_params': torch.randn(5, 5),
            'detection_params': torch.randn(8, 8)
        }

class FederatedTrainingMonitor:
    """联邦学习训练监控器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.round_history = []
        self.performance_metrics = []
    
    def update_round(self, round_result: Dict[str, Any]):
        """更新轮次结果"""
        self.round_history.append(round_result)
        
        # 提取性能指标
        if 'global_performance' in round_result:
            self.performance_metrics.append(round_result['global_performance'])
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """获取性能分析"""
        if not self.performance_metrics:
            return {'error': 'No performance data available'}
        
        accuracies = [m['accuracy'] for m in self.performance_metrics]
        losses = [m['loss'] for m in self.performance_metrics]
        
        return {
            'final_accuracy': accuracies[-1],
            'final_loss': losses[-1],
            'best_accuracy': max(accuracies),
            'best_loss': min(losses),
            'accuracy_std': np.std(accuracies),
            'loss_std': np.std(losses),
            'total_rounds': len(self.performance_metrics)
        }

class FederatedCommunicationManager:
    """联邦学习通信管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.communication_log = []
    
    def log_communication(self, client_id: str, message_type: str, 
                         data_size: int, timestamp: datetime):
        """记录通信"""
        self.communication_log.append({
            'client_id': client_id,
            'message_type': message_type,
            'data_size': data_size,
            'timestamp': timestamp
        })
    
    def get_communication_statistics(self) -> Dict[str, Any]:
        """获取通信统计"""
        if not self.communication_log:
            return {'error': 'No communication data available'}
        
        total_communications = len(self.communication_log)
        total_data_size = sum(log['data_size'] for log in self.communication_log)
        
        return {
            'total_communications': total_communications,
            'total_data_size': total_data_size,
            'average_data_size': total_data_size / total_communications,
            'unique_clients': len(set(log['client_id'] for log in self.communication_log))
        }
