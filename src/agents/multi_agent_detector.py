import torch
from typing import Dict, Any, List
from .base_agent import BaseAgent
from .trend_agent import TrendAgent

class MultiAgentAnomalyDetector:
    """多智能体异常检测器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化智能体
        self.agents = {
            'trend': TrendAgent(config.get('trend_agent', {})),
            # 后续添加其他智能体
        }
        
        # 通信总线
        self.communication_bus = CommunicationBus(config.get('communication_bus', {}))
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor(config.get('performance_monitor', {}))
    
    def detect_anomaly(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """多智能体协同异常检测"""
        try:
            # 1. 数据预处理
            processed_data = self._preprocess_data(input_data)
            
            # 2. 各智能体处理
            agent_outputs = []
            for agent_id, agent in self.agents.items():
                output = agent.process_data(processed_data)
                agent_outputs.append(output)
            
            # 3. 协调器决策
            final_decision = self._fuse_decisions(agent_outputs)
            
            # 4. 性能监控
            performance_metrics = self.performance_monitor.collect_metrics(self.agents)
            
            # 5. 结果整合
            result = {
                'final_decision': final_decision,
                'agent_outputs': agent_outputs,
                'performance_metrics': performance_metrics,
                'detection_time': self.performance_monitor.get_total_time()
            }
            
            return result
            
        except Exception as e:
            # 错误处理
            return {
                'error': str(e),
                'status': 'failed'
            }
    
    def _preprocess_data(self, data: torch.Tensor) -> torch.Tensor:
        """数据预处理"""
        # 简单的数据预处理
        if data.dim() > 2:
            # 对于多维数据，展平为2D
            data = data.view(data.shape[0], -1)
        return data
    
    def _fuse_decisions(self, agent_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """融合决策"""
        if not agent_outputs:
            return {'anomaly_score': 0.5, 'confidence': 0.0, 'fusion_method': 'none'}
        
        # 简单的加权平均融合
        total_confidence = sum(output.get('confidence', 0.0) for output in agent_outputs)
        
        if total_confidence > 0:
            # 基于置信度的加权平均
            weights = [output.get('confidence', 0.0) / total_confidence for output in agent_outputs]
            weighted_anomaly_score = sum(
                output.get('anomaly_score', 0.0) * weight 
                for output, weight in zip(agent_outputs, weights)
            )
            weighted_confidence = sum(
                output.get('confidence', 0.0) * weight 
                for output, weight in zip(agent_outputs, weights)
            )
        else:
            # 如果所有置信度都为0，使用简单平均
            weighted_anomaly_score = sum(output.get('anomaly_score', 0.0) for output in agent_outputs) / len(agent_outputs)
            weighted_confidence = sum(output.get('confidence', 0.0) for output in agent_outputs) / len(agent_outputs)
        
        return {
            'anomaly_score': weighted_anomaly_score,
            'confidence': weighted_confidence,
            'fusion_method': 'weighted_average'
        }
    
    def get_agent_states(self) -> Dict[str, Any]:
        """获取所有智能体状态"""
        states = {}
        for agent_id, agent in self.agents.items():
            states[agent_id] = agent.get_state()
        return states
    
    def reset_all_agents(self):
        """重置所有智能体"""
        for agent in self.agents.values():
            agent.reset()
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        self.config.update(new_config)
        # 更新各智能体配置
        for agent_id, agent in self.agents.items():
            if agent_id in new_config:
                agent.config.update(new_config[agent_id])

class CommunicationBus:
    """通信总线"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.message_queue = []
        self.max_queue_size = config.get('max_queue_size', 1000)
    
    def send_message(self, sender: str, receiver: str, message: Dict[str, Any]):
        """发送消息"""
        if len(self.message_queue) >= self.max_queue_size:
            # 移除最旧的消息
            self.message_queue.pop(0)
        
        self.message_queue.append({
            'sender': sender,
            'receiver': receiver,
            'message': message,
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        })
    
    def get_messages(self, receiver: str) -> List[Dict[str, Any]]:
        """获取消息"""
        messages = [msg for msg in self.message_queue if msg['receiver'] == receiver]
        return messages
    
    def clear_messages(self):
        """清空消息队列"""
        self.message_queue.clear()

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = {}
        self.start_time = None
    
    def start_timing(self):
        """开始计时"""
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time:
            self.start_time.record()
    
    def collect_metrics(self, agents: Dict[str, BaseAgent]) -> Dict[str, Any]:
        """收集性能指标"""
        metrics = {}
        for agent_id, agent in agents.items():
            state = agent.get_state()
            metrics[agent_id] = {
                'processing_time': state.processing_time,
                'confidence': state.confidence_score,
                'error_count': state.error_count,
                'status': state.status,
                'memory_usage': state.memory_usage
            }
        return metrics
    
    def get_total_time(self) -> float:
        """获取总处理时间"""
        if self.start_time and torch.cuda.is_available():
            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()
            torch.cuda.synchronize()
            return self.start_time.elapsed_time(end_time) / 1000.0  # 转换为秒
        return 0.0
