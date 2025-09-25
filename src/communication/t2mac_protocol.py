"""
T2MAC (Target-oriented Multi-Agent Communication) 协议实现
基于2024年最新研究的目标导向多智能体通信协议
"""

import torch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime

class CommunicationType(Enum):
    """通信类型枚举"""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    COORDINATION = "coordination"
    EMERGENCY = "emergency"

class MessagePriority(Enum):
    """消息优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

@dataclass
class CommunicationMessage:
    """通信消息数据结构"""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: CommunicationType
    priority: MessagePriority
    content: Dict[str, Any]
    timestamp: datetime
    target_achievement: float  # 目标达成度
    confidence: float  # 消息置信度
    requires_response: bool = False
    response_timeout: float = 5.0  # 响应超时时间(秒)

class T2MACProtocol:
    """T2MAC通信协议实现"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.communication_strategies = {}
        self.efficiency_metrics = CommunicationEfficiencyMetrics()
        self.target_tracker = TargetTracker(config.get('target_tracker', {}))
        self.adaptive_optimizer = AdaptiveCommunicationOptimizer(config.get('optimizer', {}))
        
        # 通信参数
        self.max_communication_rounds = config.get('max_communication_rounds', 5)
        self.communication_threshold = config.get('communication_threshold', 0.7)
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        
    def generate_communication_plan(self, agent_states: Dict[str, Any], 
                                  global_target: Dict[str, Any]) -> Dict[str, Any]:
        """生成目标导向的通信计划"""
        try:
            # 1. 分析智能体状态和目标
            state_analysis = self._analyze_agent_states(agent_states)
            target_analysis = self._analyze_global_target(global_target)
            
            # 2. 计算通信需求
            communication_needs = self._calculate_communication_needs(
                state_analysis, target_analysis
            )
            
            # 3. 生成通信策略
            communication_strategy = self._generate_strategy(communication_needs)
            
            # 4. 优化通信效率
            optimized_plan = self.adaptive_optimizer.optimize(communication_strategy)
            
            # 5. 更新效率指标
            self.efficiency_metrics.update(optimized_plan)
            
            return {
                'communication_plan': optimized_plan,
                'efficiency_metrics': self.efficiency_metrics.get_metrics(),
                'target_progress': self.target_tracker.get_progress(),
                'strategy_type': 't2mac'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'communication_plan': self._get_fallback_plan(),
                'strategy_type': 'fallback'
            }
    
    def _analyze_agent_states(self, agent_states: Dict[str, Any]) -> Dict[str, Any]:
        """分析智能体状态"""
        analysis = {
            'agent_count': len(agent_states),
            'active_agents': [],
            'idle_agents': [],
            'error_agents': [],
            'confidence_scores': {},
            'processing_times': {},
            'communication_readiness': {}
        }
        
        for agent_id, state in agent_states.items():
            analysis['confidence_scores'][agent_id] = getattr(state, 'confidence_score', 0.0)
            analysis['processing_times'][agent_id] = getattr(state, 'processing_time', 0.0)
            
            status = getattr(state, 'status', 'unknown')
            if status == 'active':
                analysis['active_agents'].append(agent_id)
            elif status == 'idle':
                analysis['idle_agents'].append(agent_id)
            elif status == 'error':
                analysis['error_agents'].append(agent_id)
            
            # 计算通信准备度
            readiness = self._calculate_communication_readiness(state)
            analysis['communication_readiness'][agent_id] = readiness
        
        return analysis
    
    def _analyze_global_target(self, global_target: Dict[str, Any]) -> Dict[str, Any]:
        """分析全局目标"""
        return {
            'target_type': global_target.get('type', 'anomaly_detection'),
            'target_priority': global_target.get('priority', 1.0),
            'target_deadline': global_target.get('deadline', None),
            'target_requirements': global_target.get('requirements', {}),
            'current_progress': global_target.get('progress', 0.0)
        }
    
    def _calculate_communication_needs(self, state_analysis: Dict[str, Any], 
                                     target_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """计算通信需求"""
        needs = {
            'coordination_required': False,
            'information_sharing': [],
            'resource_requests': [],
            'emergency_communications': [],
            'priority_level': MessagePriority.NORMAL
        }
        
        # 检查是否需要协调
        if len(state_analysis['error_agents']) > 0:
            needs['coordination_required'] = True
            needs['priority_level'] = MessagePriority.HIGH
        
        # 检查信息共享需求
        low_confidence_agents = [
            agent_id for agent_id, conf in state_analysis['confidence_scores'].items()
            if conf < self.communication_threshold
        ]
        if low_confidence_agents:
            needs['information_sharing'] = low_confidence_agents
        
        # 检查紧急通信需求
        if target_analysis['target_priority'] > 0.8:
            needs['priority_level'] = MessagePriority.URGENT
        
        return needs
    
    def _generate_strategy(self, communication_needs: Dict[str, Any]) -> Dict[str, Any]:
        """生成通信策略"""
        strategy = {
            'communication_rounds': [],
            'message_priorities': {},
            'communication_schedule': {},
            'expected_outcomes': {}
        }
        
        # 生成通信轮次
        for round_id in range(self.max_communication_rounds):
            round_plan = self._generate_round_plan(round_id, communication_needs)
            strategy['communication_rounds'].append(round_plan)
        
        # 设置消息优先级
        strategy['message_priorities'] = self._assign_message_priorities(communication_needs)
        
        # 生成通信调度
        strategy['communication_schedule'] = self._create_communication_schedule(strategy)
        
        return strategy
    
    def _generate_round_plan(self, round_id: int, needs: Dict[str, Any]) -> Dict[str, Any]:
        """生成单轮通信计划"""
        return {
            'round_id': round_id,
            'messages': self._generate_messages_for_round(round_id, needs),
            'expected_duration': 1.0,  # 秒
            'success_criteria': self._define_success_criteria(round_id)
        }
    
    def _generate_messages_for_round(self, round_id: int, needs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """为单轮生成消息"""
        messages = []
        
        if needs['coordination_required']:
            # 协调消息
            messages.append({
                'type': CommunicationType.COORDINATION,
                'priority': MessagePriority.HIGH,
                'content': {'action': 'coordinate', 'round': round_id}
            })
        
        if needs['information_sharing']:
            # 信息共享消息
            for agent_id in needs['information_sharing']:
                messages.append({
                    'type': CommunicationType.REQUEST,
                    'priority': MessagePriority.NORMAL,
                    'content': {'action': 'share_info', 'target_agent': agent_id}
                })
        
        return messages
    
    def _assign_message_priorities(self, needs: Dict[str, Any]) -> Dict[str, MessagePriority]:
        """分配消息优先级"""
        priorities = {}
        
        if needs['coordination_required']:
            priorities['coordination'] = MessagePriority.HIGH
        
        if needs['information_sharing']:
            for agent_id in needs['information_sharing']:
                priorities[f'info_share_{agent_id}'] = MessagePriority.NORMAL
        
        return priorities
    
    def _create_communication_schedule(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """创建通信调度"""
        schedule = {
            'total_rounds': len(strategy['communication_rounds']),
            'estimated_duration': sum(
                round_plan['expected_duration'] 
                for round_plan in strategy['communication_rounds']
            ),
            'parallel_communications': self._identify_parallel_opportunities(strategy)
        }
        return schedule
    
    def _identify_parallel_opportunities(self, strategy: Dict[str, Any]) -> List[List[int]]:
        """识别并行通信机会"""
        # 简单的并行识别：相邻轮次可以并行
        parallel_groups = []
        for i in range(0, len(strategy['communication_rounds']), 2):
            group = list(range(i, min(i + 2, len(strategy['communication_rounds']))))
            if len(group) > 1:
                parallel_groups.append(group)
        return parallel_groups
    
    def _calculate_communication_readiness(self, agent_state: Any) -> float:
        """计算智能体通信准备度"""
        confidence = getattr(agent_state, 'confidence_score', 0.0)
        status = getattr(agent_state, 'status', 'unknown')
        error_count = getattr(agent_state, 'error_count', 0)
        
        # 基于置信度、状态和错误数计算准备度
        readiness = confidence * 0.5  # 基础准备度
        
        if status == 'active':
            readiness += 0.3
        elif status == 'idle':
            readiness += 0.2
        elif status == 'error':
            readiness -= 0.5
        
        # 错误惩罚
        readiness -= min(0.3, error_count * 0.1)
        
        return max(0.0, min(1.0, readiness))
    
    def _define_success_criteria(self, round_id: int) -> Dict[str, Any]:
        """定义成功标准"""
        return {
            'min_response_rate': 0.8,
            'max_response_time': 2.0,
            'min_confidence_improvement': 0.1
        }
    
    def _get_fallback_plan(self) -> Dict[str, Any]:
        """获取备用通信计划"""
        return {
            'communication_rounds': [{
                'round_id': 0,
                'messages': [{
                    'type': CommunicationType.BROADCAST,
                    'priority': MessagePriority.NORMAL,
                    'content': {'action': 'status_check'}
                }],
                'expected_duration': 1.0,
                'success_criteria': {'min_response_rate': 0.5}
            }],
            'message_priorities': {'status_check': MessagePriority.NORMAL},
            'communication_schedule': {
                'total_rounds': 1,
                'estimated_duration': 1.0,
                'parallel_communications': []
            }
        }

class CommunicationEfficiencyMetrics:
    """通信效率指标"""
    
    def __init__(self):
        self.metrics = {
            'total_messages': 0,
            'successful_communications': 0,
            'failed_communications': 0,
            'average_response_time': 0.0,
            'communication_overhead': 0.0,
            'target_achievement_rate': 0.0
        }
    
    def update(self, communication_plan: Dict[str, Any]):
        """更新效率指标"""
        # 计算消息总数
        total_messages = sum(
            len(round_plan.get('messages', []))
            for round_plan in communication_plan.get('communication_rounds', [])
        )
        self.metrics['total_messages'] += total_messages
        
        # 更新其他指标（简化实现）
        self.metrics['successful_communications'] += total_messages * 0.9  # 假设90%成功
        self.metrics['failed_communications'] += total_messages * 0.1
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取效率指标"""
        return self.metrics.copy()

class TargetTracker:
    """目标跟踪器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_targets = {}
        self.target_history = []
    
    def get_progress(self) -> Dict[str, Any]:
        """获取目标进度"""
        return {
            'active_targets': len(self.current_targets),
            'completion_rate': 0.0,  # 简化实现
            'target_status': 'in_progress'
        }

class AdaptiveCommunicationOptimizer:
    """自适应通信优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimization_history = []
    
    def optimize(self, communication_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """优化通信策略"""
        # 简单的优化：减少不必要的通信轮次
        optimized_strategy = communication_strategy.copy()
        
        # 如果只有低优先级消息，减少轮次
        if all(
            msg.get('priority', MessagePriority.NORMAL).value <= MessagePriority.NORMAL.value
            for round_plan in communication_strategy.get('communication_rounds', [])
            for msg in round_plan.get('messages', [])
        ):
            # 保留前3轮，合并后面的轮次
            if len(optimized_strategy['communication_rounds']) > 3:
                optimized_strategy['communication_rounds'] = optimized_strategy['communication_rounds'][:3]
        
        return optimized_strategy
