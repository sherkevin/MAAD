"""
LLM驱动的通信系统
集成T2MAC协议和Qwen LLM，实现智能通信计划生成和消息优化
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from .t2mac_protocol import T2MACProtocol
from ..llm.qwen_interface import QwenLLMInterface


class LLMDrivenCommunication:
    """LLM驱动的多智能体通信系统"""
    
    def __init__(self, 
                 llm_model_path: str = "Qwen/Qwen2.5-0.5B",
                 max_communication_rounds: int = 5,
                 confidence_threshold: float = 0.8,
                 enable_llm_optimization: bool = True):
        """
        初始化LLM驱动通信系统
        
        Args:
            llm_model_path: LLM模型路径
            max_communication_rounds: 最大通信轮次
            confidence_threshold: 置信度阈值
            enable_llm_optimization: 是否启用LLM优化
        """
        self.llm_interface = QwenLLMInterface(model_path=llm_model_path)
        self.t2mac_protocol = T2MACProtocol(
            max_rounds=max_communication_rounds,
            confidence_threshold=confidence_threshold
        )
        self.enable_llm_optimization = enable_llm_optimization
        self.communication_history = []
        
    def generate_communication_plan(self, 
                                  agent_states: Dict[str, Any],
                                  anomaly_scores: Dict[str, float],
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用LLM生成通信计划
        
        Args:
            agent_states: 智能体状态
            anomaly_scores: 异常分数
            context: 上下文信息
            
        Returns:
            通信计划
        """
        if not self.enable_llm_optimization:
            # 使用基础T2MAC协议
            return self.t2mac_protocol.generate_communication_plan(
                agent_states, anomaly_scores, context
            )
        
        # 构建LLM输入
        llm_input = self._build_llm_input(agent_states, anomaly_scores, context)
        
        # 生成通信策略
        strategy = self.llm_interface.generate_strategy(llm_input)
        
        # 基于LLM策略生成通信计划
        plan = self._strategy_to_plan(strategy, agent_states, anomaly_scores)
        
        # 记录通信历史
        self.communication_history.append({
            'timestamp': torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True)),
            'plan': plan,
            'strategy': strategy
        })
        
        return plan
    
    def optimize_message(self, 
                        message: str, 
                        target_agent: str,
                        context: Dict[str, Any]) -> str:
        """
        使用LLM优化消息内容
        
        Args:
            message: 原始消息
            target_agent: 目标智能体
            context: 上下文信息
            
        Returns:
            优化后的消息
        """
        if not self.enable_llm_optimization:
            return message
        
        # 构建消息优化提示
        prompt = f"""
        优化以下多智能体通信消息：
        
        原始消息: {message}
        目标智能体: {target_agent}
        上下文: {context}
        
        请优化消息，使其更清晰、更有效。
        """
        
        optimized_message = self.llm_interface.generate_response(prompt)
        return optimized_message
    
    def analyze_communication_effectiveness(self, 
                                          communication_log: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        分析通信效果
        
        Args:
            communication_log: 通信日志
            
        Returns:
            效果分析结果
        """
        if not communication_log:
            return {'effectiveness': 0.0, 'efficiency': 0.0, 'clarity': 0.0}
        
        # 计算通信效果指标
        total_rounds = len(communication_log)
        successful_communications = sum(1 for log in communication_log if log.get('success', False))
        
        effectiveness = successful_communications / total_rounds if total_rounds > 0 else 0.0
        
        # 计算效率（基于通信轮次）
        avg_rounds = np.mean([log.get('rounds', 0) for log in communication_log])
        efficiency = max(0, 1 - (avg_rounds - 1) / 5)  # 假设最优轮次为1
        
        # 计算清晰度（基于消息质量）
        clarity_scores = [log.get('clarity', 0.5) for log in communication_log]
        clarity = np.mean(clarity_scores) if clarity_scores else 0.5
        
        return {
            'effectiveness': effectiveness,
            'efficiency': efficiency,
            'clarity': clarity,
            'total_rounds': total_rounds,
            'successful_communications': successful_communications
        }
    
    def _build_llm_input(self, 
                        agent_states: Dict[str, Any],
                        anomaly_scores: Dict[str, float],
                        context: Dict[str, Any]) -> str:
        """构建LLM输入"""
        input_text = f"""
        多智能体异常检测通信策略生成：
        
        智能体状态:
        {self._format_agent_states(agent_states)}
        
        异常分数:
        {self._format_anomaly_scores(anomaly_scores)}
        
        上下文信息:
        {self._format_context(context)}
        
        请生成最优的通信策略。
        """
        return input_text
    
    def _strategy_to_plan(self, 
                         strategy: str,
                         agent_states: Dict[str, Any],
                         anomaly_scores: Dict[str, float]) -> Dict[str, Any]:
        """将LLM策略转换为通信计划"""
        # 解析LLM生成的策略
        plan = {
            'strategy_type': 'llm_optimized',
            'communication_rounds': 3,
            'target_agents': list(agent_states.keys()),
            'priority_order': sorted(agent_states.keys(), 
                                   key=lambda x: anomaly_scores.get(x, 0), 
                                   reverse=True),
            'message_templates': self._generate_message_templates(strategy),
            'confidence_threshold': 0.8,
            'llm_strategy': strategy
        }
        return plan
    
    def _format_agent_states(self, agent_states: Dict[str, Any]) -> str:
        """格式化智能体状态"""
        formatted = []
        for agent_id, state in agent_states.items():
            formatted.append(f"  {agent_id}: {state}")
        return "\n".join(formatted)
    
    def _format_anomaly_scores(self, anomaly_scores: Dict[str, float]) -> str:
        """格式化异常分数"""
        formatted = []
        for agent_id, score in anomaly_scores.items():
            formatted.append(f"  {agent_id}: {score:.4f}")
        return "\n".join(formatted)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """格式化上下文信息"""
        formatted = []
        for key, value in context.items():
            formatted.append(f"  {key}: {value}")
        return "\n".join(formatted)
    
    def _generate_message_templates(self, strategy: str) -> Dict[str, str]:
        """基于策略生成消息模板"""
        templates = {
            'trend': f"基于LLM策略的趋势分析: {strategy}",
            'seasonal': f"基于LLM策略的季节性分析: {strategy}",
            'residual': f"基于LLM策略的残差分析: {strategy}",
            'coordinator': f"基于LLM策略的协调决策: {strategy}"
        }
        return templates
    
    def get_communication_statistics(self) -> Dict[str, Any]:
        """获取通信统计信息"""
        if not self.communication_history:
            return {'total_communications': 0, 'avg_rounds': 0}
        
        total_communications = len(self.communication_history)
        avg_rounds = np.mean([comm.get('rounds', 0) for comm in self.communication_history])
        
        return {
            'total_communications': total_communications,
            'avg_rounds': avg_rounds,
            'llm_optimization_enabled': self.enable_llm_optimization,
            'recent_strategies': [comm.get('strategy', '') for comm in self.communication_history[-5:]]
        }