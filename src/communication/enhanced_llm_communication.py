# -*- coding: utf-8 -*-
"""
增强版LLM驱动通信系统
集成阿里云百炼平台，提供更智能的多智能体协作
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import json

from src.llm.aliyun_qwen_interface import AliyunQwenInterface
from src.communication.t2mac_protocol import T2MACProtocol

class EnhancedLLMCommunication:
    """增强版LLM驱动通信系统"""
    
    def __init__(self, api_key: str, model: str = "qwen-turbo"):
        """
        初始化增强版通信系统
        
        Args:
            api_key: 阿里云百炼API密钥
            model: 模型名称
        """
        self.logger = logging.getLogger(__name__)
        
        # 初始化LLM接口
        self.llm_interface = AliyunQwenInterface(api_key, model)
        
        # 初始化T2MAC协议
        self.t2mac = T2MACProtocol(config={})
        
        # 通信状态
        self.communication_history = []
        self.agent_performance = {}
        self.fusion_strategies = {}
        
        # 统计信息
        self.total_communications = 0
        self.successful_communications = 0
        self.llm_calls = 0
        
    def intelligent_agent_coordination(self, agent_results: List[Dict], context: Dict) -> Dict:
        """
        智能智能体协调
        
        Args:
            agent_results: 智能体检测结果
            context: 上下文信息
            
        Returns:
            协调结果
        """
        self.logger.info("🤖 开始智能智能体协调")
        
        # 1. 分析各智能体结果
        analysis_result = self._analyze_agent_results(agent_results, context)
        
        # 2. 生成融合策略
        fusion_strategy = self._generate_fusion_strategy(agent_results)
        
        # 3. 协调智能体协作
        coordination_result = self._coordinate_agents(agent_results, analysis_result)
        
        # 4. 生成最终决策
        final_decision = self._generate_final_decision(
            agent_results, analysis_result, fusion_strategy, coordination_result
        )
        
        # 更新统计信息
        self.total_communications += 1
        if final_decision.get('success', False):
            self.successful_communications += 1
        
        # 记录通信历史
        self.communication_history.append({
            'timestamp': time.time(),
            'agent_results': agent_results,
            'coordination_result': coordination_result,
            'final_decision': final_decision
        })
        
        return final_decision
    
    def _analyze_agent_results(self, agent_results: List[Dict], context: Dict) -> Dict:
        """分析智能体结果"""
        self.logger.info("📊 分析智能体检测结果")
        
        # 提取分数和置信度
        scores = [result.get('score', 0) for result in agent_results]
        confidences = [result.get('confidence', 0) for result in agent_results]
        agent_names = [result.get('agent', f'Agent_{i}') for i, result in enumerate(agent_results)]
        
        # 使用LLM分析异常模式
        analysis_result = self.llm_interface.analyze_anomaly_patterns(agent_results, context)
        self.llm_calls += 1
        
        # 计算统计信息
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        
        # 一致性分析
        consistency = 1.0 - (std_score / max(mean_score, 0.001))
        
        return {
            'scores': scores,
            'confidences': confidences,
            'agent_names': agent_names,
            'mean_score': mean_score,
            'std_score': std_score,
            'max_score': max_score,
            'min_score': min_score,
            'consistency': consistency,
            'llm_analysis': analysis_result.get('analysis', ''),
            'timestamp': time.time()
        }
    
    def _generate_fusion_strategy(self, agent_results: List[Dict]) -> Dict:
        """生成融合策略"""
        self.logger.info("🔗 生成智能融合策略")
        
        # 提取分数和权重
        scores = [result.get('score', 0) for result in agent_results]
        weights = [result.get('weight', 1.0) for result in agent_results]
        
        # 使用LLM生成融合策略
        strategy_result = self.llm_interface.generate_fusion_strategy(scores, weights)
        self.llm_calls += 1
        
        # 计算自适应权重
        adaptive_weights = self._calculate_adaptive_weights(agent_results)
        
        # 计算融合分数
        fused_score = np.average(scores, weights=adaptive_weights)
        
        return {
            'original_weights': weights,
            'adaptive_weights': adaptive_weights,
            'fused_score': fused_score,
            'strategy_advice': strategy_result.get('strategy', ''),
            'confidence': strategy_result.get('confidence', 0.5),
            'timestamp': time.time()
        }
    
    def _calculate_adaptive_weights(self, agent_results: List[Dict]) -> List[float]:
        """计算自适应权重"""
        weights = []
        
        for result in agent_results:
            # 基于置信度和历史性能调整权重
            confidence = result.get('confidence', 0.5)
            agent_name = result.get('agent', 'Unknown')
            
            # 获取历史性能
            historical_performance = self.agent_performance.get(agent_name, 0.5)
            
            # 计算自适应权重
            adaptive_weight = confidence * historical_performance
            weights.append(adaptive_weight)
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        return weights
    
    def _coordinate_agents(self, agent_results: List[Dict], analysis_result: Dict) -> Dict:
        """协调智能体协作"""
        self.logger.info("🤝 协调智能体协作")
        
        # 构建智能体消息
        agent_messages = []
        for i, result in enumerate(agent_results):
            agent_name = result.get('agent', f'Agent_{i}')
            score = result.get('score', 0)
            confidence = result.get('confidence', 0)
            
            message = {
                'agent': agent_name,
                'content': f'检测分数: {score:.4f}, 置信度: {confidence:.4f}'
            }
            agent_messages.append(message)
        
        # 使用LLM进行多智能体通信
        context = f"一致性: {analysis_result.get('consistency', 0):.4f}, 平均分数: {analysis_result.get('mean_score', 0):.4f}"
        comm_result = self.llm_interface.multi_agent_communication(agent_messages, context)
        self.llm_calls += 1
        
        return {
            'communication_success': comm_result.get('success', False),
            'coordination_advice': comm_result.get('response', ''),
            'agent_messages': agent_messages,
            'timestamp': time.time()
        }
    
    def _generate_final_decision(self, agent_results: List[Dict], analysis_result: Dict, 
                                fusion_strategy: Dict, coordination_result: Dict) -> Dict:
        """生成最终决策"""
        self.logger.info("🎯 生成最终异常检测决策")
        
        # 计算最终异常分数
        fused_score = fusion_strategy.get('fused_score', 0)
        consistency = analysis_result.get('consistency', 0)
        confidence = fusion_strategy.get('confidence', 0.5)
        
        # 基于一致性和置信度调整最终分数
        adjusted_score = fused_score * (0.7 + 0.3 * consistency) * (0.5 + 0.5 * confidence)
        
        # 确定异常阈值
        threshold = self._calculate_dynamic_threshold(agent_results, analysis_result)
        
        # 判断是否异常
        is_anomaly = adjusted_score > threshold
        
        # 计算风险等级
        risk_level = self._calculate_risk_level(adjusted_score, threshold)
        
        # 生成解释
        explanation = self._generate_explanation(
            agent_results, analysis_result, fusion_strategy, 
            coordination_result, adjusted_score, threshold, is_anomaly
        )
        
        return {
            'success': True,
            'anomaly_score': adjusted_score,
            'threshold': threshold,
            'is_anomaly': is_anomaly,
            'risk_level': risk_level,
            'confidence': confidence,
            'consistency': consistency,
            'explanation': explanation,
            'fusion_strategy': fusion_strategy,
            'coordination_result': coordination_result,
            'timestamp': time.time()
        }
    
    def _calculate_dynamic_threshold(self, agent_results: List[Dict], analysis_result: Dict) -> float:
        """计算动态异常阈值"""
        # 基于历史数据和一致性调整阈值
        base_threshold = 0.5
        consistency = analysis_result.get('consistency', 0.5)
        std_score = analysis_result.get('std_score', 0.1)
        
        # 一致性越高，阈值越严格
        consistency_factor = 1.0 - 0.2 * consistency
        
        # 标准差越大，阈值越宽松
        std_factor = 1.0 + 0.1 * std_score
        
        dynamic_threshold = base_threshold * consistency_factor * std_factor
        
        return max(0.1, min(0.9, dynamic_threshold))
    
    def _calculate_risk_level(self, score: float, threshold: float) -> str:
        """计算风险等级"""
        ratio = score / max(threshold, 0.001)
        
        if ratio < 0.5:
            return "低风险"
        elif ratio < 1.0:
            return "中风险"
        elif ratio < 1.5:
            return "高风险"
        else:
            return "极高风险"
    
    def _generate_explanation(self, agent_results: List[Dict], analysis_result: Dict,
                            fusion_strategy: Dict, coordination_result: Dict,
                            final_score: float, threshold: float, is_anomaly: bool) -> str:
        """生成决策解释"""
        explanation_parts = []
        
        # 基本决策信息
        explanation_parts.append(f"最终异常分数: {final_score:.4f} (阈值: {threshold:.4f})")
        explanation_parts.append(f"检测结果: {'异常' if is_anomaly else '正常'}")
        
        # 智能体贡献分析
        explanation_parts.append("\n智能体贡献分析:")
        for i, result in enumerate(agent_results):
            agent_name = result.get('agent', f'Agent_{i}')
            score = result.get('score', 0)
            weight = fusion_strategy.get('adaptive_weights', [1.0] * len(agent_results))[i]
            explanation_parts.append(f"  {agent_name}: 分数={score:.4f}, 权重={weight:.4f}")
        
        # 一致性分析
        consistency = analysis_result.get('consistency', 0)
        explanation_parts.append(f"\n一致性分析: {consistency:.4f}")
        
        # LLM分析结果
        llm_analysis = analysis_result.get('llm_analysis', '')
        if llm_analysis:
            explanation_parts.append(f"\n智能分析: {llm_analysis[:200]}...")
        
        return "\n".join(explanation_parts)
    
    def update_agent_performance(self, agent_name: str, performance_score: float):
        """更新智能体性能"""
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = performance_score
        else:
            # 指数移动平均
            alpha = 0.1
            self.agent_performance[agent_name] = (
                alpha * performance_score + 
                (1 - alpha) * self.agent_performance[agent_name]
            )
    
    def get_communication_statistics(self) -> Dict:
        """获取通信统计信息"""
        llm_stats = self.llm_interface.get_statistics()
        
        return {
            'total_communications': self.total_communications,
            'successful_communications': self.successful_communications,
            'success_rate': self.successful_communications / max(1, self.total_communications),
            'llm_calls': self.llm_calls,
            'llm_statistics': llm_stats,
            'agent_performance': self.agent_performance,
            'communication_history_length': len(self.communication_history)
        }
    
    def save_communication_log(self, filepath: str):
        """保存通信日志"""
        log_data = {
            'statistics': self.get_communication_statistics(),
            'communication_history': self.communication_history[-100:],  # 保存最近100条
            'agent_performance': self.agent_performance,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

# 测试函数
def test_enhanced_communication():
    """测试增强版通信系统"""
    # 注意：需要替换为实际的API密钥
    api_key = "sk-dc7f3086d0564eb6ac282c7d66faea12"
    
    # 创建增强版通信系统
    comm_system = EnhancedLLMCommunication(api_key)
    
    # 模拟智能体结果
    agent_results = [
        {'agent': 'TrendAgent', 'score': 0.8, 'confidence': 0.9, 'weight': 1.0},
        {'agent': 'VarianceAgent', 'score': 0.3, 'confidence': 0.7, 'weight': 1.0},
        {'agent': 'ResidualAgent', 'score': 0.7, 'confidence': 0.8, 'weight': 1.0},
        {'agent': 'StatisticalAgent', 'score': 0.6, 'confidence': 0.6, 'weight': 1.0}
    ]
    
    context = {
        'dataset': 'MSL',
        'features': 55,
        'samples': 73729,
        'window_size': 100
    }
    
    # 测试智能协调
    print("测试增强版多智能体通信系统...")
    result = comm_system.intelligent_agent_coordination(agent_results, context)
    
    print(f"协调结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    # 显示统计信息
    stats = comm_system.get_communication_statistics()
    print(f"\n统计信息: {json.dumps(stats, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    test_enhanced_communication()
