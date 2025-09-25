# -*- coding: utf-8 -*-
"""
阿里云百炼平台Qwen LLM接口
支持多智能体通信和协作
"""

import requests
import json
import logging
import time
from typing import Dict, List, Optional, Any
import numpy as np

class AliyunQwenInterface:
    """阿里云百炼平台Qwen LLM接口"""
    
    def __init__(self, api_key: str, model: str = "qwen-turbo", base_url: str = None):
        """
        初始化阿里云百炼接口
        
        Args:
            api_key: API密钥
            model: 模型名称，默认qwen-turbo
            base_url: API基础URL
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.logger = logging.getLogger(__name__)
        
        # 设置请求头
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # 统计信息
        self.request_count = 0
        self.total_tokens = 0
        self.error_count = 0
        
    def _make_request(self, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.7) -> Dict:
        """
        发送请求到阿里云百炼API
        
        Args:
            messages: 消息列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            
        Returns:
            API响应结果
        """
        data = {
            'model': self.model,
            'input': {
                'messages': messages
            },
            'parameters': {
                'max_tokens': max_tokens,
                'temperature': temperature
            }
        }
        
        try:
            self.request_count += 1
            response = requests.post(
                self.base_url, 
                headers=self.headers, 
                json=data, 
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # 更新token统计
                if 'usage' in result:
                    self.total_tokens += result['usage'].get('total_tokens', 0)
                return result
            else:
                self.error_count += 1
                self.logger.error(f"API请求失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"API请求异常: {e}")
            return None
    
    def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        生成文本响应
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            
        Returns:
            生成的文本
        """
        messages = [{'role': 'user', 'content': prompt}]
        result = self._make_request(messages, max_tokens, temperature)
        
        if result and 'output' in result:
            return result['output'].get('text', '')
        return ""
    
    def multi_agent_communication(self, agent_messages: List[Dict], context: str = "") -> Dict:
        """
        多智能体通信处理
        
        Args:
            agent_messages: 智能体消息列表
            context: 上下文信息
            
        Returns:
            通信结果
        """
        # 构建多智能体对话
        messages = []
        
        # 添加系统提示
        system_prompt = f"""
你是一个多智能体异常检测系统的协调者。你的任务是：
1. 分析各个智能体的检测结果
2. 识别异常模式和趋势
3. 提供综合的异常检测建议
4. 协调智能体之间的协作

上下文信息：{context}

请根据以下智能体的消息进行分析和协调：
"""
        messages.append({'role': 'system', 'content': system_prompt})
        
        # 添加智能体消息
        for i, msg in enumerate(agent_messages):
            agent_name = msg.get('agent', f'Agent_{i+1}')
            content = msg.get('content', '')
            messages.append({'role': 'user', 'content': f'{agent_name}: {content}'})
        
        # 生成协调响应
        result = self._make_request(messages, max_tokens=1500, temperature=0.5)
        
        if result and 'output' in result:
            return {
                'success': True,
                'response': result['output'].get('text', ''),
                'usage': result.get('usage', {}),
                'request_id': result.get('request_id', '')
            }
        else:
            return {
                'success': False,
                'response': '',
                'error': 'API请求失败'
            }
    
    def analyze_anomaly_patterns(self, detection_results: List[Dict], data_context: Dict) -> Dict:
        """
        分析异常模式
        
        Args:
            detection_results: 检测结果列表
            data_context: 数据上下文
            
        Returns:
            分析结果
        """
        # 构建分析提示
        prompt = f"""
作为异常检测专家，请分析以下检测结果：

数据上下文：
- 数据集：{data_context.get('dataset', 'Unknown')}
- 特征维度：{data_context.get('features', 'Unknown')}
- 样本数量：{data_context.get('samples', 'Unknown')}

检测结果：
"""
        
        for i, result in enumerate(detection_results):
            agent_name = result.get('agent', f'Agent_{i+1}')
            score = result.get('score', 0)
            confidence = result.get('confidence', 0)
            prompt += f"\n{agent_name}: 异常分数={score:.4f}, 置信度={confidence:.4f}"
        
        prompt += """

请提供：
1. 异常模式分析
2. 风险等级评估
3. 建议的后续行动
4. 各智能体结果的一致性分析
"""
        
        response = self.generate_response(prompt, max_tokens=1000, temperature=0.3)
        
        return {
            'analysis': response,
            'timestamp': time.time(),
            'agent_count': len(detection_results)
        }
    
    def generate_fusion_strategy(self, agent_scores: List[float], agent_weights: List[float] = None) -> Dict:
        """
        生成融合策略建议
        
        Args:
            agent_scores: 智能体分数列表
            agent_weights: 智能体权重列表
            
        Returns:
            融合策略
        """
        if agent_weights is None:
            agent_weights = [1.0] * len(agent_scores)
        
        # 计算加权平均
        weighted_avg = np.average(agent_scores, weights=agent_weights)
        
        # 计算方差
        variance = np.var(agent_scores)
        
        # 生成策略建议
        prompt = f"""
基于以下智能体检测结果，请提供融合策略建议：

智能体分数：{agent_scores}
智能体权重：{agent_weights}
加权平均：{weighted_avg:.4f}
分数方差：{variance:.4f}

请提供：
1. 最优融合权重建议
2. 异常阈值建议
3. 置信度评估
4. 是否需要重新调整智能体权重
"""
        
        response = self.generate_response(prompt, max_tokens=800, temperature=0.2)
        
        return {
            'strategy': response,
            'weighted_average': weighted_avg,
            'variance': variance,
            'recommended_weights': agent_weights,
            'confidence': min(1.0, max(0.0, 1.0 - variance))
        }
    
    def get_statistics(self) -> Dict:
        """获取接口统计信息"""
        return {
            'request_count': self.request_count,
            'total_tokens': self.total_tokens,
            'error_count': self.error_count,
            'success_rate': (self.request_count - self.error_count) / max(1, self.request_count),
            'model': self.model
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.request_count = 0
        self.total_tokens = 0
        self.error_count = 0

# 测试函数
def test_aliyun_qwen_interface():
    """测试阿里云百炼接口"""
    # 注意：这里需要替换为实际的API密钥
    api_key = "sk-dc7f3086d0564eb6ac282c7d66faea12"
    
    # 创建接口实例
    qwen = AliyunQwenInterface(api_key)
    
    # 测试基本功能
    print("测试基本文本生成...")
    response = qwen.generate_response("请简单介绍一下多智能体系统")
    print(f"响应: {response}")
    
    # 测试多智能体通信
    print("\n测试多智能体通信...")
    agent_messages = [
        {'agent': 'TrendAgent', 'content': '检测到上升趋势异常，分数0.8'},
        {'agent': 'VarianceAgent', 'content': '方差波动正常，分数0.3'},
        {'agent': 'ResidualAgent', 'content': '残差分析显示异常，分数0.7'}
    ]
    
    comm_result = qwen.multi_agent_communication(agent_messages, "MSL数据集异常检测")
    print(f"通信结果: {comm_result}")
    
    # 测试异常模式分析
    print("\n测试异常模式分析...")
    detection_results = [
        {'agent': 'TrendAgent', 'score': 0.8, 'confidence': 0.9},
        {'agent': 'VarianceAgent', 'score': 0.3, 'confidence': 0.7},
        {'agent': 'ResidualAgent', 'score': 0.7, 'confidence': 0.8}
    ]
    
    data_context = {
        'dataset': 'MSL',
        'features': 55,
        'samples': 73729
    }
    
    analysis_result = qwen.analyze_anomaly_patterns(detection_results, data_context)
    print(f"分析结果: {analysis_result}")
    
    # 显示统计信息
    print(f"\n统计信息: {qwen.get_statistics()}")

if __name__ == "__main__":
    test_aliyun_qwen_interface()
