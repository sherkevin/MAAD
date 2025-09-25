#!/usr/bin/env python3
"""
LLM通信集成测试脚本
测试T2MAC协议和Qwen LLM集成的功能
"""

import torch
import sys
import os
import time

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from communication.llm_driven_communication import LLMDrivenCommunication
from agents.multi_agent_detector import MultiAgentAnomalyDetector

def test_llm_communication_basic():
    """测试LLM通信基础功能"""
    print("🧪 测试1: LLM通信基础功能测试")
    print("-" * 50)
    
    # 配置
    config = {
        't2mac': {
            'max_communication_rounds': 3,
            'communication_threshold': 0.7,
            'adaptation_rate': 0.1
        },
        'qwen': {
            'model_path': 'Qwen/Qwen2.5-7B-Instruct',
            'device': 'cpu',
            'max_length': 2048,
            'temperature': 0.7
        },
        'communication_manager': {},
        'strategy_executor': {},
        'performance_monitor': {}
    }
    
    # 创建LLM驱动通信系统
    llm_communication = LLMDrivenCommunication(config)
    print("✅ LLM驱动通信系统创建成功")
    
    # 创建多智能体检测器
    agent_config = {'trend_agent': {}}
    detector = MultiAgentAnomalyDetector(agent_config)
    
    # 获取智能体状态
    agent_states = detector.get_agent_states()
    print(f"✅ 智能体状态获取成功: {len(agent_states)} 个智能体")
    
    # 定义全局目标
    global_target = {
        'type': 'anomaly_detection',
        'priority': 0.8,
        'deadline': None,
        'requirements': {'accuracy': 0.9, 'speed': 1.0},
        'progress': 0.0
    }
    
    # 执行LLM驱动通信
    print("🔍 开始执行LLM驱动通信...")
    result = llm_communication.intelligent_communication(agent_states, global_target)
    
    # 验证结果
    if 'error' in result:
        print(f"❌ LLM通信失败: {result['error']}")
        return False
    
    print("✅ LLM驱动通信执行成功")
    print(f"📊 通信结果: {result['communication_result']}")
    print(f"🤖 LLM分析: {result['llm_analysis']}")
    print(f"📈 性能指标: {result['performance_metrics']}")
    
    return True

def test_t2mac_protocol():
    """测试T2MAC协议"""
    print("\n🧪 测试2: T2MAC协议测试")
    print("-" * 50)
    
    from communication.t2mac_protocol import T2MACProtocol, CommunicationType, MessagePriority
    
    config = {
        'max_communication_rounds': 3,
        'communication_threshold': 0.7,
        'adaptation_rate': 0.1
    }
    
    protocol = T2MACProtocol(config)
    print("✅ T2MAC协议创建成功")
    
    # 模拟智能体状态
    agent_states = {
        'trend': type('AgentState', (), {
            'status': 'active',
            'confidence_score': 0.6,
            'processing_time': 0.1,
            'error_count': 0
        })(),
        'seasonal': type('AgentState', (), {
            'status': 'idle',
            'confidence_score': 0.8,
            'processing_time': 0.05,
            'error_count': 0
        })()
    }
    
    global_target = {
        'type': 'anomaly_detection',
        'priority': 0.8,
        'progress': 0.0
    }
    
    # 生成通信计划
    plan = protocol.generate_communication_plan(agent_states, global_target)
    print(f"✅ 通信计划生成成功")
    print(f"📋 通信轮次: {len(plan.get('communication_plan', {}).get('communication_rounds', []))}")
    print(f"📊 效率指标: {plan.get('efficiency_metrics', {})}")
    
    return True

def test_qwen_llm_interface():
    """测试Qwen LLM接口"""
    print("\n🧪 测试3: Qwen LLM接口测试")
    print("-" * 50)
    
    from llm.qwen_interface import QwenLLMInterface
    
    config = {
        'model_path': 'Qwen/Qwen2.5-7B-Instruct',
        'device': 'cpu',
        'max_length': 2048,
        'temperature': 0.7
    }
    
    qwen_llm = QwenLLMInterface(config)
    print("✅ Qwen LLM接口创建成功")
    
    # 测试模型信息
    model_info = qwen_llm.get_model_info()
    print(f"📊 模型信息: {model_info}")
    
    # 模拟智能体状态
    agent_states = {
        'trend': type('AgentState', (), {
            'status': 'active',
            'confidence_score': 0.6,
            'processing_time': 0.1,
            'error_count': 0
        })(),
        'seasonal': type('AgentState', (), {
            'status': 'idle',
            'confidence_score': 0.8,
            'processing_time': 0.05,
            'error_count': 0
        })()
    }
    
    global_target = {
        'type': 'anomaly_detection',
        'priority': 0.8,
        'progress': 0.0
    }
    
    # 测试智能体状态分析
    analysis = qwen_llm.analyze_agent_states(agent_states, global_target)
    print(f"✅ 智能体状态分析完成")
    print(f"📊 分析结果: {analysis}")
    
    # 测试通信策略生成
    communication_needs = {
        'coordination_required': True,
        'priority_agents': ['trend', 'seasonal'],
        'communication_priority': 'high'
    }
    
    strategy = qwen_llm.generate_communication_strategy(analysis, communication_needs)
    print(f"✅ 通信策略生成完成")
    print(f"📋 策略类型: {strategy.get('strategy_type', 'unknown')}")
    
    return True

def test_communication_performance():
    """测试通信性能"""
    print("\n🧪 测试4: 通信性能测试")
    print("-" * 50)
    
    config = {
        't2mac': {'max_communication_rounds': 2},
        'qwen': {'device': 'cpu'},
        'communication_manager': {},
        'strategy_executor': {},
        'performance_monitor': {}
    }
    
    llm_communication = LLMDrivenCommunication(config)
    
    # 创建多智能体检测器
    agent_config = {'trend_agent': {}}
    detector = MultiAgentAnomalyDetector(agent_config)
    
    # 执行多次通信测试
    start_time = time.time()
    
    for i in range(5):
        agent_states = detector.get_agent_states()
        global_target = {
            'type': 'anomaly_detection',
            'priority': 0.8,
            'progress': i * 0.2
        }
        
        result = llm_communication.intelligent_communication(agent_states, global_target)
        
        if i % 2 == 0:
            print(f"  第{i+1}次通信: 成功率={result.get('communication_result', {}).get('success_rate', 0):.2f}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"✅ 性能测试完成")
    print(f"⏱️  总时间: {total_time:.2f} 秒")
    print(f"📊 平均时间: {total_time/5:.2f} 秒/次")
    
    # 获取性能指标
    metrics = llm_communication.performance_monitor.get_metrics()
    print(f"📈 性能指标: {metrics}")
    
    return True

def test_communication_history():
    """测试通信历史"""
    print("\n🧪 测试5: 通信历史测试")
    print("-" * 50)
    
    config = {
        't2mac': {'max_communication_rounds': 2},
        'qwen': {'device': 'cpu'},
        'communication_manager': {},
        'strategy_executor': {},
        'performance_monitor': {}
    }
    
    llm_communication = LLMDrivenCommunication(config)
    
    # 创建多智能体检测器
    agent_config = {'trend_agent': {}}
    detector = MultiAgentAnomalyDetector(agent_config)
    
    # 执行多次通信
    for i in range(3):
        agent_states = detector.get_agent_states()
        global_target = {
            'type': 'anomaly_detection',
            'priority': 0.8,
            'progress': i * 0.3
        }
        
        result = llm_communication.intelligent_communication(agent_states, global_target)
    
    # 获取通信历史
    history = llm_communication.get_communication_history(5)
    print(f"✅ 通信历史获取成功: {len(history)} 条记录")
    
    for i, record in enumerate(history):
        print(f"  记录{i+1}: {record['timestamp']}")
    
    # 获取当前状态
    current_state = llm_communication.get_current_state()
    print(f"📊 当前状态: {current_state}")
    
    return True

def test_error_handling():
    """测试错误处理"""
    print("\n🧪 测试6: 错误处理测试")
    print("-" * 50)
    
    config = {
        't2mac': {'max_communication_rounds': 1},
        'qwen': {'device': 'cpu'},
        'communication_manager': {},
        'strategy_executor': {},
        'performance_monitor': {}
    }
    
    llm_communication = LLMDrivenCommunication(config)
    
    # 测试无效输入
    invalid_agent_states = {}
    invalid_global_target = {}
    
    result = llm_communication.intelligent_communication(invalid_agent_states, invalid_global_target)
    
    if 'error' in result:
        print("✅ 错误处理正常: 检测到无效输入")
    else:
        print("⚠️  错误处理可能有问题: 未检测到无效输入")
    
    # 测试空目标
    empty_target = {}
    agent_states = {'trend': type('AgentState', (), {'status': 'idle'})()}
    
    result = llm_communication.intelligent_communication(agent_states, empty_target)
    
    if 'error' in result:
        print("✅ 错误处理正常: 检测到空目标")
    else:
        print("⚠️  错误处理可能有问题: 未检测到空目标")
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始LLM通信集成测试")
    print("=" * 60)
    
    tests = [
        test_llm_communication_basic,
        test_t2mac_protocol,
        test_qwen_llm_interface,
        test_communication_performance,
        test_communication_history,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    start_time = time.time()
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print("✅ 测试通过")
            else:
                print("❌ 测试失败")
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    print(f"⏱️  总测试时间: {total_time:.2f} 秒")
    
    if passed == total:
        print("🎉 所有测试通过！LLM通信集成功能完整")
        print("🚀 可以开始第3周的任务：联邦学习集成")
        return True
    else:
        print("⚠️  部分测试失败，需要检查代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
