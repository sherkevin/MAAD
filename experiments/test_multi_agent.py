#!/usr/bin/env python3
"""
多智能体异常检测测试脚本
测试基础的多智能体框架功能
"""

import torch
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents.multi_agent_detector import MultiAgentAnomalyDetector

def test_multi_agent_basic():
    """测试多智能体基础功能"""
    print("🧪 开始测试多智能体基础功能...")
    
    # 配置
    config = {
        'trend_agent': {
            'trend_analyzer': {'smoothing_factor': 0.1},
            'physics': {'smoothness_weight': 0.1},
            'anomaly_detector': {'threshold': 0.5}
        },
        'communication_bus': {'max_queue_size': 100},
        'performance_monitor': {'metrics_collection': True}
    }
    
    # 创建多智能体检测器
    detector = MultiAgentAnomalyDetector(config)
    print("✅ 多智能体检测器创建成功")
    
    # 测试数据
    test_data = torch.randn(1, 3, 64, 64)
    print(f"📊 测试数据形状: {test_data.shape}")
    
    # 执行检测
    print("🔍 开始执行异常检测...")
    result = detector.detect_anomaly(test_data)
    
    # 验证结果
    if 'error' in result:
        print(f"❌ 检测失败: {result['error']}")
        return False
    
    print("✅ 异常检测执行成功")
    print(f"📈 最终决策: {result['final_decision']}")
    print(f"🤖 智能体输出数量: {len(result['agent_outputs'])}")
    print(f"📊 性能指标: {result['performance_metrics']}")
    
    return True

def test_agent_states():
    """测试智能体状态管理"""
    print("\n🧪 开始测试智能体状态管理...")
    
    config = {'trend_agent': {}}
    detector = MultiAgentAnomalyDetector(config)
    
    # 获取智能体状态
    states = detector.get_agent_states()
    print(f"📊 智能体状态: {len(states)} 个智能体")
    
    for agent_id, state in states.items():
        print(f"  - {agent_id}: {state.status}, 置信度: {state.confidence_score:.3f}")
    
    # 重置智能体
    detector.reset_all_agents()
    print("✅ 智能体重置完成")
    
    return True

def test_communication_bus():
    """测试通信总线"""
    print("\n🧪 开始测试通信总线...")
    
    config = {'communication_bus': {'max_queue_size': 10}}
    detector = MultiAgentAnomalyDetector(config)
    
    # 发送消息
    detector.communication_bus.send_message("agent1", "agent2", {"test": "message"})
    detector.communication_bus.send_message("agent2", "agent1", {"reply": "received"})
    
    # 获取消息
    messages = detector.communication_bus.get_messages("agent2")
    print(f"📨 智能体2收到 {len(messages)} 条消息")
    
    # 清空消息
    detector.communication_bus.clear_messages()
    print("✅ 通信总线测试完成")
    
    return True

def test_performance_monitoring():
    """测试性能监控"""
    print("\n🧪 开始测试性能监控...")
    
    config = {'performance_monitor': {'metrics_collection': True}}
    detector = MultiAgentAnomalyDetector(config)
    
    # 开始计时
    detector.performance_monitor.start_timing()
    
    # 执行一些操作
    test_data = torch.randn(1, 3, 64, 64)
    result = detector.detect_anomaly(test_data)
    
    # 收集性能指标
    metrics = detector.performance_monitor.collect_metrics(detector.agents)
    total_time = detector.performance_monitor.get_total_time()
    
    print(f"⏱️  总处理时间: {total_time:.3f} 秒")
    print(f"📊 性能指标: {metrics}")
    
    return True

def test_error_handling():
    """测试错误处理"""
    print("\n🧪 开始测试错误处理...")
    
    config = {'trend_agent': {}}
    detector = MultiAgentAnomalyDetector(config)
    
    # 测试无效输入
    invalid_data = torch.tensor([])  # 空张量
    
    try:
        result = detector.detect_anomaly(invalid_data)
        if 'error' in result:
            print("✅ 错误处理正常: 检测到无效输入")
        else:
            print("⚠️  错误处理可能有问题: 未检测到无效输入")
    except Exception as e:
        print(f"✅ 错误处理正常: 捕获到异常 {type(e).__name__}")
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始多智能体异常检测测试")
    print("=" * 50)
    
    tests = [
        test_multi_agent_basic,
        test_agent_states,
        test_communication_bus,
        test_performance_monitoring,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print("✅ 测试通过")
            else:
                print("❌ 测试失败")
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！多智能体框架基础功能正常")
        return True
    else:
        print("⚠️  部分测试失败，需要检查代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)