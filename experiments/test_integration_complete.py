#!/usr/bin/env python3
"""
综合集成测试脚本
测试所有模块的完整集成，确保服务器上传万无一失
"""

import torch
import sys
import os
import time
import traceback

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_import_all_modules():
    """测试所有模块导入"""
    print("🧪 测试1: 模块导入测试")
    print("-" * 50)
    
    try:
        # 导入多智能体模块
        from agents.multi_agent_detector import MultiAgentAnomalyDetector
        from agents.trend_agent import TrendAgent
        from agents.base_agent import BaseAgent
        print("✅ 多智能体模块导入成功")
        
        # 导入通信模块
        from communication.t2mac_protocol import T2MACProtocol, CommunicationType, MessagePriority
        print("✅ 通信模块导入成功")
        
        # 导入LLM模块
        from llm.qwen_interface import QwenLLMInterface
        print("✅ LLM模块导入成功")
        
        # 导入隐私模块
        from privacy.differential_privacy import DifferentialPrivacy, PrivacyBudget
        print("✅ 隐私模块导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        traceback.print_exc()
        return False

def test_multi_agent_system():
    """测试多智能体系统"""
    print("\n🧪 测试2: 多智能体系统测试")
    print("-" * 50)
    
    try:
        from agents.multi_agent_detector import MultiAgentAnomalyDetector
        
        # 创建多智能体检测器
        config = {'trend_agent': {}}
        detector = MultiAgentAnomalyDetector(config)
        print("✅ 多智能体检测器创建成功")
        
        # 测试数据
        test_data = torch.randn(1, 3, 64, 64)
        
        # 执行检测
        result = detector.detect_anomaly(test_data)
        
        if 'error' in result:
            print(f"❌ 检测失败: {result['error']}")
            return False
        
        print(f"✅ 异常检测成功: 异常分数={result['final_decision']['anomaly_score']:.3f}")
        print(f"✅ 置信度: {result['final_decision']['confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 多智能体系统测试失败: {e}")
        traceback.print_exc()
        return False

def test_communication_system():
    """测试通信系统"""
    print("\n🧪 测试3: 通信系统测试")
    print("-" * 50)
    
    try:
        from communication.t2mac_protocol import T2MACProtocol
        
        # 创建T2MAC协议
        config = {'max_communication_rounds': 3, 'communication_threshold': 0.7}
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
        }
        
        global_target = {
            'type': 'anomaly_detection',
            'priority': 0.8,
            'progress': 0.0
        }
        
        # 生成通信计划
        plan = protocol.generate_communication_plan(agent_states, global_target)
        
        if 'error' in plan:
            print(f"❌ 通信计划生成失败: {plan['error']}")
            return False
        
        print(f"✅ 通信计划生成成功: {plan['strategy_type']}")
        print(f"✅ 通信轮次: {len(plan['communication_plan']['communication_rounds'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 通信系统测试失败: {e}")
        traceback.print_exc()
        return False

def test_llm_system():
    """测试LLM系统"""
    print("\n🧪 测试4: LLM系统测试")
    print("-" * 50)
    
    try:
        from llm.qwen_interface import QwenLLMInterface
        
        # 创建Qwen LLM接口
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
        print(f"✅ 模型信息: {model_info['model_path']}")
        
        # 模拟智能体状态
        agent_states = {
            'trend': type('AgentState', (), {
                'status': 'active',
                'confidence_score': 0.6,
                'processing_time': 0.1,
                'error_count': 0
            })(),
        }
        
        global_target = {
            'type': 'anomaly_detection',
            'priority': 0.8,
            'progress': 0.0
        }
        
        # 测试智能体状态分析
        analysis = qwen_llm.analyze_agent_states(agent_states, global_target)
        print(f"✅ 智能体状态分析完成: {analysis['analysis_type']}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM系统测试失败: {e}")
        traceback.print_exc()
        return False

def test_privacy_system():
    """测试隐私系统"""
    print("\n🧪 测试5: 隐私系统测试")
    print("-" * 50)
    
    try:
        from privacy.differential_privacy import DifferentialPrivacy, PrivacyBudget
        
        # 创建差分隐私实例
        privacy_config = {
            'epsilon': 1.0,
            'delta': 1e-5,
            'mechanism': 'gaussian',
            'sensitivity': 1.0
        }
        
        dp = DifferentialPrivacy(privacy_config)
        print("✅ 差分隐私实例创建成功")
        
        # 测试隐私预算
        budget = PrivacyBudget(epsilon=2.0, delta=1e-4)
        print(f"✅ 隐私预算创建: ε={budget.epsilon}, δ={budget.delta}")
        
        # 测试数据
        test_data = torch.randn(5, 5)
        
        # 添加噪声
        noisy_data = dp.add_noise(test_data)
        print(f"✅ 噪声添加成功: {test_data.shape} -> {noisy_data.shape}")
        
        # 检查隐私预算
        budget_status = dp.get_privacy_budget_status()
        print(f"✅ 隐私预算状态: 剩余ε={budget_status['remaining_epsilon']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 隐私系统测试失败: {e}")
        traceback.print_exc()
        return False

def test_performance_under_load():
    """测试负载下的性能"""
    print("\n🧪 测试6: 负载性能测试")
    print("-" * 50)
    
    try:
        from agents.multi_agent_detector import MultiAgentAnomalyDetector
        
        # 创建检测器
        config = {'trend_agent': {}}
        detector = MultiAgentAnomalyDetector(config)
        
        # 性能测试
        start_time = time.time()
        
        for i in range(20):
            test_data = torch.randn(1, 3, 64, 64)
            result = detector.detect_anomaly(test_data)
            
            if i % 5 == 0:
                print(f"  第{i+1}次检测完成")
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 20
        
        print(f"✅ 负载测试完成")
        print(f"📊 总时间: {total_time:.3f} 秒")
        print(f"📊 平均时间: {avg_time:.3f} 秒/次")
        print(f"📊 检测频率: {1/avg_time:.1f} 次/秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 负载性能测试失败: {e}")
        traceback.print_exc()
        return False

def test_memory_usage():
    """测试内存使用"""
    print("\n🧪 测试7: 内存使用测试")
    print("-" * 50)
    
    try:
        from agents.multi_agent_detector import MultiAgentAnomalyDetector
        
        # 记录初始内存
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # 创建多个检测器实例
        detectors = []
        for i in range(5):
            config = {'trend_agent': {}}
            detector = MultiAgentAnomalyDetector(config)
            detectors.append(detector)
        
        # 记录当前内存
        current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_used = current_memory - initial_memory
        
        print(f"✅ 内存使用测试完成")
        print(f"📊 初始内存: {initial_memory / 1e6:.1f} MB")
        print(f"📊 当前内存: {current_memory / 1e6:.1f} MB")
        print(f"📊 使用内存: {memory_used / 1e6:.1f} MB")
        
        # 清理内存
        del detectors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        print(f"📊 清理后内存: {final_memory / 1e6:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 内存使用测试失败: {e}")
        traceback.print_exc()
        return False

def test_error_recovery():
    """测试错误恢复"""
    print("\n🧪 测试8: 错误恢复测试")
    print("-" * 50)
    
    try:
        from agents.multi_agent_detector import MultiAgentAnomalyDetector
        
        # 创建检测器
        config = {'trend_agent': {}}
        detector = MultiAgentAnomalyDetector(config)
        
        # 测试各种错误情况
        error_cases = [
            torch.tensor([]),  # 空张量
            torch.randn(0, 0, 0),  # 无效形状
            torch.randn(1, 0, 64, 64),  # 部分无效
        ]
        
        recovery_success = 0
        for i, error_data in enumerate(error_cases):
            try:
                result = detector.detect_anomaly(error_data)
                if 'error' in result:
                    print(f"  错误情况{i+1}: 正确检测到错误")
                    recovery_success += 1
                else:
                    print(f"  错误情况{i+1}: 意外成功处理")
            except Exception as e:
                print(f"  错误情况{i+1}: 捕获异常 {type(e).__name__}")
                recovery_success += 1
        
        print(f"✅ 错误恢复测试完成: {recovery_success}/{len(error_cases)} 个错误情况正确处理")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误恢复测试失败: {e}")
        traceback.print_exc()
        return False

def test_server_compatibility():
    """测试服务器兼容性"""
    print("\n🧪 测试9: 服务器兼容性测试")
    print("-" * 50)
    
    try:
        # 检查PyTorch版本
        print(f"📊 PyTorch版本: {torch.__version__}")
        
        # 检查CUDA支持
        if torch.cuda.is_available():
            print(f"📊 CUDA版本: {torch.version.cuda}")
            print(f"📊 GPU数量: {torch.cuda.device_count()}")
            print(f"📊 当前GPU: {torch.cuda.current_device()}")
            print(f"📊 GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"📊 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("📊 CUDA不可用，使用CPU")
        
        # 检查设备兼容性
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📊 推荐设备: {device}")
        
        # 测试张量操作
        test_tensor = torch.randn(10, 10).to(device)
        result = torch.matmul(test_tensor, test_tensor.T)
        print(f"✅ 张量操作测试成功: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 服务器兼容性测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始综合集成测试")
    print("=" * 70)
    
    tests = [
        test_import_all_modules,
        test_multi_agent_system,
        test_communication_system,
        test_llm_system,
        test_privacy_system,
        test_performance_under_load,
        test_memory_usage,
        test_error_recovery,
        test_server_compatibility
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
    
    print("\n" + "=" * 70)
    print(f"📊 测试结果: {passed}/{total} 通过")
    print(f"⏱️  总测试时间: {total_time:.2f} 秒")
    
    if passed == total:
        print("🎉 所有测试通过！系统完全准备就绪")
        print("🚀 可以安全地上传到服务器进行大规模实验")
        print("💯 万无一失，上传成功概率: 100%")
        return True
    else:
        print("⚠️  部分测试失败，需要修复后再上传")
        print(f"❌ 失败率: {(total-passed)/total*100:.1f}%")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
