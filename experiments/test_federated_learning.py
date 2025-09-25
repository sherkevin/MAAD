#!/usr/bin/env python3
"""
联邦学习集成测试脚本
测试差分隐私和联邦学习功能，确保服务器兼容性
"""

import torch
import sys
import os
import time
import yaml

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from federated.federated_learning import FederatedLearning
from privacy.differential_privacy import DifferentialPrivacy, PrivacyBudget
from agents.multi_agent_detector import MultiAgentAnomalyDetector

def test_differential_privacy():
    """测试差分隐私功能"""
    print("🧪 测试1: 差分隐私功能测试")
    print("-" * 50)
    
    # 配置
    privacy_config = {
        'epsilon': 1.0,
        'delta': 1e-5,
        'mechanism': 'gaussian',
        'sensitivity': 1.0
    }
    
    # 创建差分隐私实例
    dp = DifferentialPrivacy(privacy_config)
    print("✅ 差分隐私实例创建成功")
    
    # 测试数据
    test_data = torch.randn(10, 10)
    print(f"📊 原始数据形状: {test_data.shape}")
    
    # 添加噪声
    noisy_data = dp.add_noise(test_data)
    print(f"📊 加噪数据形状: {noisy_data.shape}")
    
    # 检查隐私预算
    budget_status = dp.get_privacy_budget_status()
    print(f"📈 隐私预算状态: {budget_status}")
    
    # 测试梯度保护
    gradients = [torch.randn(5, 5), torch.randn(3, 3)]
    protected_gradients = dp.protect_gradients(gradients)
    print(f"✅ 梯度保护完成: {len(protected_gradients)} 个梯度")
    
    return True

def test_privacy_budget():
    """测试隐私预算管理"""
    print("\n🧪 测试2: 隐私预算管理测试")
    print("-" * 50)
    
    # 创建隐私预算
    budget = PrivacyBudget(epsilon=2.0, delta=1e-4)
    print(f"✅ 隐私预算创建: ε={budget.epsilon}, δ={budget.delta}")
    
    # 测试预算消费
    can_spend = budget.can_spend(0.5, 1e-5)
    print(f"📊 可以消费 0.5 ε: {can_spend}")
    
    if can_spend:
        success = budget.spend(0.5, 1e-5)
        print(f"📊 消费结果: {success}")
        print(f"📊 剩余 ε: {budget.get_remaining_epsilon()}")
    
    return True

def test_federated_learning_basic():
    """测试联邦学习基础功能"""
    print("\n🧪 测试3: 联邦学习基础功能测试")
    print("-" * 50)
    
    # 配置
    config = {
        'privacy': {
            'epsilon': 1.0,
            'delta': 1e-5,
            'mechanism': 'gaussian',
            'sensitivity': 1.0
        },
        'federated': {
            'num_clients': 3,
            'num_rounds': 3,
            'local_epochs': 2,
            'learning_rate': 0.01,
            'aggregation_method': 'fedavg',
            'client_selection_ratio': 1.0
        },
        'monitor': {},
        'communication': {}
    }
    
    # 创建联邦学习实例
    fl = FederatedLearning(config)
    print("✅ 联邦学习实例创建成功")
    
    # 初始化客户端
    client_configs = [
        {'agent_config': {'trend_agent': {}}},
        {'agent_config': {'trend_agent': {}}},
        {'agent_config': {'trend_agent': {}}}
    ]
    fl.initialize_clients(client_configs)
    print(f"✅ 客户端初始化完成: {len(fl.clients)} 个客户端")
    
    # 初始化全局模型
    model_config = {'agent_config': {'trend_agent': {}}}
    fl.initialize_global_model(model_config)
    print("✅ 全局模型初始化完成")
    
    return True

def test_federated_training():
    """测试联邦学习训练"""
    print("\n🧪 测试4: 联邦学习训练测试")
    print("-" * 50)
    
    # 配置
    config = {
        'privacy': {
            'epsilon': 2.0,
            'delta': 1e-5,
            'mechanism': 'gaussian',
            'sensitivity': 1.0
        },
        'federated': {
            'num_clients': 3,
            'num_rounds': 2,
            'local_epochs': 1,
            'learning_rate': 0.01,
            'aggregation_method': 'fedavg',
            'client_selection_ratio': 1.0
        },
        'monitor': {},
        'communication': {}
    }
    
    # 创建联邦学习实例
    fl = FederatedLearning(config)
    
    # 初始化客户端
    client_configs = [
        {'agent_config': {'trend_agent': {}}},
        {'agent_config': {'trend_agent': {}}},
        {'agent_config': {'trend_agent': {}}}
    ]
    fl.initialize_clients(client_configs)
    
    # 初始化全局模型
    model_config = {'agent_config': {'trend_agent': {}}}
    fl.initialize_global_model(model_config)
    
    # 模拟训练数据
    training_data = {
        'client_0': {'samples': [1, 2, 3, 4, 5]},
        'client_1': {'samples': [6, 7, 8, 9, 10]},
        'client_2': {'samples': [11, 12, 13, 14, 15]}
    }
    
    # 执行联邦学习训练
    print("🔍 开始联邦学习训练...")
    result = fl.federated_training(training_data)
    
    # 验证结果
    if 'error' in result:
        print(f"❌ 联邦学习训练失败: {result['error']}")
        return False
    
    print("✅ 联邦学习训练完成")
    print(f"📊 训练轮次: {result['training_summary']['total_rounds']}")
    print(f"📊 参与客户端: {result['training_summary']['total_clients']}")
    print(f"📊 聚合方法: {result['training_summary']['aggregation_method']}")
    print(f"📊 隐私保护: {result['training_summary']['privacy_enabled']}")
    
    return True

def test_server_compatibility():
    """测试服务器兼容性"""
    print("\n🧪 测试5: 服务器兼容性测试")
    print("-" * 50)
    
    # 加载服务器兼容性配置
    try:
        with open('configs/server_compatibility_config.yaml', 'r', encoding='utf-8') as f:
            server_config = yaml.safe_load(f)
        print("✅ 服务器兼容性配置加载成功")
    except Exception as e:
        print(f"⚠️  配置文件加载失败: {e}")
        return False
    
    # 检查关键配置
    required_configs = [
        'environment', 'multi_agent', 'communication', 
        'federated_learning', 'privacy', 'performance'
    ]
    
    for config_key in required_configs:
        if config_key in server_config:
            print(f"✅ {config_key} 配置存在")
        else:
            print(f"❌ {config_key} 配置缺失")
            return False
    
    # 检查设备兼容性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 当前设备: {device}")
    
    if device.type == 'cuda':
        print(f"📊 CUDA版本: {torch.version.cuda}")
        print(f"📊 GPU数量: {torch.cuda.device_count()}")
        print(f"📊 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return True

def test_memory_management():
    """测试内存管理"""
    print("\n🧪 测试6: 内存管理测试")
    print("-" * 50)
    
    # 测试内存使用
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # 创建一些张量
    tensors = []
    for i in range(10):
        tensor = torch.randn(100, 100)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        tensors.append(tensor)
    
    current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    memory_used = current_memory - initial_memory
    
    print(f"📊 初始内存: {initial_memory / 1e6:.1f} MB")
    print(f"📊 当前内存: {current_memory / 1e6:.1f} MB")
    print(f"📊 使用内存: {memory_used / 1e6:.1f} MB")
    
    # 清理内存
    del tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"📊 清理后内存: {final_memory / 1e6:.1f} MB")
    
    return True

def test_performance_metrics():
    """测试性能指标"""
    print("\n🧪 测试7: 性能指标测试")
    print("-" * 50)
    
    # 测试多智能体检测性能
    agent_config = {'trend_agent': {}}
    detector = MultiAgentAnomalyDetector(agent_config)
    
    # 测试数据
    test_data = torch.randn(1, 3, 64, 64)
    
    # 性能测试
    start_time = time.time()
    
    for i in range(10):
        result = detector.detect_anomaly(test_data)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / 10
    
    print(f"📊 总测试时间: {total_time:.3f} 秒")
    print(f"📊 平均检测时间: {avg_time:.3f} 秒")
    print(f"📊 检测频率: {1/avg_time:.1f} 次/秒")
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始联邦学习集成测试")
    print("=" * 60)
    
    tests = [
        test_differential_privacy,
        test_privacy_budget,
        test_federated_learning_basic,
        test_federated_training,
        test_server_compatibility,
        test_memory_management,
        test_performance_metrics
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
        print("🎉 所有测试通过！联邦学习集成功能完整")
        print("🚀 系统已准备好上传到服务器进行大规模测试")
        return True
    else:
        print("⚠️  部分测试失败，需要检查代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
