# -*- coding: utf-8 -*-
"""
测试阿里云百炼平台集成
验证增强版多智能体通信系统
"""

import sys
import os
import logging
import numpy as np
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm.aliyun_qwen_interface import AliyunQwenInterface
from src.communication.enhanced_llm_communication import EnhancedLLMCommunication

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_aliyun_qwen_interface():
    """测试阿里云百炼接口"""
    print("=" * 60)
    print("🧪 测试阿里云百炼接口")
    print("=" * 60)
    
    # API密钥
    api_key = "sk-dc7f3086d0564eb6ac282c7d66faea12"
    
    try:
        # 创建接口实例
        qwen = AliyunQwenInterface(api_key)
        print("✅ 接口创建成功")
        
        # 测试1: 基本文本生成
        print("\n📝 测试基本文本生成...")
        response = qwen.generate_response("请简单介绍一下多智能体异常检测系统")
        print(f"响应: {response[:200]}...")
        
        # 测试2: 多智能体通信
        print("\n🤖 测试多智能体通信...")
        agent_messages = [
            {'agent': 'TrendAgent', 'content': '检测到上升趋势异常，分数0.8'},
            {'agent': 'VarianceAgent', 'content': '方差波动正常，分数0.3'},
            {'agent': 'ResidualAgent', 'content': '残差分析显示异常，分数0.7'}
        ]
        
        comm_result = qwen.multi_agent_communication(agent_messages, "MSL数据集异常检测")
        print(f"通信成功: {comm_result.get('success', False)}")
        print(f"响应: {comm_result.get('response', '')[:200]}...")
        
        # 测试3: 异常模式分析
        print("\n🔍 测试异常模式分析...")
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
        print(f"分析结果: {analysis_result.get('analysis', '')[:200]}...")
        
        # 测试4: 融合策略生成
        print("\n🔗 测试融合策略生成...")
        agent_scores = [0.8, 0.3, 0.7]
        agent_weights = [1.0, 1.0, 1.0]
        
        strategy_result = qwen.generate_fusion_strategy(agent_scores, agent_weights)
        print(f"策略建议: {strategy_result.get('strategy', '')[:200]}...")
        print(f"融合分数: {strategy_result.get('weighted_average', 0):.4f}")
        
        # 显示统计信息
        stats = qwen.get_statistics()
        print(f"\n📊 统计信息:")
        print(f"  请求次数: {stats['request_count']}")
        print(f"  总Token数: {stats['total_tokens']}")
        print(f"  错误次数: {stats['error_count']}")
        print(f"  成功率: {stats['success_rate']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_enhanced_communication():
    """测试增强版通信系统"""
    print("\n" + "=" * 60)
    print("🚀 测试增强版多智能体通信系统")
    print("=" * 60)
    
    # API密钥
    api_key = "sk-dc7f3086d0564eb6ac282c7d66faea12"
    
    try:
        # 创建增强版通信系统
        comm_system = EnhancedLLMCommunication(api_key)
        print("✅ 增强版通信系统创建成功")
        
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
        print("\n🤝 测试智能智能体协调...")
        result = comm_system.intelligent_agent_coordination(agent_results, context)
        
        print(f"协调成功: {result.get('success', False)}")
        print(f"异常分数: {result.get('anomaly_score', 0):.4f}")
        print(f"异常阈值: {result.get('threshold', 0):.4f}")
        print(f"是否异常: {result.get('is_anomaly', False)}")
        print(f"风险等级: {result.get('risk_level', 'Unknown')}")
        print(f"置信度: {result.get('confidence', 0):.4f}")
        print(f"一致性: {result.get('consistency', 0):.4f}")
        
        # 显示解释
        explanation = result.get('explanation', '')
        print(f"\n📋 决策解释:")
        print(explanation)
        
        # 显示统计信息
        stats = comm_system.get_communication_statistics()
        print(f"\n📊 通信统计:")
        print(f"  总通信次数: {stats['total_communications']}")
        print(f"  成功通信次数: {stats['successful_communications']}")
        print(f"  成功率: {stats['success_rate']:.2%}")
        print(f"  LLM调用次数: {stats['llm_calls']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_real_data():
    """使用真实数据测试集成"""
    print("\n" + "=" * 60)
    print("📊 使用真实数据测试集成")
    print("=" * 60)
    
    # API密钥
    api_key = "sk-dc7f3086d0564eb6ac282c7d66faea12"
    
    try:
        # 创建增强版通信系统
        comm_system = EnhancedLLMCommunication(api_key)
        
        # 模拟MSL数据集的真实检测结果
        msl_agent_results = [
            {'agent': 'TrendAgent', 'score': 0.5711, 'confidence': 0.85, 'weight': 1.0},
            {'agent': 'VarianceAgent', 'score': 0.4200, 'confidence': 0.70, 'weight': 1.0},
            {'agent': 'ResidualAgent', 'score': 0.6800, 'confidence': 0.90, 'weight': 1.0},
            {'agent': 'StatisticalAgent', 'score': 0.5200, 'confidence': 0.75, 'weight': 1.0},
            {'agent': 'FrequencyAgent', 'score': 0.4800, 'confidence': 0.65, 'weight': 1.0}
        ]
        
        msl_context = {
            'dataset': 'MSL',
            'features': 55,
            'samples': 73729,
            'window_size': 100,
            'anomaly_ratio': 0.1
        }
        
        print("🔍 测试MSL数据集异常检测...")
        msl_result = comm_system.intelligent_agent_coordination(msl_agent_results, msl_context)
        
        print(f"MSL检测结果:")
        print(f"  异常分数: {msl_result.get('anomaly_score', 0):.4f}")
        print(f"  是否异常: {msl_result.get('is_anomaly', False)}")
        print(f"  风险等级: {msl_result.get('risk_level', 'Unknown')}")
        
        # 模拟SMAP数据集的真实检测结果
        smap_agent_results = [
            {'agent': 'TrendAgent', 'score': 0.4869, 'confidence': 0.60, 'weight': 1.0},
            {'agent': 'VarianceAgent', 'score': 0.3500, 'confidence': 0.55, 'weight': 1.0},
            {'agent': 'ResidualAgent', 'score': 0.4200, 'confidence': 0.70, 'weight': 1.0},
            {'agent': 'StatisticalAgent', 'score': 0.3800, 'confidence': 0.60, 'weight': 1.0},
            {'agent': 'FrequencyAgent', 'score': 0.4500, 'confidence': 0.65, 'weight': 1.0}
        ]
        
        smap_context = {
            'dataset': 'SMAP',
            'features': 25,
            'samples': 427617,
            'window_size': 100,
            'anomaly_ratio': 0.05
        }
        
        print("\n🔍 测试SMAP数据集异常检测...")
        smap_result = comm_system.intelligent_agent_coordination(smap_agent_results, smap_context)
        
        print(f"SMAP检测结果:")
        print(f"  异常分数: {smap_result.get('anomaly_score', 0):.4f}")
        print(f"  是否异常: {smap_result.get('is_anomaly', False)}")
        print(f"  风险等级: {smap_result.get('risk_level', 'Unknown')}")
        
        # 对比分析
        print(f"\n📈 数据集对比分析:")
        print(f"  MSL: 分数={msl_result.get('anomaly_score', 0):.4f}, 异常={msl_result.get('is_anomaly', False)}")
        print(f"  SMAP: 分数={smap_result.get('anomaly_score', 0):.4f}, 异常={smap_result.get('is_anomaly', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始阿里云百炼平台集成测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试结果
    test_results = []
    
    # 测试1: 阿里云百炼接口
    test_results.append(("阿里云百炼接口", test_aliyun_qwen_interface()))
    
    # 测试2: 增强版通信系统
    test_results.append(("增强版通信系统", test_enhanced_communication()))
    
    # 测试3: 真实数据集成
    test_results.append(("真实数据集成", test_integration_with_real_data()))
    
    # 显示测试结果
    print("\n" + "=" * 60)
    print("📋 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！阿里云百炼平台集成成功！")
    else:
        print("⚠️ 部分测试失败，请检查错误信息")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
