#!/usr/bin/env python3
"""
GPU兼容性测试脚本
确保在服务器上能正确使用GPU运行
"""

import torch
import sys
import os
import time

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_gpu_availability():
    """测试GPU可用性"""
    print("🧪 测试1: GPU可用性测试")
    print("-" * 50)
    
    print(f"📊 PyTorch版本: {torch.__version__}")
    print(f"📊 CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"📊 CUDA版本: {torch.version.cuda}")
        print(f"📊 GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"📊 GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"📊 GPU {i} 内存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
            print(f"📊 GPU {i} 计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        return True
    else:
        print("❌ CUDA不可用，需要安装CUDA版本的PyTorch")
        return False

def test_gpu_tensor_operations():
    """测试GPU张量操作"""
    print("\n🧪 测试2: GPU张量操作测试")
    print("-" * 50)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过GPU测试")
        return True
    
    try:
        # 创建GPU张量
        device = torch.device('cuda')
        print(f"📊 使用设备: {device}")
        
        # 测试基本张量操作
        a = torch.randn(1000, 1000).to(device)
        b = torch.randn(1000, 1000).to(device)
        
        # 矩阵乘法
        start_time = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"✅ GPU矩阵乘法成功: {c.shape}")
        print(f"📊 GPU计算时间: {gpu_time:.4f} 秒")
        
        # 测试内存使用
        memory_allocated = torch.cuda.memory_allocated() / 1e6
        memory_reserved = torch.cuda.memory_reserved() / 1e6
        print(f"📊 GPU内存分配: {memory_allocated:.1f} MB")
        print(f"📊 GPU内存保留: {memory_reserved:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU张量操作失败: {e}")
        return False

def test_multi_agent_gpu():
    """测试多智能体系统GPU运行"""
    print("\n🧪 测试3: 多智能体系统GPU测试")
    print("-" * 50)
    
    try:
        from agents.multi_agent_detector import MultiAgentAnomalyDetector
        
        # 创建检测器
        config = {'trend_agent': {}}
        detector = MultiAgentAnomalyDetector(config)
        
        # 测试数据
        if torch.cuda.is_available():
            device = torch.device('cuda')
            test_data = torch.randn(1, 3, 64, 64).to(device)
            print(f"📊 测试数据设备: {test_data.device}")
        else:
            test_data = torch.randn(1, 3, 64, 64)
            print("📊 使用CPU测试数据")
        
        # 执行检测
        start_time = time.time()
        result = detector.detect_anomaly(test_data)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        gpu_time = time.time() - start_time
        
        if 'error' in result:
            print(f"❌ 检测失败: {result['error']}")
            return False
        
        print(f"✅ 异常检测成功: 异常分数={result['final_decision']['anomaly_score']:.3f}")
        print(f"📊 检测时间: {gpu_time:.4f} 秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 多智能体GPU测试失败: {e}")
        return False

def test_gpu_memory_management():
    """测试GPU内存管理"""
    print("\n🧪 测试4: GPU内存管理测试")
    print("-" * 50)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过GPU内存测试")
        return True
    
    try:
        device = torch.device('cuda')
        
        # 记录初始内存
        initial_memory = torch.cuda.memory_allocated()
        print(f"📊 初始GPU内存: {initial_memory / 1e6:.1f} MB")
        
        # 创建大量张量
        tensors = []
        for i in range(10):
            tensor = torch.randn(1000, 1000).to(device)
            tensors.append(tensor)
        
        # 记录当前内存
        current_memory = torch.cuda.memory_allocated()
        print(f"📊 创建张量后内存: {current_memory / 1e6:.1f} MB")
        print(f"📊 内存增长: {(current_memory - initial_memory) / 1e6:.1f} MB")
        
        # 清理内存
        del tensors
        torch.cuda.empty_cache()
        
        # 记录清理后内存
        final_memory = torch.cuda.memory_allocated()
        print(f"📊 清理后内存: {final_memory / 1e6:.1f} MB")
        
        # 检查内存是否被正确释放
        if final_memory <= initial_memory * 1.1:  # 允许10%的误差
            print("✅ GPU内存管理正常")
            return True
        else:
            print("⚠️  GPU内存可能未完全释放")
            return False
        
    except Exception as e:
        print(f"❌ GPU内存管理测试失败: {e}")
        return False

def test_gpu_performance():
    """测试GPU性能"""
    print("\n🧪 测试5: GPU性能测试")
    print("-" * 50)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过GPU性能测试")
        return True
    
    try:
        device = torch.device('cuda')
        
        # 性能测试
        sizes = [100, 500, 1000, 2000]
        
        for size in sizes:
            # GPU测试
            a_gpu = torch.randn(size, size).to(device)
            b_gpu = torch.randn(size, size).to(device)
            
            start_time = time.time()
            c_gpu = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            # CPU测试
            a_cpu = a_gpu.cpu()
            b_cpu = b_gpu.cpu()
            
            start_time = time.time()
            c_cpu = torch.matmul(a_cpu, b_cpu)
            cpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            
            print(f"📊 矩阵大小 {size}x{size}:")
            print(f"   GPU时间: {gpu_time:.4f} 秒")
            print(f"   CPU时间: {cpu_time:.4f} 秒")
            print(f"   加速比: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU性能测试失败: {e}")
        return False

def test_gpu_error_handling():
    """测试GPU错误处理"""
    print("\n🧪 测试6: GPU错误处理测试")
    print("-" * 50)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过GPU错误处理测试")
        return True
    
    try:
        device = torch.device('cuda')
        
        # 测试内存不足错误
        try:
            # 尝试分配超大张量
            large_tensor = torch.randn(10000, 10000).to(device)
            print("⚠️  意外成功分配超大张量")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("✅ 正确捕获内存不足错误")
            else:
                print(f"⚠️  捕获到其他错误: {e}")
        
        # 测试设备不匹配错误
        try:
            cpu_tensor = torch.randn(10, 10)
            gpu_tensor = torch.randn(10, 10).to(device)
            result = cpu_tensor + gpu_tensor
            print("⚠️  意外成功执行设备不匹配操作")
        except RuntimeError as e:
            if "device" in str(e).lower():
                print("✅ 正确捕获设备不匹配错误")
            else:
                print(f"⚠️  捕获到其他错误: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU错误处理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始GPU兼容性测试")
    print("=" * 60)
    
    tests = [
        test_gpu_availability,
        test_gpu_tensor_operations,
        test_multi_agent_gpu,
        test_gpu_memory_management,
        test_gpu_performance,
        test_gpu_error_handling
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
        print("🎉 所有GPU测试通过！系统GPU兼容性良好")
        print("🚀 可以安全地在GPU服务器上运行")
        return True
    else:
        print("⚠️  部分GPU测试失败，需要检查GPU配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
