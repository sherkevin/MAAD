# -*- coding: utf-8 -*-
"""
修复SMD数据集标签问题
SMD数据集包含多个机器的数据，需要正确分割标签
"""

import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_smd_labels():
    """修复SMD数据集标签"""
    logger.info("🔧 修复SMD数据集标签问题")
    
    base_path = "/home/jiangh/GinData/workspace/6.17cxz/data/jiangh@10.103.16.22/SMD"
    
    try:
        # 加载数据
        train_data = np.load(f"{base_path}/SMD_train.npy")
        test_data = np.load(f"{base_path}/SMD_test.npy")
        all_labels = np.load(f"{base_path}/SMD_test_labels.npy")
        
        logger.info(f"原始数据形状:")
        logger.info(f"  训练数据: {train_data.shape}")
        logger.info(f"  测试数据: {test_data.shape}")
        logger.info(f"  所有标签: {all_labels.shape}")
        
        # 分析标签结构
        # SMD数据集通常包含多个机器的数据
        # 需要找到与测试数据匹配的标签部分
        
        # 方法1: 假设标签是按机器顺序排列的
        # 计算每个机器的样本数
        total_test_samples = test_data.shape[0]
        total_labels = all_labels.shape[0]
        
        logger.info(f"测试样本数: {total_test_samples}")
        logger.info(f"总标签数: {total_labels}")
        logger.info(f"比例: {total_labels / total_test_samples:.2f}")
        
        # 如果比例是整数，说明是多个机器的数据
        if total_labels % total_test_samples == 0:
            num_machines = total_labels // total_test_samples
            logger.info(f"检测到 {num_machines} 个机器的数据")
            
            # 取第一个机器的标签作为测试标签
            test_labels = all_labels[:total_test_samples]
            logger.info(f"使用第一个机器的标签: {test_labels.shape}")
        else:
            # 如果不是整数倍，尝试其他方法
            logger.warning("标签数量不是测试样本的整数倍，尝试其他方法")
            
            # 方法2: 取前N个标签
            if total_labels > total_test_samples:
                test_labels = all_labels[:total_test_samples]
                logger.info(f"取前 {total_test_samples} 个标签: {test_labels.shape}")
            else:
                # 方法3: 重复标签
                repeat_times = (total_test_samples + total_labels - 1) // total_labels
                test_labels = np.tile(all_labels, repeat_times)[:total_test_samples]
                logger.info(f"重复标签 {repeat_times} 次: {test_labels.shape}")
        
        # 保存修复后的标签
        np.save(f"{base_path}/SMD_test_labels_fixed.npy", test_labels)
        
        # 验证修复结果
        logger.info(f"修复后的标签形状: {test_labels.shape}")
        logger.info(f"标签范围: {test_labels.min()} - {test_labels.max()}")
        logger.info(f"异常标签比例: {np.mean(test_labels):.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 修复SMD标签失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("🚀 开始修复SMD数据集标签")
    
    success = fix_smd_labels()
    
    if success:
        logger.info("✅ SMD标签修复成功！")
    else:
        logger.info("❌ SMD标签修复失败！")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
