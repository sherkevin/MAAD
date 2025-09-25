# -*- coding: utf-8 -*-
"""
数据集格式转换脚本
将SMD、PSM、SWAT数据集转换为标准格式
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_smd_dataset():
    """转换SMD数据集"""
    logger.info("🔄 转换SMD数据集...")
    
    base_path = "/home/jiangh/GinData/workspace/6.17cxz/data/jiangh@10.103.16.22/SMD"
    output_path = "/home/jiangh/GinData/workspace/6.17cxz/data/jiangh@10.103.16.22/SMD"
    
    try:
        # 加载训练数据
        with open(f"{base_path}/machine-1-6_train.pkl", 'rb') as f:
            train_data = pickle.load(f)
        
        # 加载测试数据
        with open(f"{base_path}/machine-1-6_test.pkl", 'rb') as f:
            test_data = pickle.load(f)
        
        # 加载测试标签
        test_labels = np.load(f"{base_path}/SMD_test_label.npy")
        
        # 转换为numpy数组
        if isinstance(train_data, pd.DataFrame):
            train_data = train_data.values
        if isinstance(test_data, pd.DataFrame):
            test_data = test_data.values
        
        # 保存为npy格式
        np.save(f"{output_path}/SMD_train.npy", train_data)
        np.save(f"{output_path}/SMD_test.npy", test_data)
        np.save(f"{output_path}/SMD_test_labels.npy", test_labels)
        
        logger.info(f"✅ SMD数据集转换完成")
        logger.info(f"  训练数据形状: {train_data.shape}")
        logger.info(f"  测试数据形状: {test_data.shape}")
        logger.info(f"  测试标签形状: {test_labels.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SMD数据集转换失败: {e}")
        return False

def convert_psm_dataset():
    """转换PSM数据集"""
    logger.info("🔄 转换PSM数据集...")
    
    base_path = "/home/jiangh/GinData/workspace/6.17cxz/data/jiangh@10.103.16.22/PSM"
    output_path = "/home/jiangh/GinData/workspace/6.17cxz/data/jiangh@10.103.16.22/PSM"
    
    try:
        # 加载训练数据
        train_df = pd.read_csv(f"{base_path}/train.csv")
        
        # 加载测试数据
        test_df = pd.read_csv(f"{base_path}/test.csv")
        
        # 加载测试标签
        test_labels_df = pd.read_csv(f"{base_path}/test_label.csv")
        
        # 提取特征和标签
        # 假设最后一列是标签，其他是特征
        train_data = train_df.iloc[:, :-1].values
        test_data = test_df.iloc[:, :-1].values
        test_labels = test_labels_df.iloc[:, -1].values
        
        # 保存为npy格式
        np.save(f"{output_path}/PSM_train.npy", train_data)
        np.save(f"{output_path}/PSM_test.npy", test_data)
        np.save(f"{output_path}/PSM_test_labels.npy", test_labels)
        
        logger.info(f"✅ PSM数据集转换完成")
        logger.info(f"  训练数据形状: {train_data.shape}")
        logger.info(f"  测试数据形状: {test_data.shape}")
        logger.info(f"  测试标签形状: {test_labels.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PSM数据集转换失败: {e}")
        return False

def convert_swat_dataset():
    """转换SWAT数据集"""
    logger.info("🔄 转换SWAT数据集...")
    
    base_path = "/home/jiangh/GinData/workspace/6.17cxz/data/jiangh@10.103.16.22/SWAT"
    output_path = "/home/jiangh/GinData/workspace/6.17cxz/data/jiangh@10.103.16.22/SWAT"
    
    try:
        # 加载正常数据
        normal_df = pd.read_csv(f"{base_path}/SWaT_Dataset_Normal_v1.csv")
        
        # 加载攻击数据
        attack_df = pd.read_csv(f"{base_path}/SWaT_Dataset_Attack_v0.csv")
        
        # 合并数据
        combined_df = pd.concat([normal_df, attack_df], ignore_index=True)
        
        # 假设最后一列是标签，其他是特征
        # 需要根据实际数据结构调整
        data = combined_df.iloc[:, :-1].values
        labels = combined_df.iloc[:, -1].values
        
        # 分割训练和测试数据 (80% 训练, 20% 测试)
        split_idx = int(0.8 * len(data))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        test_labels = labels[split_idx:]
        
        # 保存为npy格式
        np.save(f"{output_path}/SWAT_train.npy", train_data)
        np.save(f"{output_path}/SWAT_test.npy", test_data)
        np.save(f"{output_path}/SWAT_test_labels.npy", test_labels)
        
        logger.info(f"✅ SWAT数据集转换完成")
        logger.info(f"  训练数据形状: {train_data.shape}")
        logger.info(f"  测试数据形状: {test_data.shape}")
        logger.info(f"  测试标签形状: {test_labels.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SWAT数据集转换失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("🚀 开始数据集格式转换")
    logger.info("=" * 60)
    
    # 转换结果
    results = []
    
    # 转换SMD数据集
    results.append(("SMD", convert_smd_dataset()))
    
    # 转换PSM数据集
    results.append(("PSM", convert_psm_dataset()))
    
    # 转换SWAT数据集
    results.append(("SWAT", convert_swat_dataset()))
    
    # 显示结果
    logger.info("=" * 60)
    logger.info("📋 转换结果汇总")
    logger.info("=" * 60)
    
    success_count = 0
    for dataset_name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        logger.info(f"{dataset_name}数据集: {status}")
        if success:
            success_count += 1
    
    logger.info(f"\n总体结果: {success_count}/{len(results)} 数据集转换成功")
    
    if success_count == len(results):
        logger.info("🎉 所有数据集转换成功！")
    else:
        logger.info("⚠️ 部分数据集转换失败，请检查错误信息")
    
    return success_count == len(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
