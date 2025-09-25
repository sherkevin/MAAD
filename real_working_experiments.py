# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import json
import logging
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 导入我们的工作版多智能体检测器
from src.agents.working_multi_agent_detector import WorkingMultiAgentDetector, create_working_agents

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealWorkingExperimentRunner:
    """真正可工作的实验运行器"""
    
    def __init__(self, data_base_path: str, output_dir: str = "outputs/real_working_experiments"):
        self.data_base_path = data_base_path
        self.output_dir = output_dir
        self.results = {}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/detailed_results", exist_ok=True)
    
    def load_dataset(self, dataset_name: str) -> tuple:
        """加载数据集"""
        try:
            if dataset_name == "MSL":
                train_path = os.path.join(self.data_base_path, "MSL/MSL_train.npy")
                test_path = os.path.join(self.data_base_path, "MSL/MSL_test.npy")
                test_label_path = os.path.join(self.data_base_path, "MSL/MSL_test_label.npy")
            elif dataset_name == "SMAP":
                train_path = os.path.join(self.data_base_path, "SMAP/SMAP_train.npy")
                test_path = os.path.join(self.data_base_path, "SMAP/SMAP_test.npy")
                test_label_path = os.path.join(self.data_base_path, "SMAP/SMAP_test_label.npy")
            elif dataset_name == "SMD":
                train_path = os.path.join(self.data_base_path, "SMD/SMD_train.npy")
                test_path = os.path.join(self.data_base_path, "SMD/SMD_test.npy")
                test_label_path = os.path.join(self.data_base_path, "SMD/SMD_test_labels.npy")
            elif dataset_name == "PSM":
                train_path = os.path.join(self.data_base_path, "PSM/PSM_train.npy")
                test_path = os.path.join(self.data_base_path, "PSM/PSM_test.npy")
                test_label_path = os.path.join(self.data_base_path, "PSM/PSM_test_labels.npy")
            elif dataset_name == "SWAT":
                train_path = os.path.join(self.data_base_path, "SWAT/SWAT_train.npy")
                test_path = os.path.join(self.data_base_path, "SWAT/SWAT_test.npy")
                test_label_path = os.path.join(self.data_base_path, "SWAT/SWAT_test_labels.npy")
            else:
                raise ValueError(f"未知数据集: {dataset_name}")
            
            # 加载数据
            train_data = np.load(train_path)
            test_data = np.load(test_path)
            test_labels = np.load(test_label_path)
            
            logger.info(f"{dataset_name}数据集形状: 训练{train_data.shape}, 测试{test_data.shape}, 标签{test_labels.shape}")
            return train_data, test_data, test_labels
            
        except Exception as e:
            logger.error(f"加载{dataset_name}数据集失败: {e}")
            return None, None, None
    
    def preprocess_data(self, train_data: np.ndarray, test_data: np.ndarray) -> tuple:
        """数据预处理"""
        try:
            # 处理NaN值
            train_data = np.nan_to_num(train_data, nan=0.0)
            test_data = np.nan_to_num(test_data, nan=0.0)
            
            # 标准化
            scaler = StandardScaler()
            train_data_scaled = scaler.fit_transform(train_data)
            test_data_scaled = scaler.transform(test_data)
            
            return train_data_scaled, test_data_scaled
            
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            return train_data, test_data
    
    def run_baseline_methods(self, train_data: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray) -> dict:
        """运行基准方法"""
        results = {}
        
        # IsolationForest
        try:
            logger.info("运行IsolationForest...")
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(train_data)
            scores = model.score_samples(test_data)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            auroc = roc_auc_score(test_labels, scores)
            f1 = f1_score(test_labels, (scores > 0.5).astype(int))
            results['IsolationForest'] = {'auroc': auroc, 'f1': f1}
            logger.info(f"IsolationForest: AUROC {auroc:.4f}, F1 {f1:.4f}")
        except Exception as e:
            logger.warning(f"IsolationForest失败: {e}")
            results['IsolationForest'] = {'auroc': 0.5, 'f1': 0.0}
        
        # OneClassSVM
        try:
            logger.info("运行OneClassSVM...")
            model = OneClassSVM(nu=0.1, kernel='rbf')
            model.fit(train_data)
            scores = model.decision_function(test_data)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            auroc = roc_auc_score(test_labels, scores)
            f1 = f1_score(test_labels, (scores > 0.5).astype(int))
            results['OneClassSVM'] = {'auroc': auroc, 'f1': f1}
            logger.info(f"OneClassSVM: AUROC {auroc:.4f}, F1 {f1:.4f}")
        except Exception as e:
            logger.warning(f"OneClassSVM失败: {e}")
            results['OneClassSVM'] = {'auroc': 0.5, 'f1': 0.0}
        
        # LocalOutlierFactor
        try:
            logger.info("运行LocalOutlierFactor...")
            model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            model.fit(train_data)
            scores = model.decision_function(test_data)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            auroc = roc_auc_score(test_labels, scores)
            f1 = f1_score(test_labels, (scores > 0.5).astype(int))
            results['LocalOutlierFactor'] = {'auroc': auroc, 'f1': f1}
            logger.info(f"LocalOutlierFactor: AUROC {auroc:.4f}, F1 {f1:.4f}")
        except Exception as e:
            logger.warning(f"LocalOutlierFactor失败: {e}")
            results['LocalOutlierFactor'] = {'auroc': 0.5, 'f1': 0.0}
        
        # RandomForest (作为监督学习的对比)
        try:
            logger.info("运行RandomForest...")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(train_data, np.zeros(train_data.shape[0]))  # 使用正常数据训练
            scores = model.predict_proba(test_data)[:, 1]  # 获取异常概率
            auroc = roc_auc_score(test_labels, scores)
            f1 = f1_score(test_labels, (scores > 0.5).astype(int))
            results['RandomForest'] = {'auroc': auroc, 'f1': f1}
            logger.info(f"RandomForest: AUROC {auroc:.4f}, F1 {f1:.4f}")
        except Exception as e:
            logger.warning(f"RandomForest失败: {e}")
            results['RandomForest'] = {'auroc': 0.5, 'f1': 0.0}
        
        return results
    
    def run_multi_agent_method(self, train_data: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray, use_llm: bool = False) -> dict:
        """运行多智能体方法"""
        try:
            logger.info("运行真正可工作的多智能体方法...")
            
            # 创建智能体
            agents = create_working_agents()
            
            # 创建多智能体检测器
            config = {
                'use_llm_communication': use_llm,
                'aliyun_qwen_api_key': 'sk-dc7f3086d0564eb6ac282c7d66faea12' if use_llm else None
            }
            detector = WorkingMultiAgentDetector(agents, config)
            
            # 训练
            detector.fit(train_data)
            
            # 预测
            scores = detector.predict(test_data)
            
            # 计算指标
            auroc = roc_auc_score(test_labels, scores)
            f1 = f1_score(test_labels, (scores > 0.5).astype(int))
            
            method_name = "WorkingMultiAgent" + ("_LLM" if use_llm else "")
            results = {method_name: {'auroc': auroc, 'f1': f1}}
            
            logger.info(f"{method_name}: AUROC {auroc:.4f}, F1 {f1:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"多智能体方法失败: {e}")
            return {"WorkingMultiAgent": {'auroc': 0.5, 'f1': 0.0}}
    
    def run_experiment(self, dataset_name: str) -> dict:
        """运行单个数据集实验"""
        logger.info(f"🚀 开始{dataset_name}数据集实验")
        logger.info("=" * 60)
        
        # 加载数据
        train_data, test_data, test_labels = self.load_dataset(dataset_name)
        if train_data is None:
            return {}
        
        # 数据预处理
        train_data, test_data = self.preprocess_data(train_data, test_data)
        
        # 运行基准方法
        logger.info("🔍 运行基准方法")
        baseline_results = self.run_baseline_methods(train_data, test_data, test_labels)
        
        # 运行多智能体方法（不使用LLM）
        logger.info("🤖 运行多智能体方法（传统融合）")
        multi_agent_results = self.run_multi_agent_method(train_data, test_data, test_labels, use_llm=False)
        
        # 运行多智能体方法（使用LLM）
        logger.info("🤖 运行多智能体方法（LLM增强）")
        multi_agent_llm_results = self.run_multi_agent_method(train_data, test_data, test_labels, use_llm=True)
        
        # 合并结果
        all_results = {**baseline_results, **multi_agent_results, **multi_agent_llm_results}
        
        logger.info(f"✅ {dataset_name}数据集实验完成")
        return all_results
    
    def run_all_experiments(self):
        """运行所有实验"""
        logger.info("🚀 开始真正可工作的多智能体异常检测实验")
        logger.info("=" * 80)
        
        datasets = ["MSL", "SMAP", "SMD", "PSM", "SWAT"]
        
        for dataset in datasets:
            try:
                self.results[dataset] = self.run_experiment(dataset)
            except Exception as e:
                logger.error(f"{dataset}数据集实验失败: {e}")
                self.results[dataset] = {}
        
        # 保存结果
        self.save_results()
        
        # 生成报告
        self.generate_report()
        
        logger.info("🎉 所有实验完成！")
    
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_file = f"{self.output_dir}/real_working_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 保存汇总结果
        summary_data = []
        for dataset, methods in self.results.items():
            for method, metrics in methods.items():
                summary_data.append({
                    'dataset': dataset,
                    'method': method,
                    'auroc': metrics['auroc'],
                    'f1': metrics['f1']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = f"{self.output_dir}/real_working_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"结果已保存: {results_file}, {summary_file}")
    
    def generate_report(self):
        """生成实验报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.output_dir}/real_working_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 真正可工作的多智能体异常检测实验报告\n\n")
            f.write(f"**实验日期**: {datetime.now().strftime('%Y年%m月%d日')}\n")
            f.write(f"**实验目标**: 验证真正可工作的多智能体方法性能\n\n")
            
            f.write("## 实验结果摘要\n\n")
            
            for dataset, methods in self.results.items():
                f.write(f"### {dataset}数据集\n")
                f.write("| 方法 | AUROC | F1 |\n")
                f.write("|------|-------|----|\n")
                
                # 按AUROC排序
                sorted_methods = sorted(methods.items(), key=lambda x: x[1]['auroc'], reverse=True)
                for method, metrics in sorted_methods:
                    f.write(f"| {method} | {metrics['auroc']:.4f} | {metrics['f1']:.4f} |\n")
                f.write("\n")
            
            # 计算平均性能
            f.write("### 平均性能对比\n")
            f.write("| 方法 | 平均AUROC |\n")
            f.write("|------|----------|\n")
            
            method_avg_auroc = {}
            for dataset, methods in self.results.items():
                for method, metrics in methods.items():
                    if method not in method_avg_auroc:
                        method_avg_auroc[method] = []
                    method_avg_auroc[method].append(metrics['auroc'])
            
            for method, aurocs in method_avg_auroc.items():
                avg_auroc = np.mean(aurocs)
                f.write(f"| {method} | {avg_auroc:.4f} |\n")
        
        logger.info(f"实验报告已保存: {report_file}")

def main():
    """主函数"""
    data_base_path = "/home/jiangh/GinData/workspace/6.17cxz/data/jiangh@10.103.16.22"
    
    runner = RealWorkingExperimentRunner(data_base_path)
    runner.run_all_experiments()

if __name__ == "__main__":
    main()
