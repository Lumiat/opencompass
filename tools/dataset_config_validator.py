#!/usr/bin/env python3
"""
OpenCompass 数据集配置检查脚本
用于显示配置文件生成的具体prompt，验证配置正确性
"""

import sys
import os
import importlib.util
import json
from typing import Dict, Any, List
from datasets import load_dataset
import argparse


class OpenCompassConfigInspector:
    def __init__(self, config_path: str):
        """初始化配置检查器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config_module = None
        self.datasets_config = None
        
    def load_config(self):
        """加载配置文件"""
        try:
            # 动态导入配置文件
            spec = importlib.util.spec_from_file_location("config", self.config_path)
            self.config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.config_module)
            
            # 获取数据集配置（通常以 _datasets 结尾的变量）
            config_vars = dir(self.config_module)
            dataset_vars = [var for var in config_vars if var.endswith('_datasets')]
            
            if not dataset_vars:
                raise ValueError("未找到数据集配置变量（应以 '_datasets' 结尾）")
            
            # 如果有多个，使用第一个或让用户选择
            if len(dataset_vars) > 1:
                print(f"找到多个数据集配置: {dataset_vars}")
                print(f"使用第一个: {dataset_vars[0]}")
            
            self.datasets_config = getattr(self.config_module, dataset_vars[0])
            print(f"成功加载配置文件: {self.config_path}")
            print(f"数据集配置变量: {dataset_vars[0]}")
            
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            sys.exit(1)
    
    def extract_prompt_info(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """从数据集配置中提取prompt相关信息"""
        info = {
            'abbr': dataset_config.get('abbr', 'Unknown'),
            'type': dataset_config.get('type', 'Unknown'),
            'path': dataset_config.get('path', 'Unknown'),
            'name': dataset_config.get('name', 'Unknown'),
            'reader_cfg': dataset_config.get('reader_cfg', {}),
            'infer_cfg': dataset_config.get('infer_cfg', {}),
            'eval_cfg': dataset_config.get('eval_cfg', {})
        }
        
        # 提取prompt模板
        infer_cfg = info['infer_cfg']
        if 'prompt_template' in infer_cfg:
            prompt_template = infer_cfg['prompt_template']
            if 'template' in prompt_template:
                info['prompt_template'] = prompt_template['template']
        
        return info
    
    def load_sample_data(self, dataset_config: Dict[str, Any], num_samples: int = 3):
        """加载数据集样本数据"""
        try:
            path = dataset_config.get('path')
            name = dataset_config.get('name')
            
            if not path:
                print("未找到数据集路径")
                return None
            
            # 尝试加载数据集
            print(f"尝试加载数据集: {path}")
            if name:
                dataset = load_dataset(path, name, split='test')
            else:
                dataset = load_dataset(path, split='test')
            
            # 获取前几个样本
            samples = dataset.select(range(min(num_samples, len(dataset))))
            return samples
            
        except Exception as e:
            print(f"加载数据集失败: {e}")
            print("请检查数据集路径和名称是否正确")
            return None
    
    def render_prompt(self, template_dict: Dict, sample_data: Dict) -> str:
        """根据模板和样本数据渲染prompt"""
        try:
            if 'round' in template_dict:
                # 处理对话式模板
                full_prompt = ""
                for round_item in template_dict['round']:
                    role = round_item.get('role', 'HUMAN')
                    prompt = round_item.get('prompt', '')
                    
                    # 格式化prompt
                    formatted_prompt = prompt.format(**sample_data)
                    full_prompt += f"[{role}]: {formatted_prompt}\n"
                
                return full_prompt.strip()
            else:
                # 处理简单模板
                return str(template_dict).format(**sample_data)
                
        except KeyError as e:
            return f"模板渲染失败，缺少字段: {e}"
        except Exception as e:
            return f"模板渲染失败: {e}"
    
    def display_config_summary(self):
        """显示配置摘要"""
        print("\n" + "="*80)
        print("配置文件摘要")
        print("="*80)
        
        for i, dataset_config in enumerate(self.datasets_config):
            info = self.extract_prompt_info(dataset_config)
            
            print(f"\n数据集 {i+1}:")
            print(f"  名称: {info['abbr']}")
            print(f"  类型: {info['type']}")
            print(f"  路径: {info['path']}")
            print(f"  子集: {info['name']}")
            
            # 显示输入输出列配置
            reader_cfg = info['reader_cfg']
            if reader_cfg:
                print(f"  输入列: {reader_cfg.get('input_columns', [])}")
                print(f"  输出列: {reader_cfg.get('output_column', 'Unknown')}")
            
            # 显示推理配置
            infer_cfg = info['infer_cfg']
            if infer_cfg:
                retriever = infer_cfg.get('retriever', {})
                retriever_type = retriever.get('type', 'Unknown') if isinstance(retriever, dict) else str(retriever)
                print(f"  检索器: {retriever_type}")
                
                inferencer = infer_cfg.get('inferencer', {})
                inferencer_type = inferencer.get('type', 'Unknown') if isinstance(inferencer, dict) else str(inferencer)
                print(f"  推理器: {inferencer_type}")
    
    def display_prompts(self, num_samples: int = 3):
        """显示具体的prompt示例"""
        print("\n" + "="*80)
        print("Prompt 示例")
        print("="*80)
        
        for i, dataset_config in enumerate(self.datasets_config):
            print(f"\n数据集 {i+1}: {dataset_config.get('abbr', 'Unknown')}")
            print("-" * 50)
            
            info = self.extract_prompt_info(dataset_config)
            
            # 加载样本数据
            samples = self.load_sample_data(dataset_config, num_samples)
            
            if samples is None:
                print("无法加载数据集样本，将使用模拟数据")
                # 创建模拟数据
                reader_cfg = info['reader_cfg']
                input_columns = reader_cfg.get('input_columns', [])
                output_column = reader_cfg.get('output_column', 'answer')
                
                mock_data = {}
                for col in input_columns:
                    mock_data[col] = f"示例{col}"
                mock_data[output_column] = "示例答案"
                samples = [mock_data]
            
            # 显示prompt模板
            if 'prompt_template' in info:
                print("Prompt 模板:")
                template = info['prompt_template']
                if isinstance(template, dict):
                    print(json.dumps(template, indent=2, ensure_ascii=False))
                else:
                    print(template)
                
                print("\n渲染后的 Prompt 示例:")
                # 渲染样本
                for j, sample in enumerate(samples):
                    if j >= num_samples:
                        break
                    print(f"\n样本 {j+1}:")
                    print("-" * 30)
                    
                    # 显示原始数据
                    print("原始数据:")
                    if hasattr(sample, 'keys'):
                        for key in sample.keys():
                            print(f"  {key}: {sample[key]}")
                    else:
                        print(f"  {sample}")
                    
                    print("\n生成的Prompt:")
                    rendered_prompt = self.render_prompt(template, sample)
                    print(rendered_prompt)
                    print("-" * 30)
            else:
                print("未找到prompt模板配置")
    
    def run_inspection(self, num_samples: int = 3):
        """运行完整的配置检查"""
        print(f"OpenCompass 配置文件检查工具")
        print(f"配置文件: {self.config_path}")
        
        # 加载配置
        self.load_config()
        
        # 显示配置摘要
        self.display_config_summary()
        
        # 显示prompt示例
        self.display_prompts(num_samples)
        
        print("\n" + "="*80)
        print("检查完成！")
        print("请仔细检查上述prompt是否符合你的需求。")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='OpenCompass配置文件检查工具')
    parser.add_argument('--config_path', help='配置文件路径')
    parser.add_argument('--samples', '-s', type=int, default=3, 
                       help='显示的样本数量 (默认: 3)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config_path):
        print(f"错误: 配置文件不存在: {args.config_path}")
        sys.exit(1)
    
    inspector = OpenCompassConfigInspector(args.config_path)
    inspector.run_inspection(args.samples)


if __name__ == "__main__":
    main()
