# dataset_config_validator.py
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any, List
import traceback

class OpenCompassConfigValidator:
    """通用的OpenCompass配置验证器"""
    
    def __init__(self, config_file_path: str):
        """
        Args:
            config_file_path: 配置文件路径，如 'ARC_c_test_gen.py'
        """
        self.config_path = Path(config_file_path)
        self.config_module = self._load_config_module()
        
    def _load_config_module(self):
        """动态加载配置模块"""
        spec = importlib.util.spec_from_file_location("config", self.config_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["config"] = module
        spec.loader.exec_module(module)
        return module
    
    def validate_all_datasets(self, num_samples: int = 3) -> Dict[str, Any]:
        """验证配置文件中的所有数据集配置"""
        # 自动找到配置中的数据集列表
        datasets_var = None
        for attr_name in dir(self.config_module):
            attr = getattr(self.config_module, attr_name)
            if isinstance(attr, list) and len(attr) > 0:
                if isinstance(attr[0], dict) and 'type' in attr[0]:
                    datasets_var = attr
                    print(f"🔍 找到数据集配置变量: {attr_name}")
                    break
        
        if datasets_var is None:
            return {"error": "未找到数据集配置列表"}
        
        results = {}
        for i, dataset_config in enumerate(datasets_var):
            dataset_name = dataset_config.get('abbr', f'dataset_{i}')
            print(f"\n{'='*50}")
            print(f"验证数据集: {dataset_name}")
            print(f"{'='*50}")
            
            result = self._validate_single_dataset(dataset_config, num_samples)
            results[dataset_name] = result
            
        return results
    
    def _validate_single_dataset(self, dataset_config: Dict, num_samples: int = 3) -> Dict[str, Any]:
        """验证单个数据集配置"""
        result = {
            'dataset_loaded': False,
            'columns_match': False,
            'prompt_generated': False,
            'samples': [],
            'generated_prompts': [],
            'errors': []
        }
        
        try:
            # 1. 加载数据集
            dataset_class = dataset_config['type']
            dataset_kwargs = {k: v for k, v in dataset_config.items() 
                            if k not in ['type', 'reader_cfg', 'infer_cfg', 'eval_cfg', 'abbr']}
            
            dataset = dataset_class(**dataset_kwargs)
            result['dataset_loaded'] = True
            print(f"✅ 数据集加载成功，共 {len(dataset)} 个样本")
            
            if len(dataset) == 0:
                result['errors'].append("数据集为空")
                return result
                
            # 2. 检查数据结构
            sample = dataset[0]
            actual_columns = list(sample.keys())
            print(f"📋 实际数据列: {actual_columns}")
            
            reader_cfg = dataset_config.get('reader_cfg', {})
            if 'input_columns' in reader_cfg:
                input_columns = reader_cfg['input_columns']
                missing_cols = [col for col in input_columns if col not in actual_columns]
                if missing_cols:
                    result['errors'].append(f"缺少输入列: {missing_cols}")
                    print(f"❌ 缺少输入列: {missing_cols}")
                else:
                    result['columns_match'] = True
                    print(f"✅ 所有输入列都存在: {input_columns}")
            
            if 'output_column' in reader_cfg:
                output_col = reader_cfg['output_column']
                if output_col not in actual_columns:
                    result['errors'].append(f"缺少输出列: {output_col}")
                    print(f"❌ 缺少输出列: {output_col}")
                else:
                    print(f"✅ 输出列存在: {output_col}")
            
            # 3. 收集样本数据
            for i in range(min(num_samples, len(dataset))):
                sample = dataset[i]
                result['samples'].append(dict(sample))
                
            # 4. 测试prompt生成
            infer_cfg = dataset_config.get('infer_cfg', {})
            if 'prompt_template' in infer_cfg:
                self._test_prompt_generation(dataset, infer_cfg, result, num_samples)
            
            # 5. 测试后处理器
            eval_cfg = dataset_config.get('eval_cfg', {})
            if 'pred_postprocessor' in eval_cfg:
                self._test_postprocessor(eval_cfg, result)
                
        except Exception as e:
            result['errors'].append(f"验证过程出错: {str(e)}")
            print(f"❌ 验证失败: {e}")
            print(f"详细错误: {traceback.format_exc()}")
            
        return result
    
    def _test_prompt_generation(self, dataset, infer_cfg, result, num_samples):
        """测试prompt生成"""
        try:
            from opencompass.openicl.icl_prompt_template import PromptTemplate
            
            prompt_template_cfg = infer_cfg['prompt_template']
            if isinstance(prompt_template_cfg, dict) and prompt_template_cfg.get('type') == PromptTemplate:
                template = prompt_template_cfg['template']
            else:
                template = prompt_template_cfg
                
            prompt_template = PromptTemplate(template=template)
            
            print(f"\n🎯 测试Prompt生成:")
            for i in range(min(num_samples, len(dataset))):
                sample = dataset[i]
                try:
                    generated_prompt = prompt_template.generate_prompt_for_generate_task(sample)
                    result['generated_prompts'].append({
                        'sample_index': i,
                        'prompt': generated_prompt,
                        'sample_data': dict(sample)
                    })
                    print(f"\n--- 样本 {i+1} 的Prompt ---")
                    print(generated_prompt[:500] + "..." if len(generated_prompt) > 500 else generated_prompt)
                    
                except Exception as e:
                    result['errors'].append(f"样本{i}生成prompt失败: {str(e)}")
                    print(f"❌ 样本{i}生成prompt失败: {e}")
                    
            result['prompt_generated'] = len(result['generated_prompts']) > 0
            if result['prompt_generated']:
                print(f"✅ 成功生成 {len(result['generated_prompts'])} 个prompt")
                
        except Exception as e:
            result['errors'].append(f"Prompt生成测试失败: {str(e)}")
            print(f"❌ Prompt测试失败: {e}")
    
    def _test_postprocessor(self, eval_cfg, result):
        """测试后处理器"""
        try:
            postprocessor_cfg = eval_cfg.get('pred_postprocessor', {})
            if not postprocessor_cfg:
                return
                
            print(f"\n🔧 测试后处理器:")
            
            # 动态导入后处理函数
            if 'type' in postprocessor_cfg:
                func = postprocessor_cfg['type']
                kwargs = {k: v for k, v in postprocessor_cfg.items() if k != 'type'}
                
                test_responses = ["A", "The answer is B", "C.", "I think the answer is D", "选择A"]
                for resp in test_responses:
                    try:
                        processed = func(resp, **kwargs)
                        print(f"  '{resp}' -> '{processed}'")
                    except Exception as e:
                        print(f"  '{resp}' -> 处理失败: {e}")
                        
        except Exception as e:
            result['errors'].append(f"后处理器测试失败: {str(e)}")
            print(f"❌ 后处理器测试失败: {e}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成验证报告"""
        report = ["\n" + "="*60]
        report.append("OpenCompass配置验证报告")
        report.append("="*60)
        
        for dataset_name, result in results.items():
            if 'error' in result:
                report.append(f"\n❌ {dataset_name}: {result['error']}")
                continue
                
            report.append(f"\n📊 数据集: {dataset_name}")
            report.append(f"  数据集加载: {'✅' if result['dataset_loaded'] else '❌'}")
            report.append(f"  列名匹配: {'✅' if result['columns_match'] else '❌'}")
            report.append(f"  Prompt生成: {'✅' if result['prompt_generated'] else '❌'}")
            report.append(f"  样本数量: {len(result['samples'])}")
            report.append(f"  生成的Prompt数: {len(result['generated_prompts'])}")
            
            if result['errors']:
                report.append("  错误信息:")
                for error in result['errors']:
                    report.append(f"    - {error}")
                    
        return "\n".join(report)

def validate_config(config_file: str, num_samples: int = 2):
    """验证配置文件的便捷函数"""
    validator = OpenCompassConfigValidator(config_file)
    results = validator.validate_all_datasets(num_samples)
    report = validator.generate_report(results)
    print(report)
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        validate_config(config_file, num_samples)
    else:
        print("用法: python config_validator.py <配置文件路径> [样本数量]")
