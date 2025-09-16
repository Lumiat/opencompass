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
                        if k not in ['type', 'infer_cfg', 'eval_cfg', 'abbr']}
        
        dataset = dataset_class(**dataset_kwargs)
        result['dataset_loaded'] = True
        print("✅ 数据集加载成功")  # 不显示长度
        
        # 2. 获取样本数据 - 简单直接
        samples_collected = 0
        try:
            for i in range(num_samples):
                sample = dataset[i]  # 直接索引访问
                result['samples'].append(dict(sample))
                samples_collected += 1
                
            print(f"📋 成功获取 {samples_collected} 个样本")
            
        except Exception as e:
            result['errors'].append(f"获取样本数据失败: {str(e)}")
            print(f"❌ 获取样本失败: {e}")
            return result
        
        # 3. 检查数据结构
        if result['samples']:
            sample_data = result['samples'][0]
            actual_columns = list(sample_data.keys())
            print(f"📋 实际数据列: {actual_columns}")
            
            # 验证列名匹配逻辑...
            
        # 其余验证逻辑保持不变...
        
    except Exception as e:
        result['errors'].append(f"验证过程出错: {str(e)}")
        print(f"❌ 验证失败: {e}")
        
    return result
