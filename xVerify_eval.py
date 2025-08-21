import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, List, Tuple

from src.xVerify.model import Model
from src.xVerify.eval import Evaluator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def load_converted_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def initialize_xverify_model(model_name: str = 'xVerify-0.5B-I', 
                           url: str = 'IAAR-Shanghai/xVerify-0.5B-I',
                           inference_mode: str = 'local',
                           api_key: str = None):
    try:
        model = Model(
            model_name=model_name,
            model_path_or_url=url,
            inference_mode=inference_mode,
            api_key=api_key
        )
        evaluator = Evaluator(model=model, process_num=1)
        print(f"✅ xVerify模型初始化成功: {model_name} (单进程模式)")
        return evaluator
    except Exception as e:
        print(f"❌ xVerify模型初始化失败: {e}")
        return None

def batch_evaluate(evaluator: Evaluator, data_path: str, 
                   max_samples: int = None, output_path: str = None) -> Dict:
    try:
        print(f"🔄 开始批量评估: {data_path}")
        
        if output_path is None:
            output_path = "data/xverify_results"
        
        os.makedirs(output_path, exist_ok=True)
        
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        
        original_data = load_converted_data(data_path)
        if max_samples:
            original_data = original_data[:max_samples]
        
        temp_data_path = data_path.replace('.json', '_temp.json')
        temp_data = []
        for item in original_data:
            temp_item = {
                "question": item["question"],
                "llm_output": item["llm_output"],
                "correct_answer": item["correct_answer"]
            }
            temp_data.append(temp_item)
        
        with open(temp_data_path, 'w', encoding='utf-8') as f:
            json.dump(temp_data, f, indent=2, ensure_ascii=False)
        
        results = evaluator.evaluate(
            data_path=temp_data_path,
            output_path=output_path,
            data_size=max_samples
        )
        
        if os.path.exists(temp_data_path):
            os.remove(temp_data_path)
        
        if isinstance(results, dict) and "results" in results:
            for i, result in enumerate(results["results"]):
                if i < len(original_data):
                    original_class = original_data[i].get("class", "")
                    if original_class in ["Biochemistry", "Organic"]:
                        result["class"] = "Biochemistry_Organic"
                    else:
                        result["class"] = original_class
                    result["index"] = original_data[i].get("index", i)
        
        print(f"✅ 批量评估完成")
        return results
        
    except Exception as e:
        print(f"❌ 批量评估失败: {e}")
        import traceback
        print(f"详细错误信息:")
        traceback.print_exc()
        return {"error": str(e)}

def save_results(results: List[Dict], output_file: str, model_name: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    output_data = {
        "model_name": model_name,
        "total_samples": len(results),
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 评估结果已保存到: {output_file}")

def analyze_results(results: Dict) -> Dict:
    if "error" in results:
        print(f"❌ 评估失败: {results['error']}")
        return results
    
    print(f"\n📊 评估结果分析:")
    print(f"评估结果: {results}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="使用xVerify本地模型评估转换后的数据")
    parser.add_argument("--input", "-i", type=str, 
                       default="data/converted_results/USD-guiji_deepseek-r1_converted.json",
                       help="输入文件路径")
    parser.add_argument("--output", "-o", type=str, 
                       default="data/xverify_results",
                       help="输出目录")
    parser.add_argument("--model-name", type=str, 
                       default="xVerify-0.5B-I",
                       help="xVerify模型名称")
    parser.add_argument("--url", type=str, 
                       default="IAAR-Shanghai/xVerify-0.5B-I",
                       help="Hugging Face模型标识符或本地模型路径")
    parser.add_argument("--inference-mode", type=str, 
                       default="local",
                       choices=["api", "local"],
                       help="推理模式")
    parser.add_argument("--api-key", type=str, 
                       default=None,
                       help="API密钥")
    parser.add_argument("--max-samples", type=int, 
                       default=1000,
                       help="最大评估样本数（用于测试）")
    parser.add_argument("--delay", type=float, 
                       default=0.0,
                       help="请求间隔延迟（秒，本地模式建议设为0）")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    print(f"📊 使用数据文件: {args.input}")
    
    print(f"🔧 正在初始化xVerify模型: {args.model_name}")
    evaluator = initialize_xverify_model(
        model_name=args.model_name,
        url=args.url,
        inference_mode=args.inference_mode,
        api_key=args.api_key
    )
    
    if evaluator is None:
        print("❌ 模型初始化失败，退出程序")
        return
    
    results = batch_evaluate(
        evaluator=evaluator,
        data_path=args.input,
        max_samples=args.max_samples,
        output_path=args.output
    )
    
    analysis = analyze_results(results)
    
    print(f"🎯 评估完成！")
    print(f"📄 结果已保存到: {args.output}")

if __name__ == "__main__":
    main() 