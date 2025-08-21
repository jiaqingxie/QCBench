import json
import os
import argparse
from collections import defaultdict
import swanlab
from typing import Dict, List, Tuple

def load_results(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_class_accuracy(data: List[Dict]) -> Dict[str, float]:
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for item in data:
        class_name = item.get("class", "Unknown")
        score = item.get("score", 0)
        
        if class_name in ["Biochemistry", "Organic"]:
            class_name = "BOC"
        
        class_stats[class_name]["total"] += 1
        if score == 1:
            class_stats[class_name]["correct"] += 1
    
    accuracies = {}
    for class_name, stats in class_stats.items():
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"]
            accuracies[class_name] = accuracy
        else:
            accuracies[class_name] = 0.0
    
    return accuracies

def calculate_overall_accuracy(data: List[Dict]) -> float:
    total_correct = sum(1 for item in data if item.get("score", 0) == 1)
    total_items = len(data)
    return total_correct / total_items if total_items > 0 else 0.0

def get_class_distribution(data: List[Dict]) -> Dict[str, int]:
    distribution = defaultdict(int)
    for item in data:
        class_name = item.get("class", "Unknown")
        if class_name in ["Biochemistry", "Organic"]:
            class_name = "BOC"
        distribution[class_name] += 1
    return dict(distribution)

def log_to_swanlab(accuracies: Dict[str, float], overall_acc: float, 
                   distribution: Dict[str, int], model_name: str):
    run = swanlab.init(
        experiment_name=f"QCBench_{model_name}_Analysis",
        config={
            "model": model_name,
            "total_samples": sum(distribution.values()),
            "num_classes": len(accuracies)
        }
    )
    
    for class_name, accuracy in accuracies.items():
        run.log({f"accuracy/{class_name}": accuracy})
        run.log({f"count/{class_name}": distribution.get(class_name, 0)})
    
    run.log({"accuracy/overall": overall_acc})
    
    for class_name, count in distribution.items():
        run.log({f"distribution/{class_name}": count})
    
    print(f"✅ 结果已记录到SwanLab，实验名称: ChemScope_{model_name}_Analysis")
    
    return run

def print_summary(accuracies: Dict[str, float], overall_acc: float, 
                 distribution: Dict[str, int], model_name: str):
    print(f"\n{'='*60}")
    print(f"ChemScope {model_name} 分析报告")
    print(f"{'='*60}")
    
    print(f"\n总体准确率: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"总样本数: {sum(distribution.values())}")
    print(f"类别数量: {len(accuracies)}")
    
    print(f"\n各类别准确率:")
    print(f"{'类别':<15} {'准确率':<10} {'样本数':<8} {'正确数':<8}")
    print(f"{'-'*50}")
    
    sorted_accuracies = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, accuracy in sorted_accuracies:
        count = distribution.get(class_name, 0)
        correct = int(accuracy * count)
        print(f"{class_name:<15} {accuracy:.4f} ({accuracy*100:.1f}%) {count:<8} {correct:<8}")
    
    print(f"\n{'='*60}")

def save_results_to_file(accuracies: Dict[str, float], overall_acc: float, 
                        distribution: Dict[str, int], model_name: str, output_dir: str = "reports"):
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "model": model_name,
        "overall_accuracy": overall_acc,
        "class_accuracies": accuracies,
        "class_distribution": distribution,
        "total_samples": sum(distribution.values()),
        "num_classes": len(accuracies)
    }
    
    output_file = os.path.join(output_dir, f"report_{model_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"📄 结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="分析ChemScope模型结果并记录到SwanLab")
    parser.add_argument("--input", "-i", type=str, 
                       default="data/results_acc/results_gpt-4o.json",
                       help="输入文件路径")
    parser.add_argument("--model", "-m", type=str, 
                       default="gpt-4o",
                       help="模型名称")
    parser.add_argument("--no-swanlab", action="store_true",
                       help="不记录到SwanLab")
    parser.add_argument("--output-dir", "-o", type=str,
                       default="reports",
                       help="输出目录")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 文件不存在: {args.input}")
        return
    
    print(f"📊 正在分析 {args.input}...")
    
    try:
        data = load_results(args.input)
        print(f"✅ 成功加载 {len(data)} 个样本")
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return
    
    class_accuracies = calculate_class_accuracy(data)
    overall_accuracy = calculate_overall_accuracy(data)
    class_distribution = get_class_distribution(data)
    
    print_summary(class_accuracies, overall_accuracy, class_distribution, args.model)
    
    save_results_to_file(class_accuracies, overall_accuracy, class_distribution, args.model, args.output_dir)
    
    if not args.no_swanlab:
        try:
            run = log_to_swanlab(class_accuracies, overall_accuracy, class_distribution, args.model)
            print(f"🎯 分析完成！")
        except Exception as e:
            print(f"❌ 记录到SwanLab失败: {e}")
            print("请确保已安装swanlab: pip install swanlab")
    else:
        print(f"🎯 分析完成！(未记录到SwanLab)")

if __name__ == "__main__":
    main() 