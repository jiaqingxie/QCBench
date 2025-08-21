import requests
import json
import time
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict

Baseurl =  your_url
Skey = your_api_key
url = Baseurl + "/v1/chat/completions"

headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {Skey}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json',
}

lock = Lock()

def get_answer_threadsafe(idx, item, max_retries=1):
    question = item.get("question", "")
    
    for attempt in range(1, max_retries + 1):
        payload = json.dumps({
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert chemist. Please read the following question and provide a step-by-step solution. "
                        "Your final answer must be presented as a readable LaTeX formula, enclosed in a \\boxed{} environment. "
                        "If the final answer is numerical, write only the numeric value inside \\boxed{}; place the unit immediately after the box (not inside), using the unit specified in the problem."
                    )
                },
                {
                    "role": "user",
                    "content": question + (
                        f" The unit of the final answer is {item.get('unit','').strip()}. "
                        "Do not put the unit inside the \\boxed{{}}; place it right after the box."
                        if item.get('unit','') and item.get('unit','').strip() else ""
                    )
                }
            ]
        })
        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=1500)
            response.raise_for_status()
            data = response.json()
            elapsed = time.time() - start_time
            llm_answer = data['choices'][0]['message']['content']
            break
        except Exception as e:
            elapsed = time.time() - start_time
            llm_answer = f"Error (attempt {attempt}): {str(e)}"

            if attempt == max_retries:
                break
            else:
                time.sleep(1)

    record = {
        "index": idx,
        "question": question,
        "gt_answer": item.get("answer", ""),
        "unit" : item.get("unit", ""),
        "reference": item.get("reference", ""),
        "source": item.get("source", ""),
        "class": item.get("class", ""),
        "llm_answer": llm_answer,
        "elapsed_time": round(elapsed, 2)
    }

    with lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    return idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM on orderly_qa dataset")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro-thinking-4096", help="LLM model name, e.g., o1, claude-2, etc.")
    parser.add_argument("--workers", type=int, default=30, help="Number of concurrent threads")
    args = parser.parse_args()
    model_name = args.model
    num_workers = args.workers

    with open('QCBench.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(len(dataset))
    safe_model_name = model_name.replace("/", "_")
    output_file = f"results/results_{safe_model_name}.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(output_file)
    final = list(range(1,351))

    index_to_item = {}
    for item in dataset:
        idx = item.get("index")
        if idx is not None:
            index_to_item[idx] = item

    cleaned_records = {}
    duplicate_indexes = defaultdict(list)

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    idx = data.get("index")
                    answer = data.get("llm_answer", "")
                    if answer.startswith("Error") or idx in cleaned_records:
                        duplicate_indexes[idx].append(line_num)
                        continue
                    cleaned_records[idx] = data
                except json.JSONDecodeError:
                    continue

        if cleaned_records:
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in cleaned_records.values():
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            print("⚠️ 清洗后无有效记录，原文件未覆盖")

    processed_indexes = set(cleaned_records.keys())
    print(processed_indexes)
    tasks = [(idx, item) for idx, item in index_to_item.items()
                if idx not in processed_indexes and idx in final]
    print(f"待处理数量: {len(tasks)}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(get_answer_threadsafe, idx, item) for idx, item in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"并发处理 ({model_name})"):
            pass

    print(f"\n✅ 所有处理完成，结果保存在 {output_file}") 