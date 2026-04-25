import json
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_f1_scores(file_path):
    # 使用字典存储数据：{ 方法名: { n_value: f1_score } }
    data_map = defaultdict(dict)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                # 解析 JSON 行
                record = json.loads(line)
                for key, values in record.items():
                    # 假设格式总是 "PREFIX_METHOD_nNUMBER"
                    # 1. 提取方法名：第一个 '_' 和最后一个 '_' 之间的部分
                    parts = key.split('_')
                    if len(parts) < 3:
                        continue
                        
                    method_name = parts[1]
                    
                    # 2. 提取 n 的数值：最后一个部分去掉开头的 'n'
                    n_str = parts[-1].replace('n', '')
                    try:
                        n_val = int(n_str)
                    except ValueError:
                        continue
                    
                    # 3. 过滤条件：n 在 1000 到 3000 之间
                    if 1000 <= n_val <= 3000:
                        f1_score = values[-1] # 取列表最后一个值
                        data_map[method_name][n_val] = f1_score

        # 开始绘图
        plt.figure(figsize=(10, 6))
        
        for method, points in data_map.items():
            # 对 n 进行排序，确保折线顺序正确
            sorted_n = sorted(points.keys())
            sorted_f1 = [points[n] for n in sorted_n]
            
            plt.plot(sorted_n, sorted_f1, marker='o', label=method)

        plt.xlabel('Value of n')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs n (100-300)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        plt.savefig('f1_score_analysis.png', dpi=300, bbox_inches='tight')

    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
    except Exception as e:
        print(f"发生错误: {e}")

# 调用函数，替换为你的文件名
plot_f1_scores('result.jsonl')