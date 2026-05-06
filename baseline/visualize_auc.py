import json
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_metrics(file_path):
    # 使用字典存储数据：{ 方法名: { n_value: [roc_auc, pr_auc] } }
    data_map = defaultdict(dict)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                record = json.loads(line)
                for key, values in record.items():
                    # 分割 Key，例如 "LFR_mymodel_n100auc" -> ['LFR', 'mymodel', 'n100auc']
                    parts = key.split('_')
                    if len(parts) < 3:
                        continue
                    
                    last_part = parts[-1] # 拿到 "n100auc"
                    
                    # 严格校验：必须以 'n' 开头，以 'auc' 结尾
                    if last_part.startswith('n') and last_part.endswith('auc'):
                        method_name = parts[1]
                        
                        # 提取中间的数字：去掉开头的 'n' (1位) 和 结尾的 'auc' (3位)
                        n_str = last_part[1:-3] 
                        
                        try:
                            n_val = int(n_str)
                            data_map[method_name][n_val] = values
                        except ValueError:
                            continue

        if not data_map:
            print("未找到符合条件的数据，请检查文件内容和格式。")
            return

        # ---------------- 绘图 1: ROC-AUC ----------------
        plt.figure(figsize=(10, 6))
        for method, points in data_map.items():
            sorted_n = sorted(points.keys())
            # 取列表第一个值作为 ROC-AUC
            sorted_roc = [points[n][0] for n in sorted_n]
            plt.plot(sorted_n, sorted_roc, marker='o', label=method)

        plt.xlabel('Number of Nodes (n)')
        plt.ylabel('ROC-AUC Score')
        plt.title('ROC-AUC vs n')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('roc_auc_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # ---------------- 绘图 2: PR-AUC ----------------
        plt.figure(figsize=(10, 6))
        for method, points in data_map.items():
            sorted_n = sorted(points.keys())
            # 取列表第二个值作为 PR-AUC
            sorted_pr = [points[n][1] for n in sorted_n]
            plt.plot(sorted_n, sorted_pr, marker='s', linestyle='--', label=method)

        plt.xlabel('Number of Nodes (n)')
        plt.ylabel('PR-AUC Score')
        plt.title('PR-AUC vs n')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('pr_auc_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
    except Exception as e:
        print(f"发生错误: {e}")

# 调用
plot_metrics('result.jsonl')