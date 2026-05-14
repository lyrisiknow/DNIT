import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # 导入刻度工具
from collections import defaultdict

def plot_metrics(file_path):
    data_map = defaultdict(dict)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                for key, values in record.items():
                    parts = key.split('_')
                    if len(parts) < 3: continue
                    
                    last_part = parts[-1]
                    if last_part.startswith('n') and last_part.endswith('auc'):
                        method_name = parts[1]
                        n_str = last_part[1:-3] 
                        try:
                            n_val = int(n_str)
                            if n_val > 500: continue # 过滤大于500的点
                            data_map[method_name][n_val] = values
                        except ValueError: continue

        if not data_map:
            print("未找到符合条件的数据。")
            return

        # 获取所有出现过的 n 值，用来确定刻度范围
        all_n = []
        for points in data_map.values():
            all_n.extend(points.keys())
        max_n = max(all_n) if all_n else 500

        # ---------------- 绘图函数 (减少重复代码) ----------------
        def format_ax(ax, title, ylabel):
            ax.set_xlabel('Number of Nodes (n)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # --- 关键修改：设置横坐标每 100 一个刻度 ---
            ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
            # 限制 x 轴范围，防止最后留白过多
            ax.set_xlim(left=min(all_n)-20, right=max_n+20)

        # 绘图 1: ROC-AUC
        plt.figure(figsize=(10, 6))
        for method, points in data_map.items():
            sorted_n = sorted(points.keys())
            sorted_roc = [points[n][0] for n in sorted_n]
            plt.plot(sorted_n, sorted_roc, marker='o', label=method)
        format_ax(plt.gca(), 'ROC-AUC vs n', 'ROC-AUC Score')
        plt.savefig('roc_auc_analysis100.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 绘图 2: PR-AUC
        plt.figure(figsize=(10, 6))
        for method, points in data_map.items():
            sorted_n = sorted(points.keys())
            sorted_pr = [points[n][1] for n in sorted_n]
            plt.plot(sorted_n, sorted_pr, marker='s', linestyle='--', label=method)
        format_ax(plt.gca(), 'PR-AUC vs n', 'PR-AUC Score')
        plt.savefig('pr_auc_analysis100.png', dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"发生错误: {e}")

plot_metrics('result.jsonl')