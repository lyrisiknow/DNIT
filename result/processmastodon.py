import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# 1. 初始化数据结构：{ 模型名称: { 次数: (roc_auc, pr_auc) } }
data_map = defaultdict(dict)

# 你的 jsonl 文件路径
file_path = 'result_process.jsonl' 

# 匹配 "LFR_模型名称_n1000p次数auc" 的正则表达式
pattern = re.compile(r'^mastodon_(.+)_p(\d+)auc$')

# 2. 读取并解析 JSONL 文件
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            for key, val in obj.items():
                match = pattern.match(key)
                if match:
                    model_name = match.group(1)       # 提取模型名称
                    process_count = int(match.group(2)) # 提取次数
                    roc_auc = val[0]                  # 第一个值：ROC-AUC
                    pr_auc = val[1]                   # 第二个值：PR-AUC
                    
                    # 以元组形式保存两个指标
                    data_map[model_name][process_count] = (roc_auc, pr_auc)
        except json.JSONDecodeError:
            print(f"跳过无效的JSON行: {line}")

# 3. 创建画布：1行2列的子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 4. 遍历数据并绘图
for model_name, points in data_map.items():
    # 将次数排序，确保 X 轴顺序正确
    sorted_counts = sorted(points.keys())
    
    # 分别提取排序后的 ROC-AUC 和 PR-AUC 列表
    sorted_roc = [points[c][0] for c in sorted_counts]
    sorted_pr = [points[c][1] for c in sorted_counts]
    
    # 在左边子图画 ROC-AUC
    ax1.plot(sorted_counts, sorted_roc, marker='o', label=model_name)
    
    # 在右边子图画 PR-AUC
    ax2.plot(sorted_counts, sorted_pr, marker='s', linestyle='--', label=model_name)

# 5. 美化左子图 (ROC-AUC)
ax1.set_title('Model Performance: ROC-AUC', fontsize=14, fontweight='bold')
ax1.set_xlabel('Process Count (p)', fontsize=12)
ax1.set_ylabel('ROC-AUC Score', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(title='Models', loc='best')

# 6. 美化右子图 (PR-AUC)
ax2.set_title('Model Performance: PR-AUC', fontsize=14, fontweight='bold')
ax2.set_xlabel('Process Count (p)', fontsize=12)
ax2.set_ylabel('PR-AUC Score', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(title='Models', loc='best')

# 7. 调整布局并展示
plt.tight_layout()
plt.show()
# 如果需要保存图片，可以取消下行注释
plt.savefig('auc_process_mastodon.png', dpi=300)