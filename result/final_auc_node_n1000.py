import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

auc_pattern = re.compile(r'^LFR_(.+)_n(\d+)auc$')
auc_file_path = 'result_process.jsonl'
fn = 16

# 1. 先读取样式配置，确定模型顺序
model_styles = {}
ordered_model_names = [] # 明确定义顺序列表

try:
    with open('model_style_config.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            model_name = entry['model_name']
            
            # 确保每个模型只记录一次顺序
            if model_name not in model_styles:
                ordered_model_names.append(model_name)
                style = entry['linestyle']
                if isinstance(style, str) and style.startswith('('):
                    style = eval(style)
                model_styles[model_name] = {
                    'color': entry['color'],
                    'marker': entry['marker'],
                    'linestyle': style
                }
except FileNotFoundError:
    print("警告: 未找到 model_style_config.jsonl")

# 2. 读取数据
data_map = defaultdict(dict)
all_existing_n = set()

with open(auc_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
            for key, val in obj.items():
                match = auc_pattern.match(key)
                if match:
                    model_name = match.group(1)
                    n_val = int(match.group(2))
                    
                    # 只处理在样式文件中存在的模型
                    if model_name in model_styles and 1000 <= n_val <= 3000:
                        if n_val not in data_map[model_name]:
                            data_map[model_name][n_val] = {}
                        data_map[model_name][n_val]['roc_auc'] = val[0]
                        data_map[model_name][n_val]['pr_auc'] = val[1]
                        all_existing_n.add(n_val)
        except json.JSONDecodeError:
            pass

# 3. 绘图 (严格按照 ordered_model_names 顺序)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
sorted_x_ticks = sorted(list(all_existing_n))

lines_handles = []
labels_list = []

for model_name in ordered_model_names:
    if model_name not in data_map: continue # 跳过没有数据的模型
    
    points = data_map[model_name]
    sorted_n = sorted([n for n in points.keys()])
    sorted_roc = [points[n]['roc_auc'] for n in sorted_n]
    sorted_pr = [points[n]['pr_auc'] for n in sorted_n]
    
    style = model_styles[model_name]
    
    line1, = ax1.plot(sorted_n, sorted_roc, 
                      color=style['color'], marker=style['marker'], markersize=5.5, 
                      linewidth=1.8, linestyle=style['linestyle'], label=model_name)
    
    ax2.plot(sorted_n, sorted_pr, 
             color=style['color'], marker=style['marker'], markersize=5.5, 
             linewidth=1.8, linestyle=style['linestyle'], label=model_name)
    
    lines_handles.append(line1)
    labels_list.append(model_name)

# 设置轴标签
for ax, title in zip([ax1, ax2], ['ROC-AUC score', 'PR-AUC score']):
    ax.set_xlabel('Network size', fontsize=fn, labelpad=6)
    ax.set_ylabel(title, fontsize=fn, labelpad=6)
    ax.set_xticks(sorted_x_ticks)
    # ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', labelsize=fn-2)
    # 设置边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

# 图例
num_models = len(labels_list)
ncol_val = (num_models + 1) // 2  # 自动计算，让图例分为两行

fig.legend(
    handles=lines_handles, 
    labels=labels_list, 
    loc='upper center', 
    ncol=num_models,             # 设置为总数的一半，强制换行
    bbox_to_anchor=(0.5, 1.0), # 向上稍微移出绘图区，防止遮挡
    frameon=True, 
    fontsize=fn - 2            # 略微调小字体，防止溢出
)

# 调整子图布局，留出更多上方空间给两行图例
plt.subplots_adjust(top=0.80, bottom=0.15, left=0.08, right=0.98, hspace=0.2, wspace=0.25)
plt.savefig('roc_pr_auc_n1000.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.savefig('roc_pr_auc_n1000.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.show()