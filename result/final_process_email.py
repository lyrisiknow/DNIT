import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

# --- [全局设置] ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
fn = 16 # 统一下字体大小

data_map = defaultdict(lambda: defaultdict(dict))
auc_file_path = 'result_process.jsonl'
time_file_path = 'time_process.jsonl'

auc_pattern = re.compile(r'^email_(.+)_p(\d+)auc$')
time_pattern = re.compile(r'^email_(.+)_p(\d+)$')

all_existing_counts = set()

# 读取数据
with open(auc_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
            for key, val in obj.items():
                match = auc_pattern.match(key)
                if match:
                    model_name, process_count = match.group(1), int(match.group(2))
                    data_map[model_name][process_count]['pr_auc'] = val[1]
                    all_existing_counts.add(process_count)
        except: pass

with open(time_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
            for key, val in obj.items():
                match = time_pattern.match(key)
                if match:
                    model_name, process_count = match.group(1), int(match.group(2))
                    data_map[model_name][process_count]['time'] = val
                    all_existing_counts.add(process_count)
        except: pass

# 读取样式
ordered_model_names = []
model_styles = {}
try:
    with open('model_style_config.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            m_name = entry['model_name']
            ordered_model_names.append(m_name) # 记录读取顺序
            style = entry['linestyle']
            if isinstance(style, str) and style.startswith('('):
                style = eval(style)
            model_styles[m_name] = {'color': entry['color'], 'marker': entry['marker'], 'linestyle': style}
except FileNotFoundError:
    print("警告: 未找到 model_style_config.jsonl")
    ordered_model_names = sorted(data_map.keys()) # 如果没找到文件，按名称字母排序

# 过滤出真正有数据的有序模型列表，并确保总数（如你所说是7个）
active_models = [m for m in ordered_model_names if m in data_map]
num_models = len(active_models)

# 1. 初始化画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
sorted_x_ticks = sorted(list(all_existing_counts)) # 例如 [1, 2, 4, 8]

# 2. 柱状图的基础参数计算
x_indices = np.arange(len(sorted_x_ticks))  # X轴基础索引位置: [0, 1, 2, ...]
total_width = 0.8                           # 每组柱形占据的总宽度空间（小于1留出组间距）
bar_width = total_width / num_models        # 每一个单独立柱的宽度

lines_handles, labels_list = [], []

# 3. 循环渲染子图
for i, model_name in enumerate(active_models):
    points = data_map[model_name]
    
    # --- 子图 1 (ax1): 依然采用折线图表示 PR-AUC ---
    valid_counts = sorted([c for c in points.keys() if 'pr_auc' in points[c] and 'time' in points[c]])
    if not valid_counts: continue
    
    sorted_pr = [points[c]['pr_auc'] for c in valid_counts]
    style = model_styles.get(model_name, {'color': 'black', 'marker': 'o', 'linestyle': '-'})
    
    line1, = ax1.plot(valid_counts, sorted_pr, color=style['color'], marker=style['marker'], 
                      markersize=5.5, linewidth=1.8, linestyle=style['linestyle'], label=model_name)
    
    lines_handles.append(line1)
    labels_list.append(model_name)
    
    # --- 子图 2 (ax2): 改为并行分组柱状图表示 Time ---
    # 提取与 sorted_x_ticks 一一对应的时间数据，若缺失则设为 0
    sorted_time = [points.get(c, {}).get('time', 0) for c in sorted_x_ticks]
    
    # 计算当前模型在每个刻度上的水平偏移量（居中对齐处理）
    offset = (i - num_models / 2) * bar_width + bar_width / 2
    
    # 绘制柱状图，使用相同的颜色，并加入细黑边框（edgecolor）拔高顶会图表质感
    ax2.bar(x_indices + offset, sorted_time, width=bar_width, 
            color=style['color'], edgecolor='black', linewidth=0.6, alpha=0.9, label=model_name)

# --- 4. 坐标轴与细节微调 ---
# Ax1 (PR-AUC) 轴设置
ax1.set_xlabel('Process Count', fontsize=fn, labelpad=6)
ax1.set_ylabel('PR-AUC Score', fontsize=fn, labelpad=6)
ax1.set_xticks(sorted_x_ticks)

# Ax2 (Time) 柱状图轴设置
ax2.set_xlabel('Process Count', fontsize=fn, labelpad=6)
ax2.set_ylabel('Time (seconds)', fontsize=fn, labelpad=6)
ax2.set_xticks(x_indices)              # 将刻度定在索引位置 [0, 1, 2...]
ax2.set_xticklabels(sorted_x_ticks)    # 将索引标签替换为实际的 Process Count (如 1, 2, 4, 8)
ax2.set_yscale('log')                  # 依然保持对数坐标轴，以兼容耗时差距巨大的模型

# 统一边框和字体样式
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(False)

# 解除多余的 Locator 控制，交给 matplotlib log 轴自动优化渲染
ax2.yaxis.set_minor_locator(ticker.NullLocator())

# --- 5. 图例与全局布局 ---
fig.legend(handles=lines_handles, labels=labels_list, loc='upper center', ncol=num_models, 
           bbox_to_anchor=(0.5, 1.02), frameon=True, fontsize=fn-2)

plt.subplots_adjust(top=0.80, bottom=0.15, left=0.08, right=0.98, hspace=0.2, wspace=0.25)
plt.savefig('auc_time_process_email.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.savefig('auc_time_process_email.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.show()