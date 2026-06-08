import json
import re
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

auc_pattern = re.compile(r'^LFR_(.+)_n1000p(\d+)auc$')
time_pattern = re.compile(r'^LFR_(.+)_n1000p(\d+)$')

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

# 2. 绘图时按照 ordered_model_names 循环
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
sorted_x_ticks = sorted(list(all_existing_counts))
lines_handles, labels_list = [], []

# 这里不再循环 data_map.items()，而是循环有序的 model 列表
for model_name in ordered_model_names:
    if model_name not in data_map: continue
    
    points = data_map[model_name]
    valid_counts = sorted([c for c in points.keys() if 'pr_auc' in points[c] and 'time' in points[c]])
    if not valid_counts: continue
    
    sorted_pr = [points[c]['pr_auc'] for c in valid_counts]
    sorted_time = [points[c]['time'] for c in valid_counts]
    style = model_styles.get(model_name, {'color': 'black', 'marker': 'o', 'linestyle': '-'})
    
    line1, = ax1.plot(valid_counts, sorted_pr, color=style['color'], marker=style['marker'], 
                      markersize=5.5, linewidth=1.8, linestyle=style['linestyle'], label=model_name)
    ax2.plot(valid_counts, sorted_time, color=style['color'], marker=style['marker'], 
             markersize=5.5, linewidth=1.8, linestyle=style['linestyle'], label=model_name)
    
    lines_handles.append(line1)
    labels_list.append(model_name)

# 轴设置
for ax, title in zip([ax1, ax2], ['PR-AUC Score', 'Time (seconds)']):
    ax.set_xlabel('Process Count', fontsize=fn, labelpad=6)
    ax.set_ylabel(title, fontsize=fn, labelpad=6)
    ax.set_xticks(sorted_x_ticks)
    ax.tick_params(axis='both', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(False)

ax2.set_yscale('log')
ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
ax2.yaxis.set_minor_locator(ticker.NullLocator())

# 图例与布局
num_models = len(labels_list)
fig.legend(handles=lines_handles, labels=labels_list, loc='upper center', ncol=num_models, 
           bbox_to_anchor=(0.5, 1.02), frameon=True, fontsize=fn-2)

plt.subplots_adjust(top=0.80, bottom=0.15, left=0.08, right=0.98, hspace=0.2, wspace=0.25)
plt.savefig('auc_time_process_n1000.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.savefig('auc_time_process_n1000.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.show()