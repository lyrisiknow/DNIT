import json
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict
import matplotlib

# 设置学术字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

data_map = defaultdict(lambda: defaultdict(dict))

# 文件路径配置
auc_file_path = 'result_process.jsonl' 
time_file_path = 'time_process.jsonl'           

auc_pattern = re.compile(r'^LFR_(.+)_n1000p(\d+)auc$')
time_pattern = re.compile(r'^LFR_(.+)_n1000p(\d+)$')

all_existing_counts = set()

# 数据读取逻辑
for path, pattern, key_name in [(auc_file_path, auc_pattern, 'pr_auc'), (time_file_path, time_pattern, 'time')]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                for key, val in obj.items():
                    match = pattern.match(key)
                    if match:
                        model_name = match.group(1)
                        process_count = int(match.group(2))
                        data_map[model_name][process_count][key_name] = val[1] if key_name == 'pr_auc' else val
                        all_existing_counts.add(process_count)
            except json.JSONDecodeError:
                pass

# --- 核心修改：固定颜色与线型映射 ---
model_names = sorted(data_map.keys())
# 使用 tab10 配色方案，自动对模型数量进行重采样
academic_colors = [
    '#1F77B4', 
    '#D62728', 
    "#308F30", 
    "#694789",
    "#736E6E",
    "#C36184",
    "#D68636", # 柔和橙
    '#E15759', # 柔和红
    '#76B7B2', # 蓝绿色
    '#59A14F', # 绿色
    '#EDC948', # 柔和黄
    '#B07AA1', # 紫色
    '#FF9DA7', # 粉色
    '#9C755F', # 棕色
    '#BAB0AC'  # 灰色
]

line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 5)), (0, (1, 1))]
markers = ['o', 's', '^', 'D', 'v', 'p', 'h']

color_map = {name: academic_colors[i % len(academic_colors)] for i, name in enumerate(model_names)}
style_map = {name: line_styles[i % len(line_styles)] for i, name in enumerate(model_names)}
marker_map = {name: markers[i % len(markers)] for i, name in enumerate(model_names)}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
sorted_x_ticks = sorted(list(all_existing_counts))

lines_handles = []
labels_list = []

# 绘图循环
for model_name in model_names:
    points = data_map[model_name]
    valid_counts = sorted([c for c in points.keys() if 'pr_auc' in points[c] and 'time' in points[c]])
    if not valid_counts: continue
        
    sorted_pr = [points[c]['pr_auc'] for c in valid_counts]
    sorted_time = [points[c]['time'] for c in valid_counts]
    
    # 使用统一映射
    props = {
        'color': color_map[model_name],
        'marker': marker_map[model_name],
        'linestyle': style_map[model_name],
        'markersize': 5.5,
        'linewidth': 1.2,
        'label': model_name
    }
    
    line1, = ax1.plot(valid_counts, sorted_pr, **props)
    ax2.plot(valid_counts, sorted_time, **props)
    
    lines_handles.append(line1)
    labels_list.append(model_name)

# 绘图美化
ax1.set_title('Model Performance: PR-AUC', fontsize=11, fontweight='bold', pad=12)
ax1.set_xlabel('Process Count (p)', fontsize=10, labelpad=6)
ax1.set_ylabel('PR-AUC Score', fontsize=10, labelpad=6)
ax1.set_xticks(sorted_x_ticks)                                  
ax1.set_ylim(bottom=0)                                           
ax1.tick_params(axis='both', labelsize=9)                        
ax1.grid(False)                                                  

ax2.set_title('Computational Efficiency: Time', fontsize=11, fontweight='bold', pad=12)
ax2.set_xlabel('Process Count (p)', fontsize=10, labelpad=6)
ax2.set_ylabel('Time (seconds)', fontsize=10, labelpad=6)
ax2.set_yscale('log')                                            
ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,))) 
ax2.set_xticks(sorted_x_ticks)                                   
ax2.tick_params(axis='both', labelsize=9)
ax2.grid(False)                                                  

for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)    
    ax.spines['right'].set_visible(False)   
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

fig.legend(handles=lines_handles, labels=labels_list, loc='upper center', 
           ncol=len(labels_list), bbox_to_anchor=(0.5, 0.95), 
           frameon=True, edgecolor='none', fontsize=9.5)

plt.subplots_adjust(top=0.76, bottom=0.16, left=0.10, right=0.92, hspace=0.2, wspace=0.30)
plt.show()
plt.savefig('auc_time_process_n1000.png', dpi=300)