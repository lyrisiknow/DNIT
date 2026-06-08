import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

# --- [全局样式设置] ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
fn = 16  # 统一标签字体大小

# 1. 定义 5 个数据集的配置
datasets = [
    {
        "name": "LFR (Small)",
        "auc_file": "result_process.jsonl",
        "time_file": "time_process.jsonl",
        "auc_regex": r'^LFR_(.+)_n100p(\d+)auc$',
        "time_regex": r'^LFR_(.+)_n100p(\d+)$'
    },
    {
        "name": "LFR (Medium)",
        "auc_file": "result_process.jsonl",
        "time_file": "time_process.jsonl",
        "auc_regex": r'^LFR_(.+)_n1000p(\d+)auc$',
        "time_regex": r'^LFR_(.+)_n1000p(\d+)$'
    },
    {
        "name": "Email",
        "auc_file": "result_process.jsonl",
        "time_file": "time_process.jsonl",
        "auc_regex": r'^email_(.+)_p(\d+)auc$',
        "time_regex": r'^email_(.+)_p(\d+)$'
    },
    {
        "name": "Workplace",
        "auc_file": "result_process.jsonl",
        "time_file": "time_process.jsonl",
        "auc_regex": r'^workplace_(.+)_p(\d+)auc$',
        "time_regex": r'^workplace_(.+)_p(\d+)$'
    },
    {
        "name": "mastodon",
        "auc_file": "result_process.jsonl",
        "time_file": "time_process.jsonl",
        "auc_regex": r'^mastodon_(.+)_p(\d+)auc$',
        "time_regex": r'^mastodon_(.+)_p(\d+)$'
    }
]

# 2. 读取通用的样式配置
ordered_model_names = []
model_styles = {}
try:
    with open('model_style_config.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            m_name = entry['model_name']
            ordered_model_names.append(m_name)
            style = entry['linestyle']
            if isinstance(style, str) and style.startswith('('):
                style = eval(style)
            model_styles[m_name] = {'color': entry['color'], 'marker': entry['marker'], 'linestyle': style}
except FileNotFoundError:
    print("警告: 未找到 model_style_config.jsonl，将使用默认样式。")

# 3. 初始化画布：修改为 3 行 5 列，调整 figsize 适应高度
fig, axes = plt.subplots(3, 5, figsize=(22, 11))

# 用于收集全局图例的句柄
legend_handles = {}

# 4. 开始横向循环 5 个数据集
for col_idx, ds in enumerate(datasets):
    data_map = defaultdict(lambda: defaultdict(dict))
    all_existing_counts = set()
    
    auc_pattern = re.compile(ds["auc_regex"])
    time_pattern = re.compile(ds["time_regex"])
    
    # 读取当前数据集的 AUC 数据
    try:
        with open(ds["auc_file"], 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                for key, val in obj.items():
                    match = auc_pattern.match(key)
                    if match:
                        model_name, process_count = match.group(1), int(match.group(2))
                        # 新增：提取 roc_auc (第一个值) 和 pr_auc (第二个值)
                        data_map[model_name][process_count]['roc_auc'] = val[0]
                        data_map[model_name][process_count]['pr_auc'] = val[1]
                        all_existing_counts.add(process_count)
    except FileNotFoundError:
        print(f"跳过：未找到文件 {ds['auc_file']}")
        continue

    # 读取当前数据集的 Time 数据
    try:
        with open(ds["time_file"], 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                for key, val in obj.items():
                    match = time_pattern.match(key)
                    if match:
                        model_name, process_count = match.group(1), int(match.group(2))
                        data_map[model_name][process_count]['time'] = val
                        all_existing_counts.add(process_count)
    except FileNotFoundError:
        print(f"跳过：未找到文件 {ds['time_file']}")
        continue

    active_models = [m for m in ordered_model_names if m in data_map]
    if not active_models:
        active_models = sorted(data_map.keys())
    num_models = len(active_models)
    
    sorted_x_ticks = sorted(list(all_existing_counts))
    x_indices = np.arange(len(sorted_x_ticks))
    total_width = 0.8
    bar_width = total_width / max(num_models, 1)

    # 确定当前的 上、中、下 三个子图轴
    ax_roc = axes[0, col_idx]   # 第一排：ROC-AUC (新增)
    ax_pr = axes[1, col_idx]    # 第二排：PR-AUC
    ax_time = axes[2, col_idx]  # 第三排：Time

    # 5. 渲染当前数据集的模型数据
    for i, model_name in enumerate(active_models):
        points = data_map[model_name]
        style = model_styles.get(model_name, {'color': 'black', 'marker': 'o', 'linestyle': '-'})
        
        # 过滤出同时拥有 auc 和 time 的有效 X 轴节点
        valid_counts = sorted([c for c in points.keys() if 'roc_auc' in points[c] and 'pr_auc' in points[c] and 'time' in points[c]])
        
        if valid_counts:
            # --- 第一排 (ax_roc): ROC-AUC 折线图 ---
            sorted_roc = [points[c]['roc_auc'] for c in valid_counts]
            line1, = ax_roc.plot(valid_counts, sorted_roc, color=style['color'], marker=style['marker'], 
                                 markersize=6, linewidth=2.0, linestyle=style['linestyle'], label=model_name)
            if model_name not in legend_handles:
                legend_handles[model_name] = line1

            # --- 第二排 (ax_pr): PR-AUC 折线图 ---
            sorted_pr = [points[c]['pr_auc'] for c in valid_counts]
            ax_pr.plot(valid_counts, sorted_pr, color=style['color'], marker=style['marker'], 
                       markersize=6, linewidth=2.0, linestyle=style['linestyle'], label=model_name)

        # --- 第三排 (ax_time): 柱状图 ---
        sorted_time = [points.get(c, {}).get('time', 0) for c in sorted_x_ticks]
        offset = (i - num_models / 2) * bar_width + bar_width / 2
        ax_time.bar(x_indices + offset, sorted_time, width=bar_width, 
                    color=style['color'], edgecolor='black', linewidth=0.6, alpha=0.9, label=model_name)

    # 6. 单个子图细节美化与顶会规范
    # 第一排 (ROC-AUC) 设置
    ax_roc.set_title(ds["name"], fontsize=fn+2, fontweight='bold', pad=10) # 保持在最上方
    ax_roc.set_xlabel('Process Count', fontsize=fn, labelpad=6)
    ax_roc.set_xticks(sorted_x_ticks)
    if col_idx == 0:
        ax_roc.set_ylabel('ROC-AUC Score', fontsize=fn, labelpad=6)
        
    # 第二排 (PR-AUC) 设置
    ax_pr.set_xlabel('Process Count', fontsize=fn, labelpad=6)
    ax_pr.set_xticks(sorted_x_ticks)
    if col_idx == 0:
        ax_pr.set_ylabel('PR-AUC Score', fontsize=fn, labelpad=6)
        
    # 第三排 (Time) 设置
    ax_time.set_xlabel('Process Count', fontsize=fn, labelpad=6)
    ax_time.set_xticks(x_indices)
    ax_time.set_xticklabels(sorted_x_ticks)
    ax_time.set_yscale('log')
    ax_time.yaxis.set_minor_locator(ticker.NullLocator())
    if col_idx == 0:
        ax_time.set_ylabel('Time (seconds)', fontsize=fn, labelpad=6)

    # 统一样式
    for ax in [ax_roc, ax_pr, ax_time]:
        ax.tick_params(axis='both', labelsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.grid(False)

# --- 7. 全局图例与紧凑布局优化 ---
ordered_labels = [m for m in ordered_model_names if m in legend_handles]
ordered_handles = [legend_handles[m] for m in ordered_labels]

fig.legend(handles=ordered_handles, labels=ordered_labels, loc='upper center', 
           ncol=len(ordered_labels), bbox_to_anchor=(0.5, 1.03), frameon=True, fontsize=fn-1)

# 微调间距，防止多出一排后横纵向文字挤压
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.98, hspace=0.42, wspace=0.24)

# 导出图片
plt.savefig('process.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.savefig('process.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.show()