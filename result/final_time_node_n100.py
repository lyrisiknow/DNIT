import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.ticker as ticker

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

time_pattern = re.compile(r'^LFR_(.+)_n(\d+)$')
data_file_path = 'time_process.jsonl'
fn = 16

# 1. 读取样式配置，确定模型顺序
model_styles = {}
ordered_model_names = []
try:
    with open('model_style_config.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            m_name = entry['model_name']
            if m_name not in model_styles:
                ordered_model_names.append(m_name)
                style = entry['linestyle']
                if isinstance(style, str) and style.startswith('('):
                    style = eval(style)
                model_styles[m_name] = {'color': entry['color'], 'marker': entry['marker'], 'linestyle': style}
except FileNotFoundError:
    print("未找到样式文件")

# 2. 读取数据 (存储两个区间)
data_map_small = defaultdict(dict) # 100-300
data_map_large = defaultdict(dict) # 1000-3000

with open(data_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            obj = json.loads(line)
            for key, val in obj.items():
                match = time_pattern.match(key)
                if match:
                    model_name = match.group(1)
                    n_val = int(match.group(2))
                    if model_name in model_styles:
                        if 100 <= n_val <= 300:
                            data_map_small[model_name][n_val] = val
                        elif 1000 <= n_val <= 3000:
                            data_map_large[model_name][n_val] = val
        except: continue

# 3. 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4.8))

# 增加了 x_spacing 参数用来控制横坐标分度值
def plot_data(ax, data_map, title, x_spacing):
    lines_handles = []
    labels_list = []
    for model_name in ordered_model_names:
        if model_name not in data_map: continue
        points = data_map[model_name]
        sorted_n = sorted(points.keys())
        sorted_times = [points[n] for n in sorted_n]
        style = model_styles[model_name]
        line, = ax.plot(sorted_n, sorted_times, color=style['color'], marker=style['marker'], 
                        markersize=5.5, linewidth=1.8, linestyle=style['linestyle'], label=model_name)
        lines_handles.append(line)
        labels_list.append(model_name)
    ax.tick_params(axis='both', labelsize=fn - 2)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    
    # --- 核心修改：设置横坐标主要刻度间隔 ---
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_spacing))
    
    ax.set_title(title, fontsize=fn)
    ax.set_xlabel('Network size', fontsize=fn)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    return lines_handles, labels_list

# 分别传入分度值 50 和 500
handles1, labels1 = plot_data(ax1, data_map_small, 'LFR1-5', x_spacing=50)
handles2, labels2 = plot_data(ax2, data_map_large, 'LFR6-10', x_spacing=500)

ax1.set_ylabel('Time (seconds)', fontsize=fn)

num_models = len(labels1)
fig.legend(
    handles=handles1, 
    labels=labels1, 
    loc='upper center', 
    ncol=num_models,          # 关键点：ncol 等于模型数量，即单行显示
    bbox_to_anchor=(0.5, 1.0), # 调整高度，确保不遮挡子图标题
    frameon=True,
    fontsize=fn - 2           # 若模型较多，单行可能需要缩减字体大小
)

# 调整子图布局，给单行图例留出上方空间
plt.subplots_adjust(top=0.80, bottom=0.15, left=0.08, right=0.98, hspace=0.2, wspace=0.25)
plt.savefig('time_node_n100.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.savefig('time_node_n100.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.show()