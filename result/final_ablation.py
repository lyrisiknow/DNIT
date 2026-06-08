import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- [1. 全局样式设置] ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
fn = 16  # 统一字体大小

# --- [2. 自定义学术配色区分 auc 和 auc-] ---
# 选用经典的深蓝与暖橘搭配，对比清晰且美观
COLOR_AUC = '#4048c0'        # 深蓝色 (代表标准 auc)
COLOR_AUC_MINUS = '#2aa598'  # 暖橘色 (代表 auc-)
OPACITY = 0.85

# --- [3. 数据读取与解析] ---
# 结构: data_map[n_value]['auc'] = val, data_map[n_value]['auc-'] = val
data_map = defaultdict(dict)

# 精准匹配 mymodel 的正则，捕获 n 的数值以及是否有后缀 '-'
pattern = re.compile(r'^LFR_ComInf_n(\d+)auc(-?)$')

with open('result_process.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
            for key, val in obj.items():
                match = pattern.match(key)
                if match:
                    n_val = int(match.group(1))
                    suffix = match.group(2)  # '-' 或者 ''
                    
                    type_key = 'auc-' if suffix == '-' else 'auc'
                    
                    # 取列表最右边的值作为 pr-auc
                    if isinstance(val, list) and len(val) > 0:
                        pr_auc = val[-1]
                    else:
                        pr_auc = float(val)
                        
                    data_map[n_val][type_key] = pr_auc
        except Exception as e:
            print(f"Error parsing line: {line}. Exception: {e}")

# --- [4. 拆分左右子图的 n 值群组] ---
all_n_values = sorted(list(data_map.keys()))
n_group_left = [n for n in all_n_values if 100 <= n <= 300]
n_group_right = [n for n in all_n_values if 1000 <= n <= 3000]

# --- [5. 开始创建画布] ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
bar_width = 0.35  # 柱状图宽度

def draw_sub_barplot(ax, n_list, title_text):
    """在指定子图上绘制 mymodel 的并列柱状图"""
    if not n_list:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
        return []
    
    # 离散化 X 轴刻度基础位置
    x_indices = np.arange(len(n_list))
    
    # 提取对应的数值
    auc_vals = [data_map[n].get('auc', 0) for n in n_list]
    auc_minus_vals = [data_map[n].get('auc-', 0) for n in n_list]
    
    # 【细节优化点】：定义一个带透明度的半透明黑色，让暗纹和轮廓变柔和
    # #000000 是纯黑，后面的 50 代表十六进制透明度（相当于约 30% 的不透明度）
    LIGHT_EDGE = '#00000050' 
    
    # 绘制两组并列的柱子
    # 1. 轮廓颜色 edgecolor 改为 LIGHT_EDGE，此时暗纹会自动跟着变浅
    # 2. 如果希望线条依然有分量，linewidth 保持 1.5 即可
    rects1 = ax.bar(x_indices - bar_width/2, auc_vals, bar_width,
                    alpha=OPACITY, color=COLOR_AUC, edgecolor=LIGHT_EDGE, 
                    linewidth=1.5, hatch='//', label='auc')
    rects2 = ax.bar(x_indices + bar_width/2, auc_minus_vals, bar_width,
                    alpha=OPACITY, color=COLOR_AUC_MINUS, edgecolor=LIGHT_EDGE, 
                    linewidth=1.5, hatch='\\\\', label='auc-')
    
    # 坐标轴与标签美化
    ax.set_xlabel('Network size', fontsize=fn, labelpad=6)
    ax.set_ylabel('PR-AUC score', fontsize=fn, labelpad=6)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(n) for n in n_list], fontsize=fn-2)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylim(bottom=0)
    
    # 移除上方和右侧的边框（保持坐标轴本身是清晰的纯黑）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#000000')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_color('#000000')
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(False)
    
    return [rects1, rects2]

# 分别绘制左、右子图
handles = draw_sub_barplot(ax1, n_group_left, 'Small-Scale Networks ($n \\in [100, 300]$)')
draw_sub_barplot(ax2, n_group_right, 'Large-Scale Networks ($n \\in [1000, 3000]$)')

# --- [6. 全局图例与白边优化] ---
if handles:
    fig.legend(handles=handles, labels=['ComInf', 'ComInf w/o Community Prior'], 
               loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02),
               frameon=True, facecolor='white', edgecolor='none', fontsize=fn-2)

# 调整子图间距，确保紧凑无白边
plt.subplots_adjust(top=0.80, bottom=0.15, left=0.08, right=0.98, hspace=0.2, wspace=0.25)

# 保存图片
plt.savefig('mymodel_auc_comparison.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.savefig('mymodel_auc_comparison.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)

plt.show()