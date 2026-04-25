import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_method_metrics(file_path):
    methods = []
    precision_list = []
    recall_list = []
    f1_list = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                record = json.loads(line)
                for key, values in record.items():
                    # 过滤逻辑：只处理类似 "email_METHOD" 这种不带 n1000 后缀的行
                    # 或者你可以根据 key 是否包含 "email" 前缀来判断
                    if "email_" in key and len(values) == 3:
                        # 提取方法名，例如 "email_PIND" -> "PIND"
                        method_name = key.split('_')[-1]
                        
                        methods.append(method_name)
                        precision_list.append(values[0])
                        recall_list.append(values[1])
                        f1_list.append(values[2])

        if not methods:
            print("未找到匹配 email_METHOD 格式的数据")
            return

        # 设置柱状图的位置
        x = np.arange(len(methods))  # 方法名的标签位置
        width = 0.25  # 每个柱子的宽度

        fig, ax = plt.subplots(figsize=(12, 7))

        # 绘制三组柱子
        rects1 = ax.bar(x - width, precision_list, width, label='Precision', color='#3498db')
        rects2 = ax.bar(x, recall_list, width, label='Recall', color='#2ecc71')
        rects3 = ax.bar(x + width, f1_list, width, label='F1-score', color='#e74c3c')

        # 添加文本说明
        ax.set_xlabel('Methods')
        ax.set_ylabel('Scores')
        ax.set_title('Comparison of Precision, Recall, and F1-score by Method')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()

        # 在柱子上方标注数值
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3点垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        fig.tight_layout()

        # 保存图片
        plt.savefig('method_comparison_bars.png', dpi=300)
        print("柱状图已保存为 method_comparison_bars.png")
        
        plt.show()

    except Exception as e:
        print(f"发生错误: {e}")

# 调用函数
plot_method_metrics('result.jsonl')