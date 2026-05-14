import json
import matplotlib.pyplot as plt
import numpy as np

def plot_email_metrics(file_path):
    models = []
    roc_aucs = []
    pr_aucs = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                record = json.loads(line)
                for key, values in record.items():
                    # 匹配 email_模型_auc 格式
                    if key.startswith("workplace_") and key.endswith("_auc"):
                        # 提取模型名：去掉前缀 'email_' 和后缀 '_auc'
                        model_name = key[10:-4] 
                        
                        models.append(model_name)
                        roc_aucs.append(values[0])
                        pr_aucs.append(values[1])

        if not models:
            print("未找到匹配 email_..._auc 格式的数据。")
            return

        # --- 绘图逻辑 ---
        x = np.arange(len(models))  # 标签位置
        width = 0.35  # 条形图宽度

        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 绘制两组条形
        # 替换代码中的这两行
# 替换代码中的这两行
# 替换代码中的这两行
        rects1 = ax.bar(x - width/2, roc_aucs, width, label='ROC-AUC', color='#4A90E2')
        rects2 = ax.bar(x + width/2, pr_aucs, width, label='PR-AUC', color='#FF6B6B')

        # 添加文本标签、标题和自定义轴标签
        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Comparison (Workplace Dataset)')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()

        # 在条形上方数值标注 (可选)
        ax.bar_label(rects1, padding=3, fmt='%.3f', fontsize=9)
        ax.bar_label(rects2, padding=3, fmt='%.3f', fontsize=9)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存并显示
        plt.savefig('workplace_model_comparison.png', dpi=300)
        plt.show()
        print("图表已保存为: workplace_model_comparison.png")

    except Exception as e:
        print(f"解析或绘图时出错: {e}")

# 调用
plot_email_metrics('result.jsonl')