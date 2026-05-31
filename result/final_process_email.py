import json
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

data_map = defaultdict(lambda: defaultdict(dict))

auc_file_path = 'result_process.jsonl'  
time_file_path = 'time_process.jsonl'            

auc_pattern = re.compile(r'^email_(.+)_p(\d+)auc$')
time_pattern = re.compile(r'^email_(.+)_p(\d+)$')

all_existing_counts = set()

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
                    process_count = int(match.group(2))
                    pr_auc = val[1]  
                    
                    data_map[model_name][process_count]['pr_auc'] = pr_auc
                    all_existing_counts.add(process_count)
        except json.JSONDecodeError:
            pass

with open(time_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
            for key, val in obj.items():
                match = time_pattern.match(key)
                if match:
                    model_name = match.group(1)
                    process_count = int(match.group(2))
                    run_time = val   
                    
                    data_map[model_name][process_count]['time'] = run_time
                    all_existing_counts.add(process_count)
        except json.JSONDecodeError:
            pass

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

icdm_colors = ['#1F77B4', '#D62728', '#2CA02C', '#9467BD', '#7F7F7F', '#A0522D']
markers = ['o', 's', '^', 'D', 'v', 'p', 'h']
sorted_x_ticks = sorted(list(all_existing_counts))

lines_handles = []
labels_list = []

for idx, (model_name, points) in enumerate(data_map.items()):
    valid_counts = sorted([c for c in points.keys() if 'pr_auc' in points[c] and 'time' in points[c]])
    if not valid_counts:
        continue  
        
    sorted_pr = [points[c]['pr_auc'] for c in valid_counts]
    sorted_time = [points[c]['time'] for c in valid_counts]
    
    current_color = icdm_colors[idx % len(icdm_colors)]
    current_marker = markers[idx % len(markers)]
    
    line1, = ax1.plot(valid_counts, sorted_pr, 
                      color=current_color, 
                      marker=current_marker, 
                      markersize=5.5,          
                      linewidth=1,         
                      linestyle='--',        
                      label=model_name)
    
    ax2.plot(valid_counts, sorted_time, 
             color=current_color, 
             marker=current_marker, 
             markersize=5.5, 
             linewidth=1, 
             linestyle='--',        
             label=model_name)
    
    lines_handles.append(line1)
    labels_list.append(model_name)

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
ax2.yaxis.set_minor_locator(ticker.NullLocator())                  
ax2.set_xticks(sorted_x_ticks)                                     
ax2.tick_params(axis='both', labelsize=9)
ax2.grid(False)                                                    

for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)     
    ax.spines['right'].set_visible(False)   
    ax.spines['left'].set_color('#000000')  
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_color('#000000')
    ax.spines['bottom'].set_linewidth(1.2)

fig.legend(handles=lines_handles, 
           labels=labels_list, 
           loc='upper center', 
           ncol=len(labels_list), 
           bbox_to_anchor=(0.5, 0.95), 
           frameon=True, 
           facecolor='white', 
           edgecolor='none', 
           fontsize=9.5,
           title_fontsize=9.5)

plt.subplots_adjust(top=0.76, bottom=0.16, left=0.10, right=0.92, hspace=0.2, wspace=0.30)

plt.show()
plt.savefig('auc_time_process_email.png', dpi=300)