"""Nemenyi / Bonferroni-Dunn critical-difference analysis for the TREND
component replacement variants. Reproduces paper Figure 12 / Table 9
(trend variants; CD value and average ranks are reported in the paper).

Input (--input): Excel workbook in the layout of
`STMIXER在各个数据集中趋势项组件的替换实验.xlsx`
(sheet `Sheet1`, WITH a header row, i.e. pandas default header=0):
  - column `Models`: dataset names, forward-filled down the rows;
  - column `Unnamed: 1`: prediction horizon (96/192/336/720); `Avg.` rows are
    skipped;
  - one column per model holding MSE, with the following column holding MAE.

Output (--output-dir): Nemenyi and Bonferroni-Dunn CD diagram PDFs plus
per-test result workbooks. Run from the repo root:
    python evaluation/nemenyi_trend_component_experiment.py --input <xlsx> --output-dir <dir>
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

def parse_excel_data(file_path):
    """解析Excel文件，同时提取MSE和MAE数据"""
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    all_cols = df.columns.tolist()
    model_cols = [col for col in all_cols if not col.startswith('Unnamed') and col not in ['Models']]

    models_info = {}
    for i, col in enumerate(all_cols):
        if col in model_cols:
            models_info[col] = {'mse_idx': i, 'mae_idx': i+1}

    data_mse = []
    data_mae = []
    current_dataset = None

    for idx, row in df.iterrows():
        if idx == 0:
            continue

        dataset_name = row['Models']
        pred_len = row['Unnamed: 1']

        if pd.notna(dataset_name) and str(dataset_name) != 'NaN':
            current_dataset = dataset_name

        if pd.notna(pred_len) and pred_len != 'Avg.' and current_dataset is not None:
            config_name = f'{current_dataset}_{int(pred_len)}'
            
            row_data_mse = {'config': config_name, 'dataset': current_dataset, 'pred_len': int(pred_len)}
            row_data_mae = {'config': config_name, 'dataset': current_dataset, 'pred_len': int(pred_len)}

            for model_name, col_info in models_info.items():
                mse_val = row.iloc[col_info['mse_idx']]
                mae_val = row.iloc[col_info['mae_idx']]
                row_data_mse[model_name] = mse_val
                row_data_mae[model_name] = mae_val

            data_mse.append(row_data_mse)
            data_mae.append(row_data_mae)

    return pd.DataFrame(data_mse), pd.DataFrame(data_mae), list(models_info.keys())

def nemenyi_test(data_df_mse, data_df_mae, models):
    """使用MSE和MAE排名的均值进行检验"""
    valid_df_mse = data_df_mse.dropna(subset=models)
    valid_df_mae = data_df_mae.dropna(subset=models)
    
    common_configs = list(set(valid_df_mse['config']) & set(valid_df_mae['config']))
    valid_df_mse = valid_df_mse[valid_df_mse['config'].isin(common_configs)]
    valid_df_mae = valid_df_mae[valid_df_mae['config'].isin(common_configs)]
    
    n_valid = len(valid_df_mse)
    config_list = valid_df_mse['config'].tolist()
    
    if n_valid < 2:
        return None
    
    data_matrix_mse = valid_df_mse[models].values
    ranks_mse = np.zeros_like(data_matrix_mse)
    for i in range(len(data_matrix_mse)):
        ranks_mse[i] = rankdata(data_matrix_mse[i])
    
    data_matrix_mae = valid_df_mae[models].values
    ranks_mae = np.zeros_like(data_matrix_mae)
    for i in range(len(data_matrix_mae)):
        ranks_mae[i] = rankdata(data_matrix_mae[i])
    
    ranks_avg = (ranks_mse + ranks_mae) / 2.0
    avg_ranks = np.mean(ranks_avg, axis=0)
    
    k = len(models)
    N = n_valid
    
    q_005 = {2: 2.77, 3: 3.31, 4: 3.63, 5: 3.86, 6: 4.03, 7: 4.17, 8: 4.29, 9: 4.39, 10: 4.47}
    q_010 = {2: 2.34, 3: 2.81, 4: 3.10, 5: 3.32, 6: 3.48, 7: 3.62, 8: 3.74, 9: 3.85, 10: 3.94}
    
    q05 = q_005.get(k, 4.47)
    q10 = q_010.get(k, 3.94)
    
    cd_005 = q05 * np.sqrt(k * (k + 1) / (6 * N))
    cd_010 = q10 * np.sqrt(k * (k + 1) / (6 * N))
    
    friedman_stat, friedman_p = stats.friedmanchisquare(*[ranks_avg[:, i] for i in range(k)])
    
    results = {
        'models': models,
        'avg_ranks': avg_ranks,
        'cd_005': cd_005,
        'cd_010': cd_010,
        'friedman_stat': friedman_stat,
        'friedman_p': friedman_p,
        'valid_configs': config_list,
        'N': N,
        'k': k,
        'ranks_df': pd.DataFrame(ranks_avg, columns=models, index=valid_df_mse['config'].values),
        'ranks_mse': ranks_mse,
        'ranks_mae': ranks_mae
    }
    
    return results

def bonferroni_dunn_test(data_df_mse, data_df_mae, models, control_model='ST-Mixer', alpha=0.05):
    """使用MSE和MAE排名的均值进行Bonferroni-Dunn检验"""
    if control_model not in models:
        print(f"警告: 控制模型 {control_model} 不在有效模型列表中")
        control_model = models[0]
    
    valid_df_mse = data_df_mse.dropna(subset=models)
    valid_df_mae = data_df_mae.dropna(subset=models)
    
    common_configs = list(set(valid_df_mse['config']) & set(valid_df_mae['config']))
    valid_df_mse = valid_df_mse[valid_df_mse['config'].isin(common_configs)]
    valid_df_mae = valid_df_mae[valid_df_mae['config'].isin(common_configs)]
    
    n_valid = len(valid_df_mse)
    config_list = valid_df_mse['config'].tolist()
    
    if n_valid < 2:
        return None
    
    data_matrix_mse = valid_df_mse[models].values
    ranks_mse = np.zeros_like(data_matrix_mse)
    for i in range(len(data_matrix_mse)):
        ranks_mse[i] = rankdata(data_matrix_mse[i])
    
    data_matrix_mae = valid_df_mae[models].values
    ranks_mae = np.zeros_like(data_matrix_mae)
    for i in range(len(data_matrix_mae)):
        ranks_mae[i] = rankdata(data_matrix_mae[i])
    
    ranks_avg = (ranks_mse + ranks_mae) / 2.0
    avg_ranks = np.mean(ranks_avg, axis=0)
    
    k = len(models)
    N = n_valid
    
    control_idx = models.index(control_model)
    control_rank = avg_ranks[control_idx]
    
    alpha_adjusted = alpha / (k - 1)
    z_alpha = stats.norm.ppf(1 - alpha_adjusted / 2)
    cd = z_alpha * np.sqrt(k * (k + 1) / (6 * N))
    
    friedman_stat, friedman_p = stats.friedmanchisquare(*[ranks_avg[:, i] for i in range(k)])
    
    comparisons = []
    for i in range(k):
        if i != control_idx:
            diff = abs(avg_ranks[i] - control_rank)
            comparisons.append({
                'method': models[i],
                'method_idx': i,
                'mean_rank': avg_ranks[i],
                'diff': diff,
                'significant': diff > cd
            })
    
    results = {
        'models': models,
        'avg_ranks': avg_ranks,
        'cd': cd,
        'z_alpha': z_alpha,
        'alpha_adjusted': alpha_adjusted,
        'control_idx': control_idx,
        'control_model': control_model,
        'control_rank': control_rank,
        'friedman_stat': friedman_stat,
        'friedman_p': friedman_p,
        'valid_configs': config_list,
        'N': N,
        'k': k,
        'ranks_df': pd.DataFrame(ranks_avg, columns=models, index=valid_df_mse['config'].values),
        'comparisons': comparisons
    }
    
    return results

def plot_cd_diagram(results, title, filename, exp_name, test_type='Nemenyi'):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 28
    
    models = results['models']
    avg_ranks = results['avg_ranks']
    cd = results['cd_005'] if test_type == 'Nemenyi' else results['cd']
    N = results['N']
    k = results['k']
    
    fig, ax = plt.subplots(figsize=(36, 18))
    
    sorted_indices = np.argsort(avg_ranks)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_ranks = avg_ranks[sorted_indices]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c', '#e67e22', '#16a085', '#c0392b', '#8e44ad']
    
    y_center = 0
    max_rank = max(avg_ranks)
    min_rank = min(avg_ranks)
    x_min = min_rank - 1.0
    x_max = max_rank + cd + 2.0
    
    ax.annotate('', xy=(x_min, y_center), xytext=(x_max, y_center),
                arrowprops=dict(arrowstyle='->', color='black', lw=3))
    ax.hlines(y=y_center, xmin=x_min, xmax=x_max, colors='black', linewidths=3)
    
    label_info = []
    for i, (method, rank) in enumerate(zip(sorted_models, sorted_ranks)):
        color_idx = models.index(method)
        color = colors[color_idx % len(colors)]
        
        ax.hlines(y=y_center, xmin=rank, xmax=rank + cd,
                  colors=color, linestyles='-', linewidths=14, alpha=0.6, zorder=3)
        ax.scatter(rank, y_center, s=1400, c=color, zorder=5,
                  edgecolors='black', linewidths=4)
        
        conflict_above = False
        conflict_below = False
        for prev_rank, prev_pos in label_info:
            if abs(rank - prev_rank) < 2.5:
                if prev_pos == 'above':
                    conflict_above = True
                else:
                    conflict_below = True
        
        if conflict_above and not conflict_below:
            pos = 'below'
        elif conflict_below and not conflict_above:
            pos = 'above'
        elif conflict_above and conflict_below:
            pos = 'above'
        else:
            pos = 'above'
        
        label_info.append((rank, pos))
        
        if pos == 'above':
            label_y = y_center + 0.2
            va = 'bottom'
            rank_y = y_center - 0.5
            rank_va = 'top'
            line_y_start = y_center + 0.15
            line_y_end = y_center + 0.9
        else:
            label_y = y_center - 0.2
            va = 'top'
            rank_y = y_center + 0.5
            rank_va = 'bottom'
            line_y_start = y_center - 0.15
            line_y_end = y_center - 0.9
        
        ax.plot([rank, rank], [line_y_start, line_y_end], color='gray', linewidth=1.5, linestyle='-', alpha=0.7)
        ax.text(rank, label_y, f'{method}',
                ha='center', va=va, fontsize=26, fontweight='bold', color='black')
        ax.text(rank, rank_y, f'{rank:.2f}',
                ha='center', va=rank_va, fontsize=24, color='black')
    
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.text(x_max - 0.5, y_center - 0.8, 'Average Rank',
            fontsize=30, fontweight='bold', ha='right', va='top')
    
    ax.text(x_min + 0.15, y_center + 1.5, f'CD={cd:.3f}',
            fontsize=26, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"CD图已保存: {filename}")
    plt.close()

def plot_bonferroni_dunn_diagram(results, title, filename, exp_name):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 28
    
    models = results['models']
    avg_ranks = results['avg_ranks']
    cd = results['cd']
    N = results['N']
    k = results['k']
    control_idx = results['control_idx']
    control_model = results['control_model']
    
    fig, ax = plt.subplots(figsize=(40, 20))
    
    sorted_indices = np.argsort(avg_ranks)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_ranks = avg_ranks[sorted_indices]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c', '#e67e22', '#16a085', '#c0392b', '#8e44ad']
    
    # ============================================================
    # 可配置：横轴长度参数
    # x_min_offset: 横轴左端超出最小rank的距离（负值向左延伸）
    # x_max_offset: 横轴右端超出最大rank+CD的距离（正值向右延伸）
    # ============================================================
    x_min_offset = 0.5   # ← 修改这里调整左端长度
    x_max_offset = .0   # ← 修改这里调整右端长度
    # ============================================================
    
    y_center = 0
    max_rank = max(avg_ranks)
    min_rank = min(avg_ranks)
    x_min = min_rank - x_min_offset
    x_max = max_rank + cd + x_max_offset
    
    ax.annotate('', xy=(x_min, y_center), xytext=(x_max, y_center),
                arrowprops=dict(arrowstyle='->', color='black', lw=3))
    ax.hlines(y=y_center, xmin=x_min, xmax=x_max, colors='black', linewidths=3)
    
    n = len(sorted_models)
    label_data = []
    
    for i, (method, rank) in enumerate(zip(sorted_models, sorted_ranks)):
        color_idx = models.index(method)
        is_control = (models.index(method) == control_idx)
        color = colors[color_idx % len(colors)]
        
        if is_control:
            ax.hlines(y=y_center, xmin=rank, xmax=rank + cd,
                      colors=color, linestyles='-', linewidths=14, alpha=0.6, zorder=3)
        ax.scatter(rank, y_center, s=1400, c=color, zorder=5,
                  edgecolors='black', linewidths=4)
        
        label_text = f'{method}'
        if is_control:
            label_text += ' (Control)'
        
        label_data.append({
            'rank': rank,
            'method': method,
            'label_text': label_text,
            'color': color,
            'is_control': is_control,
            'index': i
        })
    
    # ============================================================
    # 可配置：每个方法的竖线位置（y_offset）和 rank 数字位置（rank_offset）
    # 格式: {'方法名': (y_offset, rank_offset)}
    # y_offset: 标签距离横线的距离（越大竖线越长）
    # rank_offset: rank数字距离横线的距离（正值在横线上方，负值在下方）
    # ============================================================
    custom_offsets = {
        'ST-Mixer': (0.5, 0.3),           # (竖线长度, rank位置)
        'ST-Mixer-Odd-Even (S)': (1., -0.3),
        'ST-Mixer-Downsampling (S)': (.5, 0.3),
        'ST-Mixer-Bottom-Up (S)': (0.8, -0.3),
        'ST-Mixer-Top-Down (S)': (1., 0.3),
    }
    # ============================================================
    
    above_positions = []
    below_positions = []
    
    for i, data in enumerate(label_data):
        if i % 2 == 0:
            above_positions.append(data)
        else:
            below_positions.append(data)
    
    def assign_y_positions(positions, is_above):
        assigned = []
        for i, data in enumerate(positions):
            method_name = data['method']
            # 使用自定义的offset，如果没有设置则使用默认值
            offsets = custom_offsets.get(method_name, (1.5, 0.6 if is_above else -0.6))
            y_offset, rank_offset = offsets
            assigned.append({
                **data,
                'y_offset': y_offset,
                'rank_offset': rank_offset,
                'pos': 'above' if is_above else 'below'
            })
        return assigned
    
    above_assigned = assign_y_positions(above_positions, True)
    below_assigned = assign_y_positions(below_positions, False)
    
    all_labels = above_assigned + below_assigned
    all_labels.sort(key=lambda x: x['index'])
    
    for label in all_labels:
        rank = label['rank']
        y_offset = label['y_offset']
        rank_offset = label['rank_offset']
        label_text = label['label_text']
        color = label['color']
        pos = label['pos']
        
        # 竖线终点距离标签的间距（可调整）
        label_gap = 0.1  # 标签和竖线终点的间距
        
        if pos == 'above':
            label_y = y_center + y_offset
            va = 'bottom'
            rank_y = y_center + rank_offset  # 使用自定义的rank位置
            rank_va = 'top'
            line_y_start = y_center + 0.15
            line_y_end = y_center + y_offset - label_gap
        else:
            label_y = y_center - y_offset
            va = 'top'
            rank_y = y_center + rank_offset  # 使用自定义的rank位置
            rank_va = 'bottom'
            line_y_start = y_center - 0.15
            line_y_end = y_center - y_offset + label_gap
        
        ax.plot([rank, rank], [line_y_start, line_y_end], color='gray', linewidth=1.5, linestyle='-', alpha=0.7)
        ax.text(rank, label_y, label_text,
                ha='center', va=va, fontsize=26, fontweight='bold', color='black')
        ax.text(rank, rank_y, f'{rank:.2f}',
                ha='center', va=rank_va, fontsize=24, color='black')
    
    max_above_offset = max([l['y_offset'] for l in above_assigned]) if above_assigned else 4.0
    max_below_offset = max([l['y_offset'] for l in below_assigned]) if below_assigned else 4.0
    
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(-(max_below_offset + 2.0), max_above_offset + 2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.text(x_max , y_center - 0.1, 'Average Rank',
            fontsize=40, fontweight='bold', ha='right', va='top')
    
    control_rank = avg_ranks[control_idx]
    ax.text(control_rank + cd/2, y_center + 0.1, f'CD={cd:.3f}',
            fontsize=26, fontweight='bold', ha='center', va='bottom',
            color='black')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Bonferroni-Dunn CD图已保存: {filename}")
    plt.close()

def save_nemenyi_to_excel(results, output_dir, exp_name):
    excel_path = os.path.join(output_dir, f'{exp_name}_Nemenyi_Results.xlsx')
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_data = []
        for i, model in enumerate(results['models']):
            summary_data.append({
                'Model': model,
                'Average_Rank': results['avg_ranks'][i]
            })
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Average_Rank')
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        results['ranks_df'].to_excel(writer, sheet_name='Ranks_Per_Config')
        
        test_info = pd.DataFrame([{
            'Friedman_Statistic': results['friedman_stat'],
            'Friedman_p_value': results['friedman_p'],
            'N_valid_configs': results['N'],
            'K_models': results['k'],
            'CD_alpha_0.05': results['cd_005'],
            'CD_alpha_0.10': results['cd_010'],
            'Note': 'Average rank of MSE and MAE'
        }])
        test_info.to_excel(writer, sheet_name='Test_Info', index=False)
        
        pairwise_data = []
        models = results['models']
        avg_ranks = results['avg_ranks']
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                diff = abs(avg_ranks[i] - avg_ranks[j])
                sig_005 = 'Yes' if diff > results['cd_005'] else 'No'
                sig_010 = 'Yes' if diff > results['cd_010'] else 'No'
                pairwise_data.append({
                    'Model_1': models[i],
                    'Model_2': models[j],
                    'Rank_Difference': diff,
                    'Significant_alpha_0.05': sig_005,
                    'Significant_alpha_0.10': sig_010
                })
        pairwise_df = pd.DataFrame(pairwise_data)
        pairwise_df.to_excel(writer, sheet_name='Pairwise_Comparison', index=False)
    
    print(f"Nemenyi Excel结果已保存: {excel_path}")

def save_bonferroni_dunn_to_excel(results, output_dir, exp_name):
    excel_path = os.path.join(output_dir, f'{exp_name}_BonferroniDunn_Results.xlsx')
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_data = []
        for i, model in enumerate(results['models']):
            is_control = (i == results['control_idx'])
            comp = next((c for c in results['comparisons'] if c['method_idx'] == i), None)
            summary_data.append({
                'Model': model,
                'Is_Control': is_control,
                'Average_Rank': results['avg_ranks'][i],
                'Diff_to_Control': 0 if is_control else (comp['diff'] if comp else None),
                'Significant': 'N/A' if is_control else ('Yes' if comp and comp['significant'] else 'No')
            })
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Average_Rank')
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        results['ranks_df'].to_excel(writer, sheet_name='Ranks_Per_Config')
        
        test_info = pd.DataFrame([{
            'Friedman_Statistic': results['friedman_stat'],
            'Friedman_p_value': results['friedman_p'],
            'N_valid_configs': results['N'],
            'K_models': results['k'],
            'Control_Model': results['control_model'],
            'CD': results['cd'],
            'z_alpha': results['z_alpha'],
            'alpha_adjusted': results['alpha_adjusted'],
            'Note': 'Average rank of MSE and MAE'
        }])
        test_info.to_excel(writer, sheet_name='Test_Info', index=False)
        
        comparisons_df = pd.DataFrame(results['comparisons'])
        comparisons_df.to_excel(writer, sheet_name='Comparisons', index=False)
    
    print(f"Bonferroni-Dunn Excel结果已保存: {excel_path}")

def process_experiment(file_path, exp_name, output_dir):
    print('\n' + '='*80)
    print(f'处理实验: {exp_name}')
    print('='*80)

    os.makedirs(output_dir, exist_ok=True)
    
    data_df_mse, data_df_mae, models = parse_excel_data(file_path)
    print(f'模型: {models}')
    print(f'数据形状: {data_df_mse.shape}')
    
    print('\n' + '-'*80)
    print('Nemenyi检验 (使用MSE和MAE排名的均值)')
    print('-'*80)
    
    nemenyi_results = nemenyi_test(data_df_mse, data_df_mae, models)
    
    if nemenyi_results is None:
        print('有效配置太少，无法进行Nemenyi检验')
        return None
    
    print(f'\n有效配置数量: {nemenyi_results["N"]}')
    print(f'有效配置: {nemenyi_results["valid_configs"]}')
    print(f'\nFriedman检验:')
    print(f'  统计量: {nemenyi_results["friedman_stat"]:.4f}')
    print(f'  p值: {nemenyi_results["friedman_p"]:.6f}')
    
    print(f'\n各模型平均排名 (排名越小越好):')
    for model, rank in zip(nemenyi_results['models'], nemenyi_results['avg_ranks']):
        print(f'  {model}: {rank:.4f}')
    
    print(f'\nNemenyi临界距离:')
    print(f'  α=0.05时: CD = {nemenyi_results["cd_005"]:.4f}')
    print(f'  α=0.10时: CD = {nemenyi_results["cd_010"]:.4f}')
    
    print(f'\n模型间排名差异:')
    models_list = nemenyi_results['models']
    avg_ranks = nemenyi_results['avg_ranks']
    for i in range(len(models_list)):
        for j in range(i+1, len(models_list)):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            sig_005 = '显著(α=0.05)' if diff > nemenyi_results['cd_005'] else ''
            sig_010 = '显著(α=0.10)' if diff > nemenyi_results['cd_010'] else ''
            print(f'  |{models_list[i]} - {models_list[j]}| = {diff:.4f} {sig_005} {sig_010}')
    
    pdf_path = os.path.join(output_dir, f'{exp_name}_Nemenyi_CD_Diagram.pdf')
    plot_cd_diagram(nemenyi_results, f'{exp_name}: Nemenyi Test', pdf_path, exp_name, test_type='Nemenyi')
    save_nemenyi_to_excel(nemenyi_results, output_dir, exp_name)
    
    print('\n' + '-'*80)
    print('Bonferroni-Dunn检验 (使用MSE和MAE排名的均值)')
    print('-'*80)
    
    bd_results = bonferroni_dunn_test(data_df_mse, data_df_mae, models, control_model='ST-Mixer', alpha=0.05)
    
    if bd_results is None:
        print('有效配置太少，无法进行Bonferroni-Dunn检验')
        return None
    
    print(f'\n有效配置数量: {bd_results["N"]}')
    print(f'\nFriedman检验:')
    print(f'  统计量: {bd_results["friedman_stat"]:.4f}')
    print(f'  p值: {bd_results["friedman_p"]:.6f}')
    
    print(f'\n各模型平均排名 (排名越小越好):')
    for i, model in enumerate(bd_results['models']):
        marker = " (对照)" if i == bd_results['control_idx'] else ""
        print(f'  {model}: {bd_results["avg_ranks"][i]:.4f}{marker}')
    
    print(f'\nBonferroni-Dunn临界距离:')
    print(f'  CD = {bd_results["cd"]:.4f}')
    print(f'  z_alpha = {bd_results["z_alpha"]:.3f}')
    print(f'  校正后alpha = {bd_results["alpha_adjusted"]:.4f}')
    
    print(f'\n与对照方法({bd_results["control_model"]})的比较:')
    for comp in bd_results['comparisons']:
        sig = "显著" if comp['significant'] else "不显著"
        print(f'  {comp["method"]}: 排名={comp["mean_rank"]:.4f}, 差异={comp["diff"]:.4f} ({sig})')
    
    bd_pdf_path = os.path.join(output_dir, f'{exp_name}_BonferroniDunn_CD_Diagram.pdf')
    plot_bonferroni_dunn_diagram(bd_results, f'{exp_name}: Bonferroni-Dunn Test', bd_pdf_path, exp_name)
    save_bonferroni_dunn_to_excel(bd_results, output_dir, exp_name)
    
    return nemenyi_results, bd_results

def main():
    parser = argparse.ArgumentParser(
        description='Nemenyi/Bonferroni-Dunn CD analysis for the trend component variants (paper Fig. 12 / Table 9).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--input', default='results/trend_component_experiment.xlsx',
                        help='Path to the trend component experiment workbook (layout: see module docstring).')
    parser.add_argument('--output-dir', default='results/statistical_tests/Trend_Component_Experiment',
                        help='Directory for CD diagram PDFs and result workbooks.')
    args = parser.parse_args()

    process_experiment(args.input, 'Trend_Component_Experiment', args.output_dir)

    print('\n' + '='*80)
    print('实验处理完成!')
    print('='*80)
    print(f'\n结果保存在: {args.output_dir}')


if __name__ == '__main__':
    main()
