"""Nemenyi / Bonferroni-Dunn critical-difference analysis on the LMTS
(cross-domain) dataset alone.

Input (--input): Excel workbook in the layout of `单数据集结果汇总.xlsx`
(sheet `Sheet1`, NO header row, 0-based row/column indices):
  - row 42: LMTS model names; each model occupies an MSE column (even index)
    and an MAE column (next odd index), starting from column 2.
  - rows 44-47: 4 horizon rows (96/192/336/720 in column 1).

Output (--output-dir): Nemenyi and Bonferroni-Dunn CD diagram PNGs plus
per-test result workbooks. Run from the repo root:
    python evaluation/nemenyi_lmts_dataset.py --input <xlsx> --output-dir <dir>
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

def parse_lmts_data(file_path):
    """解析Excel文件，提取LMTS数据集的结果（同时提取MSE和MAE）"""
    df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)

    model_row_idx = 42
    metric_row_idx = 43
    data_start_idx = 44

    model_row = df.iloc[model_row_idx]
    
    models = []
    mse_col_indices = []
    mae_col_indices = []
    
    for i in range(2, len(model_row), 2):
        model_val = model_row[i]
        if pd.notna(model_val) and str(model_val).strip() != '' and str(model_val).strip() != 'Models':
            model_name = str(model_val).strip()
            models.append(model_name)
            mse_col_indices.append(i)
            mae_col_indices.append(i + 1)

    print(f"从第{model_row_idx+1}行读取到的模型列表: {models}")
    print(f"MSE列索引: {mse_col_indices}")
    print(f"MAE列索引: {mae_col_indices}")

    data_mse = []
    data_mae = []
    
    for i in range(4):
        row_idx = data_start_idx + i
        if row_idx >= len(df):
            break
            
        row = df.iloc[row_idx]
        pred_len = row[1]

        if pd.notna(pred_len) and str(pred_len).strip() != 'Avg.':
            try:
                pred_len_int = int(float(pred_len))
                config_name = f'LMTS_{pred_len_int}'
                
                row_data_mse = {'config': config_name, 'dataset': 'LMTS', 'pred_len': pred_len_int}
                row_data_mae = {'config': config_name, 'dataset': 'LMTS', 'pred_len': pred_len_int}

                for model, mse_col, mae_col in zip(models, mse_col_indices, mae_col_indices):
                    mse_val = row[mse_col]
                    mae_val = row[mae_col]
                    row_data_mse[model] = float(mse_val) if pd.notna(mse_val) else np.nan
                    row_data_mae[model] = float(mae_val) if pd.notna(mae_val) else np.nan

                data_mse.append(row_data_mse)
                data_mae.append(row_data_mae)
            except (ValueError, TypeError):
                continue

    data_df_mse = pd.DataFrame(data_mse)
    data_df_mae = pd.DataFrame(data_mae)

    valid_models = []
    for model in models:
        if model in data_df_mse.columns and not data_df_mse[model].isna().all():
            if model in data_df_mae.columns and not data_df_mae[model].isna().all():
                valid_models.append(model)

    return data_df_mse, data_df_mae, valid_models

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
    
    q_005 = {2: 2.77, 3: 3.31, 4: 3.63, 5: 3.86, 6: 4.03, 7: 4.17, 8: 4.29, 9: 4.39, 10: 4.47, 11: 4.55, 12: 4.62, 13: 4.69, 14: 4.75, 15: 4.81}
    q_010 = {2: 2.34, 3: 2.81, 4: 3.10, 5: 3.32, 6: 3.48, 7: 3.62, 8: 3.74, 9: 3.85, 10: 3.94, 11: 4.02, 12: 4.10, 13: 4.17, 14: 4.24, 15: 4.30}
    
    q05 = q_005.get(k, 4.81)
    q10 = q_010.get(k, 4.30)
    
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
    
    fig, ax = plt.subplots(figsize=(18, max(10, k * 1.2)))
    
    sorted_indices = np.argsort(avg_ranks)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_ranks = avg_ranks[sorted_indices]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c', '#e67e22', '#16a085', '#c0392b', '#8e44ad', '#2980b9', '#27ae60', '#d35400', '#7f8c8d', '#34495e']
    
    for i, (method, rank) in enumerate(zip(sorted_models, sorted_ranks)):
        color_idx = models.index(method)
        ax.hlines(y=i, xmin=rank, xmax=rank + cd,
                  colors=colors[color_idx % len(colors)], linestyles='-', linewidths=6, alpha=0.4)
        ax.scatter(rank, i, s=600, c=colors[color_idx % len(colors)], zorder=5,
                  edgecolors='black', linewidths=3)
        ax.text(rank - 0.2, i, f'{method}\n({rank:.2f})',
                ha='right', va='center', fontsize=26, fontweight='bold')
    
    ax.set_xlim(0, max(avg_ranks) + cd + 0.5)
    ax.set_ylim(-0.5, len(models) - 0.5)
    ax.set_xlabel('Average Rank (lower is better)', fontsize=30, fontweight='bold')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    ax.tick_params(axis='x', labelsize=28)
    
    ax.text(0.02, 0.98, f'Control: ST-Mixer\nCD={cd:.3f}',
            transform=ax.transAxes, fontsize=26,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
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
    
    fig, ax = plt.subplots(figsize=(18, max(10, k * 1.2)))
    
    sorted_indices = np.argsort(avg_ranks)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_ranks = avg_ranks[sorted_indices]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c', '#e67e22', '#16a085', '#c0392b', '#8e44ad', '#2980b9', '#27ae60', '#d35400', '#7f8c8d', '#34495e']
    
    for i, (method, rank) in enumerate(zip(sorted_models, sorted_ranks)):
        color_idx = models.index(method)
        is_control = (models.index(method) == control_idx)
        
        if is_control:
            ax.hlines(y=i, xmin=rank, xmax=rank + cd,
                      colors=colors[color_idx % len(colors)], linestyles='-', linewidths=6, alpha=0.4)
        ax.scatter(rank, i, s=600, c=colors[color_idx % len(colors)], zorder=5,
                  edgecolors='black', linewidths=3)
        ax.text(rank - 0.2, i, f'{method}\n({rank:.2f})',
                ha='right', va='center', fontsize=26, fontweight='bold')
    
    ax.set_xlim(0, max(avg_ranks) + cd + 0.5)
    ax.set_ylim(-0.5, len(models) - 0.5)
    ax.set_xlabel('Average Rank (lower is better)', fontsize=30, fontweight='bold')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    ax.tick_params(axis='x', labelsize=28)
    
    ax.text(0.02, 0.98, f'Control: {control_model}\nCD={cd:.3f}',
            transform=ax.transAxes, fontsize=26,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
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

def process_lmts_experiment(file_path, exp_name, output_dir):
    print('\n' + '='*80)
    print(f'处理LMTS数据集实验: {exp_name}')
    print('='*80)

    os.makedirs(output_dir, exist_ok=True)
    
    data_df_mse, data_df_mae, models = parse_lmts_data(file_path)
    print(f'\n模型: {models}')
    print(f'数据形状: {data_df_mse.shape}')
    print(f'LMTS配置: {data_df_mse["config"].tolist()}')
    
    if len(data_df_mse) < 2:
        print('有效配置太少，无法进行统计检验')
        return None
    
    print('\n' + '-'*80)
    print('Nemenyi检验 (使用MSE和MAE排名的均值)')
    print('-'*80)
    
    nemenyi_results = nemenyi_test(data_df_mse, data_df_mae, models)
    
    if nemenyi_results is None:
        print('有效配置太少，无法进行Nemenyi检验')
        return None
    
    print(f'\n有效配置数量: {nemenyi_results["N"]}')
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
    
    png_path = os.path.join(output_dir, f'{exp_name}_Nemenyi_CD_Diagram.png')
    plot_cd_diagram(nemenyi_results, f'{exp_name}: Nemenyi Test', png_path, exp_name, test_type='Nemenyi')
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
    
    bd_png_path = os.path.join(output_dir, f'{exp_name}_BonferroniDunn_CD_Diagram.png')
    plot_bonferroni_dunn_diagram(bd_results, f'{exp_name}: Bonferroni-Dunn Test', bd_png_path, exp_name)
    save_bonferroni_dunn_to_excel(bd_results, output_dir, exp_name)
    
    return nemenyi_results, bd_results

def main():
    parser = argparse.ArgumentParser(
        description='Nemenyi/Bonferroni-Dunn CD analysis on the LMTS dataset.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--input', default='results/single_dataset_summary.xlsx',
                        help='Path to the single-dataset summary workbook (layout: see module docstring).')
    parser.add_argument('--output-dir', default='results/statistical_tests/LMTS_Dataset',
                        help='Directory for CD diagram PNGs and result workbooks.')
    args = parser.parse_args()

    process_lmts_experiment(args.input, 'LMTS_Dataset', args.output_dir)

    print('\n' + '='*80)
    print('LMTS数据集实验处理完成!')
    print('='*80)
    print(f'\n结果保存在: {args.output_dir}')


if __name__ == '__main__':
    main()
