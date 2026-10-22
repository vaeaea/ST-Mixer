"""Welch's t-test of ST-Mixer vs PatchTST on the seven single-domain datasets
(per dataset over the 4 horizons, plus a pooled 28-config test). Supports
paper Table 6.

Input (--input): Excel workbook in the layout of `单数据集结果汇总.xlsx`
(sheet `Sheet1`, NO header row, 0-based row/column indices):
  - row 0: model names; each model occupies an MSE column (even index) and an
    MAE column (next odd index), starting from column 2.
  - 7 dataset blocks of 5 rows each; first data row of each block:
    Traffic 2, Electricity 7, Weather 12, ETTh1 17, ETTh2 22, ETTm1 27, ETTm2 32.
    Each block: 4 horizon rows (96/192/336/720 in column 1) + one `Avg.` row.

Output (--output-dir): result workbook with Welch results, raw data and a
win/loss summary. Run from the repo root:
    python evaluation/welch_test_stmixer_patchtst_seven.py --input <xlsx> --output-dir <dir>
"""

import pandas as pd
import numpy as np
from scipy import stats
import argparse
import os

# 目标模型列（MSE在偶数列，MAE在奇数列）
TARGET_MODELS = ['ST-Mixer', 'PatchTST']

def parse_seven_datasets(file_path):
    """解析Excel文件，提取前7个单一领域数据集中 ST-Mixer 与 PatchTST 的MSE/MAE"""
    df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)

    model_row = df.iloc[0]
    model_cols = {}
    for i in range(2, len(model_row), 2):
        model_val = model_row[i]
        if pd.notna(model_val) and str(model_val).strip() != '' and str(model_val).strip() != 'Models':
            model_cols[str(model_val).strip()] = (i, i + 1)

    print(f"前7个数据集的模型列表: {list(model_cols.keys())}")
    for m in TARGET_MODELS:
        assert m in model_cols, f"模型 {m} 不在数据中"

    # 数据集位置
    datasets_info = [
        {'name': 'Traffic', 'start_row': 2},
        {'name': 'Electricity', 'start_row': 7},
        {'name': 'Weather', 'start_row': 12},
        {'name': 'ETTh1', 'start_row': 17},
        {'name': 'ETTh2', 'start_row': 22},
        {'name': 'ETTm1', 'start_row': 27},
        {'name': 'ETTm2', 'start_row': 32},
    ]

    all_data = []
    for dataset_info in datasets_info:
        dataset_name = dataset_info['name']
        start_row = dataset_info['start_row']

        for i in range(4):
            row_idx = start_row + i
            if row_idx >= len(df):
                break

            row = df.iloc[row_idx]
            pred_len = row[1]

            if pd.notna(pred_len) and str(pred_len).strip() != 'Avg.':
                try:
                    pred_len_int = int(float(pred_len))
                    rec = {'dataset': dataset_name, 'pred_len': pred_len_int}
                    for m in TARGET_MODELS:
                        mse_col, mae_col = model_cols[m]
                        rec[f'{m}_MSE'] = float(row[mse_col]) if pd.notna(row[mse_col]) else np.nan
                        rec[f'{m}_MAE'] = float(row[mae_col]) if pd.notna(row[mae_col]) else np.nan
                    all_data.append(rec)
                except (ValueError, TypeError):
                    continue

    return pd.DataFrame(all_data)

def welch_report(data_df, metric):
    """对指定指标（MSE或MAE）进行Welch检验：逐数据集 + 全部28个配置合并"""
    st_col = f'ST-Mixer_{metric}'
    pt_col = f'PatchTST_{metric}'

    rows = []
    # 逐数据集检验（每个数据集4个pred_len）
    for dataset_name, g in data_df.groupby('dataset', sort=False):
        st = g[st_col].dropna().values
        pt = g[pt_col].dropna().values
        t, p = stats.ttest_ind(st, pt, equal_var=False)  # Welch's t-test
        rows.append({
            'Metric': metric,
            'Dataset': dataset_name,
            'ST-Mixer Mean': np.mean(st), 'ST-Mixer Std': np.std(st, ddof=1),
            'PatchTST Mean': np.mean(pt), 'PatchTST Std': np.std(pt, ddof=1),
            'Mean Diff (ST-PT)': np.mean(st) - np.mean(pt),
            't_statistic': t, 'p_value': p,
            'df_welch': (st.var(ddof=1)/len(st) + pt.var(ddof=1)/len(pt))**2 /
                        ((st.var(ddof=1)/len(st))**2/(len(st)-1) + (pt.var(ddof=1)/len(pt))**2/(len(pt)-1)),
            'Significant_0.05': 'Yes' if p < 0.05 else 'No',
            'Significant_0.01': 'Yes' if p < 0.01 else 'No',
        })

    # 合并全部28个配置检验（跨数据集混合量纲，需谨慎解读）
    st_all = data_df[st_col].dropna().values
    pt_all = data_df[pt_col].dropna().values
    t, p = stats.ttest_ind(st_all, pt_all, equal_var=False)
    rows.append({
        'Metric': metric,
        'Dataset': 'Pooled (28 configs)',
        'ST-Mixer Mean': np.mean(st_all), 'ST-Mixer Std': np.std(st_all, ddof=1),
        'PatchTST Mean': np.mean(pt_all), 'PatchTST Std': np.std(pt_all, ddof=1),
        'Mean Diff (ST-PT)': np.mean(st_all) - np.mean(pt_all),
        't_statistic': t, 'p_value': p,
        'df_welch': (st_all.var(ddof=1)/len(st_all) + pt_all.var(ddof=1)/len(pt_all))**2 /
                    ((st_all.var(ddof=1)/len(st_all))**2/(len(st_all)-1) + (pt_all.var(ddof=1)/len(pt_all))**2/(len(pt_all)-1)),
        'Significant_0.05': 'Yes' if p < 0.05 else 'No',
        'Significant_0.01': 'Yes' if p < 0.01 else 'No',
    })

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(
        description="Welch's t-test: ST-Mixer vs PatchTST on the seven single-domain datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--input', default='results/single_dataset_summary.xlsx',
                        help='Path to the single-dataset summary workbook (layout: see module docstring).')
    parser.add_argument('--output-dir', default='results/statistical_tests/Welch_Test_ST_Mixer_vs_PatchTST',
                        help='Directory for the result workbook.')
    args = parser.parse_args()

    exp_name = 'Welch_Test_ST_Mixer_vs_PatchTST'
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    data_df = parse_seven_datasets(args.input)
    print(f'\n共 {len(data_df)} 个配置')
    print(data_df.to_string(index=False))

    # 胜负统计
    mse_wins = int((data_df['ST-Mixer_MSE'] < data_df['PatchTST_MSE']).sum())
    mae_wins = int((data_df['ST-Mixer_MAE'] < data_df['PatchTST_MAE']).sum())
    st_mse_impr = np.mean((data_df['PatchTST_MSE'] - data_df['ST-Mixer_MSE']) / data_df['PatchTST_MSE'] * 100)
    st_mae_impr = np.mean((data_df['PatchTST_MAE'] - data_df['ST-Mixer_MAE']) / data_df['PatchTST_MAE'] * 100)

    result_mse = welch_report(data_df, 'MSE')
    result_mae = welch_report(data_df, 'MAE')
    results = pd.concat([result_mse, result_mae], ignore_index=True)

    excel_path = os.path.join(output_dir, f'{exp_name}_Seven_Datasets.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results.to_excel(writer, sheet_name='Welch_Results', index=False)
        data_df.to_excel(writer, sheet_name='Raw_Data', index=False)
        summary = pd.DataFrame([
            {'Item': 'MSE: ST-Mixer wins', 'Value': f'{mse_wins}/28'},
            {'Item': 'MAE: ST-Mixer wins', 'Value': f'{mae_wins}/28'},
            {'Item': 'ST-Mixer mean relative MSE improvement over PatchTST', 'Value': f'{st_mse_impr:.2f}%'},
            {'Item': 'ST-Mixer mean relative MAE improvement over PatchTST', 'Value': f'{st_mae_impr:.2f}%'},
        ])
        summary.to_excel(writer, sheet_name='Summary', index=False)
    print(f'\nWelch检验Excel结果已保存: {excel_path}')

    print('\n' + '='*80)
    print('Welch检验结果 (ST-Mixer vs PatchTST, 7个单一领域数据集)')
    print('='*80)
    for metric, res in [('MSE', result_mse), ('MAE', result_mae)]:
        print(f'\n----- {metric} -----')
        for _, r in res.iterrows():
            better = 'ST-Mixer更优' if r['Mean Diff (ST-PT)'] < 0 else 'PatchTST更优'
            print(f"  {r['Dataset']:<22} ST-Mixer={r['ST-Mixer Mean']:.4f}±{r['ST-Mixer Std']:.4f}  "
                  f"PatchTST={r['PatchTST Mean']:.4f}±{r['PatchTST Std']:.4f}  "
                  f"t={r['t_statistic']:.3f}  p={r['p_value']:.4f}  "
                  f"({'显著(0.05)' if r['Significant_0.05']=='Yes' else '不显著'}), {better}")

    print(f'\n胜负统计: MSE ST-Mixer胜 {mse_wins}/28, MAE ST-Mixer胜 {mae_wins}/28')
    print(f'平均相对提升: MSE {st_mse_impr:.2f}%, MAE {st_mae_impr:.2f}%')

if __name__ == '__main__':
    main()
