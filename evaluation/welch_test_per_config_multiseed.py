"""Welch's t-test per configuration (dataset x horizon x metric) between
ST-Mixer and PatchTST / TimeMixer, from seed-level mean/std over 3 runs.
Supports paper Table 6.

Input (--error-bar): Excel workbook in the layout of `ERROR BAR.xlsx`
(sheet `Sheet1`, NO header row):
  - sections introduced by a model name in column 0 (STMIXER / PATCHTST /
    TIMEMIXER, mapped to ST-Mixer / PatchTST / TimeMixer);
  - within a section, a dataset name in column 0 starts a dataset block
    (LTMS is normalised to LMTS);
  - column 1: prediction horizon (96/192/336/720) or `Avg.`;
  - columns 2-5: MSE mean, MSE std, MAE mean, MAE std over the 3 seeds.

Optional (--summary): workbook in the `单数据集结果汇总.xlsx` layout (see
nemenyi_all_datasets_common.py) used only for a consistency check; skipped if
the file does not exist.

Output (--output-dir): result workbook with per-config Welch results, the
parsed error-bar data, the consistency check and a summary sheet.
Run from the repo root:
    python evaluation/welch_test_per_config_multiseed.py --error-bar <xlsx> --output-dir <dir>
"""

import pandas as pd
import numpy as np
from scipy import stats
import argparse
import os

# ERROR BAR.xlsx 中每个配置的 mean/std 来自3次运行（由PatchTST LTMS块的3个单seed值验证：
# 均值与样本std(ddof=1)均精确吻合），因此检验在seed层面进行，每组n=3
N_SEEDS = 3

# 模型名映射（表内名称 -> 统一名称）
MODEL_MAP = {'STMIXER': 'ST-Mixer', 'PATCHTST': 'PatchTST', 'TIMEMIXER': 'TimeMixer'}

# 检验的模型对（ST-Mixer为主对照）
MODEL_PAIRS = [('ST-Mixer', 'PatchTST'), ('ST-Mixer', 'TimeMixer')]

def parse_error_bar(file_path):
    """解析ERROR BAR.xlsx，提取每个模型在每个数据集-步长上的MSE/MAE均值与std"""
    df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)

    records = []
    current_model = None
    current_dataset = None

    for _, row in df.iterrows():
        c0 = row[0]
        if pd.notna(c0):
            c0s = str(c0).strip()
            if c0s.upper() in MODEL_MAP:
                current_model = MODEL_MAP[c0s.upper()]
                current_dataset = None
                continue
            current_dataset = c0s

        if current_model is None or pd.isna(row[1]):
            continue

        horizon = str(row[1]).strip()
        if horizon == 'Avg.':
            continue

        records.append({
            'model': current_model,
            'dataset': current_dataset,
            'pred_len': int(float(horizon)),
            'MSE_mean': float(row[2]), 'MSE_std': float(row[3]),
            'MAE_mean': float(row[4]), 'MAE_std': float(row[5]),
        })

    return pd.DataFrame(records)

def welch_from_stats(m1, s1, m2, s2, n):
    """由两组的mean/std/n直接构造Welch t检验（Welch-Satterthwaite自由度）"""
    v1, v2 = s1 * s1 / n, s2 * s2 / n
    se = np.sqrt(v1 + v2)
    if se == 0:
        return np.nan, np.nan, np.nan
    t = (m1 - m2) / se
    dfw = (v1 + v2) ** 2 / (v1 ** 2 / (n - 1) + v2 ** 2 / (n - 1))
    p = 2.0 * stats.t.sf(abs(t), dfw)
    return t, dfw, p

def run_welch_per_config(data_df, model_pairs):
    """对每个 数据集-步长-指标 组合做Welch检验"""
    rows = []
    for (dataset, pred_len), g in data_df.groupby(['dataset', 'pred_len'], sort=False):
        for metric in ['MSE', 'MAE']:
            stats_map = {}
            for _, r in g.iterrows():
                stats_map[r['model']] = (r[f'{metric}_mean'], r[f'{metric}_std'])

            for m1, m2 in model_pairs:
                if m1 not in stats_map or m2 not in stats_map:
                    continue
                mu1, s1 = stats_map[m1]
                mu2, s2 = stats_map[m2]
                t, dfw, p = welch_from_stats(mu1, s1, mu2, s2, N_SEEDS)
                rows.append({
                    'Pair': f'{m1} vs {m2}',
                    'Metric': metric,
                    'Dataset': dataset,
                    'Pred_len': pred_len,
                    f'{m1} Mean': mu1, f'{m1} Std': s1,
                    f'{m2} Mean': mu2, f'{m2} Std': s2,
                    'Mean Diff (1-2)': mu1 - mu2,
                    't_statistic': t, 'df_welch': dfw, 'p_value': p,
                    'Significant_0.05': 'Yes' if p < 0.05 else 'No',
                    'Significant_0.01': 'Yes' if p < 0.01 else 'No',
                    'Better': m1 if mu1 < mu2 else m2,
                })
    return pd.DataFrame(rows)

def consistency_check(data_df, summary_path):
    """检查ERROR BAR.xlsx的均值与 单数据集结果汇总.xlsx 中单一结果的一致性"""
    src = summary_path
    if not os.path.exists(src):
        print(f'\n[!] 未找到单数据集结果汇总工作簿 ({src})，跳过一致性检查')
        return pd.DataFrame()
    df = pd.read_excel(src, sheet_name='Sheet1', header=None)

    model_row = df.iloc[0]
    model_cols = {}
    for i in range(2, len(model_row), 2):
        model_val = model_row[i]
        if pd.notna(model_val) and str(model_val).strip() != '' and str(model_val).strip() != 'Models':
            model_cols[str(model_val).strip()] = (i, i + 1)

    datasets_info = [
        {'name': 'Traffic', 'start_row': 2}, {'name': 'Electricity', 'start_row': 7},
        {'name': 'Weather', 'start_row': 12}, {'name': 'ETTh1', 'start_row': 17},
        {'name': 'ETTh2', 'start_row': 22}, {'name': 'ETTm1', 'start_row': 27},
        {'name': 'ETTm2', 'start_row': 32},
    ]

    rows = []
    for info in datasets_info:
        ds_name, start_row = info['name'], info['start_row']
        for i in range(4):
            row = df.iloc[start_row + i]
            try:
                pred_len = int(float(row[1]))
            except (ValueError, TypeError):
                continue

            g = data_df[(data_df['dataset'] == ds_name) & (data_df['pred_len'] == pred_len)]
            for _, r in g.iterrows():
                if r['model'] not in model_cols:
                    continue
                mse_col, mae_col = model_cols[r['model']]
                src_mse = float(row[mse_col]) if pd.notna(row[mse_col]) else np.nan
                src_mae = float(row[mae_col]) if pd.notna(row[mae_col]) else np.nan
                rows.append({
                    'Model': r['model'], 'Dataset': ds_name, 'Pred_len': pred_len,
                    'Summary_MSE': src_mse, 'ErrorBar_MSE': r['MSE_mean'],
                    'MSE_Diff': r['MSE_mean'] - src_mse,
                    'Summary_MAE': src_mae, 'ErrorBar_MAE': r['MAE_mean'],
                    'MAE_Diff': r['MAE_mean'] - src_mae,
                })
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(
        description="Welch's t-test per configuration from seed-level mean/std over 3 runs (paper Table 6).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--error-bar', default='results/error_bar.xlsx',
                        help='Path to the ERROR BAR workbook (layout: see module docstring).')
    parser.add_argument('--summary', default='results/single_dataset_summary.xlsx',
                        help='Path to the single-dataset summary workbook for the consistency check '
                             '(skipped if the file does not exist).')
    parser.add_argument('--output-dir', default='results/statistical_tests/Welch_Test_PerConfig_MultiSeed',
                        help='Directory for the result workbook.')
    args = parser.parse_args()

    exp_name = 'Welch_Test_PerConfig_MultiSeed'
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    data_df = parse_error_bar(args.error_bar)
    print(f'解析到 {len(data_df)} 条记录')
    print(f"模型: {data_df['model'].unique().tolist()}, 数据集: {data_df['dataset'].unique().tolist()}")

    results = run_welch_per_config(data_df, MODEL_PAIRS)

    # 一致性检查（与单数据集结果汇总.xlsx对比）
    check_df = consistency_check(data_df, args.summary)
    if len(check_df) > 0:
        big_diff = check_df[(check_df['MSE_Diff'].abs() > 0.002) | (check_df['MAE_Diff'].abs() > 0.002)]
        if len(big_diff) > 0:
            print('\n[!] ERROR BAR均值与单数据集结果汇总.xlsx差异超过0.002的条目:')
            print(big_diff.to_string(index=False))

    excel_path = os.path.join(output_dir, f'{exp_name}_Results.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results.to_excel(writer, sheet_name='Welch_Per_Config', index=False)
        data_df.to_excel(writer, sheet_name='ErrorBar_Data', index=False)
        check_df.to_excel(writer, sheet_name='Consistency_Check', index=False)

        # 汇总：每个模型对/指标的显著配置数量
        summary_rows = []
        for m1, m2 in MODEL_PAIRS:
            for metric in ['MSE', 'MAE']:
                pair_name = f'{m1} vs {m2}'
                sub = results[(results['Pair'] == pair_name) & (results['Metric'] == metric)].copy()
                sub = sub.dropna(subset=[f'{m2} Mean'])
                summary_rows.append({
                    'Pair': f'{m1} vs {m2}', 'Metric': metric,
                    'Total_Configs': len(sub),
                    f'{m1}_Better': int((sub['Better'] == m1).sum()),
                    f'{m2}_Better': int((sub['Better'] == m2).sum()),
                    'Sig_0.05_Count': int((sub['Significant_0.05'] == 'Yes').sum()),
                    'Sig_0.05_Better_is_M1': int(((sub['Significant_0.05'] == 'Yes') & (sub['Better'] == m1)).sum()),
                    'Sig_0.01_Count': int((sub['Significant_0.01'] == 'Yes').sum()),
                })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f'\nWelch检验Excel结果已保存: {excel_path}')

    # 控制台输出每对模型的逐配置结果
    for m1, m2 in MODEL_PAIRS:
        pair_name = f'{m1} vs {m2}'
        print('\n' + '='*90)
        print(f'Welch逐配置检验: {pair_name} (每组n={N_SEEDS}个seed)')
        print('='*90)
        sub = results[results['Pair'] == pair_name].dropna(subset=[f'{m2} Mean'])
        for metric in ['MSE', 'MAE']:
            print(f'\n----- {metric} -----')
            sub_m = sub[sub['Metric'] == metric]
            for _, r in sub_m.iterrows():
                if r['Significant_0.01'] == 'Yes':
                    sig = '显著(0.01)'
                elif r['Significant_0.05'] == 'Yes':
                    sig = '显著(0.05)'
                else:
                    sig = '不显著'
                print(f"  {r['Dataset']:<12} {r['Pred_len']:>3}  "
                      f"{m1}={r[f'{m1} Mean']:.4f}±{r[f'{m1} Std']:.4f}  "
                      f"{m2}={r[f'{m2} Mean']:.4f}±{r[f'{m2} Std']:.4f}  "
                      f"t={r['t_statistic']:+.2f}  df={r['df_welch']:.1f}  p={r['p_value']:.4f}  {sig}  优:{r['Better']}")

    # 汇总打印
    print('\n' + '='*90)
    print('汇总统计')
    print('='*90)
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    main()
