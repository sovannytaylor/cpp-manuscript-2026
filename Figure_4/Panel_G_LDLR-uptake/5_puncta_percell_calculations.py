import os
import numpy as np
import pandas as pd
import importlib.util
import sys
from loguru import logger

puncta_ana_path = 'punctalyze-SRT/src/4_puncta_detection.py'

spec = importlib.util.spec_from_file_location('puncta_detection', puncta_ana_path)
puncta_detection_utils = importlib.util.module_from_spec(spec)
sys.modules['puncta_detection_utils'] = puncta_detection_utils
spec.loader.exec_module(puncta_detection_utils)
aggregate_features_by_group = puncta_detection_utils.aggregate_features_by_group

logger.info('import ok')

input_folder = 'results/summary_calculations-02/'
output_folder = 'results/summary_calculations-02/'

puncta_file = f'{input_folder}puncta_features.csv'
cell_file = f'{input_folder}cell_features.csv'


def calculate_cell_features(puncta_df, all_cells_df):
    """Calculate summarized puncta features per cell, including cells with 0 puncta."""

    group_cols = ['image_name', 'cell_number']

    puncta_summary = puncta_df.groupby(group_cols).agg({
        'puncta_minor_axis_length': 'mean',
        'puncta_major_axis_length': 'mean',
        'puncta_aspect_ratio': 'mean',
        'puncta_circularity': 'mean',
        'puncta_area': ['mean', 'sum'],
        'cell_size': 'mean',
        'puncta_eccentricity': 'mean',
        'puncta_cv': 'mean',
        'puncta_skew': 'mean',
        'cell_std': 'mean',
        'cell_cv': 'mean',
        'cell_skew': 'mean',
        'cell_coi1_intensity_mean': 'mean',
        'cell_coi2_intensity_mean': 'mean',
        'puncta_intensity_mean': 'mean',
        'puncta_intensity_mean_in_coi2': 'mean'
    })

    puncta_summary.columns = [
        '_'.join(col).strip() if isinstance(col, tuple) else col
        for col in puncta_summary.columns.values
    ]

    puncta_summary = puncta_summary.reset_index()
    puncta_counts = (
        puncta_df
        .groupby(group_cols)['puncta_area']
        .apply(lambda x: (x > 0).sum())
        .reset_index(name='puncta_count')
    )

    puncta_summary = puncta_summary.merge(
        puncta_counts,
        on=group_cols,
        how='left'
    )

    puncta_summary = puncta_summary.rename(columns={
        'puncta_minor_axis_length_mean': 'puncta_mean_minor_axis',
        'puncta_major_axis_length_mean': 'puncta_mean_major_axis',
        'puncta_aspect_ratio_mean': 'puncta_mean_aspect_ratio',
        'puncta_circularity_mean': 'puncta_mean_circularity',
        'puncta_area_mean': 'mean_puncta_area',
        'puncta_area_sum': 'puncta_area_sum',
        'puncta_eccentricity_mean': 'avg_eccentricity',
        'puncta_cv_mean': 'puncta_cv_mean',
        'puncta_skew_mean': 'puncta_skew_mean',
        'cell_std_mean': 'cell_std',
        'cell_cv_mean': 'cell_cv',
        'cell_skew_mean': 'cell_skew',
        'cell_coi1_intensity_mean_mean': 'cell_coi1_intensity_mean',
        'cell_coi2_intensity_mean_mean': 'cell_coi2_intensity_mean',
        'puncta_intensity_mean_mean': 'puncta_coi1_intensity_mean',
        'puncta_intensity_mean_in_coi2_mean': 'puncta_coi2_intensity_mean',
        'cell_size_mean': 'cell_size'
    })

    all_cells = all_cells_df[group_cols].drop_duplicates().copy()

    # Keep useful cell-level measurements from all_cells if present
    possible_cell_cols = [
        'image_name',
        'cell_number',
        'cell_size',
        'cell_std',
        'cell_cv',
        'cell_skew',
        'cell_coi1_intensity_mean',
        'cell_coi2_intensity_mean'
    ]

    keep_cols = [c for c in possible_cell_cols if c in all_cells_df.columns]
    all_cells = all_cells_df[keep_cols].drop_duplicates(subset=group_cols).copy()

    summary = all_cells.merge(
        puncta_summary,
        on=group_cols,
        how='left',
        suffixes=('', '_from_puncta')
    )

    # If duplicated cell-level columns exist from puncta_summary, prefer all-cell values
    for col in ['cell_size', 'cell_std', 'cell_cv', 'cell_skew',
                'cell_coi1_intensity_mean', 'cell_coi2_intensity_mean']:
        puncta_col = f'{col}_from_puncta'
        if puncta_col in summary.columns:
            if col not in summary.columns:
                summary[col] = summary[puncta_col]
            summary = summary.drop(columns=[puncta_col])

    # Zero-puncta cells
    summary['puncta_count'] = summary['puncta_count'].fillna(0).astype(int)
    summary['puncta_area_sum'] = summary['puncta_area_sum'].fillna(0)

    summary['puncta_area_proportion'] = (
        summary['puncta_area_sum'] / summary['cell_size']
    ) * 100

    summary['puncta_area_proportion'] = summary['puncta_area_proportion'].fillna(0)

    # Keep puncta-specific averages as NaN for no-uptake cells
    # because those cells do not have puncta to average.

    return summary


def save_dataframes(df, features, group_cols=['cell', 'peptide', 'rep']):
    df.to_csv(f'{output_folder}percell_puncta_features.csv', index=False)

    rep_df = aggregate_features_by_group(df, group_cols, features)
    rep_df.to_csv(f'{output_folder}percell_puncta_features_reps.csv', index=False)

    logger.info('Saved raw per-cell and replicate-level summaries')


if __name__ == '__main__':

    puncta_features = pd.read_csv(puncta_file)

    all_cells = puncta_features.copy()

    summary = calculate_cell_features(
        puncta_features,
        all_cells
    )

    summary = calculate_cell_features(puncta_features, all_cells)

    summary['peptide'] = summary['image_name'].str.split('_').str[1]
    summary['cell'] = summary['image_name'].str.split('_').str[2]
    summary['rep'] = summary['image_name'].str.extract(r'(REP\d+)')

    cols = [
        'cell_size',
        'mean_puncta_area',
        'puncta_area_proportion',
        'puncta_count',
        'puncta_mean_minor_axis',
        'puncta_mean_major_axis',
        'puncta_mean_aspect_ratio',
        'puncta_mean_circularity',
        'avg_eccentricity',
        'puncta_cv_mean',
        'puncta_skew_mean',
        'cell_std',
        'cell_cv',
        'cell_skew',
        'cell_coi1_intensity_mean',
        'cell_coi2_intensity_mean',
        'puncta_coi1_intensity_mean',
        'puncta_coi2_intensity_mean'
    ]

    cols = [c for c in cols if c in summary.columns]

    save_dataframes(summary, cols)

    logger.info(f"Min puncta_count: {summary['puncta_count'].min()}")
    logger.info(f"Zero-puncta cells: {(summary['puncta_count'] == 0).sum()}")
    logger.info('saved puncta feature averaged-per-cell dataframes')