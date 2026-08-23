import os
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from loguru import logger

logger.info('import ok')

plt.rcParams.update({'font.size': 14})
sns.set_palette('Paired')

input_folder = 'results/summary_calculations-02/'
output_folder = 'results/plotting-02/'

os.makedirs(output_folder, exist_ok=True)


def load_summary_data(input_folder):
    return {
        'puncta_features': pd.read_csv(f'{input_folder}puncta_features.csv'),
        'percell': pd.read_csv(f'{input_folder}percell_puncta_features.csv'),
        'percell_reps': pd.read_csv(f'{input_folder}percell_puncta_features_reps.csv')
    }


def clean_names(dfs):
    for df in dfs.values():

        if 'cell' in df.columns:
            df['cell'] = df['cell'].astype(str).str.strip()

            df['cell'] = df['cell'].replace({
                'OE-EGFP': 'OE',
                'OE-eGFP': 'OE',
                'OE-eGFP0': 'OE',
                'OE-GFP': 'OE',
            })

        if 'peptide' in df.columns:
            df['peptide'] = df['peptide'].astype(str).str.strip()

    return dfs


def plot_stats(
    data_raw,
    data_agg,
    features,
    title,
    save_name,
    x='peptide',
    hue='cell',
    pairs=None,
    order=None,
    hue_order=None
):
    features = [
        f for f in features
        if f in data_raw.columns and f in data_agg.columns
    ]

    if len(features) == 0:
        logger.warning(f'No matching features found for {title}')
        return

    ncols = 3
    nrows = math.ceil(len(features) / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(20, 5 * nrows)
    )

    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]

        sns.stripplot(
            data=data_raw,
            x=x,
            y=feature,
            hue=hue,
            order=order,
            hue_order=hue_order,
            dodge=True,
            edgecolor='white',
            linewidth=1,
            size=5,
            alpha=0.25,
            ax=ax,
            zorder=0
        )

        sns.boxplot(
            data=data_agg,
            x=x,
            y=feature,
            hue=hue,
            order=order,
            hue_order=hue_order,
            dodge=True,
            showcaps=True,
            boxprops={'facecolor': 'none'},
            showfliers=False,
            whiskerprops={'linewidth': 1},
            ax=ax,
            zorder=1
        )

        sns.stripplot(
            data=data_agg,
            x=x,
            y=feature,
            hue=hue,
            order=order,
            hue_order=hue_order,
            dodge=True,
            edgecolor='black',
            linewidth=1,
            size=8,
            ax=ax,
            zorder=2
        )

        ax.set_title(feature)
        ax.tick_params(axis='x', rotation=45)
        sns.despine(ax=ax)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        if pairs:
            try:
                annotator = Annotator(
                    ax,
                    pairs,
                    data=data_agg,
                    x=x,
                    y=feature,
                    hue=hue,
                    order=order,
                    hue_order=hue_order
                )
                annotator.configure(test='Mann-Whitney', verbose=0)
                annotator.apply_test()
                annotator.annotate()
            except Exception as e:
                logger.warning(f'Stats skipped for {feature}: {e}')

    for ax in axes[len(features):]:
        ax.axis('off')

    handles, labels = axes[0].get_legend_handles_labels()

    fig.suptitle(title, fontsize=18, y=0.995)
    fig.tight_layout()

    if handles and labels:
        fig.legend(
            handles[:len(hue_order)] if hue_order else handles,
            labels[:len(hue_order)] if hue_order else labels,
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            title=hue
        )

    fig.savefig(
        os.path.join(output_folder, save_name),
        bbox_inches='tight',
        pad_inches=0.1,
        dpi=300
    )

    plt.close(fig)

def plot_one_feature(data_raw, data_agg, feature, save_name, order=None, hue_order=None):
    plt.figure(figsize=(10, 6))

    ax = sns.stripplot(
        data=data_raw,
        x='peptide',
        y=feature,
        hue='cell',
        dodge=True,
        alpha=0.08,
        size=3,
        linewidth=0,
        order=order,
        hue_order=hue_order,
    )

    sns.boxplot(
        data=data_agg,
        x='peptide',
        y=feature,
        hue='cell',
        dodge=True,
        showfliers=False,
        boxprops={'facecolor': 'none'},
        linewidth=1.5,
        order=order,
        hue_order=hue_order,
    )

    sns.stripplot(
        data=data_agg,
        x='peptide',
        y=feature,
        hue='cell',
        dodge=True,
        size=8,
        edgecolor='black',
        linewidth=1,
        order=order,
        hue_order=hue_order,
    )

    ax.set_title(feature)
    ax.tick_params(axis='x', rotation=45)
    sns.despine()

    handles, labels = ax.get_legend_handles_labels()
    n = data_agg['cell'].nunique()
    ax.legend(handles[:n], labels[:n], title='cell', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f'results/plotting/{save_name}', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    logger.info('Loading data...')
    dfs = load_summary_data(input_folder)
    dfs = clean_names(dfs)

    percell_features = [
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

    peptide_order = sorted(
        dfs['percell']['peptide'].dropna().unique().tolist()
    )

    desired_cell_order = ['KO', 'WT', 'OE']
    cell_order = [
        c for c in desired_cell_order
        if c in dfs['percell']['cell'].dropna().unique()
    ]

    plotting_configs = [
        (
            'per cell, raw',
            dfs['percell'],
            dfs['percell_reps'],
            'percell_raw_by_peptide.png'
        ),
    ]

    logger.info('Generating stats plots...')

    pairs = []

    for peptide in peptide_order:
        pairs.extend([
            ((peptide, 'KO'), (peptide, 'WT')),
            ((peptide, 'KO'), (peptide, 'OE')),
            ((peptide, 'WT'), (peptide, 'OE')),
        ])

    plot_stats(
        data_raw=dfs['percell'],
        data_agg=dfs['percell_reps'],
        features=percell_features,
        title='Per-cell features',
        save_name='percell_stats.png',
        x='peptide',
        hue='cell',
        pairs=pairs,
        order=peptide_order,
        hue_order=cell_order
    )

    logger.info('plotting complete.')

    