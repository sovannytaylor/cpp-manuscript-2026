import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_rel

plt.rcParams["svg.fonttype"] = "none"

# =========================================================
# SETTINGS
# =========================================================

INPUT_FILE = Path(
    r"C:\Users\u244278\OneDrive - Baylor College of Medicine"
    r"\Documents\python_projects\ANA_26027_LDLR_Endo-inhib"
    r"\results\tables\26027_annotated_raw_all_6_bioreps.csv"
)

RESULTS_DIR = Path(
    r"C:\Users\u244278\OneDrive - Baylor College of Medicine"
    r"\Documents\python_projects\ANA_26027_LDLR_Endo-inhib"
    r"\results"
)

TABLE_DIR = RESULTS_DIR / "tables"
NORMALIZED_PLOT_DIR = RESULTS_DIR / "barplots_dmso_normalized"
RAW_PLOT_DIR = RESULTS_DIR / "barplots_gp30sub_raw"

for folder in [
    TABLE_DIR,
    NORMALIZED_PLOT_DIR,
    RAW_PLOT_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

PARAMS_TO_PLOT = [
    "ssc_median",
    "ssc_mean",
]

PARAM_LABELS = {
    "ssc_median": "Median AF594 fluorescence",
    "ssc_mean": "Mean AF594 fluorescence",
}

TREATMENT_ORDER = [
    "dmso",
    "dynasore",
    "amiloride",
    "pitstop",
    "genistein",
]

PLOT_PEPTIDES = [
    "GR30",
    "LL37",
    "CROT",
]

# GP30 subtraction will be applied only to these peptides.
# Add "CROT" here if CROT should also be GP30-subtracted.
GP30_SUBTRACT_PEPTIDES = {
    "GR30",
    "LL37",
}

# Rep C was excluded as a complete biological replicate because it failed
# replicate-level QC: plate-wide fluorescence was abnormally low and the
# expected treatment-response pattern was not preserved across peptides.
# The exclusion is applied uniformly to every peptide, treatment, metric,
# plot, and statistical test.



BAD_REPS = ["C"]

SHOW_PLOTS = False

OUTPUT_BIOREP_TABLE = (
    TABLE_DIR
    / "26027_5_qc_passing_reps_technical_wells_averaged.csv"
)

OUTPUT_CORRECTED_TABLE = (
    TABLE_DIR
    / "26027_5_qc_passing_reps_gp30sub_and_dmso_normalized.csv"
)

OUTPUT_STATS_TABLE = (
    TABLE_DIR
    / "26027_5_qc_passing_reps_paired_dmso_stats.csv"
)

OUTPUT_EXCLUDED_REPS_TABLE = (
    TABLE_DIR
    / "26027_excluded_rep_C_raw_rows_QC_audit.csv"
)

# =========================================================
# HELPERS
# =========================================================

def safe_name(text):
    return "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in str(text)
    )


def p_to_stars(p_value):
    if pd.isna(p_value):
        return ""

    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return "ns"


def holm_adjust(p_values):
    """
    Holm-adjust a group of p-values.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        len(p_values),
        np.nan,
        dtype=float,
    )

    valid_indices = np.where(
        np.isfinite(p_values)
    )[0]

    if len(valid_indices) == 0:
        return adjusted

    ordered_indices = valid_indices[
        np.argsort(
            p_values[valid_indices]
        )
    ]

    number_of_tests = len(
        ordered_indices
    )

    running_maximum = 0.0

    for rank, original_index in enumerate(
        ordered_indices
    ):
        candidate = (
            number_of_tests - rank
        ) * p_values[original_index]

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted[original_index] = min(
            running_maximum,
            1.0,
        )

    return adjusted


def save_figure(fig, output_directory, file_stem):
    png_path = output_directory / f"{safe_name(file_stem)}.png"
    svg_path = output_directory / f"{safe_name(file_stem)}.svg"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)

    if not png_path.exists():
        raise RuntimeError(
            f"PNG was not created:\n{png_path}"
        )

    if not svg_path.exists():
        raise RuntimeError(
            f"SVG was not created:\n{svg_path}"
        )

    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")


def plot_all_peptides(
    data,
    metric,
    value_column,
    normalized,
    stats_df,
    output_directory,
):
    """
    Make one combined plot with:
    - peptide on the x-axis
    - treatment represented by bar color
    - bar height = biological-replicate mean
    - error bar = SEM across biological replicates
    - one overlaid point per biological replicate
    - biological-replicate N shown below every bar
    """

    sub = (
        data[
            data["Peptide"].isin(PLOT_PEPTIDES)
            & data["Treatment"].isin(TREATMENT_ORDER)
        ]
        .dropna(
            subset=[
                "Peptide",
                "bio_rep",
                "Treatment",
                value_column,
            ]
        )
        .copy()
    )

    if sub.empty:
        print(
            f"Skipping combined plot | {metric}: "
            f"no values in {value_column}"
        )
        return

    available_peptides = [
        peptide
        for peptide in PLOT_PEPTIDES
        if peptide in sub["Peptide"].unique()
    ]

    available_treatments = [
        treatment
        for treatment in TREATMENT_ORDER
        if treatment in sub["Treatment"].unique()
    ]

    fig, ax = plt.subplots(
        figsize=(9.5, 6.5)
    )

    treatment_colors = dict(
        zip(
            available_treatments,
            plt.get_cmap("tab10")(
                np.arange(
                    len(available_treatments)
                )
            ),
        )
    )

    group_width = 0.84
    bar_width = (
        group_width
        / max(
            len(available_treatments),
            1,
        )
    )

    summary_rows = []

    for peptide_index, peptide in enumerate(
        available_peptides
    ):
        for treatment_index, treatment in enumerate(
            available_treatments
        ):
            group = (
                sub[
                    (sub["Peptide"] == peptide)
                    & (
                        sub["Treatment"]
                        == treatment
                    )
                ][
                    [
                        "bio_rep",
                        value_column,
                    ]
                ]
                .dropna()
                .copy()
            )

            if group.empty:
                continue

            values = group[
                value_column
            ].to_numpy(dtype=float)

            n_bioreps = group[
                "bio_rep"
            ].nunique()

            mean_value = np.mean(values)

            if n_bioreps > 1:
                sem_value = (
                    np.std(
                        values,
                        ddof=1,
                    )
                    / np.sqrt(n_bioreps)
                )
            else:
                sem_value = np.nan

            bar_offset = (
                -group_width / 2
                + bar_width / 2
                + treatment_index * bar_width
            )

            bar_x = (
                peptide_index
                + bar_offset
            )

            ax.bar(
                bar_x,
                mean_value,
                width=bar_width * 0.92,
                color=treatment_colors[treatment],
                edgecolor="black",
                linewidth=0.6,
                label=(
                    treatment.upper()
                    if peptide_index == 0
                    else None
                ),
                zorder=2,
            )

            if np.isfinite(sem_value):
                ax.errorbar(
                    bar_x,
                    mean_value,
                    yerr=sem_value,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1,
                    capsize=3,
                    capthick=1,
                    zorder=4,
                )

            # Deterministic horizontal offsets keep points
            # visible without random movement between runs.
            if len(values) == 1:
                point_offsets = np.array([0.0])
            else:
                point_offsets = np.linspace(
                    -bar_width * 0.24,
                    bar_width * 0.24,
                    len(values),
                )

            point_group = group.sort_values(
                "bio_rep"
            )

            ax.scatter(
                bar_x + point_offsets,
                point_group[
                    value_column
                ].to_numpy(dtype=float),
                s=25,
                facecolor="white",
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
            )

            summary_rows.append(
                {
                    "Peptide": peptide,
                    "Treatment": treatment,
                    "bar_x": bar_x,
                    "mean": mean_value,
                    "sem": sem_value,
                    "n_bioreps": n_bioreps,
                }
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    if normalized:
        ax.axhline(
            1,
            color="0.5",
            linestyle="--",
            linewidth=1,
            zorder=0,
        )

    ax.set_xticks(
        range(len(available_peptides))
    )

    ax.set_xticklabels(
        available_peptides,
    )

    finite_values = (
        sub[value_column]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    minimum_value = np.min(finite_values)
    maximum_value = np.max(finite_values)

    value_range = maximum_value - minimum_value

    if (
        not np.isfinite(value_range)
        or value_range == 0
    ):
        value_range = max(
            abs(maximum_value),
            1.0,
        )

    # Add Holm-adjusted statistics above each
    # treatment bar on normalized plots.
    if normalized:
        for _, summary_row in (
            summary_df.iterrows()
        ):
            treatment = summary_row[
                "Treatment"
            ]

            if treatment == "dmso":
                continue

            result = stats_df[
                (stats_df["metric"] == metric)
                & (
                    stats_df["Peptide"]
                    == summary_row["Peptide"]
                )
                & (
                    stats_df["Treatment"]
                    == treatment
                )
            ]

            if result.empty:
                continue

            label = result[
                "stars_holm"
            ].iloc[0]

            error_height = (
                summary_row["sem"]
                if np.isfinite(
                    summary_row["sem"]
                )
                else 0
            )

            ax.text(
                summary_row["bar_x"],
                summary_row["mean"]
                + error_height
                + 0.05 * value_range,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Reserve space below zero for the N labels.
    n_label_y = min(
        0,
        minimum_value,
    ) - 0.10 * value_range

    for _, summary_row in (
        summary_df.iterrows()
    ):
        ax.text(
            summary_row["bar_x"],
            n_label_y,
            f"N={int(summary_row['n_bioreps'])}",
            ha="center",
            va="top",
            fontsize=7,
            rotation=90,
        )

    lower_limit = (
        n_label_y
        - 0.16 * value_range
    )

    upper_limit = (
        maximum_value
        + 0.28 * value_range
    )

    ax.set_ylim(
        lower_limit,
        upper_limit,
    )

    metric_label = PARAM_LABELS.get(
        metric,
        metric,
    )

    if normalized:
        y_label = (
            f"{metric_label} "
            "(fold of matched DMSO)"
        )

        file_suffix = (
            "all_peptides_fold_matched_dmso"
        )
        title_suffix = (
            "DMSO-normalized"
        )

    else:
        y_label = (
            f"{metric_label} "
            "(corrected value)"
        )
        file_suffix = (
            "all_peptides_corrected_raw"
        )
        title_suffix = (
            "corrected values"
        )

    ax.set_ylabel(y_label)
    ax.set_xlabel("Peptide")

    ax.set_title(
        f"All peptides | {title_suffix}"
    )

    ax.legend(
        title="Treatment",
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    save_figure(
        fig=fig,
        output_directory=output_directory,
        file_stem=(
            f"{metric}_{file_suffix}"
        ),
    )


# =========================================================
# LOAD CONCATENATED CSV
# =========================================================

print(f"Reading: {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input CSV was not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "well",
    "bio_rep",
    "Peptide",
    "Treatment",
    *PARAMS_TO_PLOT,
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        f"{sorted(missing_columns)}"
    )

df["bio_rep"] = (
    df["bio_rep"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["Peptide"] = (
    df["Peptide"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["Treatment"] = (
    df["Treatment"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace(
        {
            "pistop": "pitstop",
        }
    )
)

for metric in PARAMS_TO_PLOT:
    df[metric] = pd.to_numeric(
        df[metric],
        errors="coerce",
    )

if BAD_REPS:
    excluded_df = df[
        df["bio_rep"].isin(BAD_REPS)
    ].copy()

    if excluded_df.empty:
        raise ValueError(
            "No rows matched BAD_REPS. "
            f"Requested exclusions: {BAD_REPS}; "
            "available replicates: "
            f"{sorted(df['bio_rep'].unique())}"
        )

    excluded_df.to_csv(
        OUTPUT_EXCLUDED_REPS_TABLE,
        index=False,
    )

    print(
        "Excluded biological replicates:",
        BAD_REPS,
    )

    print(
        "Saved excluded raw rows for QC audit:",
        OUTPUT_EXCLUDED_REPS_TABLE,
    )

    df = df[
        ~df["bio_rep"].isin(BAD_REPS)
    ].copy()


# ADD IT HERE: exclude replicate A from GR30 only
df = df[
    ~(
        (df["Peptide"] == "GR30")
        & (df["bio_rep"] == "A")
    )
].copy()


# GP30 must remain in the table so it can be used
# as the matched background reference.
df = df[
    df["Peptide"].isin(
        PLOT_PEPTIDES + ["GP30"]
    )
    & df["Treatment"].isin(
        TREATMENT_ORDER
    )
].copy()
# GP30 must remain in the table so it can be used
# as the matched background reference.
df = df[
    df["Peptide"].isin(
        PLOT_PEPTIDES + ["GP30"]
    )
    & df["Treatment"].isin(
        TREATMENT_ORDER
    )
].copy()

if df.empty:
    raise ValueError(
        "No rows remained after filtering."
    )

print(
    "Biological replicates:",
    sorted(df["bio_rep"].unique()),
)

print(
    "Peptides:",
    sorted(df["Peptide"].unique()),
)

print(
    "Treatments:",
    sorted(df["Treatment"].unique()),
)

# =========================================================
# STEP 1: AVERAGE TECHNICAL WELLS
#
# One row per:
# bio_rep + peptide + treatment
# =========================================================

aggregation = {
    "well": "nunique",
}

for metric in PARAMS_TO_PLOT:
    aggregation[metric] = "mean"

biorep_df = (
    df
    .groupby(
        [
            "bio_rep",
            "Peptide",
            "Treatment",
        ],
        as_index=False,
    )
    .agg(aggregation)
    .rename(
        columns={
            "well": "n_technical_wells",
        }
    )
)

biorep_df.to_csv(
    OUTPUT_BIOREP_TABLE,
    index=False,
)

print(
    "Saved technical-well averaged table:",
    OUTPUT_BIOREP_TABLE,
)

# =========================================================
# STEP 2: MATCHED GP30 BACKGROUND SUBTRACTION
#
# Match using:
# bio_rep + treatment
#
# Example:
# GR30, rep A, dynasore
# minus
# GP30, rep A, dynasore
# =========================================================

corrected_df = biorep_df.copy()

for metric in PARAMS_TO_PLOT:
    gp30_reference_column = (
        f"{metric}_matched_gp30"
    )

    corrected_column = (
        f"{metric}_gp30sub"
    )

    gp30_reference = (
        biorep_df[
            biorep_df["Peptide"] == "GP30"
        ][
            [
                "bio_rep",
                "Treatment",
                metric,
            ]
        ]
        .rename(
            columns={
                metric: gp30_reference_column,
            }
        )
    )

    corrected_df = corrected_df.merge(
        gp30_reference,
        on=[
            "bio_rep",
            "Treatment",
        ],
        how="left",
        validate="many_to_one",
    )

    # Start with the original value.
    corrected_df[
        corrected_column
    ] = corrected_df[metric]

    subtraction_mask = (
        corrected_df["Peptide"].isin(
            GP30_SUBTRACT_PEPTIDES
        )
    )

    # Subtract GP30 from the matched replicate
    # and matched treatment.
    corrected_df.loc[
        subtraction_mask,
        corrected_column,
    ] = (
        corrected_df.loc[
            subtraction_mask,
            metric,
        ]
        - corrected_df.loc[
            subtraction_mask,
            gp30_reference_column,
        ]
    ).clip(lower=0)

    missing_gp30 = corrected_df[
        subtraction_mask
        & corrected_df[
            gp30_reference_column
        ].isna()
    ]

    if not missing_gp30.empty:
        print(
            f"\nWARNING: Missing matched GP30 "
            f"references for {metric}:"
        )

        print(
            missing_gp30[
                [
                    "bio_rep",
                    "Peptide",
                    "Treatment",
                ]
            ]
            .drop_duplicates()
            .to_string(index=False)
        )

# =========================================================
# STEP 3: MATCH EACH TREATMENT TO THAT REP'S DMSO
#
# Match using:
# bio_rep + peptide
#
# Example:
# corrected GR30, rep A, dynasore
# divided by
# corrected GR30, rep A, DMSO
# =========================================================

for metric in PARAMS_TO_PLOT:
    corrected_column = (
        f"{metric}_gp30sub"
    )

    dmso_reference_column = (
        f"{corrected_column}_matched_dmso"
    )

    normalized_column = (
        f"{metric}_fold_dmso"
    )

    dmso_reference = (
        corrected_df[
            (
                corrected_df["Treatment"]
                == "dmso"
            )
            & corrected_df["Peptide"].isin(
                PLOT_PEPTIDES
            )
        ][
            [
                "bio_rep",
                "Peptide",
                corrected_column,
            ]
        ]
        .rename(
            columns={
                corrected_column:
                dmso_reference_column,
            }
        )
    )

    corrected_df = corrected_df.merge(
        dmso_reference,
        on=[
            "bio_rep",
            "Peptide",
        ],
        how="left",
        validate="many_to_one",
    )

    corrected_df[
        normalized_column
    ] = (
        corrected_df[corrected_column]
        / corrected_df[dmso_reference_column]
    )

    # A corrected DMSO value of zero cannot be used
    # as a normalization denominator.
    invalid_dmso = (
        corrected_df[
            dmso_reference_column
        ].isna()
        | corrected_df[
            dmso_reference_column
        ].le(0)
    )

    corrected_df.loc[
        invalid_dmso,
        normalized_column,
    ] = np.nan

    invalid_groups = corrected_df[
        invalid_dmso
        & corrected_df["Peptide"].isin(
            PLOT_PEPTIDES
        )
    ][
        [
            "bio_rep",
            "Peptide",
        ]
    ].drop_duplicates()

    if not invalid_groups.empty:
        print(
            f"\nWARNING: {metric} could not be "
            "DMSO-normalized for these groups "
            "because the matched corrected DMSO "
            "value was missing or <= 0:"
        )

        print(
            invalid_groups.to_string(
                index=False
            )
        )

corrected_df.to_csv(
    OUTPUT_CORRECTED_TABLE,
    index=False,
)

print(
    "Saved corrected and normalized table:",
    OUTPUT_CORRECTED_TABLE,
)

# =========================================================
# STEP 4: PAIRED STATISTICS
#
# Pairing variable:
# bio_rep
#
# Each inhibitor is compared with DMSO from the
# same peptide and biological replicate.
# =========================================================

stats_rows = []

for metric in PARAMS_TO_PLOT:
    normalized_column = (
        f"{metric}_fold_dmso"
    )

    for peptide in PLOT_PEPTIDES:
        peptide_data = corrected_df[
            corrected_df["Peptide"] == peptide
        ]

        wide = peptide_data.pivot_table(
            index="bio_rep",
            columns="Treatment",
            values=normalized_column,
            aggfunc="mean",
        )

        for treatment in TREATMENT_ORDER:
            if treatment == "dmso":
                continue

            if (
                "dmso" not in wide.columns
                or treatment not in wide.columns
            ):
                paired = pd.DataFrame()
                t_statistic = np.nan
                p_value = np.nan

            else:
                paired = wide[
                    [
                        "dmso",
                        treatment,
                    ]
                ].dropna()

                if len(paired) >= 2:
                    (
                        t_statistic,
                        p_value,
                    ) = ttest_rel(
                        paired[treatment],
                        paired["dmso"],
                    )

                else:
                    t_statistic = np.nan
                    p_value = np.nan

            stats_rows.append(
                {
                    "metric": metric,
                    "Peptide": peptide,
                    "Treatment": treatment,
                    "comparison": (
                        f"{treatment}_vs_dmso"
                    ),
                    "n_paired": len(paired),
                    "t_stat": t_statistic,
                    "p_value": p_value,
                }
            )

stats_df = pd.DataFrame(stats_rows)

stats_df["p_value_holm"] = np.nan

for _, group_indices in (
    stats_df
    .groupby(
        [
            "metric",
            "Peptide",
        ]
    )
    .groups
    .items()
):
    group_indices = list(group_indices)

    stats_df.loc[
        group_indices,
        "p_value_holm",
    ] = holm_adjust(
        stats_df.loc[
            group_indices,
            "p_value",
        ].to_numpy()
    )

stats_df["stars_holm"] = (
    stats_df["p_value_holm"]
    .map(p_to_stars)
)

stats_df.to_csv(
    OUTPUT_STATS_TABLE,
    index=False,
)

print(
    "Saved paired statistics table:",
    OUTPUT_STATS_TABLE,
)

# =========================================================
# STEP 5: MAKE AND SAVE PLOTS
# =========================================================

print("\nStarting plot generation...")

for metric in PARAMS_TO_PLOT:
    corrected_column = (
        f"{metric}_gp30sub"
    )

    normalized_column = (
        f"{metric}_fold_dmso"
    )

    plot_all_peptides(
        data=corrected_df,
        metric=metric,
        value_column=normalized_column,
        normalized=True,
        stats_df=stats_df,
        output_directory=NORMALIZED_PLOT_DIR,
    )

    plot_all_peptides(
        data=corrected_df,
        metric=metric,
        value_column=corrected_column,
        normalized=False,
        stats_df=stats_df,
        output_directory=RAW_PLOT_DIR,
    )

print("\nDONE")

print(
    "Normalized plots saved to:",
    NORMALIZED_PLOT_DIR,
)

print(
    "Raw or GP30-subtracted plots saved to:",
    RAW_PLOT_DIR,
)