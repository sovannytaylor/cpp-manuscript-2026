# Cationic cell-penetrating peptide uptake and trafficking

This repository contains the analysis code and source data associated with the manuscript **“[Insert manuscript title]”** (citation and DOI pending).

This study investigates how natural cationic cell-penetrating peptides (+CPPs) interact with extracellular lipoproteins, enter cells through LDL receptor family-associated pathways, and traffic through intracellular compartments. The repository is organized by manuscript figure and panel so that the code, input data, and output associated with each analysis can be readily located.

## Biological background

Cationic cell-penetrating peptides are positively charged peptides capable of entering cells and transporting molecular cargo. These peptides originate from diverse biological sources, including antimicrobial peptides, venoms, and disease-associated proteins. Although +CPPs are widely used as delivery tools, the mechanisms controlling their cellular uptake and intracellular trafficking remain incompletely understood.

In this study, we examine a diverse panel of natural +CPPs to determine how extracellular interactions and peptide properties influence cellular entry. We investigate the contributions of lipoprotein binding, the LDL receptor family, and endocytic pathways to peptide uptake. We also characterize the intracellular trafficking of selected peptides across endosomal, lysosomal, secretory, autophagic, and damage-associated compartments.

Our findings support a model in which +CPP uptake and intracellular behavior depend on peptide identity, concentration, extracellular environment, receptor interactions, and cellular context.

## Repository organization

This repository is organized by manuscript figure and panel:

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── Figure_1/
│   ├── Panel_A/
│   │   ├── code/
│   │   ├── data/
│   │   │   ├── raw/
│   │   │   └── processed/
│   │   ├── output/
│   │   └── README.md
│   └── assembled_figure/
├── Figure_2/
├── Figure_3/
├── Figure_4/
├── Figure_5/
├── Figure_6/
├── Supplementary_Figures/
├── shared_code/
├── shared_data/
└── raw_data/
```

Each panel folder may contain:

- `code/`: scripts used to process the data and generate the panel
- `data/raw/`: original tabular measurements used in the analysis
- `data/processed/`: processed data supplied directly to the plotting script
- `output/`: generated plots and other panel outputs
- `README.md`: panel-specific descriptions and instructions

Data and functions used by multiple panels are stored in `shared_data/` and `shared_code/`, respectively.

## Data management

Plot-level source data underlying the quantitative figures are provided as CSV files in the corresponding figure and panel directories. Depending on the analysis, these files contain measurements summarized at the biological-replicate, well, cell, vesicle, punctum, or other object level.

Large instrument-generated files, including raw microscopy images, flow cytometry files, and intermediate image-analysis arrays, are not stored directly in this GitHub repository because of their size. These files will be deposited in an appropriate public data repository. Accession numbers and download instructions will be added to the [`raw_data`](raw_data/) directory when available.

Processed masks, quality-control images, and intermediate analysis outputs may also be deposited externally when they are necessary to reproduce an analysis but are too large for GitHub.

Any exclusions, manual quality-control decisions, or other changes to the data are documented in the relevant analysis directory.

## Analyses

### Peptide uptake analysis

Cellular uptake of fluorescently labeled +CPPs was quantified using flow cytometry and fluorescence microscopy. Analysis scripts were used to import measurements, apply sample annotations, perform background correction or normalization where appropriate, summarize biological replicates, conduct statistical comparisons, and generate plots.

The relevant scripts and source data are provided in the directories corresponding to each manuscript panel.

### LDL receptor analysis

The contribution of the LDL receptor and associated pathways to +CPP uptake was assessed using genetic screening and validation experiments. These analyses include CRISPR interference screening, comparisons between LDLR knockout and overexpression models, and evaluation of peptide uptake across cell populations.

Where applicable, per-cell measurements and biological-replicate summaries are provided with the corresponding figure panels.

### Chemical-inhibitor analysis

The effects of pharmacological perturbations on peptide uptake were analyzed across independent biological replicates. Fluorescence measurements were corrected for drug-associated background when appropriate and normalized to the matched peptide-treated control within each replicate.

Scripts used for data assembly, normalization, statistical testing, and visualization are supplied with the associated figure panels.

### Intracellular trafficking analysis

Multiplexed fluorescence imaging was used to characterize the intracellular localization of selected +CPPs relative to markers of early endosomes, late endosomes, lysosomes, recycling compartments, the Golgi apparatus, autophagic compartments, and other cellular structures.

Cells and peptide-positive puncta were segmented using custom Python workflows. Segmentation masks were manually inspected and corrected where necessary. Marker intensities, spatial overlap, enrichment, and distance-based features were then calculated for individual peptide-positive structures.

Dimensionality-reduction analyses, including principal component analysis and UMAP, were used to examine trafficking profiles across peptides and intracellular structures. Peptide fluorescence intensity was used as an annotation or overlay when specified and was not necessarily included as a feature in the embedding itself.

### Endosomal damage and galectin analysis

Galectin enrichment was examined in LAMP1- and RAB7-positive structures to identify compartments associated with membrane damage or repair. Peptide intensity was compared between structures with and without significant GAL3 or GAL8 enrichment.

The applicable object-level and cell-level data, analysis scripts, and statistical results are included in the corresponding figure directories.

### Image segmentation and quality control

Image-analysis workflows were developed in Python using packages including [Cellpose](https://www.cellpose.org/), [napari](https://napari.org/), [scikit-image](https://scikit-image.org/), NumPy, and pandas.

Depending on the experiment, the workflows included:

1. Image import and preprocessing
2. Cell, nucleus, punctum, or organelle segmentation
3. Manual mask inspection and correction
4. Image and object quality control
5. Feature extraction
6. Data aggregation
7. Statistical analysis and visualization

Analysis-specific segmentation settings and quality-control criteria are documented with the relevant scripts.

### Statistical analysis

Statistical analyses were performed using Python and, where indicated in the manuscript, GraphPad Prism. The statistical test, experimental unit, number of biological replicates, and treatment of individual cells or objects are documented in the relevant figure legends, source-data files, or panel-specific README files.

## Reproducing the analyses

### Prerequisites

The Python dependencies required to run the analyses are listed in [`requirements.txt`](requirements.txt) and/or [`environment.yml`](environment.yml).

To create the conda environment, run:

```bash
conda env create -f environment.yml
conda activate cpp-analysis
```

Alternatively, dependencies can be installed using pip:

```bash
pip install -r requirements.txt
```

Exact package requirements may differ among analyses. Additional requirements are documented in the applicable panel directories when necessary.

### Workflow

To reproduce a figure panel:

1. Navigate to the directory for the relevant figure and panel.
2. Read the panel-specific `README.md`.
3. Confirm that the necessary input files are present in `data/`.
4. Run the scripts in numerical order.
5. Locate the generated panel in `output/`.

For example:

```bash
cd Figure_4/Panel_C
python code/01_prepare_data.py
python code/02_generate_figure.py
```

Scripts should be executed from the location specified in the panel-specific README. Relative file paths are used whenever possible so that the repository can run on different computers.

Some analyses require large files stored in an external data repository. Download instructions will be provided in the relevant panel directory or in [`raw_data/README.md`](raw_data/README.md).

## Source-data conventions

Source-data filenames identify the corresponding manuscript figure and panel. For example:

```text
figure_4C_per_vesicle_source_data.csv
figure_4C_per_cell_summary.csv
figure_4C_statistical_results.csv
figure_4C_split_violin.svg
```

Where possible, source-data tables include the following identifying information:

- Experiment or biological replicate
- Plate and well
- Experimental condition
- Peptide identity
- Peptide concentration and treatment duration
- Cell or object identifier
- Measurement name
- Measurement value and units

A data dictionary is included when column names or values are not self-explanatory.

## Citation

If you use the code or data from this repository, please cite:

> [Authors]. “[Manuscript title].” [Journal or preprint server], [year]. DOI: [pending].

A permanent archival DOI for this repository will be added following deposition of a release on Zenodo.

## License

Code in this repository is available under the license specified in [`LICENSE`](LICENSE). Data reuse is subject to the terms described in the manuscript and associated data repository.

## Contact

For questions about the analysis or data, please contact:

**Sophie Taylor**  
Boeynaems Laboratory  
Baylor College of Medicine  
sovanny.taylor@bcm.edu
