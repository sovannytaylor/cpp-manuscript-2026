# Cationic Cell-Penetrating Peptide Uptake and Trafficking

This repository contains the analysis code and source data associated with the manuscript **“Lipoproteins modulate the uptake and biological function of cationic cell-penetrating peptides across the animal lineage”** (citation and DOI pending). Links to the Zenodo archive and the appropriate microscopy data repository will be added when available.

This study investigates how natural cationic cell-penetrating peptides (+CPPs) interact with extracellular lipoproteins, enter cells through pathways associated with the LDL receptor family, and traffic through intracellular compartments. The repository is organized by manuscript figure and panel so that the code, input data, and outputs associated with each analysis can be readily located.

## Biological Background

Cationic cell-penetrating peptides are positively charged peptides capable of entering cells and transporting molecular cargo. They originate from diverse biological sources, including antimicrobial peptides, venoms, and disease-associated proteins. Although +CPPs are widely used as delivery tools, the mechanisms governing their cellular uptake and intracellular trafficking remain incompletely understood.

In this study, we examine a diverse panel of natural +CPPs to determine how extracellular interactions and peptide properties influence cellular entry. We investigate the contributions of lipoprotein binding, the LDL receptor family, and endocytic pathways to peptide uptake. We also characterize the intracellular trafficking of selected peptides through endosomal, lysosomal, secretory, autophagic, and damage-associated compartments.

Our findings support a model in which +CPP uptake and intracellular behavior depend on peptide identity and concentration, the extracellular environment, receptor interactions, and cellular context.

## Repository Organization

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
│   │   ├── output/
│   │   └── README.md
│   └── ...
├── Figure_3/
├── Figure_4/
├── Figure_5/
├── Figure_6/
└── Supplementary_Figures/
```

Each panel directory may contain:

- `code/`: scripts used to process data and generate the corresponding panel
- `output/`: generated plots and other analysis outputs
- `README.md`: panel-specific descriptions and instructions

Raw data can be downloaded using the Zenodo link that will be provided upon deposition.

## Data Management

Plot-level source data underlying the quantitative figures are provided as CSV files in the corresponding figure and panel directories. Depending on the analysis, these files contain measurements summarized at the biological-replicate, well, cell, vesicle, punctum, or other object level.

Large instrument-generated files, including raw microscopy images, flow cytometry files, and intermediate image-analysis arrays, are not stored directly in this GitHub repository because of their size. These files will be deposited in an appropriate public data repository. Accession numbers and download instructions will be added to the [`raw_data`](raw_data/) directory when available.

Processed masks, quality-control images, and intermediate analysis outputs may also be deposited externally when they are required to reproduce an analysis but are too large for GitHub.

Any exclusions, manual quality-control decisions, or other data-processing changes are documented in the relevant analysis directory.

## Analyses

### Peptide Uptake Analysis (Figure 1, Panels C–D)

Colocalization between fluorescently labeled +CPPs and EEA1-positive puncta was quantified using fluorescence microscopy. Analysis scripts were used to segment cells, +CPP-positive puncta, and EEA1-positive structures. A parameter grid search was performed to identify an appropriate thresholding method for each peptide, with the negative-control peptide GP30 used to establish the global lower threshold. Structures were considered colocalized when their masks overlapped directly or were separated by no more than 2 pixels. Cell-level measurements were then generated for visualization and statistical analysis.

### Endocytosis-Inhibitor Analysis (Figure 1, Panel L)

Cellular uptake of fluorescently labeled +CPPs following treatment with endocytosis inhibitors was quantified using flow cytometry. The analysis scripts use CSV files exported from FlowJo that include the well ID, mean and median fluorescence intensities, frequency of the defined subset, event count, and other parameters used to characterize the data distributions.

### Iterative Indirect Immunofluorescence Imaging (4i) Analysis (Figure 3 and Figures S8–S10)

Multiplexed fluorescence imaging was used to characterize the intracellular localization of selected +CPPs relative to markers of early endosomes, late endosomes, lysosomes, recycling compartments, the Golgi apparatus, autophagic compartments, and other cellular structures.

Cells and peptide-positive puncta were segmented using custom Python workflows. Segmentation masks were manually inspected and corrected where necessary. Marker intensities, spatial overlap, enrichment, and distance-based features were then calculated for individual peptide-positive structures. Additional details are provided in the manuscript Methods and Supplementary Information.

### 4i Visualizations (Figure 3, Panel B)

Cell-level montages were generated from stored cell coordinates and registered marker images. The Python scripts used to extract and assemble the PNG images are provided in the corresponding panel directory.

### Dimensionality-Reduction Analyses for 4i Data (Figure 3, Panels E, F, and H; Figure S10)

Dimensionality-reduction methods, including principal component analysis (PCA) and uniform manifold approximation and projection (UMAP), were used to examine trafficking profiles across peptides and intracellular structures. Where indicated, peptide fluorescence intensity was displayed as an annotation or overlay but was not included as a feature when calculating the embedding. Additional details are provided in the Supplementary Information.

### Endosomal Damage and Galectin Analysis (Figure 3, Panel G; Figures S3 and S8)

Galectin enrichment was examined in LAMP1- and RAB7-positive structures to identify compartments associated with membrane damage or repair. Peptide intensity was compared between structures with and without significant GAL3 or GAL8 enrichment.

The applicable object-level and cell-level data, analysis scripts, and statistical results are provided in the corresponding figure directories.

### LDL Receptor Analysis (Figure 4, Panel G; Figure S5)

The contribution of the LDL receptor and associated pathways to +CPP uptake was assessed using LDLR-knockout and LDLR-overexpression models. Peptide uptake was then evaluated across the resulting cell populations.

Where applicable, per-cell measurements and biological-replicate summaries are provided with the corresponding figure panels.

### HDL, VLDL, and LDL Analysis (Figure 5, Panel C; Figure S6)

Colocalization between fluorescently labeled +CPPs and LDL-, HDL-, or VLDL-positive puncta was quantified using fluorescence microscopy. Analysis scripts were used to segment cells, +CPP-positive puncta, and lipoprotein-positive structures. A parameter grid search was performed to identify an appropriate thresholding method for each peptide and lipoprotein species, with the negative-control peptide GP30 used to establish the global lower threshold. Structures were considered colocalized when their masks overlapped directly or were separated by no more than 2 pixels. Cell-level measurements were then generated for visualization and statistical analysis.

### LDL Analysis Across All +CPPs (Figure 5, Panel E; Figure S6)

This analysis followed the same workflow used for Figure 5, Panel C, but included the complete peptide panel and evaluated colocalization only with LDL-positive puncta.

### Microscale Thermophoresis and Spectral-Shift Analysis (Figure 5, Panel F; Table 8)

The scripts in this directory visualize measurements obtained from microscale thermophoresis and spectral-shift assays. Experimental measurements were supplied to the analysis workflows as CSV files.

### Zeta-Potential Visualization (Figure 5, Panel G)

The scripts in this directory visualize zeta-potential measurements supplied as CSV files.

### Minimum Inhibitory Concentration Assay Analysis (Figure 6, Panel A)

The scripts in this directory visualize measurements obtained from minimum inhibitory concentration (MIC) assays and supplied as CSV files.

### *Naegleria gruberi* Morphology and Propidium Iodide Staining (Figure 6, Panels D–E; Figure S7)

This workflow imports CZI files and converts the image data to NumPy arrays while preserving channel-intensity information. Cells are then segmented, and the resulting masks are manually validated in napari. Morphological features are quantified, and propidium iodide (PI) staining is thresholded using the no-peptide controls to establish background signal. Subsequent scripts generate cell-level measurements, statistical analyses, and visualizations.

### Cryo-Electron Tomography Data Visualization (Figure S6)

Cryo-electron tomography (cryo-ET) data were analyzed using separate software. The resulting measurements were imported as CSV files and visualized using the scripts provided here.

### Image Segmentation and Quality Control

Image-analysis workflows were developed in Python using packages including [Cellpose](https://www.cellpose.org/), [napari](https://napari.org/), [scikit-image](https://scikit-image.org/), NumPy, and pandas.

Depending on the experiment, the workflows included:

1. Image import and preprocessing
2. Cell, nucleus, punctum, or organelle segmentation
3. Manual mask inspection and correction
4. Image- and object-level quality control
5. Feature extraction
6. Data aggregation
7. Statistical analysis and visualization

Analysis-specific segmentation settings and quality-control criteria are documented with the relevant scripts.

### Statistical Analysis

Statistical analyses were performed using Python and, where indicated in the manuscript, GraphPad Prism. The statistical test, experimental unit, number of biological replicates, and treatment of individual cells or objects are documented in the relevant figure legends, source-data files, or panel-specific README files.

## Prerequisites

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

## Source-Data Conventions

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

A permanent archival DOI for this repository will be added after a release is deposited on Zenodo.

## License

Code in this repository is available under the license specified in [`LICENSE`](LICENSE). Data reuse is subject to the terms described in the manuscript and associated data repository.

## Contact

For questions about the analysis or data, please contact:

**Sophie Taylor**  
Boeynaems Laboratory  
Baylor College of Medicine  
[sovanny.taylor@bcm.edu](mailto:sovanny.taylor@bcm.edu)
