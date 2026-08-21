# Cationic cell-penetrating peptide uptake and trafficking

This repository contains the analysis code and source data associated with the manuscript **“Lipoproteins modulate the uptake and biological function
of cationic cell-penetrating peptides across the animal lineage.
”** (citation and DOI pending).

This study investigates how natural cationic cell-penetrating peptides (+CPPs) interact with extracellular lipoproteins, enter cells through LDL receptor family–associated pathways, and traffic through intracellular compartments. The repository is organized by manuscript figure and panel so that the code, input data, and output associated with each analysis can be readily located.

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
