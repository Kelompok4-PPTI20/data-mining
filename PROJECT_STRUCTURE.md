# Project_DATAMINING — File Structure

## Overview
This document outlines the directory structure and organization of the Project_DATAMINING repository.

## Directory Tree

```
Project_DATAMINING/
├── requirements.txt         
├── data/                  
│   ├── raw/                 
│   │   └── churn.csv        
│   └── processed/
|        └── churn_clean.csv           # Dropped exited label
├── notebooks/                
│   └── notebook.ipynb      
└── src/                      
```

## Directory Descriptions

### `requirements.txt`
Python package dependencies required to run the project. Install with:
```bash
pip install -r requirements.txt
```

### `data/`
Stores all project data files.

- **`raw/`** — Original, unmodified data files
  - `churn.csv` — Customer churn dataset
  
- **`processed/`** — Cleaned, transformed, and feature-engineered data ready for modeling

### `notebooks/`
Jupyter notebooks for exploratory data analysis, visualization, and experimentation.

- `notebook.ipynb` — Main analysis and modeling notebook

### `src/`
Python source code and reusable utilities for the project.

---

**Last Updated:** 2026-04-11
