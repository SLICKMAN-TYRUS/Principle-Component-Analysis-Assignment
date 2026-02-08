# Principal Component Analysis Assignment

## Overview

This project implements PCA from scratch using World Bank development indicators for Sub-Saharan Africa. The implementation covers eigenvalues, eigenvectors, covariance matrices, and dimensionality reduction.

### Objectives

- Implement PCA without using sklearn
- Apply eigendecomposition on covariance matrices
- Select principal components based on explained variance threshold
- Reduce dimensionality while retaining most variance
- Benchmark the implementation
- Create visualizations of the transformed data

## Project Structure

```
Principle-Component-Analysis Assignment/
│
├── API_SSF_DS2_en_csv_v2_1688.csv                    # World Bank raw data
├── Metadata_Country_API_SSF_DS2_en_csv_v2_1688.csv   # Country metadata
├── Metadata_Indicator_API_SSF_DS2_en_csv_v2_1688.csv # Indicator metadata
├── data_exploration.ipynb                            # Data exploration & preparation
├── Template_PCA_Formative_1[Peer_Pair_Number].ipynb  # Main PCA implementation
├── prepared_data_for_pca.csv                         # Cleaned data (generated)
└── README.md                                         # This file
```

## Installation

### Requirements

- Python 3.8+
- Jupyter Notebook

### Setup Steps

Clone the repository:
```bash
git clone <your-repo-url>
cd Principle-Component-Analysis Assignment
```

Create a virtual environment (optional but recommended):

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

### Alternative: Install from requirements.txt

```bash
pip install -r requirements.txt
```

## How to Run

### Running the Data Exploration Notebook

First, run the data exploration notebook:

```bash
jupyter notebook data_exploration.ipynb
```

This will load the World Bank dataset, analyze missing values, select indicators, and save the cleaned data as `prepared_data_for_pca.csv`.

Note: Run all cells in order.

### Running the PCA Implementation

After preparing the data, open and run the main notebook:

```bash
jupyter notebook Template_PCA_Formative_1[Peer_Pair_Number].ipynb
```

Run all cells sequentially.

## Dataset

**Source:** World Bank - World Development Indicators (Sub-Saharan Africa)

**Characteristics:**
- Region: Sub-Saharan Africa (SSF)
- Time Period: 2000-2023 (24 years)
- Indicators: 20 development indicators
- Missing Values: Yes (handled in preprocessing)
- Non-numeric Columns: Yes (Country Name, Indicator Name)

**Indicator Types:**
- Economic indicators (exports, imports, trade)
- Social indicators (tourism, governance)
- Development metrics (high-tech exports, financial services)

The dataset meets all assignment requirements (missing values, non-numeric columns, more than 10 columns, African focus).

## Implementation

### Algorithm Steps

1. **Standardization** (using NumPy only)
   ```python
   standardized_data = (data - mean) / std_deviation
   ```

2. **Covariance Matrix Calculation**
   ```python
   cov_matrix = (X^T × X) / (n - 1)
   ```

3. **Eigendecomposition**
   ```python
   eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
   ```

4. **Sort Components by Variance**
   ```python
   sorted_indices = np.argsort(eigenvalues)[::-1]
   ```

5. **Dynamic Component Selection**
   - Target: 85% explained variance
   - Automatically selects optimal number of components

6. **Data Projection**
   ```python
   reduced_data = standardized_data × principal_components
   ```

### Key Features

- ✅ **No sklearn for PCA**: Pure NumPy implementation
- ✅ **Manual standardization**: Statistical normalization from scratch
- ✅ **Dynamic component selection**: Based on variance threshold
- ✅ **Performance benchmarking**: Execution time and memory analysis
6. **Data Projection**
   ```python
   reduced_data = standardized_data × principal_components
   ```

### Implementation Features

- No sklearn for PCA core functionality
- Manual standardization using NumPy
- Dynamic component selection based on variance threshold (85%)
- Performance benchmarking with execution time and memory analysis
- Visualizations: before/after PCA, biplots, 3D plots, component loadings
- Reconstruction analysis with quality metrics

## Expected Results

**Task 1**: PCA from Scratch
- Input: 24 observations × 20 features
- Output: 24 observations × 8-10 components (85% variance retained)
- Dimensionality reduction: approximately 50-60%

**Task 2**: Dynamic Component Selection
- Variance threshold: 85%
- Components automatically selected to meet threshold

**Task 3**: Performance Benchmarking
- Execution time: typically under 1 second
- Memory reduction: approximately 50%
- Reconstruction error: under 5% relative error

## Visualizations

The notebooks include:
- Data distribution histograms
- Correlation matrix heatmap
- Eigenvalue scree plot
- Cumulative variance plot
- Before/after PCA scatter plots (2D)
- 3D PCA visualization
- Biplot with feature loadings
- Component loadings bar charts
- Reconstruction quality comparisons
- Error analysis plots

## Assignment Requirements

All requirements met:
- Individual work
- Dataset has missing values
- Dataset has non-numeric columns
- More than 10 columns (20 indicators)
- African-focused data (Sub-Saharan Africa)
- Not a generic dataset (World Bank indicators)
- PCA implemented from scratch (Task 1)
- Manual standardization without sklearn
- Eigendecomposition applied
- Dynamic component selection (Task 2, 85% variance)
- Performance optimization (Task 3, benchmarking)
- All outputs visible in notebooks
- GitHub repository with documentation

## Mathematical Foundation

**Standardization:**
z = (x - μ) / σ

**Covariance Matrix:**
Cov(X) = (X^T × X) / (n - 1)

**Eigendecomposition:**
Cov(X)v = λv

where v is the eigenvector (principal component) and λ is the eigenvalue (variance explained)

**Explained Variance Ratio:**
EVR_i = λ_i / Σλ_j

**Data Projection:**
Z = X × W_k

where X is standardized data, W_k contains top k eigenvectors, and Z is the reduced data

## Troubleshooting

**"File not found: prepared_data_for_pca.csv"**
Run data_exploration.ipynb first to generate the file.

**"Memory Error"**
Reduce the number of features or use a smaller time range.

**Import errors**
```bash
pip install numpy pandas matplotlib seaborn jupyter
```

**Kernel crashes**
Restart the kernel and run cells from the beginning.

## References

- StatQuest: Principal Component Analysis (YouTube)
- How Is Explained Variance Used In PCA? (YouTube)
- World Bank World Development Indicators
- NumPy Linear Algebra Documentation

## Author

[Your Name]
February 2026
Advanced Linear Algebra Assignment

