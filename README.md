# Principal Component Analysis Assignment

## Overview

This project implements PCA from scratch using World Bank development indicators for Sub-Saharan Africa. The implementation covers eigenvalues, eigenvectors, covariance matrices, and dimensionality reduction.

### Goals

- Implement PCA without sklearn
- Calculate covariance matrices and apply eigendecomposition
- Select principal components based on explained variance
- Reduce dimensions while keeping most of the variance
- Benchmark the implementation
- Create visualizations

## Viewing the Notebooks

**NOTE:** The notebooks have all the executed cells and outputs (752KB and 589KB), but GitHub might not render large notebooks directly in the browser.

**How to view:**

1. **Clone and open locally (recommended for grading):**
   ```bash
   git clone https://github.com/SLICKMAN-TYRUS/Principle-Component-Analysis-Assignment.git
   cd Principle-Component-Analysis-Assignment
   jupyter notebook
   ```

2. **Download files:** Click on each `.ipynb` file then click "Download" button and open in Jupyter

3. **Use nbviewer:**
   - [Template_PCA_Formative_1[Peer_Pair_Number].ipynb](https://nbviewer.org/github/SLICKMAN-TYRUS/Principle-Component-Analysis-Assignment/blob/main/Template_PCA_Formative_1%5BPeer_Pair_Number%5D.ipynb)
   - [data_exploration.ipynb](https://nbviewer.org/github/SLICKMAN-TYRUS/Principle-Component-Analysis-Assignment/blob/main/data_exploration.ipynb)

## Project Structure

```
├── API_SSF_DS2_en_csv_v2_1688.csv                    # Raw World Bank data
├── Metadata_Country_API_SSF_DS2_en_csv_v2_1688.csv   # Country metadata
├── Metadata_Indicator_API_SSF_DS2_en_csv_v2_1688.csv # Indicator metadata
├── data_exploration.ipynb                            # Data prep notebook
├── Template_PCA_Formative_1[Peer_Pair_Number].ipynb  # Main PCA notebook
├── prepared_data_for_pca.csv                         # Cleaned data
└── README.md                                         # This file
```

## Setup

### Requirements

- Python 3.8 or higher
- Jupyter Notebook

### Installation

Clone the repo:
```bash
git clone <your-repo-url>
cd Principle-Component-Analysis Assignment
```

Create virtual environment (optional):

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

Install packages:

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Data Exploration

Run the data exploration notebook first:

```bash
jupyter notebook data_exploration.ipynb
```

This loads the World Bank dataset, handles missing values, selects indicators, and saves cleaned data as `prepared_data_for_pca.csv`.

Run all cells in order.

### 2. PCA Implementation

After preparing data, run the main notebook:

```bash
jupyter notebook Template_PCA_Formative_1[Peer_Pair_Number].ipynb
```

Run all cells sequentially.

## Dataset

**Source:** World Bank Development Indicators (Sub-Saharan Africa)

- Region: Sub-Saharan Africa
- Period: 2000-2023 (24 years)
- Indicators: 20 development indicators
- Has missing values (handled in preprocessing)
- Has non-numeric columns (Country Name, Indicator Name)

Indicators include economic metrics (exports, imports, trade), social indicators (tourism, governance), and development metrics.

## Implementation

### Steps

1. **Standardization** (NumPy only)
   ```python
   standardized_data = (data - mean) / std_deviation
   ```

2. **Covariance Matrix**
   ```python
   cov_matrix = (X^T × X) / (n - 1)
   ```

3. **Eigendecomposition**
   ```python
   eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
   ```

4. **Sort Components**
   ```python
   sorted_indices = np.argsort(eigenvalues)[::-1]
   ```

5. **Select Components** (85% variance threshold)
   ```python
   num_components = np.argmax(cumulative_variance >= 0.85) + 1
   ```

6. **Project Data**
   ```python
   reduced_data = standardized_data × principal_components
   ```

## Results

**Dimensionality Reduction:**
- Input: 24 observations × 20 features
- Output: 24 observations × 3 components
- Variance retained: 89.45%
- Reduction: 85%

**Performance:**
- Execution time: 0.0069 seconds
- Memory saved: 85%
- Reconstruction R² Score: 0.8945

## Visualizations

Includes:
- Data distributions and correlation matrix
- Eigenvalue scree plots
- Variance plots (individual and cumulative)
- 2D and 3D PCA plots
- Biplots with feature loadings
- Component loadings charts
- Reconstruction quality and error analysis

## Requirements Met

- PCA from scratch (Task 1)
- Dynamic component selection based on 85% variance threshold (Task 2)
- Performance benchmarking (Task 3)
- African data (Sub-Saharan Africa)
- Dataset has missing values and non-numeric columns
- More than 10 columns (20 indicators)
- Manual standardization (no sklearn for PCA)
- All outputs visible

## Math Formulas

Standardization: z = (x - μ) / σ

Covariance: Cov(X) = (X^T × X) / (n - 1)

Eigendecomposition: Cov(X)v = λv

Explained Variance: EVR = λ_i / Σλ_j

Projection: Z = X × W_k

## Common Issues

**"File not found: prepared_data_for_pca.csv"**
Run data_exploration.ipynb first.

**Import errors**
```bash
pip install numpy pandas matplotlib seaborn jupyter
```

**Kernel crashes**
Restart kernel and run from beginning.

## References

- StatQuest: Principal Component Analysis
- World Bank World Development Indicators
- NumPy Documentation

