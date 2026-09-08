# StackHPpred

> **A stacking-based ensemble learning framework for the identification of peptide hormones using multi-view feature representations**

StackHPpred identifies hormone peptides from protein sequences using a two-layer
stacking ensemble. The framework combines sequence-derived physicochemical and
composition descriptors (CTDD, KSCTriad) with contextual embeddings from a
pretrained protein language model (ProtTrans ALBERT BFD). Probabilities from 30
first-layer models are combined by a meta-classifier, so that composition-based
and contextual representations contribute complementary information to the
final prediction.

## 🧬 Key Features
- **Multi-view Feature Extraction:**
  - **CTDD:** Composition, Transition, and Distribution of physicochemical properties.
  - **KSCTriad:** k-Spaced Conjoint Triads capturing local spatial sequence arrangements.
  - **PTAB (ProtTrans ALBERT BFD):** Contextual embeddings from an ALBERT-based
  transformer pretrained on the BFD protein corpus. Per-residue embeddings are
  mean-pooled across the sequence to yield a fixed-length 4096-dimensional
  vector per peptide.
- **Ensemble Architecture:** A two-layer stacking design. In the first layer,
  10 classifiers (AdaBoost, ANN, CatBoost, Extremely Randomized Trees, Gradient
  Boosting, LightGBM, LRT, Random Forest, SVM, XGBoost) are trained on each of
  the three feature representations, giving 30 base models. Their predicted
  probabilities form a 30-dimensional meta-feature vector, which is passed to a
  second-layer meta-classifier that produces the final prediction.

## ⚙️ Installation

We highly recommend using `conda` to create an isolated environment for running StackHPpred to avoid dependency conflicts.

1. Clone this repository (or download the source code):
   ```bash
   git clone https://github.com/yourusername/StackHPpred.git
   cd StackHPpred
   ```

2. Create a fresh Conda environment using Python 3.9.19:
   ```bash
   conda create -n StackHPpred python=3.9.19 -y
   ```

3. Activate the environment:
   ```bash
   conda activate StackHPpred
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The first time you run the tool, it will automatically download the ProtTrans ALBERT pre-trained weights from HuggingFace, which may take a few minutes.)*

## 🚀 Usage

StackHPpred is designed to be easily run from the command line.

### Command Line Interface

```bash
python predict.py -i <input_fasta> -o <output_csv>
```

**Arguments:**
* `-i` / `--input`: Path to your input `.fasta` file containing the peptide sequences you want to evaluate.
* `-o` / `--output`: Path where the resulting predictions will be saved as a `.csv` file.

### Example

We have provided a testing dataset in the `Dataset/` directory. You can run a test prediction using the following command:

```bash
python predict.py -i Dataset/Testing.fasta -o results.csv
```

The tool will process the features, generate base probabilities, and save the final predictions to `results.csv`.

## 📁 Repository Structure
* `predict.py` - The main executable script for predictions.
* `Dataset/` - Contains the `Training.fasta` and `Testing.fasta` datasets used in this study.
* `models/` - Contains the pre-trained `.pkl` files for the 30 baseline models, data scalers, and the final meta-model.
* `requirements.txt` - All necessary Python packages required to run the pipeline.

## 📝 Output Format
The output `.csv` file will contain three columns:
1. `Sequence_Name`: The identifier from your FASTA header.
2. `Probability`: The computed probability score (0.0 to 1.0).
3. `Prediction`: The binary classification (`HP` for Hormone Peptide, `non-HP` for non-Hormone Peptide).