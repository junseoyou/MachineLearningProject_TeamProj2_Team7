# CS 53744 - Machine Learning Project 2
## Team 7: Predicting Human Preferences for LLM Response Enhancement

This repository contains the code for the "LLM Classification Finetuning" Kaggle competition.

Our final solution is an ensemble of three main candidates:
1.  **Candidate 1:** `DeBERTa-v3-base` (fine-tuned with LoRA)
2.  **Candidate 2:** `e5-base-v2` Embeddings + `XGBoost`/`LightGBM`
3.  **Candidate 3:** `e5-base-v2` Embeddings + Strong Lexical Features + `MLP`

This is supplemented by calibration (`IsotonicRegression`) for each candidate before final ensembling.

### Repository Structure

-   `/notebooks/01_local_training.ipynb`: (Local) Run this notebook on a local GPU machine to train all candidate models and save the model artifacts (LoRA adapters, `.pkl` files, scalers).
-   `/notebooks/02_kaggle_inference.ipynb`: (Kaggle) Upload this notebook to Kaggle. It loads the pre-trained artifacts from the local step, performs inference on `test.csv`, runs calibration, and creates the final ensemble submission.
-   `/notebooks/**03_validation**.ipynb`: This notebook serves as our complete development log, documenting all intermediate experiments, model validation, and the iterative "trial and error" process that informed our final model architecture.
-   `/Assignment2_7_20.._Sooho_Moon.pdf`: The final PDF report.
-   `/requirements.txt`: Python dependencies required to run the code.

### How to Reproduce Results

This project uses a "Local Train / Kaggle Infer" strategy to satisfy the "no-internet constraint".

**Step 1: Local Setup (Model Training)**

1.  **Clone Repository & Install:**
    ```bash
    git clone [Your_Repo_URL]
    cd Team7_MLP_Project2
    pip install -r requirements.txt
    ```
2.  **Download Data:** Download `train.csv`, `test.csv`, and `sample_submission.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/llm-classification-finetuning/data) and place them in the root directory.
3.  **Download Base Models:**
    Run the `Step 2. [Model Download]` cell in `01_local_training.ipynb` to download the `sentence-transformers` models (like `e5-base-v2`) into a local `./models/` directory. `DeBERTa-v3-base` will be downloaded automatically by the `transformers` library during training.
4.  **Run Training Notebook:**
    Open `notebooks/01_local_training.ipynb` and execute all cells. This will:
    * Train Candidate 1 (DeBERTa) and save the LoRA adapter to `./models/lora_adapter_deberta-v3-base/`.
    * Train Candidate 2 (XGBoost/LGBM) and save the model to `./models/candidate_2_...pkl`.
    * Train Candidate 3 (MLP) and save the model and scaler to `./models/candidate_3_...pkl`.

**Step 2: Kaggle Setup (Inference & Submission)**

1.  **Upload Trained Artifacts:**
    * Create a **new Kaggle Dataset** (e.g., `team7-final-models`).
    * Upload all the generated model files from your local `./models/` folder (the LoRA adapter folder, all `.pkl` files, etc.) to this dataset.
2.  **Upload Base Models (for No-Internet):**
    * To comply with the "no-internet" rule, you must also upload the *base models*.
    * Upload `microsoft/deberta-v3-base` to Kaggle Models (like you did for `e5-small-v2`).
    * Upload `e5-base-v2` (from your local `./models/e5-base-v2` folder) to Kaggle Models or as another Kaggle Dataset.
3.  **Run Inference Notebook:**
    * Create a new notebook in the Kaggle competition.
    * Set **Settings -> Internet -> OFF**.
    * **"Add Input"** and add the following:
        1.  `llm-classification-finetuning` (Competition data)
        2.  `team7-final-models` (Your *trained* models from Step 2.1)
        3.  `deberta-v3-base` (The *base* model from Step 2.2)
        4.  `e5-base-v2` (The *base* model from Step 2.2)
    * Copy the code from `notebooks/02_kaggle_inference.ipynb` into your Kaggle notebook.
    * In the first cell of `02_kaggle_inference.ipynb`, **update the path variables** (e.g., `PATH_TRAINED_MODELS`, `PATH_BASE_MODELS`) to match your Kaggle input paths.
    * Run all cells. The notebook will load your trained models, perform calibration, run the final ensemble, and save `submission.csv`.