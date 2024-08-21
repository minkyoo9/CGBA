# CGBA

Implementation of [Claim-Guided Textual Backdoor Attack for Practical Applications](https://openreview.net/pdf/ab0aa5302ab0ebc22cebd8e1961228620fea77e8.pdf), *ARR preprint*

## Requirements

1. **Install Anaconda**: Download and install Anaconda from [here](https://www.anaconda.com/download).
2. **Create and Activate Python Environment**:
    ```bash
    conda env create -f environments.yml -n cgba
    conda activate cgba
    
## Evaluation
### Model Preparation
1. **Download Contrastive Models**:
   - Download the models from [this link](https://github.com/PaperCGBA/CGBA/releases/download/models/Contrastive_FakeNews.tar.gz).
   - Decompress the file and move the models:
     ```bash
     tar -zxvf Contrastive_FakeNews.tar.gz
     mv Models_dist_scale_margin0.2_alpha0.1 Contrastive_Learning/Models/
     ```

2. **Download Backdoored Models**:
   - Download the models from [this link](https://github.com/PaperCGBA/CGBA/releases/download/models/BestModels_FakeNews.tar.gz).
   - Decompress the file and move the models:
     ```bash
     tar -zxvf BestModels_FakeNews.tar.gz
     mv BestModels_dist_alpha0.1_scale0.2_aug10 Model_Training/BestModels/
     ```

### Run
```bash
cd Model_Training
python COVID_Make_poisonedBERT_Eval.py -c 5 # Evaluate attack result for cluster ID: 5
```

## Train
### Dataset Preparation / We already provide datasets for FakeNews (COVID)
  - Sequentially run the following scripts with the appropriate paths to extract claims from the dataset:
    ```bash
    python Claim_Extraction/Extract_NEs.py
    python Claim_Extraction/Extract_Questions.py
    python Claim_Extraction/Extract_Claims.py
    ```

  - Extract embeddings and conduct clustering with the appropriate paths:
    ```bash
    python Embedding_Extraction/Embedding_Extraction.py
    ```

### Contrastive Learning / We already provide a trained contrastive model (see the release section)
   - Train the contrastive model:
     ```bash
     python Contrastive_Learning/ContrastiveLearning.py
     ```

### Backdoor Training
   - Conduct backdoor training:
     ```bash
     cd Model_Training
     chmod +x run_clusters.sh
     ./run_clusters.sh # This process takes several hours and requires approximately 23 GiB of storage for the models.
     ```


### Evaluation
  - Evaluate attack performance acorss entire clusters:
     ```bash
     python Analyze_results.py
     ```


