# TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data
This is the official repository for the paper TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data.

## Abstract
Price Trend Prediction (PTP) based on Limit Order Book (LOB) data is a fundamental challenge in financial markets. Despite advances in deep learning, existing models fail to generalize across different market conditions and assets. Surprisingly, by adapting a simple MLP-based architecture to LOB, we show that we surpass SoTA performance; thus, challenging the necessity of complex architectures. Unlike past work that shows robustness issues, we propose TLOB, a transformer-based model that uses a dual attention mechanism to capture spatial and temporal dependencies in LOB data. This allows it to adaptively focus on the market microstructure, making it particularly effective for longer-horizon predictions and volatile market conditions.
We also introduce a new labeling method that improves on previous ones, removing the horizon bias.
We evaluate TLOB's effectiveness across four horizons, using the established FI-2010 benchmark, which exceeds the state-of-the-art by an average of 3.7 F1-score. Additionally, TLOB shows average improvements on Tesla and Intel with a 1.3 and 7.7 increase in F1-score, respectively. Finally, we tested TLOB on a recent Bitcoin dataset, and TLOB outperforms the SoTA performance by an average of 1.1 in F1-score.
Additionally, we empirically show how stock price predictability has declined over time, -6.68 in F1-score, highlighting the growing market efficiency. 
Predictability must be considered in relation to transaction costs, so we experimented with defining trends using an average spread, reflecting the primary transaction cost. The resulting performance deterioration underscores the complexity of translating trend classification into profitable trading strategies.
We argue that our work provides new insights into the evolving landscape of stock price trend prediction and sets a strong foundation for future advancements in financial AI. We commit to releasing the code publicly. 

# Getting Started 
These instructions will get you a copy of the project up and running on your local machine for development and reproducibility purposes.

## Prerequisities
This project requires Windows, Python, and pip. If you don't have them installed, please do so first. It is possible to do it using conda or/and Linux, but in that case, you are on your own.

## Installing
To set up the environment for this project, follow these steps:

1. Clone the repository:
```sh
git clone https://github.com/LeonardoBerti00/TLOB.git
```
2. Navigate to the project directory
3. Create a virtual environment:
```sh
python -m venv env
```
4. Activate the new pip environment:
```sh
env\Scripts\activate
```
5. Download the necessary packages:
```sh
pip install -r requirements.txt
```

# Training
If your objective is to train a TLOB or MLPLOB model or implement your model, you should follow those steps.

## Data 
If you have some LOBSTER data, you can follow those steps:
1. The format of the data should be the same of LOBSTER: f"{year}-{month}-{day}_34200000_57600000_{type}" and the data should be saved in f"data/{stock_name}/{stock_name}_{year}-{start_month}-{start_day}_{year}-{end_month}-{end_day}". Type can be or message, or orderbook.
2. Inside the config file, you need to set the names of the training stock and the testing stocks, and also the dataset to LOBSTER. Currently, you can add only one for the training, but several for testing. 
3. You need to do the pre-processing step. To do so, set config.is_data_preprocessed to False.

Otherwise, you can train and test with the BTC and FI-2010 datasets that will be automatically downloaded from Kaggle or unzipped, respectively. You need to set config.is_data_preprocessed to False.

## Training a TLOB, MLPLOB, DeepLOB, or BiNCTABL Model 
To train a TLOB, MLPLOB, DeepLOB, or BiNCTABL Model, you need to set the type variable in the config file to TRAINING, then run this command:
```sh
python main.py +model={model_name} +dataset={dataset_name} hydra.job.chdir=False
```
A checkpoint will be saved in data/checkpoints/. You can see all the models and dataset names in the config file. 

## Implementing and Training a New Model 
To implement a new model, follow these steps:
1. Implement your model class in the models/ directory. Your model class will take in an input of dimension [batch_size, seq_len, num_features], and should output a tensor of dimension [batch_size, 3].
2. Add your model to pick_model in utils_models.
3. Update the config file to include your model and its hyperparameters. If you are using the FI-2010 dataset, it is suggested to set the hidden dim to 40 and the hp all_features to false if you want to use only the LOB as input, or if you want to use the LOB and market features, the hidden dim should be 144 and all features true. If you are using LOBSTER data, it is suggested to set the hidden dim to 46 and all features to true to use LOB and orders, while if you want to use only the LOB, set all features to False. 
4. Add your model with cs.store, similar to the other models
5. Run the training script:
```sh
python main.py +model={your_model_name} +dataset={dataset_name} hydra.job.chdir=False
```
6. You can set whatever configuration using the Hydra style of prompt.
7. A checkpoint will be saved in data/checkpoints/ 
Optionally, you can also log the run with wandb or run a sweep, changing the config experiment options.

# Citation
```sh
@article{berti2025tlob,
  title={TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data},
  author={Berti, Leonardo and Kasneci, Gjergji},
  journal={arXiv preprint arXiv:2502.15757},
  year={2025}
}
```
# tranformers-in-time-series-for-price-trend-prediction
