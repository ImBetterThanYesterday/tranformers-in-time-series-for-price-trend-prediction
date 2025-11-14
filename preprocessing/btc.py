import os
from utils.utils_data import z_score_orderbook, labeling, reset_indexes
import pandas as pd
import numpy as np
import torch
import constants as cst
from constants import SamplingType
import kagglehub


def btc_load(path, len_smooth, h, seq_size):
    set = np.load(path)
    if h == 10:
        tmp = 4
    elif h == 20:
        tmp = 3
    elif h == 50:
        tmp = 2
    elif h == 100:
        tmp = 1
    labels = set[seq_size-len_smooth:, -tmp]
    labels = labels[np.isfinite(labels)]
    labels = torch.from_numpy(labels).long()
    input = torch.from_numpy(set[:, :cst.N_LOB_LEVELS*4]).float()
    return input, labels


class BTCDataBuilder:
    def __init__(
        self,
        data_dir,
        date_trading_days,
        split_rates,
        sampling_type,
        sampling_time,
        sampling_quantity,
    ):
        self.n_lob_levels = cst.N_LOB_LEVELS
        self.data_dir = data_dir
        self.date_trading_days = date_trading_days
        self.split_rates = split_rates
        
        self.sampling_type = sampling_type
        self.sampling_time = sampling_time
        self.sampling_quantity = sampling_quantity


    def prepare_save_datasets(self):
        
        # Create directory if it doesn't exist
        # Continue with the existing code
        save_dir = "{}/{}/{}_{}_{}".format(
            self.data_dir,
            "BTC",
            "BTC",
            self.date_trading_days[0],
            self.date_trading_days[1],
        )
        os.makedirs(save_dir, exist_ok=True)
        # check if the directory is empty
        if len(os.listdir(save_dir)) == 0:  
            print("Downloading BTC dataset from Kaggle...")
            # Download the dataset from Kaggle
            path = kagglehub.dataset_download("siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data")

            # Get all CSV files in the downloaded directory
            file = os.listdir(path)[0]
            file_path = os.path.join(path, file)
            print(f"Processing {file}...")
            
            # Load the CSV file
            df = pd.read_csv(filepath_or_buffer=file_path, index_col='Unnamed: 0', parse_dates=True)
            df.columns = np.arange(42)
            
            # Select specific columns for the order book and 
            # order in such a way that we have sell, vsell, buy, vbuy
            df = df.loc[:,[1, 22,23, 2,3, 24,25, 4,5, 26,27, 6,7, 28,29, 8,9, 30,31, 10,11, 32,33, 12,13, 34,35, 14,15, 36,37, 16,17, 38,39, 18,19, 40,41, 20,21]]        
            # Rename the columns for better readability
            df.columns = ["timestamp", 
                        "sell1", "vsell1", "buy1", "vbuy1",
                        "sell2", "vsell2", "buy2", "vbuy2",
                        "sell3", "vsell3", "buy3", "vbuy3",
                        "sell4", "vsell4", "buy4", "vbuy4",
                        "sell5", "vsell5", "buy5", "vbuy5",
                        "sell6", "vsell6", "buy6", "vbuy6",
                        "sell7", "vsell7", "buy7", "vbuy7",
                        "sell8", "vsell8", "buy8", "vbuy8",
                        "sell9", "vsell9", "buy9", "vbuy9",
                        "sell10", "vsell10", "buy10", "vbuy10",
                        ]

            #transform string into timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')  # Let pandas infer format

            print("Splitting data by day and saving CSV files...")
            unique_dates = df["timestamp"].apply(lambda x: x.date()).unique()

            for date in unique_dates:
                # Convert date to string format YYYY-MM-DD
                date_str = date.strftime('%Y-%m-%d')
                # Filter data for the current date
                day_data = df[df["timestamp"].apply(lambda x: x.date()) == date]
                #day_data = day_data.drop(columns=["timestamp"])
                
                # Create the filename in the specified format
                filename = f"BTC_{date_str}_34200000_57600000_orderbook_10.csv"
                file_path = os.path.join(save_dir, filename)
                
                # Save the data to CSV without header and index
                day_data.to_csv(file_path, index=False, header=False)
                print(f"Saved {filename} with {len(day_data)} records")
            
        self.dataframes = []
        self._prepare_dataframes(save_dir)

        path_where_to_save = "{}/{}".format(
            self.data_dir,
            "BTC",
        )
        train_input = self.dataframes[0].values
        val_input = self.dataframes[1].values
        test_input = self.dataframes[2].values
        self.train_set = np.concatenate([train_input, self.train_labels_horizons.values], axis=1)
        self.val_set = np.concatenate([val_input, self.val_labels_horizons.values], axis=1)
        self.test_set = np.concatenate([test_input, self.test_labels_horizons.values], axis=1)
        self._save(path_where_to_save)


    def _prepare_dataframes(self, path):
        COLUMNS_NAMES = {"orderbook": ["timestamp",
                                       "sell1", "vsell1", "buy1", "vbuy1",
                                       "sell2", "vsell2", "buy2", "vbuy2",
                                       "sell3", "vsell3", "buy3", "vbuy3",
                                       "sell4", "vsell4", "buy4", "vbuy4",
                                       "sell5", "vsell5", "buy5", "vbuy5",
                                       "sell6", "vsell6", "buy6", "vbuy6",
                                       "sell7", "vsell7", "buy7", "vbuy7",
                                       "sell8", "vsell8", "buy8", "vbuy8",
                                       "sell9", "vsell9", "buy9", "vbuy9",
                                       "sell10", "vsell10", "buy10", "vbuy10"]}
        self.num_trading_days = len(os.listdir(path))
        split_days = self._split_days()
        self._create_dataframes_splitted(path, split_days, COLUMNS_NAMES)
        
        train_input = self.dataframes[0].values
        val_input = self.dataframes[1].values
        test_input = self.dataframes[2].values
        #create a dataframe for the labels
        for i in range(len(cst.LOBSTER_HORIZONS)):
            if i == 0:
                train_labels = labeling(train_input, cst.LEN_SMOOTH, cst.LOBSTER_HORIZONS[i])
                val_labels = labeling(val_input, cst.LEN_SMOOTH, cst.LOBSTER_HORIZONS[i])
                test_labels = labeling(test_input, cst.LEN_SMOOTH, cst.LOBSTER_HORIZONS[i])
                train_labels = np.concatenate([train_labels, np.full(shape=(train_input.shape[0] - train_labels.shape[0]), fill_value=np.inf)])
                val_labels = np.concatenate([val_labels, np.full(shape=(val_input.shape[0] - val_labels.shape[0]), fill_value=np.inf)])
                test_labels = np.concatenate([test_labels, np.full(shape=(test_input.shape[0] - test_labels.shape[0]), fill_value=np.inf)])
                self.train_labels_horizons = pd.DataFrame(train_labels, columns=["label_h{}".format(cst.LOBSTER_HORIZONS[i])])
                self.val_labels_horizons = pd.DataFrame(val_labels, columns=["label_h{}".format(cst.LOBSTER_HORIZONS[i])])
                self.test_labels_horizons = pd.DataFrame(test_labels, columns=["label_h{}".format(cst.LOBSTER_HORIZONS[i])])
            else:
                train_labels = labeling(train_input, cst.LEN_SMOOTH, cst.LOBSTER_HORIZONS[i])
                val_labels = labeling(val_input, cst.LEN_SMOOTH, cst.LOBSTER_HORIZONS[i])
                test_labels = labeling(test_input, cst.LEN_SMOOTH, cst.LOBSTER_HORIZONS[i])
                train_labels = np.concatenate([train_labels, np.full(shape=(train_input.shape[0] - train_labels.shape[0]), fill_value=np.inf)])
                val_labels = np.concatenate([val_labels, np.full(shape=(val_input.shape[0] - val_labels.shape[0]), fill_value=np.inf)])
                test_labels = np.concatenate([test_labels, np.full(shape=(test_input.shape[0] - test_labels.shape[0]), fill_value=np.inf)])
                self.train_labels_horizons["label_h{}".format(cst.LOBSTER_HORIZONS[i])] = train_labels
                self.val_labels_horizons["label_h{}".format(cst.LOBSTER_HORIZONS[i])] = val_labels
                self.test_labels_horizons["label_h{}".format(cst.LOBSTER_HORIZONS[i])] = test_labels
        
        # to conclude the preprocessing we normalize the dataframes
        self._normalize_dataframes()


    def _create_dataframes_splitted(self, path, split_days, COLUMNS_NAMES):
        # Initialize empty dataframes for each split
        train_orderbooks = None
        val_orderbooks = None
        test_orderbooks = None
        for i, filename in enumerate(sorted(os.listdir(path))):
            f = os.path.join(path, filename)
            if os.path.isfile(f):
                df_ob = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                # sample the dataframes according to the sampling type
                if self.sampling_type == SamplingType.TIME:
                    df_ob = self._sampling_time(df_ob, self.sampling_time)
                if i < split_days[0]:
                    train_orderbooks = df_ob if train_orderbooks is None else pd.concat([train_orderbooks, df_ob], axis=0)
                elif split_days[0] <= i < split_days[1]:
                    val_orderbooks = df_ob if val_orderbooks is None else pd.concat([val_orderbooks, df_ob], axis=0)
                else:
                    test_orderbooks = df_ob if test_orderbooks is None else pd.concat([test_orderbooks, df_ob], axis=0)
            else:
                raise ValueError(f"File {f} is not a file")
        # Save the splitted dataframes
        train_orderbooks = train_orderbooks.drop(columns=["timestamp"])
        val_orderbooks = val_orderbooks.drop(columns=["timestamp"])
        test_orderbooks = test_orderbooks.drop(columns=["timestamp"])
        self.dataframes = [train_orderbooks, val_orderbooks, test_orderbooks]


    def _normalize_dataframes(self):
        #apply z score to orderbooks
        for i in range(len(self.dataframes)):
            if (i == 0):
                self.dataframes[i], mean_size, mean_prices, std_size, std_prices = z_score_orderbook(self.dataframes[i])
            else:
                self.dataframes[i], _, _, _, _ = z_score_orderbook(self.dataframes[i], mean_size, mean_prices, std_size, std_prices)


    def _save(self, path_where_to_save):
        np.save(path_where_to_save + "/train.npy", self.train_set)
        np.save(path_where_to_save + "/val.npy", self.val_set)
        np.save(path_where_to_save + "/test.npy", self.test_set)


    def _split_days(self):
        train = int(self.num_trading_days * self.split_rates[0])
        val = int(self.num_trading_days * self.split_rates[1]) + train
        test = int(self.num_trading_days * self.split_rates[2]) + val
        print(f"There are {train} days for training, {val - train} days for validation and {test - val} days for testing")
        return [train, val, test]


    def _sampling_time(self, dataframe, time):
        # Convert the time column to datetime format if it's not already
        dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], errors='coerce')
        # Resample the messages dataframe to get data at every second
        dataframe = dataframe.set_index('timestamp').resample(time).first().dropna().reset_index()
        return dataframe
