import os
from utils.utils_data import reset_indexes, z_score_orderbook, normalize_messages, labeling
import pandas as pd
import numpy as np
import torch
import constants as cst
from torch.utils import data


def lobster_load(path, all_features, len_smooth, h, seq_size):
    set = np.load(path)
    if h == 10:
        tmp = 5
    if h == 20:
        tmp = 4
    elif h == 50:
        tmp = 3
    elif h == 100:
        tmp = 2
    elif h == 200:
        tmp = 1
    labels = set[seq_size-len_smooth:, -tmp]
    labels = labels[np.isfinite(labels)]
    labels = torch.from_numpy(labels).long()
    if all_features:
        input = set[:, cst.LEN_ORDER:cst.LEN_ORDER + 40]
        orders = set[:, :cst.LEN_ORDER]
        input = torch.from_numpy(input).float()
        orders = torch.from_numpy(orders).float()
        input = torch.cat((input, orders), dim=1)
    else:
        input = set[:, cst.LEN_ORDER:cst.LEN_ORDER + 40]
        input = torch.from_numpy(input).float()

    return input, labels


class LOBSTERDataBuilder:
    def __init__(
        self,
        stocks,
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
        self.stocks = stocks
        self.split_rates = split_rates
        
        self.sampling_type = sampling_type
        self.sampling_time = sampling_time
        self.sampling_quantity = sampling_quantity


    def prepare_save_datasets(self):
        for i in range(len(self.stocks)):
            stock = self.stocks[i]
            path = "{}/{}/{}_{}_{}".format(
                self.data_dir,
                stock,
                stock,
                self.date_trading_days[0],
                self.date_trading_days[1],
            )
            self.dataframes = []
            self._prepare_dataframes(path, stock)

            path_where_to_save = "{}/{}".format(
                self.data_dir,
                stock,
            )

            self.train_input = pd.concat(self.dataframes[0], axis=1).values
            self.val_input = pd.concat(self.dataframes[1], axis=1).values
            self.test_input = pd.concat(self.dataframes[2], axis=1).values
            self.train_set = pd.concat([pd.DataFrame(self.train_input), pd.DataFrame(self.train_labels_horizons)], axis=1).values
            self.val_set = pd.concat([pd.DataFrame(self.val_input), pd.DataFrame(self.val_labels_horizons)], axis=1).values
            self.test_set = pd.concat([pd.DataFrame(self.test_input), pd.DataFrame(self.test_labels_horizons)], axis=1).values
            self._save(path_where_to_save)


    def _prepare_dataframes(self, path, stock):
        COLUMNS_NAMES = {"orderbook": ["sell1", "vsell1", "buy1", "vbuy1",
                                       "sell2", "vsell2", "buy2", "vbuy2",
                                       "sell3", "vsell3", "buy3", "vbuy3",
                                       "sell4", "vsell4", "buy4", "vbuy4",
                                       "sell5", "vsell5", "buy5", "vbuy5",
                                       "sell6", "vsell6", "buy6", "vbuy6",
                                       "sell7", "vsell7", "buy7", "vbuy7",
                                       "sell8", "vsell8", "buy8", "vbuy8",
                                       "sell9", "vsell9", "buy9", "vbuy9",
                                       "sell10", "vsell10", "buy10", "vbuy10"],
                         "message": ["time", "event_type", "order_id", "size", "price", "direction"]}
        self.num_trading_days = len(os.listdir(path))//2
        split_days = self._split_days()
        split_days = [i * 2 for i in split_days]
        self._create_dataframes_splitted(path, split_days, COLUMNS_NAMES)
        # divide all the price, both of lob and messages, by 10000, to have dollars as unit
        for i in range(len(self.dataframes)):
            self.dataframes[i][0]["price"] = self.dataframes[i][0]["price"] / 10000
            self.dataframes[i][1].loc[:, ::2] /= 10000
        train_input = self.dataframes[0][1].values
        val_input = self.dataframes[1][1].values
        test_input = self.dataframes[2][1].values
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


    def _sparse_representation(self):
        tick_size = 0.01
        for i in range(len(self.dataframes)):
            dense_repr = self.dataframes[i][1].values
            sparse_repr = np.zeros((dense_repr.shape[0], dense_repr.shape[1] + 1))
            for row in range(dense_repr.shape[0]):
                sparse_pos_ask = 0
                sparse_pos_bid = 0
                mid_price = (dense_repr[row][0] + dense_repr[row][2]) / 2
                sparse_repr[row][-1] = mid_price
                for col in range(0, dense_repr.shape[1], 2):
                    if col == 0:
                        start_ask = dense_repr[row][col]
                    elif col == 2:
                        start_bid = dense_repr[row][col]
                    elif col % 4 == 0:
                        if sparse_pos_ask < (sparse_repr.shape[1]) - 1 / 2:
                            actual_ask = dense_repr[row][col]
                            for level in range(0, actual_ask-start_ask, -tick_size):
                                if sparse_pos_ask < (sparse_repr.shape[1]) - 1 / 2:
                                    if level == actual_ask - start_ask - tick_size:
                                        sparse_repr[row][sparse_pos_ask] = dense_repr[row][col+1]
                                    else:
                                        sparse_repr[row][sparse_pos_ask] = 0
                                    sparse_pos_ask += 1
                                else:
                                    break
                            start_ask = actual_ask
                        else:
                            continue
                    elif col % 4 == 2:
                        if sparse_pos_bid < (sparse_repr.shape[1]) - 1 / 2:
                            actual_bid = dense_repr[row][col]
                            for level in range(0, start_bid-actual_bid, -tick_size):
                                if sparse_pos_bid < (sparse_repr.shape[1]) - 1 / 2:
                                    if level == start_bid - actual_bid - tick_size:
                                        sparse_repr[row][sparse_pos_ask] = dense_repr[row][col+1]
                                    else:
                                        sparse_repr[row][sparse_pos_ask] = 0
                                    sparse_pos_bid += 1
                                else:
                                    break
                            start_bid = actual_bid
                        else:
                            continue
                

    def _create_dataframes_splitted(self, path, split_days, COLUMNS_NAMES):
        # iterate over files in the data directory of self.STOCK_NAME
        total_shape = 0
        for i, filename in enumerate(sorted(os.listdir(path))):
            f = os.path.join(path, filename)
            print(f)
            if os.path.isfile(f):
                # then we create the df for the training set
                if i < split_days[0]:
                    if (i % 2) == 0:
                        if i == 0:
                            train_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                        else:
                            train_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])

                    else:
                        if i == 1:
                            train_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            total_shape += train_orderbooks.shape[0]
                            train_orderbooks, train_messages = self._preprocess_message_orderbook([train_messages, train_orderbooks], self.n_lob_levels, self.sampling_type, self.sampling_time, self.sampling_quantity)
                            if (len(train_orderbooks) != len(train_messages)):
                                raise ValueError("train_orderbook length is different than train_messages")
                        else:
                            train_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            total_shape += train_orderbook.shape[0]
                            train_orderbook, train_message = self._preprocess_message_orderbook([train_message, train_orderbook], self.n_lob_levels, self.sampling_type, self.sampling_time, self.sampling_quantity)
                            train_messages = pd.concat([train_messages, train_message], axis=0)
                            train_orderbooks = pd.concat([train_orderbooks, train_orderbook], axis=0)

                elif split_days[0] <= i < split_days[1]:  # then we are creating the df for the validation set
                    if (i % 2) == 0:
                        if (i == split_days[0]):
                            self.dataframes.append([train_messages, train_orderbooks])
                            val_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                        else:
                            val_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                    else:
                        if i == split_days[0] + 1:
                            val_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            total_shape += val_orderbooks.shape[0]
                            val_orderbooks, val_messages = self._preprocess_message_orderbook([val_messages, val_orderbooks], self.n_lob_levels, self.sampling_type, self.sampling_time, self.sampling_quantity)
                            if (len(val_orderbooks) != len(val_messages)):
                                raise ValueError("val_orderbook length is different than val_messages")
                        else:
                            val_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            total_shape += val_orderbook.shape[0]
                            val_orderbook, val_message = self._preprocess_message_orderbook([val_message, val_orderbook], self.n_lob_levels, self.sampling_type, self.sampling_time, self.sampling_quantity)
                            val_messages = pd.concat([val_messages, val_message], axis=0)
                            val_orderbooks = pd.concat([val_orderbooks, val_orderbook], axis=0)

                else:  # then we are creating the df for the test set

                    if (i % 2) == 0:
                        if (i == split_days[1]):
                            self.dataframes.append([val_messages, val_orderbooks])
                            test_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                        else:
                            test_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])

                    else:
                        if i == split_days[1] + 1:
                            test_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            test_orderbooks, test_messages = self._preprocess_message_orderbook([test_messages, test_orderbooks], self.n_lob_levels, self.sampling_type, self.sampling_time, self.sampling_quantity)
                            if (len(test_orderbooks) != len(test_messages)):
                                raise ValueError("test_orderbook length is different than test_messages")
                        else:
                            test_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            test_orderbook, test_message = self._preprocess_message_orderbook([test_message, test_orderbook], self.n_lob_levels, self.sampling_type, self.sampling_time, self.sampling_quantity)
                            test_messages = pd.concat([test_messages, test_message], axis=0)
                            test_orderbooks = pd.concat([test_orderbooks, test_orderbook], axis=0)
            else:
                raise ValueError("File {} is not a file".format(f))
        self.dataframes.append([test_messages, test_orderbooks])
        print(f"Total shape of the orderbooks is {total_shape}")


    def _normalize_dataframes(self):
        #apply z score to orderbooks
        for i in range(len(self.dataframes)):
            if (i == 0):
                self.dataframes[i][1], mean_size, mean_prices, std_size, std_prices = z_score_orderbook(self.dataframes[i][1])
            else:
                self.dataframes[i][1], _, _, _, _ = z_score_orderbook(self.dataframes[i][1], mean_size, mean_prices, std_size, std_prices)

        #apply z-score to size and prices of messages with the statistics of the train set
        for i in range(len(self.dataframes)):
            if (i == 0):
                self.dataframes[i][0], mean_size, mean_prices, std_size, std_prices, mean_time, std_time, mean_depth, std_depth = normalize_messages(self.dataframes[i][0])
            else:
                self.dataframes[i][0], _, _, _, _, _, _, _, _ = normalize_messages(self.dataframes[i][0], mean_size, mean_prices, std_size, std_prices, mean_time, std_time, mean_depth, std_depth)

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
    
    
    def _sampling_quantity(self, dataframes, quantity):
        messages_df, orderbook_df = dataframes
        
        # Calculate cumulative sum and create boolean mask
        cumsum = messages_df['size'].cumsum()
        sample_mask = (cumsum % quantity < messages_df['size'])
        
        # Get indices where we need to sample
        sampled_indices = messages_df.index[sample_mask].tolist()
        
        # Update both dataframes efficiently using loc
        messages_df = messages_df.loc[sampled_indices].reset_index(drop=True)
        orderbook_df = orderbook_df.loc[sampled_indices].reset_index(drop=True)
        
        return [messages_df, orderbook_df]


    def _sampling_time(self, dataframes, time):
        # Convert the time column to datetime format if it's not already
        dataframes[0]['time'] = pd.to_datetime(dataframes[0]['time'], unit='s')

        # Resample the messages dataframe to get data at every second
        resampled_messages = dataframes[0].set_index('time').resample(time).first().dropna().reset_index()

        # Resample the orderbook dataframe to get data at every second
        resampled_orderbook = dataframes[1].set_index(dataframes[0]['time']).resample(time).first().dropna().reset_index(drop=True)

        # Update the dataframes with the resampled data
        dataframes[0] = resampled_messages
        
        # Transform the time column to seconds
        dataframes[0]['time'] = dataframes[0]['time'].dt.second + dataframes[0]['time'].dt.minute * 60 + dataframes[0]['time'].dt.hour * 3600 + dataframes[0]['time'].dt.microsecond / 1e6
        dataframes[1] = resampled_orderbook

        return dataframes
    
    def _preprocess_message_orderbook(self, dataframes, n_lob_levels, sampling_type, time=None, quantity=None):
        dataframes = reset_indexes(dataframes)
        # take only the first n_lob_levels levels of the orderbook and drop the others
        dataframes[1] = dataframes[1].iloc[:, :n_lob_levels * cst.LEN_LEVEL]

        # take the indexes of the dataframes that are of type 
        # 2 (partial deletion), 5 (execution of a hidden limit order), 
        # 6 (cross trade), 7 (trading halt) and drop it
        indexes_to_drop = dataframes[0][dataframes[0]["event_type"].isin([2, 5, 6, 7])].index
        dataframes[0] = dataframes[0].drop(indexes_to_drop)
        dataframes[1] = dataframes[1].drop(indexes_to_drop)

        dataframes = reset_indexes(dataframes)

        # sample the dataframes according to the sampling type
        if sampling_type == "time":
            dataframes = self._sampling_time(dataframes, time)
        elif sampling_type == "quantity":
            dataframes = self._sampling_quantity(dataframes, quantity)
            
        dataframes = reset_indexes(dataframes)
        
        # drop index column in messages
        dataframes[0] = dataframes[0].drop(columns=["order_id"])

        # do the difference of time row per row in messages and subsitute the values with the differences
        # Store the initial value of the "time" column
        first_time = dataframes[0]["time"].values[0]
        # Calculate the difference using diff
        dataframes[0]["time"] = dataframes[0]["time"].diff()
        # Set the first value directly
        dataframes[0].iat[0, dataframes[0].columns.get_loc("time")] = first_time - 34200
            
        # add depth column to messages
        dataframes[0]["depth"] = 0

        # we compute the depth of the orders with respect to the orderbook
        # Extract necessary columns
        prices = dataframes[0]["price"].values
        directions = dataframes[0]["direction"].values
        event_types = dataframes[0]["event_type"].values
        bid_sides = dataframes[1].iloc[:, 2::4].values
        ask_sides = dataframes[1].iloc[:, 0::4].values
        
        # Initialize depth array
        depths = np.zeros(dataframes[0].shape[0], dtype=int)

        # Compute the depth of the orders with respect to the orderbook
        for j in range(1, len(prices)):
            order_price = prices[j]
            direction = directions[j]
            event_type = event_types[j]
            
            index = j if event_type == 1 else j - 1
            
            if direction == 1:
                bid_price = bid_sides[index, 0]
                depth = (bid_price - order_price) // 100
            else:
                ask_price = ask_sides[index, 0]
                depth = (order_price - ask_price) // 100
            
            depths[j] = max(depth, 0)
        
        # Assign the computed depths back to the DataFrame
        dataframes[0]["depth"] = depths
            
        # we eliminate the first row of every dataframe because we can't deduce the depth
        dataframes[0] = dataframes[0].iloc[1:, :]
        dataframes[1] = dataframes[1].iloc[1:, :]
        dataframes = reset_indexes(dataframes)
        
        dataframes[0]["direction"] = dataframes[0]["direction"] * dataframes[0]["event_type"].apply(
            lambda x: -1 if x == 4 else 1)
            
        return dataframes[1], dataframes[0]