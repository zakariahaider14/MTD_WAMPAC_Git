import numpy as np
import pandas as pd
import logging
import datetime
import traceback
import gym
from   gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import random
import time
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models 
from ortools.linear_solver import pywraplp  # Add this import statement
import copy
from collections import deque
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import time
from typing import Union, List
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
from tensorflow.keras.optimizers import Adam  # Add this line
import os
import math

import sys
import datetime

os.makedirs('evaluation_logs', exist_ok=True)

# Create a log file with timestamp
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"training_log_{current_time}.txt"

# Create a custom logger class
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to both terminal and file
sys.stdout = Logger(log_file)



class DataLoader:
    """Handles all data loading and preprocessing"""
    def __init__(self, numOfBuses: int = 14, numOfLines: int = 20, freq='sec'):
        self.numOfBuses = numOfBuses
        self.numOfLines = numOfLines
        self.numOfZ = self.numOfBuses + 2 * self.numOfLines + 1  # This should be 55 based on your data
        self.file_name = f"Bus Data//IEEE_{numOfBuses}.xlsx"
        self.localize = True
        self.H_mat = None
        self.H_org = None
        # Common variables
        self.Z_org = None
        self.Z_dec = None
        self.Z_rec = None
        self.H_org = None
        self.W_list = None
        self.consideredDecoyIDs = []
        self.consideredFixedIDs = []
        self.attackerTypeList = []
        self.attackerLevelList = []
        self.impute_data_list = []  # Add this line
                # Use default values instead of get_copy
        self.Threshold_min = 1 # Default value
        self.Threshold_max = 5 

        self.Attack_Data = None
        self.freq = freq
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        # self.th = 5
        self.fixedIDDs = []
        self.runfilter = False
        self.runImpute = False
        self.Z_est_init = []
        self.Noisy_Indx_actu = []
        self.Zpair = []
        self.numOfAttacks = 0
        self.meanZ = 0
        self.bus_data_df = []
        self.line_data_df = []
        self.topo_mat = []
        self.line_data = []
        self.Z_msr_org = []
        self.bus_data = []
        self.attackerLevelList=[1, 2, 3]
        self.attacked_Bus = 5
        self.lookback = 10
        self.Z_mat = []
        self.th = 5
        self.current_seed = random.randint(0, 1000)
        self.attackertype = 1  # Initialize current_seed
        self.recoveryType = 1
        self.attackerLevel = 1
        # self.attackertype = 1
        # self.recoveryType = 1
        self.attackCat = 'FDI'
        self.addDecoy = False
        self.server = []
        self.clusters = []
        self.hubs = []
        self.IDbank = []
        self.load_data()
        self.fit_scaler()  # Add this line
    


    def fit_scaler(self):
        self.scaler.fit(self.dataset_org)
        self.dataset = self.scaler.transform(self.dataset_org)
        
    def get_initial_data(self):
        return self.dataset_org

    def load_system_data(self):
        """Load bus and line data"""
        # Your existing code for loading bus_data_df and line_data_df
        # Load bus and line data
        try:
            self.bus_data_df = pd.read_excel(self.file_name, sheet_name="Bus")
            self.line_data_df = pd.read_excel(self.file_name, sheet_name="Branch")
            
            # Update indices
            self.bus_data_df.set_index(pd.Series(range(1, self.numOfBuses+1)), inplace=True)
            self.line_data_df.set_index(pd.Series(range(1, self.numOfLines+1)), inplace=True)
            
            # Load topology matrix and create H_matrix
            self.H_org = self.load_topology_data()
            # print(f"H_org : {self.H_org}")

            self.H_mat = self.H_org.copy()
            # print(f"H_mat : {self.H_mat}")

            self.W_list = np.ones(self.numOfZ)  # Initialize with ones for all measurements
            # print(f"W_list : {self.W_list}")

            
            # Load measurement data
            self.load_measurement_data()
            # self.load_attack_data(self.attacked_Bus)
            
            
            return True
        except Exception as e:
            print(f"Error loading system data: {str(e)}")
            return False

    def load_topology_data(self):
        file_name = "Bus Data//IEEE_14.xlsx"  # Update the path to match the uploaded file location

        try:
            # Attempt to load the topology matrix and line data
            print(f"Loading topology data from: {file_name}")
            self.topo_mat = pd.read_excel(file_name, sheet_name="Topology Matrix")
            self.line_data = pd.read_excel(file_name, sheet_name="Line Data")
            
            # print(f"Loaded topology matrix with shape: {self.topo_mat.shape}")
            # print(f"Loaded line data with shape: {self.line_data.shape}")

        except Exception as e:
            # If loading fails, initialize with zeros
            # print(f"Warning: Could not load topology data from {file_name}. Error: {str(e)}. Initializing with zeros.")
            self.topo_mat = pd.DataFrame(np.zeros((self.numOfLines, self.numOfBuses)))
            self.line_data = pd.DataFrame(np.zeros((self.numOfLines, 4)))

        self.Topo = self.line_data.values.astype(int)

        # Verify if topology matrix was successfully loaded
        if self.topo_mat is None or self.topo_mat.empty:
            raise ValueError("Topology matrix is not initialized. Please ensure the Excel file and sheet names are correct.")

        # Create H_matrix based on system topology
        H_org = self.topo_mat.values.copy()
        self.H_mat = H_org
        return H_org



    def load_measurement_data(self):
        """Load measurement data from file"""
        try:
            measurement_data = pd.read_excel(self.file_name, sheet_name="Measurement Data")
            self.Z_msr_org = pd.DataFrame(measurement_data)
            
            # Add ID and Reported columns if they don't exist
            if 'ID' not in self.Z_msr_org.columns:
                self.Z_msr_org.insert(0, 'ID', range(len(self.Z_msr_org)))
            if 'Reported' not in self.Z_msr_org.columns:
                self.Z_msr_org.insert(1, 'Reported', 1)
                
            print(f"Loaded measurement data shape: {self.Z_msr_org.shape}")
            
        except Exception as e:
            print(f"Error loading measurement data: {str(e)}")
            # Create empty DataFrame with required columns
            self.Z_msr_org = pd.DataFrame(columns=['ID', 'Reported', 'Data'])
        

        
        # Use the actual length of Z_msr_org instead of self.numOfZ+1
        actual_length = len(self.Z_msr_org)

        
        self.Z_org = self.Z_msr_org.values.copy()
        self.Z_mat = self.Z_msr_org.values.copy()
        print(f"Z_mat shape: {self.Z_mat.shape}")
        # print(f"Z_mat: {self.Z_mat}")

        # Update numOfZ if it doesn't match the actual data
        if actual_length != self.numOfZ:
            print(f"Warning: numOfZ ({self.numOfZ}) doesn't match the actual data length ({actual_length - 1}). Updating numOfZ.")
            self.numOfZ = actual_length

    def load_attack_data(self, attacked_Bus: int):
        """
        Load attack data for a specific bus
        Args:
            attacked_Bus: The bus under attack
        """
        try:
            file_Name_ = f"Attack_Space_{self.numOfBuses}_{self.numOfLines}_{attacked_Bus}.csv"
            print(f"Loading attack data from: Attack Data//{file_Name_}")
            
            # Print original data shape
            attack_data = pd.read_csv("Attack Data//"+file_Name_)
            print(f"Original attack data shape: {attack_data.shape}")  # Should be (7095, 3)
            
            # Reshape into chunks of 55 measurements
            num_chunks = len(attack_data) // 55
            reshaped_data = []
            
            for i in range(num_chunks):
                chunk = attack_data.iloc[i*55:(i+1)*55].values
                if len(chunk) == 55:  # Only add complete chunks
                    reshaped_data.append(chunk)
            
            # Convert to numpy array
            self.Attack_Data = np.array(reshaped_data)
            print(f"Reshaped Attack_Data shape: {self.Attack_Data.shape}")  # Should be (128, 55, 3)
            
            # Verify the reshaping
            print(f"Number of complete chunks: {len(reshaped_data)}")
            print(f"Each chunk shape: {reshaped_data[0].shape if reshaped_data else 'No chunks'}")
            
            # Add noise if localization is enabled
            if self.localize:
                self._add_noise_to_attack_data()
            
            self.numOfAttacks = num_chunks
            
            # Calculate meanZ from dataset_org instead of Z_msr_org
            if hasattr(self, 'dataset_org') and self.dataset_org is not None:
                self.meanZ = np.abs(self.dataset_org).mean()
            else:
                print("Warning: dataset_org not initialized. Setting meanZ to 0.")
                self.meanZ = 0
                
            if self.Attack_Data is None:
                print("Warning: Attack_Data is not loaded properly.")
            else:
                print(f"Attack_Data shape: {self.Attack_Data.shape}")
            
            self.verify_attack_data_shape()
                
            return self.Attack_Data
            
        except Exception as e:
            print(f"Error loading attack data: {str(e)}")
            traceback.print_exc()
            return None

    def _add_noise_to_attack_data(self):
        """Add noise to attack data for localization"""
        Noise_Data = self.Attack_Data.copy()
        # Generate noise matching the shape of the third "column"
        noise = np.random.randint(-20, 20, size=(Noise_Data.shape[0], Noise_Data.shape[1]))
        Noise_Data[:,:,2] = noise
        Noise_Data[Noise_Data[:,:,1] == 0, 2] = 0
        # np.random.shuffle(Noise_Data[:,:,1:])
        self.Attack_Data[:,:,2] += Noise_Data[:,:,2]    

    def filterData(self, Z_processed: np.ndarray, 
                   predictions: pd.DataFrame, 
                   threshold: float = .5,
                   fixedIDs: Union[List[int], int] = None) -> np.ndarray:
        """Apply filtering to processed measurements"""
        Z_df = pd.DataFrame(Z_processed, columns=['ID', 'Reported', 'Data'])
        
        # Convert fixedIDs to a list if it's an integer
        if isinstance(fixedIDs, int):
            fixedIDs = [fixedIDs]
        elif fixedIDs is None:
            fixedIDs = []
        
        # Only process measurements that aren't fixed
        for idx, row in Z_df.iterrows():
            if row['ID'] in fixedIDs or row['ID'] == 0:
                continue
                
            if row['Reported'] == 1:
                pred_value = predictions['MS'][int(row['ID'])]
                if abs(row['Data'] - pred_value) > abs(pred_value * threshold/100):
                    Z_df.at[idx, 'Data'] = pred_value
                    
        return Z_df.values

    def imputeData(self, Z_processed: np.ndarray, 
                         predictions: pd.DataFrame) -> np.ndarray:
        """Apply imputation to missing measurements"""
        Z_df = pd.DataFrame(Z_processed, columns=['ID', 'Reported', 'Data'])
        
        # Impute missing or invalid measurements
        for idx, row in Z_df.iterrows():
            if row['Reported'] <= 0:  # Missing or invalid
                Z_df.at[idx, 'Reported'] = 1
                Z_df.at[idx, 'Data'] = predictions['MS'][int(row['ID'])]
        
        self.impute_data_list.append((Z_processed, predictions))  # Add this line
        
        return Z_df.values


    def initialize_network(self):
                sensors = [Node() for _ in range(54)]
                for i, sensor in enumerate(sensors):
                    sensor.nodeType = 'Sensor'
                    sensor.nodeID = i + 1
                    sensor.leaf = True
                    sensor.totSensor = 1
                    sensor.decSensor = 0
                    sensor.remSensor = sensor.decSensor
                    sensor.ids.append(sensor.nodeID)
                    sensor.reported.append(1)
                    sensor.deceptive.append(sensor.decSensor)
                    sensor.values.append(0)

                # Create clusters and assign sensors
                self.clusters = []
                for i in range(12):
                    cluster = Node()
                    cluster.nodeType = 'Cluster'
                    cluster.nodeID = i + 1
                    for j in range(3):
                        sensor = sensors[i*3 + j]
                        cluster.addChild(sensor)
                    self.clusters.append(cluster)

                # Create hubs and assign clusters and sensors
                self.hubs = []
                for i in range(2):
                    hub = Node()
                    hub.nodeType = 'Hub'
                    hub.nodeID = i + 1
                    for cluster in self.clusters[i*6:(i+1)*6]:
                        hub.addChild(cluster)
                    for sensor in sensors[36+i*5:36+(i+1)*5]:
                        hub.addChild(sensor)
                    self.hubs.append(hub)

                # Create server and assign hubs and remaining sensors
                self.server = Node()
                self.server.nodeType = 'Server'
                self.server.nodeID = 1
                for hub in self.hubs:
                    self.server.addChild(hub)
                for sensor in sensors[46:]:
                    self.server.addChild(sensor)
    @staticmethod
    def update_IDs(seed=0, shuffle_percent=0):
        random.seed(seed)
        all_sensor_ids = list(range(1, 55))  # IDs from 1 to 54

        print(f"ID update with shuffle_percent : {shuffle_percent}")
        
        # Cluster level: randomize each cluster of 3 sensors
        cluster_ids = all_sensor_ids[:36]
        dec_cluster_ids = DataLoader.shuffle_partial(cluster_ids, shuffle_percent)
        
        # Hub level: shuffle the specified percentage of the directly connected sensors
        hub_ids = all_sensor_ids[36:46]
        dec_hub_ids = DataLoader.shuffle_partial(hub_ids, shuffle_percent)
        
        # Server level: shuffle 50% of the directly connected sensors
        server_ids = all_sensor_ids[46:]
        dec_server_ids = DataLoader.shuffle_partial(server_ids, shuffle_percent)
        
        # Prepare the ID bank
        org_ids = all_sensor_ids
        dec_ids_cluster = dec_cluster_ids + hub_ids + server_ids
        dec_ids_hub = dec_cluster_ids + dec_hub_ids + server_ids
        dec_ids_server = dec_cluster_ids + dec_hub_ids + dec_server_ids
        
        IDbank = [
            (org_ids, dec_ids_cluster),
            (org_ids, dec_ids_hub),
            (org_ids, dec_ids_server)
        ]
        
        return IDbank
    
    @staticmethod
    def shuffle_partial(ids, shuffle_percent):
        """Shuffle a percentage of IDs while keeping the rest unchanged."""
        if not ids:
            return []
        
        # Convert shuffle_percent to float if it's a numpy array
        if isinstance(shuffle_percent, np.ndarray):
            shuffle_percent = float(shuffle_percent.item())
        
        # Ensure shuffle_percent is within valid range
        shuffle_percent = max(0.0, min(100.0, shuffle_percent))
        
        # Calculate number of IDs to shuffle
        num_to_shuffle = max(0, min(len(ids), int(len(ids) * shuffle_percent / 100)))
        
        if num_to_shuffle == 0:
            return ids.copy()
        
        try:
            # Select IDs to shuffle
            to_shuffle = random.sample(ids, num_to_shuffle)
            not_to_shuffle = [id for id in ids if id not in to_shuffle]
            
            # Shuffle selected IDs
            shuffled = random.sample(to_shuffle, len(to_shuffle))
            
            # Create result maintaining original order
            result = ids.copy()
            shuffle_indices = [ids.index(id) for id in to_shuffle]
            
            # Print shuffling details
            print("\nID Shuffling Details:")
            print("--------------------")
            print(f"Shuffle percent: {shuffle_percent}%")
            # print(f"Number of IDs shuffled: {num_to_shuffle}")
            # print("\nShuffled ID Mappings:")
            # print("Original ID -> New ID")
            for i, idx in enumerate(shuffle_indices):
                original_id = ids[idx]
                new_id = shuffled[i]
                result[idx] = new_id
                # print(f"ID {original_id:2d} -> {new_id:2d}")
            
            # print("\nUnchanged IDs:", not_to_shuffle.shape)
            
            return result
            
        except ValueError as e:
            print(f"Warning: Error during shuffle (ids: {len(ids)}, shuffle_percent: {shuffle_percent}, "
                  f"num_to_shuffle: {num_to_shuffle})")
            return ids.copy()

    def get_copy(self, attr: str):
        if hasattr(self, attr):
            value = getattr(self, attr)
            if isinstance(value, np.ndarray):
                return value.copy()
            elif isinstance(value, list):
                return value.copy()
            else:
                return copy.deepcopy(value)
        else:
            raise AttributeError(f"DataLoader has no attribute '{attr}'")

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"DataLoader has no attribute '{key}'")



    def load_data(self):
        # ... (other data loading code)   
        # Initialize W_list
        self.W_list = self.load_W_list()  # You might need to implement this method
        
        if self.W_list is None:
            # If W_list is not available, initialize it with a default value
            self.W_list = np.ones(self.numOfZ)
        


        if self.freq == 'sec':
            print("Loading Second wise data")
            senData = pd.read_csv(f"Bus Data//IEEE_{self.numOfBuses}_Bus_States_30_Sec.csv", index_col=0)
            senData.index = pd.to_datetime(senData.index)
            senData = senData.iloc[0:200000]
            try:
                senData = senData.drop(columns=['1'])
                print("Dropped state 1")
            except:
                pass

        self.dataset_org = senData.values.astype('float32')
        self.dataset = self.scaler.fit_transform(self.dataset_org)
        
        train_size = int(len(self.dataset) * 0.75)
        self.train = self.dataset[0:train_size, :]
        self.test = self.dataset[train_size:, :]

                # Ensure W_list has the correct length
        if len(self.W_list) != self.numOfZ:
            print(f"Adjusting W_list length from {len(self.W_list)} to {self.numOfZ}")
            self.W_list = np.ones(self.numOfZ)

        # # Update numOfZ based on actual data length
        # self.Z_mat = np.array(self.dataset_org)
        # if self.numOfZ != self.Z_mat.shape[1]:
        #     print(f"Warning: numOfZ ({self.numOfZ}) doesn't match the actual data length ({self.Z_mat.shape[1]}). Updating numOfZ.")
        #     self.numOfZ = self.Z_mat.shape[1]

        self.load_attack_data(self.attacked_Bus)
        
        # ... (rest of the method)
    
    def load_W_list(self):
        # Implement this method to load W_list from your data source
        self.W_list = np.ones(self.numOfZ)
        # Return the loaded W_list
        return self.W_list


    def create_dataset(self, dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), :]
            X.append(a)
            Y.append(dataset[i + look_back, :])
        return np.array(X), np.array(Y)

    def get_train_test_data(self, look_back=10):
        X_train, Y_train = self.create_dataset(self.train, look_back)
        X_test, Y_test = self.create_dataset(self.test, look_back)
        Y_test = self.scaler.inverse_transform(Y_test)
        Y_train = self.scaler.inverse_transform(Y_train)
        return X_train, Y_train, X_test, Y_test

    def get_initial_data(self):
        return self.dataset_org

    def verify_attack_data_shape(self):
        """Verify the attack data shape and content"""
        try:
            if self.Attack_Data is None:
                print("Attack data not loaded")
                return
                
            print(f"Attack_Data shape: {self.Attack_Data.shape}")
            print(f"Original rows: {7095}")
            print(f"Complete chunks possible: {7095 // 55}")
            print(f"Remainder measurements: {7095 % 55}")
            
            # Print first chunk details
            # print("\nFirst chunk details:")
            # print(self.Attack_Data[0])
            
        except Exception as e:
            print(f"Error verifying attack data: {str(e)}")
            traceback.print_exc()

class AttackScenarioManager:
    """Handles attack scenarios and tracking"""
    def __init__(self, numOfBuses: int, max_attacked_buses: int, data_loader: DataLoader):
        self.data_loader = data_loader
        self.numOfBuses = self.data_loader.numOfBuses
        self.max_attacked_buses = max_attacked_buses
        self.Attack_Data = self.data_loader.get_copy('Attack_Data')
        self.current_attack = 0

    def get_attack_scenario(self, Attack_Data, attack_index):
        """Get random attack scenario"""
        try:
            if self.Attack_Data is None:
                print("Attack data not loaded")
                return None
                
            # Generate random attack index between 0 and 127
            # attack_index = np.random.randint(0, 128)
            
            # Get the specific attack scenario
            attack_scenario = Attack_Data[attack_index]
            print(f"Selected attack scenario {attack_index}, shape: {attack_scenario.shape}")  # Should be (55, 3)
            
            return attack_scenario
            
        except Exception as e:
            print(f"Error getting attack scenario: {str(e)}")
            traceback.print_exc()
            return None

    # def identify_attacked_sensors(self, attack_scenario: np.ndarray) -> List[int]:
    #     """Identify attacked sensors in scenario"""
    #     return np.where(attack_scenario[:, 2] != 0)[0].tolist()

    # def apply_attack(self, attack_index):
    #     # Apply the attack based on the attack_index
    #     # This might involve modifying the system state or measurements
    #     attack = self.Attack_Data[attack_index]
    #     # ... (apply the attack logic)

class StateEstimator:

    def __init__(self, data_loader: DataLoader):
        import numpy as np
        self.np = np
        self.data_loader = data_loader
        self.H_mat = self.data_loader.get_copy('H_mat')
        self.W_list = self.data_loader.get_copy('W_list')
        self.Threshold_min = 1
        self.Threshold_max = 5
        self.Threshold_step =1
        self.runfilter = self.data_loader.runfilter
        self.runImpute = self.data_loader.runImpute

    def validate_inputs(self, Z_mat: np.ndarray) -> bool:
        """Validate inputs before state estimation"""
        try:
            if Z_mat.shape[1] != 3:
                raise ValueError("Invalid measurement format")
            if np.any(np.isnan(Z_mat)):
                raise ValueError("NaN values in measurements")
            if Z_mat.shape[0] != self.H_mat.shape[0]:
                raise ValueError("Measurement/H_matrix dimension mismatch")
            return True
        except Exception as e:
            print(f"Input validation failed: {str(e)}")
            return False

    def SE_BDD_COR(self, H_mat, Z_mat, W_list, Threshold_min, Threshold_max, Corr=False, Verbose="False"):
        if not isinstance(Z_mat, (self.np.ndarray, list)):
            raise ValueError(f"Z_mat must be a numpy array or list, not {type(Z_mat)}")
        

        print("###############Starting State Estimation.....###########")
        
        # Initialize variables
        self.H_mat = H_mat.copy()
        self.W_list = W_list.copy()
        self.Z_mat = Z_mat.copy()
        self.Threshold_min = Threshold_min
        self.Threshold_max = Threshold_max
        self.Threshold = Threshold_min
        self.doneFlag = 0  # Make sure this is a scalar integer
        self.fullRank = True
        self.Corr = Corr
        
        numberOfMeasurements = H_mat.shape[0]
        numberOfStates = H_mat.shape[1]

        print(f"numberOfMeasurements: {numberOfMeasurements}")
        print(f"numberOfStates: {numberOfStates}")

        while(self.doneFlag == 0):
            self.consideredIndx = np.where(self.Z_mat[:,1] == 1)[0]
            # print(f"consideredIndx: {self.consideredIndx}")
            # print(f"Z_mat shape: {self.Z_mat.shape}")
            # print(f"Number of measurements with flag=1: {len(self.consideredIndx)}")
            # print(f"Z_mat[:,1] unique values: {np.unique(self.Z_mat[:,1])}")

            self.Z_msr = self.Z_mat[self.consideredIndx][:,2]
            # print(f"Z_msr shape: {self.Z_msr.shape}")

            # considering only the corresponding columns in H
            self.H_msr = self.H_mat[self.consideredIndx]

            # Measurement Covariance Matrix
            if np.isscalar(self.W_list) or self.W_list.ndim == 0:
                self.R_msr = np.eye(len(self.consideredIndx)) * self.W_list
            else:
                valid_indices = self.consideredIndx[self.consideredIndx < len(self.W_list)]
                self.R_msr = np.diag(self.W_list[valid_indices])

            self.R_inv = np.linalg.inv(self.R_msr)
            self.Rank = np.linalg.matrix_rank(self.H_msr) if self.H_msr.shape[0] > 0 else 0

            if Verbose == "True":
                print(f"Current Rank: {self.Rank}")
                print(f"H_msr Shape: {self.H_msr.shape}")
                print(f"Z_msr Shape: {self.Z_msr.shape}")

            # Check if system is observable
            if self.Rank < numberOfStates:
                print(f"Warning: System under-determined. Rank {self.Rank} < States {numberOfStates}")
                
                try:
                    H_pinv = np.linalg.pinv(self.H_msr)
                    self.States = H_pinv @ self.Z_msr
                    self.Z_est = self.H_mat @ self.States
                    
                    # Calculate residuals using pseudo-inverse
                    self.Omega_mat = self.R_msr - (self.H_msr @ H_pinv)
                    
                    # Check noise with relaxed threshold
                    M_Noise_temp, P_Noise_temp, done_flag_temp = self.CheckNoiseCor(
                        self.Z_est, self.Z_mat, self.Omega_mat, self.R_msr, 
                        self.Threshold * 1.5,  # Relaxed threshold
                        False,  # Not full rank
                        self.Corr, 
                        Verbose
                    )
                    
                    # Safe conversion of done_flag_temp to scalar
                    if isinstance(done_flag_temp, np.ndarray):
                        if done_flag_temp.size > 0:
                            done_flag_temp = int(done_flag_temp[0])
                        else:
                            done_flag_temp = 0
                    else:
                        done_flag_temp = int(done_flag_temp)
                    
                    if done_flag_temp == 1:
                        self.M_Noise = M_Noise_temp
                        self.P_Noise = P_Noise_temp
                        self.doneFlag = 1
                        self.Noisy_Indx = np.where(self.Z_mat[:,1] == -1)[0]
                        return self.States, self.Z_est, self.Z_mat, self.M_Noise, self.Noisy_Indx, False, self.Threshold
                    
                except np.linalg.LinAlgError as e:
                    print(f"Pseudo-inverse method failed: {str(e)}")
                except Exception as e:
                    print(f"Error in pseudo-inverse section: {str(e)}")

                # If pseudo-inverse failed or threshold not met, try relaxing threshold
                if self.Threshold < self.Threshold_max:
                    self.Threshold += self.Threshold_step
                    print(f"Relaxing the threshold to {self.Threshold}")
                    continue
                else:
                    print(f"\nSystem Unobservable !, Rank = {self.Rank}")
                    self.fullRank = False
                    self.doneFlag = -1
                    self.Z_est = np.zeros(numberOfMeasurements)
                    self.States = np.zeros(numberOfStates)
                    M_Noise = np.zeros(numberOfMeasurements)
                    self.Noisy_Indx = np.where(self.Z_mat[:,1] == -1)[0]
                    return self.States, self.Z_est, self.Z_mat, M_Noise, self.Noisy_Indx, self.fullRank, self.Threshold

            # Full rank case
            inv__Ht_Rinv_H__Ht = np.linalg.inv(self.H_msr.T @ self.R_inv @ self.H_msr) @ self.H_msr.T
            self.States = inv__Ht_Rinv_H__Ht @ self.R_inv @ self.Z_msr
            self.Omega_mat = self.R_msr - (self.H_msr @ inv__Ht_Rinv_H__Ht)
            self.Z_est = self.H_mat @ self.States

            if Verbose == "True":
                # print(f"\nStates:\n{self.States}")
                # print(f"\nZ_est:\n{self.Z_est}")
                print("Checking noise...")

            # Check for bad data
            self.M_Noise, self.P_Noise, self.doneFlag = self.CheckNoiseCor(
                self.Z_est, self.Z_mat, self.Omega_mat, self.R_msr, 
                self.Threshold, self.fullRank, self.Corr, Verbose
            )

        # Safe conversion of final doneFlag
        if isinstance(self.doneFlag, np.ndarray):
            if self.doneFlag.size > 0:
                self.doneFlag = int(self.doneFlag[0])
            else:
                self.doneFlag = 0

        # Final noise index calculation
        self.Noisy_Indx = np.where(self.Z_mat[:,1] == -1)[0]
        
        # Debug information
        print(f"Debug - Final doneFlag: {self.doneFlag}")
        print(f"Debug - Final Noisy_Indx size: {len(self.Noisy_Indx)}")
        
        return self.States, self.Z_est, self.Z_mat, self.M_Noise, self.Noisy_Indx, self.fullRank, self.Threshold


    def CheckNoiseCor (self, Z_est, Z_mat, Omega_mat, R_msr, Threshold,  fullRank, Corr, Verbose):

        print("###############Starting Noise Checking.....###########")

        self.Z_est = Z_est
        self.Z_mat = Z_mat
        self.Omega_mat = Omega_mat
        self.R_msr = R_msr
        self.Threshold = Threshold
        self.fullRank = fullRank
        self.Corr = Corr
        self.Verbose = Verbose


        if self.fullRank != True:
        #         if Verbose == "True":
        #             print("System Unobservable!"
            return None, None, self.Z_mat[:,0]

    #     print ("Z_mat from SE", Z_mat )
        self.Z_msr = self.Z_mat[self.Z_mat[:, 1] == 1][:,2].copy()
    #     print ("Z_msr from SE", Z_msr )

        ####################################################   Here -------------------->
        '''boolean index did not match indexed array along dimension 0; dimension is 53
        but corresponding boolean dimension is 55'''

        if Verbose == "True":
            print("Starting BDD")
            print("Z_est: ", self.Z_est.shape)
            print("Z_msr: inside noise checking", self.Z_msr.shape)
            #print("Z_mat[:, 1] == 1", Z_mat[:, 1] == 1)
            #print(Z_mat)

        self.Z_est_msr = self.Z_est[self.Z_mat[:, 1] == 1]

        # Calculating the measurement error

        self.M_Noise = (self.Z_msr - self.Z_est_msr)
    #     print ("M_Noise :", self.M_Noise)

        self.M_Noise_norm = self.M_Noise.copy()

        # Calculating the normalized residuals
        for index, _ in enumerate(self.M_Noise):
            if index == 0: continue
            try:
                self.M_Noise_norm [index] = np.absolute(self.M_Noise [index])/math.sqrt(self.Omega_mat[index, index])
            except:
                self.M_Noise_norm [index] = 0
                if Verbose == "True":
                    print("index: ", index, np.absolute(self.M_Noise [index]))
                    print(f" Value Error, Expected postive, Got {self.Omega_mat[index, index]}")

    #     Noise_mat_actual = np.zeros(Z_mat.shape[0])
    #     Noise_mat_actual[Z_mat[:,1] == 1] = M_Noise
    #     Noise_mat_norm = np.zeros(Z_mat.shape[0])
    #     Noise_mat_norm[Z_mat[:,1] == 1] = M_Noise_norm

    #     print(M_Noise.shape)

        self.Noise_mat_actual = self.M_Noise.copy()
        self.Noise_mat_norm = self.M_Noise_norm.copy()
        print(f"Noise_mat_actual: {self.Noise_mat_actual.shape}")
        print(f"Noise_mat_norm: {self.Noise_mat_norm.shape}")

    #     print("Value of Normalized Noise", self.M_Noise_norm)

    #     print("Highest value of Noise_mat_norm : ", Noise_mat_norm.max() )

        active_idx = np.where(Z_mat[:,1] == 1)[0]

    #     print ("Active Index", active_idx )

        # Checking for Noisy data
        if np.max(self.Noise_mat_norm) > self.Threshold:
            tIndx = np.argmax(self.Noise_mat_norm)
            if Verbose == "True":
                print(f"targetedIndx in cut: {tIndx}--> Value : {self.Noise_mat_norm[tIndx]}")
                print("Updating Z_mat...")
                print("Before: ", Z_mat[tIndx])
            #print("R_msr: ", R_msr.shape, Omega_mat.shape)
            if self.Corr == True:
                correction_value= self.R_msr[tIndx,tIndx]/self.Omega_mat[tIndx,tIndx]*self.M_Noise[tIndx]
                print("Correction Value: ", correction_value )
                print("value of active_idx[tIndx]: ", active_idx[tIndx])
                print("Before: ", self.Z_mat[active_idx[tIndx],2])
                self.Z_mat[active_idx[tIndx], 2] = self.Z_mat[active_idx[tIndx], 2] - correction_value
                print("After: ", self.Z_mat[active_idx[tIndx],2])
            else:
                self.Z_mat[active_idx[tIndx], 1] = -1

            self.doneFlag = 0

            if Verbose == "True":
                print("After: ", self.Z_mat[tIndx])
        else:
            if Verbose == "True": print("No Bad Data Detected....")
            self.doneFlag = 1

            self.Noise_mat_actual = np.zeros(self.Z_mat.shape[0])
            self.Noise_mat_actual[self.Z_mat[:,1] == 1] = self.M_Noise.copy()
            self.Noise_mat_norm = np.zeros(self.Z_mat.shape[0])
            self.Noise_mat_norm[self.Z_mat[:,1] == 1] = self.M_Noise_norm.copy()
        ##############################################

        return self.Noise_mat_actual, self.Noise_mat_norm, self.doneFlag
    ##################################################################

    ##################################################################


class Node:
    def __init__(self):
        self.nodeType = ''
        self.nodeID = None
        self.parent = None
        self.leaf = False
        self.totSensor = 0
        self.decSensor = 0
        self.remSensor = 0
        self.nchild = 0
        self.child = []
        self.ids = []
        self.reported = []
        self.deceptive = []
        self.values = []

    def addChild(self, child_node):
        if not self.leaf:
            child_node.parent = self
            self.nchild += 1
            self.totSensor += child_node.totSensor
            self.decSensor += child_node.decSensor
            self.remSensor += child_node.remSensor
            self.child.append(child_node)
            self.ids += child_node.ids
            self.reported += child_node.reported
            self.deceptive += child_node.deceptive
            self.values += child_node.values
        else:
            print("This is already a leaf node!")




class HistoricalDataManager:
    def __init__(self, max_history: int, numOfBuses: int, H_mat: np.ndarray):
        self.max_history = max_history
        self.numOfBuses = numOfBuses
        self.H_mat = H_mat
        self.numOfZ = H_mat.shape[0]
        self.stateHist = deque(maxlen=self.max_history)
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        self.prediction_model = "polynomialRegression"  # or "linearRegression"

    def performanceAnalysis(self, stateHist, prediction_model, degree=2, expected_states=13):
        # Convert deque to numpy array
        stateHist_array = np.array(list(stateHist))
        
        lookBack = len(stateHist)
        if lookBack < 2:
            print("Not enough historical data for prediction. Using zeros.")
            return np.zeros(expected_states)

        try:
            # Reshape stateHist_array if necessary
            if stateHist_array.ndim == 1:
                stateHist_array = stateHist_array.reshape(-1, 1)
            elif stateHist_array.ndim > 2:
                stateHist_array = stateHist_array.reshape(stateHist_array.shape[0], -1)

            # Ensure we have the expected number of states
            if stateHist_array.shape[1] != expected_states:
                print(f"Warning: Expected {expected_states} states, but got {stateHist_array.shape[1]}.")
                # Pad or truncate to match expected_states
                if stateHist_array.shape[1] < expected_states:
                    stateHist_array = np.pad(stateHist_array, ((0, 0), (0, expected_states - stateHist_array.shape[1])))
                else:
                    stateHist_array = stateHist_array[:, :expected_states]

            x = np.arange(lookBack).reshape(-1, 1)
            
            if prediction_model == "polynomialRegression":
                print("###############Starting Polynomial Regression.....###########")
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(x)
                model = LinearRegression()
                model.fit(X_poly, stateHist_array)
                X_pred = poly.transform(np.array([[lookBack]]))
                statePre = model.predict(X_pred)
            elif prediction_model == "linearRegression":
                model = LinearRegression()
                model.fit(x, stateHist_array)
                statePre = model.predict(np.array([[lookBack]]))
            else:
                raise ValueError(f"Unsupported prediction model: {prediction_model}")

            return statePre.flatten()
        except Exception as e:
            print(f"Error in performanceAnalysis: {str(e)}")
            return np.zeros(expected_states)

    def update_stateHist(self, state):
        print("Updating StateHist")
        
        # Ensure state is 2D
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # Check for NaN or infinite values
        if np.isnan(state).any() or np.isinf(state).any():
            print("Warning: NaN or infinite values detected in state")
            state = np.nan_to_num(state)  # Replace NaN with 0 and inf with large finite numbers
        
        # Fit or update the scaler
        if not self.is_scaler_fitted:
            self.scaler.fit(state)
            self.is_scaler_fitted = True
        else:
            self.scaler.partial_fit(state)
        
        # Transform the state
        scaled_state = self.scaler.transform(state)
        
        # Append the new state to the history
        self.stateHist.append(scaled_state.flatten())
        
        # If the history is full, remove the oldest state
        if len(self.stateHist) > self.max_history:
            self.stateHist.popleft()

    def get_scaled_state(self, state):
        if not self.is_scaler_fitted:
            raise ValueError("Scaler is not fitted yet. Update state history first.")
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return self.scaler.transform(state)

    def pred_msr(self, statePre):
        print(f"H_mat shape: {self.H_mat.shape}")
        print(f"Original statePre shape: {statePre.shape}")
        
        # Flatten statePre if it's not already 1D
        if statePre.ndim > 1:
            statePre = statePre.flatten()
        
        # Concatenate a zero at the beginning of statePre
        statePre = np.concatenate((np.array([0]), statePre), axis=0)
        print(f"statePre shape after concatenation: {statePre.shape}")
        
        expected_size = self.H_mat.shape[1]
        if statePre.shape[0] != expected_size:
            print(f"Warning: statePre size ({statePre.shape[0]}) doesn't match H_mat columns ({expected_size}). Adjusting...")
            if statePre.shape[0] > expected_size:
                statePre = statePre[:expected_size]
            else:
                statePre = np.pad(statePre, (0, expected_size - statePre.shape[0]), mode='constant')
            print(f"Adjusted statePre shape: {statePre.shape}")
        
        Zpre = pd.DataFrame(index=range(self.numOfZ), columns=['MS'])
        Zpre['MS'] = (self.H_mat @ statePre).flatten()
        return Zpre

    def predict_next_state(self):
        return self.performanceAnalysis(
            np.array(self.stateHist), 
            self.prediction_model, 
            degree=2, 
            expected_states=self.H_mat.shape[1]
        )


class DataProcessor:
    """Enhanced data processing pipeline"""
    def __init__(self, data_loader: DataLoader):

        self.np = np
        self.data_loader = data_loader
        self.scaler = data_loader.scaler  # Use the same scaler from DataLoader
        self.numOfZ = self.data_loader.numOfZ
        self.H_mat = self.data_loader.get_copy('H_mat')
        self.state_estimator = StateEstimator(self.data_loader)  # Pass data_loader instead of individual parameters
        self.prediction_model = "polynomialRegression"
        self.lookback = self.data_loader.get_copy('lookback')
        self.scaler = StandardScaler()
        self.state_history = deque(maxlen=self.lookback)
        self.Z_mat = self.data_loader.get_copy('Z_mat')
        
        # Fit the scaler with initial data
        initial_data = self.data_loader.get_initial_data()  # You need to implement this method
        self.fit_scaler(initial_data)

    def fit_scaler(self, data):
        self.scaler.fit(data)

    def inverse_transform_state(self, state):
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return self.scaler.inverse_transform(state)


    def transform_state(self, state):
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return self.scaler.transform(state)

    def process_measurements(self, Z_mat: np.ndarray) -> Dict:
        """Complete measurement processing"""
        # 1. Validation
        if not self.validate_inputs(Z_mat):
            return None
            
        # 2. State Estimation
        self.Z_mat = Z_mat.copy()
    
        SE_results = self.state_estimator.SE_BDD_COR(
            self.H_mat, Z_mat, self.data_loader.get_copy('W_list'),
            self.state_estimator.Threshold_min, self.state_estimator.Threshold_max,
            False, "False"
        )
        
        if SE_results is None:
            return None
            
        States, Z_est, Z_processed, M_Noise, Noisy_Indx, fullRank, Threshold = SE_results
        
        # 3. Historical Processing
        if fullRank:
            scaled_states = self.scaler.transform(States[1:].reshape(1,-1))
            self.update_state_history(scaled_states)
            
        # 4. Return Results
        return {
            'States': States,
            'Z_est': Z_est,
            'fullRank': fullRank,
            'Noisy_Indx': Noisy_Indx,
            'M_Noise': M_Noise
        }

    def _validate_input(self, Z_mat: np.ndarray) -> None:
        """Validate input measurements"""
        if Z_mat.shape[1] != 3:  # [ID, Reported, Data]
            raise ValueError("Invalid measurement format")
        if np.any(np.isnan(Z_mat)):
            raise ValueError("NaN values in measurements")

    def apply_id_mapping(self, Z_mat: np.ndarray, IDbank: list) -> np.ndarray:
        """Apply ID mapping to measurements"""
        Z_dec = Z_mat.copy()
        # print(f"Z_dec before mapping: {Z_dec}")
        selectedIDs, deceptiveIDs = IDbank[0]  # Using first level of ID bank
        Z_dec[selectedIDs, 0] = deceptiveIDs.copy()
        # print(f"Z_dec after mapping: {Z_dec}")
        Z_dec[Z_dec[:, 0].argsort(kind='mergesort')]
        # print(f"Z_dec after sorting: {Z_dec}")
        return Z_dec

    def addDecoy_Data(self, Z_dec, consideredDecoyIDs):
        """Add decoy data to measurements"""
        # print(f"Z_dec shape: {Z_dec.shape}")
        # print(f"consideredDecoyIDs: {consideredDecoyIDs}")
        # print(f"W_list shape: {self.data_loader.W_list.shape}")
        # print(f"H_mat shape: {self.data_loader.H_mat.shape}")

        W_list = self.data_loader.W_list
        if W_list is None or self.np.isscalar(W_list) or W_list.ndim == 0:
            W_list = self.np.ones(self.data_loader.numOfZ)

        States_decoy, Z_est_decoy, _, _, _, _, _ = self.state_estimator.SE_BDD_COR(
            H_mat=self.data_loader.H_mat,
            Z_mat=Z_dec,
            W_list=W_list,
            Threshold_min=self.state_estimator.Threshold_min,
            Threshold_max=self.state_estimator.Threshold_max,
            Corr=False,
            Verbose="False"
        )
        
        # Iteratively refine decoy values
        Z_repeat = Z_dec.copy()
        for _ in range(5):
            Z_repeat[consideredDecoyIDs, 2] = Z_est_decoy[consideredDecoyIDs]
            States_decoy, Z_est_decoy, _, _, _, _, _ = self.state_estimator.SE_BDD_COR(
                H_mat=self.data_loader.H_mat,
                Z_mat=Z_repeat,
                W_list=W_list,
                Threshold_min=self.state_estimator.Threshold_min,
                Threshold_max=self.state_estimator.Threshold_max,
                Corr=False,
                Verbose="False"
            )
            
        return Z_repeat[consideredDecoyIDs, 2]

    def _get_predictions(self) -> pd.DataFrame:
        """Get predictions from historical data"""
        if len(self.state_history) < self.lookback:
            return pd.DataFrame()  # Return empty if not enough history
            
        # Convert history to array
        hist_array = np.array(self.state_history)
        
        # Get predictions based on model type
        if self.prediction_model == "polynomialRegression":
            predicted_state = self._polynomial_regression(hist_array)
        elif self.prediction_model == "linearRegression":
            predicted_state = self._linear_regression(hist_array)
        else:
            raise ValueError(f"Unsupported prediction model: {self.prediction_model}")
            
        # Convert predictions to measurements
        return self._state_to_measurements(predicted_state)

    def _polynomial_regression(self, hist_array: np.ndarray) -> np.ndarray:
        """Polynomial regression prediction"""
        x = np.array(range(self.lookback)).reshape(-1, 1)
        x_pred = np.array([self.lookback]).reshape(-1, 1)
        
        poly = PolynomialFeatures(degree=2)
        x_poly = poly.fit_transform(x)
        x_pred_poly = poly.transform(x_pred)
        
        model = LinearRegression()
        model.fit(x_poly, hist_array)
        
        return model.predict(x_pred_poly)

    def _state_to_measurements(self, state: np.ndarray) -> pd.DataFrame:
        """Convert state predictions to measurements"""
        state = np.concatenate((np.array([0]), state.flatten()))
        measurements = self.H_mat @ state
        
        return pd.DataFrame({
            'ID': np.arange(0, self.numOfZ + 1),
            'MS': measurements
        })

    def update_state_history(self, state):
        """Update state history"""
        self.state_history.append(state[0])

    def inverse_transform_state(self, state):
        """Safely inverse transform the state with proper shape handling."""
        if not hasattr(self.scaler, 'mean_'):
            print("Scaler is not fitted. Using raw state.")
            return state.flatten()
        
        # Ensure state has the expected shape for the scaler
        expected_size = self.scaler.mean_.shape[0]  # 14 in your case
        if state.ndim == 1:
            state = state.reshape(1, -1)

        # Pad or trim the state to match the expected size
        if state.shape[1] != expected_size:
            print(f"Warning: Adjusting state shape from {state.shape[1]} to {expected_size}")
            state = np.pad(state, ((0, 0), (0, max(0, expected_size - state.shape[1]))), mode='constant')
            state = state[:, :expected_size]

        return self.scaler.inverse_transform(state).flatten()


class DefenseStrategy:
    """Implements defense mechanisms"""
    def __init__(self, data_loader: DataLoader, historical_manager: HistoricalDataManager, data_processor: DataProcessor):
        self.data_loader = data_loader
        self.data_processor = data_processor
        self.historical_manager = historical_manager
        self.Z_org = self.data_loader.get_copy('Z_org')
        self.numOfZ = self.data_loader.numOfZ
        self.H_mat = self.data_loader.get_copy('H_mat')
        self.W_list = self.data_loader.get_copy('W_list')
        self.state_estimator = StateEstimator(self.data_loader)
        # self.historical_manager = HistoricalDataManager(100, self.data_loader)
        self.prediction_model = "polynomialRegression"
        self.Z_mat= self.data_loader.get_copy('Z_mat')
        self.recoveryType = self.data_loader.get_copy('recoveryType')
        self.th = self.data_loader.get_copy('th')
        self.attackCat = self.data_loader.get_copy('attackCat')

        self.server = self.data_loader.get_copy('server')
        self.clusters = self.data_loader.get_copy('clusters')
        self.hubs = self.data_loader.get_copy('hubs')
        self.data_loader.initialize_network()

        
        # Add these new attributes
        self.Threshold_min = self.data_loader.get_copy('Threshold_min')  # You may want to adjust this value
        self.Threshold_max = self.data_loader.get_copy('Threshold_max')  # You may want to adjust this value
        self.attackerLevel = self.data_loader.get_copy('attackerLevel')
        self.attackertype = self.data_loader.get_copy('attackertype')
        self.recoveryType = self.data_loader.get_copy('recoveryType')
        self.Correction = False  # You may want to adjust this value

        # Initialize other necessary attributes
        self.consideredDecoyIDs = []
        self.consideredFixedIDs = []
        self.attackerTypeList = []
        self.attackerLevelList = []
        self.Noisy_Indx_actu = []  # Initialize as an empty list
        self.detection_check = 0

    def set_random_seed(self, seed):
        """
        Set the random seed for reproducible randomization.
        """
        random.seed(seed)

    def randomize_clusters(self, ids):
        """
        Randomize each group of 3 IDs (representing a cluster) among themselves.
        """
        result = []
        for i in range(0, len(ids), 3):
            group = ids[i:i+3]
            random.shuffle(group)
            result.extend(group)
        return result
    
    # def addDecoy_Data(self, Z_dec, consideredDecoyIDs):
    #     """Add decoy data to measurements"""
    #     print(f"Z_dec shape: {Z_dec.shape}")
    #     print(f"consideredDecoyIDs: {consideredDecoyIDs}")
    #     print(f"W_list shape: {self.data_loader.W_list.shape}")
    #     print(f"H_mat shape: {self.data_loader.H_mat.shape}")

    # @staticmethod
    
    
    def filterData(self, Z_processed: np.ndarray, 
                   predictions: pd.DataFrame, 
                   threshold: float = 5,
                   fixedIDs: Union[List[int], int] = None) -> np.ndarray:
        """Apply filtering to processed measurements"""
        Z_df = pd.DataFrame(Z_processed, columns=['ID', 'Reported', 'Data'])
        
        # Convert fixedIDs to a list if it's an integer
        if isinstance(fixedIDs, int):
            fixedIDs = [fixedIDs]
        elif fixedIDs is None:
            fixedIDs = []
        
        # Only process measurements that aren't fixed
        for idx, row in Z_df.iterrows():
            if row['ID'] in fixedIDs or row['ID'] == 0:
                continue
                
            if row['Reported'] == 1:
                pred_value = predictions['MS'][int(row['ID'])]
                if abs(row['Data'] - pred_value) > abs(pred_value * threshold/100):
                    Z_df.at[idx, 'Data'] = pred_value
                    
        return Z_df.values

    def imputeData(self, Z_processed: np.ndarray, 
                   predictions: pd.DataFrame) -> np.ndarray:
        """Apply imputation to missing measurements"""
        Z_df = pd.DataFrame(Z_processed, columns=['ID', 'Reported', 'Data'])
        
        # Impute missing or invalid measurements
        for idx, row in Z_df.iterrows():
            if row['Reported'] <= 0:  # Missing or invalid
                Z_df.at[idx, 'Reported'] = 1
                Z_df.at[idx, 'Data'] = predictions['MS'][int(row['ID'])]
                
        return Z_df.values
    
    def reset_to_original(self, IDbank):
        """
        Reset the ID bank to the original state.
        """
        original_ids = IDbank[0][0]  # The original IDs are the same for all levels
        return [(original_ids, original_ids) for _ in range(3)]
    
    def mapOrgID(self, attackedIndx, selectedIDs, deceptiveIDs):
        IDs = []
        for i in attackedIndx:
            found = np.where(deceptiveIDs == i)[0]
            if found.size > 0:
                IDs.append(selectedIDs[found[0]])
        return np.array(IDs)

    def update_state_history(self, state):
        self.historical_manager.update_stateHist(state)

    def defenseEval(self, Attack_Data, attackIndx, IDbank, attackerLevelList, recoveryType, addDecoy, consideredFixedIDs, consideredDecoyIDs, attackCat, attackertype, verbose_=True, runfilter=False, runImpute=False, Z_dec=None):
        # Initialize attackID at the start of the method
        
        print("###############Starting Attack Evaluation.....###########")

        self.attackID = attackIndx  # Change this line
        self.attackIndx = attackIndx  # Add this line

        self.Attack_Data = Attack_Data
        if self.Attack_Data is None:
            raise ValueError("Attack_Data is None in defenseEval method")

        print(f"Attack_Data shape inside defenseEval: {self.Attack_Data.shape}")
        print(f"attackIndx: {self.attackID}")

        self.outliers_suspect_List_List = []
        self.th = self.data_loader.get_copy('th')
        self.Z_org  = self.data_loader.get_copy('Z_org')
        self.H_mat  = self.data_loader.get_copy('H_mat')
        self.W_list = self.data_loader.get_copy('W_list')
        self.recoveryType = recoveryType
        self.attackCat = attackCat
        self.consideredFixedIDs = consideredFixedIDs
        self.consideredDecoyIDs = consideredDecoyIDs
        

        # Use the provided attackertype
        attackertype_ = self.attackertype

        # If attackertype is 3, randomly choose from 0, 1, 2
        if attackertype_ == 3:
            attackertype_ = random.choice([0, 1, 2])

        # self.attackerLevel = attackerLevelList[attackertype_]
        self.attackerLevel = 1

        # selectedIDs = IDbank[attackertype_][0]
        # deceptiveIDs = IDbank[attackertype_][1]


        self.selectedIDs, self.deceptiveIDs = IDbank[attackertype_]
        print(f"selectedIDs : {self.selectedIDs}")
        print(f"deceptiveIDs : {self.deceptiveIDs}")

        if verbose_:
            print(f"Attack type: {self.attackertype}, Attack level: {self.attackerLevel}")

        # Implement the attack
        Z_dec = self.Z_org.copy()

        if self.attackertype != -1:
            print(f"selectedIDs : {self.selectedIDs}")
            print(f"deceptiveIDs : {self.deceptiveIDs}")
            # print(f"Z_dec before mapping: {Z_dec[:, 0]}") # Using first level of ID bank
            Z_dec[self.selectedIDs, 0] = self.deceptiveIDs.copy()
            # print(f"Z_dec after mapping: {Z_dec[:, 0]}")
            Z_dec = Z_dec[Z_dec[:, 0].argsort(kind='mergesort')]
            # print(f"Z_dec after sorting: {Z_dec}")

        if len(self.consideredDecoyIDs) > 0 and addDecoy == True:
            Z_dec[self.consideredDecoyIDs, 2] = self.data_processor.addDecoy_Data(Z_dec, self.consideredDecoyIDs)

        try :
            self.attackID = self.attackIndx
            self.Attack_Data = Attack_Data

            if self.Attack_Data is None:
                raise ValueError("Attack_Data is None")

            # print(f"Attack_Data shape: {Attack_Data.shape}")
            # print(f"attackIndx: {self.attackID}")

            # Initialize Z_dec if None
            if Z_dec is None:
                Z_dec = self.Z_org.copy()

            # Create FData with same shape as Z_dec
            FData = np.zeros_like(Z_dec)  # This ensures FData has same shape as Z_dec
            
            # Copy attack data into FData
            if self.attackID >= 0 and self.attackID < Attack_Data.shape[0]:
                attack_data = Attack_Data[self.attackID]
            else:
                print(f"Warning: Invalid attackIndx {attackIndx}. Using default attack data.")
                attack_data = Attack_Data[self.attackID]

            # Ensure attack_data is 2D
            if attack_data.ndim == 1:
                attack_data = attack_data.reshape(1, -1)

            # Copy available columns
            num_cols = min(attack_data.shape[1], FData.shape[1])
            FData[:attack_data.shape[0], :num_cols] = attack_data[:, :num_cols]

            # print(f"FData shape: {FData.shape}")
            # print(f"Z_dec shape: {Z_dec.shape}")

            # Calculate attack gain
            attackGain = 2 if self.attackID >= 0 else 0

            # Limit attack magnitude
            if abs(FData[:,2]).max() > 50:
                FData[:,2] = FData[:,2] * 50/abs(FData[:,2]).max()

            # Calculate injection
            injection = FData[:,2] * attackGain
            # print(f"injection shape: {injection}")

            # Add injection to measurements
            FData[:,2] = injection + Z_dec[:,2]

            # Expected attack indices
            self.expected_attack_indices = np.where(injection != 0)[0]
            print(f"Expected attack indices: {self.expected_attack_indices}")

            # Map attack to shuffled IDs
            Z_att = Z_dec.copy()

            if attackCat == 'FDI':
                Z_att[self.expected_attack_indices, 2] = FData[self.expected_attack_indices, 2]
            elif attackCat == 'DoS':
                Z_att[self.expected_attack_indices, 1] = -1


            


            # if verbose_:
            #     # print("Expected to attack:", expected_attack_indices)
            #     # print("Actually attacked:", actually_attacked)
            #     print("Attack type:", self.attackCat)
            #     print("Attack level:", self.attackerLevel)
        
        except Exception as e:
            print(f"Error in defenseEval: {str(e)}")
            print(f"Attack_Data shape: {Attack_Data.shape if hasattr(Attack_Data, 'shape') else 'no shape'}")
            if 'FData' in locals():
                print(f"FData shape: {FData.shape}")
                # print(f"FData content: {FData}")
            if 'Z_dec' in locals():
                print(f"Z_dec shape: {Z_dec.shape}")
                # print(f"Z_dec content: {Z_dec}")
            if 'injection' in locals():
                print(f"injection shape: {injection.shape}")
                # print(f"injection content: {injection}")
            return (
                None,  # AttackReturn
                0,    # totalAttackedSensors
                0,    # successCount
                0,    # successCount_avg
                0     # Deviation
            )

        # Received Data after Attack
        received = pd.DataFrame(Z_att[Z_att[:, 1] == 1][:, [0, 2]], columns=['ID', 'MS'])
        received['ID'] = received['ID'].astype(int)

        fixedIDDs = list(set(self.consideredFixedIDs).intersection(set(received['ID'].values)))
        print(f"fixedIDDs : {fixedIDDs}")
        decoyIDDs = list(set(self.consideredDecoyIDs).intersection(set(received['ID'].values)))
        print(f"decoyIDDs : {decoyIDDs}")

        #--------------------------------------------------------------------------#
        #&&&&&&&&&&&&&&&&&&&&&&&&&& Recovering from the attack &&&&&&&&&&&&&&&&&&
        #--------------------------------------------------------------------------#
        if runfilter == True or runImpute == True or recoveryType == 2 or recoveryType == 3:
            # Zpre Data using prediction models
            statePre = self.historical_manager.performanceAnalysis(
                np.array(self.historical_manager.stateHist), 
                self.historical_manager.prediction_model, 
                degree=2, 
                expected_states=self.H_mat.shape[1]
            )
            print(f"statePre shape after prediction: {statePre.shape}")

            # Ensure statePre is 2D
            if statePre.ndim == 1:
                statePre = statePre.reshape(1, -1)
            
            if not hasattr(self.historical_manager.scaler, 'mean_'):
                # Scaler not fitted; cannot inverse transform
                print("Scaler not fitted. Cannot inverse transform state predictions.")
                statePre_inv = statePre
            else:
                statePre_inv = self.data_processor.inverse_transform_state(statePre.reshape(1, -1))

            # statePre_inv = self.historical_manager.scaler.inverse_transform(statePre)
            
            # Ensure statePre_inv is 1D for pred_msr
            statePre_inv = self.data_processor.inverse_transform_state(statePre)
            print(f"statePre_inv shape: {statePre_inv.shape}")

            
            Zpre = self.historical_manager.pred_msr(statePre_inv)

        Z_rec = Z_att.copy()
        if recoveryType == 1 or recoveryType == 3:
            
            Z_rec[self.deceptiveIDs, 0] = self.selectedIDs.copy()
            Z_rec = Z_rec [Z_rec[:,0].argsort(kind = 'mergesort')]
            # print(f"Z_rec after recovery: {Z_rec[:, 0]}")
            Zpair = (self.Z_mat, Z_rec)
            # self.z_matrices.append(Zpair)

        if recoveryType == 2 or recoveryType == 3:
            recoZdf = self.runMatch(self.numOfBuses, self.numOfLines, Zpre, received, fixedIDDs, decoyIDDs)
            self.get_data.append((self.numOfBuses, self.numOfLines, Zpre, received, recoZdf, fixedIDDs, decoyIDDs))

            recoZdf = recoZdf.sort_values(['rem_ID'])
            Z_rec   = self.Z_org.copy()
            Z_rec[1:,1] = 0
            Z_rec[recoZdf['rem_ID'].values.astype(int),1] = 1
            Z_rec[recoZdf['rem_ID'].values.astype(int),2] = recoZdf['MS'].copy()
            Zpair = (self.Z_mat, Z_rec)
            self.z_matrices.append(Zpair)



                    # Actually attacked indices after shuffling
        self.actually_attacked = np.sort(self.mapOrgID(self.expected_attack_indices.copy(), 
                                                list(self.selectedIDs), 
                                                list(self.deceptiveIDs)).astype(int))    
        
        print(f"actually_attacked : {self.actually_attacked}")


        if verbose_: print("Calling State Estimation for original data..")

        # State Estimation and Bad Data Detection for original data
        States_org, Z_est_init, Z_processed_org, M_Noise_org, Noisy_Indx_org, fullRank_org, Threshold_org = self.state_estimator.SE_BDD_COR(
            self.H_mat.copy(), self.Z_org.copy(), self.W_list, self.Threshold_min, self.Threshold_max, self.Correction, Verbose="True")

        # Set Noisy_Indx_actu
        self.Noisy_Indx_actu = Noisy_Indx_org
        print(f"Noisy Indx without attack{self.Noisy_Indx_actu}")

        if verbose_: print("Calling State Estimation for recovered data..")
        # State Estimation and Bad Data Detection for recovered data
        States_check, Z_est_check, Z_mat_check, M_Noise_check, Noisy_index_check, Rank_check, Threshold_check = self.state_estimator.SE_BDD_COR(
            self.H_mat.copy(), Z_rec.copy(), self.W_list, self.Threshold_min, self.Threshold_max, self.Correction, Verbose = "True")

        savedState = (States_check, Z_est_check, Z_mat_check, M_Noise_check, Noisy_index_check, Rank_check, Threshold_check)

        runningfilter = False; runningImputer = False

        # actually_attacked = np.sort((self.mapOrgID(expected_attack_indices.copy(), selectedIDs, deceptiveIDs)).astype(int))

        # if verbose_: 
        #     print("Expected to attack:", expected_attack_indices)
        #     print("Actually attacked:", actually_attacked)

        Z_rec_df = pd.DataFrame(Z_mat_check, columns = ['ID', 'Taken', 'MS'])
        #########################  Adding LSTM Features ####################
        if Rank_check == True:
            if runfilter == True:
                Z_reco = self.filterData(Z_rec_df, Zpre, self.th, fixedIDDs)
                runningfilter = True
        else:
            if runImpute == True:
                print("System is not observable!!")
                self.data_loader.imputeData(Z_rec_df, Zpre)  # Change this line
                Z_reco = self.data_loader.imputeData( Z_rec_df, Zpre)
                runningImputer = True

        if runningfilter == True or runningImputer == True:
            # State Estimation and Bad Data Detection
            States_check, Z_est_check, Z_mat_check, M_Noise_check, Noisy_index_check, Rank_check, Threshold_check = self.state_estimator.SE_BDD_COR(
                self.H_mat.copy(), Z_reco.copy(), self.W_list, self.Threshold_min, self.Threshold_max, self.Correction, Verbose = "False")

            if runfilter == True and Rank_check == False:
                print("prediction error!!") if runfilter == True else print("Imputation Failed")
                (States_check, Z_est_check, Z_mat_check, M_Noise_check, Noisy_index_check, Rank_check, Threshold_check) = savedState

            if runImpute == True and Rank_check == False:
                print("Imputation failed!!!!!!")

        self.update_state_history(States_check[1:])  # Assuming States[0] is a timestamp or something to be excluded

        M_Noise_check = Z_rec[:,2] - Z_est_check

        print("\nDebug Information:")
        print(f"Noisy_index_check type: {type(Noisy_index_check)}")
        print(f"Noisy_index_check: {Noisy_index_check}")
        print(f"Noisy_Indx_actu type: {type(self.Noisy_Indx_actu)}")
        print(f"Noisy_Indx_actu: {self.Noisy_Indx_actu}")
        print(f"actually_attacked type: {type(self.actually_attacked)}")
        print(f"actually_attacked: {self.actually_attacked}")

        # Ensure arrays are numpy arrays
        Noisy_index_check = np.array(Noisy_index_check) if not isinstance(Noisy_index_check, np.ndarray) else Noisy_index_check
        self.Noisy_Indx_actu = np.array(self.Noisy_Indx_actu) if not isinstance(self.Noisy_Indx_actu, np.ndarray) else self.Noisy_Indx_actu
        self.actually_attacked = np.array(self.actually_attacked) if not isinstance(self.actually_attacked, np.ndarray) else self.actually_attacked

        # Calculate foundFDI_Idx with additional error checking
        try:
            self.foundFDI_Idx = sorted((set(Noisy_index_check) - set(self.Noisy_Indx_actu)) & set(self.actually_attacked))
            print(f"foundFDI_Idx: {self.foundFDI_Idx}")
        except Exception as e:
            print(f"Error calculating foundFDI_Idx: {str(e)}")
            self.foundFDI_Idx = []

        # printing noisy indeces
        if Noisy_index_check.size > 0 and verbose_ == True:
            pass

        # Initialize counters
        self.successful_attack_count = 0
        self.successfulDetectionCount = 0
        self.successfulDetectionCount_avg = 0
        self.Deviation = 0
        self.totalAttackedSensors = 0


        # system Unobservable
        if Rank_check == False:
            if verbose_:
                print("-----------  System Unobservable  --------------")
            Deviation = 0

            AttackEval = [self.attackID, "unobservable", Deviation]
            AttackReturn = {}
            AttackReturn['StatesOriginal'] =  0
            AttackReturn['StatesDeceived'] =  0
            AttackReturn['Deviation']  = 0
            AttackReturn['Check'] = 0
            AttackReturn['Zpair'] = (self.Z_org.copy(), self.Z_org.copy())  # Ensure Zpair is always set
            AttackReturn['Z_dec'] = 0
        else:
            # system is observable
            # calculating the percent of deviation in the estimated measurements
            self.Deviation = np.linalg.norm(Z_est_check - Z_est_init) / np.linalg.norm(Z_est_init) * 100 if np.linalg.norm(Z_est_init) != 0 else 0
            print(f"Deviation: {self.Deviation}")

            ########################
            # totalAttackedSensors = np.sum(Z_rec[actually_attacked, 1])
            if hasattr(self, 'actually_attacked') and len(self.actually_attacked) > 0:
                self.totalAttackedSensors = np.sum(Z_rec[self.actually_attacked, 1])
            else:
                self.totalAttackedSensors = 0

            # Calculate success count
            if self.totalAttackedSensors > 0:
                self.successfulDetectionCount = np.sum(np.isin(self.foundFDI_Idx, self.actually_attacked))
                self.successfulDetectionCount_avg = (self.successfulDetectionCount / self.totalAttackedSensors) * 100    
            else:
                self.successfulDetectionCount = 0
                self.successfulDetectionCount_avg = 0

            # Debug prints
            print(f"Total attacked sensors: {self.totalAttackedSensors}")
            print(f"Successful Detection count: {self.successfulDetectionCount}")
            print(f"Successful Detectioncount average: {self.successfulDetectionCount_avg}")

            ########################
            if self.attackID > 0:
                pass

            # Detected as Bad Data
            if len(self.foundFDI_Idx) > 0:
                AttackEval = [self.attackID, "detected", self.Deviation]
                self.detection_check += 1
                print("Detected Measurements: ", self.foundFDI_Idx)

                if verbose_:
                    print("\nFDI Attack Detection Results:")
                    print("-" * 40)
                    print(f"Total sensors attacked: {len(self.actually_attacked)}")
                    print(f"Sensors detected as attacked: {len(self.foundFDI_Idx)}")
                    print(f"True positives: {len(set(self.foundFDI_Idx) & set(self.actually_attacked))}")
                    print(f"False positives: {len(set(self.foundFDI_Idx) - set(self.actually_attacked))}")
                    print(f"False negatives: {len(set(self.actually_attacked) - set(self.foundFDI_Idx))}")
                    print(f"Detection accuracy: {self.successfulDetectionCount_avg:.2f}%")
                    
                    if len(self.foundFDI_Idx) > 0:
                        print("\nDetailed Attack Information:")
                        print("Sensor ID | Actually Attacked | Detected as Attack")
                        print("-" * 50)
                        all_sensors = sorted(set(self.foundFDI_Idx) | set(self.actually_attacked))
                        for sensor_id in all_sensors:
                            was_attacked = "✓" if sensor_id in self.actually_attacked else "X"
                            was_detected = "✓" if sensor_id in self.foundFDI_Idx else "X"
                            print(f"{sensor_id:^9} | {was_attacked:^16} | {was_detected:^16}")

            # Attack was undetected
            else:
                if verbose_:
                    print("$$$$$$$$$$$$$$  Attack Successful as undetected $$$$$$$$$$$$$")
                    print("$$$$$$$$$$$$$$  Attack Successful as undetected , Attack ID: $$$$$$$$$$$$$", self.attackID)
                AttackEval = [self.attackID, "success", self.Deviation]
                self.successful_attack_count = 1

            AttackReturn = {}

            AttackReturn['StatesOriginal'] =  States_org
            AttackReturn['StatesDeceived'] =  States_check

            AttackReturn['Deviation']  = AttackEval[2]
            AttackReturn['Check'] = AttackEval[1]
            AttackReturn['Zpair'] = Zpair if isinstance(Zpair, tuple) else (self.Z_org.copy(), self.Z_org.copy())
            AttackReturn['Z_dec'] = Z_dec

            print("############### Attack Evaluation Done.....###########")

        return AttackReturn, self.totalAttackedSensors, self.successfulDetectionCount, self.successfulDetectionCount_avg, self.Deviation   

    def runMatch(self, numOfBuses, numOfLines, z_pre, z_rep, fixedIDs, decoyIDDs):

        self.th = self.data_loader.get_copy('th')

        th_max = abs(z_pre['MS']*self.th/100)
        th_max = np.ceil(th_max)
        th_max[th_max < 1] = 1

        n_pre = z_pre.shape[0]
        n_rep = z_rep.shape[0]

        costs = []
        for p in z_pre['MS']:
            costs.append(list(abs(z_rep['MS'] - p)))

        for l in fixedIDs:
            try:
                i = pd.Index(z_pre['ID']).get_loc(l)
                j = pd.Index(z_rep['ID']).get_loc(l)
                costs[i][j] = 0
            except:
                print(f"Sensor {l} does not exit")
                break

        numOfSoln = 0
        maxSoln = 1

        rem_index_list = []
        M_list = []

        while(numOfSoln < maxSoln):
            solver = pywraplp.Solver('SolveAssignmentProblemMIP', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

            M = {}
            for i in range(n_pre):
                for j in range(n_rep):
                    M[i, j] = solver.BoolVar('M[%i,%i]' % (i, j))

            for i in range(n_pre):
                solver.Add(solver.Sum([M[i, j] for j in range(n_rep)]) <= 1)

            for j in range(n_rep):
                solver.Add(solver.Sum([M[i, j] for i in range(n_pre)]) <= 1)

            for d in decoyIDDs:
                try:
                    j = pd.Index(z_rep['ID']).get_loc(d)
                    solver.Add(solver.Sum([M[i, j] for i in range(n_pre)]) == 0)
                except:
                    print(f"Sensor {d} does not exit")
                    break

            for l in fixedIDs:
                try:
                    i = pd.Index(z_pre['ID']).get_loc(l)
                    j = pd.Index(z_rep['ID']).get_loc(l)
                    solver.Add(M[i,j] == 1)
                except:
                    print(f"Sensor {l} does not exit")
                    break

            for i in range(n_pre):
                for j in range(n_rep):
                    if costs[i][j] > th_max[i]:
                        solver.Add(M[i,j] == 0)

            if numOfSoln > 0:
                for x_prev in M_list:
                    cond = []
                    totalX = sum(sum(x_prev))
                    for i in range(n_pre):
                        for j in range(n_rep):
                            cond.append(M[i,j] * x_prev[i,j])
                    solver.Add(solver.Sum(cond) <= totalX-1)

            objective_terms_loss = []
            objective_terms_assign = []

            for i in range(n_pre):
                for j in range(n_rep):
                    objective_terms_loss.append(costs[i][j] *M[i, j])
                    objective_terms_assign.append(2*th*M[i, j])

            solver.Minimize(solver.Sum(objective_terms_loss) - solver.Sum(objective_terms_assign))

            status = solver.Solve()

            total_rem_sen = 0
            total_cost = 0

            M_values = np.zeros((n_pre, n_rep), dtype= int)
            rem_index = []
            recoverData = pd.DataFrame([], columns = ['MS', 'rep_ID', 'rem_ID', 'Var'])

            if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
                numOfSoln += 1

                for i in range(n_pre):
                    org_Sensor = -1
                    for j in range(n_rep):
                        M_values[i,j] = int(M[i,j].solution_value())
                        if M[i, j].solution_value() > 0.5:
                            org_Sensor = j
                            total_rem_sen += 1
                            total_cost += costs[i][j]
                    rem_index.append(org_Sensor)

                rem_index_list.append(rem_index)
                M_list.append(M_values)

            else:
                print("Cant Solve!!!!!!")
                break

            optionsIndx = pd.DataFrame(np.array(rem_index_list))

            for i, c in enumerate (optionsIndx.columns):
                indexx = np.unique(optionsIndx[c])
                indexx = indexx[indexx > 0]

                if indexx.shape[0] == 0:
                    continue

                preSensor = z_pre['ID'].iloc[c]

                recoverData.loc[preSensor] = [0, 0, 0, preSensor]
                recoverData['rem_ID'].loc[preSensor] = preSensor
                recoverData['rep_ID'].loc[preSensor] = z_rep['ID'].iloc[indexx[0]]

                recoverData['MS'].loc[preSensor] = np.mean(z_rep['MS'].iloc[indexx].values)
                recoverData['Var'].loc[preSensor] = np.square(np.std(z_rep['MS'].iloc[indexx].values))+1

            recoverData = recoverData.dropna()

        return recoverData 



class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, numOfZ: int, hidden_size: int = 256, sequence_length: int = 10, feature_dim: int = 13):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.numOfZ = numOfZ
        self.memory = deque(maxlen=200000)
        # Increase batch size for more stable learning
        self.batch_size = 128
        self.steps = 0
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim

        # Adjust gamma for longer-term rewards
        self.gamma = 0.99
        self.epsilon = 1.0
        # Increase min epsilon for more exploration
        self.epsilon_min = 0.4
        # Slower decay for better exploration
        self.epsilon_decay = 0.95
        # Reduce learning rate for more stable learning
        self.learning_rate = 0.005

                # Add exploration parameters
        self.exploration_phase = 10000  # Steps to maintain high exploration
        self.random_action_prob = 0.4 # from 0.2 to 0.4 # Probability of taking completely random action
        self.exploration_noise = 0.3

        self.default_deception = 20.0
        self.default_sensor_selection = np.zeros(self.numOfZ, dtype=int)
        
        # Add experience replay parameters
        # self.memory_size = 200000
        self.min_memory_size = 10000  # Min experiences before training
        self.target_update_freq = 500  # Update target network every N steps
        self.train_freq = 10  # Train every N steps
        
        # Initialize models and optimizer - CORRECTED VERSION
        self.model = self.build_model(hidden_size)
        self.target_model = self.build_model(hidden_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)  # Remove loss parameter
        
        self.update_target_model()

    def get_default_action(self):
        """Return a safe default action when errors occur"""
        return {
        'deception_amount': np.array([self.default_deception]),
        'sensor_selection': self.default_sensor_selection.copy()
        }

    def build_model(self, hidden_size):
        model = tf.keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.feature_dim)),
            layers.BatchNormalization(),
            layers.Bidirectional(layers.LSTM(hidden_size, activation='relu', 
                                           return_sequences=True)),
            layers.Dropout(0.2),
            
            layers.Bidirectional(layers.LSTM(hidden_size, activation='relu', 
                                           return_sequences=True)),
            layers.Dropout(0.2),
            
            layers.Bidirectional(layers.LSTM(hidden_size, activation='relu', 
                                           return_sequences=False)),
            layers.Dropout(0.2),
            
            layers.Dense(hidden_size, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(hidden_size // 2, activation='relu'),
            
            layers.Dense(self.action_dim + 1)
        ])
        
        model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        try:
            deception_amount = float(action['deception_amount'].item())
            sensor_selection = np.array(action['sensor_selection']).flatten()
            
            # Ensure consistent shapes
            state = np.array(state, dtype=np.float32).reshape(10, 13)
            next_state = np.array(next_state, dtype=np.float32).reshape(10, 13)
            
            action_array = np.concatenate(([deception_amount], sensor_selection))
            self.memory.append((state, action_array, reward, next_state, done))
            
        except Exception as e:
            print(f"Error in remember: {str(e)}")
            print(f"State shape: {np.array(state).shape}")
            print(f"Action shape: {action_array.shape if 'action_array' in locals() else 'N/A'}")

    # 
    
    def act(self, state, training=True):
        try:
            self.steps += 1
            state = np.array(state).reshape(1, 10, 13)

            # Define possible deception amounts
            # possible_deceptions = np.arange(20, 85, 5) 
            possible_deceptions = np.arange(20, 80, 2)  # [20, 25, 30, ..., 75, 80]

            if training:
                # High exploration phase
                if self.steps < self.exploration_phase:
                    exploration_prob = 0.8
                else:
                    exploration_prob = self.epsilon
                    
                if np.random.rand() <= exploration_prob:
                    # Enhanced exploration strategy
                    if np.random.rand() < self.random_action_prob:
                        # Completely random action from discrete values
                        deception_amount = np.random.choice(possible_deceptions)
                        sensor_selection = np.random.randint(2, size=self.numOfZ)
                    else:
                        # Strategic exploration
                        if np.random.rand() < 0.2:  # 20% chance of extreme values
                            deception_amount = np.random.choice([20,50,80])  # Choose from low, mid, high
                        else:
                            # Random value from possible deceptions
                            deception_amount = np.random.choice(possible_deceptions)
                        
                        # More diverse sensor selection
                        base_selection = np.random.randint(2, size=self.numOfZ)
                        noise = np.random.normal(0, self.exploration_noise, size=self.numOfZ)
                        sensor_selection = (base_selection + noise > 0.5).astype(int)
                else:
                    # Exploitation with discrete values
                    q_values = self.model.predict(state, verbose=0)[0]
                    
                    # Convert continuous Q-value to discrete deception amount
                    raw_deception = q_values[0] * 80
                    # Find closest discrete value
                    deception_amount = possible_deceptions[
                        np.abs(possible_deceptions - raw_deception).argmin()
                    ]
                    
                    sensor_probs = self.sigmoid(q_values[1:self.numOfZ+1] + np.random.normal(0, 0.1, self.numOfZ))
                    sensor_selection = (sensor_probs > 0.5).astype(int)
            else:
                # Pure exploitation for evaluation
                q_values = self.model.predict(state, verbose=0)[0]
                
                # Convert continuous Q-value to discrete deception amount
                raw_deception = q_values[0] * 80
                # Find closest discrete value
                deception_amount = possible_deceptions[
                    np.abs(possible_deceptions - raw_deception).argmin()
                ]
                
                sensor_selection = (q_values[1:self.numOfZ+1] > 0.5).astype(int)
                
            return {
                'deception_amount': np.array([deception_amount]),
                'sensor_selection': sensor_selection
            }
        except Exception as e:
            print(f"Error in act: {e}")
            return self.get_default_action()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        try:
            minibatch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = [], [], [], [], []
            
            for state, action, reward, next_state, done in minibatch:
                states.append(state)
                # Clip action values to be within valid range
                clipped_action = np.clip(action, 0, self.action_dim)  # Add this line
                actions.append(clipped_action.astype(np.int32))
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                
            # Convert to numpy arrays with proper types
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.int32)
            rewards = np.array(rewards, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)
            
            with tf.GradientTape() as tape:
                current_q = self.model(states)
                next_q = self.target_model(next_states)
                
                # Ensure actions are within valid range
                next_actions = tf.clip_by_value(
                    tf.argmax(self.model(next_states), axis=1),
                    0,
                    self.action_dim - 1
                )
                next_q_values = tf.gather(next_q, next_actions, batch_dims=1)
                
                targets = rewards + (1 - dones) * self.gamma * next_q_values
                
                # Ensure actions are within valid range for gathering
                actions_for_gather = tf.clip_by_value(
                    tf.cast(actions, tf.int32),
                    0,
                    self.action_dim - 1
                )
                predicted_q_values = tf.gather(current_q, actions_for_gather, batch_dims=1)
                loss = tf.reduce_mean(tf.square(targets - predicted_q_values))
                
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            return float(loss.numpy())
            
        except Exception as e:
            print(f"Error in replay: {str(e)}")
            print(f"Action shape: {actions.shape if 'actions' in locals() else 'N/A'}")
            print(f"Action values: {actions if 'actions' in locals() else 'N/A'}")
            print(f"Current Q shape: {current_q.shape if 'current_q' in locals() else 'N/A'}")
            return None

    def update_target_model(self):
            """Update target network"""
            self.target_model.set_weights(self.model.get_weights())


class MetricsManager:
    def __init__(self):
        self.step_metrics = []
        self.episode_averages = {
            'success_rate': [], 'deviation': [], 'reward': [], 'deception_amount': []
        }
        os.makedirs('training_logs', exist_ok=True)
        os.makedirs('metrics_plots', exist_ok=True)



    def add_step_data(self, Deviation, totalAttackedSensors, successfulDetectionCount, successfulDetectionCount_avg, deception_amount):
        self.step_metrics.append({
            'state_deviation': Deviation,
            'successful_attack': (totalAttackedSensors-successfulDetectionCount),
            'successful_detection': successfulDetectionCount,
            'successful_detection_count': successfulDetectionCount_avg,
            'deception_amount': deception_amount

        })

       

    def save_episode_metrics(self, episode, total_reward, state_deviation, successful_attack, 
                           successful_detection, successful_detection_count, processed_attacks, 
                           success_count,detection,deception_amount,Current_step,Episode_ending):
        # Save to CSV
        episode_data = pd.DataFrame([{
            'episode': episode,
            'total_reward': total_reward,
            'state_deviation': state_deviation,
            'successful_attack': successful_attack,
            'successful_detection': successful_detection,
            'successful_detection_count': successful_detection_count,
            'processed_attacks': processed_attacks,
            'success_count': success_count,
            'detection': detection,
            'deception_amount': deception_amount,
            'Current_step': Current_step,
            'Episode_ending': Episode_ending
        }])
        
        filename = 'training_logs/episode_metrics.csv'
        if not os.path.exists(filename):
            episode_data.to_csv(filename, mode='w', index=False)
        else:
            episode_data.to_csv(filename, mode='a', header=False, index=False)

    def plot_metrics(self, episode_rewards, average_detection_rates, average_successful_attacks,
                    average_state_deviations, average_success_count, average_deception_amount):
        try:
            # Debug print statements
            print("Plotting metrics...")
            print(f"Length of metrics arrays:")
            print(f"Episode rewards: {len(episode_rewards)}")
            print(f"Detection rates: {len(average_detection_rates)}")
            print(f"Successful attacks: {len(average_successful_attacks)}")
            print(f"State deviations: {len(average_state_deviations)}")
            print(f"Success count: {len(average_success_count)}")
            print(f"Deception amount: {len(average_deception_amount)}")

            # Create figure
            plt.figure(figsize=(15, 10))
            
            # Plot all metrics
            plots = [
                (episode_rewards, 'Episode Rewards', 'Total Reward', 250),
                (average_detection_rates, 'Attack Detection Rates', 'Detection Rate', 250),
                (average_successful_attacks, 'Successful Attack Counts', 'Number of Successful Attacks', 20),
                (average_state_deviations, 'Average State Deviations', 'Average Deviation', 250),
                (average_success_count, 'Average Success Count', 'Success Count', 100),
                (average_deception_amount, 'Average Deception Amount', 'Deception Amount', 100)
            ]
            
            for idx, (data, title, ylabel, ylim) in enumerate(plots, 1):
                if len(data) > 0:  # Only plot if we have data
                    plt.subplot(3, 2, idx)
                    plt.plot(data)
                    plt.title(title)
                    plt.xlabel('Episode')
                    plt.ylim(0, ylim)
                    plt.ylabel(ylabel)
                else:
                    print(f"Warning: No data for {title}")

            plt.tight_layout()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'metrics_plots/training_results_{timestamp}.png')
            plt.close()
            print("Metrics plot saved successfully!")

        except Exception as e:
            print(f"Error in plot_metrics: {str(e)}")
            traceback.print_exc()


class IEEE14BusPowerSystemEnv(gym.Env):
    def __init__(self, data_loader: DataLoader, attack_scenario_manager: AttackScenarioManager,
                 state_estimator: StateEstimator, historical_manager: HistoricalDataManager,
                 data_processor: DataProcessor, defense_strategy: DefenseStrategy, attack_indices,  episode_metrics: MetricsManager):
        super(IEEE14BusPowerSystemEnv, self).__init__()

        # Initialize components
        self.data_loader = data_loader
        self.attack_scenario_manager = attack_scenario_manager
        self.state_estimator = state_estimator
        self.historical_manager = historical_manager
        self.data_processor = data_processor
        self.defense_strategy = defense_strategy
        self.defense_strategy.historical_manager = historical_manager  # Add this line
        self.attack_indices = attack_indices
        self.attack_index_pointer = 0  # Pointer to current position in attack_indices
        self.current_attack_index = self.attack_indices[0]  # Start with first attack index
        self.seed = data_loader.get_copy('current_seed')
        self.episode_metrics = episode_metrics
        
        # Initialize state-related attributes
        self.numOfZ = data_loader.numOfZ
        self.state = None
        self.consideredFixedIDs = []
        self.consideredDecoyIDs = []
        
        # Initialize other attributes
        self.attackerLevelList = [1]
        self.recoveryType = 1
        self.addDecoy = False
        self.attackCat = 'FDI'
        self.attackertype = 0
        self.current_step = 0
        self.max_steps = 25  # Adjust as needed
        self.runfilter = False
        self.runImpute = False

        X_train, Y_train, X_test, Y_test = self.data_loader.get_train_test_data()
        self.sequence_length = X_train.shape[1]  # Time steps in sequence
        self.feature_dim = X_train.shape[2]      # Number of features per time step

        # System parameters
        self.numOfBuses = data_loader.numOfBuses
        self.numOfLines = data_loader.numOfLines
        self.numOfStates = self.numOfBuses
        self.attacked_Bus = data_loader.attacked_Bus
        self.localize = data_loader.localize
        self.attackFreq = data_loader.attackFreq if hasattr(data_loader, 'attackFreq') else 1
        self.server   = []
        self.clusters = []
        self.hubs     = []
        self.IDbank   = []
        self.shuffle_percent = 0
        self.current_seed = self.seed if self.seed is not None else random.randint(0, 1000)
        
        # Load necessary data
        self.Z_org = self.data_loader.Z_org
        self.H_mat = self.data_loader.H_mat
        self.Attack_Data = self.data_loader.Attack_Data
        self.W_list = self.data_loader.W_list
        
        self.IDbank = self.data_loader.update_IDs(seed=self.current_seed, shuffle_percent=0)
        self.shuffle_partial = DataLoader.shuffle_partial
        
        self.previous_deception_amount = 0
        self.success_count = 0
        self.detection_count = 0

        # self.deviation=0
        # self.successfulAttack=0
        # self.successfulDetectionCount=0
        # self.successfulDetectionCount_avg=0
        # self.deception_amount=0


        self.AttackReturn, self.totalAttackedSensors, self.successfulDetectionCount, self.successfulDetectionCount_avg, self.Deviation = None, 0, 0, 0, 0
        
        # Update observation space to use training data dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.sequence_length, self.feature_dim),
                dtype=np.float32
        )   
        
        # # Update observation space to match DQN agent expectations
        # self.observation_space = spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(10, 13),  # Update to match DQN agent's expected shape
        #     dtype=np.float32
        # )
        
        # Define action space
        self.action_space = spaces.Dict({
            'deception_amount': spaces.Box(low=0, high=80, shape=(1,), dtype=np.float32),
            'sensor_selection': spaces.MultiBinary(self.numOfZ)
        })
        
        
        
        # Initialize state
        self.reset()

    def _convert_state_to_array(self, state):
        """Convert state dictionary to numpy array using training data dimensions"""
        try:
            if isinstance(state, dict) and 'sensor_readings' in state:
                sensor_readings = state['sensor_readings']
                
                # Convert to numpy array if needed
                if not isinstance(sensor_readings, np.ndarray):
                    sensor_readings = np.array(sensor_readings, dtype=np.float32)
                
                # Reshape to match training data dimensions
                if sensor_readings.ndim == 1:
                    sensor_readings = sensor_readings.reshape(1, -1)
                
                # Pad or truncate to match expected dimensions
                if sensor_readings.shape[0] < self.sequence_length:
                    pad_width = ((self.sequence_length - sensor_readings.shape[0], 0), 
                               (0, max(0, self.feature_dim - sensor_readings.shape[1])))
                    sensor_readings = np.pad(sensor_readings, pad_width, mode='constant')
                else:
                    sensor_readings = sensor_readings[-self.sequence_length:]
                
                if sensor_readings.shape[1] != self.feature_dim:
                    if sensor_readings.shape[1] < self.feature_dim:
                        pad_width = ((0, 0), (0, self.feature_dim - sensor_readings.shape[1]))
                        sensor_readings = np.pad(sensor_readings, pad_width, mode='constant')
                    else:
                        sensor_readings = sensor_readings[:, :self.feature_dim]
                
                return sensor_readings.astype(np.float32)
            else:
                print(f"Invalid state format: {type(state)}")
                return np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)
                
        except Exception as e:
            print(f"Error in _convert_state_to_array: {str(e)}")
            traceback.print_exc()
            return np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)

    def reset(self):
        """Reset environment to initial state"""

        print("###############Resetting IEEE14BusPowerSystemEnv.....###########")
        self.current_step = 0
        self.total_reward = 0
        self.attack_index_pointer = 0  # Reset attack index pointer
        self.current_attack_index = self.attack_indices[0]  # Reset to first attack index
        self.consideredFixedIDs = []
        self.consideredDecoyIDs = []
        self.reward = 0
        
        self.AttackReturn, self.successCount, self.successCount_avg, self.Deviation = None, 0, 0, 0
        
        # Initialize state with original sensor readings
        initial_state = {
            'sensor_readings': self.Z_org.copy()
        }
        
        # Convert to array format
        self.state = self._convert_state_to_array(initial_state)
        
        return self.state

    def step(self, action):
        print("############### .....Starting Step Function of IEEE14BusPowerSystemEnv.....###########")
        try:
            self.current_step += 1
            total_reward = 0  # Initialize reward
            
            # Get deception amount from action
            deception_amount = action['deception_amount'][0]
            print(f"deception_amount : {deception_amount}")
            self.shuffle_percent = int(deception_amount)
            
            # Randomly select one attack index
            self.current_attack = np.random.choice(self.attack_indices)
            print(f"Selected attack index: {self.current_attack}")
            
            # Update IDbank based on shuffle percentage
            self.IDbank = self.data_loader.update_IDs(seed=self.current_seed, shuffle_percent=self.shuffle_percent)
            
            # Evaluate defense for the single selected attack
            self.AttackReturn, self.totalAttackedSensors, self.successfulDetectionCount, self.successfulDetectionCount_avg, self.Deviation = self.defense_strategy.defenseEval(
                self.Attack_Data,
                self.current_attack,
                self.IDbank,
                self.attackerLevelList,
                self.recoveryType,
                self.addDecoy,
                self.consideredFixedIDs,
                self.consideredDecoyIDs,
                self.attackCat,
                self.attackertype,
                verbose_=True
            )
            
            # Calculate reward if AttackReturn is valid
            if self.AttackReturn is not None:
                step_reward = self._calculate_reward(
                    state_deviation=self.Deviation,
                    successful_attack=(self.totalAttackedSensors-self.successfulDetectionCount),
                    successful_detection=self.successfulDetectionCount,
                    successful_detection_count=self.successfulDetectionCount_avg,
                    deception_amount=deception_amount
                )
                total_reward += step_reward
            else:
                total_reward -= 100
            
            # Prepare next state
            next_state = self._convert_state_to_array({
                'sensor_readings': self.AttackReturn.get('StatesRec', np.zeros((10, 13), dtype=np.float32))
            })
            
            # Check if episode is done
            done = bool(self.current_step >= self.max_steps)
            
            # Prepare info dictionary
            info = {
                'state_deviation': self.Deviation,
                'successful_attack': (self.totalAttackedSensors-self.successfulDetectionCount),
                'successful_detection': self.successfulDetectionCount,
                'successful_detection_count': self.successfulDetectionCount_avg,
                'processed_attacks': 1,  # Always 1 since we're processing one attack per step
                'success_count': self.successfulDetectionCount,
                'detection': self.successfulDetectionCount,
                'deception_amount': deception_amount,
                'current_attack_index': self.current_attack
            }
            
            self.state = next_state
            return next_state, total_reward, done, info
                
        except Exception as e:
            print(f"Error in step function: {str(e)}")
            traceback.print_exc()
            return self.state, -100, False, {
                'state_deviation': 0,
                'successful_attack': 0,
                'successful_detection': 0,
                'successful_detection_count': 0,
                'processed_attacks': 0,
                'success_count': 0,
                'detection': 0,
                'deception_amount': 0,
                'current_attack_index': -1
            }

    def _calculate_reward(self, state_deviation, successful_attack, successful_detection, successful_detection_count, deception_amount):
        """
        Calculate the reward based on the action results
        """
        reward = 0
        
        # Penalize high state deviation
        reward -= state_deviation * 0.1
        
        # Penalize successful attacks
        if successful_attack:
            reward -= 30*successful_attack
        
        # Reward detection
        if successful_detection:
            reward += 30*successful_detection_count
        
        # Penalize excessive deception
        reward -= (deception_amount) *10
        
        # Reward successful defense
        if successful_detection_count > 0:
            reward += 20
        
        return reward



    
def main():
    try:
        # Load data
        data_loader = DataLoader()
        metrics_manager = MetricsManager()          # data_loader.load_data()
        data_loader.load_system_data()
        data_loader.load_measurement_data()
        # data_loader.load_H_mat()
        # Initialize components
        state_estimator = StateEstimator(data_loader)
        
        # Get the required parameters from data_loader
        max_history = 100  # You can adjust this value as needed
        numOfBuses = data_loader.numOfBuses

        H_mat = data_loader.load_topology_data()
        Z_mat = data_loader.get_copy('Z_mat')
        Z_org = data_loader.get_copy('Z_org')


        # Ensure H_mat is properly loaded
        if data_loader.H_mat is None:
            raise ValueError("H_mat is None. Please ensure it is correctly loaded before proceeding.")

        # Initialize components
        historical_manager = HistoricalDataManager(max_history, numOfBuses, H_mat)
        
        data_processor = DataProcessor(data_loader)
        defense_strategy = DefenseStrategy(data_loader, historical_manager, data_processor)

        # Initialize the attack scenario manager
        max_attacked_buses = 5  # Adjust this value as needed
        attack_scenario_manager = AttackScenarioManager(numOfBuses, max_attacked_buses, data_loader)

        # Generate shuffled attack indices
        numOfAttacks = 129
        attackIndxxx = np.arange(0, numOfAttacks)
        np.random.shuffle(attackIndxxx)

        # Initialize the environment
        env = IEEE14BusPowerSystemEnv(
            data_loader=data_loader,
            attack_scenario_manager=attack_scenario_manager,
            state_estimator=state_estimator,
            historical_manager=historical_manager,
            data_processor=data_processor,
            defense_strategy=defense_strategy,
            attack_indices=attackIndxxx,
            episode_metrics=metrics_manager
        )

        # Calculate state and action dimensions
        state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]  # 10 * 13
        action_dim = env.numOfZ  # sensor selection dimensions
        
        print(f"state_dim: {state_dim}, action_dim: {action_dim}, numOfZ: {env.numOfZ}")
        
        # Initialize agent
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, numOfZ=env.numOfZ)

        # Initialize metrics lists
        episode_wise_rewards = []
        episode_wise_detection_rates = []
        episode_wise_successful_attacks = []
        episode_wise_state_deviations = []
        episode_wise_success_count = []
        episode_wise_deception_amount = []

        # Training loop
        episodes = 100
        completed_episodes = 0
        for episode in range(episodes):
            print(f"\nStarting Episode {episode}")
            state = env.reset()
            total_reward = 0
            episode_metrics = []
            step_count = 0
            done = False

            while not done and step_count < env.max_steps:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                # Debug prints for metrics
                print("\nStep Metrics:")
                print(f"state_deviation: {info['state_deviation']}")
                print(f"successful_attack: {info['successful_attack']}")
                print(f"successful_detection: {info['successful_detection']}")
                print(f"successful_detection_count: {info['successful_detection_count']}")
                print(f"processed_attacks: {info['processed_attacks']}")
                print(f"success_count: {info['success_count']}")
                print(f"detection: {info['detection']}")
                print(f"deception_amount: {info['deception_amount']}")
                print(f"Current step: {step_count}/{env.max_steps}")
                print(f"Episode ending: {done}")

                episode_metrics.append({
                    'state_deviation': info['state_deviation'],
                    'successful_attack': info['successful_attack'],
                    'successful_detection': info['successful_detection'],
                    'successful_detection_count': info['successful_detection_count'],
                    'processed_attacks': info['processed_attacks'],
                    'success_count': info['success_count'],
                    'detection': info['detection'],
                    'deception_amount': info['deception_amount'],
                    'Current_step': step_count,
                    'Episode_ending': done

                })

                metrics_manager.save_episode_metrics(
                episode=episode,
                total_reward=reward,
                state_deviation=info['state_deviation'],
                successful_attack=info['successful_attack'],
                successful_detection=info['successful_detection'],
                successful_detection_count=info['successful_detection_count'],
                processed_attacks=info['processed_attacks'],
                success_count=info['success_count'],
                detection=info['detection'],
                deception_amount=info['deception_amount'],
                Current_step=step_count,
                Episode_ending=done
                )

                total_reward += reward
                step_count += 1
                
                agent.remember(state, action, reward, next_state, done)
                        
                if len(agent.memory) > agent.batch_size:
                    agent.replay()
                
                state = next_state

                if step_count >= env.max_steps:
                    print(f"Episode {episode} completed all {env.max_steps} steps")
                    done = True

            # After episode ends, calculate proper averages
            avg_success_count = np.mean([m['successful_detection_count'] for m in episode_metrics])
            print(f"\nEpisode {episode} Summary:")
            print(f"Average Success Count:{avg_success_count}")
            print(f"Total Steps: {step_count}")
            
            # Append to metrics lists
            episode_wise_rewards.append(total_reward)
            episode_wise_detection_rates.append(np.mean([m['successful_detection'] for m in episode_metrics]))
            episode_wise_successful_attacks.append(np.mean([m['successful_attack'] for m in episode_metrics]))
            episode_wise_state_deviations.append(np.mean([m['state_deviation'] for m in episode_metrics]))
            episode_wise_success_count.append(avg_success_count)
            episode_wise_deception_amount.append(np.mean([m['deception_amount'] for m in episode_metrics]))

            # Save detailed metrics
        # After all episodes, plot metrics


            completed_episodes += 1
            print(f"Completed {completed_episodes}/{episodes} episodes")

            if episode % 10 == 0 or episode == episodes - 1:
                agent.update_target_model()
                print(f"Episode: {episode}, Total Reward: {total_reward}")

        print(f"\nTraining Summary:")
        print(f"Completed Episodes: {completed_episodes}")
        print(f"Average Steps per Episode: {np.mean([m['step_count'] for m in metrics_manager.step_metrics])}")
        

        # After all episodes, plot metrics
        metrics_manager.plot_metrics(
            episode_wise_rewards,             
            episode_wise_detection_rates,     
            episode_wise_successful_attacks,  
            episode_wise_state_deviations,   
            episode_wise_success_count,      
            episode_wise_deception_amount    
        )

    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
    
if __name__ == "__main__":

    main()