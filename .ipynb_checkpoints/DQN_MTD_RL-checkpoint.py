import numpy as np
import pandas as pd
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



class DataLoader:
    """Handles all data loading and preprocessing"""
    def __init__(self, numOfBuses: int = 14, numOfLines: int = 20):
        self.numOfBuses = numOfBuses
        self.numOfLines = numOfLines
        self.numOfZ = numOfBuses + 2 * numOfLines + 1  # This should be 55 based on your data
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
        self.Attack_Data = None
        self.scaler = StandardScaler()
        self.th = 5
        self.fixedIDDs = []
        self.runfilter = False
        self.runImpute = False
        self.Z_est_init = []
        self.Noisy_Indx_actu = []
        self.Zpair = []
        self.numOfAttacks = 0
        self.meanZ = 0
        self.bus_data_df = None
        self.line_data_df = None
        self.topo_mat = None
        self.line_data = None
        self.Z_msr_org = None
        self.bus_data = None
        self.attackerLevelList=[1, 2, 3]
        self.attacked_Bus = 5
        self.lookback = 20
        self.Zmat = None
        self.th = 5
        self.bus_data_df = None
        self.line_data_df = None
    
    def load_data(self):
        # ... (other data loading code)
        
        # Update numOfZ based on actual data length
        if self.numOfZ != self.Z_mat.shape[0]:
            print(f"Warning: numOfZ ({self.numOfZ}) doesn't match the actual data length ({self.Z_mat.shape[0]}). Updating numOfZ.")
            self.numOfZ = self.Z_mat.shape[0]
        
        # Ensure W_list has the correct length
        if len(self.W_list) != self.numOfZ:
            print(f"Adjusting W_list length from {len(self.W_list)} to {self.numOfZ}")
            self.W_list = np.ones(self.numOfZ)

        # ... (rest of the method)

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
            print(f"H_org : {self.H_org}")

            self.H_mat = self.H_org.copy()
            print(f"H_mat : {self.H_mat}")

            self.W_list = np.ones(self.numOfZ)  # Initialize with ones for all measurements
            print(f"W_list : {self.W_list}")

            
            # Load measurement data
            self.load_measurement_data()
            self.load_attack_data(self.attacked_Bus)
            
            return True
        except Exception as e:
            print(f"Error loading system data: {str(e)}")
            return False

    def load_topology_data(self):
        file_name = f"Bus Data//IEEE_{self.numOfBuses}.xlsx"
        # Load or generate topology data
        try:
            self.topo_mat = pd.read_excel(file_name, sheet_name="Topology Matrix")
            self.line_data = pd.read_excel(file_name, sheet_name="Line Data")
        except:
            print(f"Warning: Could not load topology data from {file_name}. Initializing with zeros.")
            self.topo_mat = pd.DataFrame(np.zeros((self.numOfLines, self.numOfBuses)))
            self.line_data = pd.DataFrame(np.zeros((self.numOfLines, 4)))
        
        self.Topo = self.line_data.values.astype(int)
        # Implement H_matrix creation based on system topology
        if self.topo_mat is None:
            raise ValueError("Topology matrix is not initialized. Call load_topology_data() first.")
        H_org = self.topo_mat.values.copy()
        # Add logic to populate H_mat based on system topology
        return H_org

    def load_measurement_data(self):
        try:
            self.Z_msr_org = pd.read_excel(self.file_name, sheet_name="Measurement Data")
            print(f"Z_msr_org shape: {self.Z_msr_org.shape}")
            self.bus_data = pd.read_excel(self.file_name, sheet_name="Bus Data")
            print(f"bus_data shape: {self.bus_data.shape}")
        except Exception as e:
            print(f"Error loading measurement data: {str(e)}")
            self.Z_msr_org = pd.DataFrame(np.zeros((1, self.numOfZ+1)), columns=['Data'])
            self.bus_data = pd.DataFrame(np.zeros((1, self.numOfBuses)))
        
        # Set up measurement data
        self.Z_msr_org.insert(0, 'ID', list(self.Z_msr_org.index.values))
        print(f"Z_msr_org after inserting ID: {self.Z_msr_org.head()}")
        
        # Use the actual length of Z_msr_org instead of self.numOfZ+1
        actual_length = len(self.Z_msr_org)
        self.Z_msr_org.insert(1, 'Reported', [1] * actual_length)
        print(f"Z_msr_org after inserting Reported: {self.Z_msr_org.head()}")
        
        self.Z_org = self.Z_msr_org.values.copy()
        self.Z_mat = self.Z_msr_org.values.copy()
        print(f"Z_mat shape: {self.Z_mat.shape}")
        print(f"Z_mat: {self.Z_mat}")

        # Update numOfZ if it doesn't match the actual data
        if actual_length != self.numOfZ + 1:
            print(f"Warning: numOfZ ({self.numOfZ}) doesn't match the actual data length ({actual_length - 1}). Updating numOfZ.")
            self.numOfZ = actual_length - 1

    def load_attack_data(self, attacked_Bus: int):
        file_Name_ = f"Attack_Space_{self.numOfBuses}_{self.numOfLines}_{attacked_Bus}.csv"
        try:
            raw_data = np.genfromtxt("Attack Data//"+file_Name_, delimiter=',')
            print(f"Raw data shape: {raw_data.shape}")  # Debugging line
            
            # Reshape the data into 129 scenarios, each with 55 measurements
            num_scenarios = 129
            measurements_per_scenario = 55
            self.Attack_Data = np.zeros((num_scenarios, measurements_per_scenario, 3))
            print(f"Initialized Attack_Data shape: {self.Attack_Data.shape}")  # Debugging line
            
            for i in range(num_scenarios):
                scenario_data = raw_data[i*54:(i+1)*54]
                # Filter out rows where the middle column is 0
                filtered_data = scenario_data[scenario_data[:, 1] != 0]
                # Ensure we have exactly 55 measurements (pad with zeros if necessary)
                padded_data = np.zeros((measurements_per_scenario, 3))
                padded_data[:len(filtered_data)] = filtered_data
                self.Attack_Data[i] = padded_data
            
            print(f"Processed Attack_Data shape: {self.Attack_Data.shape}")  # Debugging line
            
        except Exception as e:
            print(f"Error loading attack data: {str(e)}")
            self.Attack_Data = np.zeros((num_scenarios, measurements_per_scenario, 3))
        
        # Add noise if localization is enabled
        if self.localize:
            self._add_noise_to_attack_data()
        
        self.numOfAttacks = num_scenarios
        if self.Z_msr_org is not None:
            self.meanZ = abs(self.Z_msr_org['Data']).mean()
        else:
            print("Warning: Z_msr_org is not initialized. Setting meanZ to 0.")
            self.meanZ = 0

    def _add_noise_to_attack_data(self):
        """Add noise to attack data for localization"""
        Noise_Data = self.Attack_Data.copy()
        # Generate noise matching the shape of the third "column"
        noise = np.random.randint(-20, 20, size=(Noise_Data.shape[0], Noise_Data.shape[1]))
        Noise_Data[:,:,2] = noise
        Noise_Data[Noise_Data[:,:,1] == 0, 2] = 0
        np.random.shuffle(Noise_Data[:,:,1:])
        self.Attack_Data = np.concatenate((self.Attack_Data, Noise_Data), axis=0)

    def filterData(self, Z_processed: np.ndarray, 
                   predictions: pd.DataFrame, 
                   threshold: float = 5.0,
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
        self.numOfBuses = 14  # Set this to the correct number of buses in your system
        
        # Initialize W_list
        self.W_list = self.load_W_list()  # You might need to implement this method
        
        if self.W_list is None:
            # If W_list is not available, initialize it with a default value
            self.W_list = np.ones(self.numOfZ)
        
        # Ensure W_list has the correct length
        if len(self.W_list) != self.numOfZ:
            print(f"Adjusting W_list length from {len(self.W_list)} to {self.numOfZ}")
            self.W_list = np.ones(self.numOfZ)
        
        # ... (rest of the method)
    
    def load_W_list(self):
        # Implement this method to load W_list from your data source
        # Return the loaded W_list
        pass
        # ... (rest of the method)

class AttackScenarioManager:
    """Handles attack scenarios and tracking"""
    def __init__(self, numOfBuses: int, max_attacked_buses: int, data_loader: DataLoader):
        self.data_loader = data_loader
        self.numOfBuses = self.data_loader.numOfBuses
        self.max_attacked_buses = max_attacked_buses
        self.Attack_Data = self.data_loader.get_copy('Attack_Data')
        self.current_attack = 0

    def get_attack_scenario(self, attack_index: int) -> np.ndarray:
        """Get specific attack scenario"""
        return self.Attack_Data[attack_index].copy()

    def identify_attacked_sensors(self, attack_scenario: np.ndarray) -> List[int]:
        """Identify attacked sensors in scenario"""
        return np.where(attack_scenario[:, 2] != 0)[0].tolist()

class StateEstimator:

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.H_mat = self.data_loader.get_copy('H_mat')
        self.W_list = self.data_loader.get_copy('W_list')
        self.Threshold_min = 3.0
        self.Threshold_max = 10.0
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
#     print("***********************************************")
        import numpy as np
        from numpy.random import seed
        from numpy.random import randn
        from numpy import mean
        from numpy import std

        if Verbose == "True":
            print("\n **************  State Estimation ************* \n")

        doneFlag = 0
        fullRank = True
        Threshold = Threshold_min
        Threshold_step = 2
        numberOfMeasurements = H_mat.shape[0]
        numberOfStates = H_mat.shape[1]

        ############# starting State Estimation ###############
        #Z_mat = Z_mat[Z_mat[:, 0].argsort(kind='mergesort')]
        Z_mat_Copy = Z_mat.copy()
        W_list = np.array(W_list)

        # Space holder for Z_est
        #Z_est = np.zeros(numberOfMeasurements)
        #list_of_removed_sensors = []

        ################# Starting the loop ###################################
        while(doneFlag == 0):
            #list_of_removed_sensors = []
            #considering only the taken measurements

            consideredIndx = np.where(Z_mat[:,1] == 1)[0]
    #         print("Shape of Z_mat", Z_mat.shape)

    #         print("Value of Z_mat", Z_mat)

            # Ensure consideredIndx doesn't exceed the size of W_list
            consideredIndx = consideredIndx[consideredIndx < len(W_list)]

            if len(consideredIndx) == 0:
                print("Warning: No valid measurements found.")
                return None, None, Z_mat, None, None, False, None

            Z_msr = Z_mat[consideredIndx][:,2]

            # considering only the corresponding columns in H
            H_msr = H_mat[consideredIndx]

            # Measurement Covarriance Residual Matrix
            # Check if W_list is a scalar or 0-dimensional array
            if np.isscalar(W_list) or W_list.ndim == 0:
                R_msr = np.eye(len(consideredIndx)) * W_list
            else:
                # Ensure we're not indexing out of bounds
                R_msr = np.diag(W_list[consideredIndx[consideredIndx < len(W_list)]])
            #print(R_msr)
            R_inv = np.linalg.inv(R_msr)
            #print(R_inv)
            #Chekcing rank of H
            Rank = np.linalg.matrix_rank(H_msr) if H_msr.shape[0] > 0 else 0

            if Verbose == "True":
                print("Current Rank", Rank)
                print("H_msr Shape: ", H_msr.shape)
                print("Z_msr Shape: ", Z_msr.shape)

            ###### H is full rank --> Start Estimating
            if Rank == numberOfStates:

                #Estimating the states using WMSE estimator
                inv__Ht_Rinv_H__Ht = np.linalg.inv(H_msr.T@R_inv@H_msr)@H_msr.T
                States = inv__Ht_Rinv_H__Ht@R_inv@Z_msr

                # Ω = R − H. (H_T.R−1.H)−1.HT
                # Omega is a residual covarience matrix

                Omega_mat = R_msr - (H_msr@inv__Ht_Rinv_H__Ht)

                #print("Check :\n", R_msr - (H_msr@np.linalg.inv(H_msr.T@R_inv@H_msr)@H_msr.T))
                #print(f"R_msr: {R_msr} \n Shape {R_msr.shape}")
                #print(f"Omega_mat: {Omega_mat} \n Shape {Omega_mat.shape}")

                # Estimating the measurement from the estimated states

                Z_est = H_mat@States


                if Verbose == "True":
                    print("\n Initital Z_m: \n"+str(Z_mat))
                    print("\n Sates: \n"+str(States))
                    print("\n Z_est: \n"+str(Z_est))
                    print("Calling Noise.. CheckNoise...")

                #####################  Checking for Bad Data ##################################
                # Calculating the Noise
                M_Noise, P_Noise, doneFlag = self.CheckNoiseCor(
                    Z_est, Z_mat, Omega_mat, R_msr, Threshold,  fullRank, Corr,  Verbose)

            # H is not a full rank matrix ............. abort estimation
            else:
                if Threshold < Threshold_max:
                    Threshold += Threshold_step
                    Z_mat = Z_mat_Copy.copy()
                    if Verbose == "True": print(f"Relaxing the threshold to {Threshold}")
                else:
                    doneFlag = -1 #system unobservable
                    if Verbose == "True": print(f"\n\n\nSystem Unobservable !, Rank = {Rank}")
                    #####  Returning ##############
                    fullRank = False

                    Z_est = np.zeros(numberOfMeasurements)
                    States = np.zeros(numberOfStates)
                    M_Noise = np.zeros(numberOfMeasurements)
        ##############################################################################
        Noisy_Indx = np.where(Z_mat[:,1] == -1)[0]

        return States, Z_est, Z_mat, M_Noise, Noisy_Indx, fullRank, Threshold


    def CheckNoiseCor (self, Z_est, Z_mat, Omega_mat, R_msr, Threshold,  fullRank, Corr, Verbose):

        import math
        import numpy as np

        if fullRank != True:
        #         if Verbose == "True":
        #             print("System Unobservable!"
            return None, None, Z_mat[:,0]

    #     print ("Z_mat from SE", Z_mat )
        Z_msr = Z_mat[Z_mat[:, 1] == 1][:,2].copy()
    #     print ("Z_msr from SE", Z_msr )

        ####################################################   Here -------------------->
        '''boolean index did not match indexed array along dimension 0; dimension is 53
        but corresponding boolean dimension is 55'''

        if Verbose == "True":
            print("Starting BDD")
            print("Z_est: ", Z_est.shape)
            print("Z_msr: inside noise checking", Z_msr.shape)
            #print("Z_mat[:, 1] == 1", Z_mat[:, 1] == 1)
            #print(Z_mat)

        Z_est_msr = Z_est[Z_mat[:, 1] == 1]

        # Calculating the measurement error

        M_Noise = (Z_msr - Z_est_msr)
    #     print ("M_Noise :", M_Noise)

        M_Noise_norm = M_Noise.copy()

        # Calculating the normalized residuals
        for index, _ in enumerate(M_Noise):
            if index == 0: continue
            try:
                M_Noise_norm [index] = np.absolute(M_Noise [index])/math.sqrt(Omega_mat[index, index])
            except:
                M_Noise_norm [index] = 0
                if Verbose == "True":
                    print("index: ", index, np.absolute(M_Noise [index]))
                    print(f" Value Error, Expected postive, Got {Omega_mat[index, index]}")

    #     Noise_mat_actual = np.zeros(Z_mat.shape[0])
    #     Noise_mat_actual[Z_mat[:,1] == 1] = M_Noise
    #     Noise_mat_norm = np.zeros(Z_mat.shape[0])
    #     Noise_mat_norm[Z_mat[:,1] == 1] = M_Noise_norm

    #     print(M_Noise.shape)

        Noise_mat_actual = M_Noise.copy()
        Noise_mat_norm = M_Noise_norm.copy()

    #     print("Value of Normalized Noise", M_Noise_norm)

    #     print("Highest value of Noise_mat_norm : ", Noise_mat_norm.max() )

        active_idx = np.where(Z_mat[:,1] == 1)[0]

    #     print ("Active Index", active_idx )

        # Checking for Noisy data
        if np.max(Noise_mat_norm) > Threshold:
            tIndx = np.argmax(Noise_mat_norm)
            if Verbose == "True":
                print(f"targetedIndx in cut: {tIndx}--> Value : {Noise_mat_norm[tIndx]}")
                print("Updating Z_mat...")
                print("Before: ", Z_mat[tIndx])
            #print("R_msr: ", R_msr.shape, Omega_mat.shape)
            if Corr == True:
                correction_value= R_msr[tIndx,tIndx]/Omega_mat[tIndx,tIndx]*M_Noise[tIndx]
                print("Correction Value: ", correction_value )
                print("value of active_idx[tIndx]: ", active_idx[tIndx])
                print("Before: ", Z_mat[active_idx[tIndx],2])
                Z_mat[active_idx[tIndx], 2] = Z_mat[active_idx[tIndx], 2] - correction_value
                print("After: ", Z_mat[active_idx[tIndx],2])
            else:
                Z_mat[active_idx[tIndx], 1] = -1

            doneFlag = 0

            if Verbose == "True":
                print("After: ", Z_mat[tIndx])
        else:
            if Verbose == "True": print("No Bad Data Detected....")
            doneFlag = 1

            Noise_mat_actual = np.zeros(Z_mat.shape[0])
            Noise_mat_actual[Z_mat[:,1] == 1] = M_Noise.copy()
            Noise_mat_norm = np.zeros(Z_mat.shape[0])
            Noise_mat_norm[Z_mat[:,1] == 1] = M_Noise_norm.copy()
        ##############################################

        return Noise_mat_actual, Noise_mat_norm, doneFlag
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

    def performanceAnalysis(self, data, model_type, degree=2):
        if len(data) == 0:
            print("Warning: stateHist is empty. Returning zeros.")
            return np.zeros((self.numOfBuses - 1, 1))  # Return a 2D array

        lookBack = len(data)
        stateHist = np.array(data)

        if model_type == "linearRegression":
            trainX = np.array([i + 1 for i in range(lookBack)]).reshape((-1, 1))
            testX  = np.array([lookBack + 1]).reshape((-1, 1))
            model  = LinearRegression().fit(trainX, stateHist)
            predictedState = model.predict(testX)

        elif model_type == "polynomialRegression":
            trainX = np.array([i + 1 for i in range(lookBack)]).reshape((-1, 1))
            testX  = np.array([lookBack + 1]).reshape((-1, 1))
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            trainX_poly = poly_features.fit_transform(trainX)
            testX_poly = poly_features.transform(testX)
            model = LinearRegression().fit(trainX_poly, stateHist)
            predictedState = model.predict(testX_poly)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Ensure the output is 2D with shape (numOfBuses - 1, 1)
        predictedState = predictedState.reshape(self.numOfBuses - 1, 1)

        return predictedState

    def update_stateHist(self, statePred):
        if statePred.ndim == 1:
            statePred = statePred.reshape(1, -1)
        
        if not self.is_scaler_fitted:
            self.scaler.fit(statePred)
            self.is_scaler_fitted = True
        else:
            # Update the scaler with the new data
            self.scaler.partial_fit(statePred)
        
        scaled_state = self.scaler.transform(statePred)
        self.stateHist.append(scaled_state.flatten())

    def get_scaled_state(self, state):
        if not self.is_scaler_fitted:
            raise ValueError("Scaler is not fitted yet. Update state history first.")
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return self.scaler.transform(state)

    def pred_msr(self, statePre):
        Zpre = pd.DataFrame(index=range(self.numOfZ), columns=['ID', 'MS'])
        Zpre['ID'] = range(self.numOfZ)
        Zpre['MS'] = (self.H_mat @ statePre).flatten()
        return Zpre


class DataProcessor:
    """Enhanced data processing pipeline"""
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.numOfZ = self.data_loader.numOfZ
        self.H_mat = self.data_loader.get_copy('H_mat')
        self.state_estimator = StateEstimator(self.data_loader)  # Pass data_loader instead of individual parameters
        self.prediction_model = "polynomialRegression"
        self.lookback = self.data_loader.get_copy('lookback')
        self.scaler = StandardScaler()
        self.state_history = deque(maxlen=self.lookback)

    def process_measurements(self, Z_mat: np.ndarray) -> Dict:
        """Complete measurement processing"""
        # 1. Validation
        if not self.validate_inputs(Z_mat):
            return None
            
        # 2. State Estimation
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

    def _apply_id_mapping(self, Z_mat: np.ndarray, ID_bank: list) -> np.ndarray:
        """Apply ID mapping to measurements"""
        Z_dec = Z_mat.copy()
        selectedIDs, deceptiveIDs = ID_bank[0]  # Using first level of ID bank
        Z_dec[selectedIDs, 0] = deceptiveIDs
        return Z_dec[Z_dec[:, 0].argsort(kind='mergesort')]

    def addDecoy_Data(self, Z_dec, consideredDecoyIDs):
        """Add decoy data to measurements"""
        print(f"Z_dec shape: {Z_dec.shape}")
        print(f"consideredDecoyIDs: {consideredDecoyIDs}")
        print(f"W_list shape: {self.data_loader.W_list.shape}")
        print(f"H_mat shape: {self.data_loader.H_mat.shape}")

        W_list = self.data_loader.W_list
        if W_list is None or np.isscalar(W_list) or W_list.ndim == 0:
            W_list = np.ones(self.data_loader.numOfZ)

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

class DefenseStrategy:
    """Implements defense mechanisms"""
    def __init__(self, data_loader: DataLoader, historical_manager: HistoricalDataManager):
        self.data_loader = data_loader
        self.historical_manager = historical_manager
        self.Z_org = self.data_loader.get_copy('Z_org')
        self.numOfZ = self.data_loader.numOfZ
        self.H_mat = self.data_loader.get_copy('H_mat')
        self.W_list = self.data_loader.get_copy('W_list')
        self.state_estimator = StateEstimator(self.data_loader)
        # self.historical_manager = HistoricalDataManager(100, self.data_loader)
        self.prediction_model = "polynomialRegression"
        self.Z_mat= self.data_loader.get_copy('Z_mat')
        
        # Add these new attributes
        self.Threshold_min = 3.0  # You may want to adjust this value
        self.Threshold_max = 10.0  # You may want to adjust this value
        self.Correction = False  # You may want to adjust this value

        # Initialize other necessary attributes
        self.consideredDecoyIDs = []
        self.consideredFixedIDs = []
        self.attackerTypeList = []
        self.attackerLevelList = []

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

    def update_IDs(self, seed, shuffle_percent):
        seed= self.set_random_seed(seed)
        all_sensor_ids = list(range(1, 55))  # IDs from 1 to 54

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
            clusters = []
            for i in range(12):
                cluster = Node()
                cluster.nodeType = 'Cluster'
                cluster.nodeID = i + 1
                for j in range(3):
                    sensor = sensors[i*3 + j]
                    cluster.addChild(sensor)
                clusters.append(cluster)

            # Create hubs and assign clusters and sensors
            hubs = []
            for i in range(2):
                hub = Node()
                hub.nodeType = 'Hub'
                hub.nodeID = i + 1
                for cluster in clusters[i*6:(i+1)*6]:
                    hub.addChild(cluster)
                for sensor in sensors[36+i*5:36+(i+1)*5]:
                    hub.addChild(sensor)
                hubs.append(hub)

            # Create server and assign hubs and remaining sensors
            server = Node()
            server.nodeType = 'Server'
            server.nodeID = 1
            for hub in hubs:
                server.addChild(hub)
            for sensor in sensors[46:]:
                server.addChild(sensor)
        
        # Cluster level: randomize each cluster of 3 sensors
        cluster_ids = all_sensor_ids[:36]
        dec_cluster_ids = self.shuffle_partial(cluster_ids, shuffle_percent)
        
        # Hub level: shuffle the specified percentage of the directly connected sensors
        hub_ids = all_sensor_ids[36:46]
        dec_hub_ids = self.shuffle_partial(hub_ids, shuffle_percent)
        
        # Server level: shuffle 50% of the directly connected sensors
        server_ids = all_sensor_ids[46:]
        dec_server_ids = self.shuffle_partial(server_ids, shuffle_percent)
        
        # Prepare the ID bank
        org_ids = all_sensor_ids
        dec_ids_cluster = dec_cluster_ids + hub_ids + server_ids
        dec_ids_hub = dec_cluster_ids + dec_hub_ids + server_ids
        dec_ids_server = dec_cluster_ids + dec_hub_ids + dec_server_ids
        
        ID_bank = [
            (org_ids, dec_ids_cluster),
            (org_ids, dec_ids_hub),
            (org_ids, dec_ids_server)
        ]
            
        return ID_bank
    

    # def filterData(self, Z_processed: np.ndarray, 
    #                 predictions: pd.DataFrame, 
    #                 threshold: float = 5.0,
    #                 fixedIDs: Union[List[int], int] = None) -> np.ndarray:
    #     """Apply filtering to processed measurements"""
    #     Z_df = pd.DataFrame(Z_processed, columns=['ID', 'Reported', 'Data'])
        
    #     # Convert fixedIDs to a list if it's an integer
    #     if isinstance(fixedIDs, int):
    #         fixedIDs = [fixedIDs]
    #     elif fixedIDs is None:
    #         fixedIDs = []
        
    #     # Only process measurements that aren't fixed
    #     for idx, row in Z_df.iterrows():
    #         if row['ID'] in fixedIDs or row['ID'] == 0:
    #             continue
                
    #         if row['Reported'] == 1:
    #             pred_value = predictions['MS'][int(row['ID'])]
    #             if abs(row['Data'] - pred_value) > abs(pred_value * threshold/100):
    #                 Z_df.at[idx, 'Data'] = pred_value
                    
    #     return Z_df.values

    # def imputeData(self, Z_processed: np.ndarray, 
    #                      predictions: pd.DataFrame) -> np.ndarray:
    #     """Apply imputation to missing measurements"""
    #     Z_df = pd.DataFrame(Z_processed, columns=['ID', 'Reported', 'Data'])
        
    #     # Impute missing or invalid measurements
    #     for idx, row in Z_df.iterrows():
    #         if row['Reported'] <= 0:  # Missing or invalid
    #             Z_df.at[idx, 'Reported'] = 1
    #             Z_df.at[idx, 'Data'] = predictions['MS'][int(row['ID'])]
                
    #     return Z_df.values
    
    def shuffle_partial(self, ids, shuffle_percent):
        """
        Shuffle a percentage of IDs while keeping the rest unchanged.
        """
        num_to_shuffle = int(len(ids) * shuffle_percent / 100)
        to_shuffle = random.sample(ids, num_to_shuffle)
        not_to_shuffle = [id for id in ids if id not in to_shuffle]
        shuffled = random.sample(to_shuffle, len(to_shuffle))
        
        result = ids.copy()
        shuffle_indices = [ids.index(id) for id in to_shuffle]
        for i, idx in enumerate(shuffle_indices):
            result[idx] = shuffled[i]
        
        return result
    
    def reset_to_original(self, ID_bank):
        """
        Reset the ID bank to the original state.
        """
        original_ids = ID_bank[0][0]  # The original IDs are the same for all levels
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

    def defenseEval(self, Attack_Data, attackIndx, IDbank, attackerLevelList, recoveryType, addDecoy, consideredDecoyIDs, attackCat, verbose_ = True, runfilter = False, runImpute = False):
        self.AttackEval = []
        self.detectionList = []
        self.outliers_suspect_List_List= []
        self.actually_attacked_list = []
        self.successful_attack_count = 0
        self.successCount_avg = 0
        self.detection_count = 0
        self.skippingList = []
        self.attackertype_list = []
        self.consideredFixedIDs = []
        self.consideredDecoyIDs = consideredDecoyIDs
        self.Z_est_init = []
        self.Noisy_Indx_actu = []
        self.Zpair = []
        self.th= 5 
        self.Z_org = self.data_loader.get_copy('Z_org')
        attackID = attackIndx #if attackIndx >= 0 else 0
        
        if consideredDecoyIDs is None:
            consideredDecoyIDs = []

        if attackCat == 3:
            import random
            attackertype_ = random.choice(range(3))
        else:
            attackertype_ = attackCat

        #--------------------------------------------------------------------------#
        #&&&&&&&&&&&&&&&&&&&&&&&&&& Implementing the attack &&&&&&&&&&&&&&&&&&&&&&&&
        #--------------------------------------------------------------------------#
        attackerLevel = attackerLevelList[attackertype_]

        selectedIDs  = IDbank[attackertype_][0]
        deceptiveIDs = IDbank[attackertype_][1]

        if verbose_:
            pass

        if attackertype_ == -1:
            selectedIDs = deceptiveIDs.copy()
            Z_dec = self.Z_org.copy() # Use self.Z_org instead of Z_mat
        else:
            # Z_dec is what attacker supposed to see
            Z_dec = self.Z_org.copy() # Use self.Z_org instead of Z_mat
            Z_dec[selectedIDs, 0] = deceptiveIDs.copy()
            Z_dec = Z_dec[Z_dec[:,0].argsort(kind = 'mergesort')]

        if len(consideredDecoyIDs) > 0 and addDecoy == True:
            Z_dec[consideredDecoyIDs, 2] = self.addDecoy_Data(Z_dec, consideredDecoyIDs)

        # running state estimation on deceived data ########

        # State Estimation and Bad Data Detection
        States_dece, Z_est, Z_processed, M_Noise, Noisy_Indx, fullRank, Threshold = self.state_estimator.SE_BDD_COR(
            self.H_mat.copy(), Z_dec.copy(), self.W_list, self.Threshold_min, self.Threshold_max, self.Correction, Verbose = "False")

        #############################################

        attackGain = 0
        startingIndx = 0
        endingIndx = 0 #New

        # Preparing the false data to be replaced
        if attackID >= 0:
            attackGain = 2
            startingIndx =  (attackID+1) * (self.numOfZ +1)
            endingIndx = (attackID+2) * (self.numOfZ +1)
            FData = Attack_Data[attackID].copy()
        else:
            attackGain = 0
            startingIndx =  (0) * (self.numOfZ +1)
            endingIndx = (1) * (self.numOfZ +1)
            FData = Attack_Data[0].copy()

        if abs(FData[:,2]).max() > 50:
            FData[:,2] = FData[:,2]*50/abs(FData[:,2]).max()

        injection = FData[:,2]* attackGain

        FData[:,2] = injection + Z_dec[:,2]

        # Attacker's intend to attack the following sensors
        attackedIndx = np.where(injection != 0)[0].copy()

        # Replacing Z_dec by Z_attack which is received by the EMS
        Z_att = Z_dec.copy()

        if attackCat == 'FDI':
            Z_att[attackedIndx, 2] = FData[attackedIndx, 2].copy()
        elif attackCat == 'DoS':
            Z_att[attackedIndx, 1] = -1

        if verbose_: print("Expected to attack: ", attackedIndx)

        #--------------------------------------------------------------------------#
        #&&&&&&&&&&&&&&&&&&&&&&&&&& Recovering from the attack &&&&&&&&&&&&&&&&&&
        #--------------------------------------------------------------------------#
        if runfilter == True or runImpute == True or recoveryType == 2 or recoveryType == 3:
            # Zpre Data using prediction models
            statePre = self.historical_manager.performanceAnalysis(self.historical_manager.stateHist, self.prediction_model, degree=2)
            
            # Ensure statePre is 2D
            if statePre.ndim == 1:
                statePre = statePre.reshape(1, -1)
            
            if not hasattr(self.historical_manager.scaler, 'mean_'):
                # Scaler not fitted; cannot inverse transform
                print("Scaler not fitted. Cannot inverse transform state predictions.")
                statePre_inv = statePre
            else:
                statePre_inv = self.historical_manager.scaler.inverse_transform(statePre)
            
            # Ensure statePre_inv is 1D for pred_msr
            statePre_inv = statePre_inv.flatten()
            
            Zpre = self.historical_manager.pred_msr(statePre_inv)

        # Received Data after Attack
            received = pd.DataFrame(Z_att[Z_att[:, 1] == 1][0:,[0,2]], columns = ['ID', 'MS'])
            received['ID'] = received['ID'].astype(int)

        fixedIDDs = list(set(self.consideredFixedIDs).intersection(set(received['ID'].values)))
        decoyIDDs = list(set(self.consideredDecoyIDs).intersection(set(received['ID'].values)))

        Z_rec = Z_att.copy()
        if recoveryType == 1 or recoveryType == 3:
            
            Z_rec[deceptiveIDs, 0] = selectedIDs.copy()
            Z_rec = Z_rec [Z_rec[:,0].argsort(kind = 'mergesort')]
            Zpair = (self.Z_mat, Z_rec)
            # self.z_matrices.append(Zpair)

        if recoveryType == 2 or recoveryType == 3:
            recoZdf = self.runMatch(self.numOfBuses, self.numOfLines, Zpre, received, fixedIDDs, decoyIDDs)
            self.get_data.append((self.numOfBuses, self.numOfLines, Zpre, received, recoZdf, fixedIDDs, decoyIDDs))

            recoZdf = recoZdf.sort_values(['rem_ID'])
            Z_rec = self.Z_org.copy()
            Z_rec[1:,1] = 0
            Z_rec[recoZdf['rem_ID'].values.astype(int),1] = 1
            Z_rec[recoZdf['rem_ID'].values.astype(int),2] = recoZdf['MS'].copy()
            Zpair = (self.Z_mat, Z_rec)
            self.z_matrices.append(Zpair)

        # mapping the attacked locations
        actually_attacked = np.sort(self.mapOrgID(attackedIndx.copy(), selectedIDs, deceptiveIDs).astype(int))
        # self.targeted.append(attackedIndx)

        if verbose_: print("Actually attacked:", actually_attacked)
        # self.self.attacked.append(actually_attacked)
        if verbose_: print("Calling State Estimation for recovered data..:")

        # State Estimation and Bad Data Detection
        States_check, Z_est_check, Z_mat_check, M_Noise_check, Noisy_index_check, Rank_check, Threshold_check = self.state_estimator.SE_BDD_COR(
            self.H_mat.copy(), Z_rec.copy(), self.W_list, self.Threshold_min, self.Threshold_max, self.Correction, Verbose = "False")

        savedState = (States_check, Z_est_check, Z_mat_check, M_Noise_check, Noisy_index_check, Rank_check, Threshold_check)

        runningfilter = False; runningImputer = False

        Z_rec_df = pd.DataFrame(Z_mat_check, columns = ['ID', 'Taken', 'MS'])
        #########################  Adding LSTM Features ####################
        if Rank_check == True:
            if runfilter == True:
                Z_reco = self.data_loader.filterData(Z_rec_df, Zpre, self.th, fixedIDDs)
                runningfilter = True
        else:
            if runImpute == True:
                print("System is not not observable!!")
                self.impute_data.append((Z_rec_df,Zpre))
                Z_reco = self.data_loader.filterData(Z_rec_df, Zpre, self.th, fixedIDDs)
                runningImputer = True

        if runningfilter == True or runningImputer == True:
            # State Estimation and Bad Data Detection
            States_check, Z_est_check, Z_mat_check, M_Noise_check, Noisy_index_check, Rank_check, Threshold_check = self.state_estimator.SE_BDD_COR(
                self.H_mat.copy(), Z_reco.copy(), self.W_list, self.Threshold_min, self.Threshold_max, self.Correction, Verbose = "False")

            if runfilter == True and Rank_check == False:
                print("LSTM prediction error!!") if runfilter == True else print("Imputation Failed")
                (States_check, Z_est_check, Z_mat_check, M_Noise_check, Noisy_index_check, Rank_check, Threshold_check) = savedState

            if runImpute == True and Rank_check == False:
                print("Imputation failed!!!!!!")

        self.update_state_history(States_check[1:])  # Assuming States[0] is a timestamp or something to be excluded

        M_Noise_check = Z_rec[:,2] - Z_est_check

        # Initialize Z_est_init if it's not already set or is empty
        if not hasattr(self, 'Z_est_init') or len(self.Z_est_init) == 0:
            # Perform initial state estimation
            States_init, Z_est_init, _, _, _, _, _ = self.state_estimator.SE_BDD_COR(
                self.H_mat.copy(), self.Z_org.copy(), self.W_list, 
                self.state_estimator.Threshold_min, self.state_estimator.Threshold_max, 
                False, "False"
            )
            self.Z_est_init = Z_est_init

        # Ensure Z_est_init has the correct shape
        if self.Z_est_init.shape != Z_est_check.shape:
            print(f"Warning: Z_est_init shape ({self.Z_est_init.shape}) doesn't match Z_est_check shape ({Z_est_check.shape}). Reinitializing Z_est_init.")
            self.Z_est_init = np.zeros_like(Z_est_check)

        # eliminated attacked sensors
        foundFDI_Idx =  sorted((set(Noisy_index_check)- set(self.Noisy_Indx_actu)) & set(actually_attacked)) # Noisy_index_actu is not defined annywhere

        # printing noisy indeces
        if Noisy_index_check.size > 0 and verbose_ == True:
            pass

        # Initialize counters
        successful_attack_count = 0
        detection_count = 0
        successCount_avg = 0

        # system Unobservable
        if Rank_check == False:
            if verbose_:
                print("-----------  System Unobservable  --------------")
            Deviation = 0

            AttackEval = [attackID, "unobservable", Deviation]
            AttackReturn = {}
            AttackReturn['StatesAttack'] =  0
            AttackReturn['StatesDeceived'] =  0
            AttackReturn['Deviation']  = 0
            AttackReturn['Check'] = 0
            AttackReturn['Zpair'] = (0,0)
            AttackReturn['Z_dec'] = 0
        else:
            # system is observable
            # calculating the percent of deviation in the estimated measurements
            Deviation = np.linalg.norm(Z_est_check - self.Z_est_init) / np.linalg.norm(self.Z_est_init) * 100 if np.linalg.norm(self.Z_est_init) != 0 else 0

            ########################
            totalAttackedSensors = np.sum(Z_rec[actually_attacked, 1])
            ########################
            successCount = len(set(Noisy_index_check) & set(actually_attacked))
            #########################
            if attackID > 0:
                pass

            # Detected as Bad Data
            if len(foundFDI_Idx) > 0:
                AttackEval = [attackID, "detected", Deviation]
                detection_count = 1
                print("Detected Measurements: ", foundFDI_Idx)

                if verbose_:
                    print("!!!!!!!!!! Detected as Bad Data  !!!!!!!!!!!!")
                    print("Detected Measurements: ", foundFDI_Idx)

            # Attack was undetected
            else:
                if verbose_:
                    print("$$$$$$$$$$$$$$  Attack Successful as undetected $$$$$$$$$$$$$")
                    print("$$$$$$$$$$$$$$  Attack Successful as undetected , Attack ID: $$$$$$$$$$$$$", attackID)
                AttackEval = [attackID, "success", Deviation]
                successful_attack_count = 1

            successCount_avg = successCount / totalAttackedSensors * 100 if totalAttackedSensors > 0 else 0
            AttackReturn = {}

            AttackReturn['StatesAttack'] =  States_check
            AttackReturn['StatesDeceived'] =  States_dece

            AttackReturn['Deviation']  = AttackEval[2]
            AttackReturn['Check'] = AttackEval[1]
            AttackReturn['Zpair'] = Zpair
            AttackReturn['Z_dec'] = Z_dec

        return AttackReturn, successful_attack_count, detection_count, successCount_avg   

    def runMatch(numOfBuses, numOfLines, z_pre, z_rep, fixedIDs, decoyIDDs):
        th = 5

        th_max = abs(z_pre['MS']*th/100)
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

class DQNetwork(tf.keras.Model):
    """Neural network for DQN agent"""
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super(DQNetwork, self).__init__()
        self.fc1 = layers.Dense(hidden_size, activation='relu', input_shape=(state_dim,))
        self.fc2 = layers.Dense(hidden_size, activation='relu')
        self.fc3 = layers.Dense(action_dim)  # Output both deception_amount and sensor_selection

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128, numOfZ: int = 55):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.numOfZ = numOfZ
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQNetwork(state_dim, action_dim, hidden_size)
        self.target_model = DQNetwork(state_dim, action_dim, hidden_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def remember(self, state, action, reward, next_state, done):
        # Convert dict actions to a single array
        action_array = np.concatenate([
            action['deception_amount'],
            action['sensor_selection']
        ])
        
        # Flatten the state and next_state
        state_array = np.concatenate([
            state['sensor_readings'].flatten(),
            state['attack_history']
        ])
        next_state_array = np.concatenate([
            next_state['sensor_readings'].flatten(),
            next_state['attack_history']
        ])
        
        self.memory.append((state_array, action_array, reward, next_state_array, done))

    def act(self, state):
        state_array = np.concatenate([
            state['sensor_readings'].flatten(),
            state['attack_history']
        ])
        state_tensor = tf.convert_to_tensor(state_array[None, :], dtype=tf.float32)
        
        if random.random() <= self.epsilon:
            return {
                'deception_amount': np.array([np.random.uniform(0, 50)]),
                'sensor_selection': np.random.randint(0, 2, self.numOfZ)
            }
        
        q_values = self.model(state_tensor)
        deception_amount = q_values[0, 0].numpy()
        sensor_selection = (q_values[0, 1:self.numOfZ+1] > np.median(q_values[0, 1:self.numOfZ+1])).numpy().astype(int)
        
        return {
            'deception_amount': np.array([deception_amount]),
            'sensor_selection': sensor_selection
        }

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Convert to TensorFlow tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Q-value estimation
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            next_q_values = self.target_model(next_states)
            
            # Use only the deception_amount for max Q-value selection
            max_next_q_values = tf.reduce_max(next_q_values[:, 0], axis=-1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
            
            # Compute loss only for the deception_amount
            loss = tf.keras.losses.MeanSquaredError()(target_q_values, q_values[:, 0])

        # Compute gradients and update the model
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_target_model(self):
        """Update target network"""
        self.target_model.set_weights(self.model.get_weights())

#overall environment

class IEEE14BusPowerSystemEnv(gym.Env):
    def __init__(self, data_loader: DataLoader, attack_scenario_manager: AttackScenarioManager,
                 state_estimator: StateEstimator, historical_manager: HistoricalDataManager,
                 data_processor: DataProcessor, defense_strategy: DefenseStrategy):
        super(IEEE14BusPowerSystemEnv, self).__init__()

        # System parameters
        self.numOfBuses = data_loader.numOfBuses
        self.numOfLines = data_loader.numOfLines
        self.numOfStates = self.numOfBuses
        self.numOfZ = data_loader.numOfZ
        self.attacked_Bus = data_loader.attacked_Bus
        self.localize = data_loader.localize
        self.attackFreq = data_loader.attackFreq if hasattr(data_loader, 'attackFreq') else 1

        # Initialize components
        self.data_loader = data_loader
        self.attack_scenario_manager = attack_scenario_manager
        self.state_estimator = state_estimator
        self.historical_manager = historical_manager
        self.data_processor = data_processor
        self.defense_strategy = defense_strategy
        self.defense_strategy.historical_manager = historical_manager  # Add this line

        # Get necessary data
        self.H_mat = self.data_loader.H_mat
        self.W_list = self.data_loader.W_list
        self.Z_org = self.data_loader.Z_org

        # Update the observation space
        self.observation_space = spaces.Dict({
            'sensor_readings': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.numOfZ, 3), dtype=np.float32
            ),
            'attack_history': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        })

        # Update the action space
        self.action_space = spaces.Dict({
            'deception_amount': spaces.Box(low=0, high=50, shape=(1,), dtype=np.float32),
            'sensor_selection': spaces.MultiBinary(self.numOfZ)
        })

    def reset(self):
        """Reset environment to initial state"""
        self.current_attack = 0
        self.current_seed = random.randint(0, 1000)
        
        # Reset to original measurements
        self.state = {
            'sensor_readings': self.Z_org.copy(),
            'attack_history': np.zeros(2)
        }
        
        return self.state

    def step(self, action):
        """Execute one environment step"""
        try:
            deception_amount = action['deception_amount'][0]
            sensor_selection = action['sensor_selection']

            print(f"Action received - deception_amount: {deception_amount}, sensor_selection shape: {sensor_selection.shape}")
            print(f"numOfZ: {self.numOfZ}")
            print(f"Z_org shape: {self.Z_org.shape}")
            print(f"H_mat shape: {self.H_mat.shape}")
            print(f"W_list shape: {self.W_list.shape}")

            # Ensure sensor_selection has the correct length
            if len(sensor_selection) != self.numOfZ:
                print(f"Warning: sensor_selection length ({len(sensor_selection)}) doesn't match numOfZ ({self.numOfZ}). Adjusting sensor_selection.")
                sensor_selection = np.pad(sensor_selection, (0, self.numOfZ - len(sensor_selection)), mode='constant')

            # Calculate number of sensors to randomize
            total_sensors = self.numOfZ
            num_randomized = int(total_sensors * deception_amount / 100)
            num_fixed = total_sensors - num_randomized

            # Select sensors for randomization
            randomized_indices = np.random.choice(total_sensors, num_randomized, replace=False)
            fixed_indices = np.setdiff1d(np.arange(total_sensors), randomized_indices)

            # Update defense strategy
            num_considered_fixed = int(3 * num_fixed / 4)
            self.defense_strategy.consideredFixedIDs = np.random.choice(
                fixed_indices, 
                min(num_considered_fixed, len(fixed_indices)), 
                replace=False
            )

            # The remaining 1/4 of fixed IDs and all randomized IDs are potential decoys
            potential_decoy_indices = np.setdiff1d(np.arange(total_sensors), self.defense_strategy.consideredFixedIDs)

            # Select decoy sensors based on the sensor_selection action
            decoy_mask = sensor_selection[potential_decoy_indices[potential_decoy_indices < self.numOfZ]] == 1
            self.defense_strategy.consideredDecoyIDs = potential_decoy_indices[potential_decoy_indices < self.numOfZ][decoy_mask]

            # 2. Update ID mappings based on sensor selection
            ID_bank = self.defense_strategy.update_IDs(
                seed=self.current_seed,
                shuffle_percent=deception_amount
            )

            # 3. Create deceived measurements (Z_dec)
            Z_dec = self.Z_org.copy()
            
            # 4. Add decoy data for consideredDecoyIDs
            if len(self.defense_strategy.consideredDecoyIDs) > 0:
                Z_dec[self.defense_strategy.consideredDecoyIDs, 2] = self.data_processor.addDecoy_Data(
                    Z_dec, self.defense_strategy.consideredDecoyIDs)

            # 6. Evaluate defense strategy
            attack_return, success_count, detection_count, success_rate = self.defense_strategy.defenseEval(
                Attack_Data=self.data_loader.Attack_Data,
                attackIndx=self.current_attack,
                IDbank=ID_bank,
                attackerLevelList=[1, 2, 3],
                recoveryType=1,
                addDecoy=False,
                consideredDecoyIDs=self.defense_strategy.consideredDecoyIDs,
                attackCat=1,
                verbose_=False,
                runfilter=True,
                runImpute=True
            )

            # 7. Get received measurements after attack
            Z_rec = attack_return['Zpair'][1]

            # 8. Perform State Estimation and Bad Data Detection
            States, Z_est, Z_processed, M_Noise, Noisy_Indx, fullRank, Threshold = \
                self.state_estimator.SE_BDD_COR(
                    H_mat=self.H_mat,
                    Z_mat=Z_rec,
                    W_list=self.W_list,
                    Threshold_min=self.state_estimator.Threshold_min,
                    Threshold_max=self.state_estimator.Threshold_max,
                    Corr=False,
                    Verbose="False"
                )

            # 9. Process data based on system observability

            if fullRank:
                # System is observable - apply filtering
                Z_rec_df = pd.DataFrame(Z_processed, columns=['ID', 'Reported', 'Data'])
                Zpre = self.historical_manager.pred_msr(
                    self.historical_manager.performanceAnalysis(
                        self.historical_manager.stateHist,
                        "polynomialRegression",
                        degree=2
                    )
                )
                Z_processed = self.data_loader.filterData(
                    Z_rec_df,
                    Zpre,
                    threshold=5,
                    fixedIDs=self.defense_strategy.consideredFixedIDs
                )
            else:
                # System not observable - apply imputation
                Z_rec_df = pd.DataFrame(Z_processed, columns=['ID', 'Reported', 'Data'])
                Zpre = self.historical_manager.pred_msr(
                    self.historical_manager.performanceAnalysis(
                        self.historical_manager.stateHist,
                        "polynomialRegression",
                        degree=2
                    )
                )
                Z_processed = self.data_processor.imputeData(Z_rec_df, Zpre)

            # 10. Update historical data
            if fullRank:
                self.historical_manager.update_stateHist(States[1:].reshape(1,-1))
                scaled_states = self.historical_manager.get_scaled_state(States[1:].reshape(1,-1))

            # 11. Update environment state
            self.state = {
                'sensor_readings': Z_processed,
                'attack_history': np.array([success_rate/100, detection_count])
            }

            # 12. Calculate reward
            reward = self._calculate_reward(
                success_count=success_count,
                detection_count=detection_count,
                sensor_selection=sensor_selection,
                state_deviation=attack_return['Deviation'],
                system_observable=fullRank
            )

            # 13. Update attack index and check if episode is done
            self.current_attack += 1
            done = self.current_attack >= 129  # Complete all attack scenarios

            # 14. Prepare info dictionary for monitoring
            info = {
                'success_count': success_count,
                'detection_count': detection_count,
                'success_rate': success_rate,
                'system_observable': fullRank,
                'state_deviation': attack_return['Deviation'],
                'shuffle_percent': deception_amount,
                'noisy_measurements': len(Noisy_Indx) if Noisy_Indx is not None else 0,
                'runfilter': self.state_estimator.runfilter,
                'runImpute': self.state_estimator.runImpute,
                'num_fixed_ids': len(self.defense_strategy.consideredFixedIDs),
                'num_decoy_ids': len(self.defense_strategy.consideredDecoyIDs)
            }

            return self.state, reward, done, info

        except Exception as e:
            print(f"Error in step function: {str(e)}")
            print(f"Action shape: {action.shape if hasattr(action, 'shape') else 'N/A'}")
            print(f"Action content: {action}")
            raise e

    def _calculate_reward(self, success_count, detection_count, sensor_selection,
                        state_deviation, system_observable):
        """Calculate reward based on multiple factors"""
        # Base rewards/penalties
        detection_reward = 5.0 * detection_count
        attack_penalty = -10.0 * success_count
        
        # Operational costs
        shuffle_cost = -0.1 * (np.sum(sensor_selection) / self.numOfZ)
        
        # System stability
        stability_factor = 1.0 if system_observable else -20.0  # Heavy penalty for losing observability
        deviation_penalty = -0.1 * state_deviation
        
        # Calculate total reward
        total_reward = (detection_reward + attack_penalty + shuffle_cost + 
                    stability_factor + deviation_penalty)
        
        # Clip reward to prevent extreme values
        total_reward = np.clip(total_reward, -100, 100)
        
        return total_reward
    
    
def main():
    try:
        # Load data
        data_loader = DataLoader()
        data_loader.load_data()

        # Initialize components
        state_estimator = StateEstimator(data_loader)
        
        # Get the required parameters from data_loader
        max_history = 100  # You can adjust this value as needed
        numOfBuses = data_loader.numOfBuses
        H_mat = data_loader.H_mat
        
        historical_manager = HistoricalDataManager(max_history, numOfBuses, H_mat)
        
        data_processor = DataProcessor(data_loader)
        defense_strategy = DefenseStrategy(data_loader, historical_manager)

        # Initialize the attack scenario manager
        max_attacked_buses = 3  # Adjust this value as needed
        attack_scenario_manager = AttackScenarioManager(numOfBuses, max_attacked_buses, data_loader)

        # Initialize the environment
        env = IEEE14BusPowerSystemEnv(
            data_loader=data_loader,
            attack_scenario_manager=attack_scenario_manager,
            state_estimator=state_estimator,
            historical_manager=historical_manager,
            data_processor=data_processor,
            defense_strategy=defense_strategy
        )

        # Initialize the DQN agent
        state_dim = env.observation_space['sensor_readings'].shape[0] * env.observation_space['sensor_readings'].shape[1] + env.observation_space['attack_history'].shape[0]
        action_dim = 1 + env.numOfZ  # deception_amount (1) + sensor_selection (numOfZ)
        print(f"state_dim: {state_dim}, action_dim: {action_dim}, numOfZ: {env.numOfZ}")
        agent = DQNAgent(state_dim, action_dim, numOfZ=env.numOfZ)

        # Training loop
        episodes = 1000
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.act(state)
                print(f"Episode {episode}, Action: {action}")
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                
                if len(agent.memory) > agent.batch_size:
                    agent.replay()
                
                state = next_state
                total_reward += reward
            
            if episode % 10 == 0:
                agent.update_target_model()
                print(f"Episode: {episode}, Total Reward: {total_reward}")

    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        import traceback
        traceback.print_exc()

    
if __name__ == "__main__":
    main()