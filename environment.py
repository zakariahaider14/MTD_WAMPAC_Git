import gym
from gym import spaces
import numpy as np

class MTDEnvironment(gym.Env):
    def __init__(self, num_sensors=54, max_steps=100):
        super(MTDEnvironment, self).__init__()
        
        self.num_sensors = num_sensors
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define action space: [num_randomized_sensors, num_fixed_sensors]
        self.action_space = spaces.MultiDiscrete([num_sensors + 1, num_sensors + 1])
        
        # Define observation space: [current_randomized, current_fixed, detection_rate, attack_success_rate]
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.current_randomized = 0
        self.current_fixed = 0
        self.detection_rate = 0
        self.attack_success_rate = 0
        
        return self._get_obs()
    
    def step(self, action):
        self.current_step += 1
        
        num_randomized, num_fixed = action
        
        # Ensure the sum of randomized and fixed sensors doesn't exceed total sensors
        total_sensors = min(num_randomized + num_fixed, self.num_sensors)
        num_randomized = min(num_randomized, total_sensors)
        num_fixed = total_sensors - num_randomized
        
        self.current_randomized = num_randomized / self.num_sensors
        self.current_fixed = num_fixed / self.num_sensors
        
        # Run the MTD simulation with the current configuration
        detection_rate, attack_success_rate = self._run_mtd_simulation(num_randomized, num_fixed)
        
        self.detection_rate = detection_rate
        self.attack_success_rate = attack_success_rate
        
        reward = self._calculate_reward(detection_rate, attack_success_rate)
        done = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        return np.array([
            self.current_randomized,
            self.current_fixed,
            self.detection_rate,
            self.attack_success_rate
        ], dtype=np.float32)
    
    def _run_mtd_simulation(self, num_randomized, num_fixed):

        numOfStates = numOfBuses
        ASEU_data = pd.DataFrame([])
        file_name = f"Bus Data//IEEE_{numOfBuses}.xlsx"

        # Load data into Dataframes
        bus_data_df = pd.read_excel (file_name, sheet_name = "Bus")
        line_data_df = pd.read_excel (file_name, sheet_name = "Branch")

        # number of lines and measurements
        numOfLines = line_data_df.shape[0]
        numOfZ = numOfBuses + numOfLines * 2
        W_list = (numOfZ + 1)*[1]

        # update the index from 1 to number of elements
        bus_data_df.set_index(pd.Series(range(1, numOfBuses+1)), inplace = True)
        line_data_df.set_index(pd.Series(range(1, numOfLines+1)), inplace = True)

        # preprocess data and update line and bus numbers
        #preprocess_data(bus_data_df, line_data_df)

        # print(bus_data_df.head().T)
        # print(line_data_df.head().T)
        ###################################################################
        # Loading Topology Data and Measurement Data
        try:
            topo_mat   = pd.read_excel(file_name, sheet_name = "Topology Matrix")
            line_data  = pd.read_excel(file_name, sheet_name = "Line Data")
            print("Topology Matrix Loaded!")
        except:
            print("Generating Topology Matrix...")
            topo_mat, line_data = generate_topology_matrix(numOfBuses, numOfLines, line_data_df, file_name)

        Topo = line_data.values.astype(int) #Another name

        # Loading Topology Data and Measurement Data
        try:
            Z_msr_org = pd.read_excel(file_name, sheet_name = "Measurement Data")
            bus_data  =  pd.read_excel(file_name, sheet_name = "Bus Data")
            print("Measurement Data Loaded!")
        except:
            print("Generating Measurement Data...")
            Z_msr_org, bus_data = generate_Z_msr_org(numOfBuses, numOfLines, bus_data_df, topo_mat, file_name)



        # Adding IDs and Reported columns
        Z_msr_org.insert(0, 'ID', list(Z_msr_org.index.values))
        Z_msr_org.insert(1, 'Reported', [1]* (numOfZ+1))
        print("size of Z_msr_org:" , Z_msr_org.shape)

        file_Name_ = "Attack_Space_"+str(numOfBuses)+"_"+str(numOfLines)+"_"+str(attacked_Bus)+".csv"
        try:
            Attack_Data = np.genfromtxt("Attack Data//"+file_Name_, delimiter=',')
            print("Attack data loaded!")
        except:
            print("Attack Data is missing! Generating attack data!")
            current_path = os.getcwd()
            Attack_Data = generate_attackdata(numOfBuses, numOfLines, line_data, attacked_Bus, current_path)

        attackertype_list = []
        numOfAttacks = int (Attack_Data.shape[0]/(numOfZ+1))
        print("numOfAttacks: ", numOfAttacks)
        meanZ= abs(Z_msr_org['Data']).mean()

        Z_org = Z_msr_org.values.copy()


        ####################################################################
        if localize == True:
            #Adding Noisy data
            Noise_Data = Attack_Data.copy()
            Noise_Data[:,2] = np.random.randint(-20,20, size = (Noise_Data.shape[0]))
            Noise_Data[Noise_Data[:,1] == 0, 2] = 0
            np.random.shuffle(Noise_Data[:,1:])

    # Updating attack data
        Attack_Data = np.concatenate((Attack_Data, Noise_Data), axis = 0)

        z_matrices = []
        targeted =[]
        attacked=[]
        pre_rec = []

        #Evaluation Matrix


        EvalSum = {}

        EvalSum['random'] = []
        EvalSum['report'] = []
        EvalSum['fixed'] = []
        EvalSum['attType'] = []
        EvalSum['noOfbuses'] = []
        EvalSum['noise'] = []
        EvalSum['timeStep'] = []
        EvalSum['StatesInit'] = []
        EvalSum['StatesAttack'] = []
        EvalSum['StatesDeceived'] = []
        EvalSum['StatesOrg'] = []
        EvalSum['Deviation'] = []
        EvalSum['Check'] = []
        EvalSum['Zpair'] = []
        EvalSum['filter'] = []
        EvalSum['duration'] = []
        EvalSum['percentSDN'] = []
        EvalSum['Z_dec'] = []
        EvalSum['duraInit'] = []
        EvalSum['attackCount'] = []


        dura= 20


        iattack=0
        attackIndx=0

        # recoveryType=3


        addDecoy= True
        runfilter= True
        runImpute= False

        successful_attack_list=[]
        detection_list =[]
        detection_success_rate =[]
        attackertype=0

        seed_value = 129

        ID_bank =[]

        for percentOfDeception in percentOfDeception_list:
                percentOfdNodes_list = percentOfDeception


                ######################### Constrcuting Clusters #################################################

                repeat = 1
                retry = 0
                maxRepeat=1
                Correction = False

                while(repeat <= maxRepeat):
                    #print("repeat: ",repeat)
                    repeat += 1
                    #------------------------------------------------------------------------------------------------
                    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&   Modeling Noise and Report and Deception &&&&&&&&&&&&&&&&&&
                    #------------------------------------------------------------------------------------------------

                    # Percent of observed sensors = 100 - reported
                    percentOfobserved = 100 - percentOfreported

                    ##########################    Copying the original Data ###########################
                    H_org = topo_mat.values.copy()
                    Z_org = Z_msr_org.values.copy()
                    ##################################################################################


                    #############################    Modeling  noise   ###############################
                    Noise = np.random.normal(noise_mu, noise_sigma, numOfZ)
                    Z_org[1:,2] = np.multiply(Z_org[1:,2], Noise/100 + 1)
                    ##################################################################################


                    ############################  Modeling Data Reporting  ###########################
                    randomIDs = np.arange(1,numOfZ)
                    np.random.shuffle(randomIDs)
                    overservedIDs = randomIDs[0: int (percentOfobserved*numOfZ/100)]
                    print("overservedIDs: ", overservedIDs)
                    Z_org[overservedIDs, 1] = 0 # updating Z_org # observed IDs are the data that are not reported directly in the system. Dont confuse it with the decoy data

                    ### ----------->>  Check observability ---->>>
                    #print(Z_org)
                    Z_mat = Z_org.copy()
                    ###################################################################################


                    ##########################   Modeling Deception   #################################
                    Deception_flag = []

                    for index in range(Z_mat[1:].shape[0]):
                        Deception_flag.append([])

                    ## Flipping coin each time for each sensor to decide on deception ##########
                    for Z in Z_mat[1:,:]:
                        toss = np.random.binomial(size = 1, n = 1, p = percentOfDeception/100)[0]
                        Deception_flag[Z[0].astype(int) - 1].append(0 if Z[1].astype(int) == 0 else toss) #if it is decoy id z[1]==0 then dont include it in the deception list

                    #------------------------------------------------------------------------------------------------
                    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& SE and BDD &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
                    #------------------------------------------------------------------------------------------------

                    # State Estimation and Bad Data Detection
                    States_init, Z_est_init, Z_mat_init, M_Noise_actu, Noisy_index_actu, fullRank, Threshold = SE_BDD_COR(
                        H_org.copy(), Z_org.copy(), W_list, Threshold_min, Threshold_max, Correction, Verbose = "False")

                    print("Value of Full Rank", fullRank )
                    if fullRank == False:
                        if retry < maxRepeat:
                            print("The systen is not observable--> Retrying")
                            repeat -= 1
                            retry += 1
                            continue
                        else:
                            print("The number of reported measurements are too low!")
                            break
                    else:
                        print("The systen is observable")
                    
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



                    shuffle_percent = percentOfdNodes_list  # You can change this to 50, 60, or 80
                    
                    
                    for attackertype in  attackertypeList:
                        successful_attack_count=0
                        detection_count =0

                        print("\nattackertype: ", attackertype)
                        if attackertype <2:

                            recoveryType = 3
                            print("Prediction Based Remapping is working")
                            runfilter = True
        #                    runImpute = True
                            addDecoy = True
        #                     recoveryType = 1
        #                     print("Seed-Based Remap is working ")
        #                     runfilter = False
        #                     runImpute = False
        #                     addDecoy =  False

        # #     # Best Performance so far is prediction level_1 and seed level 2&3
                        else:
                            recoveryType = 3
                            print(" Based Remapping is working")
                            runfilter = True
                            runImpute = False
                            addDecoy = True

        #                     recoveryType = 1
        #                     print("Seed-Based Remap is working ")
        #                     runfilter = False
        #                     runImpute = False
        #                     addDecoy =  False

                        initrange = 100-percentOfDeception #if addDecoy == False else 0
                        for percentOffixed in range(initrange, 100-percentOfDeception+1,100):
                    #         print("initrange", initrange)
                    #         print("initrange", initrange)
                    #         print("percentOffixed: ", percentOffixed)
                            totalFixedPoints = int(numOfZ*percentOffixed/100)
                    #         print("totalFixedPoints: ", totalFixedPoints)
                            iattack=0
                            successCount_avg=0

                            for timeIndx in range(init, init+timestep+1):
                                if (timeIndx-init)% attackFreq == 0:
                    #                 print("Reset stateHist")
                                    stateHist = X_train[timeIndx].copy()

                                if (timeIndx-init)% updateFreq == 0 and timeIndx>0 :
                            #             print(timeIndx-init)
                                    print("Updating IDBank...")
                                    ID_bank = update_IDs(server, clusters, hubs, seed_value, shuffle_percent)


                                for i, (org_ids, dec_ids) in enumerate(ID_bank):
                                    level = ["Cluster", "Hub", "Server"][i]
                                    print(f"{level} level:")
        #                             print(f"  Original IDs: {org_ids}")
        #                             print(f"  Deceptive IDs: {dec_ids}")
                                    match=[]
                                    for j in range(0, len(org_ids)):
                                        if org_ids[j] != dec_ids[j]:
                                            match.append(1)
                                        else:
                                            match.append(0)
                                    print(f"Number of Randomized Sensors:{sum(match)} at Level {level} ", )

                                    print()

                                    consideredFixedIDs = fixedIDs[0:totalFixedPoints]
                            #             print("IDbank :", IDbank)
                    #                 print("ObservedID",overservedIDs)
                    #                 print("fixedIDs :", len(fixedIDs))
                    #                 print("consideredFixedIDs :", consideredFixedIDs)

                                    consideredDecoyIDs = fixedIDs[totalFixedPoints:]
                    #                 print("consideredDecoyIDs :", consideredDecoyIDs)

                                if ((timeIndx-init)/attackFreq) > 1 and timeIndx%attackFreq == 1:


                                    try:
                                        stateOrg = Y_train[timeIndx:timeIndx+1].flatten()
                                        stateOrg = np.concatenate((np.array([0]), stateOrg), axis =0 )
                                        #print("stateOrg: ", stateOrg.astype(int))
                                    except:
                                        print("Except!!!")

                        #             attackIndx = attackIndxxx[timeIndx-init]
                                    duraInit = dura
                                    attackIndx = attackIndxxx[iattack]
                                    print(f"Attack Happening with Index:{iattack} and time: {timeIndx} ")
                                    iattack += 1
                                    #print("If--> duraInit: ", duraInit, "attackIndex: ", attackIndx)
                                    Z_org[:,2] = H_org@stateOrg

                                    Z_mat = Z_org.copy()

                        ###################################################################################
                        # State Estimation and Bad Data Detection
                                    States_init, Z_est_init, Z_mat_init, M_Noise_actu, Noisy_index_actu, fullRank, Threshold = SE_BDD_COR(
                                        H_org.copy(), Z_org.copy(), W_list, Threshold_min, Threshold_max, Correction, Verbose = False)

                                    attackEval, successful_attack, detection, successCount_1 = defenseEval(Attack_Data, attackIndx, ID_bank, attackerLevelList,recoveryType, verbose_ = True)
                                    print ("Success Count Avg in this round: ", successCount_1)
                                    successCount_avg+= successCount_1
                                    successful_attack_count += successful_attack
                                    detection_count += detection


                                    EvalSum['StatesOrg'].append(stateOrg)
                                    #EvalSum['StatesOrg'].append(Z_est_init)
                                    EvalSum['random'].append(percentOfDeception)
                                    EvalSum['report'].append(percentOfreported)
                                    EvalSum['fixed'].append(percentOffixed)
                                    EvalSum['attType'].append(attackertype)
                                    EvalSum['noOfbuses'].append(numOfBuses)
                                    EvalSum['noise'].append(noise_sigma)
                                    EvalSum['timeStep'].append(timeIndx)
                                    EvalSum['StatesInit'].append(States_init)
                                    #EvalSum['StatesInit'].append(Z_est_init)
                                    EvalSum['StatesDeceived'].append(attackEval['StatesDeceived'])
                                    EvalSum['StatesAttack'].append(attackEval['StatesAttack'])
                                    EvalSum['Deviation'].append(attackEval['Deviation'])
                                    EvalSum['Check'].append(attackEval['Check'])
                                    EvalSum['Zpair'].append(attackEval['Zpair'])
                                    EvalSum['filter'].append(prediction_model)
                                    EvalSum['duration'].append(dura)
                                    EvalSum['percentSDN'].append(PerOfsdnController)
                                    EvalSum['Z_dec'].append(attackEval['Z_dec'])
                                    EvalSum['duraInit'].append(duraInit)
                                    EvalSum['attackCount'].append(successful_attack)


                                # continuation of the same attack
                                elif duraInit > 1:
                                    duraInit -= 1

                                else:
                                    attackIndx = -1



                        #senMsr[-numOfBuses:] = senMsr[-numOfBuses:]*(-1)
                        print("########## Final Results ################# ")

                        print("Attack happening at Level ", attackertype)
                        print("Attack Successful with count: ", successful_attack_count)
                        print("Attack Detected with count: " ,detection_count)
            #             print("Average Successful Attack per round: ",successCount_avg)
                        print("Average Detection % per round: ",successCount_avg/(detection_count+successful_attack_count))


                        successful_attack_list.append(successful_attack_count)
                        detection_list.append(detection_count)
                        detection_success_rate.append(successCount_avg/(detection_count+successful_attack_count))

                return 0.5, 0.3
            
    def _calculate_reward(self, detection_rate, attack_success_rate):
        # Define a reward function based on the detection rate and attack success rate
        return detection_rate - attack_success_rate