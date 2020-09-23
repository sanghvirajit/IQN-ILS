import os
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
import subprocess

from multiprocessing import Process, Pool
from functools import partial
from scipy.stats.stats import pearsonr

from functions import *

from fluid import *
from solid import *

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

baseName = os.path.basename(__file__).split(".py")[0]
f = open(baseName+'.log', 'w')

sys.stdout = Tee(sys.stdout, f)

if __name__ == "__main__":
    for item in os.listdir("."):
        if item.endswith(".txt"):  # or item.endswith(".sres"):
            os.remove(os.path.join(".", item))
        if item.endswith(".png"):  # or item.endswith(".sres"):
            os.remove(os.path.join(".", item))
            
    setupFileName = "setup.dat"
       
    fluidSurfs = []
    solidSurfs = []
    
    delT_mean = []
    delT_mean_k = []
    
    delT_std = []
    delT_std_k = []
    
    mean_Temp = []
    mean_Temp_k = []
    
    Residual = []
    
    omega = 0.8
        
    setupFile = check_open(setupFileName, 'r')
    for line in setupFile:
        words = line.split()
        if len(words) == 0:
            continue
        if words[0] == "#":
            continue
        if words[0] == "simerics_path":
            words.pop(0)
            plString = " ".join(words)
            simPath = '"' + plString + '"' + " -run "
        if words[0] == "fluid_model_init":
            fluidSpro_init = (words[1])
        if words[0] == "fluid_model":
            fluidSpro = (words[1])
        if words[0] == "fluid_model_k":
            fluidSpro_k = (words[1])
        if words[0] == "solid_model":
            solidSpro = (words[1])
        if words[0] == "solid_model_k":
            solidSpro_k = (words[1])
        if words[0] == "fluid_surfaces":
            for i in range(1, len(words)):
                fluidSurfs.append((words[i]))
        if words[0] == "solid_surfaces":
            for i in range(1, len(words)):
                solidSurfs.append((words[i]))
        if words[0] == "number_cycles":
            numCycles = int(words[1])

    setupFile.close()
    
    def getBaseName(file):
        name = file.split(".")
        baseName = name[0]
        return baseName
    
    T0 = 350                        # Initial Temp
    Tk = [];                        # Intermediate value
    Tn  = [];                       # Current time step
    T1n = [];                       # Last time step
    T2n = [];                       # Previous time step
    
    def Initialize(T0, Tn, T1n, T2n):
        Tn  = T0.copy()                   
        T1n = T0.copy()                  
        T2n = T0.copy()
        return T0, Tn, T1n, T2n
        
    def shift_timeStep(T0, Tn, T1n, T2n):
        T2n = T1n
        T1n = Tn
        Tn = T0
        return T0, Tn, T1n, T2n

    fluidSproCheck = check_open(fluidSpro, 'r')
    fluidSproCheck.close()

    solidSproCheck = check_open(solidSpro, 'r')
    solidSproCheck.close()

    if (len(fluidSurfs) != len(solidSurfs)):
        print('Number of fluids and solid surfaces are not same!')
        exit()

    else:
        print("Solid and fluid surfaces recognized...")
    
    #for n loop
    sol_flux_dat = []
    fl_flux_dat = []

    sol_q_sum = []
    fl_q_sum = []

    T_sol_array = []
    q_f_array = []
    q_s_array = []

    T_sol_db = []
    
    corr = []
    
    #For k loop    
    sol_flux_dat_k = []
    fl_flux_dat_k = []

    sol_q_sum_k = []
    fl_q_sum_k = []

    T_sol_array_k = []
    q_f_array_k = []
    q_s_array_k = []

    T_sol_db_k = []

    corr_k = []
      
    absolute_residual = []     
    relative_residual = []
    residual_drop = []
     
    for i in range(0, numCycles):
    
        initNum = 0
        resultNum = int(initNum + (i + 1))
        resultStr = str(resultNum)

        solidBaseName = getBaseName(solidSpro)
        solid_integral_name = solidBaseName + "_integrals.txt"
        
        solidBaseName_k = getBaseName(solidSpro_k)
        
        if i == 0:
            fluidBaseName_init = getBaseName(fluidSpro_init)
            fluidBaseName = fluidBaseName_init
            fluidRun = simPath + fluidBaseName + ".spro " + fluidBaseName_init + "_converged.sres"
            solidRun = simPath + solidBaseName + ".spro"

            fl_integral_name = fluidBaseName + "_integrals.txt"

        elif i == 1:
            fluidBaseName = getBaseName(fluidSpro)
            fluidBaseName_init = getBaseName(fluidSpro_init)
            fluidRun = simPath + fluidBaseName + ".spro " + fluidBaseName_init + ".sres"
            solidRun = simPath + solidBaseName + ".sres"

            fl_integral_name = fluidBaseName + "_integrals.txt"
        else:
            fluidBaseName = getBaseName(fluidSpro)
            fluidRun = simPath + fluidBaseName + ".sres"
            solidRun = simPath + solidBaseName + ".sres"

            fl_integral_name = fluidBaseName + "_integrals.txt"
        
        #Extrapolation
        T = extrapolator(i, T0, Tn, T1n, T2n);
        
        print("##############################################")
        print("Time Step: " + str(resultNum) + '\n')
        
    #Fluid Solver
        fluid(fluidRun, fluidBaseName, fluidSurfs, solidSurfs, resultNum, corr);
        
    #Solid Solver
        sol_flux_dat, fl_flux_dat, sol_q_sum, fl_q_sum, T_sol_array, q_f_array, q_s_array, delT_mean, delT_std, mean_Temp = solid(i, solidRun, solidBaseName, solidSurfs, fluidSurfs, solid_integral_name, fl_integral_name, sol_flux_dat, fl_flux_dat, sol_q_sum, fl_q_sum, T_sol_array, q_f_array, q_s_array, delT_mean, delT_std, mean_Temp, resultNum);
                
        for j in range(0, len(fluidSurfs)):
            
            TnFileName = solidBaseName + "_" + solidSurfs[j] + "_temperature" + "_tilda_iter_" + str(resultNum) + ".txt"
        
            Tn = np.genfromtxt(TnFileName)
            Tn = pd.DataFrame(Tn)
                   
            if i == 0:
                
                T0 = Tn.copy()
                
                T0.iloc[:, 3] = T
            
                T0_fileName = solidBaseName + "_" + solidSurfs[j] + "_temperature" + "_iter_0.txt"
                T0.to_csv(T0_fileName, index=False, header=False,  sep=" ")
            
                Residual_vec_0 = Tn.iloc[:, 3] - T0.iloc[:, 3]
                
            else:
                
                Tn1FileName = solidBaseName + "_" + solidSurfs[j] + "_temperature" + "_iter_" + str(resultNum-1) + ".txt"
            
                Tn1 = np.genfromtxt(Tn1FileName)
                Tn1 = pd.DataFrame(Tn1)
                
                Residual_vec_0 = Tn.iloc[:, 3] - Tn1.iloc[:, 3]
        
            #print(Residual_vec_0)
        
        Residual_vec_0 = pd.DataFrame(Residual_vec_0)
        
        #Residual = Tn.copy()
        #Residual.iloc[:, 3] = Residual_vec_0
        
        ResidualFileName = "Residual_iter_" + str(resultNum) + ".txt"
        Residual_vec_0.to_csv(ResidualFileName, index=False, header=False,  sep=" ")
        
        ResidualFileName = "Residual_k_0.txt"
        Residual_vec_0.to_csv(ResidualFileName, index=False, header=False,  sep=" ")
        
        #a = calculate_L2_norm(Residual_vec_0)
        
        #print("Absolute residual: " + str(a.values))
        
        #Initialize the values
        #T0, Tn, T1n, T2n = Initialize(T0, Tn, T1n, T2n);

        Tn_corr = Tn.copy()
        relaxed_value = Tn.copy()
        
        for j in range(0, len(fluidSurfs)):        
            Tn1FileName = solidBaseName + "_" + solidSurfs[j] + "_temperature" + "_iter_" + str(resultNum-1) + ".txt"
        
        Tn1 = np.genfromtxt(Tn1FileName)
        Tn1 = pd.DataFrame(Tn1)
        
        relaxed_value.iloc[:, 3] = omega * Residual_vec_0
            
        Tn_corr.iloc[:, 3] = Tn1.iloc[:, 3] + relaxed_value.iloc[:, 3]
                    
        #print(Tn_corr)
        
        for j in range(0, len(fluidSurfs)):
        
            solidTildaFileName = solidBaseName + "_" + solidSurfs[j] + "_temperature" + "_iter_" + str(resultNum) + ".txt"
        
            Tn_corr.to_csv(solidTildaFileName, index=False, header=False,  sep=" ")
            
            ##for coupling iterations
            solidTildaFileName = solidBaseName_k + "_" + solidSurfs[j] + "_temperature" + "_k_1.txt"
            Tn_corr.to_csv(solidTildaFileName, index=False, header=False,  sep=" ")
            #####
            
            fluidFileName = "temperature_fluid_" + fluidSurfs[j] + ".txt"
        
            solidTildaFile = check_open(solidTildaFileName, 'r')
            numLines = 0
            for line in solidTildaFile:
                numLines = numLines + 1
            solidTildaFile.close()

            fluidFileName_iter = "temperature_fluid_" + fluidSurfs[j] + "_iter_" + str(resultNum) + ".txt"
            write_sim_input(fluidFileName, solidTildaFileName, numLines)
            write_sim_input(fluidFileName_iter, solidTildaFileName, numLines)
                                 
        abs_Tol = 10
        residual_criteria = 1
        Tol = -5
    
        k = 1
        
        while True:
        
            if k > 1:
                if residual_criteria < Tol:
                    print("Residual drop criteria achieved." + "\n")
                    print("Solution is converged in " + str(k) + " iterations.")
                    break
        
            if k > 1:
                if abs_residual_criteria < abs_Tol:
                    print("Absolute residual criteria achieved" + "\n")
                    print("Solution is converged in " + str(k) + " iterations.")
                    break
        
        #for k in range(1, 51):
            
            solidBaseName_k = getBaseName(solidSpro_k)         
            solid_integral_name_k = solidBaseName_k + "_integrals.txt"
            
            fluidBaseName_k = getBaseName(fluidSpro_k)
            fluidBaseName_init = getBaseName(fluidSpro_init)
            fluidRun_k = simPath + fluidBaseName_k + ".spro "
            solidRun_k = simPath + solidBaseName_k + ".spro"

            fl_integral_name_k = fluidBaseName_k + "_integrals.txt"
            
            print("##############################################")
            print("Coupling Iteration: " + str(k) + '\n')
        
            #Fluid Solver
            corr_k = fluid_k(fluidRun_k, fluidBaseName_k, fluidSurfs, solidSurfs, k, corr_k);
                    
            #Solid Solver
            sol_flux_dat, fl_flux_dat, sol_q_sum, fl_q_sum, T_sol_array, q_f_array, q_s_array, delT_mean, delT_std, mean_Temp = solid_k(i, solidRun_k, solidBaseName_k, solidSurfs, fluidSurfs, solid_integral_name_k, fl_integral_name_k, sol_flux_dat_k, fl_flux_dat_k, sol_q_sum_k, fl_q_sum_k, T_sol_array_k, q_f_array_k, q_s_array_k, delT_mean_k, delT_std_k, mean_Temp_k, k);
                
            ###Building absolute residual
            
            for j in range(0, len(fluidSurfs)):
            
                TnFileName = solidBaseName_k + "_" + solidSurfs[j] + "_temperature" + "_tilda_k_" + str(k) + ".txt"
        
                Tn = np.genfromtxt(TnFileName)
                Tn = pd.DataFrame(Tn)
                             
                Tn1FileName = solidBaseName_k + "_" + solidSurfs[j] + "_temperature" + "_k_" + str(k) + ".txt"
            
                Tn1 = np.genfromtxt(Tn1FileName)
                Tn1 = pd.DataFrame(Tn1)
                
                Residual_vec_k = Tn.iloc[:, 3] - Tn1.iloc[:, 3]
                
            Residual_vec_k = pd.DataFrame(Residual_vec_k)
               
            ResidualFileName = "Residual_k_" + str(k) + ".txt"
            Residual_vec_k.to_csv(ResidualFileName, index=False, header=False,  sep=" ")
               
            a = calculate_L2_norm(Residual_vec_k)
        
            print("Absolute residual: " + str(a.values))
            
            ####Building Relative residual
            
            for j in range(0, len(fluidSurfs)):
               
                #Relative residual calculation
                if k == 1:        
                        
                    c = a / a         
                    print("Relative Residual: " + str(c.values))
                                   
                else:
                        
                    TnFileName = solidBaseName_k + "_" + solidSurfs[j] + "_temperature" + "_tilda_k_" + str(k-1) + ".txt"
        
                    Tn = np.genfromtxt(TnFileName)
                    Tn = pd.DataFrame(Tn)
            
                    Tn2FileName = solidBaseName_k + "_" + solidSurfs[j] + "_temperature" + "_k_" + str(k-1) + ".txt"
            
                    Tn2 = np.genfromtxt(Tn2FileName)
                    Tn2 = pd.DataFrame(Tn2)
            
                    Residual_vec_k_2 = Tn.iloc[:, 3] -  Tn2.iloc[:, 3]         
                            
                    b = calculate_L2_norm(Residual_vec_k_2)
                                              
                    c = a / b
                        
                    print("Relative Residual: " + str(c.values))
                    
            rel_residual = c
            
            
            #Bluiding V and W Matrix
            R, C = np.shape(Residual_vec_k)
            
            V = np.zeros((R, k))
            V = pd.DataFrame(V)
            
            VectorV = "08_Vector_V.txt"
            V.to_csv(VectorV, index=False, header=False, sep=" ")

            W = np.zeros((R, k))
            W = pd.DataFrame(W)
            
            VectorW = "09_Vector_W.txt"
            W.to_csv(VectorW, index=False, header=False, sep=" ")
            
            ###Calculating Residual differece to build V vector                       
            for i in range(0, k):

                ResidualFileName = "Residual_k_" + str(i) + ".txt"

                Ri = np.genfromtxt(ResidualFileName)
                Ri = pd.DataFrame(Ri)

                ResidualFileName = "Residual_k_" + str(k) + ".txt"

                Rk = np.genfromtxt(ResidualFileName)
                Rk = pd.DataFrame(Rk)

                Delta_Ri = Ri - Rk

                Delta_Residual = "Delta_Residual_i_" + str(i) + ".txt"
                Delta_Ri.to_csv(Delta_Residual, index=False, header=False,  sep=" ")
                
            q = k-1
            p = R
            
            ###Buidling V vector    
            for i in range(0, k):
        
                VectorV = "08_Vector_V.txt"
                VectorV = np.genfromtxt(VectorV)
                VectorV = pd.DataFrame(VectorV)

                Delta_Residual = "Delta_Residual_i_" + str(i) + ".txt"
                Delta_Residual = np.genfromtxt(Delta_Residual)
                Delta_Residual = pd.DataFrame(Delta_Residual)

                V[-1] = 0.0
                V.iloc[:, 0] = Delta_Residual.iloc[:, 0]

                V = V.shift(periods=1, axis=1).astype(float)
                
                if i == k-1:
                    V = V.drop(V.columns[0], axis=1)
                    #print(V)
                    
                if q > p:                                   #### Generally q <<< p
                    V = V.drop(V.columns[-1], axis=1)
                
                VectorV = "08_Vector_V.txt"
                V.to_csv(VectorV, index=False, header=False, sep=" ")   
            
                        
            ###Calculating Temp difference to build W vector
            for i in range(0, k):
                
                for j in range(0, len(fluidSurfs)):
                
                    TiFileName = solidBaseName_k + "_" + solidSurfs[j] + "_temperature" + "_tilda_k_" + str(k-1) + ".txt"
        
                    Ti = np.genfromtxt(TiFileName)
                    Ti = pd.DataFrame(Ti)
            
                    TkFileName = solidBaseName_k + "_" + solidSurfs[j] + "_temperature" + "_tilda_k_" + str(k) + ".txt"
            
                    Tk = np.genfromtxt(TkFileName)
                    Tk = pd.DataFrame(Tk)
            
                    Delta_Ti = Ti.iloc[:, 3] - Tk.iloc[:, 3]
                    
                    Delta_temperature = "Delta_temp_i_" + str(i) + ".txt"
                    Delta_Ti.to_csv(Delta_temperature, index=False, header=False,  sep=" ")
                
            ###Buidling W vector    
            for i in range(0, k):

                VectorW = "09_Vector_W.txt"
                VectorW = np.genfromtxt(VectorW)
                VectorW = pd.DataFrame(VectorW)

                Delta_temperature = "Delta_temp_i_" + str(i) + ".txt"
                Delta_temperature = np.genfromtxt(Delta_temperature)
                Delta_temperature = pd.DataFrame(Delta_temperature)

                W[-1] = 0.0
                W.iloc[:, 0] = Delta_temperature.iloc[:, 0]

                W = W.shift(periods=1, axis=1).astype(float)
                
                if i == k-1:
                    W = W.drop(W.columns[0], axis=1)
                    #print(W)
                    
                if q > p:                                   #### Generally q <<< p
                    W = W.drop(V.columns[-1], axis=1)
                    
                VectorW = "09_Vector_W.txt"
                W.to_csv(VectorW, index=False, header=False, sep=" ")          
            
            V = V.to_numpy()
            
            ##### QR Decomposition
            Q, R = QR_decomposition(V)
            #print(Q)
            #print(R)
            
            ResidualFileName = "Residual_k_" + str(k) + ".txt"
            Residual = np.genfromtxt(ResidualFileName)
            Residual = pd.DataFrame(Residual)
            Residual = Residual.to_numpy()
            
            Q = pd.DataFrame(Q)
            
            b = np.dot(Q.transpose(), -Residual)
                        
            #print(R)
            #print(b)
                       
            alpha = back_substitution(R, b, k)
            #print(alpha)
            
            alpha = alpha[:k]
            #print(alpha)
                       
            delta_T = np.dot(W, alpha) + Residual
            delta_T = pd.DataFrame(delta_T)
            #print(delta_T)          
            
            ##Correcting the value               
            Tn1FileName = solidBaseName_k + "_" + solidSurfs[j] + "_temperature" + "_k_" + str(k) + ".txt"
           
            Tn1 = np.genfromtxt(Tn1FileName)
            Tn1 = pd.DataFrame(Tn1)
                
            Tk = Tn1.copy()
            
            Tk.iloc[:, 3] = Tn1.iloc[:, 3] + delta_T.iloc[:, 0]
            
            for j in range(0, len(fluidSurfs)):
        
                TkFileName = solidBaseName_k + "_" + solidSurfs[j] + "_temperature" + "_k_" + str(k+1) + ".txt"
                
                Tk.to_csv(TkFileName, index=False, header=False,  sep=" ")
                   
                fluidFileName = "temperature_fluid_" + fluidSurfs[j] + ".txt"
        
                solidTildaFile = check_open(TkFileName, 'r')
                numLines = 0
                for line in solidTildaFile:
                    numLines = numLines + 1
                solidTildaFile.close()

                fluidFileName_iter = "temperature_fluid_" + fluidSurfs[j] + "_k_" + str(k) + ".txt"
                write_sim_input(fluidFileName, TkFileName, numLines)
                write_sim_input(fluidFileName_iter, TkFileName, numLines)

            ###Building the absolute residual
            absolute_residual.append(float(a))   
        
            fileName = "08_abs_residual.txt"
            absolute_residual_df = pd.DataFrame(absolute_residual)
            absolute_residual_df.to_csv(fileName, index=False, header=False,  sep=" ")
        
            ###Building the residual drop
            maximum_value = absolute_residual_df.max()
            resi_drop = absolute_residual_df/maximum_value
            y = np.log(resi_drop)
            #residual_drop.append(float(y))
        
            fileName = "10_Residual_Drop.txt"
            #residual_drop_df = pd.DataFrame(residual_drop)
            y.to_csv(fileName, index=False, header=False,  sep=" ")
            
            ###Stopping criteria
            if i > 0:
                residual_cri = np.genfromtxt(fileName)
                residual_cri = pd.DataFrame(residual_cri)
        
                residual_criteria = residual_cri.iloc[-1, 0]
        
            abs_residual_criteria = absolute_residual_df.iloc[-1, 0]
        
            ###Building the relative residual
            relative_residual.append(float(rel_residual))
        
            fileName = "09_rel_residual.txt"
            relative_residual_df = pd.DataFrame(relative_residual)
            relative_residual_df.to_csv(fileName, index=False, header=False,  sep=" ")

            plt.figure(figsize=(15, 10))
        
            plt.subplot(2, 3, 1)
            plt.scatter(np.linspace(0, k, len(sol_q_sum)), y, color="blue", label="Residual Drop")
            plt.plot(np.linspace(0, k, len(sol_q_sum)), y, color="blue")
            plt.xlabel('Iterations')
            plt.ylabel('Residual drop')
            plt.legend(loc="best")
        
            plt.subplot(2, 3, 2)
            plt.scatter(np.linspace(0, k, len(sol_q_sum)), absolute_residual, color="blue", label="R_ab Temp")
            plt.plot(np.linspace(0, k, len(sol_q_sum)), absolute_residual, color="blue")
            plt.legend(loc="best")
                
            plt.subplot(2, 3, 3)
            #plt.xlim(0, k)
            plt.ylim(-0.5, 3)
            plt.grid()
            plt.scatter(np.linspace(0, k, len(sol_q_sum)), relative_residual, color="red", label="R_rel Temp")
            plt.plot(np.linspace(0, k, len(sol_q_sum)), relative_residual, color="red")
            plt.legend(loc="best")
                             
            plt.subplot(2, 3, 4)
            plt.scatter(np.linspace(0, k, len(sol_q_sum)), sol_q_sum, color="red", label="Solid flux")
            plt.scatter(np.linspace(0, k, len(fl_q_sum)), fl_q_sum, color="blue", label="Fluid flux")
            plt.plot(np.linspace(0, k, len(sol_q_sum)), sol_q_sum, color="red")
            plt.plot(np.linspace(0, k, len(fl_q_sum)), fl_q_sum, color="blue")
            plt.legend(loc="best")
                                              
            plt.subplot(2, 3, 5)
            plt.scatter(np.linspace(0, k, len(sol_q_sum)), delT_mean, color="blue", label="delT_mean")
            plt.plot(np.linspace(0, k, len(sol_q_sum)), delT_mean, color="blue")
            plt.legend(loc="best")
        
            plt.subplot(2, 3, 6)
            plt.ylim(320, 515)
            plt.scatter(np.linspace(0, k, len(sol_q_sum)), mean_Temp, color="blue", label="mean_Temp")
            #plt.plot(np.linspace(0, i, len(sol_q_sum)), mean_Temp, color="blue")
            plt.legend(loc="best")
        
            plt.savefig('Residual_plot_' + str(k) + '.png')
            #plt.show()
        
            k += 1