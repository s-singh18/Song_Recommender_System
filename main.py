#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:48:29 2020

Song Recommendeskr system

Machine learning algorithms separated into content based and 
collaborative filtering methods.  

Content based: similarity of item attributes

Collaborative methods: calculate similarity from interactions


Use three questions to determine user's preference and recommend them a song.

NOTE: Original data set altered slightly to remove profane language from song titles

@author: sss
"""
import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

def Print_Menu():
    print("        Recommender System Menu        ")
    print("---------------------------------------")
    print("|                                     |")
    print("| 1. Take the quiz                    |")
    print("| 2. Exit                             |")
    print("|                                     |")
    print("---------------------------------------")


    
def Question(df):
    rand_num_item  = [0, 0, 0, 0]
    flag = 1
    while (flag == 1):
        #generate random values
        for ind, val in enumerate(rand_num_item):
            rand_num_item[ind] = random.randrange(0,len(df))
        
        flag = 0
        #check if they're all different
        for ind in range(len(rand_num_item) - 1):
            if rand_num_item[ind] == rand_num_item[ind+1]:
                flag = 1
    
    print("Out of the below listed songs, which one is your favorite\n:")
    for ind, val in enumerate(rand_num_item):
        num = str(ind+1)
        print(num+". "+df[df.columns[0]][val])
        
    question_ans = input(":")
    ans_ind = rand_num_item[int(question_ans)-1]
    return ans_ind
    

    



if __name__ == "__main__":
    df = pd.read_csv("top10s.csv", encoding = "cp437")
    df = df.drop(['Unnamed: 0'], axis = 1)
    menu_inp = 0
    while (menu_inp != 2):
        Print_Menu()
        menu_inp = input("Select a menu option: ")
        menu_inp = int(menu_inp)
        if menu_inp == 1:
            ans_ind = Question(df)
            arr1 = df.iloc[ans_ind, 4:].values.reshape(1, -1)
            df = df.drop([ans_ind])
            cos_sim_arr = cosine_similarity(arr1, df.iloc[:,4:])
            rec_ind = np.argmax(cos_sim_arr)
            print("Recommended for you: \"" + df.iloc[rec_ind,0] + "\" by " 
                  + df.iloc[rec_ind,1] + "\n")
        else:
            print("Exiting..")
            
    
    