# Importing libraries
import numpy as np, pandas as pd, datetime
import random
import math
import pickle
    
#Loading the foundational dictionaries' pickle files for SLB Threshold
file1 = open("market_state_scaled_dummy_dict.pickle", "rb")
ms_dict = pickle.load(file1)
file1.close()

file2 = open("labeled_dataset_thesis_params3.pickle", "rb")
labeled_df= pickle.load(file2)
file2.close()

rewards_df=labeled_df[['Date', 'ticker','Due_Dividend','Rel_ExDD','Buy_Price', 'Qty','Trade P/L', 'sell_trigger', 'Trade Max_P/L', 
                       'Trade P/L %','Trade Max_P/L %', 'cash_fraction_used', 'sell_date', 'Label']].copy()

trading_dates=list(ms_dict.keys())
    
#List of stocks picked for this analysis
stocks_list=list(list(ms_dict.values())[0].ticker)
    
#Market state vector initial state for the 1st trading day on or after 1 Jan 2014
ms_initial=ms_dict.get(trading_dates[0]) #(pd.to_datetime(datetime.date(2014,4,7)))

#Initial internal state vector is array of cash on hand
initial_cash=1.0 #np.array([[1] for i in range(len(ms_initial))])

#Initial sell dictionary; sell_date:new_cash
sell_dict_initial={'sell_date':'new_cash'}

#Initial state is [market state,array of internal state, timestamp]
initial_state=[ms_initial,initial_cash, sell_dict_initial] #np.hstack((np.array(ms_initial),initial_int_state_vec))
          
#Create list of columns in the market state vector without OHQ, Cash & rel_buy_date_to_exdd
training_cols=['Due_Dividend', 'p_Open', 'p_High', 'p_Low', 'p_Close', 'p2_Close', 'p_Volume', 'Dividend/p_Pc', 'Weekday', 'Month',
 'Week', 'p_Open/p_Close', 'p_Low/p_Close', 'p_High/p_Close', 'Open/p_Close', 'p2_Close/p_Close', 'TTO', 'TTO_MA5', 'TTO_MA45', 'ATTO_5',
 'ATTO_45', 'Beta_22', 'Beta_64', 'MACD_short', 'MACD_Signal_short', 'MACD_hist_short', 'MACD_long', 'MACD_Signal_long',  'MACD_hist_long', 'PPO_short', 'PPO_long', 'Momentum_long', 'Momentum_short', 'StochFast_K', 'StochFast_D', 'StochSlow_K',  'StochSlow_D', 'RSI', 'Williams_R', 'ADO', 'OBV', 'CCI', 'Bias', '10_day_SMA/p_Close', '12_day_EMA/p_Close', '10_day_WMA/p_Close',
 '10_day_SMA', '12_day_EMA', '10_day_WMA', 'Quarter_1', 'Quarter_2', 'Quarter_3', 'Quarter_4', 'Rel_ExDD_-3', 'Rel_ExDD_-2',  'Rel_ExDD_-1', 'Rel_ExDD_Invalid']

class market():
    
    def __init__(self):
        """initialise your state and define your action space and initial state"""
        #I will start with 1 action per stock being possible for investing 100% of cash on hand in that stock the NEXT DAY
        #Each possible action is a decision to either buy the i'th stock or not. 
        #Hence, number of possible actions in a state=number of stocks 
        self.action_space = [ticker for ticker in stocks_list]
        self.action_space.append(-1) #keeping the action of not buying anything on a day
        self.state_init = initial_state #initial state to reset back to 07 April 2014

        # Start the first round
        self.reset()
    
    #Create function to get date of state
    def get_date(self,state):
        #tup_vec=state_tuple #tuple([tuple(vector[:-3]) for vector in encoded_state])
        state_date=state[0].Date[0] #get trading day corresponding to the market state
        return state_date
    
    #Create function to fetch the next day's market vector given a vector from ms_dict
    def next_market_vector(self, state):
        today= self.get_date(state) #get trading day corresponding to the state
        next_trading_date=trading_dates[trading_dates.index(today)+1] #get date of next trading day
        #get next day's market vector
        next_day_ms=ms_dict.get(next_trading_date)
        return next_day_ms
    
    def state_encoder(self, state): #state is [tuple,array]
        state_encod=state[0][training_cols]
        return state_encod
        
    def get_ticker_from_action(self,action): #get ticker corresponding to agent's action
        action_ticker=self.action_space[action]
        return action_ticker
    
    def get_valid_actions(self,state):
        if state[1]<=0: #buy stocks only if cash on hand is more than zero
            valid_actions=[-1]
        elif len(state[0][(state[0]['Rel_ExDD_-1']==1) | (state[0]['Rel_ExDD_-2']==1) | (state[0]['Rel_ExDD_-3']==1)])==0:
            valid_actions=[-1]
        else:
            valid_actions=list(state[0][(state[0]['Rel_ExDD_-1']==1) | (state[0]['Rel_ExDD_-2']==1) | (state[0]['Rel_ExDD_-3']==1)].ticker)
            valid_actions.append(-1) #keeping the action of buying nothing
        return valid_actions
   
    def action_success_fn(self,action,state): #state is [tuple,array, timestamp]
        #This receives 1 ticker as the action to buy; need to go ahead and put all the cash-on-hand in this ticker
        #The action is decided based on today's state vector; however, it is applied only based on TOMORROW's state vector
        #Split encoded state into ms_vec & int_state_vec
        action_tkr=action #self.get_ticker_from_action(action)
        ms_vec1=state[0] 
        #int_state_vec1=state[1]
        today=ms_vec1.Date[0]
        cash_on_hand=state[1] #int_state_vec1[0][1] #at end of today
        sell_dict=state[2] #getting the sell_dictionary
        #Get next_day_sm & next_day_ms using next_market_vector(state) #ms_vec)
        next_day_ms=self.next_market_vector(state)#ms_vec1)
        #int_state_vec2=int_state_vec1.copy()
        step_reward=0
        step_dividend=0
        trig_count=[0,0,0,0,0,0,0] #7th trigger count for the case of chosen action not within 3 days of ex-dividend date
        profit_loss=0
        dividend=0
        p_buy=0
        step_label=[]
        cash_invested=0
        
        #Update cash_on_hand only upto the initial capital since we are not reinvesting profits
        #STOP AND INSERT THE CONDITION WHICH WILL IMPOSE REWARD TO BE ADDED TO CASH ON HAND ONLY WHEN CURRENT_STATE.DATE IS SELL_DATE
        if ms_vec1.Date[0] in sell_dict.keys():
            new_cash_on_hand=0.0
            new_cash_on_hand=min(cash_on_hand+sell_dict.get(ms_vec1.Date[0]),initial_cash) 
            cash_on_hand=new_cash_on_hand
        
        if action_tkr!=-1:
            #get ticker corresponding to chosen action
            #assume the action==ticker from action space
                
            #get Trade P/L % corresponding to ticker-date combination of chosen action
            #update step_reward, step_dividend, trig_count, real_loss
            if len(rewards_df[(rewards_df.Date==today) & (rewards_df.ticker==action_tkr)])>0:
                cash_invested=min(cash_on_hand/3,initial_cash/3)
                #cash_on_hand*list(rewards_df[(rewards_df.Date==today) & (rewards_df.ticker==action_tkr)]['cash_fraction_used'])[0]
                sell_date=list(rewards_df[(rewards_df.Date==today) & (rewards_df.ticker==action_tkr)]['sell_date'])[0]
                profit_loss=list(rewards_df[(rewards_df.Date==today) & (rewards_df.ticker==action_tkr)]['Trade P/L %'])[0]
                step_label.append(list(rewards_df[(rewards_df.Date==today) & (rewards_df.ticker==action_tkr)]['Label'])[0])
                #Update sell_dict
                if sell_date in sell_dict.keys():
                    sell_dict[sell_date]+=cash_invested*(1+profit_loss)
                    #td[key1]+=rew
                else:
                    #td[key1]=rew
                    sell_dict[sell_date]=cash_invested*(1+profit_loss)
                cash_on_hand-=cash_invested
                #coh_today=cash_on_hand-cash_invested
                step_reward=cash_invested*profit_loss

                #get sell trigger corresponding to ticker-date combination
                trig_count[int(list(rewards_df[(rewards_df.Date==today) & (rewards_df.ticker==action_tkr)].sell_trigger)[0]-1)]+=1
                if int(list(rewards_df[(rewards_df.Date==today) & (rewards_df.ticker==action_tkr)].sell_trigger)[0]) in [3,4,5,6]:
                    dividend=list(rewards_df[(rewards_df.Date==today) & (rewards_df.ticker==action_tkr)]['Due_Dividend'])[0]
                    p_buy=list(rewards_df[(rewards_df.Date==today) & (rewards_df.ticker==action_tkr)]['Buy_Price'])[0]
                    step_dividend+=cash_invested*dividend/p_buy
            else:
                step_label.append('Loss')
                step_reward=-0.5 #loss of 50% if the chosen action is not within 3 days of ex-dividend date
                trig_count[6]+=1
                
            #Update cash_on_hand only upto the initial capital since we are not reinvesting profits
            #new_cash_on_hand=0.0
            #new_cash_on_hand=min(cash_on_hand+step_reward,initial_cash) 
            #cash_on_hand=new_cash_on_hand
        else:
            step_label.append('Profit') #if action is -1
        
        #Create next_int_state_vec using updated (OHQ, cash-on-hand,Buy_Rel_ExDD)
        #for vector in int_state_vec2:
        #    vector=cash_on_hand
        next_state=[next_day_ms,cash_on_hand, sell_dict] #int_state_vec2]
        return next_state, step_label[0],step_reward, step_dividend,trig_count
   
    def reset(self):
        return self.state_init
    
    def reset_random(self):
        #Choose random trading day from trading_dates less than 1/1/2019
        random_date=trading_dates[np.random.randint(0,trading_dates.index(pd.to_datetime(datetime.date(2018,12,31))))]
        #Market state vector for this date
        ms_random=ms_dict.get(random_date)
        random_state=[ms_random,initial_cash]
        return random_state