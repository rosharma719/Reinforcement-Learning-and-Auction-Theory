import numpy as np
from matplotlib import pyplot as plt 
import random
import math

class agent: 
    def __init__(self, value, cheapness, hindsight, training_episodes): 
        self.value = value
        self.epsilon = 1
        self.adjustment_factor = 0.000025
        self.hindsight = hindsight # Between 0 and 1; higher hindsight emphasizes bias towards past info VS future projections
        self.cheapness = cheapness # between 0 and 1; higher cheapness emphasizes value over probability 

        # Integer data: Bid value, #wins, #games bidded with this value
        self.int_data = np.zeros((value, 3), dtype=int)
        for x in range(value): 
            self.int_data[x, 0] = x

        # Floating-point data: Win rate, Q-values
        self.float_data = np.ones((value, 3))  
        for x in range(self.value): 
            self.float_data[x, 1] = random.random()

        # Defining loss thresholds
        self.consecutive_losses = 0 # Track consecutive Losses
        self.losing_streak_threshold = training_episodes * self.adjustment_factor # Defines the winning streak
        
        self.consecutive_wins = 0 # Track consecutive wins 
        self.winning_streak_threshold = training_episodes * self.adjustment_factor

    
    def getBid(self, agents): 
        if np.random.rand() < self.epsilon:
            # Explore
            return random.randint(0, self.value - 1)
        else:
            return np.argmax(self.float_data[:, 1])  
            # Exploit: Argmax over Q-values
        

    def updateAgent(self, i, training_episodes, episode, won, agents):
        self.int_data[i, 2] += 1  
        cutoff = 0.8*training_episodes
        alpha = math.pow(episode/training_episodes, 2)

        # Updating actual winrate
        self.float_data[i, 0] = self.int_data[i, 1]/self.int_data[i, 2]

        # Predicting bids
        predictedArr = []
        for x in agents: 
            if x != self:
                predictedArr.append(np.argmax(x.float_data[:,1]))
        
        #Updating projected winrate 
        for x in range(len(self.float_data)): 
            self.float_data[x, 2] = i/max(predictedArr)
        
        #The actual Q-Equation
        if won: 
            self.float_data[i, 1] = math.pow(self.float_data[i, 0], self.hindsight)* math.pow(self.float_data[i, 2], 1-self.hindsight) * (self.value - i)/self.value
            self.int_data[i, 1] += 1
            self.consecutive_losses = 0
            self.consecutive_wins += 1
            if self.consecutive_wins >= self.winning_streak_threshold:
                current_best_bid = np.argmax(self.float_data[:, 1])
                for bid in range(0, current_best_bid):
                    # Adjust Q-values for lower bids
                        self.float_data[bid, 1] += 0.5*self.adjustment_factor
                for bid in range(current_best_bid + 1, self.value):
                    # Adjust Q-values for higher bids
                    self.float_data[bid, 1] += 0.5*self.adjustment_factor
        else: 
            self.float_data[i, 1] = math.pow(self.float_data[i, 0], self.hindsight)* math.pow(self.float_data[i, 2], 1-self.hindsight) * (self.value - i)/self.value
            self.consecutive_wins = 0
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.losing_streak_threshold:
                current_best_bid = np.argmax(self.float_data[:, 1])
                for bid in range(current_best_bid + 1, self.value):
                    # Adjust Q-values for higher bids
                    self.float_data[bid, 1] += self.adjustment_factor
                for bid in range(0, current_best_bid):
                    self.float_data[bid, 1] -= self.adjustment_factor


        # Updating decay rate
        decay_rate = 1/(cutoff)
        if self.epsilon > 0: 
            self.epsilon = self.epsilon-decay_rate

class game: 
    def __init__(self, agents, gm, training_episodes, episode): 
        self.agents = agents
        self.gameMode = gm

        #Getting the bids for each agent
        self.bids = []
        for x in agents: 
            bid = int(x.getBid(agents))
            self.bids.append(bid)
            
        winner = self.getWinner(self.bids, self.gameMode)
        self.winning_value = self.bids[winner]

        #UPDATE AGENTS
        for x in range(len(agents)): 
            ind = self.bids[x]
            agents[x].updateAgent(ind, training_episodes, episode, x == winner, agents)


    def getWinner(self, bids, gameMode): 
        if gameMode == "standard": 
            max_bid = max(bids)
            num_max_bids = bids.count(max_bid)
            if num_max_bids > 1:
                tied_agents = [i for i, x in enumerate(bids) if x == max_bid]
                random.shuffle(tied_agents)
                return random.choice(tied_agents)
            else:
                return bids.index(max_bid)
        elif gameMode == "second-price":
            # Sort bids and find the second unique maximum value
            unique_sorted_bids = sorted(set(bids), reverse=True)
            if len(unique_sorted_bids) >= 2:
                secondMax = unique_sorted_bids[1]
                secondMaxIndices = [i for i, x in enumerate(bids) if x == secondMax]
                return random.choice(secondMaxIndices)  # Randomly choose if there are multiple second max
            else:
                # If there is no second max (all bids are equal), pick a random winner
                return random.randint(0, len(bids) - 1)

def train(): 
    training_episodes = 10000
    values = "standard" #Value generation either "random" (from 50 to 100) or "standard" (all 100)
    cheapness = "standard" #Random generation of emphasis on value VS probability of winning 
    numAgents = 3
    gameMode = "second-price" # Gamemode either "second-price" (second-highest price) or "standard"
    hindsight = "standard" # Hindsight either "random" or "standard"

    agents = []
    for x in range(numAgents): 
        v = makeValues(values)
        c = getCheapness(cheapness)
        h = getHindsight(hindsight)
        agents.append(agent(v, c, h, training_episodes))
    gameCount = 1

    #Prep for some summary stats
    bidArr = np.empty((0, len(agents)), int)
    winArr = []
    win_rates = []  
    all_bids = []   
    
    #Running the game
    for episode in range(training_episodes): 
        #Create a new game, with parameters for # of agents and gamemode 
        g = game(agents, gameMode, training_episodes, episode)


        # Collecting statistics
        bidArr = np.vstack((bidArr, g.bids))
        winArr.append(g.winning_value)
        gameCount += 1
        agent_wins = [agent.int_data[:, 1].sum() for agent in agents]
        agent_bids = g.bids
        for x in range(len(agent_wins)): 
            agent_wins[x] = agent_wins[x]/(episode+1)
        win_rates.append(agent_wins)
        all_bids.append(agent_bids)
   
    #More summary debug and stat info
    winPlot(winArr)
    
    getWinRates(win_rates, all_bids, training_episodes, agents, values, cheapness, gameMode)

def getWinRates(win_rates, all_bids, training_episodes, agents, values, cheapness, gameMode):
    win_rates = np.array(win_rates)
    all_bids = np.array(all_bids)

    # Plots win rates of each agent over episodes
    
    for i, agent_wins in enumerate(win_rates.T):
        plt.plot(range(training_episodes), agent_wins, marker = '.', linestyle = '', label=f"Agent {i}")

    plt.xlabel('Episodes')
    plt.ylabel('Win Rate Per Agent')
    plt.title('Win Rates Over Time')
    plt.legend()
    plt.show()

    # Plots average bids of each agent over episodes
    for i, agent_bids in enumerate(all_bids.T):
        plt.plot(range(training_episodes), agent_bids, marker = '.', linestyle = '', label=f"Agent {i}")

    plt.xlabel('Episodes')
    plt.ylabel('Bid Value')
    plt.title('Average Bid Value Per Game')
    plt.legend()
    plt.show()

    # Displays agent's value and games won by agents
    for idx, ag in enumerate(agents):
        print(f"Agent {idx} value:", ag.value)
        print(f"Agent {idx} cheapness:", ag.cheapness)
        print(f"Agent {idx} hindsight bias:", ag.hindsight)
        print(f"Games won by Agent {idx}:", ag.int_data[:, 1].sum(), "of", training_episodes)
        wins = agents[idx].int_data[:, 1].sum()

        # Calculate the total value of wins (number of wins multiplied by agent's value)
        total_win_value = wins * agents[idx].value

        
        if len(all_bids) == training_episodes and len(win_rates) == training_episodes:
            # Calculate the total of winning bids, count of winning episodes, and average winning bid
            winning_bids = [all_bids[episode][idx] 
                            for episode in range(training_episodes) 
                            if idx == np.argmax(all_bids[episode])]
            total_winning_bids = sum(winning_bids)
            winning_episodes_count = len(winning_bids)
            average_winning_bid = np.mean(winning_bids) if winning_episodes_count > 0 else 0
            aggregate_payoff = total_win_value - total_winning_bids

            # Print the result, total winning bids, and average winning bid for each agent
            print(f"Agent {idx}: Average Winning Bid = {average_winning_bid}")
            print(f"Agent {idx}: Aggregate Payoff = {aggregate_payoff}")

def winPlot(list_1d):
    # The length of x_values should match the number of games played
    num_games = len(list_1d)
    x_values = range(1, num_games + 1)

    # Plot the list of winning bids
    plt.plot(x_values, list_1d, marker='.', linestyle = '', color='blue', label='Winning Bid')

    # Adding labels and title
    plt.xlabel('Game Number')
    plt.ylabel('Bid Value')
    plt.title('Winning Bids Over Time')
    plt.legend()

    # Show the plot
    plt.show()

    # Call the plotting function with actual data from the training


def makeValues(values): 
    if values == "standard": 
        return(100)
    else: 
        v = random.randint(50, 100) 
        return(v)  

def getCheapness(c): 
    if c == "standard": 
        return(0.6)
    else: 
        return(random.random())


def getHindsight(h): 
    if h == "standard": 
        return(1)
    else: 
        return(0.2*random.random())

train()
#Edit the train method

