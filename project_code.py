import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns


#%matplotlib inline

GAMES = 12 # Number of games
N = 25 # Number of responders

# Survey results
survey = pd.read_csv("data/NBA_Survey_Results.csv")

# FiveThurtyEight (=Experts) predictions {elo, Carmelo, RAPTOR}
data_frame = pd.read_csv("data/nba_elo_latest.csv")
five_thirty_eight = data_frame[data_frame["date"] == "2020-01-22"].iloc[:, [4, 5, 8, 9, 14, 15, 20, 21]]

# Matchups
matchups = list((survey["matchup"]).drop_duplicates())
# Actual matchup winners
matchup_winners = ["OKC", "LAL", "MIA", "TOR", "ATL", "CHI", "HOU", "IND", "SAS", "UTA", "DET", "BOS"]

# Majority voting method
def majority():
    maj = []
    for i in range(0, N * GAMES, N):
        away, home = survey.iloc[i][0].split(" @ ")
        P_0 = survey.iloc[i:i + N].groupby(["own"])["own"].count()[0] / N # Percentage of responders who picked the away team
        P_1 = 1 - P_0 # Percentage of responders who picked the home team
        if P_0 > P_1:
            maj.append(away)
        else:
            maj.append(home)
    return maj


# Surprisingly Popular (SP) method
def surprisingly_popular():
    sp = []
    for i in range(0, N * GAMES, N):
        away, home = survey.iloc[i][0].split(" @ ")
        P_0 = survey.iloc[i:i + N].groupby(["own"])["own"].count()[0] / N # Percentage of responders who picked the away team
        P_1 = 1 - P_0 # Percentage of responders who picked the home team
        E_0 = survey.iloc[i:i + N].groupby(["own"])["meta"].mean()[0] # Average of expected result (away team)
        E_1 = survey.iloc[i:i + N].groupby(["own"])["meta"].mean()[1] # Average of expected result (home team)
        if (P_0 - E_0) > (P_1 - E_1):
            sp.append(away)
        else:
            sp.append(home)
    return sp


# Confiedence method
def confidence():
    conf = []
    for i in range(0, N * GAMES, N):
        away, home = survey.iloc[i][0].split(" @ ")
        C_0 = survey.iloc[i:i + N].groupby(["own"])["confidence"].mean()[0] # Average of confidence (away team)
        C_1 = survey.iloc[i:i + N].groupby(["own"])["confidence"].mean()[1] # Average of confidence (home team)
        if C_0 > C_1:
            conf.append(away)
        else:
            conf.append(home)
    return conf


# FiveThirtyEight predictions
def fiveThirtyEight_predictions():
    df = five_thirty_eight.copy()
    elo = []
    carmelo = []
    raptor = []
    for i in range(GAMES):
        home, away = df.iloc[i, 0], df.iloc[i, 1]
    
        # elo predictions
        elo_0, elo_1 = df.iloc[i, 2], df.iloc[i, 3]         
        if elo_0 > elo_1:
            elo.append(home)
        else:
            elo.append(away)
            
        # carmelo predictions
        carmelo_0, carmelo_1 = df.iloc[i, 4], df.iloc[i, 5] 
        if carmelo_0 > carmelo_1:
            carmelo.append(home)
        else:
            carmelo.append(away)
        
        # raptor predictions
        raptor_0, raptor_1 = df.iloc[i, 6], df.iloc[i, 7]   
        if raptor_0 > raptor_1:
            raptor.append(home)
        else:
            raptor.append(away)
        
    # Arranging to match matchup_winners
    e_s = []
    c_s = []
    r_s = []
    for i, matchup in enumerate(matchup_winners):
        if matchup in elo:
            e_s.append(matchup)
        else:
            e_s.append(elo[i])

        if matchup in carmelo:
            c_s.append(matchup)
        else:
            c_s.append(carmelo[i])

        if matchup in raptor:
            r_s.append(matchup)
        else:
            r_s.append(raptor[i])               
    return e_s, c_s, r_s


# Number of results correctly predicted  
def get_score(res):
    score = 0
    for team in matchup_winners:
        if team in res:
            score += 1
    return score


# Bar chart plot of the different methods
def results():
    # Number of results correctly predicted for each method 
    maj_score = get_score(majority())
    SP_score = get_score(surprisingly_popular())
    confidence_score = get_score(confidence())
    elo, carmelo, raptor = fiveThirtyEight_predictions()
    elo_score = get_score(elo)
    carmelo_score = get_score(carmelo)
    raptor_score = get_score(raptor)
    
    lables = ["Majority", "SP", "Confidence", "Elo", "CARMELO", "RAPTOR"]
    scores = [maj_score, SP_score, confidence_score, elo_score, carmelo_score, raptor_score]
    x_pos = np.arange(len(scores))

    plt.bar(x_pos, scores, align='center', alpha=0.8)
    plt.xticks(x_pos, lables)
    for i in range(len(scores)):
        plt.text(i, scores[i] + 0.5, scores[i])
    plt.axhline(y=GAMES, color='r', linestyle='-')
    plt.xlabel("Methods")
    plt.ylabel("Number of correctly predicted games")
    plt.title("Games Results Predictions")
    plt.show()


# Accuracy of predictions for each responder
def accuracy():
    df = survey.copy()
    df["accurate"] = (~ np.logical_xor(df["own"], df["actual"])).astype(int) # Creating new column: Accurate? (1=Yes, 0=No)
    df = df.groupby(["subject", "accurate"])["accurate"].count() # Counting number of correct predictions
    accuracy_dict = dict.fromkeys(np.arange(1, N + 1), 0)
    for i in range(1, N + 1):
        accuracy_dict[i] = df[i][1] / GAMES # Accuracy precentage
    return accuracy_dict.values()



# Plot of accuracy of predictions for each responder
def responders_accuracy():
    res_accuracy = list(accuracy())
    ax = sns.distplot(res_accuracy, hist=True, rug=True)
    ax.set(xlabel='Accuracy', ylabel='Number of responders')
    ax.set_title("Responders accuracy")
    
    # Box plot accuracy
    df = pd.DataFrame(survey["knowledge_level"].iloc[0:N])
    df["accuracy"] = res_accuracy
    ax = sns.catplot(x="knowledge_level", y="accuracy", kind="box", data=df)
    ax.set(xlabel='Knowledge level', ylabel='Accuracy')
    plt.show()


# Accuracy of methods for each matchup
def methods_map():
    # Creating data frame
    games = []
    winners = []
    for matchup, winner in zip(matchups, matchup_winners):
        for _ in range(6):
            games.append(matchup)
            winners.append(winner)
    df = pd.DataFrame()
    df["matchup"] = games
    df["actual"] = winners
    df["method"] = ["Majority", "SP", "Confidence", "Elo", "Carmelo", "RAPTOR"] * GAMES
    
    # Getting predictions
    maj = majority()
    sp = surprisingly_popular()
    conf = confidence()
    elo, carmelo, raptor = fiveThirtyEight_predictions()
    predictions = []
    for i in range(GAMES):
        predictions.append(maj[i])
        predictions.append(sp[i])
        predictions.append(conf[i])
        predictions.append(elo[i])
        predictions.append(carmelo[i])
        predictions.append(raptor[i])
    df["prediction"] = predictions
    
    # Accurate? (Yes=1, No=0)
    df["accurate"] = (df["prediction"] == df["actual"]).astype(int)

    # Draw a heatmap with the numeric values in each cell
    h_map = df.pivot("matchup", "method", "accurate")
    f, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(h_map, annot=True, fmt="d", cbar=False, ax=ax)