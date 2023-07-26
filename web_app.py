import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(
    page_title = '2022 FIFA WORLD CUP PREDICTIONS',
    page_icon = '⚽',
)

#Import Dataset "wcmatches_history.csv" and clean the data for analysis
wc_games = pd.read_csv('538_wc_matches.csv', sep=';')
wc_games = wc_games.drop(columns=['league_id', 'league', 'spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2', 'score1', 'score2', 'xg1', 'xg2', 'nsxg1', 'nsxg2', 'adj_score1', 'adj_score2'])

# Import Dataset with 2022 FIFA World Cup's Data
data = pd.read_excel('DadosCopaDoMundoQatar2022.xlsx', sheet_name='selecoes').rename(columns={'NomeEmIngles': 'team'}).set_index('team').rename(index={'United States': 'USA'})

# Create list of teams
teams = []
teams = [team for team in data.index if team not in teams]

# Import Dataset "international_matches.csv" and clean the data for analysis
hist_games = pd.read_csv('international_matches_filtered.csv', sep=';')

# Transform matches' dates into years
for i in hist_games.index:
    hist_games.at[i, 'date'] = hist_games['date'][i][-4:]
hist_games = hist_games.rename(columns={'date': 'year'})
hist_games['year'] = hist_games['year'].astype(int)
hist_games = hist_games.sort_values('year', ascending=False)

# Create Dataframe with team's status
columns = ['fifa_rank', 'fifa_points', 'goalkeeper_score', 'defense_score', 'offense_score', 'midfield_score']
team_stats = pd.DataFrame(index= teams, columns= columns)

for team in team_stats.index:
    for i in hist_games.index:
        if team == hist_games['home_team'][i]:
            team_stats['fifa_rank'][team] = hist_games['home_team_fifa_rank'][i]
            team_stats['fifa_points'][team] = hist_games['home_team_total_fifa_points'][i]
            team_stats['goalkeeper_score'][team] = hist_games['home_team_goalkeeper_score'][i]
            team_stats['defense_score'][team] = hist_games['home_team_mean_defense_score'][i]
            team_stats['offense_score'][team] = hist_games['home_team_mean_offense_score'][i]
            team_stats['midfield_score'][team] = hist_games['home_team_mean_midfield_score'][i]
            break
        elif team == hist_games['away_team'][i]:
            team_stats['fifa_rank'][team] = hist_games['away_team_fifa_rank'][i]
            team_stats['fifa_points'][team] = hist_games['away_team_total_fifa_points'][i]
            team_stats['goalkeeper_score'][team] = hist_games['away_team_goalkeeper_score'][i]
            team_stats['defense_score'][team] = hist_games['away_team_mean_defense_score'][i]
            team_stats['offense_score'][team] = hist_games['away_team_mean_offense_score'][i]
            team_stats['midfield_score'][team] = hist_games['away_team_mean_midfield_score'][i]
            break

team_stats['fifa_rank'] = team_stats['fifa_rank'].astype(int)
team_stats['fifa_points'] = team_stats['fifa_points'].astype(int)
team_stats = team_stats.sort_values('fifa_points', ascending= False)

team_stats.at['Qatar', 'goalkeeper_score'] = round((team_stats['goalkeeper_score']['Saudi Arabia'] * team_stats['fifa_points']['Qatar']) / team_stats['fifa_points']['Saudi Arabia'], 1)
team_stats.loc['Qatar', 'defense_score'] = round((team_stats['defense_score']['Saudi Arabia'] * team_stats['fifa_points']['Qatar']) / team_stats['fifa_points']['Saudi Arabia'], 1)
team_stats.loc['Qatar', 'offense_score'] = round((team_stats['offense_score']['Saudi Arabia'] * team_stats['fifa_points']['Qatar']) / team_stats['fifa_points']['Saudi Arabia'], 1)
team_stats.loc['Qatar', 'midfield_score'] = round((team_stats['midfield_score']['Saudi Arabia'] * team_stats['fifa_points']['Qatar']) / team_stats['fifa_points']['Saudi Arabia'], 1)
team_stats.loc['Tunisia', 'goalkeeper_score'] = round((team_stats['goalkeeper_score']['Canada'] * team_stats['fifa_points']['Tunisia']) / team_stats['fifa_points']['Canada'], 1)

# Find average goals per match in previous FIFA World Cups
tournaments = hist_games.groupby('tournament').mean(numeric_only= True)
mgoals = (tournaments['home_team_score']['FIFA World Cup'] + tournaments['away_team_score']['FIFA World Cup'])

# Remove matches prior to 2018 from history, to use only matches from the last 5 years
for i in range(len(hist_games['year'])):
    if hist_games['year'][i] < 2018:
        hist_games = hist_games.drop(i)
hist_games = hist_games.reset_index(drop=True)

#Remove Friendly games from dataset
for i in hist_games.index:
    if hist_games['tournament'][i] == 'Friendly':
        hist_games = hist_games.drop(index= i)
hist_games = hist_games.reset_index(drop=True)

# Define function to identify average of goals in previous matches between 2 teams, if there's none in history return World Cup's avg
def avg_goals(team1, team2):
    if team1 not in teams:
        raise ValueError(f'{team1} not  in the  World Cup!')
    elif team2 not in teams:
        raise ValueError(f'{team2} not  in the  World Cup!')

    goals = 0
    matches = 0

    for i in hist_games.index:
        if (hist_games['home_team'][i] == team1 and hist_games['away_team'][i] == team2) or (hist_games['home_team'][i] == team2 and hist_games['away_team'][i] == team1):
            goals += hist_games['home_team_score'][i] + hist_games['away_team_score'][i]
            matches += 1

    try:
        return goals / matches
    except:
        return mgoals

# Define function to identify the power of each team playing
def lam(team1, team2):
    goals = avg_goals(team1, team2)
    fifa1, fifa2 = team_stats['fifa_points'][team1], team_stats['fifa_points'][team2]
    off1, off2 = team_stats['offense_score'][team1], team_stats['offense_score'][team2]
    def1, def2 = team_stats['defense_score'][team1], team_stats['defense_score'][team2]
    mid1, mid2 = team_stats['midfield_score'][team1], team_stats['midfield_score'][team2]
    gk1, gk2  = team_stats['goalkeeper_score'][team1], team_stats['goalkeeper_score'][team2]
    
    pwr1 = (fifa1 / fifa2) * ((0.9 * off1 + 0.1 * mid1) / (0.8 * def2 + 0.2 * gk2))
    pwr2 = (fifa2 / fifa1) * ((0.9 * off2 + 0.1 * mid2) / (0.8 * def1 + 0.2 * gk1))

    l1 = goals * pwr1 / (pwr1 + pwr2)
    l2 = goals * pwr2 / (pwr1 + pwr2)

    return l1, l2

# Define function to identify the result of each match
def result(goals1, goals2):
    if goals1 > goals2:
        return 'W'
    elif goals2 > goals1:
        return 'L'
    else:
        return 'D'

# Define function to distribute the points based on matches' results
def points(goals1, goals2):
    rst = result(goals1, goals2)
    if rst == 'W':
        pts1, pts2 = 3, 0
    if rst == 'L':
        pts1, pts2 = 0, 3
    if rst == 'D':
        pts1, pts2 = 1, 1
    return pts1, pts2

# Define function to simulate each game
def game(team1, team2):
    l1, l2 = lam(team1, team2)
    goals1 = int(np.random.poisson(lam=l1 , size=1))
    goals2 = int(np.random.poisson(lam=l2 , size=1))
    gd1 = goals1 - goals2
    gd2 = -gd1
    rst = result(goals1, goals2)
    pts1, pts2 = points(goals1, goals2)
    scoreboard = f'{goals1}x{goals2}'
    return {'goals1': goals1, 'goals2': goals2, 'gd1': gd1, 'gd2': gd2, 
            'pts1': pts1, 'pts2': pts2, 'rst': rst, 'scoreboard': scoreboard}

#  Define function to calculate the probabilities of the scoreboard for each game
def distribution(avg):
    probs = []
    for i in range(7):
        probs.append(poisson.pmf(i, avg))
    probs.append(1 - sum(probs))
    return pd.Series(probs, index= ['0', '1', '2', '3', '4', '5', '6', '7+'])

# Define probabilities of each game's result
def match_probs(team1, team2):
    l1, l2 = lam(team1, team2)
    d1, d2 = distribution(l1), distribution(l2)
    matrix = np.outer(d1, d2)
    
    sort = np.sort(matrix)
    max_values = []
    for i in sort:
        for j in i:
            max_values.append(j)
    max_values.sort(reverse= True)
    scoreboards = []
    for i in range(5):
        a = int(np.where(matrix == max_values[i])[0])
        b = int(np.where(matrix == max_values[i])[1])
        scoreboards.append((f'{max_values[i] * 100 :.1f}%', f'{a}x{b}'))


    w = np.tril(matrix).sum() - np.trace(matrix)
    l = np.triu(matrix).sum() - np.trace(matrix)
    d = np.trace(matrix)

    probs = np.around([w, d, l], 3)
    probsp = [f'{100 * i :.1f}%'  for i in  probs]
    
    names = ['0', '1', '2', '3', '4', '5', '6', '7+']
    matrix = pd.DataFrame(matrix, columns = names, index = names)
    matrix.index = pd.MultiIndex.from_product([[team1], matrix.index])
    matrix.columns = pd.MultiIndex.from_product([[team2], matrix.columns])

    return {'probabilities': probsp, 'scoreboards': scoreboards, 'matrix': matrix}

######## START OF APP

st.markdown("<h1 style='text-align: center; color: blue;'>2022 FIFA WORLD CUP PREDICTIONS</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>🏆 CS50 FINAL PROJECT 🏆</h1>", unsafe_allow_html=True)

st.markdown('---')
st.markdown("## ⚽ Matches' Probabilities")

teams_list1 = list(data.index)
teams_list1.sort()
teams_list2 = teams_list1.copy()

col1, col2 = st.columns(2)
team1 = col1.selectbox('Choose first team', teams_list1)
teams_list2.remove(team1)
team2 = col2.selectbox('Choose second team', teams_list2, index= 1)
st.markdown('---')

match = match_probs(team1, team2)
prob = match['probabilities']
matriz = match['matrix']
scoreboards = match['scoreboards']

col1, col2, col3, col4, col5 = st.columns(5)
col1.image(data.loc[team1, 'LinkBandeiraGrande'])  
col2.metric(team1, prob[0])
col3.metric('Draw', prob[1])
col4.metric(team2, prob[2]) 
col5.image(data.loc[team2, 'LinkBandeiraGrande'])

st.markdown('---')
st.markdown("## 📊 Scoreboards' Probabilities") 

def aux(x):
	return f'{str(round(100*x,1))}%'
st.table(matriz.applymap(aux))

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(scoreboards[0][0], scoreboards[0][1])
col2.metric(scoreboards[1][0], scoreboards[1][1])
col3.metric(scoreboards[2][0], scoreboards[2][1])
col4.metric(scoreboards[3][0], scoreboards[3][1])
col5.metric(scoreboards[4][0], scoreboards[4][1])

st.markdown('---')
st.markdown("## 🌍 Probabilities of Group's Phase Matches") 

wc_games['Group'] = None
for i in wc_games.index:
    wc_games.at[i, 'Group'] = data['Grupo'][wc_games['team1'][i]]

wc_games['Win 1'] = None
wc_games['Draw'] = None
wc_games['Win 2'] = None

for i in wc_games.index:
    team1, team2 = wc_games['team1'][i], wc_games['team2'][i]
    w, d, l = match_probs(team1, team2)['probabilities']
    wc_games['Win 1'][i] = w
    wc_games['Draw'][i] = d
    wc_games['Win 2'][i] = l
    wc_games.at[i, 'team1'] = team1
    wc_games.at[i, 'team2'] = team2

st.table(wc_games)

st.markdown('---')
st.markdown("<p style='text-align: right;'>I'm Bruno Motta</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: right;'>This was CS50!</h1>", unsafe_allow_html=True)
