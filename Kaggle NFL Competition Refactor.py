import time
import_timeS = time.perf_counter()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
import keras
from sklearn.model_selection import train_test_split, cross_val_score, LearningCurveDisplay, KFold
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, log_loss, root_mean_squared_error, make_scorer, roc_auc_score)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import class_weight
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from keras import layers
from keras.api.layers import Dense, BatchNormalization
from keras.api.models import Sequential
from keras.api.callbacks import EarlyStopping

import_timeE = time.perf_counter()
print(f'Import statements runtime: {import_timeE - import_timeS}')


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 300)


# ########################### TODO / Ideas ###########################
#
# Features for peak acceleration? Max impulse/change in accel? 
# Create complimentary features to capture details about the game that add to energy cost/fatigue not currently accounted for by metabolic power or other features
# Perhaps start using an average of speed over 3 frames or something similar, or find a way to reduce acceleration/decceleration differences during strides/on step
# Add metrics like time to peak velocity, accel, etc. and investigate their relationship to metabolic/mechanical power already expended during the game



#### Resources ####
# https://journals.lww.com/acsm-msse/fulltext/2010/01000/energy_cost_and_metabolic_power_in_elite_soccer__a.22.aspx using foundations from https://journals.biologists.com/jeb/article/208/14/2809/15645/Sprint-running-a-new-energetic-approach
# https://support.catapultsports.com/hc/en-us/articles/360001343976-What-is-Metabolic-Power 
# 
#
### Goals/targets:
# Fatigue: Create a feature to estimate players fatigue/energy expenditures over the course of the game
# Attempt to use a players expenditures up to that point in the game (or pre-snap) in conjunction with other features already available to predict how much energy/metabolic power they will expend on a given play
#
#### Completed engineered features
# Movement Energy Cost (MEC): The amount of energy expended by an athlete due to running, given by MECr = (155.4(ES)^5 - 30.4(ES)^4 - 43.3ES^3 + 46.3(ES)^2 + 19.5(ES) + 3.6) * EM * KT. Units J * kg^-1 * m^-1 or (Work / (Weight * distance))
#       and the amount of energy expended by an athlete due to walking given by MECw = (280.5(ES)^5 − 58.1(ES)^4 − 76.8(ES)^3 + 51.9(ES)^2 + 19.6(ES) + 2.5) * EM * KT, where EM is equivalent mass and KT is the coefficient of energy expended based on terrain
# 
# Players MetPower Expended : The amount of metabolic power an athlete has expended up to that point in the game
# Time Since Play: The time since the player was last on field/involved in a play
#
#
### Important Concepts
# When running/sprinting, the human body leans forward. The angle between the terrain and body can be called X.    
# Equivalent Slope (ES): The energy cost associated with running at constant speed up a slope is equivalent to the energy cost of accelerated running on flat terrain, assuming X is the same for both, given ES = tan(90 - x). 
# Equivalent Mass (EM): The average force exerted by the muscles when sprinting is greater than body weight due to the bodys acceleration, given by g'/g where g' = vector sum of acceleration and gravity
# The angle an athlete will learn forward due to acceleration is given by x = arctan(g/a) where g = gravity and a = acceleration. This means the complement angle (90 - x) is the angle we would rotate terrain upwards by to bring the accelerating runners body to identical expenditure to that of the constant speed slope runner 
# We can use acceleration to calculate our ES and EM, then use ES then to approximate the energy cost of sprinting (based on acceleration) using literature on the energy cost of incline constant speed running
# The metabolic power of an athlete in a given moment can be found using the energy cost associated with their movement and velocity (P = EC * v)


data_loadS = time.perf_counter()

project_dir = 'C:/Datasets/Kaggle Competition Data/NFL Big Data Bowl'
games = pd.read_csv(f'{project_dir}/games.csv')
plays = pd.read_csv(f'{project_dir}/plays.csv')
players = pd.read_csv(f'{project_dir}/players.csv')
player_play = pd.read_csv(f'{project_dir}/player_play.csv')
tracking_W1 = pd.read_csv(f'{project_dir}/tracking_week_1.csv')
tracking_W2 = pd.read_csv(f'{project_dir}/tracking_week_2.csv')
tracking_W3 = pd.read_csv(f'{project_dir}/tracking_week_3.csv')
tracking_W4 = pd.read_csv(f'{project_dir}/tracking_week_4.csv')
# tracking_W5 = pd.read_csv(f'{project_dir}/tracking_week_5.csv')
# tracking_W6 = pd.read_csv(f'{project_dir}/tracking_week_6.csv')
# tracking_W7 = pd.read_csv(f'{project_dir}/tracking_week_7.csv')
# tracking_W8 = pd.read_csv(f'{project_dir}/tracking_week_8.csv')
# tracking_W9 = pd.read_csv(f'{project_dir}/tracking_week_9.csv')
data_loadE = time.perf_counter()
print(f'Data loading runtime: {data_loadE - data_loadS}')


data_filterS = time.perf_counter()
# Filter for all week 1 games for testing, extract gameID and plays (uncomment first line for only week 1 DEN games, uncomment second line for all W1 games)
#den_games_wk1 = games[((games['homeTeamAbbr'] == 'DEN') | (games['visitorTeamAbbr'] == 'DEN')) & (games['week'] == 1)]
den_games_wk1 = games[(games['week'] == 1)]
den_gameIDs = den_games_wk1['gameId'].tolist()
plays = plays[plays['gameId'].isin(den_gameIDs)] 
player_play = player_play[player_play['gameId'].isin(den_gameIDs)]
tracking_W1 = tracking_W1[tracking_W1['gameId']. isin(den_gameIDs)]
games = games[games['gameId'].isin(den_gameIDs)]

data_filterE = time.perf_counter()
print(f'Data filtering runtime: {data_filterE - data_filterS}')

# Add positions to tracking and player_play and drop rows that represent the football
tracking_W1 = pd.merge(tracking_W1, players[['position', 'nflId']], on = 'nflId', how = 'left')
tracking_W1['position'] = tracking_W1['position'].fillna('football')
tracking_W1 = tracking_W1[~tracking_W1.position.str.contains('football')]
player_play = pd.merge(player_play, players[['position', 'nflId']], on = 'nflId')
mech_calc_timeS = time.perf_counter()

# Power = work / change in time, work = force * displacement
# Calc power and work, note units are in terms of meters, seconds, kg (unlike other power and work calcs) 
tracking_W1['prevX'] = tracking_W1.groupby(['playId', 'nflId', 'gameId'])['x'].shift(1)
tracking_W1['prevY'] = tracking_W1.groupby(['playId', 'nflId', 'gameId'])['y'].shift(1)
tracking_W1['displacement'] = (((tracking_W1['x'] - tracking_W1['prevX']) ** 2 + (tracking_W1['y'] - tracking_W1['prevY']) ** 2) ** 0.5) / 1.0936132983
playersWeight = players.drop(columns = ['height',"birthDate","collegeName","position","displayName"])
playersWeight['weightKg'] = playersWeight['weight'] * 0.45359237
tracking_W1 = tracking_W1.merge(playersWeight, on = 'nflId', how = 'left', validate = 'many_to_one')
tracking_W1['force'] = (tracking_W1['a'] / 1.0936132983) * tracking_W1['weightKg']
tracking_W1['rWork'] = tracking_W1['force'] * tracking_W1['displacement']
tracking_W1['rPower'] = tracking_W1['rWork'] / (1/10)
tracking_W1['calcS'] = np.abs(((tracking_W1['x'] - tracking_W1['prevX'])**2  + (tracking_W1['y'] - tracking_W1['prevY']) **2)**0.5 / 0.1)
tracking_W1['calcS'] = tracking_W1['calcS'].fillna(tracking_W1['s'])
tracking_W1['prevCalcS'] = tracking_W1.groupby(['playId', 'nflId', 'gameId'])['calcS'].shift(1)
tracking_W1['calcA'] = np.abs((tracking_W1['calcS'] - tracking_W1['prevCalcS']) / 0.1)
tracking_W1['calcA'] = tracking_W1['calcA'].fillna(tracking_W1['a'])
tracking_W1 = tracking_W1[~np.abs(tracking_W1['calcA']).ge(12)].reset_index()
tracking_W1 = tracking_W1[~np.abs(tracking_W1['calcA'].diff().ge(8))].reset_index()
mech_calc_timeE = time.perf_counter()
print(f'Displacement and mechanical power/work calculation runtime: {mech_calc_timeE - mech_calc_timeS}')


# Calculate the equivalent slope (ES) for athletes acceleration that results in equivalent energy expenditure at constant speed on said slope
tracking_W1['ES'] = np.tan((math.pi/2 - np.arctan2(9.81, (tracking_W1['calcA'] / 1.0936132983))))

# Calculate the athletes equivalent mass ratio (EM) due to the force of their acceleration 
tracking_W1['EM'] = ((((tracking_W1['calcA'] / 1.0936132983)** 2) / (9.81 ** 2)) + 1) ** 0.5

# Calculate the energy the athlete expends due to movement, based on whether they are likely running or walking. The equations for running and walking energy cost are below, MECr and MECw respectively
time_vMEC_classStart = time.perf_counter()
ES = tracking_W1['ES']
EM = tracking_W1['EM']
s = tracking_W1['s']
walkingMask = (s <= 2.43)
runningMask = (s > 2.43)
MEC_Walk_eq = ((280.5 * ES ** 5) - (58.1 * ES ** 4) - (76.8 * ES ** 3) + (51.9 * ES ** 2) + (19.9 * ES) + 2.5) * EM * 1.29
MEC_Run_eq = ((155.4* ES ** 5) - (30.4 * ES ** 4) - (43.3 * ES ** 3) + (46.3 * ES ** 2) + (19.5 * ES) + 3.6) * EM * 1.29
tracking_W1['MEC'] = 0.0
tracking_W1.loc[walkingMask, 'MEC'] = MEC_Walk_eq[walkingMask]
tracking_W1.loc[runningMask, 'MEC'] = MEC_Run_eq[runningMask]
time_vMEC_classEnd = time.perf_counter()
print(f'Vectorized MEC Calc runtime: {time_vMEC_classEnd - time_vMEC_classStart}')

# Calculte metabolic power used at each frame, metpower and MEC for each player up to that point in the game, then add those values to the player_play df
tracking_W1['MetPower'] = ((tracking_W1['calcS'] / 1.0936132983) * tracking_W1['MEC'])



# Convert 'gameDate' to datetime format 
tracking_W1['time'] = pd.to_datetime(tracking_W1['time'], format = 'mixed')
time_spg_calcStart = time.perf_counter()


## Calculate pre/post snap MEC and metabolic power and averages for pre-snap

#Presnap
tracking_W1['playerPreSnapMEC'] = tracking_W1[tracking_W1['frameType'] == 'BEFORE_SNAP'].groupby(['gameId', 'nflId', 'playId']).MEC.cumsum()
tracking_W1['playerPreSnapMetPower'] = tracking_W1[tracking_W1['frameType'] == 'BEFORE_SNAP'].groupby(['gameId', 'nflId', 'playId']).MetPower.cumsum()
intDf = pd.DataFrame(tracking_W1.loc[tracking_W1.groupby(['nflId', 'gameId', 'playId']).playerPreSnapMEC.idxmax()])
player_play = player_play.merge(intDf[['gameId', 'playId', 'nflId', 'playerPreSnapMEC', 'playerPreSnapMetPower']], how = 'left', on = ['gameId', 'playId', 'nflId'])

#Presnap Avgs
intDf = pd.DataFrame(player_play.groupby('nflId').playerPreSnapMEC.mean())
intDf.rename(columns = {'playerPreSnapMEC' : 'playerAvgPreSnapMEC'}, inplace = True)
intDf.reset_index(inplace = True)
player_play = player_play.merge(intDf, how = 'left', on = 'nflId')
player_play['PreSnapMECDiffFromAvg'] = player_play['playerPreSnapMEC'] - player_play['playerAvgPreSnapMEC']
intDf = pd.DataFrame(player_play.groupby('nflId').playerPreSnapMetPower.mean())
intDf.rename(columns = {'playerPreSnapMetPower' : 'playerAvgPreSnapMetPower'}, inplace = True)
intDf.reset_index(inplace = True)
player_play = player_play.merge(intDf, how = 'left', on = 'nflId')
player_play['PreSnapMetPowerDiffFromAvg'] = player_play['playerPreSnapMetPower'] - player_play['playerAvgPreSnapMetPower']

#Post Snap 
tracking_W1['playerPostSnapMEC'] = tracking_W1[tracking_W1['frameType'] == 'AFTER_SNAP'].groupby(['gameId', 'nflId', 'playId']).MEC.cumsum()
tracking_W1['playerPostSnapMetPower'] = tracking_W1[tracking_W1['frameType'] == 'AFTER_SNAP'].groupby(['gameId', 'nflId', 'playId']).MetPower.cumsum()
intDf = pd.DataFrame(tracking_W1.loc[tracking_W1.groupby(['nflId', 'gameId', 'playId']).playerPostSnapMEC.idxmax()])
player_play = player_play.merge(intDf[['nflId', 'playId', 'gameId', 'playerPostSnapMEC', 'playerPostSnapMetPower']], how = 'left', on = ['nflId', 'gameId', 'playId'], validate = '1:1')
intDf = pd.DataFrame(tracking_W1.loc[tracking_W1.groupby(['nflId', 'gameId', 'playId']).MetPower.idxmax()])
intDf = intDf.rename(columns = {'MetPower' : 'peakMetPower'})
player_play = player_play.merge(intDf[['nflId', 'playId', 'gameId', 'peakMetPower']], how = 'left', on = ['nflId', 'gameId', 'playId'])

time_spg_calcEnd = time.perf_counter()
print(f'Pre/post snap MEC and Met Power Averages: {time_spg_calcEnd - time_spg_calcStart}')


time_time_convStart = time.perf_counter()

# Merge Positions for each player and possession/defending team into tracking data, also game date/start time into plays and sort plays chronologically 
plays = pd.merge(plays, games[['gameId', 'gameDate', 'gameTimeEastern']], on = 'gameId', how = 'left')
plays['gameDate'] = pd.to_datetime(plays['gameDate'], format ='%m/%d/%Y')
plays['gameTimeEastern'] = pd.to_datetime(plays['gameTimeEastern'], format = '%H:%M:%S')
tracking_W1 = tracking_W1.sort_values(by = 'time', ascending = True)
plays = plays.sort_values(by= ['gameDate', 'gameTimeEastern', 'quarter', 'gameClock'], ascending = [True, True, True, False])
tracking_W1 = tracking_W1.merge(games[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']], on = 'gameId', how = 'left')
plays = plays.merge(games[['homeTeamAbbr', 'gameId']], on = 'gameId', how = 'left')
#Merge players position and offense formation into player play
player_play = player_play.merge(plays[['gameId', 'playId', 'offenseFormation', 'receiverAlignment']], on = ['gameId', 'playId'])

# Calculate total metabolic power and metabolic energy cost for each player on each play. Tracking contains cumulative values for each row, representing their expenditures up to that point in the play, player_play represents total values for that play
tracking_W1['playerPlayMetPower'] = tracking_W1.groupby(['gameId', 'nflId', 'playId']).MetPower.cumsum()
tracking_W1['playerPlayMEC'] = tracking_W1.groupby(['gameId', 'nflId', 'playId']).MEC.cumsum()
t = pd.DataFrame(tracking_W1.loc[tracking_W1.groupby(['nflId', 'gameId', 'playId']).playerPlayMEC.idxmax()])
player_play = player_play.merge(t[['nflId', 'playId', 'gameId', 'playerPlayMEC', 'playerPlayMetPower']], on = ['nflId', 'playId', 'gameId'], validate = '1:1')


#Calculate the total MEC and MetPower expended by each player up to that point in the game.
player_play['playerTotalMetPower'] = player_play.groupby(['nflId', 'gameId']).playerPlayMetPower.cumsum()
player_play['playerTotalMEC'] = player_play.groupby(['nflId', 'gameId']).playerPlayMEC.cumsum()
# tracking_W1['playerTotalMetPower'] = tracking_W1.groupby(['gameId', 'nflId']).MetPower.cumsum()
# tracking_W1['playerTotalMEC'] = tracking_W1.groupby(['gameId', 'nflId']).MEC.cumsum()

time_time_convEnd = time.perf_counter()
print(f'Time var conversions and merges, total cumulative MEC/Met Power calc runtime: {time_time_convEnd - time_time_convStart}')

#Identify team sides 
tracking_W1['side'] = np.where(tracking_W1['club'] == tracking_W1['homeTeamAbbr'], 'home', 'away')
plays['playPossSide'] = np.where(plays['possessionTeam'] == plays['homeTeamAbbr'], 'home', 'away')

time_time_calcStart = time.perf_counter()

# Calculate the time that has elapsed since each player had last played. Values of 0 indicate this is their first play of the game/season (depending on whats loaded/filtered).
player_play_start_times = tracking_W1.groupby(['nflId', 'playId', 'gameId'])['time'].min().reset_index()
player_play_end_times = tracking_W1.groupby(['nflId', 'playId', 'gameId'])['time'].max().reset_index()
player_play_times = pd.merge(player_play_start_times, player_play_end_times, on = ['nflId', 'playId', 'gameId'], suffixes = ('_start', '_end'))
player_play_times['prev_end'] = player_play_times.groupby(['nflId', 'gameId'])['time_end'].shift(1)
player_play_times['Time Since Play'] = player_play_times['time_start'] - player_play_times['prev_end']
tracking_W1 = pd.merge(tracking_W1, player_play_times, on = ['gameId', 'nflId', 'playId'], how = 'left')
tracking_W1['Time Since Play'] = tracking_W1['Time Since Play'].fillna(pd.Timedelta(0))
player_play = player_play.merge(tracking_W1[['gameId', 'nflId', 'playId', 'Time Since Play']].drop_duplicates(subset = ['nflId', 'gameId', 'playId','Time Since Play']), on = ['gameId', 'nflId', 'playId'], validate = '1:m')
player_play['Time Since Play'] = player_play['Time Since Play'].dt.total_seconds() / 60

time_time_calcEnd = time.perf_counter()
print(f'Time since play calculation runtime: {time_time_calcEnd - time_time_calcStart}')


# Count how many plays this player has been involved in up to this point in the game
player_play['playCount'] = player_play.groupby(['nflId', 'gameId']).cumcount()


# Calculate change in directions for each player. Note: TotalCODs and CoD don't represent distinct changes in direction, just points in the data where a change greater than x number of degrees happened across the frameShift, 
# which means for 1 actual physical CoD there could be multiple rows marked as a CoD
time_vCod_calcStart = time.perf_counter()
frameShift = 3
tracking_W1['codIntervalDir'] = tracking_W1.groupby(['gameId', 'playId', 'nflId'])['dir'].shift(frameShift)
directionChange = tracking_W1['dir'] - tracking_W1['codIntervalDir']
codMask = ((directionChange > 90) & (directionChange < 270) | (directionChange > -270) & (directionChange < -90))
tracking_W1['CoD'] = codMask.astype(int)
player_play['playerTotalCODs'] = tracking_W1.groupby(['gameId', 'nflId'])['CoD'].transform('cumcount')
time_vCod_calcEnd = time.perf_counter()
print(f'Vectorized CoD calc runtime: {time_vCod_calcEnd - time_vCod_calcStart}')


# Merge Expected points added into player_play and count which play number of the game this is
player_play = player_play.merge(plays[['expectedPointsAdded', 'gameId', 'playId']], on = ['gameId', 'playId'])
plays['playNo'] = plays.groupby('gameId').cumcount()
player_play = player_play.merge(plays[['gameId', 'playId', 'playNo']], on = ['gameId', 'playId'])


# Calculate each players "active time" or time spent on the field/involved in plays
intDf = tracking_W1.loc[tracking_W1.groupby(['nflId', 'gameId', 'playId']).frameId.idxmax()]
intDf['playTime'] = intDf['frameId'].transform('cumsum') * 0.1 / 60
player_play = player_play.merge(intDf[['gameId', 'nflId', 'playId', 'playTime']], on = ['gameId', 'nflId', 'playId'])




# Count number of players per position involved in the game per club
time_pos_calcStart = time.perf_counter()
player_playFilt = player_play[['gameId', 'playId', 'nflId', 'teamAbbr']]
player_playFilt = player_playFilt.merge(players[['nflId', 'position']], on = 'nflId', how = 'left')
player_side = tracking_W1[['gameId', 'nflId', 'side']].drop_duplicates(subset = 'nflId')
player_playFilt = player_playFilt.merge(player_side, on = ['nflId', 'gameId'], how = 'left') 
clubPositionCountsTot = player_playFilt.groupby(['gameId', 'side', 'position'])['nflId'].nunique()
clubPositionCountsTot = clubPositionCountsTot.unstack().reset_index()
clubPositionCountsTot = clubPositionCountsTot.melt(id_vars = ['side', 'gameId'], value_name = 'pos_total')
time_pos_calcEnd = time.perf_counter()
print(f'Team position total player calc runtime: {time_pos_calcEnd - time_pos_calcStart}')



# Count number of players per position involved in each play of the game by side
time_posP_calcStart = time.perf_counter()
clubPositionCountsPlay = player_playFilt.groupby(['gameId', 'side', 'playId', 'position'])['nflId'].nunique()
clubPositionCountsPlay = clubPositionCountsPlay.unstack().reset_index()
clubPositionCountsPlay = clubPositionCountsPlay.melt(id_vars= ['side', 'gameId', 'playId'], value_name = "pos_play")
time_posP_calcEnd = time.perf_counter()
print(f'Team position per play player calc runtime: {time_posP_calcEnd - time_posP_calcStart}')

# Create new dataframe with cumulative power expended for each sides players in each position, with one feature (clubPosPowerPerPlay) representing average expenditure by that position type for the team on that play, and another (clubPosPowerTot) representing total expenditure by that position type on that play for each team
time_PPP_calcStart = time.perf_counter()
power_per_play =  tracking_W1.groupby(['gameId', 'playId', 'position', 'side'])['MetPower'].sum().reset_index() 
power_per_play['CumuMetPower'] = power_per_play.groupby(['gameId', 'side', 'position'])['MetPower'].cumsum() 
power_per_play = power_per_play.merge(clubPositionCountsPlay, on = ['gameId', 'playId', 'side', 'position'], how = 'left')
power_per_play = power_per_play.merge(clubPositionCountsTot, on = ['gameId', 'position', 'side'], how = 'left')


power_per_play['metPower_per_posTotal'] = power_per_play['CumuMetPower'] / power_per_play['pos_total']
power_per_play['metPower_per_posPlay'] = power_per_play['CumuMetPower'] / power_per_play['pos_play']
power_per_play['positionTot'] = power_per_play['side'] + '_' + power_per_play['position'] + '_' + 'avgMetPowerPosTotal' 
power_per_play['positionPlay'] = power_per_play['side'] + '_' + power_per_play['position'] + '_' + 'avgMetPowerPosInPlay' 
clubPosPowerPerPlay = power_per_play['positionPlay'].unique()
clubPosPowerTot = power_per_play['positionTot'].unique()

powerDfPlay = power_per_play.pivot_table(index = ['gameId', 'playId'], columns = 'positionPlay', values = 'metPower_per_posPlay', fill_value = 0).reset_index()
powerDfTot = power_per_play.pivot_table(index = ['gameId', 'playId'], columns = 'positionTot', values = 'metPower_per_posTotal', fill_value = 0).reset_index()

# Merge power info into plays
plays = plays.merge(powerDfPlay, on = ['gameId', 'playId'], how = 'left')
plays = plays.merge(powerDfTot, on = ['gameId', 'playId'], how = 'left')
time_PPP_calcEnd = time.perf_counter()
print(f'Team position per play player calc runtime: {time_PPP_calcEnd - time_PPP_calcStart}')

def met_power_graph(nflId, playId, gameId):
    yer = tracking_W1.groupby(['nflId', 'playId', 'gameId']).get_group((nflId, playId, gameId))
    plt.figure()
    plt.title(f'Player: {nflId} on play {playId} of game: {gameId}')
    #plt.plot(yer['frameId'], yer['MetPower'])
    #plt.plot(yer['frameId'], yer['calcA'].diff())
    plt.plot(yer['frameId'], yer['calcS'])
    plt.plot(yer['frameId'], yer['calcA'])
    plt.tight_layout()
    plt.show()

#met_power_graph(52430.0, 1281, 2022091112)
#met_power_graph(46170.0, 1744, 2022091101)


# Encode categorical variables (and booleans that aren't true booleans, ie True/False/Nan)
heldCatVars = ['passResult', 'unblockedPressure']
wrongGranularityCats = ['shiftSinceLineset','routeRan','causedPressure']
categoricalsToEncode = ['playPossSide']
den_data = pd.get_dummies(plays, columns=categoricalsToEncode)
# Create array of variables for features in predictive models, including encoded categoricals
wrongGranularity = ['getOffTimeAsPassRusher','timeToPressureAsPassRusher','rushingYards', 'yardageGainedAfterTheCatch','fumbles','fumbleLost','quarterbackHit','hadPassReception','hadInterception']
heldVars = ['Metabolic Power Created', 'Players MetPower Expended','Time Since Play'] + ['playClockAtSnap', 'passLength', 'expectedPointsAdded', 'isDropback']
varsForFeatures = [] + [
    col for col in plays.columns
    if any(col.startswith(f'{feat}_') for feat in categoricalsToEncode)
] + [
    teamPosP for teamPosP in clubPosPowerTot
] + [
    teamPosP for teamPosP in clubPosPowerPerPlay
]
# Array of features for correlation (includes power stats)
varsForCorr = varsForFeatures + ['Metabolic Power Created', 'Players MetPower Expended']


# Heatmap for correlations 
def correlation_table(df, cols):
    corre = df[cols].corr()
    plt.figure()
    plt.title("Correlations of selected Columns")
    sns.heatmap(corre, annot = True)
    plt.tight_layout()
    plt.show()
correlationColumns = ['playerPreSnapMEC', 'PreSnapMECDiffFromAvg', 'hadRushAttempt', 'rushingYards', 'hadDropback', 'passingYards', 'sackYardsAsOffense', 'hadPassReception', 'receivingYards', 'wasTargettedReceiver', 'yardageGainedAfterTheCatch', 'fumbles', 'fumbleLost', 'fumbleOutOfBounds', 'assistedTackle', 'forcedFumbleAsDefense', 'quarterbackHit', 'passDefensed', 'sackYardsAsDefense', 'soloTackle', 'tackleAssist', 'tackleForALoss', 'hadInterception', 'fumbleRecoveries', 'causedPressure']
#correlation_table(player_play, correlationColumns)


# Convert time since play into seconds instead of datetimedelta, fill missing values for Power stats (players who haven't/dont play in the game). 
numericFeats = ['Metabolic Power Created', 'Players MetPower Expended', 'playClockAtSnap', 'passLength', 'expectedPointsAdded', 'timeToPressureAsPassRusher', 'getOffTimeAsPassRusher', 'shiftSinceLineset']


#Split the player_play df by positions/teams for testing inference using various groups

player_play['wasRunningRoute'] = player_play['wasRunningRoute'].replace(to_replace = 'NA', value = 0)
player_play_line = player_play[player_play['position'].isin(['T', 'G', 'OLB', 'DE', 'ILB', 'DT', 'C', 'NT', 'MLB', 'LB', 'DB'])]
player_play_skill = player_play[player_play['position'].isin(['CB', 'WR', 'TE', 'FS', 'RB', 'SS', 'FB'])]
player_play_off = player_play[player_play['position'].isin(['WR', 'T', 'TE', 'C', 'RB', 'QB', 'FB'])]
player_play_def = player_play[player_play['position'].isin(['CB', 'DT', 'FS', 'SS', 'NT', 'MLB', 'LB', 'DB'])]


# Encode categorical variables
player_play = pd.get_dummies(player_play, columns = ['offenseFormation', 'position', 'receiverAlignment'], prefix = ['formation', 'position', 'alignment'])
encodedCols = [col for col in player_play.columns if 'formation' in col or 'position' in col or 'alignment' in col]


xyz = ['playerPreSnapMEC','playerTotalMEC','PreSnapMECDiffFromAvg','playerAvgPreSnapMEC', 'playNo', 'PreSnapMetPowerDiffFromAvg']
X = player_play[['Time Since Play', 'playCount', 'playerTotalCODs', 'expectedPointsAdded', 'playerPreSnapMetPower',  'playerTotalMetPower', 'playTime', 'playerAvgPreSnapMetPower'] + encodedCols]
y = player_play['playerPostSnapMetPower']

# Normalize the features and split, then train keras sequential model and evaluate
print(player_play['playerPostSnapMetPower'].describe())
X = (X - X.mean()) / X.std()
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.25, random_state = 42)


kmodel  = keras.Sequential([
    layers.Input((xTest.shape[1],)),
    layers.Dense(42, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dense(30, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate = 0.3),
    layers.Dense(512, activation = 'relu'),
    layers.Dense(1)
])
earlyStop = EarlyStopping(
    min_delta = .3,
    patience = 15,
    restore_best_weights = True
)
kmodel.compile(optimizer = 'adam', loss = 'mae')

kmodel.fit(xTrain, yTrain, epochs = 200, validation_data = (xTest, yTest), batch_size = 128, callbacks = [earlyStop])
testLoss = kmodel.evaluate(xTest, yTest)
print(f'Loss: {testLoss}')




