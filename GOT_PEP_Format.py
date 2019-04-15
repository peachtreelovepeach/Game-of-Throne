# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 08:52:20 2019

@author: Ying Li
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:00:12 2019

@author: YingLi
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.patches as mpatches
sns.set_style("white")
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import copy
import re
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

'''
I fetch the data from ice and fire API (https://anapioficeandfire.com/) to 
correct wrong or missing data of original dataset, the code is below and JSON files 
are appended in the file: Raw data from api.
'''
'''
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 22:30:35 2019

@author: yvain
"""

import csv
import json
from pprint import pprint

houses = {}
characters = {}
books = {}
alias_to_characters = {}
names = []

final_chars = []

def _get_character_info(name, character):
    return {
        'name': name,
        'male': 0 if character['IsFemale'] else 1,
        'title': ','.join(character['Titles']),
        'culture': character['Culture'],
        'house': ','.join(character['Allegiances']),
    }

with open("./names") as names_file:
    for name in names_file:
        names.append(name.rstrip())

with open("./houses.json") as houses_file:
    houses_data = json.load(houses_file)
    for house in houses_data:
        houses[house['Id']] = house['Name']

with open("./characters.json") as char_file:
    char_data = json.load(char_file)
    for char in char_data:
        if char['Name'] == "":
            continue
        allegiances = []
        for a in char['Allegiances']:
            allegiances.append(houses[a])
        new_char = char
        new_char['Allegiances'] = allegiances
        characters[new_char['Name']] = new_char
        if len(new_char['Aliases']) > 0:
            for alias in new_char['Aliases']:
                alias_to_characters[alias] = new_char['Name']

in_characters = []
for name in names:
    no_alias = name.split('(', 1)[0].rstrip()
    no_last_name = no_alias.rsplit(' ', 1)[0]
    if no_alias in characters:
        c = characters[no_alias]
        final_chars.append(_get_character_info(name, c))
    elif no_last_name in characters:
        c = characters[no_last_name]
        final_chars.append(_get_character_info(name, c))
    if name in alias_to_characters:
        c = characters[alias_to_characters[name]]
        final_chars.append(_get_character_info(name, c))

with open('./c.csv', 'w', newline='') as df:
    writer = csv.writer(df)
    header = final_chars[0].keys()
    writer.writerow(header)
    for fc in final_chars:
        writer.writerow(fc.values())


###Merge

import pandas as pd 

a = pd.read_csv('./c.csv')
b = pd.read_csv('./GOT.csv')
merged = a.merge(b, on='name')
merged.to_csv('New_GOT_data.csv', index=False)

merged.info()
'''


##############################################################################
##############################################################################
######                     Data Import                                ########
##############################################################################
##############################################################################





#Import Clean Dataset

file = 'Clean_GOT.xlsx'
old_data = pd.read_excel(file)
data = old_data.copy()



'''
Assumed Continuous/Interval Variables - 
age
S.No
dateOfBirth
numDeadRelations 
popularity 

Assumed Categorical -

title
culture   
mother                        
father                        
heir                          
house                         
spouse 

Binary Classifiers-
male 
book1_A_Game_Of_Thrones       
book2_A_Clash_Of_Kings       
book3_A_Storm_Of_Swords      
book4_A_Feast_For_Crows       
book5_A_Dance_with_Dragons  
isAliveMother                 
isAliveFather                 
isAliveHeir                   
isAliveSpouse                 
isMarried                     
isNoble    
isAlive 
'''


##############################################################################
##############################################################################
######                     Data Exploration                           ########
##############################################################################
##############################################################################

data.info()

'''
There are 1946 observations, in which father, mother, heir, isAliveMother, 
isAliveFather, isAliveHeir, isALiveSpouse, age missing over 90% data.
'''

#############
# Name
#############
#Check if 'name' is duplicated 
len(data['name'].unique())#1946 unique observations, no duplicates


#############
# Male
#############
#Plot gender
data['male'].plot.hist()
plt.title('Male vs Female Ratio', fontsize = 25)
sns.despine()
plt.xlabel('Gender')
plt.xticks(np.array([0,1]),['Female','Male'], rotation = 90, fontsize = 13)
plt.figure(figsize = (10,10))


#Gender Vs Death
gd = data.groupby(["male", "isAlive"]).count()["S.No"].unstack().copy(deep = True)
p = gd.div(gd.sum(axis = 1), axis = 0).plot.barh(stacked = True, 
                                                 rot = 0, 
                                                 width = .4)
_ = p.set(yticklabels = ["No", "Yes"], 
          xticklabels = "", 
          xlabel = "Proportion of Dead vs. Alive", 
          ylabel = "Male"), p.legend(["Dead", "Alive"])
plt.show()

'''
About 20% is Female, 80% is Male
Male dead more, but famle data is not representive
'''


#############
# Popularity
#############
#Plot popularity
data['popularity'].sort_values().plot.hist(bins = 10, color = 'b')
plt.xlabel('Popularity')
plt.title('Popularity',fontsize = 25)
plt.show()

#characters are popular
data[data['popularity']>=0.9].name

'''
Most popular Characters
5         Tommen Baratheon
50       Joffrey Baratheon
54         Margaery Tyrell
101             Robb Stark
172      Stannis Baratheon
273          Samwell Tarly
280      Aegon I Targaryen
296     Aerys II Targaryen
1466            Arya Stark
1474            Bran Stark
1481      Cersei Lannister
1651       Barristan Selmy
1677    Daenerys Targaryen
1683        Davos Seaworth
1688          Eddard Stark
1741       Jaime Lannister
1749              Jon Snow
1781          Roose Bolton
1784       Renly Baratheon
1785         Theon Greyjoy
1792           Sansa Stark
1798       Tywin Lannister
1828         Petyr Baelish
1840      Tyrion Lannister
'''

#Character are poplular and still alive
#For later recommendation use
pop_cha=data[(data['popularity']>0.8) & (data['isAlive']==1)]



#############
# Age
#############
data['age'].describe()#mean equal to negative, something wrong
aa=data.loc[data['age'] < 0 ]#two observations 110, 1350, need correction 

#Rhaego = stillborn
data.loc[data["name"]=="Rhaego","age"]=0
data.loc[data["name"]=="Rhaego","dateOfBirth"]=298


#Doreah
data.loc[data["name"]=="Doreah","age"]=20
data.loc[data["name"]=="Doreah","dateOfBirth"]=279

#visualize age
sns.distplot(data['age'].dropna())
plt.show()

'''
Most characters are inbetween 15-25, or 90-100
'''

#plot age at the time of death
plt.title('Age at time of death',fontsize = 20)
plt.xlabel('Age', fontsize = 15, rotation = 45)
sns.despine()
data[data['isAlive'] == 0][abs(data[data['isAlive']==0]['age'])<150]['age'].plot.hist()

'''
Most people died 10-30, or 90-100
'''

#############
# Titles
#############
#Plot most frequente titles
plt.figure(figsize = (6,4))
plt.xlabel('Title')
plt.ylabel('Frequency')
data.title.value_counts().sort_values(ascending = False)[:10].plot.bar()

'''
Most frequent titles are ser(around 390 counts), rest are relativly small groups,
need to regroup later
'''

#############
# Culture
#############
#Plot most frequente culture
plt.figure(figsize = (6,4))
plt.xlabel('Culture')
plt.ylabel('Frequency')
data.culture.value_counts().sort_values(ascending = False)[:10].plot.bar()


#culture Vs. popularity
unique = data.culture.unique()
p_rates = {}
dataPop = sum(data.popularity)

for sal in unique:
   if(type(sal)==str):
       p_rates[sal] = sum(data.loc[data['culture'] == sal,'popularity'])/(sum(data.culture == sal))

plot_rates = []       

for key, value in zip(p_rates.keys(), p_rates.values()):
    plot_rates.append([key,round(value,2)])

pops = sorted(plot_rates, key = lambda x: x[1], reverse = False)
pops = np.array(pops)
pops = pops[len(pops)-20:len(pops),:]
plt.figure(figsize = (10,5))
plt.xlabel('Culture')
plt.ylabel('Popularity')
plt.xticks(rotation = 90, fontsize = 12)
plt.title('Culture vs Average Popularity', fontsize = 15)
plt.bar(pops[:,0],
        pops[:,1], 
        color = sns.color_palette('Paired', n_colors=15), 
        hatch = '//', 
        linestyle = '--')
sns.despine()

'''
need regroup later
popular culture might not representative due to small samples

'''

#############
# House
#############
#Plot most frequente house
plt.figure(figsize = (6,4))
plt.xlabel('House')
plt.ylabel('Frequency')
data.house.value_counts().sort_values(ascending = False)[:10].plot.bar()
'''
Night's Watch, House Frey , House Stark, House Targaryen, House Lannister,
House Greyjoy, need to check on later
'''

#house Vs. popularity
unique = data.house.unique()
p_rates = {}
dataPop = sum(data.popularity)
for sal in unique:
    if(type(sal)==str):
        p_rates[sal] = sum(data.loc[data['house'] == sal,'popularity'])/(sum(data.house == sal))

plot_rates = []       
for key, value in zip(p_rates.keys(), p_rates.values()):
    plot_rates.append([key,round(value,2)])
pops = sorted(plot_rates, key = lambda x: x[1], reverse = False)
pops = np.array(pops)
pops = pops[len(pops)-20:len(pops),:]
plt.figure(figsize = (10,5))
plt.xlabel('House')
plt.ylabel('Popularity')
plt.xticks(rotation = 90, fontsize = 12)
plt.title('House vs Average Popularity', fontsize = 15)
plt.bar(pops[:,0],
        pops[:,1] , 
        color = sns.color_palette('Paired', n_colors=15), 
        hatch = '//', 
        linestyle = '--')
sns.despine()



#############
# DeathRelations
#############
#NumDeathRelations VS Death
dr = data.groupby(["numDeadRelations", "isAlive"]).count()["S.No"].unstack().copy(deep = True)
p = dr.div(dr.sum(axis = 1), axis = 0).plot.barh(stacked = True, 
                                                 rot = 0, 
                                                 width = .5)
_ = p.set_xlim([0, 1]), 
_ = p.set( xticklabels = "",
           xlabel = "Proportion of Dead vs. Alive", 
           ylabel = "Has Dead Relations"), 
_ = p.legend(["Dead", "Alive"])

'''
people with no dead relations being alive more
'''


#############
# Books
#############
#Books VS Death
data['book1-5']=(data['book1']+data['book2']+data['book3']+data['book4']+data['book5'])

bb = data.groupby(["book1-5", "isAlive"]).count()["S.No"].unstack().copy(deep = True)
p = bb.div(bb.sum(axis = 1), axis = 0).plot.barh(stacked = True, 
                                                 rot = 0,
                                                 figsize = (15, 8), 
                                                 width = .5)
_ = p.set(xticklabels = "", 
          xlim = [0, 1], 
          ylabel = "No. of Books", 
          xlabel = "Proportion of Dead vs. Alive"), 
_ = p.legend(["Dead", "Alive"], loc = "upper right", ncol = 2, borderpad = -.15)

'''
Characters never showed in books have higer death rate, cause it might be historical or fiction.

'''



#############
# Comapre relations
#############

p = sns.pairplot(data[(data.age >= 0)][["popularity", "numDeadRelations", "age", "isAlive"]], 
                      hue = "isAlive", 
                      vars = ["popularity", "numDeadRelations", "age"], 
                      kind = "reg",
                      height = 4.)
_ = p.axes[0][0].set_ylabel("Popularity"), 
_ = p.axes[1][0].set_ylabel("No. of Dead Relations"),
_ = p.axes[2][0].set_ylabel("Age")
_ = p.axes[2][0].set_xlabel("Popularity"), 
_ = p.axes[2][1].set_xlabel("No. of Dead Relations"),
_ = p.axes[2][2].set_xlabel("Age")

'''
pipulartiy vs death, popular or not popular cha is alive, younger popular cha 
is alive


'''


##############################################################################
##############################################################################
######                     Feature Engineering                        ########
##############################################################################
##############################################################################

##############
#Flag missing value
##############
miss_flag = ['father', 'mother', 'heir', 'isAliveMother', 
'isAliveFather', 'isAliveHeir', 'isAliveSpouse', 'age']

for c in miss_flag:
    if data[c].isnull().any():
        data['m_' + c] = data[c].isnull().astype(int)


#############
# Add new column: dateOfBirth+age
#############

#Current year is 305
data['dateOfDeath']=0

for i in range(0,1946):
    if data.loc[i,'dateOfBirth']+data.loc[i,'age']<305:
        data.loc[i,'dateOfDeath']=1
       
#############
# Popularity
#############
'''
Creat column 'pop_luck' for characters are very popular or not popular, these
character have lower risk of death
'''      
        
data['pop_luck']=0

for i in range (1,1946):
  if data.loc[i, 'popularity']> 0.8:
    data.loc[i,'pop_luck']=1
  elif data.loc[i, 'popularity']< 0.3:
    data.loc[i,'pop_luck']=1
  else: data.loc[i,'pop_luck']=0


#############
# NO.of Death Relation
#############
'''
Create ndr_luck for characters have less than 3 dead relations, those people 
have lower risk of death
'''
data['ndr_luck']=0

for i in range (1,1946):
  if data.loc[i, 'numDeadRelations']<=3:
    data.loc[i,'ndr_luck']=1

        

#############
# Culture
#############        

########Correct mistypo##########

#Check 'culture'
cul = data['culture'].unique()#similar value
len(cul)#63 unique cultrue

#flag missing culture
data['m_culture'] = data['culture'].isnull().astype(int)

data['culture']=data['culture'].fillna('cul_unknown')
#regroup some value
cult_lst = {
    'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],
    'Ghiscari': ['ghiscari', 'ghiscaricari'],
    'Asshai': ["asshai'i", 'asshai'],
    'Lysene': ['lysene', 'lyseni'],
    'Andal': ['andal', 'andals'],
    'Braavosi': ['braavosi', 'braavos'],
    'Dornish': ['dornishmen', 'dorne', 'dornish'],
    'Myrish': ['myr', 'myrish', 'myrmen'],
    'Westermen': ['westermen', 'westerman', 'westerlands'],
    'Stormlander': ['stormlands', 'stormlander'],
    'Norvoshi': ['norvos', 'norvoshi'],
    'Northmen': ['the north', 'northmen'],
    'Free Folk': ['wildling', 'first men', 'free folk'],
    'Qartheen': ['qartheen', 'qarth'],
    'Reach': ['the reach', 'reach', 'reachmen'],
    'Lhazareen':['Lhazarene']
}


def get_cult(value):
    value = value.lower()
    v = [k for (k, v) in cult_lst.items() if value in v]
    return v[0] if len(v) > 0 else value.title()


data.loc[:, 'culture'] = [get_cult(x) for x in data.culture]

# Samwell Tarly
data.loc[1871, 'culture'] = 'Andals'

# Ilyn Payne
data.loc[614, 'culture'] = 'Andals'

# Barristan Selmy
data.loc[1504, 'culture'] = 'Andals'

# Davos Seaworth
data.loc[1396, 'culture'] = 'Andals'

# Mycah
data.loc[731, 'culture'] = 'Northmen'

# Luwin
data.loc[1762, 'culture'] = 'Northmen'

# Shiera Seastar
data.loc[797, 'culture'] = 'Valyrian'

#Check for culture work
len(data['culture'].unique())#42                                 



##########Visualse Culture again########
#Culture count
plt.figure(figsize = (6,4))
plt.xlabel('Culture')
plt.ylabel('Frequency')
data.culture.value_counts().sort_values(ascending = False)[:10].plot.bar()

#Culture vs death
dr = data.groupby(["culture", "isAlive"]).count()["S.No"].unstack().copy(deep = True)
p = dr.div(dr.sum(axis = 1), axis = 0).plot.barh(stacked = True, rot = 0, width = .5)
_ = p.set_xlim([0, 1]), 
_ = p.set( xticklabels = "", 
           xlabel = "Proportion of Dead vs. Alive", 
           ylabel = "Has Dead Relations"), 
_ = p.legend(["Dead", "Alive"])


#culture Vs. popularity
unique = data.culture.unique()
p_rates = {}
dataPop = sum(data.popularity)
for sal in unique:
  if(type(sal)==str):
    p_rates[sal] = sum(data.loc[data['culture'] == sal,'popularity'])/(sum(data.culture == sal))

plot_rates = []       

for key, value in zip(p_rates.keys(), p_rates.values()):
    plot_rates.append([key,round(value,2)])

pops = sorted(plot_rates, key = lambda x: x[1], reverse = False)
pops = np.array(pops)
pops = pops[len(pops)-20:len(pops),:]
plt.figure(figsize = (10,5))
plt.xlabel('Culture')
plt.ylabel('Popularity')
plt.xticks(rotation = 90, fontsize = 12)
plt.title('Culture vs Average Popularity', fontsize = 15)
plt.bar(pops[:,0],
        pops[:,1], 
        color = sns.color_palette('Paired', n_colors=15), 
        hatch = '//', 
        linestyle = '--')
sns.despine()

'''
Seems no relations between popularity and death in cultures
'''

########Death of Each Culture#########

#find death/alive info for each culture
da_culture = data.groupby(['culture','isAlive']).count()["S.No"].unstack().copy(deep=True)
da_culture.columns = ['Dead','Alive']
da_culture = da_culture.fillna(0)
da_culture['%death']=da_culture['Dead']/(da_culture['Dead']+da_culture['Alive'])


#death rate over 60% is marked as death_culture, death rate under 40% marked as lucky_culture
lucky_culture = da_culture.loc[da_culture['%death']<=0.4]

data['lucky_culture']= 0
lc_lst = lucky_culture.index.tolist()

for h in lc_lst:
    for i in range(0,1946):
       if data.loc[i,'culture'] == h:
           data.loc[i,'lucky_culture'] = 1





#############
# House
#############   

#Set Targaryen in house lable
targaryen = r'.*[t|T]argaryen.*'
data.loc[data['name'].str.match(targaryen),'house']='Targaryen'


#Fit people named Stark in house of stark
stark = r'.*Stark.*'
data.loc[data['name'].str.match(stark),'house']='Stark'


#Fit people named Lannister in house of Lannister
Lan = r'.Lannister.*'
data.loc[data['name'].str.match(Lan),'house']='Lannister'


#Fit people named Greyjoy in house of Greyjoy
Grey = r'.*Greyjoy.*'
data.loc[data['name'].str.match(Grey),'house']='Greyjoy'


#Fit people named Tyrell in house of Tyrell
tyrell = r'.*Tyrell.*'
data.loc[data['name'].str.match(tyrell),'house']='Tyrell'

#flag missing house
data['m_house'] = data['house'].isnull().astype(int)



data['house'] = data['house'].fillna('H_Unknow')
data['house']=data['house'].apply(lambda x: x.replace('House of ',''))
data['house']=data['house'].apply(lambda x: x.replace('House ',''))
data['house']=data['house'].apply(lambda x: x.replace(' ',''))
data['house']=data['house'].apply(lambda x: x.replace("'",''))




#find death/alive info for each house
da_house = data.groupby(['house','isAlive']).count()["S.No"].unstack().copy(deep=True)
da_house.columns = ['Dead','Alive']
da_house = da_house.fillna(0)
da_house['%death']=da_house['Dead']/(da_house['Dead']+da_house['Alive'])

#Flag Targaryen and NightsWatch

for i in range(0,1946):
    if data.loc[i,'house']=='Targaryen':
        data.loc[i,'Targaryen_NW']=1
    elif data.loc[i,'house']=='NightsWatch':
        data.loc[i,'Targaryen_NW']=1
    else: data.loc[i,'Targaryen_NW']=0


#death rate over 80% is marked as death_house, death rate under 40% marked as lucky_house
lucky_house = da_house.loc[da_house['%death']<=0.4]


data['lucky_house']= 0
lh_lst = lucky_house.index.tolist()

for h in lh_lst:
    for i in range(0,1946):
       if data.loc[i,'house'] == h:
           data.loc[i,'lucky_house'] = 1




#Regroup house by region
data['house_by_region']=copy.deepcopy(data['house'])

#dictionary of region
house_region_lst = {'Crownlands':['BaratheonofKingsLanding','Citywatchofkingslanding',
                                  'Kingsguard','Queensguard',"Baratheonofking'Slanding",
                                  'Baratheon','Blackfyre','Blount','Buckwell','Byrch',
                                  'Bywater','Cargyll','Chelsted','Chyttering',
                                  'Cressey','Dargood','Darke','Darklyn','Darkwood',
                                  'Edgerton','Farring','Follard','Gaunt','Harte',
                                  'Hayford','Hogg','Hollard','Kettleblack',
                                  'Langward','Longwaters','Mallery','Manning',
                                  'Massey','Pyle','Rambton','Rollingford',
                                  'Rosby','Rykker','Staunton','Stokeworth',
                                  'Targaryen','Thorne','Wendwater','Bruneofthedyreden',
                                  'Bruneofbrownhollow','Boggs','BruneofBrownhollow',
                                  'BruneoftheDyreDen','Cave','Crabb','Hardy','Pyne',
                                  'Baratheonofdragonstone','Dayne','BarEmmon',
                                  'BaratheonofDragonstone','Celtigar','Sunglass',
                                  'Velaryon'],
                    'Dorne':['Dayneofhighhermitage','Allyrion','Blackmont',
                             'Briar','Brook','Brownhill','Dalt','DayneofHighHermitage',
                             'DayneofStarfall','Drinkwater','Dryland','Fowler',
                             'Gargalen','Holt','Hull','Jordayne','Ladybright',
                             'Lake','Manwoody','Martell','Qorgyle','Santagar',
                             'Shell','Toland','Uller','Vaith','Wade','Wells','Wyl'],
                    'IronIsland':['Harlawofthetowerofglimmering','Goodbrotherofshatterstone',
                                  'Stonehouse','Goodmasters','Kenningofharlaw',
                                  'Harlawofharlawhall','Farwyndofthelonelylight',
                                  'Harlawofharridanhill','Farwynd','Yronwood',
                                  'Blacktyde','Botley','Codd','Drumm',
                                  'FarwyndofSealskinPoint','FarwyndoftheLonelyLight',
                                  'GoodbrotherofCorpseLake','GoodbrotherofCrowSpikeKeep',
                                  'GoodbrotherofDowndelving','GoodbrotherofOrkmont',
                                  'GoodbrotherofShatterstone','GoodbrotherofHammerhorn',
                                  'Greyiron','Greyjoy','HarlawofHarlawHall',
                                  'HarlawofHarridanHill','HarlawofHarlaw',
                                  'HarlawoftheTowerofGlimmering','Humble',
                                  'Ironmaker','Kenning','Merlyn','Myre','Netley',
                                  'Orkwood','Saltcliffe','Sharp','Shepherd','Sparr',
                                  'StonehouseofOldWyk','Stonetree','Sunderly',
                                  'Tawney','Weaver','Wynch','Hoare','Harlawofgreygarden',
                                  'Andrik','Harlaw','Nute','Volmark'],
                    'North':['Stark','Nightswatch','Flint','Boltonofthedreadfort',
                             "Flintofwidow'Swatch",'Stout','Bolton','Amber',
                             'Ashwood','BoltonoftheDreadfort','BoltonofWinterfell',
                             'Cassel','Cerwyn','Condon','Dustin','FisheroftheStonyShore',
                             'FlintofBreakstoneHill',"FlintofFlint'sFinger",
                             "FlintofWidow'sWatch",'Frost','Glover','Greenwood',
                             'Greystark','Holt','Hornwood','Ironsmith','Karstark',
                             'Lake','Lightfoot','Locke','Long','Marsh','Mollen','Mormont',
                             'Moss','Overton','Poole','Redbeard','Ryder','Ryswell',
                             'SlateofBlackpool','StarkofWinterfell','StoutofGoldgrass',
                             'Tallhart','Thenn','Towers','Umber','Waterman','Wells',
                             'Whitehill','Woodfoot','Woolfield','Blackmyre','Boggs',
                             'Cray','Fenn','Greengood','Peat','Quagg','Reed','Burley',
                             'Flintofthemountains','Harclay','Knott','Liddle','Norrey',
                             'Wull','Crowl','Magnar','Stane','Bole','Branch',
                             'Forrester','Woods',],
                    'Reach_h':['Citadel','Maesters','Fossoway','Fossowayofciderhall',
                               'Tyrellofbrightwaterkeep','Fossowayofnewbarrel',
                               'Tyrell','Osgrey','Ambrose','Appleton','Ashford',
                               'Ball','Beesbury','Blackbar','Bridges','Bulwer',
                               'Bushy','Caswell','Cockshaw','Conklyn','Cordwayner',
                               'Costayne','Crane','Cuy','Dunn','Durwell','Florent',
                               'Footly','FossowayofCiderHall','FossowayofNewBarrel',
                               'Gardener','Graceford','Graves','Hastwyck','Hightower',
                               'Hunt','Hutcheson','Inchfield','Kidwell','Leygood',
                               'Lowther','Lyberr','Meadows','Merryweather',
                               'Middlebury','Mullendore','Norcross','Norridge',
                               'Oakheart','Oldflowers','Orme','OsgreyofLeafyLake',
                               'OsgreyofStandfast','Peake','Pommingham','Redding',
                               'Redwyne','Rhysling','Risley','Rowan','Roxton',
                               'Shermer','Sloane','Stackhouse','Tarly',
                               'TyrellofBrightwaterKeep','TyrellofHighgarden',
                               'Uffering','Varner','Vyrwel','Webber','Westbrook',
                               'Willum','Woodwright','Wythers','Yelshire'],
                    'ShieldIsland':['Chester','Grimm','Hewett','Serry',],
                    'Riverlands_r':['VanceofWayfarersRest',"Vanceofwayfarer'Srest",
                                    'Vanceofatranta','Darry','Freyofriverrun',
                                    'Vance','Frey','Baelish','Bigglestone',
                                    'Blanetree','Bracken','Butterwell','Chambers',
                                    'Charlton','CoxofSaltpans','DarryofDarry',
                                    'Deddings','Erenford','FisheroftheMistyIsle',
                                    'FreyofRiverrun','FreyoftheCrossing','Goodbrook',
                                    'Grell','Grey','Haigh','Harroway','Hawick',
                                    'Heddle','Hook','Justman','Keath','Lannister',
                                    'Lolliston','Lothston','Lychester','Mallister',
                                    'Mooton','Mudd','Nayland','Nutt','Paege',
                                    'Pemford','Perryn','Piper','Qoherys','Roote',
                                    'Ryger','Shawney','Smallwood','Strong','Teague',
                                    'Terrick','TowersofHarrenhal','Tully',
                                    'VanceofAtranta',"VanceofWayfarer'sRest",
                                    'Vypren','Wayn','Whent','Wode'],
                    'Stormlands':['Baratheon','Bolling','Buckler','Cafferen',
                                  'Cole','Connington','Durrandon','Errol',
                                  'Estermont','Fell','Gower','Grandison','Hasty',
                                  'Herston','Horpe','Kellington','Lonmouth',
                                  'Mertyns','Morrigen','Musgood','Peasebury',
                                  'Penrose','Rogers','Seaworth','Staedmon',
                                  'Swygert','Tarth','Toyne','Trant','Tudbury',
                                  'Wagstaff','Wensington','Whitehead','Wyldee',
                                  'Wylde','Caron','Dondarrion','FooteofNightsong',
                                  'Selmy','Swann',],
                    'NotClear':['H_Unknow','AlchemistsGuild','Goodbrother','Thirteen',
                                'Faithoftheseven','Brotherhoodwithoutbanners',
                                'Kingdomofthethreedaughters','Khal','Kingswoodbrotherhood',
                                'KingswoodBrotherhood''AlchemistsGuild', 'Rhllor', 
                                'Chatayasbrothel','Bandofnine','Blackears','Kandaq',
                                'Pureborn','Summerislands','Moonbrothers','Ironbankofbraavos',
                                'Mancerayder', 'Merreq','Peach','Pahl', 'Seawatch',
                                'Wildling', 'Undyingones', 'Unsullied','Stormcrows', 
                                'Cox', 'Graces','Companyofthecat','Facelessmen',
                                'Burnedmen','Loraq','Galare','Antlermen','Happyport',
                                "Alchemists'Guild", 'Wisemasters','Thecitadel',
                                'Three-Eyedcrow',"Chataya'Sbrothel",'Reznak','Drownedmen',
                                'Blackwood','Ghazeen','Slynt',"R'Hllor",'Manderly',
                                'Secondsons','Bravecompanions','NymerosMartell',
                                'Blacks','GoldenCompany','Windblown','Belgrave',
                                'Blackberry','Cupps','Farrow','Goode','Greenhill',
                                'Leek','Mandrake','Penny','Potter','Sawyer',
                                'Strickland','Suggs'],
                    'ValeofArryn':['Shettofgulltower','Royceofthegatesofthemoon',
                                   'Royce','Arryn','ArrynofGulltown',
                                   'ArrynoftheEyrie','Baelish','Belmore',
                                   'Breakstone','Brightstone','Coldwater',
                                   'Corbray','Donniger','Egen','Elesham','Grafton',
                                   'Hardyng','Hersy','Hunter','Lipps','Lynderly',
                                   'Melcolm','Moore','Pryor','Redfort',
                                   'RoyceofRunestone','RoyceoftheGatesoftheMoon',
                                   'Ruthermont','Shell','ShettofGullTower',
                                   'ShettofGulltown','Templeton','Tollett',
                                   'Upcliff','Waxley','Waynwood','Wydman','Borrell',
                                   'Longthorpe','Sunderland','Torrent',],
                    'Westerlands':['Lannisteroflannisport', 'Kenningofkayce',
                                   'LannisterOfCasterlyRock','Algood','Banefort',
                                   'Bettley','Brax','Broom','Casterly','Clegane',
                                   'Clifton','Crakehall','Doggett','Drox','Estren',
                                   'Falwell','Farman','Ferren','Foote','Garner',
                                   'Greenfield','Hamell','Hawthorne','Hetherspoon',
                                   'Jast','Kenning','Kyndall','Lannett',
                                   'LannisterofCasterlyRock','LannisterofLannisport'
                                   ,'Lanny','Lantell','Lefford','Lorch','Lydden',
                                   'Marbrand','Moreland','Myatt','Payne','Peckledon',
                                   'Plumm','Prester','Reyne','Ruttiger','Sarsfield',
                                   'Serrett','Spicer','Stackspear','Swyft','Tarbeck',
                                   'Turnberry','Vikary','Westerling','Yarwyck','Yew'],
                    }

#Lower all values in dic
for k, v in house_region_lst.items():
    new_v = []
    for vi in v:
        new_v.append(vi.lower())
    house_region_lst[k] = new_v
    
#replace values in dataframe by key for house_by_region column
def get_house(value):
    v = [k for (k, v) in house_region_lst.items() if value.lower() in v]
    return v[0] if len(v) > 0 else value

data.loc[:, 'house_by_region'] = [get_house(x) for x in data.house]

data['house_by_region'].unique()#13 variables now


#find death/alive info for each house_region
da_house_re = data.groupby(['house_by_region','isAlive']).count()["S.No"].unstack().copy(deep=True)
da_house_re.columns = ['Dead','Alive']
da_house_re['%death']=da_house_re['Dead']/(da_house_re['Dead']+da_house_re['Alive'])



#############
# Title
#############   
#check for 'title' 
tit_lst=data['title'].unique()
len(tit_lst)#303

#flag missing title
data['m_title'] = data['title'].isnull().astype(int)

data['title']=data['title'].fillna("tit_unknown")

data['title_rg']=copy.deepcopy(data['title'])


##regroup title
lord = r'.*[l|L]ord.*'
king = r'.*[k|K]ing.*'
queen = r'.*[q|Q]ueen.*'
princess = r'.*[p|P]rincess.*'
prince = r'.*[p|P]rince.*'
master = r'.*[m|M]aster.*'
lady = r'.*[l|L]ady.*'
measter = r'.*[m|M]aester.*'
knight = r'.*[k|K]night.*'
commander = r'.*[c|C]ommander.*'
captain = r'.*[C|c]aptain.*'
ser = r'.*[S|s]er.*'
steward = r'.*[S|s]teward.*'
castellan =  r'.*[C|c]astellan.*'

pattern_list = [lord,
                king,
                queen,
                princess,
                prince,
                master,
                lady,
                measter,
                knight,
                commander,
                captain,
                ser,
                steward,
                castellan]

#group Lord and Lady, King and Queen, Prince and Princess since the gender
#info are not representative
replace_list = ['Lord/Lady', 'King/Queen', 'King/Queen', 'Prince/Princess', 
                'Prince/Princess', 'Lord/Lady', 'Lord/Lady','Maester','Knight',
                'Ser','Knight','Ser','Knight','Lord/Lady']
title_list2 = tit_lst


title_dic = {}

for r in replace_list:
    title_dic[r] = []

for idx, p in enumerate(pattern_list):
    for t in title_list2:
        if type(t) == type('str') and re.match(p, t) is not None:
            title_dic[replace_list[idx]].append(t)
            

#add in more titles
title_dic.update({ 'With_title':['Undergaoler', 'Duskendale', 'Dragonstone',
                                 'First Builder', 'Nightsong', 'Warlock','Green Grace', 
                                 'Seneschal', "\nWater Dancer", 'Ruddy Hall',
                                 'First Ranger','Blue Grace', 'Last Hearth',
                                 'Skyreach','Bucket','Goodman', 'Mistress of whisperers',
                                 'Keeper of the Gates of the Moon','Seneschal of Sunspear',
                                 'First Sword of Braavos','First Sword of Braavos',
                                 'Magnar of Thenn','Chief Undergaoler',
                                 'Magister of Pentos','Broad Arch','Bloodrider',
                                 'High Septon', 'Septa', 'Red Priest', 'Godswife', 
                                 'Goodwife', 'Priest', 'Cupbearer', "Slave of R'hllor",
                                 'BrotherProctor','red hand','Brother']
                 })


    
#merge few titles
title_dic['Lord/Lady'].append('Starpike')
title_dic['Lord/Lady'].append('Casterly Rock')
title_dic['Lord/Lady'].append('Coldmoat')
title_dic['Maester'].append('Grand Measter')
title_dic['Maester'].append('Wisdom')
title_dic['Maester'].append('Septon')
title_dic['Maester'].append('Archmaester')
title_dic['Lord/Lady'].append('Khal')
title_dic['Lord/Lady'].append('Khalakka')
title_dic['Knight'].append('Ko')



#lower dic value and place dataframe values with dic key
for k, v in title_dic.items():
    new_v = []
    for vi in v:
        new_v.append(vi.lower())
    title_dic[k] = new_v


def get_title(value):
    v = [k for (k, v) in title_dic.items() if value.lower() in v]
    return v[0] if len(v) > 0 else value

data.loc[:, 'title_rg'] = [get_title(x) for x in data.title_rg]

#Check
data['title_rg'].unique()#eight variables 


#Title plot
plt.figure(figsize = (6,4))
plt.xlabel('Title')
plt.ylabel('Frequency')
data.title_rg.value_counts().sort_values(ascending = False)[:10].plot.bar()

#Tile with death rate
da_title = data.groupby(['title_rg','isAlive']).count()["S.No"].unstack().copy(deep=True)
da_title.columns = ['Dead','Alive']
da_title = da_title.fillna(0)
da_title['%death']=da_title['Dead']/(da_title['Dead']+da_title['Alive'])



#############
# Age
#############

#flag missing age
data['m_age'] = data['age'].isnull().astype(int)

#fill na
data['age'].fillna(data['age'].median(), inplace=True)#Swked, fill in median

#plot age 
sns.distplot(data['age'].dropna())
plt.show()



#############
# Book
############# 
#flag characters that never showed up in all books:

for i in range(0,1946):
    if data.loc[i,'book1-5']==0:
        data.loc[i,'notshow']=1
    else: data.loc[i,'notshow']=0



#############
# Check and Clean
#############   

#Check NAs
data.isnull().sum()

data.info()


#Drop columns with too many missing values and regrouped columns
clean_data=data.drop(['isAliveMother','isAliveFather','isAliveHeir','isAliveSpouse',
                      'isMarried','isNoble','dateOfBirth','mother','father','heir',
                      'spouse','name','title','house'], axis=1)



####################
#One hot encoding
###################

#Get dummies
dummie_lst = ['culture','title_rg','house_by_region']

for x in dummie_lst:
   exec('{} = pd.get_dummies(list(clean_data[x]), drop_first = True)'.format(x+'_dum'))
   

#merge the dataframe
data_dum = pd.concat(
        [clean_data.loc[:,:],
         culture_dum, 
         title_rg_dum,  
         house_by_region_dum
         ],
         axis = 1)

#drop catagracal data
data_dum = data_dum.drop(['title_rg','culture','house_by_region'], axis=1)



#Run correlation

df_corr = data_dum.corr().round(2)

corr_list = df_corr.loc['isAlive'].sort_values(ascending = False)
corr_list.head(30).index
corr_list.tail(15).index



#########################
#Split data set
########################
#Chose variables more significiant:42

for i in range(0,1946):
    if data.loc[i, 'house']=='Targaryen':
        data_dum.loc[i,'Targaryen']=1
    else: data_dum.loc[i,'Targaryen']=0

independent_1 = data_dum[['lucky_house', 'book4', 'lucky_culture', 'ndr_luck',
       'pop_luck', 'm_age', 'm_isAliveFather', 'm_isAliveMother', 'm_mother',
       'm_father', 'm_isAliveHeir', 'm_heir', 'NotClear', 'm_house', 'Maester',
       'tit_unknown', 'm_title', 'Braavosi', 'Reach_h', 'Ironborn', 'Dorne',
       'm_isAliveSpouse', 'Summer Islands', 'book1-5', 'With_title',
       'ShieldIsland','m_culture','Westermen', 'Astapori', 'book2', 'Lord/Lady', 
       'male', 'North','Prince/Princess', 'S.No', 'book1', 'age', 'popularity',
       'notshow','numDeadRelations', 'Valyrian', 'dateOfDeath']]

#Check corr between selected variables
inde_corr = independent_1.corr().round(2)



# Scaling the data
from sklearn.preprocessing import StandardScaler


# Instantiating a StandardScaler() object
scaler = StandardScaler()


# Fitting the scaler with our data
scaler.fit(independent_1)


# Transforming our data after fit
X_scaled = scaler.transform(independent_1)


# Putting our scaled data into a DataFrame
X_scaled_df = pd.DataFrame(X_scaled)


# Adding labels to our scaled DataFrame
X_scaled_df.columns = independent_1.columns


#Featrues and Target
target =  data.loc[:,'isAlive']
independent = X_scaled_df


#Split data set
X_train, X_test, y_train, y_test = train_test_split(
            independent,
            target,
            test_size = 0.10,
            random_state = 508,
            stratify = target)


##############################################################################
##############################################################################
######                          Modeling                              ########
##############################################################################
##############################################################################
'''
Following are working models but with less good values 
'''

################################################################################
## Hyperparameter Tuning with Logistic Regression
################################################################################
#
#logreg = LogisticRegression(C = 500,
#                            solver = 'lbfgs')
#
#
#logreg_fit = logreg.fit(X_train, y_train)
#
#
#logreg_pred = logreg_fit.predict(X_test)
#
#
## Let's compare the testing score to the training score.
#print('Training Score', logreg_fit.score(X_train, y_train).round(4))
#print('Testing Score:', logreg_fit.score(X_test, y_test).round(4))
#
#
#cv_lr_3 = cross_val_score(logreg, independent, target, scoring='roc_auc', cv = 3)
#
#print(cv_lr_3)
#print(pd.np.mean(cv_lr_3).round(4))
#
#'''
#Training Score 0.8607
#Testing Score: 0.8718
#[0.8620586  0.85767593 0.94007152]
#0.8867
#'''



###############################################################################
# Gradient Boosted Machines
###############################################################################

####################
# Applying RandomizedSearchCV to Blueprint GBM
####################



## Creating a hyperparameter grid
#learn_space = pd.np.arange(0.01, 2.01, 0.05)
#estimator_space = pd.np.arange(50, 1000, 50)
#depth_space = pd.np.arange(1, 10)
#leaf_space = pd.np.arange(1, 150, 15)
#criterion_space = ['friedman_mse', 'mse', 'mae']
#
#
#param_grid = {'learning_rate' : learn_space,
#              'n_estimators' : estimator_space,
#              'max_depth' : depth_space,
#              'min_samples_leaf' : leaf_space,
#              'criterion' : criterion_space}
#
#
#
## Building the model object one more time
#gbm_grid = GradientBoostingRegressor(random_state = 508)
#
#
#
## Creating a GridSearchCV object
#gbm_grid_cv = RandomizedSearchCV(estimator = gbm_grid,
#                                 param_distributions = param_grid,
#                                 n_iter = 50,
#                                 scoring = None,
#                                 cv = 3,
#                                 random_state = 508)
#
#
#
## Fit it to the training data
#gbm_grid_cv.fit(X_train, y_train.values.ravel())
#
#
#
## Print the optimal parameters and best score
#print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
#print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))
#
#
#
#########################
## Building GBM Model Based on Best Parameters
#########################
#
#gbm_optimal = GradientBoostingClassifier(criterion = 'mse',
#                                      learning_rate = 0.01,
#                                      max_depth = 3,
#                                      min_samples_leaf = 42,
#                                      n_estimators = 260)
#
#
#
#gbm_optimal_fit = gbm_optimal.fit(X_train, y_train.values.ravel())
#
#
#gbm_optimal_score = gbm_optimal.score(X_test, y_test)
#
#
#gbm_optimal_pred = gbm_optimal.predict(X_test)
#
#
## Training and Testing Scores
#print('Training Score', gbm_optimal.score(X_train, y_train).round(4))
#print('Testing Score:', gbm_optimal.score(X_test, y_test).round(4))
#
#
#cv_lr_3 = cross_val_score(gbm_optimal, independent, target, scoring='roc_auc', cv = 3)
#
#print(cv_lr_3)
#print('AUC:',pd.np.mean(cv_lr_3))
#
#'''
#Training Score 0.8646
#Testing Score: 0.8564
#[0.86421863 0.86267844 0.91895978]
#AUC: 0.8819522846831506
#'''
#
################################################################################
## XGBoosting
################################################################################
#
##########################
### Parameter tuning with GridSearchCV
##########################
#
#
#param_test1 = {
#'max_depth':range(1,6,1),
#'min_child_weight':range(1,6,1),
#'gamma':[i/10.0 for i in range(0,5)],
#'subsample':[i/10.0 for i in range(6,10)],
#'colsample_bytree':[i/10.0 for i in range(6,10)],
#'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
#
#}
#gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, 
#                                                   n_estimators=350, 
#                                                   max_depth=4,
#                                                   min_child_weight=1, 
#                                                   gamma=0.4,
#                                                   scale_pos_weight = 1,
#                                                   subsample=0.8, 
#                                                   colsample_bytree=0.8,
#                                                   objective= 'binary:logistic', 
#                                                   seed=508), 
#
#param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch1.fit(X_train, y_train)
#gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_




#########################
## Building XGBoost Model Based on Best Parameters
#########################


xgboost_optimal = XGBClassifier( learning_rate =0.01, 
                                 n_estimators=350, 
                                 max_depth=3,
                                 min_child_weight=1, 
                                 gamma=0.0, 
                                 subsample=1.0, 
                                 colsample_bytree=0.8,
                                 objective= 'binary:logistic',
                                 reg_alpha= 0.0)


xgboost_optimal.fit(X_train, y_train)
xgboost_optimal_pred = xgboost_optimal.predict(X_test)

print('Training Score', xgboost_optimal.score(X_train, y_train).round(4))
print('Testing Score:', xgboost_optimal.score(X_test, y_test).round(4))

xgboost_optimal_train = xgboost_optimal.score(X_train, y_train)
xgboost_optimal_test  = xgboost_optimal.score(X_test, y_test)


#Cross-Val
cv_lr_3 = cross_val_score(xgboost_optimal, independent, target,scoring='roc_auc', cv = 3)

print(cv_lr_3)
print('AUC:',pd.np.mean(cv_lr_3).round(3))


'''
Training Score 0.8784
Testing Score: 0.8615
[0.86172677 0.87054846 0.93990213]
AUC: 0.891
'''


########################
# Confusion Matrix
########################

from sklearn.metrics import confusion_matrix

confusion_matrix(y_true = y_test,
                 y_pred = xgboost_optimal_pred)


# Visualizing a confusion matrix

labels = ['Dead', 'Alive']

cm = confusion_matrix(y_true = y_test,
                      y_pred = xgboost_optimal_pred)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            cmap = 'Pastel1_r')


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.savefig('Confusion Matrix of Alive or Dead.png')
plt.show()


'''
Good True positive
'''
########################
# Creating a classification report
########################

from sklearn.metrics import classification_report

classification_report(y_true = y_test,
                      y_pred = xgboost_optimal_pred,
                      target_names = labels)

'''
 precision    recall  f1-score   support

        Dead       0.85      0.56      0.67        50
       Alive       0.86      0.97      0.91       145

   micro avg       0.86      0.86      0.86       195
   macro avg       0.86      0.76      0.79       195
weighted avg       0.86      0.86      0.85       195
'''


########################
# Saving Results
########################
model_predictions_df = pd.DataFrame({'Actual':y_test,
                                     'XGB': xgboost_optimal_pred})

model_predictions_df.to_excel("GOT_model_predictions.xlsx")
