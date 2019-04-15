# Game-of-Throne
A Song of Ice and Fire are a fantasy book series written by George R.R. Martin. There are 2432
named characters in all five books: 692 characters are dead, 1279 are alive, 308 are historical, 23 are
fictional, and 131 are unknown. The death of characters in this series is very unpredictable, and it
becomes a big disadvantage when one bets money on who will die first in Season 8 on the HBO series,
Game of Thrones, (which was inspired by Martin’s book series). To increase the chance of winning this
bet, a model will be created to predict deaths.
 To build the model, a dataset derived from the books’ information has been provided, which
includes 1946 observations, with 24 features in total, including Character number(by order of
appearance), character name, their title, gender, culture, date of birth, mother, father, heir, house, spouse,
book, age, number of dead relations, whether or not their mother/father/heir/spouse is alive, and whether
or not they are married or a noble. Due to the numerous data errors in name, gender, title, culture and
house, an external data resource was applied to correct the values. With the corrected information, the
gender distribution of characters was spread unequally, with about 20% being female, and 80% being
male. Thus, the gender’s information will not be used for modeling. The age of characters is from 0 to
100, with most characters being between 15 to 25 years old. The title, culture and house were regrouped
based on keywords or regions to increase sample quantities per segment. Information about mother, father,
heir have over 95% of the observations missing, therefore those features will not be used for further
analysis as well.
Key Insights
After cleaning the data and running the regression, a few patterns emerged for predicting a character’s
death. The age for dead characters is mostly between 10 to 30, or 90 to 100 years old. Characters who
are very popular (popularity > 0.9) or even very unpopular (popularity < 0.3) tend to survive longer.
3
Characters who have less than 3 death relations tend to survive longer.Characters appeared in a smaller
number of books within the series tend to survive longer; characters who do not appear in any of the
books are either historical characters or fictional characters, which therefore makes sense that they would
have a high death rate. Characters in the main cultures, like Northmen, Iron born and Free Folk, have a
higher risk of death; on the other hand, characters in smaller cultures are more likely to survive.
Characters in some of the major houses, like the Targaryens and the Night’s Watch, have a higher risk of
death compared to some of the other major houses, like Stark, Lannister and Greyjoy. In addition,
characters who have no clear house information or are in the smaller houses are more likely to survive.
Characters in the region of Crownlands have a 50% risk of death; whereas in the North, they have a 35%
risk of death.Characters with the titles of ‘Prince’ or ‘Princess’ have the highest death rate (60%); whereas
characters with the title of ‘Maester’ have the lowest death rate (12%).The model predicting alive very
well, but death prediction still has 14% mistake percentage. The final AUC value is 0.891.
Recommendation
 Based on death patterns in features such as house, regions, cultures, titles and number of dead
relations, the suggested betting strategy for the top characters for the next season as follow: Daenarys is
a Targaryen in the Crownlands, with both the title of ‘Princess’ and ‘Queen’, as well as 15 dead
relations. She is the most likely main character to die in the next season. Her mortality rate is 5 star. Jon
Snow is a Stark in the North, with the title of ‘Lord’, who has 5 dead relations. His mortality rate is 5
star. Arya is a Stark in the North, with the title of ‘Princess’, who has 8 dead relations. Her motality rate
is four star. Tyrion is a Lannister in the Westerlands, with the title of ‘Master’ and ‘Lord’, who has 12
dead relations. His mortality is three star. Samwell is in the Night’s Watch from the House Tarly,
Westeros, who has no dead relations. His mortality is two star. Margaery is a Tyrell from the Reach,
with the title of ‘Queen’, and has three dead relations. Her mortality rate is two star.
4
 Reference
Wikepedia: A Song of Ice and Fire. Retrieved
From:https://en.wikipedia.org/wiki/A_Song_of_Ice_and_Fire
A Song of Ice and Fire Character Spreadsheet. scifi.stackexchange.com
https://docs.google.com/spreadsheets/d/1K8DJTFUUIZvQsgnkxv3d142SXTcmnTV4YuHsNubHjA/edit?pref=2&pli=1#gid=2038601944
An API of Ice And Fire: https://anapioficeandfire.com
