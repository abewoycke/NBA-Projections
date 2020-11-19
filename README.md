# National Basketball Association (NBA) Game Outcome Projections

_Who will win each NBA game? This classifier takes inputs of both team's various team statistics entering a game (i.e. 2-pt FG %, ASTs/game, winning %) and outputs the probability of each team winning._

__1. Data__

A wonderful CSV file containing game data from 2004-2020 was available at: https://www.kaggle.com/nathanlauga/nba-games

__2. Data Wrangling/Cleaning__

[Data Wrangling Notebook](https://github.com/abewoycke/NBA-Projections/blob/master/2_Data_Wrangling/NBA%20Projections%20Data%20Wrangling.ipynb)

I dealt with a handful of null/missing values. I verified some individual player stats that had been missing to make sure that missing values should be filled with 0. This was accurate.

I made several functions to transform the data into the form of observations that would be most useful. I wrote a function that compiles team's individual games in a season up to the game in question and returns their stats coming into that game. The resultant dataframe has rows that are both the home and away team's stats entering the game, along with the game result.

__3. EDA__

[EDA Notebook](https://github.com/abewoycke/NBA-Projections/blob/master/3_EDA/NBA%20Projections%20EDA.ipynb)

I did some manual feature transformations based on my domain knowledge. My main goal was to remove redundancies/collinearity in the predictor variables (i.e. transforming 2-pt FGM and 2-pt FGA into 2-pt FG%). I also used a Bayesian prior for each team's win percentage so that early season win percentages (that would normally be at 0% or 100%) wouldn't be outliers.  The hope is that this would make for more meaningful predictor features, and that I wouldn't have to limit the range of avaialble classifier models to those that deal with multicollinearity well.

![Variable Heatmap](https://github.com/abewoycke/NBA-Projections/blob/master/3_EDA/heatmap.png)

The resultant heat map showed a lot of data relationships that were predictable in their direction, although interesting in their magnitude. The top 5 variables correlated with the home team winning, in order of correlation, were:

1. Winning %
2. 2-point FG%
3. Assists per Game
4. Defensive Rebounds per Game
5. 3-point FG%

![Point Plot of Variable Relationships](https://github.com/abewoycke/NBA-Projections/blob/master/3_EDA/Normalized_Pointplot.png)

The above plot also shows the variable relationships. The more the orange line is to the right of the blue line, the more the home team being better in that statistic entering the game impacts their expected win %. These results mostly align with a normal basketball viewer's expectations. For example, if the home team has more assists per game than the away team, the home team is more likely to win the game.

Conversely, if the blue line is to the right of the orange line, the home team having a higher value in this category is negatively correlated with the home team's probability of winning the game. These statistics are things normally thought to be bad for a team. including turnovers and personal fouls.

One variable I was surprised by was teams having higher offensive rebounds are more likely to _lose_. This goes against my intuition. But perhaps you only have more chances for offensive rebounds if you miss shots!


__4. Modeling and Tuning__

[Modeling Notebook](https://github.com/abewoycke/NBA-Projections/blob/master/4_Preprocessing_Modeling/NBA%20Projections%20Preprocessing%20Modeling%20Clean.ipynb)

I focused on three models: GradientBoostingClassifier, a Neural Network, and a Logistic Regression. I also built hard and soft voting classifiers from those three. I used Area Under the Receiver Operator Curve (AUROC) as my primary metric for performance, because:
a) in this use case, I don't place a stronger value on either Type 1 or Type 2 errors
2) AUROC is better than pure accuracy in this case when the data is imbalanced (home teams win 59.6% of the time)

After tuning all three of the models with a combination of HyperOpt, GridSearch, and RandomSearch, the tuned __GradientBoostingClassifier__ performed the best with an AUROC of 0.697 and a classification accuracy of 66.2%.

__5. Model Predictions__

[Documentation Notebook](https://github.com/abewoycke/NBA-Projections/blob/master/5_Documentation/NBA%20Projections%20Documentation.ipynb)

The model performed fairly well. The accuracy being about 66% is good when compared with some other classifications (expert consensus picks come in at ~59%). This represents an ~11% increase in relative performance, or a 7% increase in classification accuracy overall.

![Roc Curve](https://github.com/abewoycke/NBA-Projections/blob/master/5_Documentation/roc.png)

The overall AUROC score is alright. I think that NBA games are inherently hard to predict at too high a clip. An AUROC in the area of 0.7 represents that the model makes somewhat more meaningful predictions than just randomly guessing or always guessing the dominant class.

![Precision Recall Curve](https://github.com/abewoycke/NBA-Projections/blob/master/5_Documentation/precision_recall.png)

In this case, I don't place a different value on Type 1 or Type 2 errors, so don't have a strong opinion on maximizing precision or recall.

![Confusion Matrix](https://github.com/abewoycke/NBA-Projections/blob/master/5_Documentation/confusion_matrix.png)

I would be interested in seeing if there are tweaks that could be made that would decrease the false positive rate. I attribute the high false positives to class imbalance (home teams win 59% of the time).

__6. Future Improvements__

If the data is available, I would be interested in expanding the dataset and looking at data before 2004. I would also be interested in building a pipeline to simulate full seasons using the dataset. Also, I would be interested in seeing how the model performs against truly unseen data with pre-2004 backtesting, and how it will perform in the upcoming 2020-2021 season (although that might be weird sample due to limited home fan attendance).

__7. Project Writeup:__

__8. Credits__

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
