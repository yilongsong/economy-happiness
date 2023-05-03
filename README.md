
## Inferring Happiness from Wealth: with Data in the World Happiness Report, 2023

Computational Analysis of Big Data, DIS Copenhagen Semester, Final Project.

![Happiness Scores of 137 Countries from the World Happiness Report, 2023](https://cdn-images-1.medium.com/max/2000/1*zqYy_p4Egq1LNJRY3g8Hqg.png)

So we ask ourselves: on the global level, does money buy happiness?

The [World Happiness Report](https://worldhappiness.report/) (WHR)— a respectable, comprehensive, and philosophically robust study of happiness on the global level — is concerned with more pressing and academic matters, such as the effects of recent crises (wars, the pandemic), inequality, virtuous behavior, policies, etc. have on well-being. But it also provides data that could potentially answer our cliché of a question — upon applying some simple methods of big data analysis, that is.

To keep our analysis current and since our question lacks the time dimension (i.e. we aren’t studying any trend), we will use the World Happiness Report’s 2023 data available on [https://worldhappiness.report/ed/2023/#appendices-and-data](https://worldhappiness.report/ed/2023/#appendices-and-data). [The file](https://happiness-report.s3.amazonaws.com/2023/DataForFigure2.1WHR2023.xls) consists of 137 rows corresponding to 137 countries the report includes and 19 columns. A full, precise description of all the columns can be found in [the report](https://happiness-report.s3.amazonaws.com/2023/WHR+23.pdf). We cover briefly here the general gist and the most relevant columns.

## Data

At the center of everyone’s interest is the “Ladder score” column. “Ladder score” measures happiness, and thus is just another term for “happiness score.” These happiness scores under the “Ladder score” column are collected by the Gallup World Poll (GWP) by “[asking] respondents to evaluate their current life as a whole using the image of a ladder, with the best possible life for them as a 10 and worst possible as a 0,” receiving “around 1,000 responses … annually for each country,” and “constructing population-representative national averages for each year in each country” based on these responses. The happiness scores and ranking, additionally, are “[based] on a three-year average of these life evaluations, since the larger sample size enables more precise estimates.”

The six factors: “Logged GDP per capita,” “Social support,” “Healthy life expectancy,” “Freedom to make life choices,” “Generosity,” and “Perceptions of corruption” are features that the researchers for the WHR deem necessary and sufficient in explaining happiness. The values under “Logged GDP per capita” and “Healthy life expectancy” are objective data able to be directly collected from statistical services. The other values are weighted averages based on binary choice poll results in the corresponding territories. “Ladder score in Dystopia,” incidentally, is the happiness score of “[the] hypothetical country [of Dystopia] with values equal to the world’s lowest national averages for each of the six factors,” and provides a lower bound to the happiness scores (so that the summation does not start from zero). The “Explained by …” columns are weighted versions of the six factors that, adding “Dystopia + residual,” sum up to the “ladder score,” but they aren’t used to compute the “ladder scores” — those, as mentioned earlier, are directly determined by GWP polls.

By the nature of our question, the two columns we will mainly focus on are “Ladder score” and “Logged GDP per capita.” To motivate our investigation and to get a very rough sense of the correlation we will be investigating, we can compare the top 10 countries by happiness versus that by GDP per capita.

![Happiness and wealth rankings](https://cdn-images-1.medium.com/max/2000/1*DAh3DvrsNLtBIYnXgLuG-Q.png)

Looking at the GDP rankings of the happiest countries and the happiness rankings of the countries with the highest GDP, we see that while happiness and GDP seem to correlate somewhat in the grand scheme of things (given that there are 137 countries in total), i.e., happy countries generally have high GDP ranking, and vice versa, happiness ranking and GDP ranking do not always match up.

We can further observe the complexity in the correlation of interest by looking at the importance of wealth for happiness in each country. A naive way of accomplishing this is to simply compute the logged GDP per capita to happiness ratio.

![GDP per capita to happiness ratio](https://cdn-images-1.medium.com/max/2000/1*BXKDNJEEnYiYQxQ5opjW3A.png)

The irregularity here is more obvious: while countries in which wealth to happiness ratio is high are (understandably) all ranked near the bottom of the happiness ranking, countries in which wealth to happiness ratio is low do not consist of only countries ranked near the top of the happiness ranking.

The next section discusses the methods we will use to understand such patterns and irregularities at a deep level.

## Methods

Since we’re working with numerical data, our initial question may be translated to: how well can we predict a country’s happiness score using its per capita GDP?

We first use a simple (vanilla) neural network, taking in solely the GDP per capita values, to see how good of a happiness score prediction we may get.

    from keras.models import Sequential
    from keras.layers import Dense
    
    model = Sequential()
    
    model.add(Dense(64, input_shape=(1,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    
    model.compile(loss='mae', optimizer='adam')
    
    model.fit(shuffled_dataframe['Logged GDP per capita'][:100], shuffled_dataframe['Ladder score'][:100], epochs=1000)
    
    loss = model.evaluate(shuffled_dataframe['Logged GDP per capita'][100:137], shuffled_dataframe['Ladder score'][100:137])
    print("Test loss:", loss)

Note that we are using three fully connected hidden layers, each consisting of 64 neurons here, despite only having one number as our input. This is so that our future predictions, which add more features to the input, can run on a model with maximally similar architecture, making the testing losses more comparable. It is also empirically verifiable that a model with fewer parameters yields similar results (around a mean absolute error of 0.5 depending on the shuffling of the data frame), so this more complex model isn’t in fact causing issues that overly complex models usually cause, like overfitting.

After testing on this model and staring at the world map, I wondered how well a neural network model would perform if given not only the country’s GDP per capita values, but also its region. After adding columns to shuffled_dataframe that encode information about a country’s region, I created my training and testing array as such:

    features = ['Logged GDP per capita', 'Australia and New Zealand', 'Central Asia', 'Eastern Asia',
           'Eastern Europe', 'Latin America and the Caribbean', 'Melanesia',
           'Micronesia', 'Northern Africa', 'Northern America', 'Northern Europe',
           'Polynesia', 'South-eastern Asia', 'Southern Asia', 'Southern Europe',
           'Sub-Saharan Africa', 'Western Asia', 'Western Europe']
    x = shuffled_dataframe[features].values

The encoding is such that for each country or row, if the country belongs to the region that is the name of the column, the value of the column in that row gets set to 1, and 0 if not. And, as mentioned earlier, we use the same model (the only difference being it now takes 18 numbers as input) and see how well it predicts the happiness scores now with regions of the countries encoded.

    model = Sequential()
    
    model.add(Dense(64, input_shape=(18,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    
    model.compile(loss='mae', optimizer='adam')
    
    model.fit(x[:100], shuffled_dataframe['Ladder score'][:100], epochs=1000)
    
    loss = model.evaluate(x[100:137], shuffled_dataframe['Ladder score'][100:137])
    print("Test loss:", loss)

Then, out of curiosity, we use all six featured mentioned earlier: “Logged GDP per capita,” “Social support,” “Healthy life expectancy,” “Freedom to make life choices,” “Generosity,” and “Perceptions of corruption” for prediction with a similar architecture.

    features = ['Logged GDP per capita',
           'Social support', 'Healthy life expectancy',
           'Freedom to make life choices', 'Generosity',
           'Perceptions of corruption']
    x = shuffled_dataframe[features].values
    
    model = Sequential()
    
    model.add(Dense(64, input_shape=(6,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    
    model.compile(loss='mae', optimizer='adam')
    
    model.fit(x[:100], shuffled_dataframe['Ladder score'][:100], epochs=1000)
    
    loss = model.evaluate(x[100:137], shuffled_dataframe['Ladder score'][100:137])
    print("Test loss:", loss)

Then arises the question: how important is each of the six features in determining the happiness scores? We may thus do a feature importance analysis using sklearn’s RandomForestRegressor.

    from sklearn.ensemble import RandomForestRegressor
    
    X = dataframe[['Logged GDP per capita',
           'Social support', 'Healthy life expectancy',
           'Freedom to make life choices', 'Generosity',
           'Perceptions of corruption']]
    y = dataframe['Ladder score']
    
    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    
    rfr.fit(X, y)
    
    importances = rfr.feature_importances_

Since we have tested if supplying region information of countries improves happiness score predictions, we now test the direction and amount in which each region affect the prediction. First, we use the fitted RandomForestRegressor.

    regions = ['Australia and New Zealand', 'Central Asia', 'Eastern Asia',
           'Eastern Europe', 'Latin America and the Caribbean', 'Melanesia',
           'Micronesia', 'Northern Africa', 'Northern America', 'Northern Europe',
           'Polynesia', 'South-eastern Asia', 'Southern Asia', 'Southern Europe',
           'Sub-Saharan Africa', 'Western Asia', 'Western Europe']
    relevant_features = ['Logged GDP per capita',
           'Social support', 'Healthy life expectancy',
           'Freedom to make life choices', 'Generosity',
           'Perceptions of corruption', 'Ladder score in Dystopia']
    
    # Create X_test with the mean of all countries for the seven features
    X_test = [dataframe[feature].mean() for feature in relevant_features]
    # Add region encoding (no region encoding at first)
    for i in range(len(regions)):
        X_test.append(0)
    X_test_df = pd.DataFrame([X_test], columns=relevant_features+regions)
    
    # Get predictions
    predictions = []
    # Use region = Melanesia as baseline, as feature importance indicates that
    # Melanesia = 1 has no importance in prediction
    X_test_df['Melanesia'] = 1
    prediction = rfr.predict(X_test_df)
    print('Base:', prediction)
    predictions.append(prediction)
    X_test_df['Melanesia'] = 0
    # Now get prediction for each region
    for region in regions:
        X_test_region = X_test_df.copy()
        X_test_region.loc[0, region] = 1
        prediction = rfr.predict(X_test_region)
        print(region, prediction)
        predictions.append(prediction)

Then we train a neural network that takes in all seven features (the six factors plus “Ladder score in Dystopia”) plus the region features (the same input as the RandomForestRegressor we used) and again test how each region impact prediction for comparison.

    features = ['Logged GDP per capita',
           'Social support', 'Healthy life expectancy',
           'Freedom to make life choices', 'Generosity',
           'Perceptions of corruption', 'Ladder score in Dystopia',
           'Australia and New Zealand', 'Central Asia', 'Eastern Asia',
           'Eastern Europe', 'Latin America and the Caribbean', 'Melanesia',
           'Micronesia', 'Northern Africa', 'Northern America', 'Northern Europe',
           'Polynesia', 'South-eastern Asia', 'Southern Asia', 'Southern Europe',
           'Sub-Saharan Africa', 'Western Asia', 'Western Europe']
    x = shuffled_dataframe[features].values
    
    model = Sequential()
    
    model.add(Dense(64, input_shape=(24,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    
    model.compile(loss='mae', optimizer='adam')
    
    model.fit(x[:100], shuffled_dataframe['Ladder score'][:100], epochs=1000)

## Results

The first model in the previous section that predicts happiness score using only per capita GDP values, after 1000 epochs of training using the ADAM optimizer on data from 100 countries, yields a mean absolute error of 0.551 in testing loss in this instance. Here is the model’s happiness score predictions of the rest of the countries (of which there are 37) versus actual happiness scores of these countries plotted.

 <iframe src="https://medium.com/media/af88c863eb9f0c1c2ca152b4b679fc41" frameborder=0></iframe>

The mean absolute error varies slightly given different versions of shuffled_dataframe (i.e., given different partitions of training and test data). For comparison with the following models, we will conduct a brief but more thorough investigation of the distribution of this error later.

The second model, one that predicts using GDP per capita values and region information of the country, yields a mean absolute error of 0.446 in testing loss using the same training and testing data. The results are again plotted below.

 <iframe src="https://medium.com/media/960763f7a40bd172d4129c0a3c0a6cbd" frameborder=0></iframe>

The third model, which uses the six features to predict (not taking region into account), interestingly obtains a mean absolute error of only 0.554 in testing loss given, again, the identical training and testing data. Here is the plot.

 <iframe src="https://medium.com/media/44a35bc7ac3586c1dadc4f2ae6d46bca" frameborder=0></iframe>

While it may be surprising that the third model has a similar performance to the first, we now test and compare the performances of these models more concretely by shuffling the dataframe 20 times, training each model on each of the shuffled dataframe, and recording the mean absolute losses. Computing, first, the mean differences between the arrays of 20 errors (the exact values are available under cell 16 of analyzer.ipynb in our GitHub repository), we get that errors of the first model minus that of the second model is 0.046 on average; between first and third, we have 0.051 (predicting only with GDP having higher errors); between second and third, 0.006 (predicting with GDP and region having higher errors). Furthermore, in 17 of the 20 instances, predicting with GDP and region has a lower error than predicting only with GDP. This is true in 15 of the 20 instances for predicting with all six factors versus predicting only with GDP, and 12 out of 20 for predicting with all six factors versus with GDP and region. This shows that our initial results, showing the model that predicts using all six factors as being less accurate than both the first and second model, is an anomaly. But it is accurate to observe that predicting with all six factors is similar to predicting with GDP and region in terms of accuracy. It is also interesting to observe the mean error of each model across the 20 instances of training: the first model has a mean mean absolute error (mean MAE) of 0.505, the second of 0.459, the third of 0.453.

Now, doing a feature importance analysis using a random forest regressor on the six features plus “Ladder score in Dystopia” as described in the previous section, we get the following results:

![Result of feature importance analysis on the seven features](https://cdn-images-1.medium.com/max/2000/1*LSMDbPolL2wd8wQEoRFv3Q.png)

Note that since “Ladder score in Dystopia” is a constant (identical for every country), it expectedly has 0 importance in determining ladder scores. On the other hand, social support appears to be the most important of the rest of the six features, followed by GDP per capita, then freedom to make life choices, healthy life expectancy, perceptions of corruption, and lastly, generosity.

Given the importance of social support, we thought that it would be interesting to see how well our deep learning model predicts using the country’s social support data. It turns out to not be great. The mean absolute error is 0.634 given the same set of training and testing data as the previous deep learning models. The corresponding plot can be found immediately below this paragraph. Averaging over 20 instances of training like we did with the three models and using the same 20 shuffled dataframes, we get a mean MAE of 0.514. This means, although the MAE of 0.634 obtained earlier is significantly above average, the performance of this model is still similar to and slightly worse than the previous three models.

 <iframe src="https://medium.com/media/1b3eef3972a938ed4c2593d648dc029d" frameborder=0></iframe>

Finally, since providing the countries’ regions benefits neural network predictions, we investigate, as mentioned in the previous section, how a country belonging to different regions impact its happiness prediction differently. Here is the result using the previously fitted random forest regressor. Note that for the “Base” prediction we set the country to belong to the Melanesia region (as visible in the code we have previously supplied), because a feature importance analysis verifiably yields the result that a country belonging to this region does not its impact happiness prediction (0 in importance score) or has negligible impact.

![](https://cdn-images-1.medium.com/max/2000/1*UgFxdNLmUrcWgu9KRk1FDA.png)

The differences here are relatively small with few exceptions. However, when performing the same test using a neural network, we get the following, more drastic results.

![](https://cdn-images-1.medium.com/max/2000/1*PU2mRapCVUP4Wwm92NPjjg.png)

## Conclusion

From comparing the performances of the three deep learning models, we can make three observations:

 1. GDP per capita is a reasonably good predictor of the happiness score. With a mean MAE of around 0.5, GDP per capita predicts the happiness score of countries roughly within an interval of 1 — which is about the difference in happiness score between the country ranked first (Finland) and the country ranked nineteenth (the UK). Given that there are 137 countries on the WHR and the difference in happiness score between the country ranked first and the country ranked last being around 7, we see the effectiveness of per capita GDP in predicting happiness.

 2. Region does matter, but only slightly. We conjecture that it matters because culture may have some impact on how one might answer a poll question about their current happiness, and countries near each other often have some level of cultural similarities. On the other side of the same coin, however, the reason for it only mattering slightly, we conjecture, may be the cultural varieties in some regions.

 3. The fact that using all six factors which the WHR deems necessary and sufficient in determining happiness yields only predictions slightly better using only GDP per capita may indicate that these six factors are **not** in fact necessary and sufficient in determining happiness. Our analysis relevant to this observation is an interesting interaction between philosophy — whose principles the authors of the WHR relied on when coming up with the six factors — and computing, as the latter appears to be pointing out insufficiency in the former.

From our feature importance analysis we obtain the interesting result that social support is by far the most important feature when determining happiness. This again presents the interesting philosophical implication that social, interpersonal connections and a sense of community form the most important aspect to one’s happiness. Our later test on how a country belonging to each region affects happiness predictions also shows interesting correlations between each region and happiness.

However, through testing we find that deep learning predictions do not always display trends identical to random forest predictions. Furthermore, when observing the impact of each region on happiness prediction, the neural network displays more drastic differences in its predictions across different regions, while the random forest regressions give predictions with only small variations. By comparing the “Happiness Prediction based on Region” charts, we may also observe that trends presented by the two types of models do not entirely overlap. This may in part be due to the varying and unavoidable inaccuracies inherent in both models, but the fact that we cannot fully explain how and why these differences exist shows that we need to deepen our understanding in neural networks and random forests, which is a possible task in future investigations.
