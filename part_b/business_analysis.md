# B1: Problem Formulation
# a. ML Problem Formulation:

Problem type: Supervised Regression

Target variable: 'items_sold'
Input features:
- store_size (Categorical)
- promotion_type (Categorical)
- location_type (Categorical)
- is_weekend (Binary)
- is_festival (Binary)
- competition_density (Numerical)
- month, day_of_week (Engineered from date)

This is a regression problem as items_sold is a dynamic number not a category, it is supervised since we have historical data labelled where we know both the inputs like store conditions and promotions used as well as the outcome like items sold. This way the model learns from history sales data to predict future sales to help with promotion decisions. 

# b. Why items_sold is better than Revenue:
Since revenue is a product of Price and Quantity it is constantly affected by pricing decisions and not only by promotions, this means even during flat promotions revenue would reduce per item even when more customers are actually buying the product making it a misleading measure of promotion effectiveness. 

As items_sold would measure how much units were sold when each promotion is ran giving a clearer picture not distorted by the price changes. 

**Broader Principle:**
The principle as affect here is the target variable allignment where the variable must directly reflect what the model is trying to explain. 

# c. Alternative to Single Global Model:
Instead of a single global model which assumes all 50 stores react to the promotions in the same way which is not plausible as urban and rural stores would have a very different types of customers as well as price senstivity, we can use location based models which are trained locally for each type of store dividing them into smaller cluster of urban, semi-urban and rural stores, another advanced model would be the hierarchial model which would share global trends with all stores but still make store level adjustments to fine tune it to individual stores. 


Justification: The modelling strategy needs to reflect the real world structure of data as forcing one model to predict and assist with all stores would result in the model not fitting any of them well.

# B2: Data and EDA Strategy]
# a. Joining tables
These are the join keys for the 4 tables: 
- transactions : transaction_id, store_id, date, promotion_id
- store_attributes : store_id
- promotion_details : promotion_id
- calendar : date

**Join strategy:**

We will use transactions table as the base and bring in the other 3 tables one by one, starting with joining store_attributes on store_id populating the information about store size, location type, and etc. Join promotion_details on promotion_id which adds description and type of promotion. Next, join calendar on date which adds is_weekend and is_festival flags. 

All the joins should be left joins since we want to keep every transaction even if a row is missing in the lookup table.

**Grain of the Final Dataset:**
one row = one store, running one promotion, on one day. This ensures that each record captures a unique combination of store, date, and promotion, this is the same level at which items_sold is recorded and future predictions need to be made. 

**Aggregations before Modelling:**

- Average items_sold per store per month which helps remove noise for analysis
- Promotion frequency per store as it tells us how often does each store run promotion.
- Rolling averages of items_sold over 7 or 30 days to ensure we capture the latest sales momentum

# b. Exploratory Data Analysis
- **distribution of items_sold - Histogram**

Plot a histogram of target variable and look for skewness, outliers or multimodal peaks. If the distribution is heavily right-skewed, a log transformation of items_sold may improve model performance by reducing the influence of extreme values.

- **Average items_sold by promotion_type - Bar chart**

By plotting a mean items sold for each of the five promotions which would present what are the strongest promotions historicaly and if or not they are clear winners. If there is a single promotion that dominantes across stores, it would become either a strong predictor or flag that this data is not spread out enough for the model to learn meaningful distinctions. 

- **items_sold by location_type - Box plot**

We will make a side-by-side boxplot across urbn, semi-urban and rural stores to conclude if location influences sales and spread strongly. If it does show sales distributions differ significantly, it would be proof of building stratifed models per location type as recommended above. 

- **Correlation Heatmap - Numerical features**

Computing correlations between competition_density, is_weekend, is_festival, and items_sold. High correlations with the target confirm useful festures, if the high correlations are between feature suggest potential multicollinearity which wuld destabilise linear models and require dropping or combining some of those features. 

- **items_sold over time - Line chart**

Plotting monthly sales averages across the full dat range and looking for seasonal trends, spikes around fesivals, or a general upward?downward drift. Any obvious seasonality would justify adding month and is_festival as features and would also confirm that a temporal train-test split is necessary. 

# Promotion Balance

**The problem:** If the 80% records have no promotion, the model sees very few examples of promotion applied transactions during training, which would mkae the model learn "no promotion" sales pattern better but struggle with accurately predicting the effect of any promotion type which is mainly what the business needs. 

Steps to address it:
- Stratified sampling - This guarantees that each promotion type is represented in the correct proportions across both sets, preventing less frequent promotions from being entirely omitted from the training phase. This ensures your train/test split maintains the integrity of your data.

- Oversample minority classes — use techniques like SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic examples of underrepresented promotion types, giving the model more signal to learn from.

- Separate baseline model - We should train one model specifically on no promotion transactions to capture the baseline sales level as well as a second model on promotion transactions to isolate the uplift effect.

- Flag the imbalance in evaluation - We report the model performance per promotion type not just overall, model that would score well globally may still perform poorly for rare promotions, which would be hidden by aggregate metrics like overall RMSE. 


# B3. Model evaluation and deployment
**a. Train-test split and evaluation**

Train-test split setup: With 3 years of monthly data across 50 stores, it has a clear time order and must be split chronologically.

Recommended approach would be walk-forward validation, where training set would be Month 1 to Month 30, and test set Month 31 to Month 36.

We can also use a more advanced version rolling walk forward validation where we train om 1-13 months and test on month 14 and so so on. 

Why random split is innapropriate:

A random split is inappropriate for time-ordered datasets because it allows "future" observations to enter the training set while "past" observations are held out for testing. This creates a scenario where the model effectively learns from information that would be unavailable at the time of a real-world prediction. This is known as data leakage, grants the model advantage, resulting in over-optimistic performance metrics that cannot be replicated in production. To ensure the model’s validity, a chronological (time-based) split must be used instead.

Evaluation Metrics:
- RMSE
In this business context, a large prediction error (e.g. recommending a promotion that massively undersells) is more costly than a small one.
- MAE
This is the most interpretable metric for stakeholders. If MAE = 40, it means the model is off by about 40 items on average. Less sensitive to outliers than RMSE.
- R squared
Useful for communicating overall model quality to a non-technical audience, though it should always be reported alongside RMSE and MAE rather than in isolation.

**b. Investigating Different Recommendations via Feature Importance**
The model recommends Loyalty Points Bonus in December and Flat Discount in March for the same store. This difference is driven by the input features changing between months even though it is the same store. 

**How to investigate:**
- Step 1: Checking what features changed between the 2 months.
- Step 2: Use feature importance scores like trained Random Forest, extract global feature importances, if 'is_festival' and 'month' rank highly, this owuld confirm seasonal context is the primary driver for differing recommendations. 
- Step 3: Using SHAP values for store level explanation as it can show exaclty how much each feature pushed the prediction up or down for store 12 specifically in each month. 

**How to communicate to the marketing team:**
We should present a SHAP waterfall chart showing that in December the festival flag and high footfall month pushed the model towards Loyalty Points, which retains customers acquired during peak season, In March, with no festival and lower footfall, the model favours Flat Discount to drive volume in a quieter period.

**c. End-to-end Deployment process:**

- Step 1: Once the training is complete, we save the entire scikit-learn pipeline which consists of preprocessor and model as a single file using joblib. Saving ensures all data goes through the same pipeline removing the risk of inconsistent transformations. 
- Step 2: Preparing new monthly data for all 50 stores with the following columns - store_id, store_size, location_type, competition_density and month, is_weekend(proportion for upcoming month), is_festival as well as One row per store per candidate promotion 5 rows per store making it 250 rows total. The saved pipeline is then loaded and run on this input to generate a predicted items_sold for each store is selected as the recommendation.
- Step 3: Monitoring for Model Degradation
The model should not be assumed to remain accurate indefinitely, we need to put in following checks in place:
    * Prediction drift monitoring where each month we compare distribution of predicted items_sold to the historical prediction distribution and a significant shift signals that input data patterns have changed like a new competitor entering the market.
    * Actuals vs prediction tracking where once actual sales data arrives at the end of each month, we compute RMSE and MAE for this month's recommendations and plot these metrics on a dashboard so that if a steady increase over 2-3 months is a signal that retraining is needed. 
    * Trigger based retraining is when we set a threshold like rolling MAE exceeds a defined limit, automatically flag the model for retraining on the most recent 12 months of data
    * Feature distribution checks where we monitor features like competition_density and is_festival for unexpected values or missing data each month as missing or corrupted inputs can silently degrade predictions without triggering a metric alert. 
