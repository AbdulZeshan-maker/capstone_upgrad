# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customer_df = pd.read_csv('/mnt/data/Customer.csv')
prod_cat_df = pd.read_csv('/mnt/data/prod_cat_info.csv')
transactions_df = pd.read_csv('/mnt/data/Transactions.csv')

# Display the first few rows of each dataframe to understand the structure
print("Customer DataFrame:")
display(customer_df.head())

print("Product Category DataFrame:")
display(prod_cat_df.head())

print("Transactions DataFrame:")
display(transactions_df.head())

# Check the basic info and structure of each dataset
print("\nCustomer DataFrame Info:")
customer_df.info()

print("\nProduct Category DataFrame Info:")
prod_cat_df.info()

print("\nTransactions DataFrame Info:")
transactions_df.info()

# Check for missing values
print("\nMissing values in Customer DataFrame:")
print(customer_df.isnull().sum())

print("\nMissing values in Product Category DataFrame:")
print(prod_cat_df.isnull().sum())

print("\nMissing values in Transactions DataFrame:")
print(transactions_df.isnull().sum())

# Check for duplicates in each dataset
print("\nDuplicates in Customer DataFrame:", customer_df.duplicated().sum())
print("Duplicates in Product Category DataFrame:", prod_cat_df.duplicated().sum())
print("Duplicates in Transactions DataFrame:", transactions_df.duplicated().sum())

# Merge the datasets to form a single table for recommendation system analysis
# Assuming 'prod_cat_code' is a linking key between Transactions and Product Category DataFrames
merged_df = pd.merge(transactions_df, prod_cat_df, on='prod_cat_code', how='left')

# Assuming 'customer_Id' is the key to link Transactions and Customer DataFrames
merged_df = pd.merge(merged_df, customer_df, on='customer_Id', how='left')

# Display the first few rows of the merged dataframe
print("\nMerged DataFrame:")
display(merged_df.head())

# Explore the merged data
print("\nMerged DataFrame Info:")
merged_df.info()

# Visualizing some important relationships (optional, can be skipped for the next steps)
plt.figure(figsize=(10,6))
sns.countplot(data=merged_df, x='Gender')
plt.title('Gender Distribution of Customers')
plt.show()

# Save the merged dataframe for future use
merged_df.to_csv('/mnt/data/merged_recommender_data.csv', index=False)

# Import necessary libraries for building the collaborative filtering model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import cross_validate

# Prepare the data for collaborative filtering
# We'll use 'customer_Id' and 'prod_cat_code' for this purpose
ratings_data = merged_df[['customer_Id', 'prod_cat_code']]

# Adding a dummy 'rating' column for recommendation purposes
ratings_data['rating'] = 1  # This can be modified later based on actual customer ratings

# Use Surprise library to build a collaborative filtering model
# Reader helps to define the rating scale, but since we have binary ratings (purchased or not), we can set min=0, max=1
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(ratings_data[['customer_Id', 'prod_cat_code', 'rating']], reader)

# Split the data into training and test sets
trainset, testset = train_test_split(ratings_data, test_size=0.2, random_state=42)

# Using SVD (Singular Value Decomposition) for collaborative filtering
svd_model = SVD()

# Perform 5-fold cross-validation on the SVD model
cross_validate(svd_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train the SVD model on the full training data
train_data = data.build_full_trainset()
svd_model.fit(train_data)

# Make predictions for a specific user (for example, customer_Id = 1)
customer_id = 1
product_id = 2  # example product category code

# Predict the rating for this customer-product pair
prediction = svd_model.predict(customer_id, product_id)
print(f"Predicted rating for customer {customer_id} and product {product_id}: {prediction.est}")

# Get top N product recommendations for a specific customer
def get_top_n_recommendations(model, customer_id, product_ids, n=5):
    predictions = [model.predict(customer_id, pid) for pid in product_ids]
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    return [(pred.iid, pred.est) for pred in recommendations]

# Example usage: Get top 5 recommended products for customer with ID 1
unique_products = merged_df['prod_cat_code'].unique()
top_n_recommendations = get_top_n_recommendations(svd_model, customer_id, unique_products, n=5)

print(f"Top 5 product recommendations for customer {customer_id}: {top_n_recommendations}")

# Import libraries for content-based filtering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# We'll use the product category names to find similar products
tfidf = TfidfVectorizer(stop_words='english')

# Replace missing values in 'prod_cat_name' with empty string
prod_cat_df['prod_cat_name'] = prod_cat_df['prod_cat_name'].fillna('')

# Fit and transform the 'prod_cat_name' column
tfidf_matrix = tfidf.fit_transform(prod_cat_df['prod_cat_name'])

# Compute the cosine similarity matrix between products
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on product similarity
def get_content_based_recommendations(product_id, cosine_sim=cosine_sim, prod_cat_df=prod_cat_df, n=5):
    # Get the index of the product that matches the product_id
    idx = prod_cat_df[prod_cat_df['prod_cat_code'] == product_id].index[0]

    # Get the pairwise similarity scores of all products with the selected product
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the n most similar products
    sim_scores = sim_scores[1:n+1]

    # Get the product indices
    product_indices = [i[0] for i in sim_scores]

    # Return the top n most similar products
    return prod_cat_df['prod_cat_code'].iloc[product_indices]

# Example: Get top 5 products similar to product with 'prod_cat_code' = 2
similar_products = get_content_based_recommendations(2, n=5)
print(f"Top 5 similar products to product 2: {similar_products.tolist()}")

# Import necessary metrics
from sklearn.metrics import precision_score, recall_score, f1_score

# Import necessary metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Example ground truth and predicted data for illustration
# y_true: Actual purchases (1 if product purchased, 0 if not)
# y_pred: Model's prediction (1 if product recommended, 0 if not)
# These should ideally come from your test data and the model's recommendations

# Example data for demonstration (adjust based on actual data)
y_true = [1, 1, 0, 0, 1, 0, 1, 0, 0, 1]  # Actual purchases
y_pred = [1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # Predicted recommendations

# Calculate precision, recall, and F1-score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Display the results
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")

