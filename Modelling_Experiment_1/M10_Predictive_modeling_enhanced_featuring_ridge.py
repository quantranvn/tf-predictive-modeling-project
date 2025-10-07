# Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import Ridge
from scipy.stats import uniform
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import ast

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
# Load your data
commits_df = pd.read_csv('preprocessed_tensorflow_commits.csv', parse_dates=['date'])
issues_df = pd.read_csv('preprocessed_tensorflow_issues.csv', parse_dates=['created_at', 'closed_at'])

# Text processing for the 'message' column in commits_df
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words])

commits_df['processed_message'] = commits_df['message'].apply(preprocess_text)

# Vectorize the processed text messages
tfidf_vectorizer = TfidfVectorizer(max_features=100)  # Adjust max_features based on your dataset
tfidf_matrix = tfidf_vectorizer.fit_transform(commits_df['processed_message'])

# Categorical encoding for the 'labels' column in issues_df
def evaluate_labels(label_string):
    try:
        return ast.literal_eval(label_string)
    except ValueError:
        return []

issues_df['labels_list'] = issues_df['labels'].apply(evaluate_labels)

# Multi-label binarization
mlb = MultiLabelBinarizer()
labels_matrix = mlb.fit_transform(issues_df['labels_list'])

# Prepare your time-based features as before
commits_df['Month'] = commits_df['date'].dt.to_period('M')
issues_df['Created_Month'] = issues_df['created_at'].dt.to_period('M')
issues_df['Closed_Month'] = issues_df['closed_at'].dt.to_period('M')
monthly_commits = commits_df.groupby('Month').size()
monthly_created_issues = issues_df.groupby('Created_Month').size()
monthly_closed_issues = issues_df[issues_df['state'] == 'closed'].groupby('Closed_Month').size()

# Merge the datasets and create lagged columns for issues
combined_data = pd.DataFrame({
    'Commits': monthly_commits,
    'CreatedIssues': monthly_created_issues.reindex(monthly_commits.index, fill_value=0),
    'ClosedIssues': monthly_closed_issues.reindex(monthly_commits.index, fill_value=0)
}).fillna(0)

combined_data['ClosedIssuesNextMonth'] = combined_data['ClosedIssues'].shift(-1)

# Drop the last row which will have NaN because of the shift
combined_data.dropna(inplace=True)

# Combine TF-IDF and label features with your time-based features
# Note: You might need to adjust indices here to make sure everything aligns
combined_features = pd.concat([
    combined_data.reset_index(drop=True), 
    pd.DataFrame(tfidf_matrix.toarray()), 
    pd.DataFrame(labels_matrix)
], axis=1)

# Ensure all column names are strings
combined_features.columns = combined_features.columns.astype(str)
combined_features = combined_features.dropna(subset=['ClosedIssuesNextMonth'])

# Define features and target variable
X = combined_features.drop(columns='ClosedIssuesNextMonth')
y = combined_features['ClosedIssuesNextMonth']

# Create an imputer object with a median filling strategy
imputer = SimpleImputer(strategy='mean')

# Apply the imputer to our data, replacing each NaN with the median of the column
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y_imputed = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=0.2, random_state=42)

# Define the model
ridge_model = Ridge()

# Set up Randomized Search with a wide range
param_distributions = {'alpha': uniform(0.1, 10000)}
random_search = RandomizedSearchCV(ridge_model, param_distributions, random_state=42, n_iter=100, cv=5, scoring='neg_mean_squared_error')

# Perform Random Search
random_search.fit(X_train, y_train)

# Get the best parameters from the Random Search to use in Grid Search
best_params = random_search.best_params_
print(f"Best parameters from Random Search: {best_params}")

# Define a narrower range around the best alpha found
alpha_range = [best_params['alpha'] * 0.5, best_params['alpha'], best_params['alpha'] * 1.5]

# Set up Grid Search within the narrow range
grid_search = GridSearchCV(ridge_model, {'alpha': alpha_range}, cv=5, scoring='neg_mean_squared_error')

# Perform Grid Search
grid_search.fit(X_train, y_train)

# Get the best alpha value from Grid Search
best_alpha = grid_search.best_params_['alpha']
print(f"Optimal alpha value after Grid Search: {best_alpha}")

# Use the best alpha to train the final model
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train, y_train)

# Make predictions on the test set using the best alpha value
y_pred = final_model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the performance metrics
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Closed Issues Next Month')
plt.ylabel('Predicted Closed Issues Next Month')
plt.title('Actual vs Predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.show()

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals
plt.scatter(y_test, residuals)
plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), colors='red', linestyles='--')
plt.xlabel('Actual Closed Issues Next Month')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Check the distribution of residuals
plt.hist(residuals, bins=20)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.show()

# Perform cross-validation
cv_scores = cross_val_score(final_model, X, y, cv=5, scoring='neg_mean_squared_error')

# Output the cross-validation scores
print(f'CV MSE scores: {-cv_scores}')
print(f'Average CV MSE: {-cv_scores.mean()}')
