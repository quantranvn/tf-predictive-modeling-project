import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

FIG_DIR = Path("figures_test_2")
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Load the data
commits_df = pd.read_csv("preprocessed_tensorflow_commits.csv")
issues_df = pd.read_csv("preprocessed_tensorflow_issues.csv")

# Assuming 'contributors' column is in the commits_df
# If not, you might need to do additional data preparation to include it

# Convert the dates from string to datetime
commits_df['date'] = pd.to_datetime(commits_df['date'])
issues_df['created_at'] = pd.to_datetime(issues_df['created_at'])
issues_df['closed_at'] = pd.to_datetime(issues_df['closed_at'], errors='coerce')

# Filter closed issues and calculate resolution time in days
closed_issues_df = issues_df[issues_df['state'] == 'closed'].copy()
closed_issues_df['resolution_time'] = (closed_issues_df['closed_at'] - closed_issues_df['created_at']).dt.total_seconds() / (24 * 60 * 60)

# Descriptive statistics
## Commits per month
commits_df['date'] = pd.to_datetime(commits_df['date'])  # Assuming 'date' needs to be converted as well
commits_df['Month'] = commits_df['date'].dt.to_period('M')
commits_df_filtered = commits_df[commits_df['Month'] >= '2018-10']
commits_per_month_filtered = commits_df_filtered.groupby('Month').size()
#commits_per_month = commits_df['date'].dt.to_period('M').value_counts().sort_index()
#print("Commits per month:\n", commits_per_month)

## Commits per contributor
commits_per_contributor = commits_df['contributor'].value_counts()
print("Commits per contributor:\n", commits_per_contributor)

## Issue resolution time
average_resolution_time = closed_issues_df['resolution_time'].mean()
median_resolution_time = closed_issues_df['resolution_time'].median()
print(f"Average issue resolution time (days): {average_resolution_time}")
print(f"Median issue resolution time (days): {median_resolution_time}")

## Label analysis
# Assuming labels are stored in a stringified list format
issues_df['labels'] = issues_df['labels'].apply(lambda x: eval(x) if pd.notnull(x) else [])
all_labels = [label for sublist in issues_df['labels'].tolist() for label in sublist]
labels_df = pd.DataFrame(all_labels, columns=['label'])
labels_count = labels_df['label'].value_counts()
print("Labels count:\n", labels_count)

# Visualization

## Commits per month
#commits_per_month.sort_index().plot(kind='line', figsize=(10, 6), marker='o')
commits_per_month_filtered.sort_index().plot(kind='line', figsize=(10, 6), marker='o')
#plt.title('TensorFlow Commits Per Month')
plt.xlabel('Month')
plt.ylabel('Number of Commits')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')
plt.savefig(FIG_DIR / 'Figure_3.svg', format='svg')
plt.close()

## Issue resolution time for closed issues
# Plotting the histogram with a logarithmic scale
n, bins, patches = plt.hist(closed_issues_df['resolution_time'], bins=50, log=True, linestyle='--')
plt.ylabel('Number of Issues (log scale)')
plt.xlabel('Resolution Time (days)')
#plt.title('Histogram of Issue Resolution Times')
plt.grid(axis='y', linestyle='--')
plt.savefig(FIG_DIR / 'Figure_4.svg', format='svg')
plt.close()

## Commits per contributor
commits_per_contributor.head(10).iloc[::-1].plot(kind='barh')
#plt.title('Top 10 Contributors by Number of Commits')
plt.ylabel('Contributor')
plt.xlabel('Number of Commits')
#plt.xticks(rotation=90)
plt.grid(axis='x', linestyle='--')
plt.tight_layout()
plt.savefig(FIG_DIR / 'Figure_5.svg', format='svg')
plt.close()

## Visualizing the contributions of the top 10 contributors without bot
mask_bot = commits_per_contributor.index.str.contains("bot", case=False, na=False) | \
           commits_per_contributor.index.str.contains("gardener", case=False, na=False) | \
           commits_per_contributor.index.str.contains("Unknown", case=False, na=False)
contributors_no_bot = commits_per_contributor[~mask_bot]
top10_without_bot = contributors_no_bot.head(10).iloc[::-1]
top10_without_bot.head(10).plot(kind='barh')
#plt.title('Top 10 Contributors by Number of Contributions')
plt.xlabel('Contributor')
plt.ylabel('Number of Commits')
#plt.xticks(rotation=90)
plt.grid(axis='x', linestyle='--')
plt.tight_layout()
plt.savefig(FIG_DIR / 'Figure_6.svg', format='svg')
plt.close()

## Label frequency
#plt.figure(figsize=(10, 6))
#labels_count.head(10).plot(kind='bar', color='purple')
#plt.title('Top 10 Labels Frequency')
#plt.xlabel('Label')
#plt.ylabel('Frequency')
#plt.xticks(rotation=45)
#plt.grid(axis='y', linestyle='--')
#plt.show()

# Word clouds for commits and issues including contributor and labels
all_commit_messages = ' '.join(commits_df['message'].astype(str))
all_issue_titles = ' '.join(issues_df['title'].astype(str))

# Generate word clouds
wordcloud_commits = WordCloud(width=800, height=400, background_color='white', min_font_size=10).generate(all_commit_messages)
wordcloud_issues = WordCloud(width=800, height=400, background_color='white', min_font_size=10).generate(all_issue_titles)

# Plot the WordCloud images                        
#plt.figure(figsize=(8, 8), facecolor=None) 
#plt.imshow(wordcloud_commits) 
#plt.axis("off") 
#plt.tight_layout(pad=0)
#plt.show()

#plt.figure(figsize=(8, 8), facecolor=None) 
#plt.imshow(wordcloud_issues) 
#plt.axis("off") 
#plt.tight_layout(pad=0) 
#plt.show()
