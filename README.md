Movie Success Prediction Project
This project uses machine learning to predict a movie's success category (Flop, Average, or Hit) based on features like runtime, rating, votes, and Metascore. The project encompasses data preprocessing, model training with hyperparameter tuning, evaluation, and visualization.

📌 Features
  Data Preprocessing and Cleaning: Handles missing values, outliers, and data scaling to ensure data quality.
  Movie Categorization: Classifies movies into "Flop," "Average," and "Hit" categories based on a combination of rating and revenue.
  Model Training: Trains and compares the performance of the following machine learning models:
  Random Forest
  AdaBoost
  Voting Classifier (combining the above)
  Hyperparameter Tuning: Optimizes model performance using GridSearchCV to find the best combination of hyperparameters.
  Evaluation Metrics: Evaluates model performance using key metrics such as accuracy, precision, and recall.
  Visual Comparison: Provides visualizations to compare the performance of different models.
  Confusion Matrix Plotting: Visualizes the performance of the classification model.
  
📂 Dataset
The project expects a CSV file with the following columns:
 Runtime (Minutes)
 Rating
 Votes
 Metascore
 Revenue (Millions)
 
⚙️ Installation and Usage
Clone the repository:
 git clone https://github.com/yourusername/movie-success-predictor.git
 cd movie-success-predictor


