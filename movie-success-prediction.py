import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt

class MovieSuccessPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0

    def preprocess_data(self, data):
        # Drop missing values
        data = data.dropna()

        # Categorize movies
        data['Category'] = data.apply(self._categorize_movie, axis=1)

        # Select features (removing highly correlated ones to avoid overfitting)
        X = data[['Runtime (Minutes)', 'Rating', 'Votes', 'Metascore']]
        y = data['Category']

        return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    def _categorize_movie(self, row):
        if row['Rating'] > 7.5 or row['Revenue (Millions)'] > 100:
            return 'Hit'
        elif 5.5 <= row['Rating'] <= 7.5 and 30 <= row['Revenue (Millions)'] <= 100:
            return 'Average'
        else:
            return 'Flop'

    def train_models(self, X_train, X_test, y_train, y_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        model_params = {
            'random_forest': (RandomForestClassifier(), {
                'n_estimators': [50, 100],
                'max_depth': [10, 20],
                'min_samples_split': [5, 10],
                'max_features': ['sqrt', 'log2']
            }),
            'adaboost': (AdaBoostClassifier(), {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1]
            })
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, (model, params) in model_params.items():
            grid_search = GridSearchCV(model, params, cv=skf, n_jobs=-1, scoring='accuracy')
            grid_search.fit(X_train_scaled, y_train)

            self.models[name] = grid_search.best_estimator_
            score = grid_search.score(X_test_scaled, y_test)
            print(f"{name} best score: {score:.4f}")

            if score > self.best_score:
                self.best_score = score
                self.best_model = name

        # Voting classifier
        self.models['voting'] = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items() if name != 'voting'],
            voting='soft'
        )
        self.models['voting'].fit(X_train_scaled, y_train)

    def evaluate_models(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)

        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred) * 100,
                'precision': precision_score(y_test, y_pred, average='weighted') * 100,
                'recall': recall_score(y_test, y_pred, average='weighted') * 100,
                'report': classification_report(y_test, y_pred)
            }
        return results
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plots a confusion matrix using Seaborn heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Flop', 'Average', 'Hit'], yticklabels=['Flop', 'Average', 'Hit'])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.show()

    def plot_results(self, results):
        plt.figure(figsize=(10, 6))
        model_names = list(results.keys())
        accuracy_scores = [results[model]['accuracy'] for model in model_names]
        precision_scores = [results[model]['precision'] for model in model_names]
        recall_scores = [results[model]['recall'] for model in model_names]

        x = np.arange(len(model_names))
        width = 0.25

        plt.bar(x - width, accuracy_scores, width, label='Accuracy (%)', color='#4CAF50', edgecolor='black')
        plt.bar(x, precision_scores, width, label='Precision (%)', color='#2196F3', edgecolor='black')
        plt.bar(x + width, recall_scores, width, label='Recall (%)', color='#FF9800', edgecolor='black')

        plt.xlabel('Models', fontweight='bold')
        plt.ylabel('Scores (%)', fontweight='bold')
        plt.title('Model Performance Comparison', fontweight='bold', fontsize=14)
        plt.xticks(x, [name.replace('_', ' ').title() for name in model_names], rotation=45)
        plt.legend()

        for i, metric in enumerate([accuracy_scores, precision_scores, recall_scores]):
            for j, v in enumerate(metric):
                plt.text(j + x[0] + (i-1)*width, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.ylim(0, 110)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()



if __name__ == "__main__":
    from google.colab import files
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    data = pd.read_csv(file_name)

    predictor = MovieSuccessPredictor()
    X_train, X_test, y_train, y_test = predictor.preprocess_data(data)
    predictor.train_models(X_train, X_test, y_train, y_test)

    results = predictor.evaluate_models(X_test, y_test)
    predictor.plot_results(results)
