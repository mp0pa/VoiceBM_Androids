import os
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Avoid pandas futurewarning messages about downcasting when replacing categorical values with numeric codes
pd.set_option('future.no_silent_downcasting', True)

def logistic_regression(input_csv, task, condition, normalize_demographics=False, output_csv=None):

    # Load the dataset
    df = pd.read_csv(input_csv)

    # Filter the dataset based on the specified task and condition
    if task is not None:
        df = df[df['Task'] == task]

    if condition is not None:
        df = df[df['is_denoised'] == condition]

    if df.empty:
        print("No data available for the specified task and condition.")
        return
        
    # Avoid SettingWithCopyWarning down the line
    df = df.copy()

    # Encode categorical variables 
    df['Sex'] = df['Sex'].replace({'M': 0, 'F': 1}).infer_objects(copy=False)
    df['Task'] = df['Task'].replace({'RT': 0, 'IT': 1}).infer_objects(copy=False)
    df['Speaker_grp'] = df['Speaker_grp'].replace({'C': 0, 'P': 1}).infer_objects(copy=False)
    df['Age_grp'] = df['Age_grp'].replace({'<47': 0, '>=47': 1}).infer_objects(copy=False)
    
    # Drop missing values to prevent errors
    df = df.dropna(subset=['Mean_F0', 'F0_SD', 'Age', 'Sex', 'Age_grp'])
    
    y = df['Speaker_grp']
    X = df.drop(columns=['Speaker_grp'])
    
    # Split the data into training and testing sets before scaling/normalization to prevent data leakage
    X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if normalize_demographics:
        # Calculate baseline stats for healthy (Speaker_grp == 0) subjects using only the training set
        train_healthy = X_train_full[y_train == 0]
        
        baseline_stats = train_healthy.groupby(['Sex', 'Age_grp'])[['Mean_F0', 'F0_SD']].agg(['mean', 'std']).reset_index()
        baseline_stats.columns = ['Sex', 'Age_grp', 'Mean_F0_mean', 'Mean_F0_std', 'F0_SD_mean', 'F0_SD_std']
        
        # Fallback healthy overall means in case a demographic combo is missing from training data
        overall_m_f0_mean = train_healthy['Mean_F0'].mean()
        overall_m_f0_std = train_healthy['Mean_F0'].std()
        if pd.isna(overall_m_f0_std) or overall_m_f0_std == 0: overall_m_f0_std = 1
        
        overall_sd_f0_mean = train_healthy['F0_SD'].mean()
        overall_sd_f0_std = train_healthy['F0_SD'].std()
        if pd.isna(overall_sd_f0_std) or overall_sd_f0_std == 0: overall_sd_f0_std = 1
        
        def apply_normalization(X_data):
            X_out = X_data.copy()
            for idx, row in X_out.iterrows():
                base = baseline_stats[(baseline_stats['Sex'] == row['Sex']) & (baseline_stats['Age_grp'] == row['Age_grp'])]
                
                m_f0_mean, m_f0_std = overall_m_f0_mean, overall_m_f0_std
                sd_f0_mean, sd_f0_std = overall_sd_f0_mean, overall_sd_f0_std
                
                if not base.empty:
                    if not pd.isna(base['Mean_F0_mean'].values[0]): m_f0_mean = base['Mean_F0_mean'].values[0]
                    if not pd.isna(base['Mean_F0_std'].values[0]): m_f0_std = base['Mean_F0_std'].values[0]
                    if not pd.isna(base['F0_SD_mean'].values[0]): sd_f0_mean = base['F0_SD_mean'].values[0]
                    if not pd.isna(base['F0_SD_std'].values[0]): sd_f0_std = base['F0_SD_std'].values[0]
                    
                if m_f0_std == 0 or pd.isna(m_f0_std): m_f0_std = 1
                if sd_f0_std == 0 or pd.isna(sd_f0_std): sd_f0_std = 1
                
                X_out.at[idx, 'Mean_F0'] = (row['Mean_F0'] - m_f0_mean) / m_f0_std
                X_out.at[idx, 'F0_SD'] = (row['F0_SD'] - sd_f0_mean) / sd_f0_std
                
            return X_out[['Mean_F0', 'F0_SD']]
            
        X_train = apply_normalization(X_train_full)
        X_test = apply_normalization(X_test_full)
    else:
        features = ['Age', 'Sex', 'Age_grp', 'Mean_F0', 'F0_SD']
        X_train = X_train_full[features].copy()
        X_test = X_test_full[features].copy()
        
        # Scale the 'Age' feature properly fitting on train, transforming test
        scaler = StandardScaler()
        X_train['Age'] = scaler.fit_transform(X_train[['Age']])
        X_test['Age'] = scaler.transform(X_test[['Age']])
    
    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")
    
    results = {
        'Task': task if task else 'All',
        'Condition': condition if condition is not None else 'All',
        'Normalized': normalize_demographics,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1
    }

    print("Feature Coefficients (Log-Odds) and Odds Ratios:")
    for feature, coef in zip(X_train.columns, model.coef_[0]):
        odds_ratio = np.exp(coef)
        print(f"{feature}: {coef:.4f} | Odds Ratio: {odds_ratio:.4f}")
        results[f'{feature}_Coef'] = coef
        results[f'{feature}_OddsRatio'] = odds_ratio
        
    print("\nPermutation Feature Importance (Impact on Accuracy):")
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    for feature, imp_mean, imp_std in zip(X_test.columns, result.importances_mean, result.importances_std):
        print(f"{feature}: {imp_mean:.4f} +/- {imp_std:.4f}")
        results[f'{feature}_PermImp_Mean'] = imp_mean
        results[f'{feature}_PermImp_Std'] = imp_std
        
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save to CSV if an output path is provided
    if output_csv:
        results_df = pd.DataFrame([results])
        if os.path.isfile(output_csv) and os.path.getsize(output_csv) > 0:
            try:
                existing_df = pd.read_csv(output_csv)
                results_df = pd.concat([existing_df, results_df], ignore_index=True)
            except pd.errors.EmptyDataError:
                pass
        results_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform logistic regression on the input dataset.")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file containing the dataset")
    parser.add_argument("--task", type=str, help="The task from which to take F0 data (e.g., RT or IT)")
    parser.add_argument("--condition", type=int, choices=[0, 1], help="The condition from which to take F0 data (0 for noisy, 1 for denoised)")
    parser.add_argument("--normalize_demographics", action="store_true", help="Standardize F0 features by Sex and Age_grp healthy baselines.")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to save the evaluation metrics as a CSV file")
    
    args = parser.parse_args()
    logistic_regression(args.input_csv, args.task, args.condition, args.normalize_demographics, args.output_csv)