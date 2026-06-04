import os
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Avoid pandas futurewarning messages about downcasting when replacing categorical values with numeric codes
pd.set_option('future.no_silent_downcasting', True)

def logistic_regression(input_csv, task='Both', k_folds=5, output_csv=None, plot=False):

    # Load the dataset
    df = pd.read_csv(input_csv)

    # Filter the dataset to always use raw audio (not denoised)
    df = df[df['is_denoised'] == 0]

    # Filter by task if requested
    if task in ['RT', 'IT']:
        df = df[df['Task'] == task]

    if df.empty:
        print("No raw audio data available in the dataset.")
        return
        
    # Avoid SettingWithCopyWarning down the line
    df = df.copy()

    # Encode categorical variables 
    df['Task'] = df['Task'].replace({'RT': 0, 'IT': 1}).infer_objects(copy=False)
    df['Speaker_grp'] = df['Speaker_grp'].replace({'C': 0, 'P': 1}).infer_objects(copy=False)
    
    # Keep only the requested features, target, and Speaker_ID for grouping
    columns_to_keep = ['Speaker_ID', 'Mean_F0', 'F0_SD', 'Speaker_grp']
    if task == 'Both':
        columns_to_keep.append('Task')
    df = df[[col for col in columns_to_keep if col in df.columns]]

    # Drop missing values to prevent errors
    df = df.dropna()

    # Define features, target variable, and groups
    groups = df['Speaker_ID'].astype(str) + '_' + df['Speaker_grp'].astype(str)
    y = df['Speaker_grp']
    X = df.drop(columns=['Speaker_grp', 'Speaker_ID'])
    
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    uars, precisions, recalls, f1s = [], [], [], []
    total_cm = np.zeros((2, 2), dtype=int)
    
    if plot:
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
    
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Standardize features properly fitting on train, transforming test
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the logistic regression model
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test_scaled)
        
        # Calculate ROC metrics if plotting is enabled
        if plot:
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)
            
        # Evaluate the model
        uars.append(balanced_accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        total_cm += confusion_matrix(y_test, y_pred, labels=[0, 1])
        
    avg_uar = np.mean(uars)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    
    print(f"Results across {k_folds} folds:")
    print(f"UAR (Unweighted Average Recall): {avg_uar:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}\n")
    
    results = {
        'Tasks': task if task != 'Both' else 'Both (RT & IT)',
        'Condition': 'Raw (0)',
        'K_Folds': k_folds,
        'UAR': avg_uar,
        'Precision': avg_precision,
        'Recall': avg_recall,
        'F1_Score': avg_f1
    }
        
    print("\nAggregated Confusion Matrix:")
    print(total_cm)

    if plot:
        # 1. Plot Mean ROC Curve
        plt.figure(figsize=(8, 6))
        for i, tpr in enumerate(tprs):
            plt.plot(mean_fpr, tpr, alpha=0.3, label=f'ROC fold {i+1} (AUC = {aucs[i]:.2f})')
            
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        plt.plot(mean_fpr, mean_tpr, color='b', label=rf'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})', lw=2, alpha=0.8)
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.')
        
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve with Cross-Validation for {task} Task'.format(task=task if task != 'Both' else 'RT & IT'))
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve_cv.png', dpi=300)
        print("\nSaved ROC curve plot to 'roc_curve_cv.png'")
        
        # 2. Plot Confusion Matrix Heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Control (0)', 'Patient (1)'],
                    yticklabels=['Control (0)', 'Patient (1)'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Aggregated Confusion Matrix for {task} Task'.format(task=task if task != 'Both' else 'RT & IT'))
        plt.tight_layout()
        plt.savefig('confusion_matrix_cv.png', dpi=300)
        print("Saved Confusion Matrix plot to 'confusion_matrix_cv.png'")
        
        plt.show()

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
    parser.add_argument("--task", type=str, choices=['RT', 'IT', 'Both'], default='Both', help="Specific task to analyze (RT, IT, or Both). Default is Both.")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to save the evaluation metrics as a CSV file")
    parser.add_argument("--plot", action="store_true", help="Generate and save evaluation plots")
    
    args = parser.parse_args()
    logistic_regression(args.input_csv, args.task, args.k_folds, args.output_csv, args.plot)