import pandas as pd
import numpy as np
import statsmodels.api as sm
from tableone import TableOne
import streamlit as st

def process_data(file):
    # Load the dataset
    train_total = pd.read_excel(file)
    
    # Handle missing values
    train_total = train_total.fillna(train_total.median(numeric_only=True))
    train_total = train_total.fillna(0)

    # Replace categorical columns with numeric values
    aa = train_total['Name'].unique().tolist()
    number = 0
    for i in aa:
        train_total['Name'] = train_total['Name'].replace(i, number)
        number += 1

    aa = train_total['Burn type'].unique().tolist()
    number = 0
    for i in aa:
        train_total['Burn type'] = train_total['Burn type'].replace(i, number)
        number += 1

    aa = train_total['STSG'].unique().tolist()
    number = 0
    for i in aa:
        train_total['STSG'] = train_total['STSG'].replace(i, number)
        number += 1

    # Drop specific columns
    columns_to_drop = [
        "total extension", "total extension_2", "tripod_2", "tripod", "heavy score_2",
        "heavy score", "TBSA(4')", "affected side", "artificial dermis", "pegboardassembly",
        "Jesen taylor", "grasp power", "Pre-evaluation", "STSG OP", "Onset", "입원일자",
        "pegboard"
    ]
    train_total = train_total.drop(labels=columns_to_drop, axis=1)

    # Split the data
    ex_test = train_total[158:]
    test = train_total[100:158]
    train = train_total[0:100]

    # Display column information
    st.write(train.columns)
    st.write(train.info())

    # Prepare data for analysis
    columns_for_analysis = [
        "ID", "Name", "sex", "Age", "Burn type", "TBSA(Total)", "TBSA(2')", "TBSA(3')",
        'STSG', 'ROM', 'total flexion', 'writing Score', 'card score', 'small score',
        'feed score', 'checker score', 'light score', 'grip strength', 'tip', 'key',
        'pegboard affected (Rt)', 'pegboard both'
    ]
    train_data = train[columns_for_analysis]
    test_data = test[columns_for_analysis]
    ex_test_data = ex_test[columns_for_analysis]

    # Output the TableOne summaries
    generate_table(train, 'pegboard both_2')
    generate_table(train, 'key_2')
    generate_table(train, 'tip_2')
    generate_table(train, 'grip strength_2')
    generate_table(train, 'light score_2')
    generate_table(train, 'checker score_2')
    generate_table(train, 'feed score_2')
    generate_table(train, 'smmall score_2')
    generate_table(train, 'card score_2')
    generate_table(train, 'writing Score_2')
    generate_table(train, 'total flexion_2')
    generate_table(train, 'ROM_2')

    #classification
    y0=train_total['ROM_2'].iloc[0:100].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y0_internal = train_total['ROM_2'].iloc[100:158].values
    y0_external = train_total['ROM_2'].iloc[158:].values

    y1=train_total['total flexion_2'].iloc[0:100].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y1_internal = train_total['total flexion_2'].iloc[100:158].values
    y1_external = train_total['total flexion_2'].iloc[158:].values
    #y2=train_total['total extension_2'].iloc[0:80].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    #y2_internal = train_total['total extension_2'].iloc[80:158].values
    #y2_external = train_total['total extension_2'].iloc[158:].values
    y3=train_total['writing Score_2'].iloc[0:100].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y3_internal = train_total['writing Score_2'].iloc[100:158].values
    y3_external = train_total['writing Score_2'].iloc[158:].values
    y4=train_total['card score_2'].iloc[0:100].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y4_internal = train_total['card score_2'].iloc[100:158].values
    y4_external = train_total['card score_2'].iloc[158:].values
    y5=train_total['smmall score_2'].iloc[0:100].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y5_internal = train_total['smmall score_2'].iloc[100:158].values
    y5_external = train_total['smmall score_2'].iloc[158:].values
    y6=train_total['feed score_2'].iloc[0:100].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y6_internal = train_total['feed score_2'].iloc[100:158].values
    y6_external = train_total['feed score_2'].iloc[158:].values
    y7=train_total['checker score_2'].iloc[0:100].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y7_internal = train_total['checker score_2'].iloc[100:158].values
    y7_external = train_total['checker score_2'].iloc[158:].values
    y8=train_total['light score_2'].iloc[0:100].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y8_internal = train_total['light score_2'].iloc[100:158].values
    y8_external = train_total['light score_2'].iloc[158:].values
    #y9=train_total['heavy score_2'].iloc[0:80].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    #y9_internal = train_total['heavy score_2'].iloc[80:158].values
    #y9_external = train_total['heavy score_2'].iloc[158:].values
    y10=train_total['grip strength_2'].iloc[0:100].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y10_internal = train_total['grip strength_2'].iloc[100:158].values
    y10_external = train_total['grip strength_2'].iloc[158:].values
    y11=train_total['tip_2'].iloc[0:100].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y11_internal = train_total['tip_2'].iloc[100:158].values
    y11_external = train_total['tip_2'].iloc[158:].values
    y12=train_total['key_2'].iloc[0:100].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y12_internal = train_total['key_2'].iloc[100:158].values
    y12_external = train_total['key_2'].iloc[158:].values
    #y13=train_total['tripod_2'].iloc[0:80].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    #y13_internal = train_total['tripod_2'].iloc[80:158].values
    #y13_external = train_total['tripod_2'].iloc[158:].values
    y14=train_total['pegboard affected(Rt)_2'].iloc[0:100].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y14_internal = train_total['pegboard affected(Rt)_2'].iloc[100:158].values
    y14_external = train_total['pegboard affected(Rt)_2'].iloc[158:].values
    y15=train_total['pegboard both_2'].iloc[0:100].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y15_internal = train_total['pegboard both_2'].iloc[100:158].values
    y15_external = train_total['pegboard both_2'].iloc[158:].values
    y16=train_total['pegboardassembly_2'].iloc[0:100].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y16_internal = train_total['pegboardassembly_2'].iloc[100:158].values
    y16_external = train_total['pegboardassembly_2'].iloc[158:].values
    
    train_data=train_data.astype('float')
    test_data=test_data.astype('float')
    ex_test=ex_test.astype('float')
    ex_test['Name']

    y_variables = [y0, y1, y3, y4, y5, y6, y7, y8, y10, y11, y12, y14, y15, y16]
    y_variable_names = ['y0', 'y1', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y10', 'y11', 'y12', 'y14', 'y15', 'y16']


def generate_table(data, column_name):
    # Select relevant columns for the TableOne summary
    selected_columns = [
        "ID", "Name", "sex", "Age", "Burn type", "TBSA(Total)", "TBSA(2')", "TBSA(3')",
        'STSG', 'ROM', 'total flexion', 'writing Score', 'card score', 'small score',
        'feed score', 'checker score', 'light score', 'grip strength', 'tip', 'key',
        'pegboard affected (Rt)', 'pegboard both'
    ]
    data_subset = data[selected_columns]

    # Generate the TableOne summary
    table = TableOne(data_subset, groupby=column_name, pval=True)
    
    # Output the table to a CSV and display it
    table_csv = table.to_csv(index=True)
    st.download_button(label=f"Download TableOne for {column_name}", data=table_csv, file_name=f'train_tableone_{column_name}.csv')

import warnings
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             explained_variance_score, brier_score_loss)
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, HistGradientBoostingRegressor)
from sklearn.svm import SVR
import shap
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# 모델 정의
models = {
    'AdaBoost': AdaBoostRegressor(),
    'ExtraTrees': ExtraTreesRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'RandomForest': RandomForestRegressor(),
    'HistGradientBoosting': HistGradientBoostingRegressor(),
    'SVM': SVR()
}

# Hyperparameter grids for each model
param_grid = {
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'HistGradientBoosting': {
        'max_iter': [100, 200],
        'max_depth': [None, 10, 20],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'GradientBoosting': {
        'n_estimators': [100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    },
    'AdaBoost': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    'ExtraTrees': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.2, 0.5],
        'kernel': ['linear', 'rbf']
    }
}

# Nested Cross Validation 설정
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 모델 결과 저장 리스트
model_results = {}  # 각 y 변수와 모델 결과 저장

# ML 학습 함수 정의
def run_machine_learning(X, X_test, y_variables, y_variable_names):
    model_results = {y_name: {} for y_name in y_variable_names}  # 각 y 변수와 모델 결과 저장
    
    # X 데이터 전처리
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train_scaled = sc_X.fit_transform(X)
    X_test_scaled = sc_X.transform(X_test)

    # 분석 수행
    for y_idx, y in enumerate(y_variables):
        y_name = y_variable_names[y_idx]
        print(f"\nProcessing output variable {y_name}...")

        metrics_all = []  # 성능 지표를 저장할 리스트

        for model_name, model in models.items():
            print(f"\n  Running model: {model_name}...")
            start_model = time.time()

            # 각 모델에 맞는 하이퍼파라미터 그리드 사용
            grid = param_grid.get(model_name, {})

            # Grid search 설정
            grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=inner_cv, scoring='neg_mean_squared_error')

            # Nested CV 수행
            nested_scores = []
            for train_idx, test_idx in outer_cv.split(X_train_scaled, y):
                X_train_outer, X_test_outer = X_train_scaled[train_idx], X_train_scaled[test_idx]
                y_train_outer, y_test_outer = y[train_idx], y[test_idx]

                # 내부 CV로 최적 하이퍼파라미터 찾기
                grid_search.fit(X_train_outer, y_train_outer)
                best_model = grid_search.best_estimator_

                # 외부 검증 데이터로 평가
                y_pred_outer = best_model.predict(X_test_outer)
                nested_scores.append(r2_score(y_test_outer, y_pred_outer))

            print(f"    Nested CV Mean R2 Score: {np.mean(nested_scores):.4f}")
            if grid_search.best_params_:
                print(f"    Best Parameters: {grid_search.best_params_}")

            # 최적 모델로 전체 학습 데이터 학습
            best_model.fit(X_train_scaled, y)
            y_pred_train = best_model.predict(X_train_scaled)

            # 성능 지표 계산
            metrics = {
                'R2 Score': r2_score(y, y_pred_train),
                'Mean Absolute Error': mean_absolute_error(y, y_pred_train),
                'Mean Squared Error': mean_squared_error(y, y_pred_train),
                'Explained Variance Score': explained_variance_score(y, y_pred_train)
            }
            metrics_all.append(metrics)

            elapsed_model_time = time.time() - start_model
            print(f"    Model {model_name} completed in {elapsed_model_time // 60:.0f} minutes {elapsed_model_time % 60:.0f} seconds.")

            # 결과 저장
            model_results[y_name][model_name] = {
                'best_model': best_model,
                'metrics': metrics,
                'nested_cv_scores': nested_scores
            }
    
    return model_results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from sklearn.metrics import roc_auc_score
import seaborn as sns
import lime.lime_tabular
from pathlib import Path
from sklearn.svm import SVR

# 분석 실행
def analysis():
    # 데이터 로드
    train_total['time'] = (pd.to_datetime(df['Post evaluation']) - pd.to_datetime(df['Pre-evaluation'])).dt.days

    # 각 변수의 변화량 계산
    change_columns = ['total flexion', 'total extension', 'writing Score', 'card score',
                      'small score', 'feed score', 'checker score', 'light score',
                      'heavy score', 'grip strength', 'tip', 'key',
                      'pegboard affected (Rt)', 'pegboard both', 'pegboardassembly']

    for col in change_columns:
        df[f'change_{col}'] = abs(df[col] - df[f'{col}_2'])

    # 중앙값 계산
    medians = {col: df[f'change_{col}'].median() for col in change_columns}

    # 사건 발생 여부 정의
    for col, median in medians.items():
        df[f'event_{col}'] = (df[f'change_{col}'] >= median).astype(int)

    # Cox 모델 적합
    event_vars = [f'event_{col}' for col in change_columns]
    for event in event_vars:
        cph = CoxPHFitter()
        cph.fit(df[['time', event, 'Age', 'sex']], duration_col='time', event_col=event)

        plt.figure()
        cph.plot()
        plt.title(f'Hazard Ratios for {event}')
        plt.show()
        plt.clf()  # 그래프 초기화

    # 저장 경로 설정
    output_path = Path("./model_results")
    output_path.mkdir(parents=True, exist_ok=True)

    def save_plot(fig, filename):
        filepath = output_path / filename
        fig.savefig(filepath)
        plt.close(fig)
        print(f"Saved plot to {filepath}")

    # Bland-Altman plot 함수
    def bland_altman_plot(y_true, y_pred, title, filename):
        mean = (y_true + y_pred) / 2
        diff = y_true - y_pred
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        plt.figure(figsize=(6, 6))
        plt.scatter(mean, diff, alpha=0.5, color="darkorange")
        plt.axhline(mean_diff, color='red', linestyle='--', label='Mean Difference')
        plt.axhline(mean_diff + 1.96 * std_diff, color='blue', linestyle='--', label='Upper Limit')
        plt.axhline(mean_diff - 1.96 * std_diff, color='blue', linestyle='--', label='Lower Limit')
        plt.xlabel('Mean of True and Predicted')
        plt.ylabel('Difference')
        plt.title(title)
        plt.legend()
        plt.show()
        save_plot(plt.gcf(), filename)
    import shap
    import matplotlib.pyplot as plt
    import os

    # SHAP 분석 수행
    # Streamlit에서 사용자로부터 SHAP 결과 저장 경로 입력받기
    shap_results_dir = st.text_input('SHAP 결과를 저장할 디렉토리 경로를 입력하세요')
    # 입력된 경로가 유효한 디렉토리인지 확인하고, 없다면 생성
    os.makedirs(shap_results_dir, exist_ok=True)

    # 각 y 변수에 대해 SHAP 분석
    for y_idx, y in enumerate(y_variables):
        y_name = y_variable_names[y_idx]
        print(f"\nPerforming SHAP analysis for {y_name}...")

        for model_name, result in model_results[y_name].items():
            best_model = result['best_model']

            # 모델에 따라 explainer 선택
            if isinstance(best_model, (RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor)):
                explainer = shap.TreeExplainer(best_model)
            else:
                # AdaBoostRegressor는 KernelExplainer 사용
                explainer = shap.KernelExplainer(best_model.predict, X_train_scaled)

            # SHAP 값 계산
            shap_values = explainer.shap_values(X_train_scaled)

            # Summary plot 저장
            summary_plot_path = os.path.join(shap_results_dir, f"{y_name}_{model_name}_summary.png")
            shap.summary_plot(shap_values, X_train_scaled)
            plt.savefig(summary_plot_path)
            plt.close()

            # Dependency plot 저장
            for feature_idx in range(X_train_scaled.shape[1]):  # 모든 특성에 대해 반복
                dependency_plot_path = os.path.join(shap_results_dir, f"{y_name}_{model_name}_dependency_{feature_idx}.png")
                shap.dependence_plot(feature_idx, shap_values, X_train_scaled)  # 각 특성에 대해 dependency plot 생성
                plt.savefig(dependency_plot_path)
                plt.close()

            print(f"  SHAP plots for {model_name} saved for {y_name}.")

    # 모델 예측 분석
    for y_idx, y in enumerate(y_variables):
        y_name = y_variable_names[y_idx]
        print(f"\nPost-processing output variable {y_name}...")
        try:
            if y_name in model_results:
                best_model_dict = model_results[y_name]
            else:
                print(f"{y_name}: No results found for this variable.")
                continue

            for model_name, result in best_model_dict.items():
                best_model = result.get('best_model')
                if best_model is None:
                    print(f"{y_name}: No best model found for {model_name}.")
                    continue

                print(f"Processing with {model_name} for {y_name}...")

                if isinstance(best_model, SVR):
                    print(f"{y_name}: SVR 모델은 지원되지 않아 건너뜁니다.")
                    continue

                y_pred = best_model.predict(X_test_scaled)

                # C-index 계산
                try:
                    if hasattr(best_model, "predict_proba"):
                        y_pred_proba = best_model.predict_proba(X_test_scaled)
                        if len(np.unique(y)) > 2:
                            c_index = roc_auc_score(y, y_pred_proba, multi_class="ovr")
                        else:
                            c_index = roc_auc_score(y, y_pred_proba[:, 1])
                    else:
                        print(f"{y_name}: Model does not support probability prediction.")
                        c_index = None

                    if c_index is not None:
                        plt.figure(figsize=(6, 6))
                        plt.plot([0, 1], [0, 1], 'k--')
                        plt.title(f"C-index: {c_index:.4f}")
                        plt.show()
                        save_plot(plt.gcf(), f"{y_name}_cindex_plot.png")
                except Exception as e:
                    print(f"{y_name}: Error calculating C-index: {e}")

                # Pearson correlation map
                min_length = min(len(y), len(y_pred))
                y, y_pred = y[:min_length], y_pred[:min_length]
                plt.figure(figsize=(6, 6))
                correlation_matrix = np.corrcoef(y.flatten(), y_pred.flatten())
                sns.heatmap(correlation_matrix, annot=True, cmap="Spectral", cbar=True, square=True)
                plt.title(f"Pearson Correlation Map: {y_name}")
                plt.show()
                save_plot(plt.gcf(), f"{y_name}_pearson_correlation.png")

                # Bland-Altman plot
                bland_altman_plot(y, y_pred, f"Bland-Altman Plot: {y_name}", f"{y_name}_bland_altman_plot.png")

                # 예측 vs 실제값 차이
                plt.figure(figsize=(6, 6))
                plt.scatter(y, y_pred, alpha=0.5, color="teal")
                plt.xlabel("True Values")
                plt.ylabel("Predictions")
                plt.title(f"{y_name} - True vs Predicted")
                save_plot(plt.gcf(), f"{y_name}_true_vs_predicted.png")

                # 그룹 간 차이 시각화
                sns.boxplot(x=y > y.mean(), y=y_pred, palette="Set2")
                plt.title(f"{y_name} Group Difference")
                save_plot(plt.gcf(), f"{y_name}_group_difference.png")

                # LIME 분석
                explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train_scaled, mode="regression")
                explanation = explainer_lime.explain_instance(X_test_scaled[0], best_model.predict)
                explanation.save_to_file(output_path / f"{y_name}_lime_instance_0.html")

                # Heatmap 저장
                for heatmap_type, data in [
                    ("X_y", np.corrcoef(X_test_scaled.T, y.flatten())),
                    ("X_X", np.corrcoef(X_test_scaled.T)),
                    ("y_y", np.corrcoef(y.flatten().reshape(-1, 1)))
                ]:
                    try:
                        plt.figure(figsize=(6, 6))
                        sns.heatmap(data, annot=True, cmap="Spectral", cbar=True, square=True)
                        plt.title(f"{y_name} Heatmap ({heatmap_type})")
                        save_plot(plt.gcf(), f"{y_name}_heatmap_{heatmap_type}.png")
                    except Exception as e:
                        print(f"{y_name}: Error creating heatmap ({heatmap_type}): {e}")
        # Handle any additional exceptions for the entire block
        except Exception as e:
            print(f"Error processing {y_name}: {e}")
    print("Analysis complete!")

# Streamlit 앱 설정
def main():
    st.title("Data Processing and TableOne Generator")

    # File uploader widget
    uploaded_file = st.file_uploader("Upload your dataset", type=["xlsx"])

    if uploaded_file is not None:
        process_data(uploaded_file)

    st.title("Machine Learning Model Training")

    X = train_total.drop(columns=["target_variable"])  # 입력 변수
    X_test = X  # 예시로 동일한 X로 설정 (사용자가 테스트 데이터를 업로드할 수도 있음)
    y_variables = [df[col] for col in df.columns if col != "target_variable"]
    y_variable_names = [col for col in df.columns if col != "target_variable"]

    if st.button("Start Machine Learning"):
        model_results = run_machine_learning(X, X_test, y_variables, y_variable_names)

        # 모델 결과 출력
        for y_name, results in model_results.items():
            st.subheader(f"Results for {y_name}")
            for model_name, result in results.items():
                st.write(f"Model: {model_name}")
                st.write(f"Metrics: {result['metrics']}")
                st.write(f"Nested CV Mean R2 Score: {np.mean(result['nested_cv_scores']):.4f}")
    
    if st.button("Start Machine Learning"):
        st.write(f"SHAP 결과가 저장될 경로: {shap_results_dir}")
        analysis()


if __name__ == "__main__":
    main()
