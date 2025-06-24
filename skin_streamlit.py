import pandas as pd
import numpy as np
import statsmodels.api as sm
from tableone import TableOne
import streamlit as st
from io import StringIO

def process_data(file):
    # Load the dataset
    train_total = pd.read_excel(file)
    
    # Handle missing values
    train_total = train_total.fillna(train_total.median(numeric_only=True))
    train_total = train_total.fillna(0)

    # Replace categorical columns with numeric values
    aa=train_total['Name'].unique().tolist()
    number=0
    for i in aa:
        train_total['Name']=train_total['Name'].replace(i,number)
        number=number+1

    aa=train_total['Burn type'].unique().tolist()
    number=0
    for i in aa:
        train_total['Burn type']=train_total['Burn type'].replace(i,number)
        number=number+1

    aa=train_total['artificial dermis'].unique().tolist()
    number=0
    for i in aa:
        train_total['artificial dermis']=train_total['artificial dermis'].replace(i,number)
        number=number+1

    aa=train_total['skintest site'].unique().tolist()
    number=0
    for i in aa:
        train_total['skintest site']=train_total['skintest site'].replace(i,number)
        number=number+1

    train_total=train_total.drop(labels="STSG OP",axis=1)
    train_total=train_total.drop(labels="Onset",axis=1)
    train_total=train_total.drop(labels="입원일자",axis=1)
    train_total=train_total.drop(labels="skintest_1",axis=1)

    ex_test=train_total[336:]
    test = train_total[180:336]
    train = train_total[0:180] #서울대 부분 추출
    print(train_total.columns)

    train_data = train[["ID", "Name", "sex", "Age", "affected \nside","Burn type", "TBSA(Total)", "TBSA(2')", "TBSA(3')", "TBSA(4')",'STSG', 'artificial dermis', 'skintest site','skintest_1_Thickness','skintest_1_Melanin','skintest_1_Erythema','skintest_1_TEWL','skintest_1_Sebum','skintest_1_R0','skintest_1_R2','skintest_1_R6','skintest_1_R7']]
    test_data = test[["ID", "Name", "sex", "Age", "affected \nside","Burn type", "TBSA(Total)", "TBSA(2')", "TBSA(3')", "TBSA(4')",'STSG', 'artificial dermis', 'skintest site','skintest_1_Thickness','skintest_1_Melanin','skintest_1_Erythema','skintest_1_TEWL','skintest_1_Sebum','skintest_1_R0','skintest_1_R2','skintest_1_R6','skintest_1_R7']]
    ex_test = ex_test[["ID", "Name", "sex", "Age", "affected \nside","Burn type", "TBSA(Total)", "TBSA(2')", "TBSA(3')", "TBSA(4')",'STSG', 'artificial dermis', 'skintest site','skintest_1_Thickness','skintest_1_Melanin','skintest_1_Erythema','skintest_1_TEWL','skintest_1_Sebum','skintest_1_R0','skintest_1_R2','skintest_1_R6','skintest_1_R7']]

    def generate_tableone_and_display(train, group_var, label):
        columns = ["ID", "Name", "sex", "Age", "affected \nside", "Burn type",
                "TBSA(Total)", "TBSA(2')", "TBSA(3')", "TBSA(4')", 'STSG',
                'artificial dermis', 'skintest site', 'skintest_1_Thickness',
                'skintest_1_Melanin', 'skintest_1_Erythema', 'skintest_1_TEWL',
                'skintest_1_Sebum', 'skintest_1_R0', 'skintest_1_R2',
                'skintest_1_R6', 'skintest_1_R7']
        
        df = train[[group_var] + columns]
        table = TableOne(df, groupby=group_var, pval=True)

        st.subheader(f"TableOne - {label} 기준")
        st.dataframe(table.tableone)

        # 다운로드용 CSV 스트링 만들기
        csv = table.to_csv()
        st.download_button(
            label=f"📥 {label} 기준 TableOne CSV 다운로드",
            data=csv,
            file_name=f"tableone_{label}.csv",
            mime="text/csv"
        )
    
    generate_tableone_and_display(train, 'Thickness', 'Thickness')
    generate_tableone_and_display(train, 'Melanin', 'Melanin')
    generate_tableone_and_display(train, 'Erythema', 'Erythema')
    generate_tableone_and_display(train, 'TEWL', 'TEWL')
    generate_tableone_and_display(train, 'Sebum', 'Sebum')
    generate_tableone_and_display(train, 'R0', 'R0')
    generate_tableone_and_display(train, 'R2', 'R2')
    generate_tableone_and_display(train, 'R6', 'R6')
    generate_tableone_and_display(train, 'R7', 'R7')

    #classification
    y0=train_total['Thickness'].iloc[0:180].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y0_internal = train_total['Thickness'].iloc[180:336].values
    y0_external = train_total['Thickness'].iloc[336:].values
    y1=train_total['Melanin'].iloc[0:180].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y1_internal = train_total['Melanin'].iloc[180:336].values
    y1_external = train_total['Melanin'].iloc[336:].values
    y2=train_total['Erythema'].iloc[0:180].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y2_internal = train_total['Erythema'].iloc[180:336].values
    y2_external = train_total['Erythema'].iloc[336:].values
    y3=train_total['TEWL'].iloc[0:180].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y3_internal = train_total['TEWL'].iloc[180:336].values
    y3_external = train_total['TEWL'].iloc[336:].values
    y4=train_total['Sebum'].iloc[0:180].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y4_internal = train_total['Sebum'].iloc[180:336].values
    y4_external = train_total['Sebum'].iloc[336:].values
    y5=train_total['R0'].iloc[0:180].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y5_internal = train_total['R0'].iloc[180:336].values
    y5_external = train_total['R0'].iloc[336:].values
    y6=train_total['R2'].iloc[0:180].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y6_internal = train_total['R2'].iloc[180:336].values
    y6_external = train_total['R2'].iloc[336:].values
    y7=train_total['R6'].iloc[0:180].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y7_internal = train_total['R6'].iloc[180:336].values
    y7_external = train_total['R6'].iloc[336:].values
    y8=train_total['R7'].iloc[0:180].values # 엑셀 데이터에 'hypo' 열이 두 개여서 위치 기반으로 지정 (원래는 y0=train['hypo'].values)
    y8_internal = train_total['R7'].iloc[180:336].values
    y8_external = train_total['R7'].iloc[336:].values
    
    train_data=train_data.astype('float')
    test_data=test_data.astype('float')
    ex_test=ex_test.astype('float')
    ex_test['Name']

    #train
    X = np.array(train_data)
    X_test = np.array(test_data)
    X_test_external = np.array(ex_test)

    print("Preprocessing Done")

    y_variable_names = ['y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8']

    return train_data, test_data, ex_test, y_variables, y_variable_names


# 분석에 사용할 공통 컬럼
columns = ["ID", "Name", "sex", "Age", "affected \nside", "Burn type",
           "TBSA(Total)", "TBSA(2')", "TBSA(3')", "TBSA(4')", 'STSG',
           'artificial dermis', 'skintest site', 'skintest_1_Thickness',
           'skintest_1_Melanin', 'skintest_1_Erythema', 'skintest_1_TEWL',
           'skintest_1_Sebum', 'skintest_1_R0', 'skintest_1_R2',
           'skintest_1_R6', 'skintest_1_R7']

# Group 기준이 될 변수 리스트
group_vars = ["skintest_1_Thickness", "skintest_1_Melanin", "skintest_1_Erythema",
              "skintest_1_TEWL", "skintest_1_Sebum", "skintest_1_R0", 
              "skintest_1_R2", "skintest_1_R6", "skintest_1_R7"]

def generate_table(data, column_name):
    try:
        selected_columns = [column_name] + columns
        df = data[selected_columns].dropna(subset=[column_name])  # groupby 컬럼 결측 제거
        table = TableOne(df, groupby=column_name, pval=True)
        table_csv = table.to_csv(index=True)

        st.success(f"TableOne 생성 완료: Group by {column_name}")
        st.dataframe(table.tableone)  # 간단한 요약 테이블 미리보기
        st.download_button(label=f"Download TableOne for {column_name}",
                           data=table_csv,
                           file_name=f'tableone_{column_name}.csv')
    except Exception as e:
        st.error(f"Error generating TableOne for {column_name}: {e}")



def run_adversarial_evaluation(X, X_test, X_test_external, y_variables, save_dir="./adversarial_results"):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    import matplotlib as mpl
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
    from sklearn.model_selection import train_test_split
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import shap
    from lime import lime_tabular

    # 사용자가 설정한 y 변수 리스트에 대해 반복 실행
    y_variables = [(y0, y0_internal, y0_external, 'y0'),
                (y1, y1_internal, y1_external, 'y1'),
                (y2, y2_internal, y2_external, 'y2'),
                (y3, y3_internal, y3_external, 'y3'),
                (y4, y4_internal, y4_external, 'y4'),
                (y5, y5_internal, y5_external, 'y5'),
                (y6, y6_internal, y6_external, 'y6'),
                (y7, y7_internal, y7_external, 'y7'),
                (y8, y8_internal, y8_external, 'y8')]

    warnings.filterwarnings('ignore')
    os.makedirs(save_dir, exist_ok=True)

    def remove_multicollinearity(df, thresh=5.0):
        variables = df.columns.tolist()
        removed = []
        while True:
            vif = pd.Series([variance_inflation_factor(df[variables].values, i) for i in range(len(variables))], index=variables)
            max_vif = vif.max()
            if max_vif > thresh:
                max_var = vif.idxmax()
                variables.remove(max_var)
                removed.append(max_var)
            else:
                break
        return df[variables], removed

    def fgsm_attack(X, epsilon=0.1):
        noise = epsilon * np.sign(np.random.randn(*X.shape))
        return np.clip(X + noise, X.min(), X.max())

    def pgd_attack(X, epsilon=0.1, alpha=0.01, iterations=10):
        X_adv = X.copy()
        for _ in range(iterations):
            noise = alpha * np.sign(np.random.randn(*X.shape))
            X_adv = np.clip(X_adv + noise, X - epsilon, X + epsilon)
        return X_adv

    def cw_attack(X, confidence=0.1):
        noise = confidence * np.random.randn(*X.shape)
        return np.clip(X + noise, X.min(), X.max())

    def uniform_noise_attack(X, epsilon=0.1):
        noise = np.random.uniform(-epsilon, epsilon, X.shape)
        return np.clip(X + noise, X.min(), X.max())

    def shap_visualize(model, X_train, X_test, model_name, target_name, attack_name):
        if model_name.upper() == "SVR":
            return
        try:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            st.markdown(f"**SHAP Summary: {target_name} | {attack_name} | {model_name}**")
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(plt.gcf())
            plt.close()
        except Exception:
            st.warning(f"SHAP 시각화 실패: {model_name}, {target_name}, {attack_name}")

    def lime_visualize(model, X_train, X_test, feature_names, model_name, target_name, attack_name):
        explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, verbose=False, mode='regression')
        exp = explainer.explain_instance(X_test[0], model.predict, num_features=len(feature_names))
        fig = exp.as_pyplot_figure()
        st.markdown(f"**LIME Explanation: {target_name} | {attack_name} | {model_name}**")
        st.pyplot(fig)
        plt.close()

    def plot_radar_by_attack(results, y_name, metric="R2_Adversarial"):
        categories = list(models.keys())
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
        for attack in attack_methods:
            plt.figure(figsize=(8, 8))
            for scenario in attack_scenarios:
                values = [results[scenario][y_name][attack].get(model, {}).get(metric, np.nan) for model in categories] + [np.nan]
                ax = plt.subplot(111, polar=True)
                ax.plot(angles, values, label=scenario)
                ax.fill(angles, values, alpha=0.1)
            plt.xticks(angles[:-1], categories)
            plt.title(f"{y_name} - {attack} ({metric})")
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            st.pyplot(plt)
            plt.close()

    def plot_scenario_comparison(results, y_name, metric="R2_Adversarial"):
        scenario_names = list(attack_scenarios.keys())
        model_names = list(models.keys())
        data = pd.DataFrame(index=scenario_names, columns=model_names)
        for scenario in scenario_names:
            for model in model_names:
                vals = [results[scenario][y_name][attack].get(model, {}).get(metric, np.nan) for attack in attack_methods]
                data.loc[scenario, model] = np.nanmean(vals)
        plt.figure(figsize=(8, 6))
        sns.heatmap(data.astype(float), annot=True, fmt=".3f", cmap="viridis")
        plt.title(f"{y_name} - Scenario vs Model ({metric})")
        st.pyplot(plt)
        plt.close()

    X_df = pd.DataFrame(X)
    X_df_filtered, _ = remove_multicollinearity(X_df)
    feature_names = X_df_filtered.columns.tolist()

    scaler = StandardScaler()
    for dset, name in zip([X, X_test, X_test_external], ['X', 'X_test', 'X_test_external']):
        d = pd.DataFrame(dset, columns=X_df.columns)[X_df_filtered.columns]
        d = np.delete(d.values, [0, 1, 4], axis=1)
        if name == 'X':
            X_proc = scaler.fit_transform(d)  # fit_transform 으로 fit + transform
        elif name == 'X_test':
            X_test_proc = scaler.transform(d) # 이미 fit 된 scaler 사용
        else:
            X_ex_test_proc = scaler.transform(d) # 이미 fit 된 scaler 사용

    models = {
        'RandomForest': RandomForestRegressor(),
        'ExtraTrees': ExtraTreesRegressor(),
        'DecisionTree': DecisionTreeRegressor(),
        'HistGradientBoosting': HistGradientBoostingRegressor(),
        'SVR': SVR()
    }

    attack_scenarios = {
        "Scenario 50:50": 0.05,
        "Scenario 60:40": 0.10,
        "Scenario 70:30": 0.15,
        "Scenario 80:20": 0.20,
        "Scenario 40:60": 0.25,
        "Scenario 30:70": 0.30,
        "Scenario 20:80": 0.35
    }

    attack_methods = {
        "FGSM": fgsm_attack,
        "PGD": pgd_attack,
        "CW": cw_attack,
        "Uniform Noise": uniform_noise_attack
    }

    results = {}

    for y_main, y_internal, y_external, y_name in y_variables:
        X_train, X_test_split, y_train, y_test = train_test_split(X_proc, y_main, test_size=0.2, random_state=42)
        for scenario, epsilon in attack_scenarios.items():
            for attack_name, attack_fn in attack_methods.items():
                for model_name, model in models.items():
                    X_adv = attack_fn(X_test_split, epsilon)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test_split)
                    y_pred_adv = model.predict(X_adv)

                    mae_o = mean_absolute_error(y_test, y_pred)
                    mae_a = mean_absolute_error(y_test, y_pred_adv)
                    mse_o = mean_squared_error(y_test, y_pred)
                    mse_a = mean_squared_error(y_test, y_pred_adv)
                    evs_o = explained_variance_score(y_test, y_pred)
                    evs_a = explained_variance_score(y_test, y_pred_adv)
                    r2_o = r2_score(y_test, y_pred)
                    r2_a = r2_score(y_test, y_pred_adv)

                    results.setdefault(scenario, {}).setdefault(y_name, {}).setdefault(attack_name, {})[model_name] = {
                        "MAE_Original": mae_o,
                        "MAE_Adversarial": mae_a,
                        "MSE_Original": mse_o,
                        "MSE_Adversarial": mse_a,
                        "EVS_Original": evs_o,
                        "EVS_Adversarial": evs_a,
                        "R2_Original": r2_o,
                        "R2_Adversarial": r2_a
                    }

        # ▶탭 구분 출력
        st.markdown(f"### 🎯 Target: {y_name}")
        tab1, tab2, tab3 = st.tabs(["📋 결과 요약표", "📊 Heatmap", "📈 Radar Chart"])

        with tab1:
            df_results = pd.DataFrame([
                [scenario, y_name, attack, model, m["MAE_Original"], m["MSE_Original"], m["EVS_Original"],
                 m["MAE_Adversarial"], m["MSE_Adversarial"], m["EVS_Adversarial"]]
                for scenario in results
                for attack in results[scenario][y_name]
                for model, m in results[scenario][y_name][attack].items()
            ], columns=["Scenario", "Target", "Attack", "Model", "MAE_Original", "MSE_Original", "EVS_Original",
                        "MAE_Adversarial", "MSE_Adversarial", "EVS_Adversarial"])
            st.dataframe(df_results)

        with tab2:
            plot_scenario_comparison(results, y_name)

        with tab3:
            plot_radar_by_attack(results, y_name)

    # 저장
    df_results.to_csv(os.path.join(save_dir, "adversarial_results.csv"), index=False)
    return df_results



def main():
    st.title("Data Processing and TableOne Generator")
    uploaded_file = st.file_uploader("Upload your dataset (.xlsx)", type=["xlsx"])

    if uploaded_file is not None:
        try:
            # 엑셀 확인
            uploaded_file.seek(0)  # 중요!
            data = pd.read_excel(uploaded_file)
            st.write("데이터 미리보기", data.head())

            # 다시 초기화해서 넘김
            uploaded_file.seek(0)
            train_data, test_data, ex_test_data, y_variables, y_variable_names = process_data(uploaded_file)

            # Group 기준 선택
            group_vars = train_data.columns.tolist()
            group_col = st.selectbox("Group 기준 변수 선택", options=group_vars)

            if st.button("TableOne 생성"):
                generate_table(data, group_col)

            if st.button("Start Adversarial Evaluation"):
                df_result = run_adversarial_evaluation(train_data.values, test_data.values, ex_test_data.values, y_variables)
                st.success("Adversarial Evaluation 완료")
                st.dataframe(df_result.head())

        except Exception as e:
            st.error(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
