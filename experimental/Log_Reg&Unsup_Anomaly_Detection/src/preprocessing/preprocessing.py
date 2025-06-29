import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
from sklearn.decomposition import PCA


class TargetEncoder:
    def __init__(self, smoothing=5.0):
        self.target_means = {}
        self.global_mean = None
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.global_mean = y.mean()
        for col in X.columns:
            agg = X[col].to_frame().join(y).groupby(col)[y.name].agg(['mean', 'count'])
            smoothing = 1 / (1 + np.exp(-(agg['count'] - self.smoothing)))
            smoothed = self.global_mean * (1 - smoothing) + agg['mean'] * smoothing
            self.target_means[col] = smoothed.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.apply(lambda col: col.map(self.target_means.get(col.name, {})).fillna(self.global_mean))



class FrequencyEncoder:
    def __init__(self):
        self.freq_maps = {}

    def fit(self, X: pd.DataFrame):
        self.freq_maps = {
            col: X[col].value_counts(normalize=True).to_dict()
            for col in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.apply(lambda col: col.map(self.freq_maps.get(col.name, {})).fillna(0))


class MultiColumnLabelEncoder:
    def __init__(self):
        self.label_maps = {}

    def fit(self, X: pd.DataFrame):
        for col in X.columns:
            unique_vals = pd.Series(X[col].dropna().unique())
            self.label_maps[col] = {val: idx for idx, val in enumerate(unique_vals)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        encoded = X.copy()
        for col in X.columns:
            encoded[col] = encoded[col].map(self.label_maps[col]).fillna(-1).astype(int)
        return encoded


def simple_preprocess(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    numeric_imputer_strategy: str,
    categorical_imputer_strategy: str,
    apply_smote: bool = False,
    random_state: int = 42,
    sampling_strategy="auto",
    threshold_imbalanced: int = 0.95,
    scaler_strategy= MinMaxScaler,
    cardinality_amount: int = 42,
    label_encode_cols: list = None,
    apply_PCA: bool = False
) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:

    label_encode_cols = []

    # === 0. Drop only highly imbalanced columns (from training set only) ===
    imbalanced_cols = [
        col for col in x_train.columns
        if x_train[col].value_counts(normalize=True, dropna=False).values[0] >= threshold_imbalanced
    ]
    cols_to_drop = imbalanced_cols

    x_train = x_train.drop(columns=cols_to_drop)
    x_val = x_val.drop(columns=cols_to_drop, errors='ignore')
    x_test = x_test.drop(columns=cols_to_drop, errors='ignore')

    print(f"Dropped columns: {cols_to_drop}")

    # === 0. Identify column types ===
    numeric_cols = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = [col for col in x_train.columns if col not in numeric_cols]
    
    low_card_cols = [
        col for col in categorical_cols
        if x_train[col].nunique() <= cardinality_amount and col not in label_encode_cols
    ]
    high_card_cols = [
        col for col in categorical_cols
        if x_train[col].nunique() > cardinality_amount and col not in label_encode_cols
    ]

    # === 1. Impute numeric columns ===
    num_imputer = SimpleImputer(strategy=numeric_imputer_strategy)
    x_train_num = pd.DataFrame(num_imputer.fit_transform(x_train[numeric_cols]), columns=numeric_cols)
    x_val_num = pd.DataFrame(num_imputer.transform(x_val[numeric_cols]), columns=numeric_cols)
    x_test_num = pd.DataFrame(num_imputer.transform(x_test[numeric_cols]), columns=numeric_cols)
    print(f"[Step 1] Numeric columns imputed: {numeric_cols}")

    # === Add custom feature: INCOME_BELOW_AVG ===
    if 'TOTAL_INCOME' in x_train_num.columns:
        avg_income = x_train_num['TOTAL_INCOME'].mean()

        x_train_num['INCOME_BELOW_AVG'] = (x_train_num['TOTAL_INCOME'] < avg_income).astype(int)
        x_val_num['INCOME_BELOW_AVG'] = (x_val_num['TOTAL_INCOME'] < avg_income).astype(int)
        x_test_num['INCOME_BELOW_AVG'] = (x_test_num['TOTAL_INCOME'] < avg_income).astype(int)

        print(f"[Custom Feature] INCOME_BELOW_AVG added using avg from training set: {avg_income:.2f}")
    
    # === Bin AGE into 4 groups based on quantiles ===
    if 'AGE' in x_train_num.columns:
        age_bins = np.quantile(x_train_num['AGE'], q=[0, 0.25, 0.5, 0.75, 1.0])
        age_bins = np.unique(age_bins)
        x_train['AGE_BIN'] = pd.cut(x_train_num['AGE'], bins=age_bins, include_lowest=True).astype(str)
        x_val['AGE_BIN'] = pd.cut(x_val_num['AGE'], bins=age_bins, include_lowest=True).astype(str)
        x_test['AGE_BIN'] = pd.cut(x_test_num['AGE'], bins=age_bins, include_lowest=True).astype(str)

        # Then update categorical columns list to include AGE_BIN
        categorical_cols.append('AGE_BIN')

    # === 2. Impute and One-Hot Encode low-cardinality categorical columns ===
    cat_imputer = SimpleImputer(strategy=categorical_imputer_strategy)
    x_train_low_cat = pd.DataFrame(cat_imputer.fit_transform(x_train[low_card_cols]), columns=low_card_cols)
    x_val_low_cat = pd.DataFrame(cat_imputer.transform(x_val[low_card_cols]), columns=low_card_cols)
    x_test_low_cat = pd.DataFrame(cat_imputer.transform(x_test[low_card_cols]), columns=low_card_cols)
    print(f"[Step 2] Low-card categorical columns imputed:{low_card_cols}")

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    x_train_low_cat_encoded = pd.DataFrame(
        ohe.fit_transform(x_train_low_cat),
        columns=ohe.get_feature_names_out(low_card_cols)
    )
    x_val_low_cat_encoded = pd.DataFrame(
        ohe.transform(x_val_low_cat),
        columns=ohe.get_feature_names_out(low_card_cols)
    )
    x_test_low_cat_encoded = pd.DataFrame(
        ohe.transform(x_test_low_cat),
        columns=ohe.get_feature_names_out(low_card_cols)
    )
    print(f"[Step 2] Low-card categorical columns one-hot encoded:{low_card_cols}")

    # === 3. Impute + *Encode high-card categorical columns ===
    x_train_high_cat = pd.DataFrame(cat_imputer.fit_transform(x_train[high_card_cols]), columns=high_card_cols)
    x_val_high_cat = pd.DataFrame(cat_imputer.transform(x_val[high_card_cols]), columns=high_card_cols)
    x_test_high_cat = pd.DataFrame(cat_imputer.transform(x_test[high_card_cols]), columns=high_card_cols)
    print(f"[Step 3] High-card categorical columns imputed:{high_card_cols}")

    target_encoder = TargetEncoder(smoothing=5.0).fit(x_train_high_cat, y_train)
    x_train_high_cat_encoded = target_encoder.transform(x_train_high_cat)
    x_val_high_cat_encoded = target_encoder.transform(x_val_high_cat)
    x_test_high_cat_encoded = target_encoder.transform(x_test_high_cat)
    print(f"[Step 3] High-card categorical columns target encoded: {high_card_cols}")
    #freq_encoder = FrequencyEncoder().fit(x_train_high_cat)
    #x_train_high_cat_encoded = freq_encoder.transform(x_train_high_cat)
    #x_val_high_cat_encoded = freq_encoder.transform(x_val_high_cat)
    #x_test_high_cat_encoded = freq_encoder.transform(x_test_high_cat)
    #print(f"[Step 3] High-card categorical columns frequency encoded{high_card_cols}")

    # === 4. Impute + LabelEncode specified columns ===
    #x_train_label = pd.DataFrame(cat_imputer.fit_transform(x_train[label_encode_cols]), columns=label_encode_cols)
    #x_val_label = pd.DataFrame(cat_imputer.transform(x_val[label_encode_cols]), columns=label_encode_cols)
    #x_test_label = pd.DataFrame(cat_imputer.transform(x_test[label_encode_cols]), columns=label_encode_cols)
    #print("[Step 4] Label-encode columns imputed.")

    #label_encoder = MultiColumnLabelEncoder().fit(x_train_label)
    #x_train_label_encoded = label_encoder.transform(x_train_label)
    #x_val_label_encoded = label_encoder.transform(x_val_label)
    #x_test_label_encoded = label_encoder.transform(x_test_label)
    #print("[Step 4] Label-encoded specified columns.")

   # === 1a. Automatically detect and log-transform skewed numeric columns
    log_transform_cols = []

    for col in x_train_num.columns:
        if (x_train_num[col] > 0).all():
            skew_val = x_train_num[col].skew()
            if abs(skew_val) > 1:
                log_transform_cols.append(col)

    if log_transform_cols:
        for col in log_transform_cols:
            x_train_num[col] = np.log1p(x_train_num[col])
            x_val_num[col] = np.log1p(x_val_num[col])
            x_test_num[col] = np.log1p(x_test_num[col])

    print(f"[Step 1a] Automatically log-transformed columns: {log_transform_cols}")


    # === 5. Scale numeric columns (conditionally)
    force_scale_cols = ['AGE']
    columns_to_scale = []
    for col in x_train_num.columns:
        unique_vals = x_train_num[col].nunique()
        min_val = x_train_num[col].min()
        max_val = x_train_num[col].max()
        std_dev = x_train_num[col].std()
        skew = x_train_num[col].skew()

       
        if col in force_scale_cols or (unique_vals > 15 and (abs(skew) > 1 or std_dev > 1000 or max_val - min_val > 1000)):
            columns_to_scale.append(col)

    if columns_to_scale:
        scaler = scaler_strategy().fit(x_train_num[columns_to_scale])

        x_train_num[columns_to_scale] = scaler.transform(x_train_num[columns_to_scale])
        x_val_num[columns_to_scale] = scaler.transform(x_val_num[columns_to_scale])
        x_test_num[columns_to_scale] = scaler.transform(x_test_num[columns_to_scale])

    print(f"[Step 5] Scaled columns: {columns_to_scale}")

    # === 6. Combine all processed parts ===
    x_train_final = pd.concat([
        x_train_num,
        x_train_low_cat_encoded,
        x_train_high_cat_encoded,
        #x_train_label_encoded
    ], axis=1)

    x_val_final = pd.concat([
        x_val_num,
        x_val_low_cat_encoded,
        x_val_high_cat_encoded,
        #x_val_label_encoded
    ], axis=1)

    x_test_final = pd.concat([
        x_test_num,
        x_test_low_cat_encoded,
        x_test_high_cat_encoded,
        #x_test_label_encoded
    ], axis=1)

    # === 6b. Apply PCA if requested ===
    if apply_PCA:
        pca = PCA(n_components=0.95)  # o pca = PCA(n_components=10) para 10 componentes
        # Ajustar PCA al conjunto de entrenamiento
        x_train_pca = pca.fit_transform(x_train_final)
        # Transformar validación y test con el mismo pca
        x_val_pca = pca.transform(x_val_final)
        x_test_pca = pca.transform(x_test_final)
        # Obtener el número de componentes resultantes
        num_componentes = x_train_pca.shape[1]
        # Crear nombres para las columnas
        columnas_pca = [f'PC{i+1}' for i in range(num_componentes)]

        # Convertir a DataFrame
        x_train_final = pd.DataFrame(x_train_pca, columns=columnas_pca)
        x_val_final = pd.DataFrame(x_val_pca, columns=columnas_pca)
        x_test_final = pd.DataFrame(x_test_pca, columns=columnas_pca)
        print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
        print("Varianza total explicada:", sum(pca.explained_variance_ratio_))

    # === 7. Optionally apply SMOTE ===
    if apply_smote:
        smote = SMOTE(  sampling_strategy, random_state=random_state)
        x_train_final, y_train = smote.fit_resample(x_train_final, y_train)

    return (x_train_final, y_train), (x_val_final, y_val), (x_test_final, y_test)
