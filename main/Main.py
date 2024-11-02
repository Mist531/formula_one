import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
from enum import Enum
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


class ModelType(Enum):
    GRADIENT_BOOSTING = 'GradientBoosting'
    RANDOM_FOREST = 'RandomForest'
    LOGISTIC_REGRESSION = 'LogisticRegression'
    DECISION_TREE = 'DecisionTree'


def load_data(drivers_file, races_file, results_file, circuits_file):
    drivers = pd.read_csv(drivers_file)
    races = pd.read_csv(races_file)
    results = pd.read_csv(results_file)
    circuits = pd.read_csv(circuits_file)

    return drivers, races, results, circuits


def prepare_data(drivers, races, results, circuits):
    data = results.merge(drivers, on='driverId', how='left')
    data = data.merge(races, on='raceId', how='left')
    data = data.merge(circuits, on='circuitId', how='left')

    data['win'] = data['positionOrder'].apply(lambda x: 1 if x == 1 else 0)

    return data


def calculate_historical_stats(data, alpha=5):
    global_avg_win_rate = data['win'].mean()

    grouped = data.groupby(['driverId', 'circuitId']).agg(
        races=('raceId', 'nunique'),
        wins=('win', 'sum')
    ).reset_index()

    grouped['win_rate_regularized'] = (
            (grouped['wins'] + alpha * global_avg_win_rate) / (grouped['races'] + alpha)
    )

    return grouped, global_avg_win_rate


def encode_data(data):
    le_driver = LabelEncoder()
    data['driverId_encoded'] = le_driver.fit_transform(data['driverId'])

    le_circuit = LabelEncoder()
    data['circuitId_encoded'] = le_circuit.fit_transform(data['circuitId'])

    le_constructor = LabelEncoder()
    data['constructorId_encoded'] = le_constructor.fit_transform(data['constructorId'])

    return data, le_driver, le_circuit, le_constructor


def resample_data(x, y):
    rus = RandomUnderSampler(random_state=42)
    x_resampled, y_resampled = rus.fit_resample(x, y)
    print(f"Распределение классов после пересэмплирования: {Counter(y_resampled)}")
    return x_resampled, y_resampled


def train_ml_model(data, features, target, model, model_type):
    x = data[features]
    y = data[target]
    x_resampled, y_resampled = resample_data(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f'\nМодель: {model_type.value}')
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model


def predict_win_probabilities(data, historical_stats, drivers, circuits, model, features):
    circuits_latest = data['circuitId'].unique()
    drivers_latest = data['driverId'].unique()

    results_list = []

    for circuit_id in circuits_latest:
        circuit_name = circuits[circuits['circuitId'] == circuit_id]['name'].values[0]

        driver_probs = []
        for driver_id in drivers_latest:
            driver_stats = historical_stats[
                (historical_stats['driverId'] == driver_id) &
                (historical_stats['circuitId'] == circuit_id)
                ]

            if driver_stats.empty:
                continue

            driver_row = drivers[drivers['driverId'] == driver_id]
            driver_name = driver_row['forename'].values[0] + ' ' + driver_row['surname'].values[0]

            x_new = pd.DataFrame({
                'driverId_encoded': [data[data['driverId'] == driver_id]['driverId_encoded'].values[0]],
                'circuitId_encoded': [data[data['circuitId'] == circuit_id]['circuitId_encoded'].values[0]],
                'constructorId_encoded': [data[data['driverId'] == driver_id]['constructorId_encoded'].values[0]],
                'grid': [1],
                'year': [data['year'].max()]
            })

            prob_ml = model.predict_proba(x_new[features])[0][1]

            win_rate = driver_stats['win_rate_regularized'].values[0]
            final_prob = (win_rate + prob_ml) / 2

            driver_probs.append({'driver_name': driver_name, 'probability': final_prob})

        driver_probs_sorted = sorted(driver_probs, key=lambda x: x['probability'], reverse=True)

        top_3 = driver_probs_sorted[:3]

        results_list.append({
            'Название трассы': circuit_name,
            '1 место': f"{top_3[0]['driver_name']}, {top_3[0]['probability']:.2f}" if len(top_3) > 0 else '',
            '2 место': f"{top_3[1]['driver_name']}, {top_3[1]['probability']:.2f}" if len(top_3) > 1 else '',
            '3 место': f"{top_3[2]['driver_name']}, {top_3[2]['probability']:.2f}" if len(top_3) > 2 else ''
        })

    return results_list


def display_results(results_list):
    results_df = pd.DataFrame(results_list)
    print("\nТаблица вероятностей победы гонщиков на конкретных трассах:")
    print(tabulate(results_df, headers='keys', tablefmt='fancy_grid'))


def main(start_year, end_year, alpha=5, model_type=ModelType.GRADIENT_BOOSTING):
    drivers, races, results, circuits = load_data('dataset/drivers.csv', 'dataset/races.csv', 'dataset/results.csv',
                                                  'dataset/circuits.csv')

    data = prepare_data(drivers, races, results, circuits)

    data = data[(data['year'] >= start_year) & (data['year'] <= end_year)]

    historical_stats, global_avg_win_rate = calculate_historical_stats(data, alpha)

    data, le_driver, le_circuit, le_constructor = encode_data(data)

    features = ['driverId_encoded', 'circuitId_encoded', 'constructorId_encoded', 'grid', 'year']
    target = 'win'

    if model_type == ModelType.GRADIENT_BOOSTING:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == ModelType.RANDOM_FOREST:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    elif model_type == ModelType.LOGISTIC_REGRESSION:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    elif model_type == ModelType.DECISION_TREE:
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    else:
        raise ValueError("Invalid model type specified.")

    model = train_ml_model(data, features, target, model, model_type)

    results_list = predict_win_probabilities(data, historical_stats, drivers, circuits, model, features)
    display_results(results_list)


if __name__ == "__main__":
    main(2010, 2020, alpha=5, model_type=ModelType.GRADIENT_BOOSTING)
    main(2010, 2020, alpha=5, model_type=ModelType.RANDOM_FOREST)
    main(2010, 2020, alpha=5, model_type=ModelType.LOGISTIC_REGRESSION)
    main(2010, 2020, alpha=5, model_type=ModelType.DECISION_TREE)
