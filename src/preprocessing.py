import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    #fill missing values of Age with median
    df['Age_filled'] = df['Age'].fillna(df['Age'].median())

    #fill missing values of Embarked with mode
    df['Embarked_filled'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    #crate a new feature 'Has_Cabin' to indicate whether a passenger has a cabin or not
    df['HasCabin'] = df['Cabin'].notnull().astype(int)

    return df

def extract_title(name):
    title = name.split(',')[1].split('.')[0].strip()
    return title

def engineer_features(df):
    #extract title from Name
    df['Title'] = df['Name'].apply(extract_title)

    #map titles to a smaller set of categories
    Title_mapping = { 'Mlle': 'Miss', 'Mme' : 'Mrs', 'Ms' : 'Miss', 'Dr' : 'Rare', 'Rev' : 'Rare',
    'Major' : 'Rare', 'Col' : 'Rare', 'Don' : 'Rare', 'Lady' : 'Rare', 'Sir' : 'Rare',
    'Capt' : 'Rare', 'the Countess' : 'Rare', 'Jonkheer' : 'Rare'
    }

    #replace titles in the dataframe
    df['Title_Organised'] = df['Title'].replace(Title_mapping)

    #create a new feature 'FamilySize' by adding 'SibSp' and 'Parch' and adding 1 for the passenger themselves
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    #create a new feature 'isAlone' to indicate whether a passenger is alone or not
    df['isAlone'] = df['FamilySize'].eq(1).astype(int)

    #create a new feature 'Age_Group' by binning the 'Age_filled' feature into categories
    df['Age_Group'] = pd.cut(df['Age_filled'], bins=[0, 12, 19, 59, 100], labels = ['Child', 'Teen', 'Adult', 'Senior'])

    return df

def encode_features(df):
    # Sex: simple binary encoding
    df['Sex_encoded'] = df['Sex'].replace({'male': 0, 'female': 1}).astype(int)

    # Embarked: one-hot encode, drop 'S' as baseline
    df = pd.concat([df, pd.get_dummies(df['Embarked_filled'])], axis=1)
    df = df.drop(columns=['S'])

    # Age_Group: ordinal encoding (real order: Child < Teen < Adult < Senior)
    df['Age_Group_encoded'] = df['Age_Group'].astype(str).replace(
        {'Child': 0, 'Teen': 1, 'Adult': 2, 'Senior': 3}
    ).astype(int)

    # Title: one-hot encode, drop 'Mr' as baseline
    df = pd.concat([df, pd.get_dummies(df['Title_Organised'])], axis=1)
    df = df.drop(columns=['Mr'])

    # Convert all remaining boolean dummy columns to int
    bool_cols = ['C', 'Q', 'Master', 'Miss', 'Mrs', 'Rare']
    df[bool_cols] = df[bool_cols].astype(int)

    return df

def get_features_and_target(df):
    # Define features and target variable
    features = ['Pclass', 'FamilySize', 'isAlone', 'Sex_encoded', 'Age_filled', 'Age_Group_encoded', 'C', 'Q', 'Master', 'Miss', 'Mrs', 'Rare', 'Fare', 'HasCabin']

    X = df[features]
    y = df['Survived']

    return X, y

def full_pipeline(file_path):
    df = load_data(file_path)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_features(df)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
