#IMPORT MODULES
import pandas as pd

# Define the updated handle_unseen_labels function here
def handle_unseen_labels(encoder, data, default_label='Other'):
    unseen_labels = set(data) - set(encoder.classes_)
    if unseen_labels:
        data = [label if label in encoder.classes_ else default_label for label in data]
    return encoder.transform(data)

#READ THE DATA FILE
def prepare_dataframe(df1, model, label_encoders, scaler):
    
    #Number of Languages known
    column_name = "LanguageHaveWorkedWith"
    def languagecount(row):
        value = str(row[column_name]).split(";")
        if(value[0] == "nan"):return 0
        return len(value)
    df1["NumberOfLanguagesKnown"] = df1.apply(languagecount,axis = 1)
    
    # Number of Databases known
    column_name = "DatabaseHaveWorkedWith"
    def languagecount(row):
        value = str(row[column_name]).split(";")
        if(value[0] == "nan"):return 0
        return len(value)
    df1["NumberOfDatabasesKnown"] = df1.apply(languagecount,axis = 1)
    

    #Number of Sources from which Coding was learnt
    column_name = "LearnCode"
    def learncodecount(row):
        value = str(row[column_name]).split(";")
        if(value[0] == "nan"):return 0
        return len(value)
    df1["NumberOfLearningSources"] = df1.apply(learncodecount,axis = 1)
    
    columns_selected =  [
       'Age',
       'AISelect',
       'OrgSize',
       'DevType',
       'YearsCode',
       'WorkExp',
       'YearsCodePro',
       "RemoteWork",
       'Currency',
       "EdLevel",
       "NumberOfDatabasesKnown",
       "NumberOfLanguagesKnown",
       "NumberOfLearningSources"
    ]
    
    train_columns = [
       'Age',
       'AISelect',
       'OrgSize',
       'DevType',
       "RemoteWork",
       'Currency',
       "EdLevel",
       "ExperienceCategory",
       "YearsCodeCategory",
       "YearsCodeProCategory",
       "NumberOfDatabasesKnown",
       "NumberOfLanguagesKnown",
       "NumberOfLearningSources"
    ]
    
    df1 = df1[columns_selected]
    
    
    #CATEGORISE COLUMNS  INTO MAJORITY VALUES AND 'OTHER'
    def shorten_categories(categories, cutoff):
        categorical_map = {}
        for i in range(len(categories)):
            if categories.values[i] >= cutoff:
                categorical_map[categories.index[i]] = categories.index[i]
            else:
                categorical_map[categories.index[i]] = 'Other'
        return categorical_map
    
    
    currency_map = shorten_categories(df1.Currency.value_counts(), 400)
    df1['Currency'] = df1['Currency'].map(currency_map)
    
    #CATEGORISE THE WORK EXPERIENCE INTO BINS
    bins = [0, 2, 5, 10, 20, 30, 40, 50, float('inf')]  # Define custom bin edges
    labels = [0, 1, 2, 3, 4, 5, 6, 7]  # Define labels
    
    # Create a new column with the categories
    df1['ExperienceCategory'] = pd.cut(df1['WorkExp'], bins=bins, labels=labels)
    
    #CATEGORISE LESS THAN 1 YEAR AS 0 AND MORE THAN 50 AS 51 FOR YEARS OF CODE
    df1['YearsCode'] = df1['YearsCode'].replace("Less than 1 year", 0)
    df1['YearsCode'] = df1['YearsCode'].replace("More than 50 years", 51)
    
    df1['YearsCodePro'] = df1['YearsCodePro'].replace("Less than 1 year", 0)
    df1['YearsCodePro'] = df1['YearsCodePro'].replace("More than 50 years", 51)
    
    #CATEGORISE YEARS OF CODE INTO BINS
    bins = [0, 2, 5, 10, 20, 30, 40, 50, float('inf')]  # Define custom bin edges
    labels = [0, 1, 2, 3, 4, 5, 6, 7]  # Define labels
    
    # Create a new column with the categories
    df1["YearsCode"] = df1["YearsCode"].astype(int)
    df1["YearsCodePro"] = df1["YearsCodePro"].astype(int)
    df1['YearsCodeCategory'] = pd.cut(df1['YearsCode'], bins=bins, labels=labels)
    df1['YearsCodeProCategory'] = pd.cut(df1['YearsCodePro'], bins=bins, labels=labels)
    
    #LABEL ENCODE THE COLUMNS
    df_LE = df1.copy()
    df_LE = df_LE.dropna()
    
    for i in train_columns:
        if i == "ConvertedCompYearly":
            continue
    
        df_LE[i] = handle_unseen_labels(label_encoders[i], df_LE[i])
    
    X = df_LE[train_columns]
    
    
    #SCALE THE DATA
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)
    
    return prediction
