import pandas as pd
import random
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv('secondary_data.csv', sep=';')

    X = df.drop('class', axis=1)
    y = df['class']

    results = []

    for index, x in X.iterrows():
        if x["gill-color"] == 'w' or x['has-ring'] == 't' or x['stem-root'] == 'b' or x['stem-root'] == 's' or x['stem-color'] == 'e' or x['cap-color'] == 'e' or x['does-bruise-or-bleed'] == 't':
            results.append('p')
        else:
            results.append(random.choice(['p', 'e']))
        
    df_results = pd.DataFrame({"clas": results})
    # convert (61069, 1) to (61069,)
    df_results = df_results.clas.to_numpy()

    matches = (df_results == y).sum().sum()  # Count matching values
    total_values = len(df_results)  # Total number of values
    percentage_match = (matches / total_values) * 100  # Calculate percentage
    
    print(percentage_match)

if __name__ == '__main__':
    main()