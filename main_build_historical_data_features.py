import pandas as pd
import os
import glob

from features.build_features import add_indicators

def cancel_stock_split(df, name):
    list_split_by_name = {
        "AAPL": ("2020-08-31", 4),
        "AMZN": ("2022-06-03", 20),
        "WMT": ("2024-02-24", 3)
    }
    if name in list_split_by_name:
        df.loc[df['t'] > list_split_by_name[name][0], 'c'] *= list_split_by_name[name][1]
    return df

def build_feature_DJIA():
    chemin_csv = glob.glob(os.path.join("data/raw/DJIA", '*.csv'))

    # Lire chaque CSV en DataFrame et les stocker dans une liste
    liste_dataframes = [(pd.read_csv(fichier), fichier.split('_')[-1].split('.')[0]) for fichier in chemin_csv]

    # Préparation des DataFrames
    for i in range(len(liste_dataframes)):
        df = liste_dataframes[i][0]
        name = liste_dataframes[i][1]
        df['t'] = pd.to_datetime(df['t'])  # Assurer la conversion
        df = cancel_stock_split(df, name)
        df = add_indicators(df)  # Ajouter les indicateurs
        df.set_index('t', inplace=True)  # Utiliser 't' comme index
        df.sort_index(inplace=True)  # Trier l'index
        df.drop_duplicates(inplace=True)  # Supprimer les doublons
        df.columns = [f'{col}__{i}' for col in df.columns]

        # Réaffecter le DataFrame modifié dans la liste
        liste_dataframes[i] = df

    combined_df = pd.concat(liste_dataframes, axis=1, join='outer')
    combined_df['t'] = combined_df.index
    combined_df.reset_index(drop=True, inplace=True)

    # Remplissage des données manquantes
    combined_df.ffill(inplace=True)
    combined_df.bfill(inplace=True)

    # Récupérer les noms de toutes les colonnes
    colonnes = combined_df.columns.tolist()

    colonnes_c = [col for col in colonnes if col.startswith('c__')]
    autres_colonnes = [col for col in colonnes if not col.startswith('c__')]

    # Nouvel ordre des colonnes
    nouvel_ordre_colonnes = colonnes_c + autres_colonnes

    # Réorganiser les colonnes dans le DataFrame
    combined_df = combined_df.reindex(columns=nouvel_ordre_colonnes)

    combined_df.to_csv("data/processed/DJIA/historical_data_bars_1H_DJIA_with_indicators.csv", index=False)

#df_AAPL = pd.read_csv('data/raw/historical_data_bars_1D_AAPL.csv')
#df_AAPL = add_indicators(df_AAPL)
#df_AAPL.to_csv("data/processed/historical_data_bars_1D_AAPL_with_indicators.csv", index=False)
build_feature_DJIA()