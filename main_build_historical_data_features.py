import pandas as pd
import os
import glob

from features.build_features import add_indicators

def build_feature_DJIA():
    chemin_csv = glob.glob(os.path.join("data/raw/DJIA", '*.csv'))

    # Lire chaque CSV en DataFrame et les stocker dans une liste
    liste_dataframes = [pd.read_csv(fichier) for fichier in chemin_csv]

    # Préparation des DataFrames
    for i in range(len(liste_dataframes)):
        df = liste_dataframes[i]
        df['t'] = pd.to_datetime(df['t'])  # Assurer la conversion
        df = add_indicators(df)  # Ajouter les indicateurs
        df.set_index('t', inplace=True)  # Utiliser 't' comme index
        df.sort_index(inplace=True)  # Trier l'index
        df.drop_duplicates(inplace=True)  # Supprimer les doublons

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

    # Extraire les noms des colonnes commençant par 'c'
    colonnes_c = [col for col in colonnes if col == 'c']

    # Autres colonnes
    autres_colonnes = [col for col in colonnes if col != 'c']

    # Nouvel ordre des colonnes
    nouvel_ordre_colonnes = colonnes_c + autres_colonnes

    # Réorganiser les colonnes dans le DataFrame
    combined_df = combined_df.reindex(columns=nouvel_ordre_colonnes)

    combined_df.to_csv("data/processed/DJIA/historical_data_bars_1H_DJIA_with_indicators.csv")

#df_AAPL = pd.read_csv('data/raw/historical_data_bars_1D_AAPL.csv')
#df_AAPL = add_indicators(df_AAPL)
#df_AAPL.to_csv("data/processed/historical_data_bars_1D_AAPL_with_indicators.csv", index=False)
build_feature_DJIA()