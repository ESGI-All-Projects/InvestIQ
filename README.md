# InvestIQ

## I. Pour reproduire les expérimentations

### A. Installer les packages necessaires

Créer votre environnement virtuel python et lancer la commande
``pip install -r requirements.txt``

### B. Télécharger les données historiques
1. Créer un compte Alpaca et récupérer une paire de clé/secret
2. Créer le fichier .env à la source du projet de la forme suivante
``API_KEY="XXXXXX"
SECRET_KEY="XXXXX"
``
3. Lancer le script **main_get_historical_data.py** qui va télécharger toutes les données des stocks DJIA et AAPL

### C. Création les features
Lancer le scrypt **main_build_historical_data_features.py** qui va créer toutes les features pour les actions DJIA et AAPL

### D. Lancer les entraiements des modèles
Lancer le sript **main_train_model.py** en modifiant l'appel a la fonction en fonction du choix de l'expérimentation choisie
- train_LSTM() | entrainement du modèle LSTM pour les séries temporelles
- train_PPO_window() | entrainement du modèle RL qui prend en entrée les 30 dernières valeurs des prix
- train_PPO_indicators() | entrainement du modèle RL qui prend en entrée la dernière valeur du prix + les indicateurs détaillés dans le rapport
- train_PPO_MultiActions() | entrainement du modèle RL sur toutes les actions DJIA