import pandas as pd

from API.account import get_account
from user_metrics.database_connexion import get_connexion
"""
Objectif:
Calculer toutes les informations necessaires pour pouvoir les affichers à l'utilisateur
nécessite des appels à Alpalca + le maintiens d'une base de données
1. a chaque action de l'IA (on va dire chaque heure) on doit calculer l'évolution du portfolio en pourcentage
2. Puis calculer pour chaque utilisateur le nouveau solde et l'ajouter à une liste pour garder un historique
On aura pas une actualisation en temps réel mais par heure

Attention : On ne peut pas ajouter ou retirer de l'argent sur Alpaca, il va falloir simuler notre propre wallet
tout en utilisant l'évolution des actions avec Alpaca
"""

# def calcul_portfolio_next_ratio(last_portfolio_value):
#     account_info = get_account()
#     equity = account_info['equity']
#
#     return equity/last_portfolio_value

def update_portfolio(current_portfolio_value, user_id, amount):
    conn = get_connexion()
    cur = conn.cursor()

    # On récupère la dernière valeur en date
    query = """
        SELECT wallet
        FROM wallet_table
        ORDER BY date_colonne DESC
        LIMIT 1;
    """
    cur.execute(query)

    # Récupération du résultat
    last_portfolio_value = cur.fetchone()
    ratio = current_portfolio_value/last_portfolio_value

    # Insert nouvelle valeur
    query = f"""
        INSERT INTO wallet_table (user, depot, wallet, gain, date)
        VALUES ({user_id}, {amount}, {current_portfolio_value + amount}, {ratio}, NOW());
    """
    cur.execute(query)
    conn.commit()

    cur.close()
    conn.close()



def update_user_portfolio():
    """


    """
