# Pricing d'autocall

**Auteurs :** Alexandre Guérin - Jeanne Collot - Arthur Frachon - Alice Munier - Milo Seguin  
**Cadre :** Dauphine - cours de Produits Structurés  
**Date :** 2024  


## 📌 Sujet
Ce projet est un outil conçu pour simuler et évaluer des autocalls. Il inclut un pricer développé en Python ainsi qu'une interface utilisateur pour rendre l'application accessible et intuitive. Pour en savoir plus, consultez le sujet 2 du fichier "Sujet_PricingStructures_2024.pdf" 

Le projet est structuré autour de deux composants principaux :
- **`autocall_pricer.py`** : Le cœur du projet, un pricer permettant de modéliser et d'évaluer les rendements d'autocall.
- **`interface.py`** : Une interface utilisateur interactive qui permet de visualiser les calculs et les résultats.


## 🎯 Objectifs
- Simulation des flux financiers liés aux autocalls.
- Paramétrage des produits structurés (barrières, coupons, etc.).
- Affichage des résultats via une interface graphique.


## 🔍 Structure du projet
Le projet contient les fichiers suivants :
- **`autocall_pricer.py`** : Script Python pour le calcul des prix des autocalls.
- **`interface.py`** : Script Python pour l'interface utilisateur.
- **`Logo.png`** : Logo du projet pour l'interface.
- **`__pycache__`** : Dossier généré automatiquement pour les fichiers compilés.

## 📉 Méthode utilisée 
Le pricer est basé sur la méthode **Monte Carlo**, qui est particulièrement adaptée pour modéliser la valeur des produits structurés complexes tels que les autocalls. 
#### 1. Simulation des scénarios de marché
   - Génération de milliers de trajectoires pour les sous-jacents financiers (actions, indices, etc.) à l'aide de processus stochastiques, comme le modèle de Black-Scholes ou des variantes (ex. modèles avec volatilité stochastique).
   
#### 2. Calcul des flux financiers
   - Évaluation des paiements des coupons en fonction des conditions du produit (barrières, rendement minimum, etc.).
   - Identification des points de sortie anticipée (autocall).

#### 3. Actualisation des flux
   - Tous les flux financiers futurs sont actualisés au taux sans risque pour obtenir leur valeur présente.

#### 4. Calcul de la valeur finale
   - La moyenne des résultats des simulations donne la valeur estimée du produit structuré.

---
