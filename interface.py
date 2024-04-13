import streamlit as st
import pricer as prc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime



def organisation():
    # Classe Taux
    st.subheader('**Classe : Taux**')
    st.write('**Objectif** : Gère les calculs des taux d\'intérêt.')
    st.write('**Méthodes** :')
    st.write("""
    - `__init__` : Initialise une instance de `Taux` avec un taux d'intérêt spécifique ou une courbe de taux pour les calculs.
    - `taux` : Retourne le taux d'intérêt pour une échéance donnée.
    - `facteur_discount` : Calcule le facteur d'actualisation pour une échéance spécifiée, en utilisant éventuellement un taux d'intérêt spécifique.
    """)
    st.write('\n')


    # Classe Maturite
    st.subheader('**Classe : Maturite**')
    st.write('**Objectif** : Gère les calculs relatifs aux durées de maturité basées sur différentes conventions de comptage des jours.')
    st.write('**Méthodes** :')
    st.write("""
    - `__init__` : Initialise une instance de maturité avec des dates de début et de fin spécifiques et une convention de comptage des jours.
    - `__denom` : Calcule le dénominateur en fonction de la convention de comptage des jours pour calculer la maturité en années.
    - `maturite` : Calcule et retourne la maturité en années.
    """)
    st.write('\n')

    # Classe SimulateurAutocall
    st.subheader('**Classe : SimulateurAutocall**')
    st.write('**Objectif** : Simule des produits structurés autocallables dans le temps.')
    st.write('**Méthodes** :')
    st.write("""
    - `__init__` : Initialise le simulateur avec des paramètres pour les simulations de Monte Carlo et les caractéristiques du produit.
    - `evol_barriere` : Gère l'évolution des niveaux de barrière tout au long de la période de simulation.
    - `calculer_payoff` : Calcule le paiement du produit structuré en fonction des scénarios simulés.
    - `calculer_proba_step` : Calcule la probabilité pour certaines conditions à chaque étape de la simulation.
    """)
    st.write('\n')

    # Classe OptionInfos
    st.subheader('**Classe : OptionInfos**')
    st.write('**Objectif** : Gère la récupération des données d\'options et les calculs pour les volatilités implicites.')
    st.write('**Méthodes** :')
    st.write("""
    - `__init__` : Constructeur pour initialiser le gestionnaire des données d'options.
    - `get_all_option_data` : Récupère les données d'options pour toutes les dates d'expiration disponibles.
    - `get_underlying_price` : Récupère le prix actuel de l'actif sous-jacent.
    - `calc_implied_volatility` : Calcule la volatilité implicite en utilisant le modèle Black-Scholes.
    - `create_iv_interpolation` : Crée une fonction d'interpolation pour les volatilités implicites.
    - `plot_iv_surface` : Construit et trace la surface de volatilité implicite.
    """)
    st.write('\n')

    # Classe CourbeTauxAméricaine
    st.subheader('**Classe : CourbeTauxAméricaine**')
    st.write('**Objectif** : Gère la récupération et la manipulation des courbes de taux américaines.')
    st.write('**Méthodes** :')
    st.write("""
    - `__init__` : Initialise la classe avec des paramètres pour la récupération des données.
    - `recuperer_taux` : Récupère les dernières valeurs des taux d'intérêt pour les échéances définies.
    - `courbe_comme_dict` : Convertit la courbe de taux récupérée en un format de dictionnaire plus simple.
    """)

def travaux(tickers,T,nb_steps_par_an, nb_chemins,modele, symbol, nominal, coupon, barrier_up_initial_factor, barrier_down_factor, barrier_decrement_factor,response):
    # Liste des tickers à analyser
    analyzer = prc.StockAnalyzer(tickers)
    volatilities, spot_prices = analyzer.analyze_stocks()

    st.write("**Prix spot:**")
    st.write(spot_prices)

    start = datetime(2024, 4, 1)  # Date de début pour la récupération des données
    courbe_taux = prc.CourbeTauxAméricaine(start)
    taux_actuels = courbe_taux.recuperer_taux()
    courbe_taux_simple = courbe_taux.courbe_comme_dict()

    # Remplacer votre courbe_taux_exemple par courbe_taux_simple
    st.write("**Courbe de taux officielle**")
    st.write(courbe_taux_simple)

    # Créer des instances de Maturite et définir la courbe des taux
    courbe_taux_exemple = {
        1: 0.01,   # 1.0% pour une maturité de 1 an
        2: 0.015,  # 1.5% pour une maturité de 2 ans
        5: 0.02,   # 2.0% pour une maturité de 5 ans
        10: 0.025, # 2.5% pour une maturité de 10 ans
        15: 0.03   # 3.0% pour une maturité de 15 ans
    }
    taux = prc.Taux(courbe_taux=courbe_taux_simple, type_interpol="linear")

    # Demande à l'utilisateur
    user_input = response.strip().lower()
    use_market_data = user_input == "oui"

    # Configuration de la simulation
    option_infos = prc.OptionInfos(symbol, taux)

    # Récupération des données d'options pour toutes les dates d'expiration disponibles
    options_data = option_infos.get_all_option_data()
    # Tracer la surface de volatilité implicite pour les calls
    expirations, strikes, ivs,figures = option_infos.plot_iv_surface(plot_type='calls')
    fig,fig1 = figures
    st.write("**Surface de volatilités discrète**")
    st.pyplot(fig)
    st.write("**Surface de volatilités interpolé**")
    st.pyplot(fig1)


    if use_market_data:
            # Utilise les données de marché (prix spot et volatilité implicite)
            S0 = spot_prices[symbol]
            use_iv = True  # Utilise la volatilité implicite
            sigma = None  # La valeur de sigma sera déterminée par la volatilité implicite
    else:
            # Utilise des valeurs prédéfinies
            S0 = 100
            sigma = 0.2  # Valeur fixe pour sigma
            use_iv = False  # Ne pas utiliser la volatilité implicite

        
    # Simulation avec sigma fixe
    simulation = prc.SimulationActif(option_infos, taux, S0, sigma, T, nb_steps_par_an, nb_chemins, modele, use_iv)
    prix_simules, taux_simules, discount_factors = simulation.simuler()

    # Convertir les résultats en DataFrame
    dates = pd.date_range(start=datetime.today(), periods=T * nb_steps_par_an + 1, freq='Q')
    df = pd.DataFrame(index=dates)
    df['Stock Price'] = prix_simules.mean(axis=0)  # Moyenne sur les chemins simulés
    df['Taux'] = taux_simules
    df['Discount Factor'] = discount_factors

    st.write("Courbe des taux interpolé")
    fig, ax = plt.subplots()
    ax.plot(dates,prix_simules)
    st.pyplot(fig)

    # Sélectionnez le nombre de chemins à tracer
    nb_chemins_a_tracer = nb_chemins
    temps = range(prix_simules.shape[1])

    fig, ax = plt.subplots()
    plt.figure(figsize=(14, 8))
    nb_chemins_a_tracer = min(nb_chemins_a_tracer, prix_simules.shape[0])

    # Tracer chaque chemin sélectionné
    for i in range(nb_chemins_a_tracer):
        ax.plot(temps, prix_simules[i, :], lw=1, alpha=0.8)

    plt.title('Simulation de Monte Carlo de prix d\'actif')
    plt.xlabel('Temps')
    plt.ylabel('Prix de l\'actif')
    plt.grid(True)
    plt.show()
    st.pyplot(fig)

    # Création et évaluation de l'Autocall
    autocall = prc.Autocall(
        prix_simules=prix_simules, 
        discount_factors=discount_factors, 
        S0=S0, 
        nominal=nominal, 
        coupon=coupon, 
        barrier_up_initial_factor=barrier_up_initial_factor, 
        barrier_down_factor=barrier_down_factor, 
        T=T, 
        nb_steps_par_an=nb_steps_par_an, 
        barrier_decrement_factor=barrier_decrement_factor,
        dates=dates
    )

    # Calcul de l'évolution de la barrière up avant de l'utiliser dans calculer_payoff
    evol_barrier_up_off,  evol_barrier_down_off = autocall.evol_barriere()
    valeur_autocall, nb_steps, evol_barriere_up, evol_barriere_down, df_probabilite = autocall.calculer_payoff(evol_barrier_up_off, evol_barrier_down_off)

    # Ajouter l'évolution de la barrière up et down au DataFrame
    df["Barriere UP"] = evol_barriere_up
    df["Barriere Down"] = evol_barriere_down
    # Commande supplémentaire pour MaJ le df avec activation de la Barriere Down que à la derniere échéance
    #df["Barriere Down"].iloc[:-1] = None
    # Ajouter la probabilité d'éxécution à chaque échéance
    df["Probabilité d'exécution"] = df_probabilite["Probabilité d'exécution"].to_list()
    df["Nombre de dépassement"] = df_probabilite["Count_Exceedances"].to_list()

    df["Echeance"] = [i for i in range(len(df))]
    step = df[df["Stock Price"]>df["Barriere UP"]].Echeance.min()
    if step is np.nan:
        step = 0
    st.write(f"Échéances d'autocall (première occurrence ou finale si non autocallé) : {step}")
    st.write(f"Valeur moyenne de l'Autocall: {valeur_autocall}")

        
    # Plot le DF - Autocall 
    df = df.reset_index()
    df = df.rename(columns={'index': 'Date'})
    fig, ax = plt.subplots()
    plt.figure(figsize=(14,7))

    # Barriere UP
    ax.plot(df['Date'], df['Barriere UP'], label='Barriere UP', color='green', marker='o')

    # Barriere Down
    ax.plot(df['Date'], df['Barriere Down'], label='Barriere Down', color='red', marker='o')

    # Stock Price
    ax.plot(df['Date'], df['Stock Price'], label='Stock Price', color='blue', marker='o')

    # Titles and labels
    plt.title('Autocall contract over time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()

    # Improve layout and grid visibility
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()
    st.pyplot(fig)







# initialisation du streamlit 
logo = r".\\logo.png"
st.title("Structuration de Produits Autocallables")
st.sidebar.image(logo, width=250)
selected_tab = st.sidebar.radio("Sommaire",["Dashboard", "About the projet"])


# Dashboard
if selected_tab == "Dashboard":
    ## paramètres
    st.header("Paramètres")
    col1, col2, col3 = st.columns([33, 33, 33])
    with col1:
        tickers = st.text_input("Tickers", "AAPL")
    with col2:
        T = st.number_input("Maturité", min_value=1, max_value=10, step=1, value=2)
    with col3 :
        nb_steps_par_an = st.number_input("Nombre de steps par an", min_value=1, max_value=10, step=1, value=4)
    col1, col2, col3 = st.columns([33, 33, 33])
    with col1:
        nb_chemins = st.number_input("Nombre de simulation", min_value=1, max_value=10000, step=1, value=1000)
    with col2:
        nominal = st.number_input("Nominal", min_value=1,value=1000)
    with col3:
        coupon = st.number_input("Coupon en pourcentage", min_value=0, max_value=100,step = 1, value=5)/100
    col1, col2, col3 = st.columns([33, 33, 33])
    with col1:
        barrier_up_initial_factor = st.number_input("Barrière up initiale en pourcentage", min_value=0, max_value=1000,step = 1, value=110)/100  # Barrière supérieure initiale à 110% de S0
    with col2:
        barrier_down_factor = st.number_input("Barrière down initiale en pourcentage", min_value=0, max_value=1000,step = 1, value=90)/100   # Barrière inférieure finale à 90% de S0
    with col3:
        barrier_decrement_factor = st.number_input("Barrière up initiale en pourcentage", min_value=0, max_value=100,step = 1, value=1)/100   # Décrément annuel de la barrière supérieure de 2%
    modele = "GBM"
    symbol = tickers
    if st.button('Run'):
        st.header("Travaux")
        travaux(['AAPL', 'MSFT', 'GOOGL'],2,4,1000,"GBM","AAPL", 1000 ,0.05 ,1.1 , 0.9,0.01,"oui")



        
elif selected_tab == "About the projet":
    st.header('Sujet')
    st.write('\n')
    st.write("Notre projet vise à développer un tableau de bord pour la structuration et la valorisation de produits financiers autocallables. L'objectif est de permettre une gestion personnalisée des produits financiers sur un ou plusieurs sous-jacents grâce à la modification de paramètres. Ce tableau de bord intégre des fonctionnalités pour définir les paramètres de calcul, les données de marché lui permettant de fournir des indicateurs clés tels que le prix du produit, ses sensibilités, et la probabilité d’exercice...")
    st.header('Solutions')
    st.write('\n')
    st.header('Organisation')
    organisation()
