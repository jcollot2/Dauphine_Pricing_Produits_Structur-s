import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as pdr
from scipy import interpolate
from scipy.stats import norm
from datetime import datetime
from scipy.optimize import brentq


class Taux:
    def __init__(self, 
                 taux: float = None,
                 type_taux: str = "continu",
                 courbe_taux: dict = None,
                 type_interpol: str = "lineaire") -> None:
        
        """Initialise une instance de la classe Taux.

        Args:
            taux (float): Le taux d'intérêt fixe. Si None, utilise la courbe des taux pour le calcul.
            type_taux (str): Le type de taux ('continu' ou 'composé').
            courbe_taux (dict): Une courbe des taux représentée par un dictionnaire {Maturité: Taux}.
            type_interpol (str): Le type d'interpolation ('lineaire' ou 'cubique').

        Raises:
            Exception: Si le type d'interpolation n'est ni 'lineaire' ni 'cubique'.
        """
        self.__courbe_taux = courbe_taux
        self.__taux = taux
        self.__type_taux = type_taux
        self.__type_interpol = type_interpol

        if courbe_taux is not None:
            if self.__type_interpol not in ["linear", "cubic"]:
                raise Exception("Type d'interpolation incorrect")
            self.__interpol = interpolate.interp1d(
                list(courbe_taux.keys()),
                list(courbe_taux.values()),
                fill_value="extrapolate",
                kind=self.__type_interpol
            )
    
    def taux(self, maturite):
        """Retourne le taux d'intérêt pour une maturité donnée.

        Args:
            maturite: La maturité pour laquelle le taux est demandé.

        Returns:
            float: Le taux d'intérêt pour la maturité donnée.
        """
        if self.__taux is not None:
            return self.__taux
        return float(self.__interpol(maturite))
    
        
    def facteur_discount(self, maturite, taux_force: float = None):
        """Calcule le facteur de discount pour une maturité donnée.

        Args:
            maturite: La maturité pour laquelle le facteur de discount est calculé.
            taux_force (float): Un taux d'intérêt spécifique à utiliser à la place de celui calculé.

        Returns:
            float: Le facteur de discount pour la maturité donnée.
        """
        taux = self.taux(maturite)
        if taux_force is not None:
            taux = taux_force
            
        if self.__type_taux == "continu":
            return math.exp(-taux * maturite)
        elif self.__type_taux == "composé":
            return 1.0 / (1 + taux) ** maturite

class Maturite:
    def __init__(self, annees_maturite: float = None,
                 date_debut: datetime = None,
                 date_fin: datetime = None,
                 convention_compte_jours: str = "ACT/360") -> None:
        """Initialise une instance de la classe Maturite.

        Args:
            annees_maturite (float): La maturité en années.
            date_debut (datetime): La date de début pour calculer la maturité.
            date_fin (datetime): La date de fin pour calculer la maturité.
            convention_compte_jours (str): La convention pour le calcul du nombre de jours ('ACT/360', 'ACT/365').

        Raises:
            Exception: Si la convention de comptage des jours est inconnue.
        """
        self.convention_compte_jours = convention_compte_jours
        if annees_maturite is not None:
            self.annees_maturite = annees_maturite
        else:
            self.annees_maturite = (date_fin - date_debut).days / self.__denom()
            
    def __denom(self):
        """Calcule le dénominateur basé sur la convention de comptage des jours.

        Returns:
            int: Le dénominateur utilisé pour calculer la maturité en années.

        Raises:
            Exception: Si la convention de comptage des jours est inconnue.
        """
        if self.convention_compte_jours == "ACT/360":
            return 360
        elif self.convention_compte_jours == "ACT/365":
            return 365
        raise Exception("Convention de comptage des jours inconnue: " + self.convention_compte_jours)
   
    def maturite(self):
        """Retourne la maturité en années.

        Returns:
            float: La maturité en années.
        """
        return self.annees_maturite

class SimulationActif:
    def __init__(self, option_infos, taux: Taux, S0: float = 100, sigma: float = None, T: float = 10, nb_steps_par_an : int = None, nb_chemins: int = 10000, modele: str = "GBM",  use_iv=None):
        """
        Initialise une instance pour la simulation de prix d'actifs sous-jacents avec des ajustements pour la courbe des taux.

        Args:
            option_infos (OptionInfos): Instance de la classe OptionInfos pour accéder à la volatilité implicite.
            taux (Taux): Une instance de la classe Taux pour le calcul du drift à partir d'une courbe des taux.
            S0 (float): Le prix initial de l'actif.
            sigma (float): La volatilité annuelle de l'actif.
            T (float): La durée totale de simulation en années.
            nb_steps_par_an (int): Le nombre de simulations par an (2 pour semestriel, 4 pour trimestriel).
            nb_chemins (int): Le nombre de chemins à simuler.
            modele (str): Le modèle de simulation à utiliser ('GBM' pour le mouvement brownien géométrique).
            use_iv (bool): Indique si la volatilité implicite doit être utilisée (True) ou non (False).
        """
        if use_iv is True and sigma is not None:
            raise ValueError("Si 'use_iv' est True, 'sigma' ne doit pas être fourni.")
        if use_iv is False and sigma is None:
            raise ValueError("'sigma' doit être fourni si 'use_iv' est False.")
        
        self.option_infos = option_infos
        self.taux = taux
        self.S0 = S0
        self.sigma = sigma
        self.T = T
        self.nb_steps = nb_steps_par_an * T
        self.nb_chemins = nb_chemins
        self.dt = 1 / nb_steps_par_an 
        self.modele = modele
        self.use_iv = use_iv
        if use_iv:
            self.option_infos.create_iv_interpolation()


    def simuler(self):
        """
        Simule le prix de l'actif sous-jacent en utilisant le modèle de mouvement brownien géométrique.

        Returns:
            np.ndarray: Matrice des prix simulés de taille (nb_chemins, nb_steps * T + 1).
        """
        dt = self.dt
        prix = np.zeros((self.nb_chemins, self.nb_steps + 1))
        prix[:, 0] = self.S0
        taux_steps = np.zeros(self.nb_steps + 1)
        discount_factors = np.zeros(self.nb_steps + 1)
        
        for t in range(1, self.nb_steps + 1):
            maturite_actuelle = t * dt
            taux_actuel = self.taux.taux(maturite_actuelle)
            taux_steps[t] = taux_actuel
            if self.use_iv is True:
                # Utilise l'IV interpolée si use_iv est True
                self.sigma = self.option_infos.iv_interpolation(maturite_actuelle)
                sigma = self.sigma
                print(sigma)
            else:
                # Sinon, utilise une sigma fixe
                sigma = self.sigma
            
            if t == 1:  
                mean_S = self.S0
            else:
                mean_S = np.mean(prix[:, t-1])
            drift = taux_actuel - 0.5 * sigma ** 2
            shock = np.random.normal(0, 1, (self.nb_chemins,))
            prix[:, t] = mean_S * np.exp(drift * dt + sigma * np.sqrt(dt) * shock)  
            discount_factors[t] = math.exp(-taux_actuel * maturite_actuelle)
        
        return prix, taux_steps, discount_factors

class StockAnalyzer:
    def __init__(self, symbols):
        self.symbols = symbols

    def calculate_historical_volatility(self, symbol, period='1y', interval='1d'):
        # Récupération des données historiques
        data = yf.download(symbol, period=period, interval=interval)
        # Calcul des rendements logarithmiques
        log_returns = np.log(data['Close'] / data['Close'].shift(1))
        # Calcul de la volatilité historique
        volatility = log_returns.std() * np.sqrt(252)
        return volatility

    def get_spot_price(self, symbol):
        ticker = yf.Ticker(symbol)
        # Récupération des informations sur le titre
        data = ticker.history(period="1d")
        # Retourne le dernier prix de clôture
        return data['Close'].iloc[-1]

    def analyze_stocks(self):
        volatilities = {symbol: self.calculate_historical_volatility(symbol) for symbol in self.symbols}
        spot_prices = {symbol: self.get_spot_price(symbol) for symbol in self.symbols}

        return volatilities, spot_prices

class Autocall:
    def __init__(self, prix_simules, discount_factors, S0, nominal, coupon, barrier_up_initial_factor, barrier_down_factor, T, nb_steps_par_an, barrier_decrement_factor,dates):
        """
        Initialise l'autocall.
        
        Args:
            prix_simules (np.ndarray): Matrice des prix simulés de l'actif sous-jacent.
            discount_factors (np.ndarray): Matrice ou vecteur des facteurs de discount.
            S0 (float): Le prix initial de l'actif sous-jacent.
            nominal (float): Le nominal de l'Autocall.
            coupon (float): Le coupon annuel en tant que pourcentage.
            barrier_up_initial_factor (float): Facteur initial pour la barrière supérieure.
            barrier_down_factor (float): Facteur pour la barrière inférieure finale.
            T (float): Durée totale de simulation en années.
            nb_steps_par_an (int): Nombre de pas de temps par an.
            barrier_decrement_factor (float): Pourcentage de décrément annuel de la barrière supérieure.
        """
        self.prix_simules = prix_simules
        self.discount_factors = discount_factors
        self.S0 = S0
        self.nominal = nominal
        self.coupon = coupon
        self.barrier_up_initial_factor = barrier_up_initial_factor
        self.barrier_down_factor = barrier_down_factor
        self.T = T
        self.nb_steps_par_an = nb_steps_par_an
        self.barrier_decrement_factor = barrier_decrement_factor
        self.nb_steps = int(nb_steps_par_an * T)
        self.dates = dates

    def evol_barriere(self):
        barrier_up_ev = self.S0 * self.barrier_up_initial_factor
        barrier_down = self.S0 * self.barrier_down_factor
        evol_barrier_up_off = [barrier_up_ev]
        evol_barrier_down_off = [barrier_down for _ in range(self.nb_steps + 1)]
        
        
        for step in range(1, self.nb_steps+1):
            if (step > 0):
                #print(step)
                barrier_up_ev *= (1 - self.barrier_decrement_factor)
                evol_barrier_up_off.append(barrier_up_ev)
        return evol_barrier_up_off, evol_barrier_down_off
        
    
    def calculer_payoff(self, evol_barrier_up_off, evol_barrier_down_off):
        payoff = 0
        exercised = False
        
        # Vérifier si l'autocall est déclenché à chaque échéance
        for step in range(1, self.nb_steps +1 ):
            # Notez que step commence à 0 donc step + 1 pour l'indexation basée sur 1
            nb_coupons = step
            prix_moyen = np.mean(self.prix_simules[:, step])
            #print(prix_moyen)
            barrier_up_step = evol_barrier_up_off[step]
            
            # Condition d'autocall: le prix moyen est supérieur ou égal à la barrière UP
            if prix_moyen >= barrier_up_step:
                # Échéance actuelle basée sur l'indexation à partir de 0
                nb_coupons = step
                print(f"Nombre de coupon: {nb_coupons}")
                # Calcul du payoff à l'échéance où l'autocall est déclenché
                payoff = self.nominal * (1 + nb_coupons * self.coupon) * self.discount_factors[step]
                exercised = True
                break
        
        # Si l'autocall n'est pas déclenché, vérifier le prix moyen à la fin
        if not exercised:
            final_prix_moyen = np.mean(self.prix_simules[:, -1])
            if final_prix_moyen >= barrier_up_step:
                payoff = self.nominal * (1 + self.T * self.coupon) * self.discount_factors[-1]
            elif final_prix_moyen <= evol_barrier_down_off[-1]:
                variation = (final_prix_moyen - self.S0) / self.S0
                payoff = self.nominal * (1 + variation) * self.discount_factors[-1]
            else:  # Si le prix est entre les deux barrières à la fin
                payoff = self.nominal * self.discount_factors[-1]
                
        evolution_barriere_up, evolution_barriere_down, df_probabilite = self. calculer_proba_step(evol_barrier_up_off, evol_barrier_down_off)
        
        return payoff, self.nb_steps, evolution_barriere_up, evolution_barriere_down, df_probabilite


    def calculer_proba_step(self, evol_barrier_up_off, evol_barrier_down_off):
        
        # Préparation pour le calcul des probabilités
        prix_simules_df = pd.DataFrame(self.prix_simules.copy())
        # Initialisation du DataFrame pour stocker le compte des excédences et des probabilités
        df_probabilite = pd.DataFrame(index=range(0, self.nb_steps + 1))
        barrier_series = pd.Series(evol_barrier_up_off)
    
        for step in range(self.nb_steps + 1):
            # Correction: l'indexation dans DataFrame commence à 0, donc step - 1 pour accéder à la bonne colonne
            exceedances = prix_simules_df.iloc[:, step] > barrier_series.iloc[step]
            count = exceedances.sum()
            proba = count / len(prix_simules_df.iloc[:, step - 1])
    
            df_probabilite.at[step, 'Count_Exceedances'] = count
            df_probabilite.at[step, 'Probabilité d\'exécution'] = proba
        
        # Affichage des informations pour débogage et vérification
        #print('df_barriere : ', barrier_series)
        print()
        #print('prix_simules_df : ', prix_simules_df)
        print()
        #print('df_count_exceedances : ', df_probabilite)
        print()
        #print('longueur nombre de prix simulés : ', len(prix_simules_df.iloc[:,0]))  
        
        # Retourne l'évolution des barrières haute et basse, ainsi que le DataFrame des probabilités
        return evol_barrier_up_off, evol_barrier_down_off, df_probabilite
    
class OptionInfos:
    def __init__(self, symbol, taux):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self.taux = taux 
        
    def get_all_option_data(self):
        """
        Récupère les données des options pour toutes les dates d'expiration disponibles.
        
        :return: Un dictionnaire avec les dates d'expiration comme clés et les données d'options (calls et puts) comme valeurs.
        """
        options_data = {}
        # Récupère toutes les dates d'expiration disponibles
        expirations = self.ticker.options
        
        for expiration in expirations:
            opts = self.ticker.option_chain(expiration)
            options_data[expiration] = {
                'calls': opts.calls,
                'puts': opts.puts
            }
        return options_data
    
    def get_underlying_price(self):
        """
        Récupère le prix actuel de l'actif sous-jacent.
        """
        return self.ticker.history(period="1d")['Close'][0]
    
    def calc_implied_volatility(self, option_type, S, K, T, r, option_price):
        """
        Calcule la volatilité implicite en utilisant le modèle de Black-Scholes.
        """
        def bs_price(sigma):
            d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return price - option_price
        
        return brentq(bs_price, 0.001, 1, maxiter=1000)
    
    def create_iv_interpolation(self):
        """
        Crée une fonction d'interpolation pour les volatilités implicites basée sur les dates d'expiration disponibles.
        """
        options_data = self.get_all_option_data()
        S = self.get_underlying_price()
        expirations_dates = []
        ivs = []

        for expiration in options_data:
            T = (datetime.strptime(expiration, '%Y-%m-%d') - datetime.now()).days / 365
            r = self.taux.taux(T)
            for _, row in options_data[expiration]['calls'].iterrows():
                try:
                    iv = self.calc_implied_volatility('call', S, row['strike'], T, r, row['lastPrice'])
                    expirations_dates.append(T)
                    ivs.append(iv)
                except ValueError:
                    continue
        
        # Assurez-vous que les dates et les IVs sont triées
        sorted_indices = np.argsort(expirations_dates)
        expirations_dates = np.array(expirations_dates)[sorted_indices]
        ivs = np.array(ivs)[sorted_indices]
        
        # Créez la fonction d'interpolation
        self.iv_interpolation = interpolate.interp1d(expirations_dates, ivs, kind='linear', fill_value='extrapolate')
    
    def plot_iv_surface(self, plot_type='calls'):
        """
        Construit et trace la surface de volatilité implicite pour les calls ou les puts.
        """
        options_data = self.get_all_option_data()
        S = self.get_underlying_price()
        
        strikes = []
        expirations = []
        ivs = []
        
        for expiration in options_data:
            for index, row in options_data[expiration][plot_type].iterrows():
                T = (datetime.strptime(expiration, '%Y-%m-%d') - datetime.now()).days / 365
                r = self.taux.taux(T)
                
                try:
                    iv = self.calc_implied_volatility(plot_type[:-1], S, row['strike'], T, r, row['lastPrice'])
                    strikes.append(row['strike'])
                    expirations.append(T)
                    ivs.append(iv)
                except ValueError:
                    continue
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(expirations, strikes, ivs, c=ivs, cmap='viridis', marker='o')
        
        ax.set_xlabel('Temps jusqu\'à expiration (années)')
        ax.set_ylabel('Prix d\'exercice')
        ax.set_zlabel('Volatilité Implicite')
        
        plt.title('Surface de Volatilité Implicite des ' + plot_type.capitalize())
        plt.show()
        
        fig1 = self.plot_iv_surface_interpolate(plot_type)
        
        return expirations, strikes, ivs ,(fig,fig1)
    
    def plot_iv_surface_interpolate(self, plot_type='calls'):
        """
        Construit et trace la surface de volatilité implicite pour les calls ou les puts, y compris l'interpolation.
        """
        self.create_iv_interpolation()
        S = self.get_underlying_price()
        expirations = np.linspace(0.01, 3, 100)  # Exemple: échéances de 0.01 à 3 ans
        strikes = np.linspace(S*0.5, S*1.5, 100)  # Exemple: strikes de 50% à 150% du S0
        
        X, Y = np.meshgrid(expirations, strikes)
        Z = self.iv_interpolation(X)

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(X, Y, Z, cmap='viridis')
        
        ax.set_xlabel('Temps jusqu\'à expiration (années)')
        ax.set_ylabel('Prix d\'exercice')
        ax.set_zlabel('Volatilité Implicite')
        
        plt.title(f'Surface de Volatilité Implicite des {plot_type.capitalize()}')
        plt.show()
        return fig

class CourbeTauxAméricaine:
    def __init__(self, start_date, end_date=None):
        """Initialise la classe avec des dates de début et de fin pour la récupération des données."""
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now()
        self.series_ids = ['DGS1', 'DGS2', 'DGS5', 'DGS10', 'DGS30']
        self.courbe_taux = {}

    def recuperer_taux(self):
        """Récupère les dernières valeurs des taux d'intérêt pour les maturités définies."""
        try:
            df = pdr.DataReader(self.series_ids, 'fred', self.start_date, self.end_date)
            # Stocker la dernière valeur disponible pour chaque série
            for series_id in self.series_ids:
                # Note: Cela suppose que les données les plus récentes sont à la fin
                self.courbe_taux[series_id] = df[series_id].iloc[-1]
        except Exception as e:
            print(f"Erreur lors de la récupération des données: {e}")
        return self.courbe_taux

    def courbe_comme_dict(self):
        """Convertit la courbe des taux récupérée en un dictionnaire plus simple."""
        maturite_mapping = {
            'DGS1': 1,
            'DGS2': 2,
            'DGS5': 5,
            'DGS10': 10,
            'DGS30': 30
        }
        return {maturite_mapping[key]: value / 100 for key, value in self.courbe_taux.items() if value is not np.nan}


