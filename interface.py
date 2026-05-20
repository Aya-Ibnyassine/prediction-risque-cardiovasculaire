import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# ---------------- Chargement du modèle & scaler avec cache ----------------
@st.cache_resource
def charger_ressources():
    """Charge toutes les ressources nécessaires à l'application.
    Cette fonction est mise en cache pour ne s'exécuter qu'une seule fois."""
    try:
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        df = pd.read_csv("heart_clean.csv")
        reco_df = pd.read_excel("recommendations.xlsx")
        feature_names = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                         'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
        X = df.drop("HeartDisease", axis=1)
        y = df["HeartDisease"]
        return model, scaler, df, reco_df, feature_names, X, y
    except FileNotFoundError as e:
        st.error( #Affiche un message d'erreur rouge sur l'interface Streamlit
            f"Fichier manquant. Assurez-vous que tous les fichiers nécessaires sont dans le même dossier. Erreur: {e}")
        st.stop() # Arrête l'exécution de l'application.

#Exécute la fonction load_resources() et assigne les valeurs retournées aux variables correspondantes.
model, scaler, df, reco_df, feature_names, X, y = charger_ressources()

# ---------------- Configuration & design ----------------
st.set_page_config(page_title="Évaluation Risque Cardiaque", layout="wide")

def add_bg_from_local(image_file):  #Définit une fonction pour ajouter une image de fond.
    with open(image_file, "rb") as f: #Ouvre le fichier image en mode binaire ("rb")
        data = f.read()
    encoded = base64.b64encode(data).decode() #Encode les données binaires de l'image en chaîne de caractères Base64.
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;       #Redimensionne l'image pour qu'elle couvre toute la zone, sans déformation
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("background.PNG") #Appelle la fonction pour appliquer l'image de fond.

st.markdown("""
    <style>
    body, .stApp { color: black; }
    
    label p, label {
    font-size: 15px !important;
    font-weight: bold !important; 
    font-family: "Arial", sans-serif ;
    color: black ;
}
    .stButton > button {
    background-color: #f0f2f6;       
    color: black;
     border: 2px solid red;
     border-radius: 100px;
}
     p {font-size: 17px !important;}
     h1 {
    text-align: center;
}
    </style>
""", unsafe_allow_html=True)

st.title("❤️ Application de Prédiction de Maladies Cardiaques ❤️")

# ---------------- Initialisation des états ----------------
if "role" not in st.session_state:
    st.session_state.role = "accueil"
if "patient_inputs" not in st.session_state:
    st.session_state.patient_inputs = None
if "patient_prob" not in st.session_state:
    st.session_state.patient_prob = None
if "patient_pred" not in st.session_state:
    st.session_state.patient_pred = None
if "historique" not in st.session_state:
    st.session_state.historique = []
if "med_pred" not in st.session_state:
    st.session_state.med_pred = None
if "med_prob" not in st.session_state:
    st.session_state.med_prob = None
if "show_id_input" not in st.session_state:
    st.session_state.show_id_input = False


# Valeurs par défaut du formulaire pour la réinitialisation
DEFAULT_FORM_VALUES = {
    "age": 30, "sex": "Homme", "chest_pain": "ATA", "resting_bp": 120,
    "cholesterol": 200, "fasting_bs": "Non", "ecg": "Normal",
    "max_hr": 150, "angina": "Non", "oldpeak": 1.0, "slope": "Up"
}

# ---------------- Fonction pour les boutons de la barre latérale ----------------
def sidebar_buttons():
    if st.session_state.role != "accueil":  #Les boutons ne s'affichent que si l'utilisateur n'est pas sur la page d'accueil.
        with st.sidebar:
            st.title("Navigation")
            if st.button("🔀 Changer de profil"):
                st.session_state.clear()
                st.session_state.role = "accueil"
                st.rerun()

            if st.button("🔄 Réinitialiser le formulaire"):
                # On doit d'abord sauvegarder les variables qu'on ne veut pas perdre (le rôle et l'historique).
                historique_temp = st.session_state.get("historique", [])
                role_temp = st.session_state.get("role")

                st.session_state.clear()

                st.session_state.role = role_temp
                st.session_state.historique = historique_temp

                # Réinitialise les valeurs des widgets dans st.session_state
                for key, value in DEFAULT_FORM_VALUES.items():
                    st.session_state[key] = value

                st.rerun()


sidebar_buttons()


# ---------------- Fonction input utilisateur ----------------
def user_inputs():
    with st.expander("Cliquez ici pour masquer ou afficher le formulaire"):

            age = st.slider("Âge", 18, 100, 30, key="age")
            sex = st.selectbox("Sexe", ["Femme", "Homme"], key="sex")
            chest_pain = st.selectbox("Type de douleur thoracique", ["ATA", "NAP", "ASY", "TA"], key="chest_pain")
            resting_bp = st.slider("Pression artérielle au repos", 80, 220, 120, key="resting_bp")
            cholesterol = st.slider("Cholestérol", 50, 600, 200, key="cholesterol")
            fasting_bs = st.selectbox("Glycémie à jeun : > 120 mg/dl ?", ["Non", "Oui"], key="fasting_bs")
            ecg = st.selectbox("ECG au repos", ["Normal", "ST", "LVH"], key="ecg")
            max_hr = st.slider("Fréquence cardiaque max atteinte pendant l'effort", 60, 250, 150, key="max_hr")
            angina = st.selectbox("Angine induite par effort", ["Non", "Oui"], key="angina")
            oldpeak = st.slider("Oldpeak (dépression du segment ST par rapport au repos)", 0.0, 6.0, step=0.1, key="oldpeak")
            slope = st.selectbox("Pente du segment ST à l'effort", ["Up", "Flat", "Down"], key="slope")

    input_dict = {
        "Age": age,
        "Sex": 1 if sex == "Homme" else 0,
        "ChestPainType": {"ATA": 1, "NAP": 2, "ASY": 0, "TA": 3}[chest_pain],
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == "Oui" else 0,
        "RestingECG": {"Normal": 1, "ST": 2, "LVH": 0}[ecg],
        "MaxHR": max_hr,
        "ExerciseAngina": 1 if angina == "Oui" else 0,
        "Oldpeak": oldpeak,
        "ST_Slope": {"Up": 2, "Flat": 1, "Down": 0}[slope]
    }
    return pd.DataFrame([input_dict])

# ---------------- Interface Patient ----------------
def interface_patient():
    st.header("👤 Interface Patient")

    inputs_df = user_inputs()
    st.write("Vos données saisies :")
    st.dataframe(inputs_df)

    if st.button("1️⃣ Prédire le risque"):
        scaled_inputs = scaler.transform(inputs_df)
        st.session_state.patient_prob = model.predict_proba(scaled_inputs)[0][1] * 100
        st.session_state.patient_pred = model.predict(scaled_inputs)[0]
        st.session_state.patient_inputs = inputs_df

    if st.session_state.get("patient_pred") is not None:
        if st.session_state.patient_pred == 1:
            st.error(f"⚠️ Risque détecté (Probabilité : {st.session_state.patient_prob:.2f}%)")
        else:
            st.success(f"✅ Aucun risque détecté (Probabilité : {st.session_state.patient_prob:.2f}%)")

        st.markdown("---")
        with st.container(border=True):
            st.subheader("Explorez d'autres informations")

            col_opt1, col_opt2 = st.columns(2)

            with col_opt1:
                if st.button("2️⃣ Comparer aux normes"):
                    age_patient = st.session_state.patient_inputs["Age"].iloc[0]
                    normes = {"RestingBP": 120, "Cholesterol": 200, "MaxHR": 220 - age_patient, "Oldpeak": 0.0}
                    patient_values = st.session_state.patient_inputs[list(normes.keys())].iloc[0]
                    compare_df = pd.DataFrame({
                        "Feature": list(normes.keys()),
                        "Patient": patient_values.values,
                        "Normale": list(normes.values())
                    })
                    fig, ax = plt.subplots(figsize=(8, 5))
                    compare_df.set_index("Feature").plot(kind="bar", ax=ax)
                    plt.title("Comparaison Patient vs Normes Médicales")
                    st.pyplot(fig)

            with col_opt2:
                if st.button("3️⃣ Voir recommandations"):
                    recos = reco_df[(reco_df["risk_percentage_min"] <= st.session_state.patient_prob) &
                                    (reco_df["risk_percentage_max"] >= st.session_state.patient_prob)]
                    if not recos.empty:
                        st.subheader("💡 Recommandations :")
                        st.dataframe(recos[["category", "recommendation_fr", "details_fr"]])
                    else:
                        st.warning("⚠️ Aucune recommandation trouvée pour ce niveau de risque.")


# Nom du fichier CSV pour l'historique
HISTORIQUE_FILE = "historique_patients.csv"
# ---------------- Interface Médecin ----------------
def medecin_interface():
    st.subheader("🩺 Interface Médecin")

    # Charger le DataFrame
    try:
        historique_df = pd.read_csv("historique_patients.csv")
    except FileNotFoundError:
        st.warning("Fichier historique_patients.csv non trouvé.")
        historique_df = pd.DataFrame(
            columns=["ID", "Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "Prediction", "Probabilité (%)"])

    st.session_state.historique_df = historique_df
    if not historique_df.empty:
        historique_df["ID"] = historique_df["ID"].astype(str)

    # --- Section de saisie et de prédiction ---
    inputs_df = user_inputs()

    # Bouton de prédiction pour l'interface médecin
    if st.button("Prédire le risque"):
        scaled_inputs = scaler.transform(inputs_df)
        st.session_state.med_pred = model.predict(scaled_inputs)[0]
        st.session_state.med_prob = model.predict_proba(scaled_inputs)[0][1] * 100

    if st.session_state.get("med_pred") is not None:
        if st.session_state.med_pred == 1:
            st.error(f"⚠️ Risque détecté (Probabilité : {st.session_state.med_prob:.2f}%)")
        else:
            st.success(f"✅ Aucun risque détecté (Probabilité : {st.session_state.med_prob:.2f}%)")

        # Le bouton d'ajout n'apparaît que si une prédiction médecin a été faite
        if st.button("Ajouter ce patient à l'historique"):
            st.session_state.show_id_input = True

    # --- Section de saisie d'ID et confirmation ---
    if st.session_state.show_id_input:
        st.markdown("---")
        with st.container(border=True):
            patient_id = st.text_input("Saisissez un identifiant pour le patient :")
            if st.button("Confirmer l'ajout"):
                if patient_id:
                        # Crée un nouveau DataFrame pour le patient
                        nouveau_patient_df = pd.DataFrame([{
                            "ID": patient_id,
                            "Age": inputs_df["Age"].iloc[0],
                            "RestingBP": inputs_df["RestingBP"].iloc[0],
                            "Cholesterol": inputs_df["Cholesterol"].iloc[0],
                            "MaxHR": inputs_df["MaxHR"].iloc[0],
                            "Oldpeak": inputs_df["Oldpeak"].iloc[0],
                            "Prediction": int(st.session_state.med_pred),
                            "Probabilité (%)": round(st.session_state.med_prob, 2)
                        }])

                        # Ajoute le nouveau patient au DataFrame de l'historique
                        st.session_state.historique_df = pd.concat([st.session_state.historique_df, nouveau_patient_df],
                                                                   ignore_index=True)
                        st.session_state.historique_df.to_csv("historique_patients.csv", index=False)

                        st.success(f"Patient '{patient_id}' ajouté à l'historique avec succès.")
                        st.session_state.show_id_input = False  # Masque le champ de saisie
                else:
                    st.warning("⚠️ Veuillez saisir un identifiant avant de confirmer.")

    # --- Section Outils d'analyse et historique ---
    st.markdown("---")
    st.subheader("Outils d'analyse et historique")

    # Affichage de l'historique complet
    with st.expander("Historique", expanded=False):
     if not st.session_state.historique_df.empty:
        if st.button("📊 Afficher la liste de l'historique"):
            st.write("### Historique complet des patients")
            st.dataframe(st.session_state.historique_df)
     else:
        st.info("Aucun historique disponible. Ajoutez un patient pour commencer.")

    # Recherche d'un patient par ID
    with st.expander("🔎 Rechercher un patient par ID", expanded=False):
        search_id = st.text_input("Saisissez l'ID du patient à rechercher", key="search_patient_id")
        if st.button("Rechercher", key="search_button"):
            if search_id:
                # Nettoyer l'ID saisi pour enlever les espaces
                cleaned_search_id = search_id.strip()

                # Nettoyer la colonne ID du DataFrame pour une meilleure correspondance
                df_to_search = st.session_state.historique_df.copy()
                df_to_search["ID"] = df_to_search["ID"].str.strip()

                patient_found = df_to_search[df_to_search["ID"] == cleaned_search_id]

                if not patient_found.empty:
                    st.write(f"### Détails du patient {cleaned_search_id}")
                    st.dataframe(patient_found)
                else:
                    st.warning(f"⚠️ Patient avec l'ID '{cleaned_search_id}' non trouvé dans l'historique.")
            else:
                st.warning("Veuillez saisir un ID pour effectuer la recherche.")

    # Affichage des variables influentes
    features_import = joblib.load("top_features.pkl")
    with st.expander("📊 Variables influentes", expanded=False):
        if st.button("Afficher les variables influentes", key="feature_importance"):
            if st.session_state.med_pred is not None:
                st.dataframe(features_import)

                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(
                  x="Importance (%)",
                  y="Feature",
                  data=features_import,
                  ax=ax,
                  palette="viridis")
                plt.xlabel("Importance (%)")
                plt.ylabel("Variable")
                plt.title("Top variables influentes")
                st.pyplot(fig)

            else:
              st.warning("Veuillez d'abord faire une prédiction pour voir l'importance des variables.")


# ---------------- Page d'accueil ----------------
def accueil():
    st.subheader("Bienvenue sur l'application de prédiction de maladies cardiaques")
    st.markdown("---")
    st.markdown("""
    Les maladies cardiovasculaires constituent un défi majeur pour la santé mondiale. Une détection précoce des risques 
    est primordiale pour la prévention et la gestion des complications.

    Notre application est un outil d'aide à la décision conçu pour faciliter l'évaluation du risque de maladie cardiaque.
    Elle s'appuie sur un modèle d'apprentissage automatique, rigoureusement entraîné sur un vaste ensemble de données 
    médicales pour fournir des prédictions fiables.

    ### Fonctions principales
    * **Pour les patients :** Saisissez vos informations pour obtenir une **évaluation instantanée et personnalisée de votre risque**.
    Notre outil fournit également des recommandations ciblées et vous permet de comparer vos données aux normes médicales 
    de référence, vous donnant ainsi une perspective claire de votre situation.

    * **Pour les professionnels de santé :** Accédez à une interface avancée pour affiner votre diagnostic. Visualisez 
    l'**importance des facteurs de risque** dans la prédiction du modèle et consultez l'historique des évaluations des patients 
    pour un suivi optimisé.

    ### Avertissement important
    Cet outil est conçu pour assister dans la prise de décision et ne doit en aucun cas remplacer l'**expertise d'un 
    professionnel de la santé**. Les résultats générés ne constituent pas un diagnostic médical formel.
    """)
    st.markdown("---")
    st.subheader("Choisissez votre profil pour commencer :")
    col1, col2, col3= st.columns(3)
    with col1:
        if st.button("🧍 Patient",use_container_width=True):
            st.session_state.role = "patient"
            st.rerun()
    with col3:
        if st.button("👨‍⚕️ Médecin",use_container_width=True):
            st.session_state.role = "medecin"
            st.rerun()


# ---------------- Logique principale de l'application ----------------
if st.session_state.role == "accueil":
    accueil()
elif st.session_state.role == "patient":
    interface_patient()
elif st.session_state.role == "medecin":
    medecin_interface()
