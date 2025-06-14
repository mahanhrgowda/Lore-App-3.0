import streamlit as st
import re
from collections import Counter
import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib

# Load pre-trained neural network model
try:
    model = joblib.load('bhava_model.joblib')
except FileNotFoundError:
    st.error("Pre-trained model 'bhava_model.joblib' not found. Please ensure it is in the repository.")
    st.stop()

# Load label encoder
try:
    le = joblib.load('label_encoder.joblib')
except FileNotFoundError:
    st.error("Label encoder 'label_encoder.joblib' not found. Please ensure it is in the repository.")
    st.stop()

# Dataset with chakra mappings
chakra_data = [
    {
        "chakra": "Muladhara",
        "emoji_chakra": "🌱",
        "phonemes": ["vaṁ", "śaṁ", "ṣaṁ", "saṁ"],
        "bhava": "Stability, Security, Survival",
        "emoji_bhava": "🏠",
        "rasa": "Shanta",
        "emoji_rasa": "🕊️",
        "sthayibhava": "Shama",
        "deity": "Brahma",
        "emoji_deity": "🪔"
    },
    {
        "chakra": "Svadhisthana",
        "emoji_chakra": "🌊",
        "phonemes": ["baṁ", "bhaṁ", "maṁ", "yaṁ", "raṁ", "laṁ"],
        "bhava": "Creativity, Sexuality, Emotions",
        "emoji_bhava": "🎨",
        "rasa": "Sringara",
        "emoji_rasa": "💖",
        "sthayibhava": "Rati",
        "deity": "Vishnu",
        "emoji_deity": "🪷"
    },
    {
        "chakra": "Manipura",
        "emoji_chakra": "🔥",
        "phonemes": ["ḍaṁ", "ḍhaṁ", "ṇaṁ", "taṁ", "thaṁ", "daṁ", "dhaṁ", "naṁ", "paṁ", "phaṁ"],
        "bhava": "Personal Power, Willpower, Self-Esteem",
        "emoji_bhava": "💪",
        "rasa": "Veera",
        "emoji_rasa": "🦁",
        "sthayibhava": "Utsaha",
        "deity": "Indra",
        "emoji_deity": "⚡️"
    },
    {
        "chakra": "Anahata",
        "emoji_chakra": "💚",
        "phonemes": ["kaṁ", "khaṁ", "gaṁ", "ghaṁ", "ṅaṁ", "caṁ", "chaṁ", "jaṁ", "jhaṁ", "ñaṁ", "ṭaṁ", "ṭhaṁ"],
        "bhava": "Love, Compassion, Forgiveness",
        "emoji_bhava": "🤗",
        "rasa": "Sringara",
        "emoji_rasa": "💖",
        "sthayibhava": "Rati",
        "deity": "Vishnu",
        "emoji_deity": "🪷"
    },
    {
        "chakra": "Vishuddha",
        "emoji_chakra": "🗣️",
        "phonemes": ["aṁ", "āṁ", "iṁ", "īṁ", "uṁ", "ūṁ", "ṛṁ", "ṝṁ", "ḷṁ", "ḹṁ", "eṁ", "aiṁ", "oṁ", "auṁ", "aṁ", "aḥ"],
        "bhava": "Communication, Self-Expression, Truth",
        "emoji_bhava": "📢",
        "rasa": "Adbhuta",
        "emoji_rasa": "✨",
        "sthayibhava": "Vismaya",
        "deity": "Brahma",
        "emoji_deity": "🪔"
    },
    {
        "chakra": "Ajna",
        "emoji_chakra": "👁️",
        "phonemes": ["haṁ", "kṣaṁ"],
        "bhava": "Intuition, Insight, Wisdom",
        "emoji_bhava": "🧠",
        "rasa": "Shanta",
        "emoji_rasa": "🕊️",
        "sthayibhava": "Shama",
        "deity": "Brahma",
        "emoji_deity": "🪔"
    },
    {
        "chakra": "Sahasrara",
        "emoji_chakra": "👑",
        "phonemes": ["aum"],
        "bhava": "Spiritual Connection, Enlightenment, Bliss",
        "emoji_bhava": "🙏",
        "rasa": "Shanta",
        "emoji_rasa": "🕊️",
        "sthayibhava": "Shama",
        "deity": "Brahma",
        "emoji_deity": "🪔"
    }
]

# All unique phonemes for vectorization
all_phonemes = sorted(set(sum([chakra["phonemes"] for chakra in chakra_data], [])))
bhava_labels = [chakra["bhava"] for chakra in chakra_data]

# Simplified English-to-Sanskrit phoneme mapping
english_to_phoneme = {
    'a': 'aṁ', 'b': 'baṁ', 'c': 'caṁ', 'd': 'daṁ', 'e': 'eṁ',
    'f': 'phaṁ', 'g': 'gaṁ', 'h': 'haṁ', 'i': 'iṁ', 'j': 'jaṁ',
    'k': 'kaṁ', 'l': 'laṁ', 'm': 'maṁ', 'n': 'naṁ', 'o': 'oṁ',
    'p': 'paṁ', 'q': 'kṣaṁ', 'r': 'raṁ', 's': 'saṁ', 't': 'taṁ',
    'u': 'uṁ', 'v': 'vaṁ', 'w': 'vaṁ', 'x': 'kṣaṁ', 'y': 'yaṁ', 'z': 'ṣaṁ'
}

# Function to create phoneme frequency vector
def create_phoneme_vector(phonemes):
    freq = Counter(phonemes)
    vector = [freq.get(phoneme, 0) for phoneme in all_phonemes]
    total = sum(vector)
    if total == 0:
        return np.zeros(len(all_phonemes))
    return np.array(vector) / total  # Normalize

# Function to generate dynamic lore, story, and poem
def generate_content(name, chakra_info, phoneme):
    chakra = chakra_info["chakra"]
    bhava = chakra_info["bhava"]
    rasa = chakra_info["rasa"]
    sthayibhava = chakra_info["sthayibhava"]
    deity = chakra_info["deity"]

    # Dynamic Lore (150-200 words)
    chakra_texts = {
        "Muladhara": ("base of the spine", "*Yoga Upanishads*", "grounding"),
        "Svadhisthana": ("sacral region", "*Chandogya Upanishad*", "fluidity"),
        "Manipura": ("solar plexus", "*Katha Upanishad*", "radiant fire"),
        "Anahata": ("heart center", "*Mundaka Upanishad*", "universal love"),
        "Vishuddha": ("throat", "*Kena Upanishad*", "pure expression"),
        "Ajna": ("third eye", "*Brihadaranyaka Upanishad*", "inner vision"),
        "Sahasrara": ("crown", "*Mandukya Upanishad*", "divine union")
    }
    deity_texts = {
        "Brahma": ("creator", "*Rigveda*", "cosmic wisdom"),
        "Vishnu": ("preserver", "*Vishnu Purana*", "divine love"),
        "Indra": ("warrior king", "*Rigveda*", "valor")
    }
    location, text_ref, energy = chakra_texts[chakra]
    deity_role, deity_ref, deity_quality = deity_texts[deity]
    lore = (
        f"The name {name}, resonating with the phoneme ‘{phoneme}’, aligns with the {chakra} Chakra, the {location}’s seat of {energy}, as described in {text_ref}. "
        f"This chakra embodies the Bhava of {bhava.lower()}, guiding {name}’s journey toward {bhava.split(',')[0].lower()}. "
        f"The {rasa} Rasa, evoking {sthayibhava.lower()}, stirs the heart with {rasa.lower()}, as per *Natya Shastra*’s aesthetic wisdom. "
        f"Guided by {deity}, the {deity_role} from {deity_ref}, {name}’s essence reflects {deity_quality}. "
        f"Through the {chakra}’s vibration, {name} harmonizes personal {bhava.split(',')[0].lower()} with divine {rasa.lower()}, forging a path of spiritual resonance."
    )

    # Dynamic Story (100-150 words)
    settings = {
        "Muladhara": "village by the Ganges",
        "Svadhisthana": "Vrindavan’s lush groves",
        "Manipura": "Kurukshetra’s battlefield",
        "Anahata": "desert village oasis",
        "Vishuddha": "Himalayan peak",
        "Ajna": "forest cave",
        "Sahasrara": "sacred grove under a bodhi tree"
    }
    challenges = {
        "Stability, Security, Survival": "fear of loss",
        "Creativity, Sexuality, Emotions": "creative doubt",
        "Personal Power, Willpower, Self-Esteem": "moment of hesitation",
        "Love, Compassion, Forgiveness": "past grievances",
        "Communication, Self-Expression, Truth": "silenced voice",
        "Intuition, Insight, Wisdom": "clouded mind",
        "Spiritual Connection, Enlightenment, Bliss": "earthly attachments"
    }
    actions = {
        "Muladhara": "rebuilt their home",
        "Svadhisthana": "wove a vibrant tapestry",
        "Manipura": "charged with a blazing bow",
        "Anahata": "shared their heart’s warmth",
        "Vishuddha": "chanted a sacred hymn",
        "Ajna": "meditated in silence",
        "Sahasrara": "embraced cosmic unity"
    }
    setting = settings[chakra]
    challenge = challenges[bhava]
    action = actions[chakra]
    story = (
        f"In {setting}, {name} felt their {chakra} Chakra awaken, stirring the Bhava of {bhava.lower()}. "
        f"Facing {challenge}, their heart wavered, yet the phoneme ‘{phoneme}’ resonated within. "
        f"Through {action}, {name} channeled {energy}, and {rasa} Rasa bloomed, filling them with {sthayibhava.lower()}. "
        f"{deity}, appearing in a divine vision, bestowed {deity_quality}. "
        f"{name}’s {bhava.split(',')[0].lower()} inspired those around, uniting them in {rasa.lower()}, echoing {deity_ref}’s timeless grace.”
    )

    # Dynamic Poem (4-8 lines)
    poem = (
        f"{name}’s call, with ‘{phoneme}’ so bright,\n"
        f"{chakra}’s {energy}, a guiding light.\n"
        f"{bhava}’s heart, in {rasa}’s sway,\n"
        f"{deity}’s {deity_quality} paves the way.\n"
    )

    return lore, story, poem

# Function to map input text to chakra using neural network
def map_text_to_chakra(text):
    text = re.sub(r'[^a-zA-Z]', '', text.lower())
    if not text:
        return None, None, None, "Input contains no valid letters."

    phonemes = [english_to_phoneme.get(char, 'aṁ') for char in text]
    vector = create_phoneme_vector(phonemes)
    probas = model.predict_proba([vector])[0]
    bhava_index = np.argmax(probas)
    confidence = probas[bhava_index]
    predicted_bhava = le.inverse_transform([bhava_index])[0]

    for chakra in chakra_data:
        if chakra["bhava"] == predicted_bhava:
            return chakra, phonemes, confidence, None
    return None, phonemes, None, f"No chakra found for predicted Bhava '{predicted_bhava}'."

# Streamlit app
def main():
    st.set_page_config(page_title="Chakra-Rasa Mapper", page_icon="🕉️")
    st.title("Chakra, Bhava, Rasa, and Deity Mapper 🕉️")
    st.markdown("""
    Enter an English name to discover its associated **Chakra**, **Bhava**, **Rasa**, and **Deity** based on Sanskrit phonemes and the *Natya Shastra*. The app uses a neural network to predict the dominant Bhava from phoneme frequencies, mapping it to the corresponding chakra. Explore dynamically generated lore, a story, and a poem tailored to your name!
    """)

    with st.form(key="input_form"):
        user_input = st.text_input("Enter a name:", placeholder="e.g., Arjun, Love, Samskruthi")
        submit_button = st.form_submit_button("Map to Chakra")

    if submit_button and user_input:
        chakra_info, detected_phonemes, confidence, error = map_text_to_chakra(user_input)
        if error:
            st.error(error)
        else:
            st.success(f"Results for '{user_input}':")
            st.markdown(f"""
            - **Detected Phonemes**: {', '.join(detected_phonemes)}
            - **Predicted Bhava**: {chakra_info['bhava']} {chakra_info['emoji_bhava']} (Confidence: {confidence:.2%})
            - **Chakra**: {chakra_info['chakra']} {chakra_info['emoji_chakra']}
            - **Rasa**: {chakra_info['rasa']} {chakra_info['emoji_rasa']}
            - **Sthayibhava**: {chakra_info['sthayibhava']}
            - **Deity**: {chakra_info['deity']} {chakra_info['emoji_deity']}
            """)
            lore, story, poem = generate_content(user_input, chakra_info, detected_phonemes[0])
            with st.expander("View Dynamic Lore"):
                st.markdown(lore)
            with st.expander("Read Dynamic Story"):
                st.markdown(story)
            with st.expander("Discover Dynamic Poem"):
                st.markdown(f"```\n{poem}\n```")
            st.info("Note: The Bhava is predicted using a neural network based on normalized phoneme frequencies loaded from .joblib files. Lore, story, and poem are dynamically generated based on your input name.")

    st.markdown("""
    ---
    *Built with ❤️ by Mahan H R Gowda using Streamlit. Based on the Natya Shastra, Vedic, and Tantric traditions.*
    """)

if __name__ == "__main__":
    main()