import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import base64

# Define phonemes and Bhavas based on ancient Sanskrit texts
chakra_data = [
    {"phonemes": ["vaṁ", "śaṁ", "ṣaṁ", "saṁ"], "bhava": "Stability, Security, Survival"},
    {"phonemes": ["baṁ", "bhaṁ", "maṁ", "yaṁ", "raṁ", "laṁ"], "bhava": "Creativity, Sexuality, Emotions"},
    {"phonemes": ["ḍaṁ", "ḍhaṁ", "ṇaṁ", "taṁ", "thaṁ", "daṁ", "dhaṁ", "naṁ", "paṁ", "phaṁ"], "bhava": "Personal Power, Willpower, Self-Esteem"},
    {"phonemes": ["kaṁ", "khaṁ", "gaṁ", "ghaṁ", "ṅaṁ", "caṁ", "chaṁ", "jaṁ", "jhaṁ", "ñaṁ", "ṭaṁ", "ṭhaṁ"], "bhava": "Love, Compassion, Forgiveness"},
    {"phonemes": ["aṁ", "āṁ", "iṁ", "īṁ", "uṁ", "ūṁ", "ṛṁ", "ṝṁ", "ḷṁ", "ḹṁ", "eṁ", "aiṁ", "oṁ", "auṁ", "aṁ", "aḥ"], "bhava": "Communication, Self-Expression, Truth"},
    {"phonemes": ["haṁ", "kṣaṁ"], "bhava": "Intuition, Insight, Wisdom"},
    {"phonemes": ["aum"], "bhava": "Spiritual Connection, Enlightenment, Bliss"}
]
all_phonemes = sorted(set(sum([chakra["phonemes"] for chakra in chakra_data], [])))
bhavas = [chakra["bhava"] for chakra in chakra_data]

# Generate synthetic data aligned with Sanskrit texts
n_samples = 10000
X = []
y = []
np.random.seed(42)

for _ in range(n_samples):
    bhava_idx = np.random.randint(0, len(bhavas))
    target_phonemes = chakra_data[bhava_idx]["phonemes"]
    vector = np.zeros(len(all_phonemes))
    target_indices = [all_phonemes.index(p) for p in target_phonemes]
    weight = 0.7 / len(target_indices)
    for idx in target_indices:
        vector[idx] = np.random.uniform(weight * 0.8, weight * 1.2)
    non_target_indices = [i for i in range(len(all_phonemes)) if i not in target_indices]
    if non_target_indices:
        noise_weight = 0.3 / len(non_target_indices)
        for idx in non_target_indices:
            vector[idx] = np.random.uniform(0, noise_weight * 0.5)
    vector = vector / vector.sum() if vector.sum() > 0 else vector
    X.append(vector)
    y.append(bhavas[bhava_idx])

X = np.array(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train neural network
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X, y_encoded)

# Save models as .joblib
joblib.dump(model, 'bhava_model.joblib')
joblib.dump(le, 'label_encoder.joblib')

# Encode .joblib files to base64
for file_path, output_path in [('bhava_model.joblib', 'bhava_model_base64.txt'), ('label_encoder.joblib', 'label_encoder_base64.txt')]:
    with open(file_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    with open(output_path, 'w') as f:
        f.write(encoded)
    print(f"Base64 string saved to {output_path}")