# Chakra-Rasa Mapper

A Streamlit web application that maps English names to their associated **Chakra**, **Bhava**, **Rasa**, and **Deity** based on Sanskrit phonemes and the *Natya Shastra*. The app uses a pre-trained neural network, saved as `.joblib` files, to predict the dominant Bhava from phoneme frequency vectors, mapping it to the corresponding chakra. It dynamically generates personalized lore, a story, and a poem for each input name, grounded in ancient Sanskrit texts.

## Features
- **Machine Learning**: Uses a pre-trained feedforward neural network (scikit-learn) loaded from `bhava_model.joblib` and `label_encoder.joblib` to predict the dominant Bhava from normalized phoneme frequency vectors.
- **Phoneme Vectoring**: Converts text to a 51-dimensional vector of phoneme frequencies, covering 50 Sanskrit bijas plus "aum".
- **Bhava Prediction**: Neural network classifies Bhava with confidence scores, trained on synthetic data aligned with *Sat-Cakra-Nirupana* and *Natya Shastra*.
- **Chakra Details**: Displays the associated chakra, Bhava, Rasa, Sthayibhava, and deity, with emojis for visual appeal.
- **Dynamic Content**: Generates a 150-200 word lore, a 100-150 word story, and a 4-8 line poem tailored to the input name, reflecting the chakra’s energy, Bhava’s emotion, Rasa’s aesthetic, and deity’s guidance.
- **Portable .joblib Models**: Saves the neural network and label encoder as compact `.joblib` files, with base64-encoded versions for GitHub compatibility.
- **User-Friendly Interface**: Built with Streamlit, featuring input validation, error handling, and collapsible sections for lore, story, and poem.
- **Deployment-Ready**: Optimized for Streamlit Community Cloud with minimal dependencies.

## Prerequisites
- Python 3.8 or higher
- Streamlit 1.38.0
- scikit-learn 1.5.2
- joblib 1.4.2
- Git (for cloning the repository)
- A GitHub account (for Streamlit Cloud deployment)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/chakra-rasa-mapper.git
   cd chakra-rasa-mapper
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure `bhava_model.joblib` and `label_encoder.joblib` are in the repository root (generated via `train_model.py` or decoded from base64 text files).

## Generating the .joblib Models
The repository includes `bhava_model.joblib` and `label_encoder.joblib`. To regenerate or decode:
1. **Run the training script**:
   ```bash
   python train_model.py
   ```
   - This generates `bhava_model.joblib`, `label_encoder.joblib`, `bhava_model_base64.txt`, and `label_encoder_base64.txt`.
2. **Decode from Base64 (if using provided strings)**:
   - Save the base64 strings from `bhava_model_base64.txt` and `label_encoder_base64.txt` into `decode_joblib.py`.
   - Run:
     ```bash
     python decode_joblib.py
     ```
   - This creates `bhava_model.joblib` and `label_encoder.joblib`.

## Local Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and navigate to `http://localhost:8501`.
3. Enter an English name (e.g., "Arjun", "Love", "Samskruthi").
4. Click "Map to Chakra" to view the results, including detected phonemes, predicted Bhava with confidence, chakra details, and dynamically generated lore, story, and poem.

## Deployment on Streamlit Cloud
1. Push the repository to GitHub, including `bhava_model.joblib` and `label_encoder.joblib` (or their base64 text files):
   ```bash
   git add .
   git commit -m "Add app and .joblib models"
   git push origin main
   ```
2. Log in to [Streamlit Community Cloud](https://streamlit.io/cloud) with your GitHub account.
3. Click "New app" and select your repository.
4. Set the branch to `main` and the main file path to `app.py`.
5. Deploy the app. Streamlit will install dependencies from `requirements.txt`.
6. Access the app via the provided URL.

## Example Outputs
- **Input**: "Arjun"
  - **Detected Phonemes**: aṁ, raṁ, jaṁ, uṁ, naṁ
  - **Predicted Bhava**: Communication, Self-Expression, Truth 📢 (Confidence: e.g., 92.3%)
  - **Chakra**: Vishuddha 🗣️
  - **Rasa**: Adbhuta ✨
  - **Sthayibhava**: Vismaya
  - **Deity**: Brahma 🪔
  - **Lore**: Personalized prose linking Arjun’s name to Vishuddha’s truth and Brahma’s wisdom.
  - **Story**: A tale of Arjun chanting a hymn on a Himalayan peak, evoking Adbhuta.
  - **Poem**: A verse celebrating Arjun’s voice and Vishuddha’s clarity.
- **Input**: "Love"
  - **Detected Phonemes**: laṁ, oṁ, vaṁ, eṁ
  - **Predicted Bhava**: Creativity, Sexuality, Emotions 🎨 (Confidence: e.g., 87.5%)
  - **Chakra**: Svadhisthana 🌊
  - **Rasa**: Sringara 💖
  - **Sthayibhava**: Rati
  - **Deity**: Vishnu 🪷
  - **Lore**: Prose connecting Love’s name to Svadhisthana’s creativity and Vishnu’s love.
  - **Story**: A story of Love weaving a tapestry in Vrindavan, filled with Sringara.
  - **Poem**: A verse reflecting Love’s passion and Svadhisthana’s flow.

## Project Structure
```
chakra-rasa-mapper/
├── app.py                      # Main Streamlit application
├── train_model.py              # Script to train the neural network and save as .joblib
├── bhava_model.joblib          # Pre-trained neural network model
├── label_encoder.joblib        # Pre-trained label encoder
├── bhava_model_base64.txt      # Base64-encoded neural network model
├── label_encoder_base64.txt    # Base64-encoded label encoder
├── decode_joblib.py            # Script to decode base64 strings to .joblib
├── requirements.txt            # Dependencies for deployment
├── README.md                   # Project documentation
```

## Notes
- **Phoneme Mapping**: Uses a simplified English-to-Sanskrit mapping. For production, consider a transliteration library like `indic-transliteration`.
- **Neural Network**: Trained on synthetic data simulating phoneme distributions per *Sat-Cakra-Nirupana*. Real labeled data would improve accuracy.
- **Dynamic Content**: Lore, story, and poem are generated based on the input name, ensuring personalized narratives rooted in Vedic traditions.
- **.joblib Models**: Saved as `bhava_model.joblib` and `label_encoder.joblib`, with base64-encoded versions for portability. Direct loading is efficient.
- **Cultural Accuracy**: Content aligns with *Natya Shastra*, *Vedas*, *Upanishads*, and Tantric texts. The Sahasrara chakra uses the "aum" phoneme, reflecting its sacred sound.
- **Deployment**: Lightweight, using Streamlit, scikit-learn, and joblib, ensuring Streamlit Cloud compatibility.
- **Enhancements**: Could include real training data, advanced transliteration, or phoneme weighting by position.

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Inspired by the *Natya Shastra* and yogic traditions.
- Built with [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/).
- References: [Caleidoscope](https://www.caleidoscope.in/featured/rasas-in-bharata-munis-natya-shastra/), [Wikipedia](https://en.wikipedia.org/wiki/Rasa_(aesthetics)), [Kalyani Kala Mandir](https://www.kalyanikalamandir.org/the-navarasas/).

---
*Built with ❤️ by [Your Name]*