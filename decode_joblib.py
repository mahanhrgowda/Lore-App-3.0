import base64

# Replace with contents of bhava_model_base64.txt and label_encoder_base64.txt
bhava_model_base64 = open('bhava_model_base64.txt', 'r').read()
label_encoder_base64 = open('label_encoder_base64.txt', 'r').read()

for base64_string, output_path in [(bhava_model_base64, 'bhava_model.joblib'), (label_encoder_base64, 'label_encoder.joblib')]:
    with open(output_path, 'wb') as f:
        f.write(base64.b64decode(base64_string))
    print(f"Saved {output_path}")