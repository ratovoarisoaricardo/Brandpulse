import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# classical model matrix mock (struggles with neutral tweets)
cm_classical = np.array([
    [2400, 300, 150],  # actual negative
    [500,  850, 200],  # actual neutral
    [150,  250, 1100]  # actual positive
])

plt.figure(figsize=(6, 5))
sns.heatmap(cm_classical, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'], 
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Confusion Matrix: Classical Model (TF-IDF)')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('models/cm_classical.png')
plt.close()

# lstm model matrix mock
cm_lstm = np.array([
    [2300, 350, 200],
    [400,  900, 250],
    [100,  200, 1200]
])

plt.figure(figsize=(6, 5))
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Negative', 'Neutral', 'Positive'], 
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Confusion Matrix: Deep Learning (LSTM)')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('models/cm_lstm.png')
plt.close()

print("Confusion matrices generated successfully.")
