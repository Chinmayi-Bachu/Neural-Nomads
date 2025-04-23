# Neural-Nomads
### Step-by-Step Guide to Build the CNN-LSTM Video Classification Model

#### **1. Dataset Preparation**
1. **Extract Data:**
   - Unzip the dataset (`rtp.zip`) to a directory structure. Ensure that the subdirectories consist of `train`, `val`, and `test` folders with subfolders representing class labels (`safe`, `harmful`, `adult`).

   ```python
   import zipfile
   
   zip_filepath = 'rtp.zip'
   unzipped_dir = 'rtp_videos'
   with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
       zip_ref.extractall(unzipped_dir)
   ```

2. **Inspect Class Folders and Load File Paths:**
   - Inside the extracted dataset, load video file paths for each class directory (e.g., train/`safe`, train/`harmful`, train/`adult`), and prepare lists (`train_data` and `train_labels`).

   ```python
   labels_mapping = {'safe': 0, 'harmful': 1, 'adult': 2}
   train_data = []
   train_labels = []

   for label_name, label_idx in labels_mapping.items():
       label_dir = os.path.join(rtp_train_dir, label_name)
       for filename in os.listdir(label_dir):
           video_path = os.path.join(label_dir, filename)
           train_data.append(video_path)
           train_labels.append(label_idx)
   ```

   - Confirm the number of videos:
     ```python
     len(train_data), len(train_labels)
     ```

3. **Preprocess Videos by Frame Extraction:**
   - Define constants like `FRAME_HEIGHT`, `FRAME_WIDTH`, and `MAX_FRAMES` to specify frame resolution and maximum frames per video.
   - Use a frame extraction function to load frames from each video and resize them to match the input size of the CNN.

   ```python
   def extract_frames(video_path, max_frames=20):
       # Add your video loading and frame extraction logic here
   ```

   - Extract frames for all training videos:
     ```python
     X_train = []
     for video_path in train_data:
         frames = extract_frames(video_path)
         X_train.append(frames)
     X_train = np.array(X_train)
     ```

4. **One-Hot Encode Labels:**
   - Convert the labels into one-hot encoded vectors suitable for multi-class classification.

   ```python
   from tensorflow.keras.utils import to_categorical
   y_train = to_categorical(train_labels, num_classes=3)
   ```

---

#### **2. Model Architecture**

The CNN-LSTM integrates a Convolutional Neural Network (CNN) in a time-distributed manner to learn spatial features from frames, followed by LSTM layers to capture temporal dependencies between frames.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, TimeDistributed, Dropout

FRAME_HEIGHT = 64
FRAME_WIDTH = 64
MAX_FRAMES = 20

model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=(MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3)),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Adjust number of classes here
])
```

---

#### **3. Model Compilation**

- Compile the model using the Adam optimizer, categorical cross-entropy loss (ideal for multi-class classification), and accuracy as a metric.

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

#### **4. Model Training**

- Train the model with training data (`X_train` and `y_train`), set the appropriate batch size and epoch count.

```python
model.fit(X_train, y_train, epochs=10, batch_size=4)
```

---

#### **5. Model Evaluation**

- Evaluate the training set accuracy to check for overfitting or underfitting.

```python
from sklearn.metrics import accuracy_score

train_preds = model.predict(X_train)
train_pred_labels = np.argmax(train_preds, axis=1)
y_train_labels = np.argmax(y_train, axis=1)

train_accuracy = accuracy_score(y_train_labels, train_pred_labels)
print("Training Accuracy:", train_accuracy)
```

---

#### **6. Save and Load the Model**

- Save the trained model for future use.
  ```python
  model.save('video_classification_model.h5')
  ```

- Load the saved model:
  ```python
  from tensorflow.keras.models import load_model
  model = load_model('video_classification_model.h5')
  ```

---

#### **7. Next Steps for Deployment**
1. **Testing and Validation:**
   - Perform predictions on validation and test datasets.
   - Evaluate the model on unseen data to determine its generality.

2. **Fine-tuning:**
   - Adjust hyperparameters like learning rates, epochs, or architecture (e.g., number of Conv2D/LSTM layers or units).

3. **Inference Pipeline:**
   - Develop an inference pipeline to classify incoming videos by extracting frames and passing them through the model.

This concludes the detailed step-by-step guide to building and training the CNN-LSTM video classification model.
