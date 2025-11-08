# 🎬 Deep Learning–Based Sentiment Analysis of Movie Reviews

## 🧠 Overview
This project applies **Deep Learning** and **Natural Language Processing (NLP)** to analyze IMDB movie reviews.  
Using an **LSTM (Long Short-Term Memory)** network, it predicts whether a movie review expresses a positive or negative sentiment.  

The project demonstrates **text preprocessing**, **feature engineering**, **data visualization**, and **model training** — showcasing how neural networks interpret emotions and opinions in text data.

---

## 🎯 Objectives
1. Build a model to classify movie reviews as **positive** or **negative**.  
2. Explore NLP concepts such as **tokenization**, **padding**, and **word embeddings**.  
3. Implement and train an **LSTM-based neural network** for sentiment analysis.  
4. Visualize and analyze model performance through **accuracy and loss curves**.  
5. Derive insights into **audience emotions** from unstructured review text.

---

## 🏢 Business Problem
In the **entertainment industry**, understanding public sentiment is vital for improving content, marketing, and audience engagement.  
Analyzing thousands of user reviews manually is impractical — this project automates sentiment detection using deep learning, enabling studios and streaming platforms to **measure audience satisfaction** and make **data-driven business decisions** effectively.

## 📊 Dataset Information

1. **Dataset:** [IMDB Movie Reviews Dataset (Keras built-in)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
2. **Training Samples:** 25,000  
3. **Test Samples:** 25,000  
4. **Label 0:** Negative Review  
5. **Label 1:** Positive Review


## ⚙️ Tech Stack

1. **Language:** Python  
2. **Libraries:** TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn, WordCloud, Scikit-learn, NLTK  
3. **Environment:** Google Colab / Jupyter Notebook


## 🔍 Project Workflow

### 1. Data Loading
- Loaded IMDB dataset using Keras.
- Viewed structure, shapes, and examples of data.

### 2. Exploratory Data Analysis (EDA)
- Label distribution visualization.
- Review length analysis.
- Word clouds for positive and negative reviews.

### 3. Feature Engineering
- Created features like review length, average word index, and VADER sentiment counts.
- Added positive/negative word ratios.

### 4. Data Preprocessing
- Tokenization and sequence padding (max length = 250).
- Conversion of word indices into numerical format for neural network input.

### 5. Model Building
- Implemented **Embedding → LSTM → Dense** architecture.
- Activation: **Sigmoid** for binary output.
- Loss: **Binary Crossentropy**.
- Optimizer: **Adam**.

### 6. Model Evaluation
- Accuracy & Loss plots across epochs.
- Confusion matrix visualization.

### 7. Real Review Prediction
- Custom function to test new user reviews and predict sentiment.

## 📈 Results
- Achieved **high accuracy** on both training and validation datasets.  
- The **LSTM model** effectively learned complex patterns distinguishing positive and negative sentiments.  
- **Visualizations** (EDA, Word Clouds, Accuracy & Loss curves) provided strong interpretability of model behavior and data distribution.

---

## 💡 Business Insights
- Enables **automated sentiment understanding** from thousands of user reviews.  
- Assists **entertainment companies** in identifying audience preferences and improving future content.  
- Can be integrated into **recommendation systems** or **marketing dashboards** for real-time audience sentiment analysis.

---

## 🏁 Conclusion
This project demonstrates how **Deep Learning** and **NLP** can extract valuable insights from textual data.  
By analyzing IMDB reviews, the **LSTM-based sentiment analysis model** accurately distinguishes emotional tones and opinions.  
Such models can scale to analyze **millions of reviews**, providing actionable feedback loops that empower the **entertainment industry** to make **data-driven decisions** and enhance viewer satisfaction.

## 📦 Future Enhancements

- Implement **Bidirectional LSTM** or **GRU layers** to capture richer contextual understanding and improve accuracy.  
- Integrate **pre-trained embeddings** such as **GloVe** or **Word2Vec** for enhanced semantic representation.  
- **Deploy the model** as an interactive web application using **Streamlit** or **Flask** for real-time sentiment prediction.  
- Extend the project with **multilingual review datasets** to perform **global sentiment analysis** across different languages.

## 👤 Author: Om Patil

📧 **Data Science & Machine Learning Enthusiast**  

🔗 [Connect with me on LinkedIn](https://www.linkedin.com/in/om-patil-039863369/)  

👨‍💻 **GitHub Profile:** [Om Patil](https://github.com/OmPatil2806)
