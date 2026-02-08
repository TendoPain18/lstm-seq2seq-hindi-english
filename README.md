# Hindi-to-English Neural Machine Translation 🌐🔤

A sequence-to-sequence neural machine translation system using LSTM encoder-decoder architecture with TensorFlow/Keras. This project translates Hindi sentences to English using deep learning, achieving word-level accuracy on unseen test data.

![Project Thumbnail](images/thumbnail.png)

## 📋 Description

This project implements a complete neural machine translation (NMT) pipeline for translating Hindi text to English. The system uses an encoder-decoder architecture with LSTM (Long Short-Term Memory) networks, featuring GPU acceleration, proper train/test splitting to prevent data leakage, and comprehensive evaluation metrics.

The implementation demonstrates core concepts in sequence-to-sequence learning, including text preprocessing, vocabulary building, attention-free translation, and inference-time decoding strategies.

<br>
<div align="center">
  <a href="https://codeload.github.com/TendoPain18/hindi-english-neural-translation/legacy.zip/main">
    <img src="https://img.shields.io/badge/Download-Files-brightgreen?style=for-the-badge&logo=download&logoColor=white" alt="Download Files" style="height: 50px;"/>
  </a>
</div>

## 🎯 Project Objectives

1. **Build Seq2Seq Architecture**: Implement LSTM encoder-decoder for translation
2. **Process Bilingual Data**: Handle Hindi-English parallel corpus
3. **Prevent Data Leakage**: Proper train/test splitting with separate vectorization
4. **GPU Acceleration**: Optimize training with CUDA-enabled TensorFlow
5. **Evaluate Performance**: Calculate word-level accuracy on unseen test data

## ✨ Features

### Architecture Components
- **LSTM Encoder**: Processes Hindi input sequences and captures context
- **LSTM Decoder**: Generates English output sequences word-by-word
- **Embedding Layers**: 128-dimensional word embeddings for both languages
- **State Transfer**: Encoder final states initialize decoder

### Data Processing
- **Text Vectorization**: Convert text to integer sequences (5000 vocab size)
- **Sequence Padding**: Fixed 20-token maximum length
- **Special Tokens**: [start] and [end] markers for sentence boundaries
- **Text Standardization**: Lowercase conversion and punctuation handling

### Training Features
- **GPU Support**: Automatic GPU detection with memory growth
- **Train/Test Split**: 80-20 split with random seed for reproducibility
- **Batch Processing**: 64-sample batches with shuffling
- **Dropout Regularization**: 0.2 dropout rate to prevent overfitting

### Inference System
- **Separate Inference Models**: Encoder and decoder models for translation
- **Greedy Decoding**: Word-by-word generation with argmax selection
- **Maximum Length Control**: 20-word limit to prevent infinite generation
- **State Propagation**: Carry hidden/cell states across decoding steps

## 🔬 Theoretical Background

### Encoder-Decoder Architecture

**Encoder**: Processes the entire input sequence and compresses it into fixed-size context vectors (hidden state and cell state).

```
Hindi Input → Embedding → LSTM Encoder → Context Vectors (h, c)
```

**Decoder**: Uses context vectors to generate the output sequence one word at a time.

```
Context (h, c) → LSTM Decoder → Softmax → English Output
                      ↑
                 Previous Word
```

### LSTM (Long Short-Term Memory)

**Cell State Update**:
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # Input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C) # Candidate values
C_t = f_t * C_{t-1} + i_t * C̃_t        # New cell state
```

**Hidden State Update**:
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # Output gate
h_t = o_t * tanh(C_t)                   # New hidden state
```

### Training Process

**Input**: Hindi sentence (vectorized)
```
[245, 89, 1432, 67, 3]  # Token IDs
```

**Target**: English sentence (shifted by 1 for teacher forcing)
```
Decoder Input:  [START, I, am, learning, machine]
Expected Output: [I, am, learning, machine, END]
```

**Loss**: Sparse categorical cross-entropy
```
Loss = -∑ y_true · log(y_pred)
```

### Inference (Translation)

**Greedy Decoding Algorithm**:
```
1. Encode Hindi input → Get context vectors (h, c)
2. Initialize decoder with [START] token
3. While not [END] and length < 20:
   a. Predict next word (argmax of softmax output)
   b. Append word to translation
   c. Use predicted word as next decoder input
   d. Update states (h, c)
4. Return translated sentence
```

## 📊 Dataset Information

**Source**: Hindi-English parallel corpus (hin.txt)

**Statistics**:
- Total Samples: 3,116 sentence pairs
- Training Samples: 2,492 (80%)
- Testing Samples: 624 (20%)
- Vocabulary Size: 5,000 tokens (both languages)
- Max Sequence Length: 20 tokens

**Example Pairs**:
```
Hindi:    "मुझे हिंदी आती है।"
English:  "I know Hindi."

Hindi:    "यह किताब अच्छी है।"
English:  "This book is good."
```

## 🚀 Getting Started

### Prerequisites

**Python Requirements**:
```
Python 3.8+
TensorFlow 2.x (GPU-enabled for best performance)
NumPy
scikit-learn
CUDA 11.x (for GPU support)
cuDNN 8.x (for GPU support)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/hindi-english-neural-translation.git
cd hindi-english-neural-translation
```

2. **Install dependencies**
```bash
pip install tensorflow numpy scikit-learn
```

3. **Verify GPU setup (optional but recommended)**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

4. **Download dataset**
```bash
# Place hin.txt in the project directory
# Dataset should contain tab-separated Hindi-English pairs
```

5. **Run the training script**
```bash
python translation_model.py
```

## 📖 Usage Guide

### Training the Model

```python
# Configure hyperparameters
BATCH_SIZE = 64
TRAINING_EPOCHS = 100
LSTM_UNITS = 256
EMBED_SIZE = 128

# Load and split data
train_source, test_source, train_target, test_target = train_test_split(
    source_sentences,
    target_sentences,
    test_size=0.2,
    random_state=42
)

# Create vectorization layers (adapt on training data only)
source_vectorization.adapt(train_source)
target_vectorization.adapt(train_target)

# Train model
translation_model.fit(training_data, epochs=TRAINING_EPOCHS)
```

### Translating Sentences

```python
# Translate a Hindi sentence
hindi_input = "मैं हिंदी सीख रहा हूं।"
english_output = translate_sentence(hindi_input)
print(f"Translation: {english_output}")
# Output: "I am learning Hindi"
```

### Evaluating Performance

```python
# Calculate word-level accuracy on test set
total_accuracy = 0.0
for hindi, expected_english in zip(test_source, test_target):
    predicted_english = translate_sentence(hindi)
    accuracy = calculate_word_accuracy(expected_english, predicted_english)
    total_accuracy += accuracy

average_accuracy = (total_accuracy / len(test_source)) * 100
print(f"Test Accuracy: {average_accuracy:.2f}%")
```

## 📈 Model Performance

### Training Results

| Metric | Value |
|--------|-------|
| Final Training Loss | 0.2547 |
| Final Training Accuracy | 42.23% |
| Training Epochs | 100 |
| Batch Size | 64 |

### Test Set Evaluation

| Metric | Value |
|--------|-------|
| Word-Level Accuracy | 14.80% |
| Test Samples | 624 (unseen) |
| Vocabulary Coverage | 5,000 tokens |

### Sample Translations

**Successful Translation**:
```
Hindi:     "क्या तुम्हें गाड़ी चलाना आता है?"
Expected:  "do you know how to drive a car ?"
Predicted: "do you know how to drive a car"
✓ Highly accurate
```

**Partial Translation**:
```
Hindi:     "मेरे चाचा क्रिकेट के शौकिया खिलाड़ी हैं।"
Expected:  "my uncle is an amateur cricket player ."
Predicted: "my uncle is an amateur cricket player"
✓ Nearly perfect (missing period only)
```

**Failed Translation**:
```
Hindi:     "चीनी गर्म कॉफी में घुल जाती है।"
Expected:  "sugar dissolves in hot coffee ."
Predicted: "your sister is very fond of music"
✗ Incorrect semantic mapping
```

## 🔍 Key Insights

### Why 14.80% Test Accuracy?

1. **Small Dataset**: Only 2,492 training samples limits generalization
2. **Complex Language Pair**: Hindi-English have different grammatical structures
3. **Limited Vocabulary**: 5,000 tokens cannot capture all linguistic nuances
4. **No Attention Mechanism**: Model relies solely on fixed-size context vectors
5. **Simple Architecture**: Basic LSTM without advanced techniques (beam search, attention)

### Improvement Strategies

**To Increase Accuracy**:
- **Larger Dataset**: Use 50K+ parallel sentences
- **Attention Mechanism**: Add Bahdanau or Luong attention
- **Transformer Architecture**: Replace LSTM with Transformer
- **Beam Search**: Implement beam search instead of greedy decoding
- **Subword Tokenization**: Use BPE or WordPiece for better vocabulary
- **Pre-trained Embeddings**: Initialize with multilingual embeddings
- **More Training**: Increase epochs to 200-300 with early stopping

## 💻 GPU Acceleration

**GPU Configuration**:
```python
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

**Performance Boost**:
- CPU Training: ~30-40 seconds per epoch
- GPU Training: ~8-10 seconds per epoch
- **~3-4x speedup** with CUDA-enabled GPU

## 🎓 Learning Outcomes

This project demonstrates:

1. **Seq2Seq Architecture**: Encoder-decoder design for sequence transduction
2. **LSTM Networks**: Handling sequential data with memory cells
3. **Neural Machine Translation**: Core concepts in statistical translation
4. **Data Preprocessing**: Text vectorization and sequence handling
5. **Train/Test Splitting**: Preventing data leakage in NLP tasks
6. **Inference Strategies**: Greedy decoding and state propagation
7. **GPU Programming**: TensorFlow GPU acceleration

## 🤝 Contributing

Contributions are welcome! Potential improvements:

- Implement attention mechanism
- Add beam search decoding
- Try Transformer architecture
- Expand dataset size
- Add BLEU score evaluation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow/Keras team for deep learning framework
- Hindi-English parallel corpus contributors
- Neural machine translation research community

<br>
<div align="center">
  <a href="https://codeload.github.com/TendoPain18/hindi-english-neural-translation/legacy.zip/main">
    <img src="https://img.shields.io/badge/Download-Files-brightgreen?style=for-the-badge&logo=download&logoColor=white" alt="Download Files" style="height: 50px;"/>
  </a>
</div>

## <!-- CONTACT -->
<div id="toc" align="center">
  <ul style="list-style: none">
    <summary>
      <h2 align="center">
        🚀
        CONTACT ME
        🚀
      </h2>
    </summary>
  </ul>
</div>
<table align="center" style="width: 100%; max-width: 600px;">
<tr>
  <td style="width: 20%; text-align: center;">
    <a href="https://www.linkedin.com/in/amr-ashraf-86457134a/" target="_blank">
      <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" style="height: 33px; width: 120px;"/>
    </a>
  </td>
  <td style="width: 20%; text-align: center;">
    <a href="https://github.com/TendoPain18" target="_blank">
      <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" style="height: 33px; width: 120px;"/>
    </a>
  </td>
  <td style="width: 20%; text-align: center;">
    <a href="mailto:amrgadalla01@gmail.com">
      <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" style="height: 33px; width: 120px;"/>
    </a>
  </td>
  <td style="width: 20%; text-align: center;">
    <a href="https://www.facebook.com/amr.ashraf.7311/" target="_blank">
      <img src="https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white" style="height: 33px; width: 120px;"/>
    </a>
  </td>
  <td style="width: 20%; text-align: center;">
    <a href="https://wa.me/201019702121" target="_blank">
      <img src="https://img.shields.io/badge/WhatsApp-25D366?style=for-the-badge&logo=whatsapp&logoColor=white" style="height: 33px; width: 120px;"/>
    </a>
  </td>
</tr>
</table>
<!-- END CONTACT -->

## **Break language barriers with neural machine translation! 🌐✨**
