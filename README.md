# Recurrent Neural Network (specifically LSTM)

## Training

`python3 src/train.py`

Will create models in `models/`

## Running

`python3 src/predict.py`

Generates text based off previous text, hardcoded in `src/predict.py`

## Explanation

## LSTM, RNN

LSTM (Long Short-Term Memory) is a type of RNN (Recurrent Neural Network).

Here's a brief rundown:

Recurrent Neural Network (RNN): This is a class of neural networks where connections can loop back in time. This temporal dynamic is specifically useful for sequences and series of data, making RNNs particularly useful for tasks like time series forecasting, speech recognition, and text generation. However, vanilla RNNs suffer from problems like vanishing and exploding gradients, which make them hard to train on long sequences.

Long Short-Term Memory (LSTM): LSTMs are an evolution of RNNs introduced to combat the issues faced by vanilla RNNs. They have a more complex cell structure, which includes gates to control the flow of information, helping them to remember long-term dependencies in the data.

GRU (Gated Recurrent Unit): GRUs are another variation of RNNs, similar to LSTMs but with a simplified gating mechanism.

To summarize, an LSTM is a specific type of RNN. When you're using an LSTM, you are using an RNN, but one that has been designed to better capture long-term dependencies in sequence data compared to vanilla RNNs.

### Loss

Loss is the cumulative loss of the model over an entire epoch during training.

Loss: This is the discrepancy between the predictions of the model and the actual target values. In deep learning, this discrepancy is quantified by a loss function. For the LSTM model, we are using the CrossEntropyLoss.

179353.5834172368: This is the actual value of the loss. It indicates the total error the model made on the entire dataset for that specific epoch.

Here's what this means in the context of the model:

High Value: If the loss is high, especially in the initial epochs, it means the model is making many mistakes in its predictions. This is expected at the beginning of training.

Decreasing Over Time: As we train the model over more epochs, we'd expect this loss to decrease, meaning the model is getting better and making fewer mistakes.

Not Decreasing: If the loss stops decreasing or starts increasing again, it might indicate problems like overfitting, where the model is starting to "memorize" the training data rather than "learning" it.

Absolute Value: The absolute value of the loss, by itself, doesn't tell us much. What's more important is its trend over time. However, if we're comparing different models, architectures, or hyperparameters, the loss can give us a comparative measure of performance.

In summary, the loss provides insight into how well the model is performing. Monitoring its trend can help diagnose training issues and guide decisions about when to stop training or adjust hyperparameters.
