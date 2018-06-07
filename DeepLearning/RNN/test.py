#%%
import numpy as np

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function
    n_batches = len(int_text) // (batch_size * seq_length)
    
    xdata = np.array(int_text[: n_batches * batch_size * seq_length])
    ydata = np.array(int_text[1: n_batches * batch_size * seq_length + 1])
    
    ydata = np.zeros_like(xdata)
    ydata[:-1], ydata[-1] = xdata[1:], xdata[0]

    x_batches = np.split(xdata.reshape(batch_size, -1), n_batches, 1)
    y_batches = np.split(ydata.reshape(batch_size, -1), n_batches, 1)

    return np.array(list(zip(x_batches, y_batches)))

b = get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 2, 2)

print(type(b))
print(b)

#%%
def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # TODO: Implement Function
    id = np.random.randint(0, len(probabilities))
    if probabilities[id] < 0.6:
        id = np.argmax(probabilities)
    words = np.array([value for value in int_to_vocab.values()])
    return np.array(list(zip(probabilities, words)))[id][1]
test_probabilities = np.array([0.1, 0.8, 0.05, 0.05])
test_int_to_vocab = {word_i: word for word_i, word in enumerate(['this', 'is', 'a', 'test'])}

word = pick_word(test_probabilities, test_int_to_vocab)
print(word)