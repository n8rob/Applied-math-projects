# markov_chains.py

import numpy as np
from scipy import linalg as la


def random_chain(n):
    """Create and return a transition matrix for a random Markov chain with
    'n' states. This should be stored as an nxn NumPy array.
    """
    trans_matrix = np.random.random((n,n))
    trans_matrix = trans_matrix / trans_matrix.sum(axis=0)
    return trans_matrix


def forecast(days):
    """Forecast the weather for the next given number of days given that today
        is hot.

    Parameters:
        days (int): number of days

    Returns: a list with the given number of entries containing the day-by-day
        weather predictions
    """
    #   We use the convention that 0 is hot and 1 is cold.
    #   This is a given probability matrix, transition.
    #   The i,j entry is the probability that the weather state will be j tomorrow,
    #   given that it's i today.

    transition = np.array([[0.7, 0.6], [0.3, 0.4]])

    #   Sample from a binomial distribution to choose a new state.

    #Assume the first day is hot (hot is 0).
    current_weather = 0
    #Initialize a list that we want to return:
    list = []
    for i in range(days):
        #We want 0 = hot and 1 = cold, so we test for coldness (using row 1)
        current_weather = np.random.binomial(1, transition[1,current_weather])
        list.append(current_weather)

    return list


def four_state_forecast(days):
    """Run a simulation for the weather over the specified number of days,
    with mild as the starting state, using the four-state Markov chain.
    Return a list containing the day-by-day results, not including the
    starting day.

    Examples:
        >>> four_state_forecast(3)
        [0, 1, 3]
        >>> four_state_forecast(5)
        [2, 1, 2, 1, 1]
    """
    #   Well I mean first we gotta make the probability array. (See the previous
    #   function for an explanation.) In this case we have 0 = hot, 1 = mild,
    #   2 = cold, 3 = freezing.
    #   We'll use a multinomial distribution this time since there are 4 states.

    transition = np.array([[.5,.3,.1,0],[.3,.3,.3,.3],[.2,.3,.4,.5],[0,.1,.2,.2]])
    #Assume that first puppy is mild (or 1)
    current_weather = 1
    #Initialize list
    list_ = [1]
    #Loop through each day
    for i in range(days):
        #A multinomial distribution will take the probabilities and choose an
        #outcome (where there will be a 1). We find the location of the 1 using
        #numpy.argmax().
        current_weather = np.argmax(np.random.multinomial(1,transition[:,list_[-1]]))
        list_.append(current_weather)
    return list_[1:]

def steady_state(A, tol=1e-20, N=40):
    """Compute the steady state of the transition matrix A.

    Inputs:
        A ((n,n) ndarray): A column-stochastic transition matrix.
        tol (float): The convergence tolerance.
        N (int): The maximum number of iterations to compute.

    Raises:
        ValueError: if the iteration does not converge within N steps.

    Returns:
        x ((n,) ndarray): The steady state distribution vector of A.
    """

    #   We can find the steady state, which is
    #   lim_(k->inf) x_k = lim_(k->inf) A^k@x_0
    #   , regardless of x_0.
    #   This is the general most likely vector.

    #Note that A has gotta be column-stochastic.
    #Generate random vector x_0
    x_k = np.random.random(len(A))
    #And normalize
    x_k = x_k/np.sum(x_k)
    #Now we just have to assign x_prev to something that will let it pass
    #through the loop
    x_prev = x_k + 2000*tol
    #Initialize number of iterations
    k = 0
    while la.norm( x_prev - x_k ) >= tol:
        #Check if k (number of iterations) exceeds N (meaning we've gone too
        #long)
        if k > N:
            raise ValueError("It appears A^k does not converge.")
        #Set the value for x_k before setting x equal to x_k+1
        x_prev = x_k
        #Now set x_k+1
        x_k = A@x_k
        #Iterate k
        k += 1
    return x_k


# Problems 5 and 6
class SentenceGenerator(object):
    """Markov chain creator for simulating English.

    Attributes:
        filename (string): name of file of corpus to read in
        matrix ((n,n) array): probability transistion matrix based off the corpus
            data. Matrix is square with length equal to number of unique words.
        dict (dict): maps a word to the number of its designated row/column in
            matrix
        list_of_states (list): list of all unique words

    Example:
        >>> yoda = SentenceGenerator("Yoda.txt")
        >>> print(yoda.babble())
        The dark side of loss is a path as one with you.
    """

    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """

        #First let's just store the filename as an attribute
        self.filename = filename

        #Let's read that file in as a list of strings (each string a sentence)
        with open(filename,'r') as myfile:
            list_of_sentences = myfile.readlines()

        #Eventually we want a list of total words and a ditionary mapping words
        #to numbers (indices in the matrix).
        list_states = []
        dict_of_words = {}
        #Initialize index for word before we loop through the corpus
        index_for_word = 0

        #Recall list_of_sentences is a list of strings.
        for i in range(len(list_of_sentences)):
            #Turn the current sentence from a string to a list of words, and
            #add $tart and $top tokens, which are needed later (so that our
            #generated English sentences don't go on forever).
            current_sentence = ["$tart"] + list_of_sentences[i].split() + ["$top"]
            #Now we loop through the words
            for j in range(len(current_sentence)):
                #Now we construct a dictionary that maps words to numbers
                if current_sentence[j] not in list_states:
                    dict_of_words[current_sentence[j]] = index_for_word
                    index_for_word += 1
                    list_states.append(current_sentence[j])

        #It will be useful to have the total number of words
        num_unique_words = index_for_word

        #Let's check if this is okay (the first item in the list of words
        #should be "$tart").
        if list_states[0] != "$tart":
            raise ValueError("List did not $tart correctly.")

        #Let's initialize this transition matrix that we need!
        matrix = np.zeros((num_unique_words,num_unique_words))

        #We need to update the matrix now
        #Loop thru sentences
        for i in range(len(list_of_sentences)):
            #And turn them into lists like we did before
            current_sentence = ["$tart"] + list_of_sentences[i].split() + ["$top"]
            for j in range(len(current_sentence)-1):
                #This is complicated. Update the entry in the row of "matrix"
                #corresponding to the j+1'th word in the sentence and the
                #column of "matrix" corresponding to the j'th word of the sentence
                #(in order to show the connection between those two words)...
                matrix[dict_of_words[current_sentence[j+1]]][dict_of_words[current_sentence[j]]] += 1

        #Now we make sure $top links to itself
        matrix[dict_of_words["$top"]][dict_of_words["$top"]] = 1

        #The matrix should be good now except for being stochastic
        matrix = matrix / matrix.sum(axis=0)
        #Now we should be good

        #Let's store the matrix as an attribute
        self.matrix = matrix
        #And everything else we need
        self.dict = dict_of_words
        self.list_states = list_states


    def babble(self):
        """Begin at the start sate and use the strategy from
        four_state_forecast() to transition through the Markov chain.
        Keep track of the path through the chain and the corresponding words.
        When the stop state is reached, stop transitioning and terminate the
        sentence. Return the resulting sentence as a single string.
        """
        #   We use pretty much the same strategy here that we did in the
        #   function four_state_forecast. So we'll use a multinomial
        #   distribution and look for the '1' it generates....

        #Let's start with the word = $tart
        current_word_num = 0
        #Initialize list of sentence we want to make
        sentence_list = []
        #Go till we hit $top
        while current_word_num != self.dict["$top"]:
            #Check the index of the 1 from the multinomial
            current_word_num = np.argmax(np.random.multinomial(1,self.matrix[:,current_word_num]))
            #Add the actual word (not the index) to the list. (Note we don't want to add "$tart" or "$top")
            if current_word_num != self.dict["$tart"] and current_word_num != self.dict["$top"]:
                sentence_list.append(self.list_states[current_word_num])

        #Now we gotta turn the list into a string
        sentence = " ".join(sentence_list)

        return sentence
