import os
import csv
import subprocess
import re
import random
import numpy as np
from numpy import linalg as LA


def read_in_shakespeare():
  '''Reads in the Shakespeare dataset processes it into a list of tuples.
     Also reads in the vocab and play name lists from files.

  Each tuple consists of
  tuple[0]: The name of the play
  tuple[1] A line from the play as a list of tokenized words.

  Returns:
    tuples: A list of tuples in the above format.
    document_names: A list of the plays present in the corpus.
    vocab: A list of all tokens in the vocabulary.
  '''

  tuples = []

  with open('will_play_text.csv') as f:
    csv_reader = csv.reader(f, delimiter=';')
    for row in csv_reader:
      play_name = row[1]
      line = row[5]
      line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split()
      line_tokens = [token.lower() for token in line_tokens]

      tuples.append((play_name, line_tokens))

  with open('vocab.txt') as f:
    vocab =  [line.strip() for line in f]

  with open('play_names.txt') as f:
    document_names =  [line.strip() for line in f]

  return tuples, document_names, vocab

def get_row_vector(matrix, row_id):
  return matrix[row_id, :]

def get_column_vector(matrix, col_id):
  return matrix[:, col_id]


def create_term_document_matrix(line_tuples, document_names, vocab):
  '''Returns a numpy array containing the term document matrix for the input lines.

  Inputs:
    line_tuples: A list of tuples, containing the name of the document and 
    a tokenized line from that document.
    document_names: A list of the document names
    vocab: A list of the tokens in the vocabulary

  Let m = len(vocab) and n = len(document_names).

  Returns:
    td_matrix: A mxn numpy array where the number of rows is the number of words
        and each column corresponds to a document. A_ij contains the
        frequency with which word i occurs in document j.
  '''

  vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
  docname_to_id = dict(zip(document_names, range(0, len(document_names))))
  tdm = np.zeros(shape=(len(vocab), len(document_names)))
  for line in line_tuples:
    doc = line[0]
    y_axis = docname_to_id.get(doc)
    for i in range(0, len(line[1])):
      x_axis = vocab_to_id.get(line[1][i])
      tdm[x_axis, y_axis] += 1

  # YOUR CODE HERE
  return tdm

def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
  '''Returns a numpy array containing the term context matrix for the input lines.

  Inputs:
    line_tuples: A list of tuples, containing the name of the document and 
    a tokenized line from that document.
    vocab: A list of the tokens in the vocabulary

  Let n = len(vocab).

  Returns:
    tc_matrix: A nxn numpy array where A_ij contains the frequency with which
        word j was found within context_window_size to the left or right of
        word i in any sentence in the tuples.
  '''

  vocab_to_id = dict(zip(vocab, range(0, len(vocab))))

  # YOUR CODE HERE
  tcm = np.zeros(shape=(len(vocab),len(vocab)))
  for line in line_tuples:
    sent= line[1]
    for i in range(0,len(line[1])):
      left = i - context_window_size
      right = i + context_window_size
      left = 0 if left < 0 else left
      right = len(sent) - 1 if right >= len(sent) else right
      for j in range(left, right+1):
        x_axis = vocab_to_id.get(sent[i],None)
        y_axis = vocab_to_id.get(sent[j],None)

        tcm[x_axis][y_axis] += 1

  return tcm


def create_PPMI_matrix(term_context_matrix):
  '''Given a term context matrix, output a PPMI matrix.

  Hint: Use numpy matrix and vector operations to speed up implementation.

  Input:
    term_context_matrix: A nxn numpy array, where n is
        the numer of tokens in the vocab.

  Returns: A nxn numpy matrix, where A_ij is equal to the
     point-wise mutual information between the ith word
     and the jth word in the term_context_matrix.
  '''

  # YOUR CODE HERE
  para = term_context_matrix.shape
  tcm = term_context_matrix.copy() + 1
  n = para[0]
  totals1 = np.sum(term_context_matrix) + n * n
  totals = np.tile(totals1, (n, para[1]))
  ppmi = np.zeros(para)
  rows1 = np.sum(term_context_matrix, axis=1) + n
  rows = np.tile(rows1, (n, 1)).T
  cols1 = np.sum(term_context_matrix, axis=0) + n
  cols = np.tile(cols1, (n, 1))
  dom = np.multiply(rows, cols)
  num = np.multiply(totals, tcm)
  ppmi_final = np.divide(num, dom)
  ppmi_final = np.log2(ppmi_final)
  ppmi_final = np.maximum(ppmi_final, 0)

  return ppmi_final


def create_tf_idf_matrix(term_document_matrix):
  '''Given the term document matrix, output a tf-idf weighted version.

  Hint: Use numpy matrix and vector operations to speed up implementation.

  Input:
    term_document_matrix: Numpy array where each column represents a document
    and each row, the frequency of a word in that document.

  Returns:
    A numpy array with the same dimension as term_document_matrix, where
    A_ij is weighted by the inverse document frequency of document h.
  '''

  # YOUR CODE HERE
  tf_idf_matrix = np.zeros(term_document_matrix.shape)
  df_raw = term_document_matrix.copy()
  tf_raw = term_document_matrix.copy()
  df_raw[df_raw > 0] = 1
  df = np.sum(df_raw, axis=1)
  docs = term_document_matrix.shape[1]
  idf = np.log(docs / df[df > 0])

  tf_raw[tf_raw > 0] = np.log10(tf_raw[tf_raw > 0]) + 1

  tf = tf_raw

  for row in range(idf.shape[0]):
    tf_idf_matrix[row] = tf[row] * idf[row]

  return tf_idf_matrix

def compute_cosine_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''
  cs = np.dot(vector1,vector2)/(LA.norm(vector1)*LA.norm(vector2))
  # YOUR CODE HERE
  return cs

def compute_jaccard_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''
  
  # YOUR CODE HERE
  # Tanimoto similarity
  num = np.minimum(vector1, vector2)
  dom = np.maximum(vector1, vector2)
  js = np.sum(num) / np.sum(dom)
  return js

def compute_dice_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''

  # YOUR CODE HERE
  upper = np.minimum(vector1,vector2)
  upper_sum = np.sum(upper) * 2
  dom = np.sum(vector1 + vector2)
  return upper_sum/dom

def rank_plays(target_play_index, term_document_matrix, similarity_fn):
  ''' Ranks the similarity of all of the plays to the target play.

  Inputs:
    target_play_index: The integer index of the play we want to compare all others against.
    term_document_matrix: The term-document matrix as a mxn numpy array.
    similarity_fn: Function that should be used to compared vectors for two
      documents. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.

  Returns:
    A length-n list of integer indices corresponding to play names,
    ordered by decreasing similarity to the play indexed by target_play_index
  '''
  
  # YOUR CODE HERE
  nums = term_document_matrix.shape[1]
  target = term_document_matrix.T[target_play_index,:]
  docs_ranking = {}
  result = []
  td_in_cols = term_document_matrix.T
  for i in range(nums):
          if i!= target_play_index:
              similarity_doc = similarity_fn(td_in_cols[i,:], target)
              docs_ranking[i] = similarity_doc
  sort_ranking = sorted(docs_ranking.items(),key=lambda item: item[1],reverse=True)
  for k,v in sort_ranking:
      result.append(k)
  return result

def rank_words(target_word_index, matrix, similarity_fn):
  ''' Ranks the similarity of all of the words to the target word.

  Inputs:
    target_word_index: The index of the word we want to compare all others against.
    matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.
    similarity_fn: Function that should be used to compared vectors for two word
      ebeddings. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.

  Returns:
    A length-n list of integer word indices, ordered by decreasing similarity to the 
    target word indexed by word_index
  '''
  # YOUR CODE HERE
  result =[]
  nums = matrix.shape[0]
  target = matrix[target_word_index,:]
  word_ranking = {}
  for i in range(nums):
      if i!= target_word_index:
        word_simi = similarity_fn(matrix[i,:],target)
        word_ranking[i] = word_simi
  for k,v in sorted(word_ranking.items(),key=lambda item: item[1],reverse=True):
      result.append(k)
  return result


if __name__ == '__main__':
  tuples, document_names, vocab = read_in_shakespeare()

  print('Computing term document matrix...')
  td_matrix = create_term_document_matrix(tuples, document_names, vocab)

  print('Computing tf-idf matrix...')
  tf_idf_matrix = create_tf_idf_matrix(td_matrix)

  print('Computing term context matrix...')
  tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=2)

  print('Computing PPMI matrix...')
  PPMI_matrix = create_PPMI_matrix(tc_matrix)

  random_idx = random.randint(0, len(document_names)-1)
  similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
  for sim_fn in similarity_fns:
    print('\nThe top most similar plays to "%s" using %s are:' % (document_names[random_idx], sim_fn.__qualname__))
    ranks = rank_plays(random_idx, td_matrix, sim_fn)
    for idx in range(0, 1):
      doc_id = ranks[idx]
      print('%d: %s' % (idx+1, document_names[doc_id]))

  word = 'juliet'
  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar words to "%s" using %s on term-document frequency matrix are:' % (word, sim_fn.__qualname__))
    ranks = rank_words(vocab_to_index[word], td_matrix, sim_fn)
    for idx in range(0, 10):
      word_id = ranks[idx]
      print('%d: %s' % (idx+1, vocab[word_id]))

  word = 'juliet'
  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar words to "%s" using %s on term-context frequency matrix are:' % (word, sim_fn.__qualname__))
    ranks = rank_words(vocab_to_index[word], tc_matrix, sim_fn)
    for idx in range(0, 10):
      word_id = ranks[idx]
      print('%d: %s' % (idx+1, vocab[word_id]))

  word = 'juliet'
  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar words to "%s" using %s on PPMI matrix are:' % (word, sim_fn.__qualname__))
    ranks = rank_words(vocab_to_index[word], tf_idf_matrix, sim_fn)
    for idx in range(0, 10):
      word_id = ranks[idx]
      print('%d: %s' % (idx+1, vocab[word_id]))

  word = 'juliet'
  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar words to "%s" using %s on PPMI matrix are:' % (word, sim_fn.__qualname__))
    ranks = rank_words(vocab_to_index[word], PPMI_matrix, sim_fn)
    for idx in range(0, 10):
      word_id = ranks[idx]
      print('%d: %s' % (idx+1, vocab[word_id]))
