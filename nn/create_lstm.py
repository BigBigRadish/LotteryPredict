# -*- coding: utf-8 -*-
'''
Created on 2018年9月27日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import tensorflow as tf
import numpy as np
import preprocess.preprocess_and_save as ps
# Number of Epochs
num_epochs = 25
# Batch Size
batch_size = 32
# RNN Size
rnn_size = 1000
# Embedding Dimension Size
embed_dim = 1000
# Sequence Length
seq_length = 1
# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = 10

save_dir = './save'
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    inputs = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    LearningRate = tf.placeholder(tf.float32)
    return inputs, targets, LearningRate
def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)#num_units=embed_dim
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 2)
    InitialState = cell.zero_state(batch_size, tf.float32)
    InitialState = tf.identity(InitialState, name="initial_state")
    return cell, InitialState
def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Embedded input, embed_matrix)
    """
    embed_matrix = tf.Variable(tf.random_uniform([vocab_size, embed_dim], -1, 1))
    embed_layer = tf.nn.embedding_lookup(embed_matrix, input_data)
    return embed_layer, embed_matrix
def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    Outputs, final_State = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_State = tf.identity(final_State, name="final_state")
    return Outputs, final_State
def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState, embed_matrix)
    """
    embed_layer, embed_matrix = get_embed(input_data, vocab_size, embed_dim)
    Outputs, final_State = build_rnn(cell, embed_layer)
    logits = tf.layers.dense(Outputs, vocab_size)
    return logits, final_State, embed_matrix
def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    batchCnt = len(int_text) // (batch_size * seq_length)
    int_text_inputs = int_text[:batchCnt * (batch_size * seq_length)]
    int_text_targets = int_text[1:batchCnt * (batch_size * seq_length)+1]

    result_list = []
    x = np.array(int_text_inputs).reshape(1, batch_size, -1)
    y = np.array(int_text_targets).reshape(1, batch_size, -1)

    x_new = np.dsplit(x, batchCnt)
    y_new = np.dsplit(y, batchCnt)

    for ii in range(batchCnt):
        x_list = []
        x_list.append(x_new[ii][0])
        x_list.append(y_new[ii][0])
        result_list.append(x_list)

    return np.array(result_list)
from tensorflow.contrib import seq2seq
tf.reset_default_graph()
train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(ps.int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state, embed_matrix = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)
    
    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')
    
    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))
    #     cost = build_loss(logits, targets, vocab_size)
    
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embed_matrix), 1, keep_dims=True))
    normalized_embedding = embed_matrix / norm
    
    probs_embeddings = tf.nn.embedding_lookup(normalized_embedding, tf.squeeze(tf.argmax(probs, 2)))#np.squeeze(probs.argmax(2))
    probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_embedding))
    
    y_embeddings = tf.nn.embedding_lookup(normalized_embedding, tf.squeeze(targets))
    y_similarity = tf.matmul(y_embeddings, tf.transpose(normalized_embedding))
    
    #     data_moments = tf.reduce_mean(y_similarity, axis=0)
    #     sample_moments = tf.reduce_mean(probs_similarity, axis=0)
    similar_loss = tf.reduce_mean(tf.abs(y_similarity - probs_similarity))
    total_loss = cost + similar_loss
    
    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)
    
    # Gradient Clipping
    gradients = optimizer.compute_gradients(total_loss)  #cost
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]  #clip_by_norm
    train_op = optimizer.apply_gradients(capped_gradients)
    
    # Accuracy
    correct_pred = tf.equal(tf.argmax(probs, 2), tf.cast(targets, tf.int64))#logits <--> probs  tf.argmax(targets, 1) <--> targets
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    batches = get_batches(ps.int_text[:-(batch_size+1)], batch_size, seq_length)
    test_batches = get_batches(ps.int_text[-(batch_size+1):], batch_size, seq_length)
    top_k = 10
    topk_acc_list = []
    topk_acc = 0
    sim_topk_acc_list = []
    sim_topk_acc = 0
    
    range_k = 5
    floating_median_idx = 0
    floating_median_acc_range_k = 0
    floating_median_acc_range_k_list = []
    
    floating_median_sim_idx = 0
    floating_median_sim_acc_range_k = 0
    floating_median_sim_acc_range_k_list = []
    
    losses = {'train':[], 'test':[]}
    accuracies = {'accuracy':[], 'topk':[], 'sim_topk':[], 'floating_median_acc_range_k':[], 'floating_median_sim_acc_range_k':[]}
    
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch_i in range(num_epochs):
            state = sess.run(initial_state, {input_text: batches[0][0]})
    
            #训练的迭代，保存训练损失
            for batch_i, (x, y) in enumerate(batches):
                feed = {
                    input_text: x,
                    targets: y,
                    initial_state: state,
                    lr: learning_rate}
                train_loss, state, _ = sess.run([total_loss, final_state, train_op], feed)  #cost
                losses['train'].append(train_loss)
                
                # Show every <show_every_n_batches> batches
                if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                    print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        epoch_i,
                        batch_i,
                        len(batches),
                        train_loss))
                    
            #使用测试数据的迭代
            acc_list = []
            prev_state = sess.run(initial_state, {input_text: np.array([[1]])})#test_batches[0][0]
            for batch_i, (x, y) in enumerate(test_batches):
                # Get Prediction
                test_loss, acc, probabilities, prev_state = sess.run(
                    [total_loss, accuracy, probs, final_state],
                    {input_text: x, 
                     targets: y,
                     initial_state: prev_state})  #cost
                
                #保存测试损失和准确率
                acc_list.append(acc)
                losses['test'].append(test_loss)
                accuracies['accuracy'].append(acc)
    
                print('Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(test_batches),
                    test_loss))
        
                #利用嵌入矩阵和生成的预测计算得到相似度矩阵sim
                valid_embedding = tf.nn.embedding_lookup(normalized_embedding, np.squeeze(probabilities.argmax(2)))
                similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))
                sim = similarity.eval()
                
                #保存预测结果的Top K准确率和与预测结果距离最近的Top K准确率
                topk_acc = 0
                sim_topk_acc = 0
                for ii in range(len(probabilities)):
    
                    nearest = (-sim[ii, :]).argsort()[0:top_k]
                    if y[ii] in nearest:
                        sim_topk_acc += 1
    
                    if y[ii] in (-probabilities[ii]).argsort()[0][0:top_k]:
                        topk_acc += 1
    
                topk_acc = topk_acc / len(y)
                topk_acc_list.append(topk_acc)
                accuracies['topk'].append(topk_acc)
                
                sim_topk_acc = sim_topk_acc / len(y)
                sim_topk_acc_list.append(sim_topk_acc)
                accuracies['sim_topk'].append(sim_topk_acc)
    
                #计算真实值在预测值中的距离数据
                realInSim_distance_list = []
                realInPredict_distance_list = []
                for ii in range(len(probabilities)):
                    sim_nearest = (-sim[ii, :]).argsort()
                    idx = list(sim_nearest).index(y[ii])
                    realInSim_distance_list.append(idx)
                    
                    nearest = (-probabilities[ii]).argsort()[0]
                    idx = list(nearest).index(y[ii])
                    realInPredict_distance_list.append(idx)
                    
                print('真实值在预测值中的距离数据：')
                print('max distance : {}'.format(max(realInPredict_distance_list)))
                print('min distance : {}'.format(min(realInPredict_distance_list)))
                print('平均距离 : {}'.format(np.mean(realInPredict_distance_list)))
                print('距离中位数 : {}'.format(np.median(realInPredict_distance_list)))
                print('距离标准差 : {}'.format(np.std(realInPredict_distance_list)))
                
                print('真实值在预测值相似向量中的距离数据：')
                print('max distance : {}'.format(max(realInSim_distance_list)))
                print('min distance : {}'.format(min(realInSim_distance_list)))
                print('平均距离 : {}'.format(np.mean(realInSim_distance_list)))
                print('距离中位数 : {}'.format(np.median(realInSim_distance_list)))
                print('距离标准差 : {}'.format(np.std(realInSim_distance_list)))
    #             sns.distplot(realInPredict_distance_list, rug=True)  #, hist=False
                #plt.hist(np.log(realInPredict_distance_list), bins=50, color='steelblue', normed=True )
    
                #计算以距离中位数为中心，范围K为半径的准确率
                floating_median_sim_idx = int(np.median(realInSim_distance_list))
                floating_median_sim_acc_range_k = 0
            
                floating_median_idx = int(np.median(realInPredict_distance_list))
                floating_median_acc_range_k = 0
                for ii in range(len(probabilities)):
                    nearest_floating_median = (-probabilities[ii]).argsort()[0][floating_median_idx - range_k:floating_median_idx + range_k]
                    if y[ii] in nearest_floating_median:
                        floating_median_acc_range_k += 1
                        
                    nearest_floating_median_sim = (-sim[ii, :]).argsort()[floating_median_sim_idx - range_k:floating_median_sim_idx + range_k]
                    if y[ii] in nearest_floating_median_sim:
                        floating_median_sim_acc_range_k += 1
                        
                floating_median_acc_range_k = floating_median_acc_range_k / len(y)
                floating_median_acc_range_k_list.append(floating_median_acc_range_k)
                accuracies['floating_median_acc_range_k'].append(floating_median_acc_range_k)
                
                floating_median_sim_acc_range_k = floating_median_sim_acc_range_k / len(y)
                floating_median_sim_acc_range_k_list.append(floating_median_sim_acc_range_k)
                accuracies['floating_median_sim_acc_range_k'].append(floating_median_sim_acc_range_k)
                
            print('Epoch {:>3} floating median sim range k accuracy {} '.format(epoch_i, np.mean(floating_median_sim_acc_range_k_list)))#:.3f
            print('Epoch {:>3} floating median range k accuracy {} '.format(epoch_i, np.mean(floating_median_acc_range_k_list)))#:.3f
            print('Epoch {:>3} similar top k accuracy {} '.format(epoch_i, np.mean(sim_topk_acc_list)))#:.3f
            print('Epoch {:>3} top k accuracy {} '.format(epoch_i, np.mean(topk_acc_list)))#:.3f
            print('Epoch {:>3} accuracy {} '.format(epoch_i, np.mean(acc_list)))#:.3f
            
        # Save Model
        saver.save(sess, save_dir)  #, global_step=epoch_i
        print('Model Trained and Saved')
        embed_mat = sess.run(normalized_embedding)
