import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS

def get_embeddings(vocab):
    print("get_embedding")
    initializer = load_word_embeddings(vocab, FLAGS.embedding_dim)
    return tf.constant(initializer, name="word_embedding")

def get_char_embedding(charVocab):
    print("get_char_embedding")
    char_size = len(charVocab)
    embeddings = np.zeros((char_size, char_size), dtype='float32')
    for i in range(1, char_size):
        embeddings[i, i] = 1.0

    return tf.constant(embeddings, name="word_char_embedding")

def load_embed_vectors(fname, dim):
    # vectors = { 'the': [0.2911, 0.3288, 0.2002,...], ... }
    vectors = {}
    for line in open(fname, 'rt'):
        items = line.strip().split(' ')
        if len(items[0]) <= 0:
            continue
        vec = [float(items[i]) for i in range(1, dim+1)]
        vectors[items[0]] = vec

    return vectors

def load_word_embeddings(vocab, dim):
    vectors = load_embed_vectors(FLAGS.embedded_vector_file, dim)
    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, dim), dtype='float32')
    for word, code in vocab.items():
        if word in vectors:
            embeddings[code] = vectors[word]
        #else:
        #    embeddings[code] = np.random.uniform(-0.25, 0.25, dim) 

    return embeddings 


def lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, scope, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse) as vs:
        fw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        fw_cell  = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_prob)
        bw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        bw_cell  = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_prob)
        rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                inputs=inputs,
                                                                sequence_length=input_seq_len,
                                                                dtype=tf.float32)
        return rnn_outputs, rnn_states

def multi_lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, num_layer, scope, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse) as vs:
        multi_outputs = []
        multi_states = []
        cur_inputs = inputs
        for i_layer in range(num_layer):
            rnn_outputs, rnn_states = lstm_layer(cur_inputs, input_seq_len, rnn_size, dropout_keep_prob, scope+str(i_layer), scope_reuse)
            rnn_outputs = tf.concat(values=rnn_outputs, axis=2)
            multi_outputs.append(rnn_outputs)
            multi_states.append(rnn_states)
            cur_inputs = rnn_outputs

        # multi_layer_aggregation
        ml_weights = tf.nn.softmax(tf.get_variable("ml_scores", [num_layer, ], initializer=tf.constant_initializer(0.0)))

        multi_outputs = tf.stack(multi_outputs, axis=-1)   # [batch_size, max_len, 2*rnn_size(400), num_layer]
        max_len = multi_outputs.get_shape()[1].value
        dim = multi_outputs.get_shape()[2].value
        flattened_multi_outputs = tf.reshape(multi_outputs, [-1, num_layer])                         # [batch_size * max_len * 2*rnn_size(400), num_layer]
        aggregated_ml_outputs = tf.matmul(flattened_multi_outputs, tf.expand_dims(ml_weights, 1))    # [batch_size * max_len * 2*rnn_size(400), 1]
        aggregated_ml_outputs = tf.reshape(aggregated_ml_outputs, [-1, max_len, dim])                # [batch_size , max_len , 2*rnn_size(400)]

        return aggregated_ml_outputs


def cnn_layer(inputs, filter_sizes, num_filters, scope=None, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse):
        input_size = inputs.get_shape()[2].value

        outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv_{}".format(i)):
                w = tf.get_variable("w", [filter_size, input_size, num_filters])
                b = tf.get_variable("b", [num_filters])
            conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID") # [num_words, num_chars - filter_size, num_filters]
            h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
            pooled = tf.reduce_max(h, 1) # [num_words, num_filters]
            outputs.append(pooled)
    return tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]


def get_dist_mask(max_len):
    mask = np.zeros((max_len, max_len), dtype='float32')
    for i in range(max_len):
        for j in range(max_len):
            mask[i, j] = abs(i-j)
    return tf.constant(mask, name="distance_mask")

def self_attended(inputs, lens, max_len, scope, scope_reuse=False):
    # inputs: [batch_size, max_len, dim]
    with tf.variable_scope(scope, reuse=scope_reuse):
        self_similarity = tf.matmul(inputs, tf.transpose(inputs, perm=[0,2,1]), name='self_similarity_matrix')  # [batch_size, max_len, max_len]

        # Gaussian similarity
        dist_mask = get_dist_mask(max_len)  # [max_len, max_len]
        dist_w = tf.get_variable("dist_w", [1, ], initializer=tf.constant_initializer(0.1))
        dist_b = tf.get_variable("dist_b", [1, ], initializer=tf.constant_initializer(-0.1))
        dist_w = tf.clip_by_value(dist_w, clip_value_min=0, clip_value_max=1000)
        dist_b = tf.clip_by_value(dist_b, clip_value_min=-1000, clip_value_max=0)
        self_similarity = - tf.abs(dist_w * tf.square(dist_mask) + dist_b) + self_similarity
        print("establish gaussian prior")

        # masked similarity
        mask = tf.sequence_mask(lens, max_len, dtype=tf.float32)    # [batch_size, max_len]
        mask = tf.expand_dims(mask, 2)                              # [batch_size, max_len, 1]
        self_similarity = self_similarity * mask + -1e9 * (1-mask)  # [batch_size, max_len, max_len]

        attention_weight_for_self = tf.nn.softmax(self_similarity, dim=-1)  # [batch_size, max_len, max_len]
        attended_self = tf.matmul(attention_weight_for_self, inputs)        # [batch_size, max_len, dim]

    return attended_self

# context:  [batch_size, max_utter_num*max_utter_len, dim]
# response: [batch_size, max_response_num*max_response_len, dim]
# distance: [batch_size, max_response_num, max_utter_num]
def context_response_similarity_matrix(context, response, distance, max_utter_num, max_utter_len, max_response_num, max_response_len):

    c2 = tf.transpose(context, perm=[0, 2, 1])
    similarity = tf.matmul(response, c2, name='similarity_matrix')  # [batch_size, max_response_num*max_response_len, max_utter_num*max_utter_len]

    # exponential decay
    similarity = tf.reshape(similarity, [-1, max_response_num, max_response_len, max_utter_num, max_utter_len], name='similarity_RnumU')
    distance = tf.expand_dims(distance, 2)  # [batch_size, max_response_num, 1, max_utter_num]
    distance = tf.expand_dims(distance, -1) # [batch_size, max_response_num, 1, max_utter_num, 1]
    dist_w = tf.get_variable("dist_w", [1, ], initializer=tf.constant_initializer(0.0))
    dist_b = tf.get_variable("dist_b", [1, ], initializer=tf.constant_initializer(0.0))
    dist_w = tf.clip_by_value(dist_w, clip_value_min=0, clip_value_max=1000)
    dist_b = tf.clip_by_value(dist_b, clip_value_min=-1000, clip_value_max=0)

    similarity = - dist_w * distance + dist_b + similarity
    similarity = tf.reshape(similarity, [-1, max_response_num*max_response_len, max_utter_num*max_utter_len], name='similarity_decayed')
    print("establish exponential prior")

    return similarity

def attended_response(similarity_matrix, context, flattened_utters_len, max_utter_len, max_utter_num):
    # similarity_matrix:    [batch_size, max_response_num*response_len, max_utter_num*max_utter_len]
    # context:              [batch_size, max_utter_num*max_utter_len, dim]
    # flattened_utters_len: [batch_size* max_utter_num, ]
    
    # masked similarity_matrix
    mask_c = tf.sequence_mask(flattened_utters_len, max_utter_len, dtype=tf.float32)  # [batch_size*max_utter_num, max_utter_len]
    mask_c = tf.reshape(mask_c, [-1, max_utter_num*max_utter_len])                    # [batch_size, max_utter_num*max_utter_len]
    mask_c = tf.expand_dims(mask_c, 1)                                                # [batch_size, 1, max_utter_num*max_utter_len]
    similarity_matrix = similarity_matrix * mask_c + -1e9 * (1-mask_c)                # [batch_size, max_response_num*response_len, max_utter_num*max_utter_len]

    attention_weight_for_c = tf.nn.softmax(similarity_matrix, dim=-1)  # [batch_size, max_response_num*response_len, max_utter_num*max_utter_len]
    attended_response = tf.matmul(attention_weight_for_c, context)     # [batch_size, max_response_num*response_len, dim]

    return attended_response

def attended_context(similarity_matrix, response, flattened_responses_len, max_response_len, max_response_num):
    # similarity_matrix:    [batch_size, max_response_num*response_len, max_utter_num*max_utter_len]
    # response:             [batch_size, max_response_num*response_len, dim]
    # flattened_utters_len: [batch_size* max_response_num, ]

    # masked similarity_matrix
    mask_r = tf.sequence_mask(flattened_responses_len, max_response_len, dtype=tf.float32)  # [batch_size*max_response_num, response_len]
    mask_r = tf.reshape(mask_r, [-1, max_response_num*max_response_len])                    # [batch_size, max_response_num*response_len]
    mask_r = tf.expand_dims(mask_r, 2)                                           # [batch_size, max_response_num*response_len, 1]
    similarity_matrix = similarity_matrix * mask_r + -1e9 * (1-mask_r)           # [batch_size, max_response_num*response_len, max_utter_num*max_utter_len]

    attention_weight_for_r = tf.nn.softmax(tf.transpose(similarity_matrix, perm=[0,2,1]), dim=-1)  # [batch_size, max_utter_num*max_utter_len, max_response_num*response_len]
    attended_context = tf.matmul(attention_weight_for_r, response)                                 # [batch_size, max_utter_num*max_utter_len, dim]
    
    return attended_context


class U2U_IMN(object):
    def __init__(
      self, max_utter_len, max_utter_num, max_response_len, num_layer, vocab_size, embedding_size, vocab, rnn_size, maxWordLength, charVocab, l2_reg_lambda=0.0):

        max_response_num = 3

        self.utterances = tf.placeholder(tf.int32, [None, max_utter_num, max_utter_len], name="utterances")
        self.utterances_len = tf.placeholder(tf.int32, [None, max_utter_num], name="utterances_len")
        self.utters_num = tf.placeholder(tf.int32, [None], name="utterances_num")

        self.response = tf.placeholder(tf.int32, [None, max_response_num, max_response_len], name="response")
        self.response_len = tf.placeholder(tf.int32, [None, max_response_num], name="response_len")
        self.responses_num = tf.placeholder(tf.int32, [None], name="responses_num")

        self.distance = tf.placeholder(tf.float32, [None, max_response_num, max_utter_num], name="distance")
        self.target = tf.placeholder(tf.float32, [None], name="target")
        self.target_loss_weight = tf.placeholder(tf.float32, [None], name="target_weight")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.u_charVec = tf.placeholder(tf.int32, [None, max_utter_num, max_utter_len, maxWordLength], name="utterances_char")
        self.u_charLen = tf.placeholder(tf.int32, [None, max_utter_num, max_utter_len], name="utterances_char_len")

        self.r_charVec = tf.placeholder(tf.int32, [None, max_response_num, max_response_len, maxWordLength], name="response_char")
        self.r_charLen =  tf.placeholder(tf.int32, [None, max_response_num, max_response_len], name="response_char_len")

        l2_loss = tf.constant(1.0)

        # =============================== Embedding layer ===============================
        # word embedding
        with tf.name_scope("embedding"):
            W = get_embeddings(vocab)
            utterances_embedded = tf.nn.embedding_lookup(W, self.utterances)  # [batch_size, max_utter_num, max_utter_len,  word_dim]
            response_embedded = tf.nn.embedding_lookup(W, self.response)      # [batch_size, max_response_num, max_response_len, word_dim]
            print("original utterances_embedded: {}".format(utterances_embedded.get_shape()))
            print("original response_embedded: {}".format(response_embedded.get_shape()))

        with tf.name_scope('char_embedding'):
            char_W = get_char_embedding(charVocab)
            utterances_char_embedded = tf.nn.embedding_lookup(char_W, self.u_charVec)  # [batch_size, max_utter_num, max_utter_len,  maxWordLength, char_dim]
            response_char_embedded   = tf.nn.embedding_lookup(char_W, self.r_charVec)  # [batch_size, max_response_num, max_response_len, maxWordLength, char_dim]
            print("utterances_char_embedded: {}".format(utterances_char_embedded.get_shape()))
            print("response_char_embedded: {}".format(response_char_embedded.get_shape()))

        # char CNN
        char_dim = utterances_char_embedded.get_shape()[-1].value
        utterances_char_embedded = tf.reshape(utterances_char_embedded, [-1, maxWordLength, char_dim])  # [batch_size*max_utter_num*max_utter_len, maxWordLength, char_dim]
        response_char_embedded = tf.reshape(response_char_embedded, [-1, maxWordLength, char_dim])      # [batch_size*max_response_num*max_response_len, maxWordLength, char_dim]

        utterances_cnn_char_emb = cnn_layer(utterances_char_embedded, filter_sizes=[3, 4, 5], num_filters=50, scope="CNN_char_emb", scope_reuse=False) # [batch_size*max_utter_num*max_utter_len, emb]
        cnn_char_dim = utterances_cnn_char_emb.get_shape()[1].value
        utterances_cnn_char_emb = tf.reshape(utterances_cnn_char_emb, [-1, max_utter_num, max_utter_len, cnn_char_dim])                                # [batch_size, max_utter_num, max_utter_len, emb]

        response_cnn_char_emb = cnn_layer(response_char_embedded, filter_sizes=[3, 4, 5], num_filters=50, scope="CNN_char_emb", scope_reuse=True)      # [batch_size*max_response_num*max_response_len, emb]
        response_cnn_char_emb = tf.reshape(response_cnn_char_emb, [-1, max_response_num, max_response_len, cnn_char_dim])                              # [batch_size, max_response_num, max_response_len, emb]
                
        utterances_embedded = tf.concat(axis=-1, values=[utterances_embedded, utterances_cnn_char_emb])   # [batch_size, max_utter_num, max_utter_len, emb]
        response_embedded  = tf.concat(axis=-1, values=[response_embedded, response_cnn_char_emb])        # [batch_size, max_response_num, max_response_len, emb]
        utterances_embedded = tf.nn.dropout(utterances_embedded, keep_prob=self.dropout_keep_prob)
        response_embedded = tf.nn.dropout(response_embedded, keep_prob=self.dropout_keep_prob)
        print("utterances_embedded: {}".format(utterances_embedded.get_shape()))
        print("response_embedded: {}".format(response_embedded.get_shape()))


        # =============================== Encoding layer ===============================
        with tf.variable_scope("encoding_layer") as vs:
            rnn_scope_name = "bidirectional_rnn"
            emb_dim = utterances_embedded.get_shape()[-1].value
            flattened_utterances_embedded = tf.reshape(utterances_embedded, [-1, max_utter_len, emb_dim])  # [batch_size*max_utter_num, max_utter_len, emb]
            flattened_utterances_len = tf.reshape(self.utterances_len, [-1])                               # [batch_size*max_utter_num, ]
            flattened_responses_embedded = tf.reshape(response_embedded, [-1, max_response_len, emb_dim])  # [batch_size*max_response_num, max_response_len, emb]
            flattened_responses_len = tf.reshape(self.response_len, [-1])                                  # [batch_size*max_response_num, ]
            # 1. single_lstm_layer
            u_rnn_output, u_rnn_states = lstm_layer(flattened_utterances_embedded, flattened_utterances_len, rnn_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=False)
            utterances_output = tf.concat(axis=2, values=u_rnn_output)   # [batch_size*max_utter_num,  max_utter_len, rnn_size*2]
            r_rnn_output, r_rnn_states = lstm_layer(flattened_responses_embedded, flattened_responses_len, rnn_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=True)   # [batch_size, max_response_len, rnn_size(200)]
            response_output = tf.concat(axis=2, values=r_rnn_output)     # [batch_size*max_response_num, max_response_len, rnn_size*2]
            # 2. multi_lstm_layer
            # utterances_output = multi_lstm_layer(flattened_utterances_embedded, flattened_utterances_len, rnn_size, self.dropout_keep_prob, num_layer, rnn_scope_name, scope_reuse=False)
            # response_output = multi_lstm_layer(flattened_responses_embedded, flattened_responses_len, rnn_size, self.dropout_keep_prob, num_layer, rnn_scope_name, scope_reuse=True)
            # print("establish AHRE layers : {}".format(num_layer))
            
            # self-attention
            utterances_output = self_attended(utterances_output, flattened_utterances_len, max_utter_len, "utterance_self_attend")
            response_output = self_attended(response_output, flattened_responses_len, max_response_len, "response_self_attend")
        

        # =============================== Matching layer ===============================
        with tf.variable_scope("matching_layer") as vs:
            output_dim = utterances_output.get_shape()[-1].value
            utterances_output = tf.reshape(utterances_output, [-1, max_utter_num*max_utter_len, output_dim])   # [batch_size, max_utter_num*max_utter_len, rnn_size*2]
            response_output = tf.reshape(response_output, [-1, max_response_num*max_response_len, output_dim]) # [batch_size, max_response_num*max_response_len, rnn_size*2]

            # similarity = [batch_size, max_response_num*response_len, max_utter_num*max_utter_len]
            similarity = context_response_similarity_matrix(utterances_output, response_output, self.distance, max_utter_num, max_utter_len, max_response_num, max_response_len)
            attended_utterances_output = attended_context(similarity, response_output, flattened_responses_len, max_response_len, max_response_num)  # [batch_size, max_utter_num*max_utter_len, dim]
            attended_response_output = attended_response(similarity, utterances_output, flattened_utterances_len, max_utter_len, max_utter_num)      # [batch_size, max_response_num*response_len, dim]
            
            m_u = tf.concat(axis=2, values=[utterances_output, attended_utterances_output, tf.multiply(utterances_output, attended_utterances_output), utterances_output-attended_utterances_output])   # [batch_size, max_utter_num*max_utter_len, dim]
            m_r = tf.concat(axis=2, values=[response_output, attended_response_output, tf.multiply(response_output, attended_response_output), response_output-attended_response_output])               # [batch_size, max_response_num*response_len, dim]
            concat_dim = m_u.get_shape()[-1].value
            m_u = tf.reshape(m_u, [-1, max_utter_len, concat_dim])     # [batch_size*max_utter_num, max_utter_len, dim]
            m_r = tf.reshape(m_r, [-1, max_response_len, concat_dim])  # [batch_size*max_response_num, response_len, dim]
            print("establish matching between utterances and response")


        # =============================== Aggregation layer ===============================
        with tf.variable_scope("aggregation_layer") as vs:
            # context (maxAndState_max, maxAndState_state)
            rnn_scope_cross = 'bidirectional_rnn_cross'
            u_rnn_output_2, u_rnn_state_2 = lstm_layer(m_u, flattened_utterances_len, rnn_size, self.dropout_keep_prob, rnn_scope_cross, scope_reuse=False)
            utterances_output_cross = tf.concat(axis=2, values=u_rnn_output_2)   # [batch_size*max_utter_num, max_utter_len, 2*rnn_size]
            final_utterances_max = tf.reduce_max(utterances_output_cross, axis=1)
            final_utterances_state = tf.concat(axis=1, values=[u_rnn_state_2[0].h, u_rnn_state_2[1].h])
            final_utterances = tf.concat(axis=1, values=[final_utterances_max, final_utterances_state])
            final_utterances = tf.reshape(final_utterances, [-1, max_utter_num, output_dim*2])   # [batch_size, max_utter_num, 4*rnn_size]

            rnn_scope_aggre = "bidirectional_rnn_aggregation"
            final_utterances_output, final_utterances_state = lstm_layer(final_utterances, self.utters_num, rnn_size, self.dropout_keep_prob, rnn_scope_aggre, scope_reuse=False)
            final_utterances_output = tf.concat(axis=2, values=final_utterances_output)   # [batch_size, max_utter_num, 2*rnn_size]
            final_utterances_max = tf.reduce_max(final_utterances_output, axis=1)         # [batch_size, 2*rnn_size]
            final_utterances_state = tf.concat(axis=1, values=[final_utterances_state[0].h, final_utterances_state[1].h])  # [batch_size, 2*rnn_size]

            final_utterances =  tf.concat(axis=1, values=[final_utterances_max, final_utterances_state])
            print("establish rnn aggregation on response")

            # response
            r_rnn_output_2, r_rnn_state_2 = lstm_layer(m_r, flattened_responses_len, rnn_size, self.dropout_keep_prob, rnn_scope_cross, scope_reuse=True)
            response_output_cross = tf.concat(axis=2, values=r_rnn_output_2)     # [batch_size, max_response_len, rnn_size*2]
            final_response_max = tf.reduce_max(response_output_cross, axis=1)
            final_response_state = tf.concat(axis=1, values=[r_rnn_state_2[0].h, r_rnn_state_2[1].h])
            final_response = tf.concat(axis=1, values=[final_response_max, final_response_state])
            final_response = tf.reshape(final_response, [-1, max_response_num, output_dim*2])   # [batch_size, max_response_num, 4*rnn_size]
            
            # 1. RNN aggregation
            # final_response_output, final_response_state = lstm_layer(final_response, self.responses_num, rnn_size, self.dropout_keep_prob, rnn_scope_aggre, scope_reuse=True)
            # final_response_output = tf.concat(axis=2, values=final_response_output)   # [batch_size, max_response_num, 2*rnn_size]
            # final_response_max = tf.reduce_max(final_response_output, axis=1)         # [batch_size, 2*rnn_size]
            # final_response_state = tf.concat(axis=1, values=[final_response_state[0].h, final_response_state[1].h])  # [batch_size, 2*rnn_size]
            # final_response =  tf.concat(axis=1, values=[final_response_max, final_response_state])
            # print("establish rnn aggregation on response")

            # 2. position_attention aggregation
            res_weights = tf.get_variable("res_scores", [1, max_response_num], initializer=tf.constant_initializer(0.0))  # [1, max_response_num]
            res_mask = tf.sequence_mask(self.responses_num, max_response_num, dtype=tf.float32)  # [batch_size, max_response_num]
            res_weights = tf.nn.softmax(res_weights * res_mask + -1e9 * (1-res_mask))            # [batch_size, max_response_num]
            final_response_att = tf.matmul(tf.transpose(final_response, perm=[0,2,1]),  # [batch_size, dim, max_response_num]
                                           tf.expand_dims(res_weights, -1))             # [batch_size, max_response_num, 1]  ==> [batch_size, dim, 1]
            final_response_att = tf.squeeze(final_response_att, [-1])                   # [batch_size, dim]
            final_response = final_response_att   
            print("establish position attention aggregation on response")

            # 3. self_attention aggregation
            # proj_W = tf.get_variable("proj_W", [output_dim*2, 1], initializer=tf.orthogonal_initializer())
            # proj_b = tf.get_variable("proj_b", [1, ], initializer=tf.constant_initializer(0.0))
            # res_weights = tf.einsum('bij,jk->bik', final_response, proj_W) + proj_b              # [batch_size, max_response_num, 1]
            # res_weights = tf.squeeze(res_weights, [-1])                                          # [batch_size, max_response_num]
            # res_mask = tf.sequence_mask(self.responses_num, max_response_num, dtype=tf.float32)  # [batch_size, max_response_num]
            # res_weights = tf.nn.softmax(res_weights * res_mask + -1e9 * (1-res_mask))            # [batch_size, max_response_num]
            # final_response_att = tf.matmul(tf.transpose(final_response, perm=[0,2,1]),  # [batch_size, dim, max_response_num]
            #                                tf.expand_dims(res_weights, -1))             # [batch_size, max_response_num, 1]  ==> [batch_size, dim, 1]
            # final_response_att = tf.squeeze(final_response_att, [-1])                   # [batch_size, dim]
            # final_response = final_response_att
            # print("establish self project attention aggregation on response")

            final_response = tf.nn.dropout(final_response, keep_prob=self.dropout_keep_prob)

            joined_feature = tf.concat(axis=1, values=[final_utterances, final_response])  # [batch_size, 8*rnn_size(1600)]
            print("joined feature: {}".format(joined_feature.get_shape()))


        # =============================== Prediction layer ===============================
        with tf.variable_scope("prediction_layer") as vs:
            hidden_input_size = joined_feature.get_shape()[1].value
            hidden_output_size = 256
            regularizer = tf.contrib.layers.l2_regularizer(l2_reg_lambda)
            joined_feature = tf.nn.dropout(joined_feature, keep_prob=self.dropout_keep_prob)
            full_out = tf.contrib.layers.fully_connected(joined_feature, hidden_output_size,
                                                            activation_fn=tf.nn.relu,
                                                            reuse=False,
                                                            trainable=True,
                                                            scope="projected_layer")   # [batch_size, hidden_output_size(256)]
            full_out = tf.nn.dropout(full_out, keep_prob=self.dropout_keep_prob)

            last_weight_dim = full_out.get_shape()[1].value
            print("last_weight_dim: {}".format(last_weight_dim))
            bias = tf.Variable(tf.constant(0.1, shape=[1]), name="bias")
            s_w = tf.get_variable("s_w", shape=[last_weight_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.matmul(full_out, s_w) + bias       # [batch_size, 1]
            print("logits: {}".format(logits.get_shape()))
            
            logits = tf.squeeze(logits, [1])               # [batch_size, ]
            self.probs = tf.sigmoid(logits, name="prob")   # [batch_size, ]

            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.target)
            losses = tf.multiply(losses, self.target_loss_weight)
            self.mean_loss = tf.reduce_mean(losses, name="mean_loss") + l2_reg_lambda * l2_loss + sum(
                                                              tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.sign(self.probs - 0.5), tf.sign(self.target - 0.5))    # [batch_size, ]
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
