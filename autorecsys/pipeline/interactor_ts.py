from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.python.util import nest
from autokeras.engine.block import Block
from autokeras import keras_layers
from typing import Optional


class LSTMInteractor(Block):
    """
    """

    def __init__(self, embed_dim=None, embed_dim2=None, num_layers=None, dropout_rate=None, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.embed_dim2 = embed_dim2
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                'embed_dim': self.embed_dim,
                # 'embed_dim2': self.embed_dim2,
                # 'num_layers': self.num_layers,
                # 'dropout_rate': self.dropout_rate
            })

        return state

    def set_state(self, state):
        super().set_state(state)
        self.embed_dim = state['embed_dim']
        # self.embed_dim2 = state['embed_dim2']
        # self.num_layers = state['num_layers']
        # self.dropout_rate = state['dropout_rate']

    def build(self, hp, inputs=None):
        output_node = nest.flatten(inputs)

        # expland all the tensors to 3D tensor
        # input_node = inputs
        output_node = tf.concat(output_node, axis=0)
        # embedding_dim = self.embed_dim or hp.Choice('embedding_dim', [2, 8, 16], default=8)
        embedding_dim = self.embed_dim or hp.Choice('embedding_dim', [32, 128, 256, 512], default=128)
        # embedding_dim2 = self.embed_dim2 or hp.Choice('embedding_dim', [4, 8, 16], default=8)
        # num_layers = self.num_layers or hp.Choice('num_layers', [0, 2, 4], default=2)
        # dropout_rate = self.dropout_rate or hp.Choice('dropout_rate',
        #                                               [0.0, 0.25, 0.5],
        #                                               default=0)
        # for i in range(num_layers):
        #     output_node = tf.keras.layers.LSTM(embedding_dim, dropout=dropout_rate, recurrent_dropout=0.2, return_sequences=True)(output_node)
        # output_node = tf.keras.layers.LSTM(embedding_dim2, dropout=dropout_rate, recurrent_dropout=0.2)(output_node)
        # for i in range(num_layers):
        for i in range(2):
            output_node = tf.keras.layers.LSTM(embedding_dim, return_sequences=True)(output_node)
        output_node = tf.keras.layers.LSTM(16)(output_node)

        return output_node



class GRUInteractor(Block):
    """
    """

    def __init__(self, embed_dim=None, embed_dim2=None, num_layers=None, dropout_rate=None, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        # self.embed_dim2 = embed_dim2
        self.num_layers = num_layers
        # self.dropout_rate = dropout_rate

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                'embed_dim': self.embed_dim,
                # 'num_layers': self.num_layers,
                # 'dropout_rate': self.dropout_rate
            })

        return state

    def set_state(self, state):
        super().set_state(state)
        # self.embed_dim = state['embed_dim']
        # self.num_layers = state['num_layers']
        # self.dropout_rate = state['dropout_rate']

    def build(self, hp, inputs=None):
        output_node = nest.flatten(inputs)
        # expland all the tensors to 3D tensor
        for idx, node in enumerate(output_node):
            if len(node.shape) == 1:
                output_node[idx] = tf.expand_dims(tf.expand_dims(node, -1), -1)
            elif len(node.shape) == 2:
                output_node[idx] = tf.expand_dims(node, 1)
            elif len(node.shape) > 3:
                raise ValueError(
                    "Unexpected inputs dimensions %d, expect to be smaller than 3" % len(node.shape)
                )

        # input_node = inputs
        output_node = tf.concat(output_node, axis=-1)
        # embedding_dim = self.embed_dim or hp.Choice('embedding_dim', [2, 8, 16], default=8)
        embedding_dim = self.embed_dim or hp.Choice('embedding_dim', [256, 512], default=512)
        # embedding_dim2 = self.embed_dim2 or hp.Choice('embedding_dim', [4, 8, 16], default=8)
        #
        # num_layers = self.num_layers or hp.Choice('num_layers', [0, 2, 4], default=2)
        # dropout_rate = self.dropout_rate or hp.Choice('dropout_rate',
        #                                               [0.0, 0.10, 0.20],
        #                                               default=0)
        # for i in range(num_layers):
        for i in range(2):
            output_node = tf.keras.layers.GRU(embedding_dim, return_sequences=True)(output_node)
        output_node = tf.keras.layers.GRU(16)(output_node)


        return output_node


class CONV(Block):
    """
    """

    def __init__(self, embed_dim=None, num_layers=None, dropout_rate=None, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        # self.num_layers = num_layers
        # self.dropout_rate = dropout_rate

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                'embed_dim': self.embed_dim,
                # 'num_layers': self.num_layers,
                # 'dropout_rate': self.dropout_rate
            })

        return state

    def set_state(self, state):
        super().set_state(state)
        self.embed_dim = state['embed_dim']
        # self.num_layers = state['num_layers']
        # self.dropout_rate = state['dropout_rate']

    def build(self, hp, inputs=None):
        output_node = nest.flatten(inputs)
        print('\n\nffffn\n\n')

        # expland all the tensors to 3D tensor
        for idx, node in enumerate(output_node):
            if len(node.shape) == 1:
                output_node[idx] = tf.expand_dims(tf.expand_dims(node, -1), -1)
            elif len(node.shape) == 2:
                output_node[idx] = tf.expand_dims(node, 1)
            elif len(node.shape) > 3:
                raise ValueError(
                    "Unexpected inputs dimensions %d, expect to be smaller than 3" % len(node.shape)
                )

        # input_node = inputs
        output_node = tf.concat(output_node, axis=1)

        embedding_dim = self.embed_dim or hp.Choice('embedding_dim', [32, 64, 128], default=64)
        # num_layers = self.num_layers or hp.Choice('num_layers', [1, 2, 3], default=2)
        # dropout_rate = self.dropout_rate or hp.Choice('dropout_rate',
        #                                               [0.0, 0.25, 0.5],
        #                                               default=0)

        for i in range(2):
            output_node = tf.keras.layers.Conv2D(2, 3, padding='valid')(output_node)
        output_node = tf.keras.layers.Conv2D(2, 3, padding='valid')(output_node)



        return output_node


class TokenAndPositionEmbedding(Block):
    def __init__(self, maxlen, vocab_size, feat=None, embed_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.maxlen = maxlen
        self.feat = feat

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)

        # expland all the tensors to 3D tensor
        # for idx, node in enumerate(output_node):
        #     if len(node.shape) == 1:
        #         output_node[idx] = tf.expand_dims(tf.expand_dims(node, -1), -1)
        #     elif len(node.shape) == 2:
        #         output_node[idx] = tf.expand_dims(node, 1)
        #     elif len(node.shape) > 3:
        #         raise ValueError(
        #             "Unexpected inputs dimensions %d, expect to be smaller than 3" % len(node.shape)
        #         )

        inputs = input_node[0]
        print('input shapee', inputs.shape)
        # pos_emb = input_node[1]
        # inputs = tf.cast(input_node[0], tf.int64)
        # pos_emb = tf.cast(input_node[1], tf.int64)

        # target encoding
        ## Create a movie embedding encoder
        # movie_vocabulary = self.feat["movieID"]
        # movie_embedding_dims = 32
        #
        # movie_index_lookup = IntegerLookup(
        #     vocabulary=movie_vocabulary,
        #     num_oov_indices=0,
        # )
        #
        # movie_embedding_encoder = tf.keras.layers.Embedding(
        #     input_dim=len(movie_vocabulary),
        #     output_dim=movie_embedding_dims,
        # )
        #
        #
        # def encode_movie(movie_id):
        #     # Convert the string input values into integer indices.
        #     movie_idx = movie_index_lookup(movie_id)
        #     movie_embedding = movie_embedding_encoder(movie_idx)
        #     encoded_movie = movie_embedding
        #     # if include_movie_features:
        #     #     movie_genres_vector = movie_genres_lookup(movie_idx)
        #     #     encoded_movie = movie_embedding_processor(
        #     #         layers.concatenate([movie_embedding, movie_genres_vector])
        #     #     )
        #     return encoded_movie
        #
        # target_movie_id = pos_emb
        # pos_emb = encode_movie(target_movie_id)
        #
        # ## Encoding sequence movie_ids.
        # output_node = encode_movie(inputs)

        # rating = input_node[1]
        #
        # rating = tf.keras.layers.Embedding(input_dim=self.maxlen, output_dim=32)(rating)

        maxlen = tf.shape(inputs)[-1]

        # embedding_dim = self.embed_dim or hp.Choice('embedding_dim', [32, 64, 128], default=64)
        embedding_dim = 32
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = tf.keras.layers.Embedding(input_dim=self.maxlen, output_dim=embedding_dim)(positions)
        output_node = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=embedding_dim)(inputs)
        # pos_emb = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=embedding_dim)(pos_emb)


        output_node = output_node + positions


        encoded_transformer_features = []
        print(output_node.shape)

        # output_node = tf.concat([output_node, pos_emb], axis=1)



        # for encoded_movie in tf.unstack(
        #         output_node, axis=1
        # ):
        #     encoded_transformer_features.append(tf.expand_dims(encoded_movie, 1))
        #
        # print(len(encoded_transformer_features))
        # encoded_transformer_features.append(pos_emb)
        #
        # output_node = tf.keras.layers.concatenate(
        #     encoded_transformer_features, axis=1
        # )
        print(output_node.shape)
        # import sys
        # sys.exit()


        # mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        # output_node = output_node * rating

        # mask = tf.expand_dims(tf.cast(tf.not_equal(inputs, 0), tf.float32), -1)
        #
        # output_node *= mask


        return output_node

class TransformerBlock(Block):
    def __init__(self, embed_dim=None, num_heads=None, ff_dim=None, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate


    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)
        # expland all the tensors to 3D tensor
        # for idx, node in enumerate(input_node):
        #     if len(node.shape) == 1:
        #         input_node[idx] = tf.expand_dims(tf.expand_dims(node, -1), -1)
        #     elif len(node.shape) == 2:
        #         input_node[idx] = tf.expand_dims(node, 1)
        #     elif len(node.shape) > 3:
        #         raise ValueError(
        #             "Unexpected inputs dimensions %d, expect to be smaller than 3" % len(node.shape))
        # output_node = [tf.keras.layers.Dense(32)(node)
        #                if node.shape[2] != 32 else node for node in input_node]
        # output_node = tf.concat(output_node, axis=1)

        inputs = input_node[0]
        # mask = input_node[1]
        # mask = tf.expand_dims(tf.cast(tf.not_equal(mask, 0), tf.float32), -1)

        #embedding_dim = self.embed_dim or hp.Choice('embedding_dim', [32, 64, 128], default=32)
        # number_heads = self.num_heads or hp.Choice('num_heads', [1, 2, 3], default=1)
        ff_dims = self.ff_dim or hp.Choice('ff_dims', [16, 32], default=16)

        embedding_dim = 32

        # multihead attention
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=embedding_dim, value_dim=embedding_dim)(inputs, inputs)
        attn_output = tf.keras.layers.Dropout(self.rate)(attn_output)

        # add&norm
        out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
        # feed forward
        out2 = tf.keras.layers.Dense(ff_dims, activation='relu')(out1)
        ffn_output1 = tf.keras.layers.Dense(embedding_dim)(out2)

        # inputs = tf.keras.layers.Masking()(ffn_output)

        # ffn_output = tf.keras.Sequential(
        #     [tf.keras.layers.Dense(ff_dims, activation="relu"), tf.keras.layers.Dense(embedding_dim),]
        # )(out1)

        # add&norm
        ffn_output = tf.keras.layers.Dropout(self.rate)(ffn_output1)
        output_node = tf.keras.layers.LayerNormalization()(out1 + ffn_output)
        # output_node = tf.keras.layers.GlobalAveragePooling1D()(output_node)
        return output_node


class Transformers2(Block):
    def __init__(self, maxlen, vocab_size, embed_dim=None, num_heads=None, ff_dim=None, rate=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.maxlen = maxlen
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate


    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        maxlen = tf.shape(inputs)[-1]

        embedding_dim = self.embed_dim or hp.Choice('embedding_dim', [32, 64, 128, 256], default=128)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = tf.keras.layers.Embedding(input_dim=self.maxlen, output_dim=embedding_dim)(positions)
        output_node = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=embedding_dim)(inputs[0])

        number_heads = self.num_heads or hp.Choice('num_heads', [1, 2, 3], default=2)
        ff_dims = self.ff_dim or hp.Choice('ff_dims', [32, 64, 128, 256], default=128)
        rate = self.rate or hp.Choice('dropout_rate', [0.0, 0.25, 0.5], default=0)

        inputs = output_node + positions

        training = True
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=number_heads, key_dim=embedding_dim)(inputs,inputs)
        attn_output = tf.keras.layers.Dropout(rate)(attn_output, training=training)
        out1 = tf.keras.layers.LayerNormalization()(inputs + attn_output)
        ffn_output = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dims, activation="relu"), tf.keras.layers.Dense(embedding_dim),]
        )(out1)
        ffn_output = tf.keras.layers.Dropout(rate)(ffn_output, training=training)
        output_node = tf.keras.layers.LayerNormalization()(out1 + ffn_output)

        return output_node


class BPRMultiply(Block):
    """
    """
    def __init__(self,
                 embedding_dim=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.fixed_params = []
        self.tunable_candidates = ['embedding_dim']
        self.embedding_dim = embedding_dim

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                'embedding_dim': self.embedding_dim,
            })
        return state

    def set_state(self, state):
        super().set_state(state)
        self.embedding_dim = state['embedding_dim']

    def build(self, hp, inputs=None):
        print(inputs[0].shape)
        input_node = [tf.keras.layers.Flatten()(node) if len(node.shape) > 2 else node for node in nest.flatten(inputs)]
        print(input_node[0].shape)
        from keras import backend as K
        output_node = tf.reduce_prod(input_node, axis=0, keepdims=True)

        output_node = tf.keras.layers.Dense(1, activation=tf.nn.softmax)(output_node)
        output_node = tf.keras.layers.Activation('sigmoid')(output_node)
        print(output_node.shape)


        # output_node = tf.keras.layers.multiply(input_node)
        # output_node = K.sum(output_node, axis=-1, keepdims=True)
        print('zzzz', output_node.shape)

        return output_node

class BPRInteractor(Block):
    """
    """
    def __init__(self,
                 embedding_dim=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.fixed_params = []
        self.tunable_candidates = ['embedding_dim']
        self.embedding_dim = embedding_dim

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                'embedding_dim': self.embedding_dim,
            })
        return state

    def set_state(self, state):
        super().set_state(state)
        self.embedding_dim = state['embedding_dim']

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)
        uid = input_node[0]
        pid = input_node[1]
        nid = input_node[2]

        output_node = 1.0 - tf.sigmoid(tf.reduce_sum((uid*pid), axis=1, keepdims=True)-tf.reduce_sum(uid*nid, axis=1, keepdims=True))

        from keras import backend as K
        # output_node = -tf.reduce_sum(tf.keras.layers.subtract(input_node), axis=1, keepdims=True)

        # output_node = 1-K.sigmoid(tf.keras.layers.subtract(input_node))
        # output_node = tf.keras.layers.subtract(input_node)
        # print(output_node.shape)
        # print('outputtt', output_node)

        return output_node

class ElementwiseInteraction(Block):
    """Module for element-wise operation. this block includes the element-wise sum, average, multiply (Hadamard
        product), max, and min.
        The default operation is element-wise sum.
    # Attributes:
        elementwise_type("str"):  Can be used to select the element-wise operation. the default value is None. If the
        value of this parameter is None, the block can select the operation for the element-wise sum, average, multiply,
        max, and min, according to the search algorithm.
    """

    def __init__(self,
                 elementwise_type=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.elementwise_type = elementwise_type

    def get_state(self):
        state = super().get_state()
        state.update({
            'elementwise_type': self.elementwise_type})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.elementwise_type = state['elementwise_type']

    def build(self, hp, inputs=None):
        input_node = [tf.keras.layers.Flatten()(node) if len(node.shape) > 2 else node for node in nest.flatten(inputs)]

        shape_set = set()
        for node in input_node:
            shape_set.add(node.shape[1])  # shape[0] is the batch size
        if len(shape_set) > 1:
            # raise ValueError("Inputs of ElementwiseInteraction should have same dimension.")
            min_len = min( shape_set )

            input_node = [tf.keras.layers.Dense(min_len)(node)
                          if node.shape[1] != min_len else node for node in input_node]

        elementwise_type = self.elementwise_type or hp.Choice('elementwise_type',
                                                              ["sum", "average", "multiply", "max", "min"],
                                                              default='average')
        if elementwise_type == "sum":
            output_node = tf.add_n(input_node)
        elif elementwise_type == "average":
            output_node = tf.reduce_mean(input_node, axis=0)
        elif elementwise_type == "multiply":
            output_node = tf.reduce_prod(input_node, axis=0)
        elif elementwise_type == "max":
            output_node = tf.reduce_max(input_node, axis=[0])
        elif elementwise_type == "min":
            output_node = tf.reduce_min(input_node, axis=[0])
        else:
            output_node = tf.add_n(input_node)
        return output_node

class BertBlock(Block):
    """Block for Pre-trained BERT.
    The input should be sequence of sentences. The implementation is derived from
    this [example](https://www.tensorflow.org/official_models/fine_tuning_bert)
    # Example
    ```python
        # Using the Transformer Block with AutoModel.
        import autokeras as ak
        from autokeras import BertBlock
        from tensorflow.keras import losses
        input_node = ak.TextInput()
        output_node = BertBlock(max_sequence_length=128)(input_node)
        output_node = ak.ClassificationHead()(output_node)
        clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=10)
    ```
    # Arguments
        max_sequence_length: Int. The maximum length of a sequence that is
            used to train the model.
    """

    def __init__(
        self,
        max_sequence_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_sequence_length = max_sequence_length

    def get_config(self):
        config = super().get_config()
        config.update({"max_sequence_length": self.max_sequence_length})
        return config

    def build(self, hp, inputs=None):
        input_tensor = nest.flatten(inputs)[0]

        max_sequence_length = self.max_sequence_length or hp.Choice(
            "max_seq_len", [128, 256, 512], default=128
        )

        tokenizer_layer = keras_layers.BertTokenizer(
            max_sequence_length=max_sequence_length
        )
        output_node = tokenizer_layer(input_tensor)

        bert_encoder = keras_layers.BertEncoder()

        output_node = bert_encoder(output_node)
        bert_encoder.load_pretrained_weights()

        return output_node


class MLPInteraction(Block):
    """Module for MLP operation. This block can be configured with different layer, unit, and other settings.
    # Attributes:
        units (int). The units of all layer in the MLP block.
        num_layers (int). The number of the layers in the MLP block.
        use_batchnorm (Boolean). Use batch normalization or not.
        dropout_rate(float). The value of drop out in the last layer of MLP.
    """

    def __init__(self,
                 units=None,
                 num_layers=None,
                 use_batchnorm=None,
                 dropout_rate=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

    def get_state(self):
        state = super().get_state()
        state.update({
            'units': self.units,
            'num_layers': self.num_layers,
            'use_batchnorm': self.use_batchnorm,
            'dropout_rate': self.dropout_rate})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.units = state['units']
        self.num_layers = state['num_layers']
        self.use_batchnorm = state['use_batchnorm']
        self.dropout_rate = state['dropout_rate']

    def build(self, hp, inputs=None):
        input_node = [tf.keras.layers.Flatten()(node) if len(node.shape) > 2 else node for node in nest.flatten(inputs)]
        output_node = tf.concat(input_node, axis=1)
        num_layers = self.num_layers or hp.Choice('num_layers', [1, 2, 3], default=2)
        use_batchnorm = self.use_batchnorm
        if use_batchnorm is None:
            use_batchnorm = hp.Choice('use_batchnorm', [True, False], default=False)
        dropout_rate = self.dropout_rate or hp.Choice('dropout_rate',
                                                      [0.0, 0.25, 0.5],
                                                      default=0)

        for i in range(num_layers):
            units = self.units or hp.Choice(
                'units_{i}'.format(i=i),
                [32, 64, 128],
                default=32)
            output_node = tf.keras.layers.Dense(units)(output_node)
            if use_batchnorm:
                output_node = tf.keras.layers.BatchNormalization()(output_node)
            output_node = tf.keras.layers.ReLU()(output_node)
            output_node = tf.keras.layers.Dropout(dropout_rate)(output_node)
        return output_node





class HyperInteraction3d(Block):
    """Module for selecting different block. This block includes can select different blocks in the interactor.
    # Attributes:
        meta_interator_num (str). The total number of the meta interoctor block.
        interactor_type (str).  The type of interactor used in this block.
    """

    def __init__(self, meta_interator_num=None, interactor_type=None, **kwargs):
        super().__init__(**kwargs)
        self.meta_interator_num = meta_interator_num
        self.interactor_type = interactor_type
        self.name2interactor = {
            "LSTMInteractor": LSTMInteractor,
            "GRUInteractor": GRUInteractor,
            "SelfAttentionInteractor": SelfAttentionInteraction
        }

    def get_state(self):
        state = super().get_state()
        state.update({
            "interactor_type": self.interactor_type,
            "meta_interator_num": self.meta_interator_num,
            "name2interactor": {
                # "MLPInteraction": MLPInteraction,
                # "ElementwiseInteraction": ElementwiseInteraction,
                # "LSTMInteractor": LSTMInteractor,
                "GRUInteractor": GRUInteractor,
                # "BiLSTMInteractor": Bi_LSTMInteractor,
                # "BiGRUInteractor": Bi_GRUInteractor,
            }
        })
        return state

    def set_state(self, state):
        super().set_state(state)
        self.interactor_type = state['interactor_type']
        self.meta_interator_num = state['meta_interator_num']

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)
        meta_interator_num = self.meta_interator_num or hp.Choice('meta_interator_num',
                                                                  [1, 2, 3, 4, 5, 6],
                                                                  default=3)
        interactors_name = []
        for idx in range(meta_interator_num):
            tmp_interactor_type = self.interactor_type or hp.Choice('interactor_type_' + str(idx),
                                                                    list(self.name2interactor.keys()),
                                                                    default='GRUInteractor')
            interactors_name.append(tmp_interactor_type)
        outputs = [self.name2interactor[interactor_name]().build(hp, input_node)
                                for interactor_name in interactors_name]

        # DO WE REALLY NEED TO CAT THEM?
        outputs = [tf.keras.layers.Flatten()(node) if len(node.shape) > 2 else node for node in outputs]
        outputs = tf.concat(outputs, axis=1)
        return outputs


class HyperInteraction2d(Block):
    """Module for selecting different block. This block includes can select different blocks in the interactor.
    # Attributes:
        meta_interator_num (str). The total number of the meta interoctor block.
        interactor_type (str).  The type of interactor used in this block.
    """

    def __init__(self, meta_interator_num=None, interactor_type=None, **kwargs):
        super().__init__(**kwargs)
        self.meta_interator_num = meta_interator_num
        self.interactor_type = interactor_type
        self.name2interactor = {
            # "MLPInteraction": MLPInteraction,
            # "ElementwiseInteraction": ElementwiseInteraction,
            # "LSTMInteractor": LSTMInteractor,
            # "GRUInteractor": GRUInteractor,
            # "BiLSTMInteractor": Bi_LSTMInteractor,
            # "BiGRUInteractor": Bi_GRUInteractor,
            "TransformerInteractor": TransformerBlock,
            "BertInteractor": BertBlock,
        }

    def get_state(self):
        state = super().get_state()
        state.update({
            "interactor_type": self.interactor_type,
            "meta_interator_num": self.meta_interator_num,
            "name2interactor": {
                # "MLPInteraction": MLPInteraction,
                # "ElementwiseInteraction": ElementwiseInteraction,
                # "LSTMInteractor": LSTMInteractor,
                # "GRUInteractor": GRUInteractor,
                # "BiLSTMInteractor": Bi_LSTMInteractor,
                # "BiGRUInteractor": Bi_GRUInteractor,
                "TransformerInteractor": TransformerBlock,
                "BertInteractor": BertBlock,
            }
        })
        return state

    def set_state(self, state):
        super().set_state(state)
        self.interactor_type = state['interactor_type']
        self.meta_interator_num = state['meta_interator_num']

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)
        meta_interator_num = self.meta_interator_num or hp.Choice('meta_interator_num',
                                                                  [1, 2, 3, 4, 5, 6],
                                                                  default=3)
        interactors_name = []
        for idx in range(meta_interator_num):
            tmp_interactor_type = self.interactor_type or hp.Choice('interactor_type_' + str(idx),
                                                                    list(self.name2interactor.keys()),
                                                                    default='TransformerInteractor')
            interactors_name.append(tmp_interactor_type)
        outputs = [self.name2interactor[interactor_name]().build(hp, input_node)
                                for interactor_name in interactors_name]

        # DO WE REALLY NEED TO CAT THEM?
        outputs = [tf.keras.layers.Flatten()(node) if len(node.shape) > 2 else node for node in outputs]
        outputs = tf.concat(outputs, axis=1)
        return outputs






class TransformerBlock3sas(Block):
    def __init__(self, embed_dim=None, num_heads=None, ff_dim=None, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate


    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)

        inputs = input_node[0]
        #embedding_dim = self.embed_dim or hp.Choice('embedding_dim', [32, 64, 128], default=32)
        # number_heads = self.num_heads or hp.Choice('num_heads', [1, 2, 3], default=1)
        ff_dims = self.ff_dim or hp.Choice('ff_dims', [16, 32], default=16)

        embedding_dim = 32

        training = True
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=embedding_dim)(inputs,inputs)
        attn_output = tf.keras.layers.Dropout(self.rate, training=training)(attn_output)
        out1 = tf.keras.layers.LayerNormalization()(inputs + attn_output)
        ffn_output = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dims, activation="relu"), tf.keras.layers.Dense(embedding_dim),]
        )(out1)
        ffn_output = tf.keras.layers.Dropout(self.rate, training=training)(ffn_output)
        output_node = tf.keras.layers.LayerNormalization()(out1 + ffn_output)

        return output_node












class DistanceInteractor(Block):
    def __init__(self, embed_dim=None, num_heads=None, ff_dim=None, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate


    def build(self, hp, inputs=None):
        input_node = [tf.keras.layers.Flatten()(node) if len(node.shape) > 2 else node for node in nest.flatten(inputs)]
        output_node = tf.reduce_sum(tf.square(tf.subtract(input_node)), axis=-1)
        return output_node


class FeedForward(Block):
    def __init__(self, embed_dim=None, num_heads=None, ff_dim=None, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate


    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)

        inputs = input_node[0]
        #embedding_dim = self.embed_dim or hp.Choice('embedding_dim', [32, 64, 128], default=32)
        # number_heads = self.num_heads or hp.Choice('num_heads', [1, 2, 3], default=1)
        ff_dims = self.ff_dim or hp.Choice('ff_dims', [16, 32], default=16)

        embedding_dim = 4


        # feed forward
        out2 = tf.keras.layers.Dense(ff_dims, activation='relu')(inputs)
        output_node = tf.keras.layers.Dense(embedding_dim)(out2)
        return output_node












#
#
#
#


class SelfAttentionInteraction(Block):
    """CTR module for the multi-head self-attention layer in the autoint paper.

    Reference: https://arxiv.org/pdf/1810.11921.pdf

    This block applies multi-head self-attention on a 3D tensor of size
    (batch_size, field_size, embedding_size).

    We assume the input could be a list of tensors of 1D, 2D or 3D, and the block
    will align the dimension of tensors to 3D if they're 1D or 2D originally, and
    it will also align the last embedding dimension based on a tunable hyperaparmeter.

    # Attributes:
        embedding_dim (int). Embedding dimension for aligning embedding dimension of
                            the input tensors.
        att_embedding_dim (int). Output embedding dimension after the mulit-head self-attention.
        head_num (int). Number of attention heads.
        residual (boolean). Whether to apply residual connection after self-attention or not.
    """

    def __init__(self,
                 embedding_dim=None,
                 att_embedding_dim=None,
                 head_num=None,
                 residual=None,
                 **kwargs):
        super(SelfAttentionInteraction, self).__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.att_embedding_dim = att_embedding_dim
        # self.head_num = head_num
        # self.residual = residual

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                'embedding_dim': self.embedding_dim,
                'att_embedding_dim': self.att_embedding_dim,
                # 'head_num': self.head_num,
                # 'residual': self.residual,
            })
        return state

    def set_state(self, state):
        super().set_state(state)
        self.embedding_dim = state['embedding_dim']
        self.att_embedding_dim = state['att_embedding_dim']
        # self.head_num = state['head_num']
        # self.residual = state['residual']

    def _scaled_dot_product_attention(self, q, k, v):
        """Calculate the attention weights.

        Reference: https://www.tensorflow.org/tutorials/text/transformer

        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        # Arguments:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth_v)

        # Returns:
          single-head attention result
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # if mask is not None:
        #     scaled_attention_logits += (mask * -1e9)
        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)

        # expland all the tensors to 3D tensor
        for idx, node in enumerate(input_node):
            if len(node.shape) == 1:
                input_node[idx] = tf.expand_dims(tf.expand_dims(node, -1), -1)
            elif len(node.shape) == 2:
                input_node[idx] = tf.expand_dims(node, 1)
            elif len(node.shape) > 3:
                raise ValueError(
                    "Unexpected inputs dimensions %d, expect to be smaller than 3" % len(node.shape)
                )

        # align the embedding_dim of input nodes if they're not the same
        embedding_dim = self.embedding_dim or hp.Choice('embedding_dim', [ 128, 256, 512], default=256)
        output_node = [tf.keras.layers.Dense(embedding_dim)(node)
                       if node.shape[2] != embedding_dim else node for node in input_node]
        output_node = tf.concat(output_node, axis=1)

        att_embedding_dim = self.att_embedding_dim or hp.Choice('att_embedding_dim', [128, 256, 512], default=256)
        # head_num = self.head_num or hp.Choice('head_num', [1, 2, 3, 4], default=2)
        # residual = self.residual or hp.Choice('residual', [True, False], default=True)

        outputs = []
        for _ in range(1):
            query = tf.keras.layers.Dense(att_embedding_dim, use_bias=False)(output_node)
            key = tf.keras.layers.Dense(att_embedding_dim, use_bias=False)(output_node)
            value = tf.keras.layers.Dense(att_embedding_dim, use_bias=False)(output_node)

            outputs.append(
                self._scaled_dot_product_attention(query, key, value)
            )

        outputs = tf.concat(outputs, axis=2)

        # if self.residual:
        #     print('hkjk')
        #     outputs += tf.keras.layers.Dense(att_embedding_dim * head_num, use_bias=False)(output_node)

        return outputs

class SASREC(Block):
    """
    """
    def __init__(self,
                 embedding_dim=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.fixed_params = []
        self.tunable_candidates = ['embedding_dim']
        self.embedding_dim = embedding_dim

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                'embedding_dim': self.embedding_dim,
            })
        return state

    def set_state(self, state):
        super().set_state(state)
        self.embedding_dim = state['embedding_dim']

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)

        # expland all the tensors to 3D tensor

        # output = input_node[0]
        logits = tf.expand_dims(input_node[0], 1)
        pos_emb = tf.expand_dims(input_node[1], 1)
        # neg_emb = tf.expand_dims(input_node[1], 1)

        # print('hfhf')

        # pos_emb = tf.keras.layers.Dense(32)(pos_emb)
        # neg_emb = tf.keras.layers.Dense(32)(neg_emb)

        # logits = tf.reduce_mean(logits, axis=1, keepdims=True)

        output_node = -(tf.reduce_sum(logits * pos_emb, axis=1, keepdims=True))
        # neg_logits = tf.reduce_sum(logits * neg_emb, axis=1, keepdims=True)


        # loss = - tf.math.log(tf.sigmoid(pos_logits) + 1e-24) -tf.math.log(1-tf.sigmoid(neg_logits))
        #
        # output_node = tf.reduce_sum(tf.sigmoid(loss), axis=2, keepdims=False)



        # from keras import backend as K

        # output_node = 1-K.sigmoid(tf.keras.layers.subtract(input_node))
        # output_node = tf.keras.layers.subtract(input_node)
        # print(output_node.shape)
        # output_node = tf.keras.layers.Dense(1, activation='sigmoid')(output_node)
        # print('outputtt', output_node)

        return output_node
