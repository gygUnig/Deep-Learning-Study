# Transformer 구현
# reference : https://wikidocs.net/31379

import numpy as np
import tensorflow as tf


# Positional Encoding
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def get_angles(self, position, i, d_model):
        angles = 1/tf.pow(10000, (2*(i//2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position = tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model = d_model
        )
        
        # 배열의 짝수 인엑스(2i)에는 사인 함수 적용
        sines = tf.math.sin(angle_rads[:, 0::2])
        
        #배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
    
# scaled dot-product attention
def scaled_dot_product_attention(query, key, value, mask):
    # query 크기 : (batch_size, num_heads, query문장길이, d_model/num_heads)
    # key 크기   : (batch_size, num_heads, key문장길이, d_model/num_heads)
    # value 크기 : (batch_size, num_heads, value문장길이, d_model/num_heads)
    # padding_mask : (batch_size, 1, 1, key문장길이)
    
    # Q와 K의 곱, 어텐션 스코어 행렬
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    
    # 스케일링 - dk의 루트값으로 나눠준다
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    
    # 마스킹. 어텐션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
    # 매우 작은 값이므로 소프트 맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 된다.
    if mask is not None:
        logits += (mask * -1e9)
        
    # 소프트맥스 함수는 마지막 차원인 Key의 문장 길이 방향으로 수행된다.
    # attention weight : (batch_size, num_heads, query문장길이, key문장길이)
    attention_weights = tf.nn.softmax(logits, axis = -1)
    
    # output : (batch_size, num_heads, query문장 길이, d_model/num_heads)
    output = tf.matmul(attention_weights, value)
    
    return output, attention_weights


# Multi Head Attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        # d_model을 num_heads로 나눈 값. 논문 기준 64
        self.depth = d_model // self.num_heads
        
        # WQ, WK, WV에 해당하는 밀집층 정의
        self.query_dense = tf.keras.layers.Dense(units = d_model)
        self.key_dense = tf.keras.layers.Dense(units = d_model)
        self.value_dense = tf.keras.layers.Dense(units = d_model)
        
        # WO에 해당하는 밀집층 정의
        self.dense = tf.keras.layers.Dense(units = d_model)
        
    # num_heads 개수만큼 q, k, v를 split하는 함수
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth)
        )
        return tf.transpose(inputs, perm = [0,2,1,3])
    
    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        
        # 1. WQ, WK, WV에 해당하는 밀집층 지나기
        # q : (batch_size, query문장길이, d_model)
        # k : (batch_size, key문장길이, d_model)
        # v : (batch_size, value문장길이, d_model)
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        
        # 2. 헤드 나누기
        # q : (batch_size, num_heads, query문장길이, d_model/num_heads)
        # k : (batch_size, num_heads, key문장길이, d_model/num_heads)
        # v : (batch_size, num_heads, value문장길이, d_model/num_heads)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # 3. 스케일드 닷 프로덕트 어텐션
        # (batch_size, num_heads, query문장길이, d_model/num_heads)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        # (batch_size, query문장길이, num_heads, d_model/num_heads)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        # 4. 헤드 연결(concatenate)하기
        # (batch_size, query문장길이, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        # 5. WO에 해당하는 밀집층 지나기
        # (batch_size, query의 문장 길이, d_model)
        outputs = self.dense(concat_attention)
        
        return outputs
        
        
# 패딩 마스크
def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, key의 문장 길이)
    return mask[:, tf.newaxis, tf.newaxis, :]


# Encoder
def encoder_layer(dff, d_model, num_heads, dropout, name = "encoder_layer"):
    inputs = tf.keras.Input(shape = (None, d_model), name="inputs")
    
    # 패딩 마스크 사용
    padding_mask = tf.keras.Input(shape = (1, 1, None), name = "padding_mask")
    
    # 멀티 헤드 어텐션
    attention = MultiHeadAttention(d_model, num_heads, name="attention")({
        'query' : inputs, 'key' : inputs, 'value' : inputs,  # Q=K=V
        'mask' : padding_mask
    })
    
    # 드롭아웃 + 잔차 연결과 층 정규화
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
    
    # 포지션 와이즈 피드 포워드 신경망
    outputs = tf.keras.layers.Dense(units = dff, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units = d_model)(outputs)
    
    # 드롭아웃 + 잔차연결과 층 정규화
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)
    
    return tf.keras.Model(
        inputs = [inputs, padding_mask], outputs = outputs, name = name
    )


# Encoder 쌓기
def encoder(max_length, num_layers, dff, d_model, num_heads, dropout, name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    
    # 패딩 마스크 사용
    padding_mask = tf.keras.Input(shape=(1,1,None), name = "padding_mask")
    
    # 포지셔널 인코딩 + 드롭아웃
    embeddings = tf.keras.layers.Embedding(max_length, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(max_length, d_model)(embeddings)
    
    # 인코더를 num_layers개 쌓기
    for i in range(num_layers):
        outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
                                dropout=dropout, name="encoder_layer_{}".format(i))([outputs, padding_mask])
        
    return tf.keras.Model(
        inputs = [inputs, padding_mask], outputs = outputs, name=name
    )
  
    
# look ahead mask
def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


# Decoder
def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    
    # look ahead mask
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name = "look_ahead_mask")
    
    # padding mask
    padding_mask = tf.keras.Input(shape=(1,1,None), name="padding_mask")
    
    # 멀티 헤드 어텐션
    attention1 = MultiHeadAttention(d_model, num_heads, name="attention1")(
        inputs = {
            'query': inputs, 'key': inputs, 'value': inputs,
            'mask': look_ahead_mask  
        }
    )
    
    # 잔차 연결과 층 정규화
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)
    
    # 멀티 헤드 어텐션
    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention2")(
            inputs = {
                'query': attention1, 'key': enc_outputs, 'value': enc_outputs,
                'mask' : padding_mask
            }
    )
    
    # 드롭아웃 + 잔차 연결과 층 정규화
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)
    
    # 포지션 와이즈 피드 포워드 신경망
    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    
    # 드롭아웃 + 잔차 연결과 층 정규화
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)
    
    return tf.keras.Model(
        inputs = [inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs = outputs,
        name = name
    )
    

# Decoder 쌓기
def decoder(max_length, num_layers, dff, d_model, num_heads, dropout, name="decoder"):
    inputs = tf.keras.Input(shape=(None, d_model), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name = 'encoder_outputs')
    
    # look ahead mask, padding mask
    look_ahead_mask = tf.keras.Input(
        shape = (1, None, None), name = 'look_ahead_mask')
    padding_mask = tf.keras.Input(shape = (1,1,None), name = 'padding_mask')
    
    # 포지셔널 인코딩 + 드롭아웃
    embeddings = tf.keras.layers.Enbedding(max_length, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(max_length, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
    
    # 디코더를 num_layers개 쌓기
    for i in range(num_layers):
        outputs = decoder_layer(dff = dff, d_model=d_model, num_heads=num_heads,
                                dropout=dropout, name='decoder_layer_{}'.format(i))(
                                    inputs = [outputs, enc_outputs, look_ahead_mask, padding_mask])
                                
    return tf.keras.Model(
        inputs = [inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs = outputs,
        name = name
    )
    
    
# Transformer 구현
def transformer(vocab_size, max_length, num_layers, dff, d_model, num_heads, dropout, name="transformer"):
    
    # 인코더의 입력
    inputs = tf.keras.Input(shape=(None,), name = "inputs")
    
    # 디코더의 입력
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")
    
    # 인코더의 패딩 마스크
    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape = (1,1,None),
        name = 'enc_padding_mask')(inputs)
    
    # 디코더의 룩어헤드 마스크
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape = (1, None, None),
        name = 'look_ahead_mask')(dec_inputs)
    
    # 디코더의 패딩 마스크
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape = (1,1,None),
        name = 'dec_padding_mask')(inputs)
    
    # 인코더의 출력은 enc_outputs. 디코더로 전달된다.
    enc_outputs = encoder(max_length=max_length, num_layers=num_layers, dff=dff,
                          d_model=d_model, num_heads=num_heads, dropout=dropout)(
                              inputs = [inputs, enc_padding_mask])
    
    # 디코더의 출력은 dec_outputs. 출력층으로 전달된다.
    dec_outputs = decoder(max_length=max_length, num_layers=num_layers, dff=dff,
                          d_model=d_model, num_heads=num_heads, dropout=dropout)(
                              inputs = [dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
    
    # 다음 단어 예측을 위한 출력층
    outputs = tf.keras.layers.Dense(units = vocab_size, name = 'outputs')(dec_outputs)
    
    return tf.keras.Model(inputs = [inputs, dec_inputs], outputs = outputs, name=name)


