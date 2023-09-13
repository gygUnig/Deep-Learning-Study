# Transformer 구현 - Tensorflow
# reference : https://wikidocs.net/31379
# 목표 : 단순 필사가 아닌, docstring과 주석 꼼꼼하게 달기!


import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional Encoding 레이어
    단어의 위치 정보를 얻기 위해서 각 단어의 임베딩 벡터에 위치 정보들을 더하여 모델의 입력으로 사용한다.
    
    Attributes: 
    - pos_encoding (tf.Tensor) : Positional encoding tensor로, positional_encoding 메소드를 통해
      계산된 포지셔널 인코딩 값을 가지게 되며, call 메소드에서 inputs에 더해진다.
    """
    
    def __init__(self, position, d_model):
        """
        PositionalEncoding 클래스의 초기화 메소드
        
        Args:
            - position (int) : 최대 시퀀스 길이 
            - d_model (int) : 임베딩 벡터의 차원, 트랜스포머의 모든 층의 입-출력 차원
        """
        
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def get_angles(self, position, i, d_model):
        """
        Positional Encoding의 각도를 계산하는 메소드이다.
        1/(10000^(wi/d_model))에 해당하는 값이다.

        Args:
            - position (tf.Tensor): positional_encoding 메서드에서 int인 positional을 받아서 tf.range를 통해서
              0 ~ position-1 까지의 텐서를 생성한다. 이 텐서가 get_angles 메서드에 position으로 전달된다.
                
            - i (tf.Tensor): positional_encoding 메서드에서 int인 d_model을 받아서 tf.range를 통해서
              0 ~ d_model 까지의 텐서를 생성한다. 이 텐서가 get_angles 메서드에 i로 전달된다.
                
            - d_model (int): 임베딩 벡터의 차원

        Returns:
            - tf.Tensor : position과 angles를 곱해서 만들어진 텐서. Positional Encoding을 할 때 sin과 cos함수의
                입력값으로 사용된다. shape : (position, d_model)
        """
        
        angles = 1/tf.pow(10000, (2*(i//2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        """
        Positional encoding을 계산하는 메서드

        Args:
            - position (int): 최대 시퀀스 길이
            - d_model (int): 임베딩 벡터의 차원

        Returns:
            - tf.Tensor : Positional Encoding 값을 포함하는 3D 텐서
              입력 시퀀스의 각 위치에 대해서 특정 값들을 더해줘서 단어의 위치 정보를 얻을 수 있다.
            
        """
        
        # 위에서 정의한 get_angles 메서드의 인자로 (position,1)의 shape를 가진 텐서인 position,
        # (1, d_model)의 shape를 가진 텐서인 i, d_model을 넣어준다.
        # 이에 대한 결과인 angle_rads는 (position, d_model)의 shape를 가진 2D 텐서가 된다.
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
        pos_encoding = pos_encoding[tf.newaxis, ...] # pos_encoding의 shape는 (1, position, d_model)
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        """
        PositionalEncoding layer의 호출 메서드

        Args:
            - inputs (tf.Tensor): 임베딩 된 시퀀스

        Returns:
            - tf.Tensor : 임베딩 된 시퀀스에 Positioanl Encoding이 더해진 값
        """
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
    
    
def scaled_dot_product_attention(query, key, value, mask):
    """
    스케일드 닷 프로덕트 어텐션 함수
    
    query, key, value를 사용하여 Attention Value Matrix를 구한다.

    Args:
        - query (tf.Tensor): 쿼리 텐서
          shape : (batch_size, num_heads, query 문장 길이, d_model/num_heads)
        - key (tf.Tensor): 키 텐서
          shape : (batch_size, num_heads, key 문장 길이, d_model/num_heads)
        - value (tf.Tensor): 벨류 텐서
          shape : (batch_size, num_heads, value 문장 길이, d_model/num_heads)
        - mask (tf.Tensor): 어텐션 스코어 행렬에 적용될 마스크 텐서(softmax 적용 전)
          shape : (batch_size, 1, 1, key 문장 길이)

    Returns:
        - output(tf.Tensor) : Attention Value Tensor (softmax를 거친 후 value과 matmul한 결과값)
          shape : (batch_size, num_heads, qeury문장 길이, d_model/num_heads)
        - attention_weights(tf.Tensor) : Attention weight Tensor(softmax를 거친 후 value와 matmul하기 전)
          attention_weights와 value를 matmul하면 output이 된다.
          shape : (batch_size, num_heads, qeury 문장 길이, key 문장 길이)
    """

    # Q와 K.T의 곱
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    
    # tf.shape(key)[-1]은 d_model/num_heads이고, 이 값은 d_k이다.
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    
    # matmul_qk를 d_k의 루트값으로 나눠준 값이 attention score이 된다.
    logits = matmul_qk / tf.math.sqrt(depth)
    
    
    # 마스킹. 어텐션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
    # 매우 작은 값이므로 소프트 맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 된다.
    if mask is not None:
        logits += (mask * -1e9)  # 이때 mask의 크기는 (batch_size, 1, 1, key 문장 길이)인데, 
        # logits의 크기는 (batch_size, num_heads, query 문장 길이, key문장 길이)이다.
        # 따라서 브로드캐스팅이 되어서 key에 <PAD>가 있는 경우에는 해당 열 전체를 마스킹을 해주게 된다.
        
    # 소프트맥스 함수는 마지막 차원인 Key의 문장 길이 방향으로 수행된다.
    attention_weights = tf.nn.softmax(logits, axis = -1)
    
    # attention weights와 value를 matmul한다.
    output = tf.matmul(attention_weights, value)
    
    return output, attention_weights



class MultiHeadAttention(tf.keras.layers.Layer):
    """
    멀티 헤드 어텐션 클래스
    여러 헤드에서 각각 어텐션을 계산한 후, 결과를 concatenate하여 반환한다.

    Attributes:
    - num_heads (int) : 어텐션 헤드의 갯수
    - d_model (int) : 임베딩 벡터의 차원, 트랜스포머의 모든 층의 입-출력 차원
    - depth (int) : d_model을 num_heads로 나눈 값, 각 헤드에서의 차원
    - query_dense, key_dense, value_dense (tf.keras.layers.Dense) : WQ, WK, WV에 해당하는 밀집층
    - dense (tf.keras.layers.Dense) : WO에 해당하는 밀집층
    """
    
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        """
        멀티 헤드 어텐션 객체 초기화 메서드

        Args:
            - d_model (int) : 임베딩 벡터의 차원, 트랜스포머 모든 층의 입-출력 차원
            - num_heads (int) : 어텐션 헤드의 갯수
            - name (str): 레이어 이름, 기본값은 "multi_head_attention"
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        
        # d_model이 num_heads로 나누어 떨어지지 않는다면 assert 에러
        assert d_model % self.num_heads == 0
        
        # depth는 d_model을 num_heads로 나눈 값. 논문 기준 64
        self.depth = d_model // self.num_heads
        
        # WQ, WK, WV에 해당하는 밀집층 정의
        self.query_dense = tf.keras.layers.Dense(units = d_model)
        self.key_dense = tf.keras.layers.Dense(units = d_model)
        self.value_dense = tf.keras.layers.Dense(units = d_model)
        
        # WO에 해당하는 밀집층 정의
        self.dense = tf.keras.layers.Dense(units = d_model)
        
    # num_heads 개수만큼 q, k, v를 split하는 함수
    def split_heads(self, inputs, batch_size):
        """
        inputs을 num_heads만큼 split하는 메서드
        여기서 inputs이란? query, key, value가 될 것이다.

        Args:
            - inputs (tf.Tensor): split할 입력 텐서. 여기서는 q, k, v
            - batch_size (int): 배치 사이즈

        Returns:
            - tf.Tensor: 헤드 수만큼 나뉘어진 텐서
        """
        
        # inputs을 reshape해서 (batch_size, 문장 길이, num_heads, d_model/num_heads)의 shape를 갖도록 나눈다
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth)
        )
        return tf.transpose(inputs, perm = [0,2,1,3]) # (batch_size, num_heads, 문장길이, d_model/num_heads)
    
    
    def call(self, inputs):
        """
        멀티 헤드 어텐션의 연산을 수행하는 메서드

        Args:
            - inputs (dict): query, key, value, mask 정보가 있는 딕셔너리

        Returns:
            - tf.Tensor: 멀티 헤드 어텐션의 결과
        """
        
        # inputs은 딕셔너리 형태. query, key, value, mask 정보가 있다.
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
        
        

def create_padding_mask(x):
    """
    패딩 마스킹을 생성하는 함수
    입력된 시퀀스에서 0의 값을 가진 위치를 찾아서 해당 위치를 마스킹한다.
    패딩된 위치에 대해서 어텐션 연산 무시

    Args:
        - x (tf.Tensor): 패딩 마스크를 적용할 시퀀스. shape는 (batch_size, sequence_length)

    Returns:
        - tf.Tensor: 패딩 마스크가 적용된 텐서. shape는 (batch_size, 1, 1, sequence_length)
          0인 위치에는 1의 값을, 그 외에는 0의 값을 가진다.
    """
    
    # tf.math.equal(x, 0)은 0과 같으면 True, 아니면 False를 반환하는 것이다. 
    # 이 때 캐스팅을 tf.float32로 하므로 True는 1이 되고 False는 0이 된다.
    # 따라서 mask는 0은 1로 바꾸고, 0이 아니면 0으로 바꾼다.
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
   
    return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, key의 문장 길이)



def encoder_layer(dff, d_model, num_heads, dropout, name = "encoder_layer"):
    """
    인코더 레이어 함수
    
    두 개의 서브 레이어로 나뉘어져 있다.
    1. 멀티 헤드 어텐션
    2. 포지션 와이즈 피드 포워드 신경망
    
    각 서브 레이어는 잔차 연결과 층 정규화가 진행된다.

    Args:
        - dff (int) : 포지션 와이즈 피드 포워드 신경망의 은닉층 차원
        - d_model (int) : 임베딩 벡터의 차원, 트랜스포머 모든 층의 입-출력 차원
        - num_heads (int) : 멀티 헤드 어텐션 헤드 수
        - dropout (float) : Dropout 비율
        - name (str) : 인코더 레이어의 이름. 기본값은 "encoder_layer"
        
    Returns:
        - tf.keras.Model: 인코더 레이어 모델
    """
    
    # 입력층 정의
    # shape = (None, d_model) 에서 None은 정의가 안 된 상태이므로 어떤 길이의 문장이 들어와도 처리할 수 있다.
    inputs = tf.keras.Input(shape = (None, d_model), name="inputs")
    
    # 패딩 마스크 사용
    # shape=(1, 1, None)에서 마지막 None은 패딩 마스크의 길이(key 문장 길이)에 해당한다.
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



def encoder(max_length, vocab_size, num_layers, dff, d_model, num_heads, dropout, name="encoder"):
    """
    트랜스포머 인코더를 쌓는 함수

    Args:
        - max_length (int): 입력 시퀀스의 최대 길이. 포지셔널 인코딩 벡터를 생성할 때 사용된다.
        - vocab_size (int) : 어휘 크기. 임베딩 레이어에서 사용되는 최대 단어 수 (10000 같은 것)
        - num_layers (int): 인코더 레이어를 쌓을 개수
        - dff (int): 피드 포워드 신경망의 은닉층 차원
        - d_model (int): 임베딩 벡터의 차원, 트랜스포머 모든 층의 입-출력 차원
        - num_heads (int): 멀티 헤드 어텐션에서 헤드의 개수
        - dropout (float): 드롭아웃 비율
        - name (str): 모델 이름. 기본값은 "encoder".

    Returns:
        - tf.keras.Model: 인코더 구조를 가진 Keras 모델
        
    Notes:
    1. 입력은 임베딩 레이어를 거쳐서 임베딩 벡터로 변환된다.
    2. 포지셔널 인코딩 레이어를 통해 임베딩 벡터에 위치 정보를 추가한다.
    3. num_layers만큼의 인코더 레이어를 거쳐서 최종 출력을 얻는다. 이 떄, 각 레이어는 멀티 헤드 어텐션과
       피드 포워드 신경망을 포함하며, 잔차 연결과 층 정규화를 진행한다.
    """
    
    # Input 텐서 정의. shape = (None,)라는 것은 입력 시퀀스의 길이가 다양하다는 것을 의미한다.
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    
    # 패딩 마스크를 위한 Input 텐서 정의
    padding_mask = tf.keras.Input(shape=(1,1,None), name = "padding_mask")
    
    # 입력 텐서를 임베딩 레이어를 통해서 d_model 차원의 벡터로 변환한다.
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(max_length, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
    
    # 인코더를 num_layers개 쌓기
    # 이전 레이어의 출력이 현재 레이어의 입력으로 사용된다. 맨 처음에는 embeddings가 사용된다.
    for i in range(num_layers):
        outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
                                dropout=dropout, name="encoder_layer_{}".format(i))([outputs, padding_mask])
        
    return tf.keras.Model(
        inputs = [inputs, padding_mask], outputs = outputs, name=name
    )
  
    
# look ahead mask
def create_look_ahead_mask(x):
    """
    Look-ahead 마스크와 패딩 마스크를 생성하는 함수

    Args:
        - x (tf.Tensor): 입력 시퀀스. shape는 (batch_size, sequence_length)

    Returns:
        - tf.Tensor : Look-ahead와 패딩 마스크를 합친 마스크. shape는 (batch_size, 1, 1, sequence_length)
        
    Notes:
        - tf.linalg.band_part(input, num_lower, num_upper, name=None)
        - input 행렬에 대해서 대각선 기준(대각선이 0) num_lower만큼 대각선 아래 쪽 값을 살린다.
        - num_upper만큼 대각선 위에 값을 살린다
        - 예시)
        if 'input' is [[ 0,  1,  2, 3]
                       [-1,  0,  1, 2]
                       [-2, -1,  0, 1]
                       [-3, -2, -1, 0]],

        tf.linalg.band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
                                               [-1,  0,  1, 2]
                                               [ 0, -1,  0, 1]
                                               [ 0,  0, -1, 0]],

        tf.linalg.band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
                                              [-1,  0,  1, 0]
                                              [-2, -1,  0, 1]
                                              [ 0, -2, -1, 0]]   
    """
    seq_len = tf.shape(x)[1]  # x의 2번째는 sequence_length
    
    # look ahead mask와 padding mask
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    
    return tf.maximum(look_ahead_mask, padding_mask)



def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
    """
    디코더의 한 레이어

    Args:
        - dff (int): 피드 포워드 신경망의 은닉층 차원
        - d_model (int): 임베딩 벡터의 차원, 트랜스포머 모든 층의 입-출력 차원
        - num_heads (int): 멀티 헤드 어텐션에서 헤드의 갯수
        - dropout (float): 드롭아웃 비율
        - name (str): 모델 이름. 기본값은 "decoder_layer"

    Returns:
        tf.keras.Model: 디코더 레이어의 구조를 가진 Keras 모델
    
    Notes:
    1. 두 번의 멀티 헤드 어텐션을 거친다
        - 첫 번째는 look-ahead mask를 사용해서 현재 위치 이후의 정보에 접근하지 않게 한다.
        - 두 번째는 인코더의 출력(Value, Key)과 디코더의 Query를 이용한 어텐션이다. 여기서는 padding mask만 사용한다
    2. 멀티 헤드 어텐션 후에는 각각 잔차 연결과 층 정규화가 진행된다.
    3. 포지션 와이즈 피드 포워드 신경망을 거치고, 잔차연결과 층 정규화를 진행한다.
    
    """
    
    # 디코더 레이어의 인풋. shape = (None, d_model)인데 여기서 None은 시퀀스 길이를 의미한다.
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    
    # 인코더에서 나온 출력값을 받는다.
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    
    # look ahead mask - 디코더가 미래의 정보에 접근하지 못하게 하는데 사용된다.
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name = "look_ahead_mask")
    
    # padding mask
    padding_mask = tf.keras.Input(shape=(1,1,None), name="padding_mask")
    
    # 디코더의 첫 번째 멀티 헤드 어텐션 - self attention이다, look ahead mask를 적용한다.
    attention1 = MultiHeadAttention(d_model, num_heads, name="attention1")(
        inputs = {
            'query': inputs, 'key': inputs, 'value': inputs,
            'mask': look_ahead_mask  
        }
    )
    
    # 잔차 연결과 층 정규화
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)
    
    # 디코더의 두 번째 멀티 헤드 어텐션 - self attention이 아니다
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
    


def decoder(max_length, vocab_size, num_layers, dff, d_model, num_heads, dropout, name="decoder"):
    """
    디코더를 num_layers만큼 쌓는 함수

    Args:
        - max_length (int): 입력 시퀀스의 최대 길이
        - vocab_size (int) : 어휘 크기. 임베딩 레이어에서 사용되는 최대 단어 수 (10000 같은 것)
        - num_layers (int): 디코더 레이어를 몇 개 쌓을 것인지에 대한 수
        - dff (int): 포지션 와이즈 피드 포워드 레이어의 은닉층 크기
        - d_model (int): 임베딩 벡터의 차원, 트랜스포머 모든 층의 입-출력 차원
        - num_heads (int): 멀티 헤드 어텐션에서의 헤드 수
        - dropout (float): 드롭아웃 비율
        - name (str): 모델의 이름. 기본값은 "decoder".

    Returns:
        tf.keras.Model: 디코더 모델
    """
    
    # (None,) shape의 입력
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name = 'encoder_outputs')
    
    # look ahead mask, padding mask
    look_ahead_mask = tf.keras.Input(
        shape = (1, None, None), name = 'look_ahead_mask')
    padding_mask = tf.keras.Input(shape = (1,1,None), name = 'padding_mask')
    
    # 포지셔널 인코딩 + 드롭아웃
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
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
    
    
    
def transformer(vocab_size, max_length, num_layers, dff, d_model, num_heads, dropout, name="transformer"):
    """
    트랜스포머 모델 생성 함수

    Args:
        vocab_size (int): 어휘 크기. 출력층에 사용된다
        max_length (int): 시퀀스의 최대 길이
        num_layers (int): 인코더, 디코더 레이어의 수
        dff (int): 피드 포워드 레이어의 은닉층 크기
        d_model (int): 임베딩 벡터의 차원
        num_heads (int): 멀티 헤드 어텐션에서의 헤드 수
        dropout (float): 드롭아웃 비율
        name (str): 모델 이름. 기본값은 "transformer".

    Returns:
        tf.keras.Model : 트랜스포머 구조의 Keras 모델
    """
    
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
    enc_outputs = encoder(max_length=max_length, vocab_size=vocab_size, num_layers=num_layers, dff=dff,
                          d_model=d_model, num_heads=num_heads, dropout=dropout)(
                              inputs = [inputs, enc_padding_mask])
    
    # 디코더의 출력은 dec_outputs. 출력층으로 전달된다.
    dec_outputs = decoder(max_length=max_length, vocab_size=vocab_size, num_layers=num_layers, dff=dff,
                          d_model=d_model, num_heads=num_heads, dropout=dropout)(
                              inputs = [dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
    
    # 다음 단어 예측을 위한 출력층
    outputs = tf.keras.layers.Dense(units = vocab_size, name = 'outputs')(dec_outputs)
    
    return tf.keras.Model(inputs = [inputs, dec_inputs], outputs = outputs, name=name)


small_transformer = transformer(
    vocab_size = 9000,
    max_length=50,
    num_layers = 4,
    dff = 512,
    d_model = 128,
    num_heads = 4,
    dropout = 0.3,
    name="small_transformer")



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)