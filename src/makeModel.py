from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.activations import swish
from tensorflow.keras.initializers import glorot_uniform, RandomUniform

# TODO: Modify architecture from outside


def model_builder(representations, output_size):
    init1 = glorot_uniform(seed=1)
    init2 = glorot_uniform(seed=1)
    init3 = glorot_uniform(seed=1)
    init4 = glorot_uniform(seed=1)
    init5 = glorot_uniform(seed=1)
    inito = RandomUniform(minval=-0.05, maxval=0.05, seed=1)  # glorot_uniform(seed=1)
    layer1_size = 860
    layer2_size = 460
    layer3_size = 50
    layer4_size = 560
    layer5_size = 140
    input_layer = Input(shape=(len(representations[0]),), name='input')
    hidden_layer1 = Dense(layer1_size, kernel_initializer=init1, bias_initializer=init1, name='layer_1')(input_layer)
    hidden_layer1b = LeakyReLU()(hidden_layer1)
    hidden_layer2 = Dense(layer2_size, kernel_initializer=init1, bias_initializer=init2, name='layer_2')(hidden_layer1b)
    hidden_layer2b = LeakyReLU()(hidden_layer2)
    hidden_layer3 = Dense(layer3_size, kernel_initializer=init1, bias_initializer=init3, name='layer_3')(hidden_layer2b)
    hidden_layer3b = LeakyReLU()(hidden_layer3)
    hidden_layer4 = Dense(layer4_size, kernel_initializer=init1, bias_initializer=init4, name='layer_4')(hidden_layer3b)
    hidden_layer4b = LeakyReLU()(hidden_layer4)
    hidden_layer5 = Dense(layer5_size, kernel_initializer=init1, bias_initializer=init5, name='layer_5')(hidden_layer4b)
    output_layer = Dense(output_size, activation='linear', kernel_initializer=inito,
                         bias_initializer=inito, name='output')(hidden_layer5)
    model = Model(inputs=[input_layer], outputs=output_layer)
    model.compile(loss='mean_squared_error',
                  optimizer="Adam",
                  metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])
    return model


def small_model():
    init1 = glorot_uniform(seed=1)
    init2 = glorot_uniform(seed=2)
    inito = RandomUniform(minval=-0.05, maxval=0.05, seed=1)
    layer1_size = 50
    layer2_size = 50
    input_layer = Input(shape=(2,), name='input')
    hidden_layer1 = Dense(layer1_size, kernel_initializer=init1, bias_initializer=init1, name='layer_1')(input_layer)
    hidden_layer1b = LeakyReLU()(hidden_layer1)
    hidden_layer2 = Dense(layer2_size, kernel_initializer=init1, bias_initializer=init2, name='layer_2')(hidden_layer1b)
    output_layer = Dense(1, activation='linear', kernel_initializer=inito,
                         bias_initializer=inito, name='output')(hidden_layer2)
    model = Model(inputs=[input_layer], outputs=output_layer)
    model.compile(loss='mean_squared_error',
                  optimizer="Adam",
                  metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])
    return model
