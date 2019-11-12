import torch.nn as nn


numeric_input_layer_params = {'batch_norm': False,
                              'dropout_rate': None}

embed_layer_params = {'batch_norm': False,
                      'activation': None,
                      'dropout_rate': None,
                      'weight_init': nn.init.xavier_uniform_}

attention_layer_params = {'batch_norm': False,
                          'activation': None,
                          'dropout_rate': None,
                          'weight_init': nn.init.xavier_uniform_}

dense_layer_params = {'batch_norm': False,
                      'activation': None,
                      'dropout_rate': None,
                      'weight_init': nn.init.xavier_uniform_}

output_layer_params = {'activation': None,
                       'weight_init': nn.init.xavier_uniform_}

recurrent_layer_params = {'dropout': 0.0}
