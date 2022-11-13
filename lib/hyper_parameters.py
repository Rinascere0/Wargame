from collections import namedtuple

HyperParameter = namedtuple('HyperParameter',
                            ['state_shape', 'action_num', 'max_entity', 'my_entity', 'embed_x', 'embed_y', 'mapx',
                             'mapy',
                             'entity_size','total_map_channel', 'map_channel', 'scalar_size', 'bias_value',
                             'entity_embedding_size', 'spatial_embedding_size', 'scalar_embedding_size',
                             'total_embedding_size', 'embedding_size', 'autoregressive_embedding_size',
                             'hidden_size', 'lstm_layers', 'seq_length', 'batch_size', 'max_map_channel',
                             'original_32','original_64','original_128','original_256','original_1024'])

hyper_parameters = HyperParameter(state_shape=128,  # base info
                                  action_num=14,
                                  max_entity=6,
                                  my_entity=3,
                                  mapx=92,
                                  mapy=77,
                                  embed_x=128,
                                  embed_y=128,
                                  entity_size=240,  # preprocess params
                                  map_channel=13,
                                  total_map_channel=15,
                                  scalar_size=13,
                                  bias_value=-1e9,
                                  entity_embedding_size=64,  # encoder params
                                  spatial_embedding_size=64,
                                  scalar_embedding_size=8,
                                  total_embedding_size=128,
                                  embedding_size=128,
                                  autoregressive_embedding_size=256,
                                  hidden_size=128,  # core params
                                  lstm_layers=1,
                                  seq_length=1,
                                  batch_size=1,
                                  max_map_channel=48,
                                  original_32=16,
                                  original_64=32,
                                  original_128=48,
                                  original_256=64,
                                  original_1024=256)
