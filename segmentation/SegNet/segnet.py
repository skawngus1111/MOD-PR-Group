import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module) :
    def __init__(self, num_classes, n_init_features=3, drop_rate=0.5,
                 filter_config=(64, 128, 256, 512, 512)):
        super(SegNet, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = (2, 2, 3, 3, 3)
        decoder_n_layers = (3, 3, 3, 2, 1)

        encoder_filter_config = (n_init_features, ) + filter_config
        decoder_filter_config = filter_config[::-1] + (filter_config[0], )

        for i in range(0, 5) :
            # encoder architecture
            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i+1],
                                          encoder_n_layers[i], drop_rate))

            # decoder architecture
            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i+1],
                                          decoder_n_layers[i], drop_rate))

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Conv2d(filter_config[0], num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x

        # encoder path, keep track of pooling indices and features size
        for i in range(0, 5) :
            (feat, ind), size = self.encoders[i](feat)
            indices.append(ind)
            unpool_sizes.append(size)

        # decoder path, upsampling with corresponding indices and size
        for i in range(0, 5) :
            feat = self.decoders[i](feat, indices[4 - i], unpool_sizes[4 - i])

        return self.classifier(feat)

class _Encoder(nn.Module) :
    def __init__(self, n_in_feat, n_out_feat,
                 n_block=2, drop_rate=0.5):
        super(_Encoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_out_feat, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.BatchNorm2d(n_out_feat), nn.ReLU(inplace=True)]

        if n_block > 1 :
            layers += [nn.Conv2d(n_out_feat, n_out_feat, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.BatchNorm2d(n_out_feat), nn.ReLU(inplace=True)]

            if n_block == 3 :
                layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)

        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()

class _Decoder(nn.Module) :
    def __init__(self, n_in_feat, n_out_feat,
                 n_block=2, drop_rate=0.5):
        super(_Decoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_out_feat, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.BatchNorm2d(n_out_feat), nn.ReLU(inplace=True)]

        if n_block > 1 :
            layers += [nn.Conv2d(n_out_feat, n_out_feat, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                  nn.BatchNorm2d(n_out_feat), nn.ReLU(inplace=True)]

            if n_block == 3 :
                layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)

        return self.features(unpooled)