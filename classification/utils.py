import sys


def get_model(model_name, input_channel, num_classes) :
    if model_name == 'mobilenet-v1' :
        from models.mobilenet import MobileNet
        model = MobileNet(model_name, input_channel, num_classes)
    elif model_name == 'googlenet' :
        from models.googlenet import GoogleNet
        model = GoogleNet(input_channel, num_classes)
    elif model_name == 'inception-v3' :
        from models.inceptionv3 import InceptionV3
        model = InceptionV3(input_channel, num_classes)
    elif model_name == 'nin' :
        from models.nin import NIN
        model = NIN(input_channel, num_classes)
    else :
        sys.exit(1)

    return model