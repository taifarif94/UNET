from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Concatenate, Activation, Conv2DTranspose, \
    BatchNormalization, Dropout

class CreateUNET:

    def __init__(self, input_shape, createBigUNET=False):
        self.createBigNet = createBigUNET       # If true, we create the bigger UNET
        self.input_shape = input_shape

    def convolution_block(self, input_layer, num_filters, use_dropout=False):
        x = Conv2D(num_filters, 3, padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        if use_dropout:
            x = Dropout(0.5)(x)

        return x

    def encoder_block(self, input_layer, num_filters, use_dropout=False):
        x = self.convolution_block(input_layer, num_filters, use_dropout)
        pool_layer = MaxPool2D((2, 2))(x)
        return x, pool_layer

    def decoder_block(self, input_layer, skip_features, num_filters, use_dropout=False):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_layer)
        x = Concatenate()([x, skip_features])
        x = self.convolution_block(x, num_filters, use_dropout)
        return x

    def get_model(self):
        inputs = Input(shape=self.input_shape)
        if self.createBigNet:
            e1, pool1 = self.encoder_block(inputs, 64)
            e2, pool2 = self.encoder_block(pool1, 128)
            e3, pool3 = self.encoder_block(pool2, 256)
            e4, pool4 = self.encoder_block(pool3, 512)
            e5, pool5 = self.encoder_block(pool4, 1024, use_dropout=True)

            bridge_layer = self.convolution_block(pool5, 2048, use_dropout=True)

            d1 = self.decoder_block(bridge_layer, e5, 1024, use_dropout=True)
            d2 = self.decoder_block(d1, e4, 512)
            d3 = self.decoder_block(d2, e3, 256)
            d4 = self.decoder_block(d3, e2, 128)
            d_final = self.decoder_block(d4, e1, 64)
        else:
            e1, pool1 = self.encoder_block(inputs, 64)
            e2, pool2 = self.encoder_block(pool1, 128)
            e3, pool3 = self.encoder_block(pool2, 256)
            e4, pool4 = self.encoder_block(pool3, 512, use_dropout=True)

            bridge_layer = self.convolution_block(pool4, 1024, use_dropout=True)

            d1 = self.decoder_block(bridge_layer, e4, 512)
            d2 = self.decoder_block(d1, e3, 256)
            d3 = self.decoder_block(d2, e2, 128)
            d_final = self.decoder_block(d3, e1, 64)

        outputs = Conv2D(3, 1, padding="same", activation="relu")(d_final)
        model = Model(inputs, outputs, name="Enhanced_UNET")

        model.summary()     # get the summary
        return model
