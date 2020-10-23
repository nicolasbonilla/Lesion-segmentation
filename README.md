# MRI - Deep learning segmentation



![Brain](U-Net-Architecture.png) 
![Hist](My movie.mp4)<br /><br />


```bash
def get_unet(img_shape = None):

        dim_ordering = 'tf'

        inputs = Input(shape = img_shape)
        concat_axis = -1

        conv1 = Conv2D(64, (5, 5), padding="same", activation="relu", name="conv1_1", data_format="channels_last")(inputs)
        conv1 = Conv2D(64, (5, 5), padding="same", activation="relu", data_format="channels_last")(conv1)
        pool1 = MaxPooling2D(data_format="channels_last", pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(96, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool1)
        conv2 = Conv2D(96, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv2)
        pool2 = MaxPooling2D(data_format="channels_last", pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool2)
        conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv3)
        pool3 = MaxPooling2D(data_format="channels_last", pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool3)
        conv4 = Conv2D(256, (4, 4), padding="same", activation="relu", data_format="channels_last")(conv4)
        pool4 = MaxPooling2D(data_format="channels_last", pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool4)
        conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv5)

        up_conv5 = UpSampling2D(data_format="channels_last", size=(2, 2))(conv5)
        ch, cw = get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv4)
        
        up6 = concatenate([up_conv5, crop_conv4], axis = concat_axis)
        
        conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(up6)
        conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv6)

        up_conv6 = UpSampling2D(data_format="channels_last", size=(2, 2))(conv6)
        ch, cw = get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv3)

        up7 = concatenate([up_conv6, crop_conv3], axis = concat_axis)

        conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(up7)
        conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv7)

        up_conv7 = UpSampling2D(data_format="channels_last", size=(2, 2))(conv7)
        ch, cw = get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv2)
        
        up8 = concatenate([up_conv7, crop_conv2], axis = concat_axis)
        
        conv8 = Conv2D(96, (3, 3), padding="same", activation="relu", data_format="channels_last")(up8)
        conv8 = Conv2D(96, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv8)

        up_conv8 = UpSampling2D(data_format="channels_last", size=(2, 2))(conv8)
        ch, cw = get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv1)
        
        up9 = concatenate([up_conv8, crop_conv1], axis = concat_axis)
        
        conv9 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(up9)
        conv9 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv9)

        ch, cw = get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=(ch, cw), data_format="channels_last")(conv9)
        conv10 = Conv2D(1, (1, 1), activation="sigmoid", data_format="channels_last")(conv9)
        model = Model(inputs=inputs, outputs=conv10)
        
        model.compile(optimizer=Adam(lr=(1e-4)*2), loss=dice_coef_loss, metrics=[dice_coef_for_training])

        return model
```
Hola
