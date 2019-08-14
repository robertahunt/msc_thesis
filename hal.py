
from imports import *
config = dict(modelName='AEMnist3Conv',
              modelParams=dict(imgSize=(28, 28), 
                           out_channels=3, 
                           cuda=True ,
                           n_conv_layers=2,
                           n_filters=16,
                           filter_size=3,
                           stride=3
                           ),
             max_num_epochs = 500,
             suffix=f'_2NC_16NF_3FS',
             opt='adam',
             optParams = {'lr': 0.001},
             loss='mse',
             batchSize=20,
             earlyStopping=25,
             imgSize=(28, 28),
             cuda=True,
             save_me=True,
             datagetter_name = 'butterfly',
             datagetterParams = dict(
                     background = 'neutral', 
                     dataFilePath = '/home/rob/Dropbox/thesis/2. code/src/data/data.xlsx',
                     root = '/home/rob/Dropbox/thesis/2. code/datasets',
                     classifier_column = 'Sex',
                     transforms = [('resize',dict(size=(28, 28))),
                                   ('hflip'),
                                   ('rotateandscale',{'rotation':1, 'scaling':0.1, 'background':'neutral'}),
                                   ('totensor')],
                     sides = 'D',#'both_in_one',
                 ),
         )



ex = experiment(**config)
ex.run_experiment(n_arand_clusters=2)
