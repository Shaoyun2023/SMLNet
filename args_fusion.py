
class args():
	# training args
	epochs = 1 # "number of training epochs"
	batch_size = 1  #"batch size for training"
	# the COCO dataset path in your computer

	dataset = "./images/test"
	vgg_model_dir = './models/vgg'

	device = "cuda:0"
	HEIGHT = 256
	WIDTH = 256

	save_model_dir_autoencoder = "models/model"
	save_loss_dir = './models/model/'

	cuda = 1
	ssim_weight = [1,10,100,1000,10000,100000]
	ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4', '1e5']

	lr = 1e-4  #"learning rate"

	lr_light = 1e-4  # "learning rate"
	log_interval = 10  #"number of images after which the training loss is logged"
	resume = None

	# for test, model_default is the model used in paper
	model_default = './models/model/SMLNet.model'





