from cpsc_db import *
from torchvision import transforms
# from model import *
from model_modify3 import *
from train4_2 import *

# china physiological signal challenge 2018
# 12 channel, 9 class classification (1 normal, 8 abnormal)
# about 7000 sample
# 500Hz frequency
# unbalanced class distribution
# baseline: f1 score 0.768


# 한번만 처음에 실행하여 파일 만들기
# cpsc = CpscDataset(root_dir=None, record_list=None, pre_processing=None)

# # *** Set this root_dir containing .hea and .mat files ***
# root_dir = 'db/'
# cpsc.pre_pre_processing(root_dir=root_dir, save_dir='dataset')
# print("Finish making py files...")

# Train and Test list
tr_list = np.loadtxt('dataset/train', dtype=str, delimiter=',')
val_list = np.loadtxt('dataset/val', dtype=str, delimiter=',')
# Initialize dataset
train_db = CpscDataset(root_dir='./dataset', # /home/meng/projects/dataset/cpsc
                       record_list=tr_list,
                       pre_processing=transforms.Compose([ToTensor()]))
val_db = CpscDataset(root_dir='./dataset', # /home/meng/projects/dataset/cpsc
                     record_list=val_list,
                     pre_processing=transforms.Compose([ToTensor()]))

# Initialize data loader
batch_size = 64 # default = 32
# train_weights=make_weights_for_balanced_classes(train_db)
ld_tr = Loader(train_db, batch_size=batch_size, shuffle=True, valid_split=0, seed=4, worker=1)
ld_val = Loader(val_db, batch_size=batch_size, shuffle=True, valid_split=0, seed=4, worker=1)
loader_tr = ld_tr.fetch("train")
loader_val = ld_val.fetch("valid")
loader_tr = WrappedDataLoader(loader_tr,
                              x_dim=[12, 5000], y_dim=len(train_db.cls_list))
loader_val = WrappedDataLoader(loader_val,
                               x_dim=[12, 5000], y_dim=len(val_db.cls_list))

# Model
model = ModelArrhythmia(input_shape=[12, 5000],
                        output_shape=len(train_db.cls_list), # 9
                        block_num=3,
                        kernel_size=3,
                        stride=1,
                        padding=1)
if torch.cuda.is_available():
    model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(),
                             weight_decay=1e-4, # 1e-4
                             lr=2e-3) # 1e-3

loss_fn = torch.nn.BCEWithLogitsLoss()  # Including Sigmoid function

fit_fn = Fit(model=model, loss_fn=loss_fn, optimizer=optimizer)

fit_fn.fit(epochs=800, # 800
           dl_train=loader_tr, dl_valid=loader_val,
           classes=train_db.cls_list,
           val_classes=train_db.cls_list)
