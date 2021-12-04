from cpsc_db import *
from torchvision import transforms
from model import *
from train import *

cpsc = CpscDataset(root_dir=None,
                   record_list=None,
                   pre_processing=None)

# *** Set this root_dir containing .hea and .mat files ***
root_dir = './db/'
cpsc.pre_pre_processing(root_dir=root_dir,
                        save_dir='dataset')

# Train and Test list
tr_list = np.loadtxt('./dataset/train', dtype=str, delimiter=',')
val_list = np.loadtxt('./dataset/val', dtype=str, delimiter=',')
# Initialize dataset
train_db = CpscDataset(root_dir='./dataset',
                       record_list=tr_list,
                       pre_processing=transforms.Compose([
                           ToTensor()
                       ]))
val_db = CpscDataset(root_dir='./dataset',
                     record_list=val_list,
                     pre_processing=transforms.Compose([
                          ToTensor()
                      ]))
# Initialize data loader
batch_size = 64  # default = 32
ld_tr = Loader(train_db, batch_size=batch_size, shuffle=True, valid_split=0, seed=4, worker=1)
ld_val = Loader(val_db, batch_size=batch_size, shuffle=True, valid_split=0, seed=4, worker=1)
loader_tr = ld_tr.fetch()
loader_val = ld_val.fetch()
loader_tr = WrappedDataLoader(loader_tr,
                              x_dim=[12, 5000], y_dim=len(train_db.cls_list))
loader_val = WrappedDataLoader(loader_val,
                               x_dim=[12, 5000], y_dim=len(val_db.cls_list))

model = ModelArrhythmia(input_shape=[12, 5000],
                        output_shape=len(train_db.cls_list),
                        n_blocks=15,
                        init_channel=32,
                        kernel_size=15,
                        dilation=1)
if torch.cuda.is_available():
    model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(),
                             weight_decay=1e-4,
                             lr=1e-3)

loss_fn = torch.nn.BCEWithLogitsLoss()  # Including Sigmoid function

fit_fn = Fit(model=model, loss_fn=loss_fn,
             optimizer=optimizer)

fit_fn.fit(epochs=20,
           dl_train=loader_tr, dl_valid=loader_val,
           classes=train_db.cls_list,
           val_classes=train_db.cls_list)
