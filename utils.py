
def getGPU():
    device = torch.device("cuda:{}".format(opt.gpu_id) if torch.cuda.is_available() else "cpu")
    opt.device = device


# 设置随机种子
def setSeed():
    # First
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

   # Second 
    np.random.seed(2020)
    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)
    torch.cuda.manual_seed_all(2020)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def visual_loss():
    # train_loss_list in train(epoch) to write and get it
    size = len(train_loss_list)
    X = range(1, size + 1)

    loss_fig = plt.figure('loss').add_subplot(111)
    loss_fig.plot(X, train_loss_list, c='blue', linestyle='-')
    loss_fig.plot(X, val_loss_list, c='red', linestyle='--')
    loss_fig.set_xlabel('Number of iters')
    loss_fig.set_ylabel('Loss')
    loss_fig.legend(['train'])
    plt.show()


def get_save_Modal():
    
    if cfg.train.restore:
        print("Restore checkpoint and optimizer...")
        ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model.pt')
        ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
    
def Save_Modal():
    def save_checkpoint(epoch, model, optimizer, model_kwargs, filename):
        
        #you can save anything you want
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_kwargs': model_kwargs,
        }
        time.sleep(10)
        torch.save(state, filename)

def  visual_Train_now():
    # save log_file
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
    logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(os.path.join(log_file, 'stdout.log'), 'w+')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # show your training 
    sys.stdout.write('~~~~~~ Valid Accuracy: {valid_acc} ~~~~~~~\n'.format(
                valid_acc=colored("{:.4f}".format(valid_acc), "red", attrs=['bold'])))
    sys.stdout.flush()
    sys.stdout.write(
                    "\rProgress = {progress}   ce_loss = {ce_loss}   avg_loss = {avg_loss}    train_mse = {train_mse}    avg_mse = {avg_mse}    exp: {exp_name}".format(
                        progress=colored("{:.3f}".format(progress), "green", attrs=['bold']),
                        ce_loss=colored("{:.4f}".format(loss.item()), "blue", attrs=['bold']),
                        avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold']),
                        train_mse=colored("{:.4f}".format(batch_avg_mse), "blue",
                                          attrs=['bold']),
                        avg_mse=colored("{:.4f}".format(avg_mse), "red", attrs=['bold']),
                        exp_name=cfg.exp_name))
    sys.stdout.flush()

def optimizer_decay():

    def step_decay(cfg, optimizer):
        # compute the new learning rate based on decay rate
        cfg.train.lr *= 0.5
        logging.info("Reduced learning rate to {}".format(cfg.train.lr))
        sys.stdout.flush()
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg.train.lr

        return optimizer

def bug_free():
   #first
    assert cfg.train.k_max_clip_level <= 8
   #second
    try:
    except:

def to_Device():

    def todevice(tensor, device):
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            assert isinstance(tensor[0], torch.Tensor)
            return [todevice(t, device) for t in tensor]
        elif isinstance(tensor, torch.Tensor):
            return tensor.to(device)

def init_modal_weight():
    def init_modules(modules, w_init='kaiming_uniform'):
        if w_init == "normal":
            _init = init.normal_
        elif w_init == "xavier_normal":
            _init = init.xavier_normal_
        elif w_init == "xavier_uniform":
            _init = init.xavier_uniform_
        elif w_init == "kaiming_normal":
            _init = init.kaiming_normal_
        elif w_init == "kaiming_uniform":
            _init = init.kaiming_uniform_
        elif w_init == "orthogonal":
            _init = init.orthogonal_
        else:
            raise NotImplementedError
        for m in modules:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                _init(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, (nn.LSTM, nn.GRU)):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.zeros_(param)
                    elif 'weight' in name:
                        _init(param)
