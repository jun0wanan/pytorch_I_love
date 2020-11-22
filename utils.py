
def getGPU():
    device = torch.device("cuda:{}".format(opt.gpu_id) if torch.cuda.is_available() else "cpu")
    opt.device = device

    self.device = (
      torch.device("cuda", self.hparams.gpu_ids[0])
      if self.hparams.gpu_ids[0] >= 0
      else torch.device("cpu")
    )


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

def visual_Train_now():
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

    def adjust_lr_clevr(curr_los, prev_loss, curr_lr):
        loss_diff = prev_loss - curr_los
        not_improve = (
            (loss_diff < 0.015 and prev_loss < 0.5 and curr_lr > 0.00002) or
            (loss_diff < 0.008 and prev_loss < 0.15 and curr_lr > 0.00001) or
            (loss_diff < 0.003 and prev_loss < 0.10 and curr_lr > 0.000005))

        next_lr = curr_lr * cfg.TRAIN.SOLVER.LR_DECAY if not_improve else curr_lr
        return next_lr
    
    def adjust_lr_1():
        lr_default = args.lr if eval_loader is not None else 7e-4
        lr_decay_step = 2
        lr_decay_rate = .25
        lr_decay_epochs = range(10, 30, lr_decay_step) if eval_loader is not None else range(10, 20, lr_decay_step)
        gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]
        if epoch < 4:
            optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            lr = optim.param_groups[0]['lr']
        elif epoch in lr_decay_epochs:
            optim.param_groups[0]['lr'] *= lr_decay_rate
            lr =  optim.param_groups[0]['lr']
        else:
            lr = optim.param_groups[0]['lr']

        
    

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
    # in module use
    init_modules(self.modules(), w_init="xavier_uniform")
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

def set_optimizer():

    # see module's requires_grad
    self.trainable_params = [
            p for p in self.model.parameters() if p.requires_grad]
    self.optimizer = torch.optim.Adam(
            self.trainable_params, lr=cfg.TRAIN.SOLVER.LR)

def get_time():
    tm = timer.Timer() 
    run_duration = tm.get_duration()
    tm.reset()

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

def see_result():
    def tensor2numpy(ptdata):
        return ptdata.detach().cpu().numpy()
    def to_data(ptdata):
        if ptdata is None: return ptdata
        if isinstance(ptdata, list):
            return [tensor2numpy(dt) for dt in ptdata]
        elif isinstance(ptdata, dict):
            return {k:tensor2numpy(dt) for k,dt in ptdata.items()}
        else:
            return tensor2numpy(ptdata)

def other_():

    nn.utils.clip_grad_norm(model.parameters(), 0.25)
