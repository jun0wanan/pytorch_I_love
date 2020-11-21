

def modal_module():

    # define a module  from 1 to ...
    for t in range(cfg.MSG_ITER_NUM):
        qInput_layer2 = ops.Linear(cfg.CMD_DIM, cfg.CMD_DIM)
        setattr(self, "qInput%d" % t, qInput_layer2)

    qInput_layer2 = getattr(self, "qInput%d" % t)

