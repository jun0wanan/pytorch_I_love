

def modal_module():

    # define a module  from 1 to ...
    for t in range(cfg.MSG_ITER_NUM):
        qInput_layer2 = ops.Linear(cfg.CMD_DIM, cfg.CMD_DIM)
        setattr(self, "qInput%d" % t, qInput_layer2)

    qInput_layer2 = getattr(self, "qInput%d" % t)

    # define many same module
    nn.ModuleList([nn.Linear(self.qdim, self.qdim) for i in range(self.nse)])
    self.global_emb_fn[n](q_feats)

    # you can define a function
    def _make_modulelist(self, net, n):
        assert n > 0
        new_net_list = nn.ModuleList()
        new_net_list.append(net)
        if n > 1:
            for i in range(n-1):
                new_net_list.append(copy.deepcopy(net))
        return new_net_list
    
    def make_module2():
        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

     
def fusion_method():

    if self.mm_fusion_method == "concat":
            fused_feat = torch.cat([s_feats, q4s_feat], dim=2)
            fused_feat = torch.relu(self.lin_fn[n](fused_feat))
    elif self.mm_fusion_method == "add":
            fused_feat = s_feats + q4s_feat
    elif self.mm_fusion_method == "mul":
            fused_feat = self.fusion_fn[n]([s_feats, q4s_feat])
    else:
            raise NotImplementedError()


def get_modal():

    model = getattr(base_model, constructor)(args, args.nhid).cuda()

def activation(act):
    if act == 'RELU':
        return nn.ReLU()
    elif act == 'TANH':
        return nn.Tanh()
    elif act == 'GLU':
        return nn.GLU()
    elif act == 'ELU':
        return nn.ELU(cfg.MODEL.BILINEAR.ELU_ALPHA)
    elif act == 'CELU':
        return nn.CELU(cfg.MODEL.BILINEAR.ELU_ALPHA)
    else:
        return nn.Identity()