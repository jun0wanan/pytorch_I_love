

def base_code():

    kk.repeat(1,2,3)
    kk.contiguous()
    kk.squeeze()

    w=torch.stack(kk,0)
    w=torch.cat([kk,kkk],1)

    weighted_keys = weighted_keys.transpose(1, 2)
    weighted_keys = weighted_keys.view(1, 2)
    out = x.permute(0, 2, 1)

    alignment = torch.matmul(extended_query, weighted_keys)
    alignment = torch.mm(extended_query, weighted_keys)
    alignment = torch.bmm(extended_query, weighted_keys)

    params.get("debug_mode", False)
    
    att_k = att_k.unsqueeze(1).expand_as(att_f)

    q4s_feat = se_feats[:,n,:].unsqueeze(1).expand(B, nseg, -1)

    att_feats = attw @ feats

    

def base_code1():
    if isinstance(ptdata, list):
        return [tensor2numpy(dt) for dt in ptdata]
    elif isinstance(ptdata, dict):
        return {k:tensor2numpy(dt) for k,dt in ptdata.items()}
    else:
        return tensor2numpy(ptdata)
