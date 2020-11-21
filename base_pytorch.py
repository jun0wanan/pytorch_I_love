

def base_code():

    kk.repeat(1,2,3)
    kk.contiguous()
    kk.squeeze()

    w=torch.stack(kk,0)
    w=torch.cat([kk,kkk],1)

    weighted_keys = weighted_keys.transpose(1, 2)
    weighted_keys = weighted_keys.view(1, 2)

    alignment = torch.matmul(extended_query, weighted_keys)
    alignment = torch.mm(extended_query, weighted_keys)
    alignment = torch.bmm(extended_query, weighted_keys)

