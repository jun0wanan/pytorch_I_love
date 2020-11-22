

def mask_get():
    def apply_mask1d(attention, image_locs):
        batch_size, num_loc = attention.size()
        tmp1 = attention.new_zeros(num_loc)
        tmp1[:num_loc] = torch.arange(
            0, num_loc, dtype=attention.dtype).unsqueeze(0)

        tmp1 = tmp1.expand(batch_size, num_loc)
        tmp2 = image_locs.type(tmp1.type())
        tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
        mask = torch.ge(tmp1, tmp2)
        attention = attention.masked_fill(mask, -1e30)
        return attention

    def mask_fill():

        alpha = alpha.masked_fill(feat_masks.float().eq(0), -1e9) # dimension is same