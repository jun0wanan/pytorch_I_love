class MatrixAttn(nn.Module):

  def __init__(self,linin,linout):
    super().__init__()
    self.attnlin = nn.Linear(linin,linout)

  def get_device(self):
    # return the device of the tensor, either "cpu" 
    # or number specifiing the index of gpu. 
    dev = next(self.parameters()).get_device()
    if dev == -1:
        return "cpu"
    return dev

  def forward(self,dec,emb):
    emb,elen = emb
    # dev = emb.get_device()
    # emask and emb should be in the same device 
    emask = torch.arange(0,emb.size(1)).unsqueeze(0).repeat(emb.size(0),1).long().to(self.get_device())
    
    emask = (emask >= elen.unsqueeze(1)).unsqueeze(1)
    decsmall = self.attnlin(dec)
    unnorm = torch.bmm(decsmall,emb.transpose(1,2))
    unnorm.masked_fill_(emask,-float('inf'))
    attn = F.softmax(unnorm,dim=2)
    out = torch.bmm(attn,emb)
    return out, attn

class BahdanauAttention(nn.Module):
    def __init__(self, num_units, query_size, memory_size):
        super(BahdanauAttention, self).__init__()

        self._num_units = num_units
        self._softmax = nn.Softmax()

        self.query_layer = nn.Linear(query_size, num_units, bias=False)
        self.memory_layer = nn.Linear(memory_size, num_units, bias=False)
        self.alignment_layer = nn.Linear(num_units, 1, bias=False)

    def _score(self, query, keys):
        # Put the query and the keys into Dense layer
        processed_query = self.query_layer(query)
        values = self.memory_layer(keys)

        # since the sizes of processed_query i [B x embedding],
        # we can't directly add it with the keys. therefore, we need
        # to add extra dimension, so the dimension of the query
        # now become [B x 1 x alignment unit size]
        extended_query = processed_query.unsqueeze(1)

        # The original formula is v * tanh(extended_query + values).
        # We can just use Dense layer and feed tanh as the input
        alignment = self.alignment_layer(F.tanh(extended_query + values))

        # Now the alignment size is [B x S x 1], We need to squeeze it
        # so that we can use Softmax later on. Converting to [B x S]
        return alignment.squeeze(2)

    def forward(self, query, keys):
        # Calculate the alignment score
        alignment_score = self._score(query, keys)

        # Put it into softmax to get the weight of every steps
        weight = F.softmax(alignment_score, dim=-1)

        # To get the context, this is the original formula
        # context = sum(weight * keys)
        # In order to multiply those two, we need to reshape the weight
        # from [B x S] into [B x S x 1] for broacasting.
        # The multiplication will result in [B x S x embedding]. Remember,
        # we want the score as the sum over all the steps. Therefore, we will
        # sum it over the 1st index
        context = weight.unsqueeze(2) * keys
        total_context = context.sum(1)

        return total_context, alignment_score


class LuongAttention(nn.Module):
    _SCORE_FN = {
        "dot": "_dot_score",
        "general": "_general_score",
        "concat": "_concat_score"
    }

    def __init__(self,
                 attention_window_size,
                 num_units,
                 query_size,
                 memory_size,
                 alignment="local",
                 score_fn="dot"):
        super(LuongAttention, self).__init__()

        if score_fn not in self._SCORE_FN.keys():
            raise ValueError()

        self._attention_window_size = attention_window_size
        self._softmax = nn.Softmax()
        self._score_fn = score_fn
        self._alignment = alignment

        self.query_layer = nn.Linear(query_size, num_units, bias=False)
        self.predictive_alignment_layer = nn.Linear(num_units, 1, bias=False)
        self.alignment_layer = nn.Linear(num_units, 1, bias=False)

        if score_fn == "general":
            self.general_memory_layer = nn.Linear(
                memory_size, query_size, bias=False)
        elif score_fn == "concat":
            self.concat_memory_layer1 = nn.Linear(
                2 * memory_size, num_units, bias=False)
            self.concat_memory_layer2 = nn.Linear(num_units, 1, bias=False)

    def _dot_score(self, query, keys):
        depth = query.size(-1)
        key_units = keys.size(-1)
        if depth != key_units:
            raise ValueError(
                "Incompatible inner dimensions between query and keys. "
                "Query has units: %d. Keys have units: %d. "
                "Dot score requires you to have same size between num_units in "
                "query and keys" % (depth, key_units))

        # Expand query to [B x 1 x embedding dim] for broadcasting
        extended_query = query.unsqueeze(1)

        # Transpose the keys so that we can multiply it
        tkeys = keys.transpose(1, 2)

        alignment = torch.matmul(extended_query, tkeys)

        # Result of the multiplication will be in size [B x 1 x embedding dim]
        # we can safely squeeze the dimension
        return alignment.squeeze(1)

    def _general_score(self, query, keys):
        weighted_keys = self.general_memory_layer(keys)
        extended_query = query.unsqueeze(1)
        weighted_keys = weighted_keys.transpose(1, 2)

        alignment = torch.matmul(extended_query, weighted_keys)
        return alignment.squeeze(1)

    def _concat_score(self, query, keys):
        expanded_query = query.unsqueeze(1).expand(*keys.size())
        concatenated_hidden = torch.cat([expanded_query, keys], dim=2)
        weighted_concatenated_hidden = self.concat_memory_layer1(
            concatenated_hidden)
        temp_score = F.tanh(weighted_concatenated_hidden)
        alignment = self.concat_memory_layer2(temp_score)

        return alignment.squeeze(2)

    def forward(self, query, keys, key_lengths):
        score_fn = getattr(self, self._SCORE_FN[self._score_fn])
        alignment_score = score_fn(query, keys)

        weight = F.softmax(alignment_score, dim=-1)

        if self._alignment == "local":
            extended_key_lengths = key_lengths.unsqueeze(1)
            preprocessed_query = self.query_layer(query)

            activated_query = F.tanh(preprocessed_query)
            sigmoid_query = F.sigmoid(
                self.predictive_alignment_layer(activated_query))
            predictive_alignment = extended_key_lengths * sigmoid_query

            ai_start = predictive_alignment - self._attention_window_size
            ai_end = predictive_alignment + self._attention_window_size

            std = torch.FloatTensor([self._attention_window_size / 2.]).pow(2)
            alignment_point_dist = (
                extended_key_lengths - predictive_alignment).pow(2)

            alignment_point_dist = (
                -(alignment_point_dist / (2 * std[0]))).exp()
            weight = weight * alignment_point_dist

            contexts = []
            for i in range(weight.size(0)):
                start = ai_start[i].int().data.numpy()[0]
                end = ai_end[i].int().data.numpy()[0]

                aligned_weight = weight[i, start:end]
                aligned_keys = keys[i, start:end]

                aligned_context = aligned_weight.unsqueeze(1) * aligned_keys
                contexts.append(aligned_context.sum(0))

            total_context = torch.stack(contexts, 0)
        elif self._alignment == "global":
            context = weight.unsqueeze(2) * keys
            total_context = context.sum(1)

        return total_context, alignment_score

    @property
    def attention_window_size(self):
        return self._attention_window_size


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 dropout_p=0.5,
                 h=8,
                 is_masked=False):
        super(MultiHeadAttention, self).__init__()

        if query_dim != key_dim:
            raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        self._key_dim = torch.tensor(key_dim,requires_grad=False).float()
        self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)
        self.bn = nn.BatchNorm1d(num_units)
        self.ln = nn.LayerNorm(num_units)

    def get_device(self):
        # return the device of the tensor, either "cpu" 
        # or number specifiing the index of gpu. 
        dev = next(self.parameters()).get_device()
        if dev == -1:
            return "cpu"
        return dev

    def forward(self, query, keys, mask=None):
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)

        # calculate QK^T
        attention = torch.matmul(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)

        # attention and _key_dim should be in the same device.
        attention = attention / torch.sqrt(self._key_dim).to(self.get_device())

        if mask is not None:
          mask = mask.repeat(self._h,1,1)
          attention.masked_fill_(mask,-float('inf'))
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        attention = F.dropout(attention, self._dropout_p)
        # multiplyt it with V
        attention = torch.matmul(attention, V)
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2)
        # residual connection
        attention += query
        # apply batch normalization
        #attention = self.bn(attention.transpose(1, 2)).transpose(1, 2)
        # apply layer normalization
        #attention = self.ln(attention)

        return attention

# video
class SpatialAttentionModule(nn.Module):

    def __init__(self, input_size=3072, feat_dim=7, hidden_size=512, dropout=0.2):
        """Set the hyper-parameters and build the layers."""
        super(SpatialAttentionModule, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feat_dim = feat_dim
        
        # alignment model
        # see appendices A.1.2 of neural machine translation
        self.Wa = nn.Parameter(torch.FloatTensor(input_size, hidden_size),requires_grad=True)
        self.Ua = nn.Parameter(torch.FloatTensor(hidden_size*2, hidden_size),requires_grad=True)
        self.Va = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.ba = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)
        
        self.drop_keep_prob_image_embed_t = nn.Dropout(dropout)
         
        self.init_weights()
        
    def init_weights(self):
        self.Wa.data.normal_(0.0, 0.01)
        self.Ua.data.normal_(0.0, 0.01)
        self.Va.data.normal_(0.0, 0.01)
        self.ba.data.fill_(0)
        
        

    def forward(self, hidden_frames, hidden_text):
        
        # hidden_text:  1 x 1024 (tgif-qa paper Section 4.2, use a two layer one-directional LSTM, combining each layer's hidden)
        # hidden_frame: 1 x 7 x 7 x 3072 (from C3D and resnet, 1024+2048 = 3072)
        assert self.feat_dim==hidden_frames.size(2)
        hidden_frames = hidden_frames.view(hidden_frames.size(0), hidden_frames.size(1) * hidden_frames.size(2), hidden_frames.size(3))

        # precompute Uahj, see Appendix A.1.2 Page12, last sentence, 
        # NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
        Uh = torch.matmul(hidden_text, self.Ua)  # (1,512)
        Uh = Uh.view(Uh.size(0),1,Uh.size(1)) # (1,1,512)
        
        # see appendices A.1.2 of neural machine translation
        Ws = torch.matmul(hidden_frames, self.Wa) # (1,49,512)
        att_vec = torch.matmul( torch.tanh(Ws + Uh + self.ba), self.Va )
        att_vec = F.softmax(att_vec, dim=1) # normalize by Softmax, see Eq(15)
        att_vec = att_vec.view(att_vec.size(0),att_vec.size(1),1) # expand att_vec from 1x49 to 1x49x1
        
        # Hori ICCV 2017
        # Eq(10) c_i
        ht_weighted = att_vec * hidden_frames
        ht_sum = torch.sum(ht_weighted, dim=1)
        ht_sum = self.drop_keep_prob_image_embed_t(ht_sum)
        return ht_sum

class TemporalAttentionModule(nn.Module):

    def __init__(self, input_size, hidden_size=512):
        """Set the hyper-parameters and build the layers."""
        super(TemporalAttentionModule, self).__init__()
        self.input_size = input_size   # in most cases, 2*hidden_size
        self.hidden_size = hidden_size

        self.Wa = nn.Parameter(torch.FloatTensor(input_size, hidden_size),requires_grad=True)
        self.Ua = nn.Parameter(torch.FloatTensor(input_size, hidden_size),requires_grad=True)
        self.Va = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.ba = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)
        
        self.init_weights()
        
        
    def init_weights(self):
        self.Wa.data.normal_(0.0, 0.01)
        self.Ua.data.normal_(0.0, 0.01)
        self.Va.data.normal_(0.0, 0.01)
        self.ba.data.fill_(0)


    def forward(self, hidden_frames, hidden_text, inv_attention=False):

        Uh = torch.matmul(hidden_text, self.Ua)  # (1,512)
        Uh = Uh.view(Uh.size(0),1,Uh.size(1)) # (1,1,512)

        # see appendices A.1.2 of neural machine translation
        # Page 12 last line
        Ws = torch.matmul(hidden_frames, self.Wa) # (1,T,512)
        #print('Temporal Ws size',Ws.size())       # (1, T, 512)
        att_vec = torch.matmul( torch.tanh(Ws + Uh + self.ba), self.Va )
        
        if inv_attention==True:
            att_vec = - att_vec
        
        att_vec = F.softmax(att_vec, dim=1) # normalize by Softmax, see Eq(15)
        att_vec = att_vec.view(att_vec.size(0),att_vec.size(1),1) # expand att_vec from 1xT to 1xTx1 

        # Hori ICCV 2017
        # Eq(10) c_i
        ht_weighted = att_vec * hidden_frames
        ht_sum = torch.sum(ht_weighted, dim=1)
        return ht_sum

class MultiModalAttentionModule(nn.Module):

    def __init__(self, hidden_size=512, simple=False):
        """Set the hyper-parameters and build the layers."""
        super(MultiModalAttentionModule, self).__init__()

        self.hidden_size = hidden_size
        self.simple=simple
        
        # alignment model
        # see appendices A.1.2 of neural machine translation
        
        self.Wav = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wat = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Uav = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Uat = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vav = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.Vat = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.bav = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)
        self.bat = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)

        self.Whh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wvh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wth = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.bh = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)

        self.video_sum_encoder = nn.Linear(hidden_size, hidden_size) 
        self.question_sum_encoder = nn.Linear(hidden_size, hidden_size) 

        self.Wb = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vbv = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vbt = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.bbv = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.bbt = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.wb = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.init_weights()
        
    def init_weights(self):

        self.Wav.data.normal_(0.0, 0.1)
        self.Wat.data.normal_(0.0, 0.1)
        self.Uav.data.normal_(0.0, 0.1)
        self.Uat.data.normal_(0.0, 0.1)
        self.Vav.data.normal_(0.0, 0.1)
        self.Vat.data.normal_(0.0, 0.1)
        self.bav.data.fill_(0)
        self.bat.data.fill_(0)

        self.Whh.data.normal_(0.0, 0.1)
        self.Wvh.data.normal_(0.0, 0.1)
        self.Wth.data.normal_(0.0, 0.1)
        self.bh.data.fill_(0)

        self.Wb.data.normal_(0.0, 0.01)
        self.Vbv.data.normal_(0.0, 0.01)
        self.Vbt.data.normal_(0.0, 0.01)
        self.wb.data.normal_(0.0, 0.01)
        
        self.bbv.data.fill_(0)
        self.bbt.data.fill_(0)

        
    def forward(self, h, hidden_frames, hidden_text, inv_attention=False):
        
        #print self.Uav
        # hidden_text:  1 x T1 x 1024 (looks like a two layer one-directional LSTM, combining each layer's hidden)
        # hidden_frame: 1 x T2 x 1024 (from video encoder output, 1024 is similar from above)

        #print hidden_frames.size(),hidden_text.size()
        Uhv = torch.matmul(h, self.Uav)  # (1,512)
        Uhv = Uhv.view(Uhv.size(0),1,Uhv.size(1)) # (1,1,512)

        Uht = torch.matmul(h, self.Uat)  # (1,512)
        Uht = Uht.view(Uht.size(0),1,Uht.size(1)) # (1,1,512)
        
        #print Uhv.size(),Uht.size()
        
        Wsv = torch.matmul(hidden_frames, self.Wav) # (1,T,512)
        #print Wsv.size()
        att_vec_v = torch.matmul( torch.tanh(Wsv + Uhv + self.bav), self.Vav )
        
        Wst = torch.matmul(hidden_text, self.Wat) # (1,T,512)
        att_vec_t = torch.matmul( torch.tanh(Wst + Uht + self.bat), self.Vat )
        
        if inv_attention==True:
            att_vec_v = -att_vec_v
            att_vec_t = -att_vec_t
            

        att_vec_v = torch.softmax(att_vec_v, dim=1)
        att_vec_t = torch.softmax(att_vec_t, dim=1)

        
        att_vec_v = att_vec_v.view(att_vec_v.size(0),att_vec_v.size(1),1) # expand att_vec from 1xT to 1xTx1 
        att_vec_t = att_vec_t.view(att_vec_t.size(0),att_vec_t.size(1),1) # expand att_vec from 1xT to 1xTx1 
                
        hv_weighted = att_vec_v * hidden_frames
        hv_sum = torch.sum(hv_weighted, dim=1)
        hv_sum2 = self.video_sum_encoder(hv_sum)

        ht_weighted = att_vec_t * hidden_text
        ht_sum = torch.sum(ht_weighted, dim=1)
        ht_sum2 = self.question_sum_encoder(ht_sum)        
        
        
        Wbs = torch.matmul(h, self.Wb)
        mt1 = torch.matmul(ht_sum, self.Vbt) + self.bbt + Wbs
        mv1 = torch.matmul(hv_sum, self.Vbv) + self.bbv + Wbs
        mtv =  torch.tanh(torch.cat([mv1,mt1],dim=0))
        mtv2 = torch.matmul(mtv, self.wb)
        beta = torch.softmax(mtv2,dim=0)

        output = torch.tanh( torch.matmul(h,self.Whh) + beta[0] * hv_sum2 + 
                             beta[1] * ht_sum2 + self.bh )
        output = output.view(output.size(1),output.size(2))
        
        return output

class AttentionTwoStream(nn.Module):

    def __init__(self, feat_channel, feat_dim, text_embed_size, hidden_size, vocab_size, num_layers, word_matrix,
                    answer_vocab_size=None, max_len=20, dropout=0.2, mm_version=1, useSpatial=False, useNaive=False, mrmUseOriginFeat=False,
                    iter_num=3):
        """Set the hyper-parameters and build the layers."""
        super(AttentionTwoStream, self).__init__()
                
        
        # text input size
        self.text_embed_size = text_embed_size # should be 300
        
        # video input size
        self.feat_channel = feat_channel
        self.feat_dim = feat_dim # should be 7
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.useNaive = useNaive
        self.mrmUseOriginFeat = mrmUseOriginFeat
        self.useSpatial = useSpatial
        self.mm_version = mm_version
        self.iter_num = iter_num
        
        self.TpAtt_a = TemporalAttentionModule(hidden_size*2, hidden_size)
        self.TpAtt_m = TemporalAttentionModule(hidden_size*2, hidden_size)

        if useSpatial:
            self.SpAtt = SpatialAttentionModule(feat_channel, feat_dim, hidden_size)
        else:
            self.video_encoder = nn.Linear(feat_channel, hidden_size) 

        self.drop_keep_prob_final_att_vec = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, text_embed_size)
                    
        self.lstm_text_1 = nn.LSTMCell(text_embed_size, hidden_size)
        self.lstm_text_2 = nn.LSTMCell(hidden_size, hidden_size)

        self.lstm_video_1a = nn.LSTMCell(4096, hidden_size)
        self.lstm_video_2a = nn.LSTMCell(hidden_size, hidden_size)
        
        self.lstm_video_1m = nn.LSTMCell(4096, hidden_size)
        self.lstm_video_2m = nn.LSTMCell(hidden_size, hidden_size)
        
        
        if mm_version==1:
            self.lstm_mm_1 = nn.LSTMCell(hidden_size, hidden_size)
            self.lstm_mm_2 = nn.LSTMCell(hidden_size, hidden_size)
            self.linear_decoder_mem = nn.Linear(hidden_size * 2, hidden_size) 
            self.hidden_encoder_1 = nn.Linear(hidden_size * 2, hidden_size) 
            self.hidden_encoder_2 = nn.Linear(hidden_size * 2, hidden_size) 
        
        else:
            self.gru_mm = nn.GRUCell(hidden_size, hidden_size)
            self.linear_decoder_mem = nn.Linear(hidden_size, hidden_size) 
        

        self.mm_att = MultiModalAttentionModule(hidden_size)
        self.linear_decoder_att_a = nn.Linear(hidden_size * 2, hidden_size) 
        self.linear_decoder_att_m = nn.Linear(hidden_size * 2, hidden_size) 

        
        if answer_vocab_size is not None:
            self.linear_decoder_count_2 = nn.Linear(hidden_size * 2 + hidden_size, answer_vocab_size)
        else:
            self.linear_decoder_count_2 = nn.Linear(hidden_size * 2 + hidden_size, 1)    # Count is regression problem
                
        self.max_len = max_len

        self.mrm_vid = MemoryRamTwoStreamModule(hidden_size, hidden_size, max_len)
        self.mrm_txt = MemoryRamModule(hidden_size, hidden_size, max_len)

        self.init_weights(word_matrix)


    def init_weights(self, word_matrix):
        """Initialize weights."""

        if word_matrix is None:
            self.embed.weight.data.uniform_(-0.1, 0.1)
        else:
            # init embed from glove
            self.embed.weight.data.copy_(torch.from_numpy(word_matrix))
        
        self.mrm_vid.init_weights()
        self.mrm_txt.init_weights()

    def init_hiddens(self):
        s_t = torch.zeros(1, self.hidden_size).cuda()
        s_t2 = torch.zeros(1, self.hidden_size).cuda()
        c_t = torch.zeros(1, self.hidden_size).cuda()
        c_t2 = torch.zeros(1, self.hidden_size).cuda()
        return s_t,s_t2,c_t,c_t2
    
    
    def mm_module_v1(self,svt_tmp,memory_ram_vid,memory_ram_txt,loop=3):
    
        sm_q1,sm_q2,cm_q1,cm_q2 = self.init_hiddens()
        mm_oo = self.drop_keep_prob_final_att_vec(torch.tanh(self.hidden_encoder_1(svt_tmp)))
        
        for _ in range(loop):
        
            sm_q1, cm_q1 = self.lstm_mm_1(mm_oo, (sm_q1, cm_q1))
            sm_q2, cm_q2 = self.lstm_mm_2(sm_q1, (sm_q2, cm_q2))
        
            mm_o1 = self.mm_att(sm_q2,memory_ram_vid,memory_ram_txt)
            mm_o2 = torch.cat((sm_q2,mm_o1),dim=1)
            mm_oo = self.drop_keep_prob_final_att_vec(torch.tanh(self.hidden_encoder_2(mm_o2)))
        
        smq = torch.cat( (sm_q1,sm_q2), dim=1)

        return smq
    
    
    def mm_module_v2(self,memory_ram_vid,memory_ram_txt,loop=5):
        
        h_t = torch.zeros(1, self.hidden_size).cuda()
                
        for _ in range(loop):
            mm_o = self.mm_att(h_t,memory_ram_vid,memory_ram_txt)
            h_t = self.gru_mm(mm_o, h_t)
        
        return h_t
    
    
    def forward(self, data_dict):
        ret = self.forward_frameqa(data_dict)
        return ret

    
    def forward_frameqa(self, data_dict):
        
        video_features, numImgs = data_dict['video_features'], data_dict['video_lengths'],
        questions, question_lengths = data_dict['question_words'], data_dict['question_lengths']

        outputs = []
        predictions = []
        bsize = len(questions)
        batch_size = len(questions)  # batch size has to be 1
        features_questions = self.embed(questions)

        
        for j in range(batch_size):
            nImg = numImgs[j]
            nQuestionWords = question_lengths[j]

            ################################
            # slice the input image features
            ################################
            feature = video_features[j,video_features.size(1)-nImg:]
            #print('current video feature size', feature.size())            

            
            #############################             
            # run text encoder first time
            #############################
            s1_t1,s1_t2,c1_t1,c1_t2 = self.init_hiddens()
            
            for i in xrange(nQuestionWords):
                input_question = features_questions[j,i:i+1]
                s1_t1, c1_t1 = self.lstm_text_1(input_question, (s1_t1, c1_t1))
                s1_t2, c1_t2 = self.lstm_text_2(s1_t1, (s1_t2, c1_t2))
            
            # here s1_t1, s1_t2 is the last hidden
            s1_t = torch.cat( (s1_t1,s1_t2), dim=1)  # should be of size (1,1024)
            
            
            
            ###########################################             
            # run video encoder with spatial attention
            ###########################################
            sV_t1a,sV_t2a,cV_t1a,cV_t2a = s1_t1,s1_t2,c1_t1,c1_t2
            sV_t1m,sV_t2m,cV_t1m,cV_t2m = s1_t1,s1_t2,c1_t1,c1_t2

            # record each time t, hidden states, for later temporal attention after text encoding
            hidden_array_1a = []
            hidden_array_2a = []
            hidden_array_1m = []
            hidden_array_2m = []
            
            for i in xrange(nImg):

                if self.useSpatial:
                    input_frame = feature[i:i+1]
                    feat_att = self.SpAtt(input_frame, s1_t)
                else:
                    feat_att_m = feature[i:i+1,0,0,:4096]
                    feat_att_a = feature[i:i+1,0,0,4096:]
                
                sV_t1m, cV_t1m = self.lstm_video_1m(feat_att_m, (sV_t1m, cV_t1m))
                sV_t2m, cV_t2m = self.lstm_video_2m(sV_t1m, (sV_t2m, cV_t2m))

                sV_t1a, cV_t1a = self.lstm_video_1a(feat_att_a, (sV_t1a, cV_t1a))
                sV_t2a, cV_t2a = self.lstm_video_2a(sV_t1a, (sV_t2a, cV_t2a))
                
                sV_t1a_vec = sV_t1a.view(sV_t1a.size(0),1,sV_t1a.size(1))
                sV_t2a_vec = sV_t2a.view(sV_t2a.size(0),1,sV_t2a.size(1))
            
                hidden_array_1a.append(sV_t1a_vec)
                hidden_array_2a.append(sV_t2a_vec)
            
                sV_t1m_vec = sV_t1m.view(sV_t1m.size(0),1,sV_t1m.size(1))
                sV_t2m_vec = sV_t2m.view(sV_t2m.size(0),1,sV_t2m.size(1))
            
                hidden_array_1m.append(sV_t1m_vec)
                hidden_array_2m.append(sV_t2m_vec)
                

            sV_l1a = torch.cat(hidden_array_1a, dim=1)
            sV_l2a = torch.cat(hidden_array_2a, dim=1)
            sV_l1m = torch.cat(hidden_array_1m, dim=1)
            sV_l2m = torch.cat(hidden_array_2m, dim=1)
        
            sV_lla = torch.cat((sV_l1a,sV_l2a), dim=2)
            sV_llm = torch.cat((sV_l1m,sV_l2m), dim=2)

               
            #############################             
            # run text encoder second time
            #############################
            sT_t1,sT_t2,cT_t1,cT_t2 = self.init_hiddens()
            sT_t1,sT_t2 = sV_t1a+sV_t1m, sV_t2a+sV_t2m
            
            hidden_array_3 = []
            
                
            for i in xrange(nQuestionWords):
                input_question = features_questions[j,i:i+1]
                sT_t1, cT_t1 = self.lstm_text_1(input_question, (sT_t1, cT_t1))
                sT_t2, cT_t2 = self.lstm_text_2(sT_t1, (sT_t2, cT_t2))
                hidden_array_3.append(sT_t2)

            # here sT_t1, sT_t2 is the last hidden
            sT_t = torch.cat( (sT_t1,sT_t2), dim=1)  # should be of size (1,1024)
            
            
            #####################
            # temporal attention
            #####################
            vid_att_a = self.TpAtt_a(sV_lla, sT_t)
            vid_att_m = self.TpAtt_m(sV_llm, sT_t)


            ################
            # ram memory
            ################
            sT_rl = torch.cat(hidden_array_3, dim=0)

            memory_ram_vid = self.mrm_vid(sV_l2a[0,:,:], sV_l2m[0,:,:], nImg)
            memory_ram_txt = self.mrm_txt(sT_rl, nQuestionWords)
                
            svt_tmp = torch.cat((sV_t2a,sV_t2m),dim=1)
            smq = self.mm_module_v1(svt_tmp,memory_ram_vid,memory_ram_txt,self.iter_num)

                
            ######################### 
            # decode the final output
            ######################### 
            
            final_embed_a = torch.tanh( self.linear_decoder_att_a(vid_att_a) )
            final_embed_m = torch.tanh( self.linear_decoder_att_m(vid_att_m) )
            final_embed_2 = torch.tanh( self.linear_decoder_mem(smq) )
            final_embed = torch.cat([final_embed_a,final_embed_m,final_embed_2],dim=1)

            output = self.linear_decoder_count_2(final_embed)
            outputs.append(output)
            
            _,mx_idx = torch.max(output,1)
            predictions.append(mx_idx)

        outputs = torch.cat(outputs, 0)
        predictions = torch.cat(predictions, 0)
        return outputs, predictions
    

    def accuracy(self, logits, targets):
        correct = torch.sum(logits.eq(targets)).float()
        return correct * 100.0 / targets.size(0)