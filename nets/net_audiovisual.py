import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import copy
import math
from .models_utils import MultiHeadAttention2, PositionalEncoding2

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None):
        output_a = src_a
        output_v = src_v

        for i in range(self.num_layers):
            output_a = self.layers[i](src_a, src_v, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)
            output_v = self.layers[i](src_v, src_a, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output_a = self.norm1(output_a)
            output_v = self.norm2(output_v)

        return output_a, output_v

class HANLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(HANLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)

        src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)





class ResBlock1D(nn.Module):
    def __init__(self, idim, odim, ksize = 15, stride=1, num_res_blocks = 5, downsample = False):
        super(ResBlock1D, self).__init__() # Must call super __init__()

        self.nblocks = num_res_blocks
        self.do_downsample = downsample

        # set layers
        if self.do_downsample:
            self.downsample = nn.Sequential(
                nn.Conv1d(idim, odim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(odim),
            )
        self.blocks = nn.ModuleList()
        for i in range(self.nblocks):
            cur_block = self.basic_block(idim, odim, ksize, stride)
            self.blocks.append(cur_block)
            if (i == 0) and self.do_downsample:
                idim = odim

    def basic_block(self, idim, odim, ksize=3, stride=1):
        layers = []
        # 1st conv
        p = ksize // 2
        layers.append(nn.Conv1d(idim, odim, ksize, stride, p, bias=False))
        layers.append(nn.BatchNorm1d(odim))
        layers.append(nn.ReLU(inplace=True))
        # 2nd conv
        # layers.append(nn.Conv1d(odim, odim, ksize, stride, p, bias=False))
        # layers.append(nn.BatchNorm1d(odim))

        return nn.Sequential(*layers)

    def forward(self, inp):
        """
        Args:
            inp: [B, idim, H, w]
        Returns:
            answer_label : [B, odim, H, w]
        """
        residual = inp.permute(0, 2, 1)

        for i in range(self.nblocks):
            out = self.blocks[i](residual)
            if (i == 0) and self.do_downsample:
                residual = self.downsample(residual)
            # out += residual
            out = F.relu(out) # w/o is sometimes better
            residual = out

        return out.permute(0, 2, 1)


class HCMN(nn.Module):

    def __init__(self):
        super(HCMN, self).__init__()

        self.fc_prob = nn.Linear(512, 25)
        self.fc_frame_att = nn.Linear(512, 25)
        self.fc_av_att = nn.Linear(512, 25)
        self.fc_a =  nn.Linear(128, 512)
        self.fc_v = nn.Linear(2048, 512)
        self.fc_st = nn.Linear(512, 512)
        self.fc_fusion = nn.Linear(1024, 512)

        self.conv_x1 = nn.ModuleList([ResBlock1D(idim=512, odim=512, ksize=1, stride=1, num_res_blocks=1), ResBlock1D(idim=512, odim=512, ksize=3, stride=2, num_res_blocks=1), ResBlock1D(idim=512, odim=512, ksize=5, stride=5, num_res_blocks=1)]) 
        self.conv_x2 = nn.ModuleList([ResBlock1D(idim=512, odim=512, ksize=1, stride=1, num_res_blocks=1), ResBlock1D(idim=512, odim=512, ksize=3, stride=2, num_res_blocks=1), ResBlock1D(idim=512, odim=512, ksize=5, stride=5, num_res_blocks=1)]) 

        self.hat_encoder = Encoder(HANLayer(d_model=512, nhead=1, dim_feedforward=512), num_layers=1)


    def forward(self, audio, visual, visual_st):

        b, t, d = visual_st.size()
        x1 = self.fc_a(audio)
        x_audio = x1

        vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1)
        vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        
        vid_st = self.fc_st(visual_st)
        x2 = torch.cat((vid_s, vid_st), dim =-1)
        x2 = self.fc_fusion(x2)
        x_visual = x2


        feature_list = []
        for x1_conv_layer, x2_conv_layer in zip(self.conv_x1, self.conv_x2):

            gen_x1 = x1_conv_layer(x1)
            gen_x2 = x2_conv_layer(x2)

            # HAN
            enc_x1, enc_x2 = self.hat_encoder(gen_x1, gen_x2)
            x = torch.cat([enc_x1.unsqueeze(-2), enc_x2.unsqueeze(-2)], dim=-2)
            

            if len(feature_list) == 0:
                feature_list.append(x)
            else:
                pre_x = feature_list[-1]
                x = F.interpolate(x.permute(0,3,2,1), size=[pre_x.shape[2], pre_x.shape[1]], mode="nearest").permute(0,3,2,1) + pre_x
                feature_list.append(x)

        global_prob_list = []
        a_prob_list = []
        v_prob_list = []
        frame_prob_list = []
        for x in feature_list:
            frame_prob = torch.sigmoid(self.fc_prob(x))

            frame_att = torch.softmax(self.fc_frame_att(x), dim=1)
            av_att = torch.softmax(self.fc_av_att(x), dim=2)
            temporal_prob = (frame_att * frame_prob)
            global_prob = (temporal_prob*av_att).sum(dim=2)
            a_prob = temporal_prob[:, :, 0, :]
            v_prob =temporal_prob[:, :, 1, :]

            global_prob = F.interpolate(global_prob.permute(0,2,1), size=[global_prob.shape[1]], mode="nearest").permute(0,2,1)
            a_prob = F.interpolate(a_prob.permute(0,2,1), size=[a_prob.shape[1]], mode="nearest").permute(0,2,1)
            v_prob = F.interpolate(v_prob.permute(0,2,1), size=[v_prob.shape[1]], mode="nearest").permute(0,2,1)

            global_prob_list.append(global_prob)
            a_prob_list.append(a_prob)
            v_prob_list.append(v_prob)
            frame_prob_list.append(frame_prob)

        global_prob = torch.stack(global_prob_list, dim=0).mean(dim=0).sum(dim=1)
        a_prob = torch.stack(a_prob_list, dim=0).mean(dim=0).sum(dim=1)
        v_prob = torch.stack(v_prob_list, dim=0).mean(dim=0).sum(dim=1)
        frame_prob = torch.stack(frame_prob_list, dim=0).mean(dim=0)
   


        return global_prob, a_prob, v_prob, frame_prob, x1, x_visual



class AudioNet(nn.Module):

    def __init__(self):
        super(AudioNet, self).__init__()

        self.fc_prob = nn.Linear(512, 25)
        self.fc_frame_att = nn.Linear(512, 25)
        self.fc_a =  nn.Linear(128, 512)


        self.audio_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=512), num_layers=1)

        self.iter = [0,1,2,3]


    def forward(self, audio, visual, visual_st):

        b, t, d = visual_st.size()
        x1 = self.fc_a(audio)
        x_audio = x1


        feature_list = [x1]
        for it in self.iter:


            enc_x1 = self.audio_encoder(feature_list[-1].permute(1,0,2)).permute(1,0,2)

            if len(feature_list) == 0:
                feature_list.append(enc_x1)
            else:
                pre_x = feature_list[-1]
                enc_x1 = F.interpolate(enc_x1.permute(0,2,1), size=[pre_x.shape[1]], mode="nearest").permute(0,2,1) + pre_x
                feature_list.append(enc_x1)

        x = feature_list[-1]

        frame_prob = torch.sigmoid(self.fc_prob(x))

        # attentive MMIL pooling
        frame_att = torch.softmax(self.fc_frame_att(x), dim=1)
        temporal_prob = (frame_att * frame_prob)
        global_prob = temporal_prob.sum(dim=1)

        return global_prob, frame_prob, x1, x_audio


class VideoNet(nn.Module):

    def __init__(self):
        super(VideoNet, self).__init__()

        self.fc_prob = nn.Linear(512, 25)
        self.fc_frame_att = nn.Linear(512, 25)
        self.fc_v = nn.Linear(2048, 512)
        self.fc_st = nn.Linear(512, 512)
        self.fc_fusion = nn.Linear(1024, 512)

        self.audio_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=512), num_layers=2)

        self.iter = [0,1,2] 

    def forward(self, audio, visual, visual_st):

        vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1)
        vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        
        vid_st = self.fc_st(visual_st)
        x2 = torch.cat((vid_s, vid_st), dim =-1)
        x2 = self.fc_fusion(x2)
        x_visual = x2

        x1 = x_visual

        feature_list = [x1]
        for it in self.iter:
   

            enc_x1 = self.audio_encoder(feature_list[-1].permute(1,0,2)).permute(1,0,2)

            if len(feature_list) == 0:
                feature_list.append(enc_x1)
            else:
                pre_x = feature_list[-1]
                enc_x1 = F.interpolate(enc_x1.permute(0,2,1), size=[pre_x.shape[1]], mode="nearest").permute(0,2,1) + pre_x
                feature_list.append(enc_x1)


        x = feature_list[-1]

        frame_prob = torch.sigmoid(self.fc_prob(x))

        # attentive MMIL pooling
        frame_att = torch.softmax(self.fc_frame_att(x), dim=1)
        temporal_prob = (frame_att * frame_prob)
        global_prob = temporal_prob.sum(dim=1)

        return global_prob, frame_prob, x1, x_visual



class DHHN(nn.Module):

    def __init__(self):
        super(DHHN, self).__init__()
        self.video_net = HCMN()
        self.audio_net = HCMN()
    def forward(self, audio, visual, visual_st):
        _, audio_prob, _, audio_frame_prob, _, _ = self.audio_net(audio, visual, visual_st)
        _, _, video_prob, video_frame_prob, _, _ = self.video_net(audio, visual, visual_st)
        return audio_prob, audio_frame_prob, video_prob, video_frame_prob
    