import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
import numpy as np
from utils.utils import build_pretrain_embedding, load_embeddings
from utils.losses import AsymmetricLoss, AsymmetricLossOptimized
from math import floor, sqrt


class WordRep(nn.Module):
    def __init__(self, args, Y, dicts):
        super(WordRep, self).__init__()

        if args.embed_file:
            print("loading pretrained embeddings from {}".format(args.embed_file))
            if args.use_ext_emb:
                pretrain_word_embedding, pretrain_emb_dim = build_pretrain_embedding(args.embed_file, dicts['w2ind'],
                                                                                     True)
                W = torch.from_numpy(pretrain_word_embedding)
            else:
                W = torch.Tensor(load_embeddings(args.embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            # add 2 to include UNK and PAD
            self.embed = nn.Embedding(len(dicts['w2ind']) + 2, args.embed_size, padding_idx=0)
        self.feature_size = self.embed.embedding_dim

        self.embed_drop = nn.Dropout(p=args.dropout)

        self.conv_dict = {1: [self.feature_size, args.num_filter_maps],
                     2: [self.feature_size, 100, args.num_filter_maps],
                     3: [self.feature_size, 150, 100, args.num_filter_maps],
                     4: [self.feature_size, 200, 150, 100, args.num_filter_maps]
                     }


    def forward(self, x):
        features = [self.embed(x)]

        x = torch.cat(features, dim=2)

        x = self.embed_drop(x)
        return x


class RandomlyInitializedDecoder(nn.Module):
    """
    The original per-label attention network: query matrix is randomly initialized
    """
    def __init__(self, args, Y, dicts, input_size):
        super(RandomlyInitializedDecoder, self).__init__()

        Y = Y[-1]

        self.U = nn.Linear(input_size, Y)
        xavier_uniform(self.U.weight)


        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()


    def forward(self, x, target, text_inputs):
        # attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

        m = alpha.matmul(x)

        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        loss = self.loss_function(y, target)
        return y, loss, alpha, m
    
    def change_depth(self, depth=0):
        # placeholder
        pass


class RACDecoder(nn.Module):
    """
    The decoder proposed by Kim et al. (Code title-guided attention)
    """
    def __init__(self, args, Y, dicts, input_size):
        super(RACDecoder, self).__init__()

        Y = Y[-1]

        self.input_size = input_size

        self.register_buffer("c2title", torch.LongTensor(dicts["c2title"]))
        self.word_rep = WordRep(args, Y, dicts)

        filter_size = int(args.code_title_filter_size)
        self.code_title_conv = nn.Conv1d(self.word_rep.feature_size, input_size,
                                         filter_size, padding=int(floor(filter_size / 2)))
        xavier_uniform(self.code_title_conv.weight)
        self.code_title_maxpool = nn.MaxPool1d(args.num_code_title_tokens)

        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()
    
    def forward(self, x, target, text_inputs):
        code_title = self.word_rep(self._buffers['c2title']).transpose(1, 2)
        # attention
        U = self.code_title_conv(code_title)
        U = self.code_title_maxpool(U).squeeze(-1)
        U = torch.tanh(U)

        attention_score = U.matmul(x.transpose(1, 2)) / sqrt(self.input_size)
        alpha = F.softmax(attention_score, dim=2)

        m = alpha.matmul(x)

        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        loss = self.loss_function(y, target)
        return y, loss, alpha, m

    def change_depth(self, depth=0):
        # placeholder
        pass


class LAATDecoder(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(LAATDecoder, self).__init__()

        Y = Y[-1]

        self.attn_dim = args.attn_dim
        self.W = nn.Linear(input_size, self.attn_dim)
        self.U = nn.Linear(self.attn_dim, Y)
        xavier_uniform(self.W.weight)
        xavier_uniform(self.U.weight)

        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, x, target, text_inputs):
        z = torch.tanh(self.W(x))
        # attention
        alpha = F.softmax(self.U.weight.matmul(z.transpose(1, 2)), dim=2)

        m = alpha.matmul(x)

        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        loss = self.loss_function(y, target)
        return y, loss, alpha, m

    def change_depth(self, depth=0):
        # placeholder
        pass


class Decoder(nn.Module):
    """
    Decoder: knowledge transfer initialization and hyperbolic embedding correction
    """
    def __init__(self, args, Y, dicts, input_size):
        super(Decoder, self).__init__()

        self.dicts = dicts

        self.decoder_dict = nn.ModuleDict()
        for i in range(len(Y)):
            y = Y[i]
            self.decoder_dict[str(i) + '_' + '0'] = nn.Linear(input_size, y)
            self.decoder_dict[str(i) + '_' + '1'] = nn.Linear(input_size, y)
            xavier_uniform(self.decoder_dict[str(i) + '_' + '0'].weight)
            xavier_uniform(self.decoder_dict[str(i) + '_' + '1'].weight)
        
        self.use_hyperbolic =  args.decoder.find("Hyperbolic") != -1
        if self.use_hyperbolic:
            self.cat_hyperbolic = args.cat_hyperbolic
            if not self.cat_hyperbolic:
                self.hyperbolic_fc_dict = nn.ModuleDict()
                for i in range(len(Y)):
                    self.hyperbolic_fc_dict[str(i)] = nn.Linear(args.hyperbolic_dim, input_size)
            else:
                self.query_fc_dict = nn.ModuleDict()
                for i in range(len(Y)):
                    self.query_fc_dict[str(i)] = nn.Linear(input_size + args.hyperbolic_dim, input_size)
            
            # build hyperbolic embedding matrix
            self.hyperbolic_emb_dict = {}
            for i in range(len(Y)):
                self.hyperbolic_emb_dict[i] = np.zeros((Y[i], args.hyperbolic_dim))
                for idx, code in dicts['ind2c'][i].items():
                    self.hyperbolic_emb_dict[i][idx, :] = np.copy(dicts['poincare_embeddings'].get_vector(code))
                self.register_buffer(name='hb_emb_' + str(i), tensor=torch.tensor(self.hyperbolic_emb_dict[i], dtype=torch.float32))

        self.cur_depth = 5 - args.depth
        self.is_init = False
        self.change_depth(self.cur_depth)

        if args.loss == 'BCE':
            self.loss_function = nn.BCEWithLogitsLoss()
        elif args.loss == 'ASL':
            asl_config = [float(c) for c in args.asl_config.split(',')]
            self.loss_function = AsymmetricLoss(gamma_neg=asl_config[0], gamma_pos=asl_config[1],
                                                clip=asl_config[2], reduction=args.asl_reduction)
        elif args.loss == 'ASLO':
            asl_config = [float(c) for c in args.asl_config.split(',')]
            self.loss_function = AsymmetricLossOptimized(gamma_neg=asl_config[0], gamma_pos=asl_config[1],
                                                         clip=asl_config[2], reduction=args.asl_reduction)
    
    def change_depth(self, depth=0):
        if self.is_init:
            # copy previous attention weights to current attention network based on ICD hierarchy
            ind2c = self.dicts['ind2c']
            c2ind = self.dicts['c2ind']
            hierarchy_dist = self.dicts['hierarchy_dist']
            for i, code in ind2c[depth].items():
                tree = hierarchy_dist[depth][code]
                pre_idx = c2ind[depth - 1][tree[depth - 1]]

                self.decoder_dict[str(depth) + '_' + '0'].weight.data[i, :] = self.decoder_dict[str(depth - 1) + '_' + '0'].weight.data[pre_idx, :].clone()
                self.decoder_dict[str(depth) + '_' + '1'].weight.data[i, :] = self.decoder_dict[str(depth - 1) + '_' + '1'].weight.data[pre_idx, :].clone()

        if not self.is_init:
            self.is_init = True

        self.cur_depth = depth
        
    def forward(self, x, target, text_inputs):
        # attention
        if self.use_hyperbolic:
            if not self.cat_hyperbolic:
                query = self.decoder_dict[str(self.cur_depth) + '_' + '0'].weight + self.hyperbolic_fc_dict[str(self.cur_depth)](self._buffers['hb_emb_' + str(self.cur_depth)])
            else:
                query = torch.cat([self.decoder_dict[str(self.cur_depth) + '_' + '0'].weight, self._buffers['hb_emb_' + str(self.cur_depth)]], dim=1)
                query = self.query_fc_dict[str(self.cur_depth)](query)
        else:
            query = self.decoder_dict[str(self.cur_depth) + '_' + '0'].weight

        alpha = F.softmax(query.matmul(x.transpose(1, 2)), dim=2)
        m = alpha.matmul(x)

        y = self.decoder_dict[str(self.cur_depth) + '_' + '1'].weight.mul(m).sum(dim=2).add(self.decoder_dict[str(self.cur_depth) + '_' + '1'].bias)

        loss = self.loss_function(y, target)
        
        return y, loss, alpha, m


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out


class MultiResCNN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    args.dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        if args.decoder == "HierarchicalHyperbolic" or args.decoder == "Hierarchical":
            self.decoder = Decoder(args, Y, dicts, self.filter_num * args.num_filter_maps)
        elif args.decoder == "RandomlyInitialized":
            self.decoder = RandomlyInitializedDecoder(args, Y, dicts, self.filter_num * args.num_filter_maps)
        elif args.decoder == "CodeTitle":
            self.decoder = RACDecoder(args, Y, dicts, self.filter_num * args.num_filter_maps)
        else:
            raise RuntimeError("wrong decoder name")

        self.cur_depth = 5 - args.depth


    def forward(self, x, target, text_inputs):
        x = self.word_rep(x)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)

        y, loss, alpha, m = self.decoder(x, target, text_inputs)

        return y, loss, alpha, m

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

import os
from transformers import LongformerModel, LongformerConfig
class LongformerClassifier(nn.Module):

    def __init__(self, args, Y, dicts):
        super(LongformerClassifier, self).__init__()

        if args.longformer_dir != '':
            print("loading pretrained longformer from {}".format(args.longformer_dir))
            config_file = os.path.join(args.longformer_dir, 'config.json')
            self.config = LongformerConfig.from_json_file(config_file)
            print("Model config {}".format(self.config))
            self.longformer = LongformerModel.from_pretrained(args.longformer_dir, gradient_checkpointing=True)
        else:
            self.config = LongformerConfig(
                attention_mode="longformer",
                attention_probs_dropout_prob=0.1,
                attention_window=[
                    512,
                    512,
                    512,
                    512,
                    512,
                    512,
                ],
                bos_token_id=0,
                eos_token_id=2,
                gradient_checkpointing=False,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                hidden_size=768,
                ignore_attention_mask=False,
                initializer_range=0.02,
                intermediate_size=3072,
                layer_norm_eps=1e-05,
                max_position_embeddings=4098,
                model_type="longformer",
                num_attention_heads=12,
                num_hidden_layers=6,
                pad_token_id=1,
                sep_token_id=2,
                type_vocab_size=1,
                vocab_size=50265
            )
            self.longformer = LongformerModel(self.config)

        # decoder
        self.decoder = Decoder(args, Y, dicts, self.config.hidden_size)
        

    def forward(self, input_ids, token_type_ids, attention_mask, target):
        global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            # global_attention_mask[:, 0] = 1 # this line should be commented if using decoder
        longformer_output = self.longformer(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=False
        )

        output = longformer_output[0]
        y, loss, alpha, m = self.decoder(output, target, None)

        return y, loss, alpha, m

    def freeze_net(self):
        pass


class RACReader(nn.Module):
    def __init__(self, args, Y, dicts):
        super(RACReader, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)
        filter_size = int(args.filter_size)

        self.conv = nn.ModuleList()
        for i in range(args.reader_conv_num):
            conv = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                                padding=int(floor(filter_size / 2)))
            xavier_uniform(conv.weight)
            self.conv.add_module(f'conv_{i+1}', conv)
        
        self.dropout = nn.Dropout(p=args.dropout)

        self.trans = nn.ModuleList()
        for i in range(args.reader_trans_num):
            trans = nn.TransformerEncoderLayer(self.word_rep.feature_size, 1, args.trans_ff_dim, args.dropout, "relu")
            self.trans.add_module(f'trans_{i+1}', trans)

        if args.decoder == "HierarchicalHyperbolic" or args.decoder == "Hierarchical":
            self.decoder = Decoder(args, Y, dicts, self.word_rep.feature_size)
        elif args.decoder == "RandomlyInitialized":
            self.decoder = RandomlyInitializedDecoder(args, Y, dicts, self.word_rep.feature_size)
        elif args.decoder == "CodeTitle":
            self.decoder = RACDecoder(args, Y, dicts, self.word_rep.feature_size)
        else:
            raise RuntimeError("wrong decoder name")
    
    def forward(self, x, target, text_inputs=None):
        x = self.word_rep(x)

        x = x.transpose(1, 2)

        for conv in self.conv:
            x = conv(x)

        x = torch.tanh(x).permute(2, 0, 1)
        x = self.dropout(x)

        for trans in self.trans:
            x = trans(x)
        
        x = x.permute(1, 0, 2)
        
        y, loss, alpha, m = self.decoder(x, target, text_inputs)

        return y, loss, alpha, m
    
    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class LAAT(nn.Module):
    def __init__(self, args, Y, dicts):
        super(LAAT, self).__init__()
        self.word_rep = WordRep(args, Y, dicts)

        self.hidden_dim = args.lstm_hidden_dim
        self.biLSTM = nn.LSTM(
            input_size=self.word_rep.feature_size,
            hidden_size=self.hidden_dim,
            batch_first=True,
            dropout=args.dropout,
            bidirectional=True
        )

        self.output_dim = 2 * self.hidden_dim
        self.use_LAAT = False

        self.attn_dim = args.attn_dim
        self.decoder_name = args.decoder
        if "LAAT" in args.decoder:
            if args.decoder == "LAATHierarchicalHyperbolic" or args.decoder == "LAATHierarchical":
                self.decoder_name = args.decoder[4:]
            self.output_dim = self.attn_dim
            self.use_LAAT = True
            self.W = nn.Linear(2 * self.hidden_dim, self.attn_dim)

        if self.decoder_name == "HierarchicalHyperbolic" or self.decoder_name == "Hierarchical":
            self.decoder = Decoder(args, Y, dicts, self.output_dim)
        elif self.decoder_name == "RandomlyInitialized":
            self.decoder = RandomlyInitializedDecoder(args, Y, dicts, self.output_dim)
        elif self.decoder_name == "CodeTitle":
            self.decoder = RACDecoder(args, Y, dicts, self.output_dim)
        elif self.decoder_name == "LAATDecoder":
            self.decoder = RandomlyInitializedDecoder(args, Y, dicts, self.output_dim)
        else:
            raise RuntimeError("wrong decoder name")

        

        self.cur_depth = 5 - args.depth

    def forward(self, x, target, text_inputs):
        # lengths = (x > 0).sum(dim=1).cpu()
        x = self.word_rep(x)  # [batch, length, input_size]

        # x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x1 = self.biLSTM(x)[0]
        # x1 = pad_packed_sequence(x1, batch_first=True)[0]

        if self.use_LAAT:
            x1 = torch.tanh(self.W(x1))

        y, loss, alpha, m = self.decoder(x1, target, text_inputs)

        return y, loss, alpha, m


def pick_model(args, dicts):
    ind2c = dicts['ind2c']
    Y = [len(ind2c[i]) for i in range(5)] # total number of ICD codes
    if args.model == 'MultiResCNN':
        model = MultiResCNN(args, Y, dicts)
    elif args.model == 'longformer':
        model = LongformerClassifier(args, Y, dicts)
    elif args.model == 'RACReader':
        model = RACReader(args, Y, dicts)
    elif args.model == 'LAAT':
        model = LAAT(args, Y, dicts)
    else:
        raise RuntimeError("wrong model name")

    if args.test_model:
        model.decoder.change_depth(4)
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
    if args.tune_wordemb == False:
        model.freeze_net()
    if len(args.gpu_list) == 1 and args.gpu_list[0] != -1: # single card training
        model.cuda()
    elif len(args.gpu_list) > 1: # multi-card training
        model = nn.DataParallel(model, device_ids=args.gpu_list)
        model = model.to(f'cuda:{model.device_ids[0]}')
    return model
