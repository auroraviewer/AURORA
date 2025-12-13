import torch
import torch.nn as nn
class AURORA(nn.Module):
    def __init__(self, input_size, output_size, device, 
                 celltype_num, bulk_size = 1212, 
                 bidirectional=False,
                 use_bulk=True,
                 pred_celltype=True,
                 finetune=False):
        super(AURORA, self).__init__()
        self.input_size = input_size
        self.bulk_size = bulk_size
        self.device = device
        self.bidirectional = bidirectional
        self.use_bulk = use_bulk
        self.pred_celltype = pred_celltype
        major_num = celltype_num
        self.finetune = finetune
        #TODO: Try using self-attention for bulk_encoder as in Geneformer (or simply use their model then do another Feedforward)
        self.bulk_encoder = nn.Sequential(
            nn.Linear(bulk_size, 256),
            nn.LeakyReLU()
        )
        self.bulk_encoder.to(device)
        self.lstm = nn.LSTM(input_size, 256, batch_first=True, device=device, bidirectional=self.bidirectional)
        if not finetune:
            if self.bidirectional:
                self.decoder = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, output_size),
                    #nn.ELU()
                    nn.Sigmoid()
                )
            else:
                self.decoder = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, output_size),
                    #nn.ELU()
                    nn.Sigmoid()
                )
            self.decoder.to(device)
        else:
            self.new_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size),
            #nn.ELU()
            nn.Sigmoid()
        )
            self.new_decoder.to(device)
        self.celltype_decoder = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(32, major_num),
            #nn.ELU()
            # nn.Clamp(min=-5, max=0),
            nn.LogSoftmax(dim=-1)
        )
        self.celltype_decoder.to(device)
    #
    def forward(self, emb_hierrachy, bulk):
        if self.bidirectional:
            d = 2
        else:
            d = 1
        if self.use_bulk:
            bulk_emb = self.bulk_encoder(bulk)
            hierrachy_emb, _ = self.lstm(emb_hierrachy,
                                        (bulk_emb.repeat(d,emb_hierrachy.shape[0],1),
                                        torch.zeros(d,emb_hierrachy.shape[0],256,device=self.device)))
        else:
            hierrachy_emb, _ = self.lstm(emb_hierrachy)
        hierrachy_emb = torch.unbind(hierrachy_emb, dim=1)
        if not self.finetune:
            results = [self.decoder(s) for s in hierrachy_emb]
        else:
            results = [self.new_decoder(s) for s in hierrachy_emb]
        output = torch.stack(results, dim=1)
        if self.pred_celltype:
            major_prop = [self.celltype_decoder(hierrachy_emb[s]) for s in range(len(hierrachy_emb))]
            major_prop = torch.stack(major_prop, dim=1)
        else:
            major_prop = None
        return output, major_prop