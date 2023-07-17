import random
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim      # 19215
        self.emb_dim = emb_dim          # 32
        self.enc_hid_dim = enc_hid_dim  # 64
        self.dec_hid_dim = dec_hid_dim  # 64
        self.dropout = dropout

        # nn.Embeddingを用いるとOneHotにする必要がなく効率的。
        # 入力は(batch, seq)が一般的だが、今回は(seq, batch)とする。
        # 内部的に(vocab_size, emb_dim)のembeddingマトリックスの行列積をした出力値となる。
        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)  # ×2はBidirectionalのため

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Tensor]:
        # 今回の入力は(seq, batch)の縦ベクトル、出力は(seq, batch, embed).
        # GRUがその形式の入力を期待しているため.
        embedded = self.dropout(self.embedding(src))

        # outputs shape: (seq, batch, hidden*2)  例：torch.Size([24, 11, 128])
        # 入力token毎の64×2（双方向）の単語ベクトルを出力
        # hidden shape : (2, batch, hidden)  例：torch.Size([2, 11, 64])
        # 最後の隠れ状態ベクトルを出力
        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):  # 8
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,  # (batch, embed)
                encoder_outputs: Tensor  # (seq, batch, embed*2)
                ) -> Tensor:

        src_len = encoder_outputs.shape[0]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch, seq, embed)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (batch, seq, embed*2)
        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim=2)))
        attention = torch.sum(energy, dim=2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor,  # (seq, batch, hidden*2)
                              ) -> Tensor:
        a = self.attention(decoder_hidden, encoder_outputs)
        a = a.unsqueeze(1)  # (batch, 1, seq)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (batch, seq, hidden*2)

        # softmax()を通したattention（どのトークンが重要かの特徴量）とencoder_outputsの内積計算。
        # 注目すべき単語に重みのかかった特徴量を出力する。
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)  # (batch, 1, hidden*2)
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)  # (1, batch, hidden*2)
        return weighted_encoder_rep

    def forward(self,
                input_data: Tensor,  # t-1の出力値
                decoder_hidden: Tensor,   # (batch, 64)
                encoder_outputs: Tensor,  # (seq, batch, hidden*2)
                ) -> Tuple[Tensor]:
        input_data = input_data.unsqueeze(0)
        embedded = self.dropout(self.embedding(input_data))
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)  # (1, batch, hidden*2 + embed)
        # output=decoder_hidden=(1, batch, hidden)
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim=1))
        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:
        """

        Args:
            src（source）: モデルの入力となる系列データ。（トークンID）
                            この例では元の言語の文章（あるいはその他の系列データ）が該当します。
                            縦ベクトルのため、(文章の長さ, バッチサイズ)となる。
            trg（target）: モデルが生成すべき目標の系列データ。
                            この例では翻訳後の言語の文章（あるいはその他の系列データ）が該当します。
                            縦ベクトルのため、(文章の長さ, バッチサイズ)となる。
            teacher_forcing_ratio: トレーニング中に "teacher forcing" という手法をどれだけ使うかを制御します。
                                    "teacher forcing" は、モデルが一つ前の時間ステップで生成した出力ではなく、
                                    正解の出力（target）を次の時間ステップの入力として使用する手法です。
                                    この手法はモデルのトレーニングを早く進めることができますが、
                                    一方でトレーニングとテストでの挙動が異なることから、
                                    過剰に依存するとモデルの性能に悪影響を及ぼす可能性もあります。
        """

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0, :]

        for t in range(1, max_len):
            # Decoder: (t-1の出力、隠れ状態ベクトル、encoderの出力)を入力
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs
