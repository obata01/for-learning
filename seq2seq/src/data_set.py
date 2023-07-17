import io
import os
import logging
import torch
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url, extract_archive


logger = logging.getLogger()


class DataSets:  # pylint: disable=R0902,R0903
    """datasets"""
    def __init__(self, mode: str, output_dir: str | None = None):
        if mode not in ["train", "val", "test"]:
            raise ValueError()
        self._mode = mode
        self._output_dir = output_dir or '/app/seq2seq/data/inputs/'
        # サンプルデータセットの定義
        self._url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
        self._urls = (f'{self._mode}.de.gz', f'{self._mode}.en.gz')
        if self._mode == "test":
            self._urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')
        # tokenizer
        self._de_tokenizer = get_tokenizer('spacy', language='de')
        self._en_tokenizer = get_tokenizer('spacy', language='en')
        # file paths
        self._filepaths: list = self._download_data()
        # vocab
        self._default_vocab = '<unk>'
        self._specials = ['<unk>', '<pad>', '<bos>', '<eos>']
        self._de_vocab = self._build_vocab(self._filepaths[0], self._de_tokenizer)
        self._en_vocab = self._build_vocab(self._filepaths[1], self._en_tokenizer)
        # data
        self._data = self._data_process(self._filepaths)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    @property
    def data(self) -> torch.Tensor:
        """dataset"""
        return self._data

    @property
    def bos_idx(self) -> int:
        """BOSのインデックス"""
        return self._de_vocab['<bos>']

    @property
    def eos_idx(self) -> int:
        """EOSのインデックス"""
        return self._de_vocab['<eos>']

    @property
    def pad_idx(self) -> int:
        """PADのインデックス"""
        return self._de_vocab['<pad>']

    @property
    def unk_idx(self) -> int:
        """PADのインデックス"""
        return self._de_vocab['<unk>']

    @property
    def n_de_vocab(self) -> int:
        """ドイツ語の単語数"""
        return len(self._de_vocab)

    @property
    def n_en_vocab(self) -> int:
        """英語の単語数"""
        return len(self._en_vocab)

    @property
    def de_vocab(self) -> list:
        """ドイツ語の単語一覧"""
        return self._de_vocab.get_itos()

    @property
    def en_vocab(self) -> list:
        """英語の単語一覧"""
        return self._en_vocab.get_itos()

    def _download_data(self) -> list[str]:
        """データセットをダウンロード"""
        return [
            extract_archive(
                download_from_url(
                    self._url_base + url,
                    path=os.path.join(self._output_dir + url)
                    ))[0]
            for url in self._urls
            ]

    def _tokens(self, filepath: str, tokenizer):
        """行毎のtokensを返すイテレータ"""
        with io.open(filepath, encoding="utf8") as file:
            for row in file:
                yield tokenizer(row)

    def _build_vocab(self, filepath: str, tokenizer):
        """単語のデータセットを作成"""
        vocab = build_vocab_from_iterator(
            self._tokens(filepath, tokenizer),
            specials=self._specials,
            )
        # 未知語はunknownトークンのインデックスとして設定する
        vocab.set_default_index(vocab[self._default_vocab])
        return vocab

    def _data_process(self, filepaths: str) -> list[tuple[torch.tensor, torch.tensor]]:
        """dataset"""
        raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))  # pylint: disable=consider-using-with
        raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))  # pylint: disable=consider-using-with
        data = []
        logger.info("data_process start.")
        for i, (raw_de, raw_en) in enumerate(zip(raw_de_iter, raw_en_iter), 1):
            # 全量ではなく動作テスト用に一部を対象にする。
            if i > 10:
                break
            de_tensor_ = torch.tensor([self._de_vocab[token] for token in self._de_tokenizer(raw_de)],
                                      dtype=torch.long)
            en_tensor_ = torch.tensor([self._en_vocab[token] for token in self._en_tokenizer(raw_en)],
                                      dtype=torch.long)
            data.append((de_tensor_, en_tensor_))
        return data
