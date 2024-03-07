import torch
import numpy as np
from typing import List, Optional, Tuple
from pythomata import SimpleDFA

from ..config import DataSegmentConfig
from .utils import DataSegment

import numpy as np

class RegBenchConfig(DataSegmentConfig):
    name: str="regbench"

    def build(self, seed: int) -> DataSegment:
        return regbench(**self.model_dump(), seed=seed)

def dataset2tensor(dataset, seq_len, char2id, ret_dfa=False):
    # input: list of string where each string might contain example separator
    # output: list of string where each string might contain pad token (example separator is replaced with pad token)
    examples = dataset.data

    def tokenize(inp_str, type="input"):
        assert type in ["input", "output"]
        ret = []
        for char in inp_str.split(" "):
            if char in char2id:
                ret.append(char2id[char])
            else:
                assert char == "<unk>" and type == "output"
                ret.append(-100)

        # pad to seq_len, pad_token is 0 for input and -100 for output
        if type == "input":
            pad_token = 0
        else:
            pad_token = -100
        ret += [pad_token] * (seq_len - len(ret))
        return ret

    inputs, labels = [], []
    dfas = []
    for example in examples:
        inp, out, dfa = example
        inputs.append(tokenize(inp, "input"))
        labels.append(tokenize(out, "output"))
        dfas.append(dfa)

    if ret_dfa:
        return torch.LongTensor(inputs), torch.LongTensor(labels), dfas
    else: 
        return torch.LongTensor(inputs), torch.LongTensor(labels)


def regbench(
        split: str="train",
        vocab_size: int=18, 
        input_seq_len: int=512,
        num_examples: int=1000,
        seed: int=42,
        eval_flag: bool=False,
        **kwargs,
    ):
    """
    Args:
        eval_flag: if True, return all the meta information needed for DFA evaluation
    """
    regbench = RegBench(split, num_examples=num_examples, vocab_size=vocab_size, seed=seed, max_input_seq_len=input_seq_len)

    vocab = regbench.vocab
    vocab = ["<unk>"] + vocab + ["|"] # the first token is the pad token, | is the example separator
    char2id = {c: i for i, c in enumerate(vocab)}

    if eval_flag:
        inputs, labels, dfas = dataset2tensor(regbench, input_seq_len, char2id, ret_dfa=True)
        return inputs, vocab, dfas
    else:
        inputs, labels = dataset2tensor(regbench, input_seq_len, char2id, ret_dfa=False)
        return DataSegment(inputs, labels)

class DFA:
    """Represents a DFA"""

    def __init__(
        self,
        num_nodes: int,
        alphabet: Tuple[str],
        transitions: Tuple[dict],
        rng: np.random.Generator,
    ):
        assert len(transitions) == num_nodes
        transitions = {i: v for i, v in enumerate(transitions)}
        dfa = SimpleDFA(
            states=set(list(range(num_nodes))),
            alphabet=set(alphabet),
            initial_state=0,
            accepting_states=set(list(range(num_nodes))),
            transition_function=transitions,
        )
        self.dfa = dfa
        self.rng = rng

    @classmethod
    def from_pythomata_dfa(cls, dfa: SimpleDFA, rng: Optional[np.random.Generator] = None):
        if rng is None:
            rng = np.random.default_rng()

        obj = cls(
            num_nodes=1,
            alphabet=(""),
            transitions=[{}],
            rng=rng,
        )

        obj.dfa = dfa
        return obj

    def __repr__(self) -> str:
        return f"""DFA.from_pythomata_dfa(pythomata.SimpleDFA(
            states={self.dfa._states},
            alphabet={self.dfa._alphabet.symbols},
            initial_state={self.dfa._initial_state},
            accepting_states={self.dfa._accepting_states},
            transition_function={self.dfa._transition_function},
        ))"""


    def _sorted_transitions(self):
        nodes = sorted(list(self.dfa._transition_function.keys()))
        transitions = []
        for node in nodes:
            node_transitions = self.dfa._transition_function[node]
            # sort node transitions by outgoing state
            transitions.append(
                tuple(sorted(node_transitions.items(), key=lambda item: item[1]))
            )
        return tuple(transitions)

    def minimize(self):
        # minimize super
        self.dfa = self.dfa.minimize()
        return self

    def trim(self):
        # trim super
        self.dfa = self.dfa.trim()
        return self

    def __hash__(self):
        # Here I assume the initial state is always the smallest node
        return hash(self._sorted_transitions())

    def __call__(self, word: List[str]):
        current_node = self.dfa._initial_state
        for symbol in word:
            if symbol not in self.dfa._transition_function[current_node]:
                return False
            else:
                current_node = self.dfa._transition_function[current_node][symbol]
        return True

    def forward(self, word: List[str]):
        current_node = self.dfa._initial_state
        for symbol in word:
            if symbol not in self.dfa._transition_function[current_node]:
                return None
            else:
                current_node = self.dfa._transition_function[current_node][symbol]
        return current_node

    def trace(self, word: List[str]):
        current_node = self.dfa._initial_state
        path = [current_node]
        for symbol in word:
            try:
                self.dfa._transition_function[current_node]
            except:
                ValueError("Invalid node in transition function")

            if symbol not in self.dfa._transition_function[current_node]:
                return path
            else:
                current_node = self.dfa._transition_function[current_node][symbol]
                path.append(current_node)
        return path

    def sample(self, length: int = 1):
        """Samples a random word from the DFA"""
        current_node = self.dfa._initial_state
        word = []
        for _ in range(length):
            outgoing_symbols = list(self.dfa._transition_function[current_node].keys())
            symbol = self.rng.choice(outgoing_symbols)
            word.append(symbol)
            current_node = self.dfa._transition_function[current_node][symbol]
        return word


class RandomDFASampler:
    """Samples random DFAs given configs"""

    num_nodes: int
    alphabet: Tuple[str]
    max_outgoing_edge: int
    rng: np.random.Generator = None

    def __init__(
        self,
        num_nodes: int,
        alphabet: Tuple[str],
        max_outgoing_edge: int,
        seed: int = 42,
    ):
        self.num_nodes = num_nodes
        self.alphabet = alphabet
        self.max_outgoing_edge = max_outgoing_edge
        self.rng = np.random.default_rng(seed)

    def sample(self):
        transitions = [{} for _ in range(self.num_nodes)]
        for node in range(self.num_nodes):
            num_transitions = self.rng.integers(1, self.max_outgoing_edge)
            transition_symbols = self.rng.choice(
                self.alphabet, size=num_transitions, replace=False
            )
            # exclude self loops
            possible_nodes = [n for n in range(self.num_nodes) if n != node]
            transition_nodes = self.rng.choice(
                possible_nodes, size=num_transitions, replace=False
            )
            transitions[node] = dict(zip(transition_symbols, transition_nodes))
        dfa_rng = np.random.default_rng(self.rng.integers(0, 2**32))
        return DFA(self.num_nodes, self.alphabet, tuple(transitions), dfa_rng)

class RegBench:
    def __init__(
        self,
        split: str = "val",
        num_examples: int = 5000,
        vocab_size: int = 18,
        max_num_nodes: int = 12,
        max_num_in_context_examples: int = 20,
        min_num_in_context_examples: int = 10,
        max_outgoing_edges: int = 4,
        max_len_per_example: int = 50,
        seed: int = 0,
        example_seperator: str = "|",
        token_seperator: str = " ",
        pad_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        alphabet_type: str = "chr", # "num" or "chr"
        max_input_seq_len: Optional[int] = None,  # truncates examples to this length
    ):
        self.num_examples = num_examples
        self.vocab_size = vocab_size
        self.seed = seed

        # special tokens
        self.example_seperator = example_seperator
        self.token_seperator = token_seperator
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        # end

        self.alphabet_type = alphabet_type
        self.vocab = self.get_vocab()

        self.max_num_nodes = max_num_nodes
        self.max_num_in_context_examples = max_num_in_context_examples
        self.min_num_in_context_examples = min_num_in_context_examples
        self.max_outgoing_edges = max_outgoing_edges
        self.max_len_per_example = max_len_per_example
        self.max_input_seq_len = max_input_seq_len

        self.data = self.generate(split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


    def get_special_tokens(self):
        return {
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "example_seperator": self.example_seperator,
            "token_seperator": self.token_seperator,
        }

    def get_vocab(self) -> List[str]:
        special_tokens = list(self.get_special_tokens().values())

        if self.alphabet_type == "num":
            self.vocab = [str(i) for i in range(self.vocab_size)]
        elif self.alphabet_type == "chr":
            self.vocab = []
            i = 97
            while len(self.vocab) < self.vocab_size:
                if chr(i) not in special_tokens:
                    self.vocab.append(chr(i))
                i += 1
            self.vocab = sorted(list(set(self.vocab)))

        assert len(self.vocab) == self.vocab_size
        return self.vocab

    def get_tokenizer_info(self):
        return {
            "vocabulary": self.vocab,
            "special_tokens": self.get_special_tokens(),
        }

    def generate(self, split="train") -> List[Tuple[str, str, DFA]]:
        seed = self.seed + abs(hash(split))

        self.rng = np.random.default_rng(seed)

        DFAs = set([])

        while len(DFAs) < self.num_examples:
            num_nodes = self.rng.integers(
                self.max_outgoing_edges, self.max_num_nodes + 1
            )
            num_alphabet = self.rng.integers(
                self.max_outgoing_edges, self.vocab_size + 1
            )
            alphabet = self.rng.choice(
                self.vocab_size, size=num_alphabet, replace=False
            )
            alphabet = tuple([self.vocab[i] for i in alphabet])
            sampler = RandomDFASampler(
                num_nodes,
                alphabet,
                self.max_outgoing_edges,
            )
            sampler.rng = np.random.default_rng(self.rng.integers(0, 2**32))
            dfa = sampler.sample()
            dfa.minimize().trim()
            DFAs.add(dfa)

        DFAs = list(DFAs)

        self.rng.shuffle(DFAs)

        data = []

        for dfa in DFAs:
            num_samples = self.rng.integers(
                self.min_num_in_context_examples,
                self.max_num_in_context_examples,
            )
            inp, out = self.generate_example(dfa, num_samples)

            data.append((inp, out, dfa))
        return data

    def generate_example(self, dfa: DFA, num_examples: int) -> Tuple[str, str]:
        sample = []

        for _ in range(num_examples):
            length = self.rng.integers(1, self.max_len_per_example)
            word = dfa.sample(length=length)
            sample += word
            sample += [self.example_seperator]

        sample = sample[:-1]

        if self.max_input_seq_len is not None and len(sample) > self.max_input_seq_len:
            sample = sample[: self.max_input_seq_len]

        inp = sample[:-1]
        out = sample[1:]
        # replace the example seperator with the pad token for output
        out = [self.pad_token if token == self.example_seperator else token for token in out]
        inp = self.encode(inp)
        out = self.encode(out)
        return inp, out

    def encode(self, example: List[str]) -> str:
        return self.token_seperator.join(example)

    def decode(self, example: str) -> List[str]:
        return example.split(self.token_seperator)

