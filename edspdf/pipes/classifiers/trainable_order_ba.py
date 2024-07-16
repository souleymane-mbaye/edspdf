import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Set

import torch
import torch.nn.functional as F
from foldedtensor import as_folded_tensor
from tqdm import tqdm

from edspdf.layers.vocabulary import Vocabulary
from edspdf.pipeline import Pipeline
from edspdf.pipes.embeddings import EmbeddingOutput
from edspdf.registry import registry
from edspdf.structures import PDFDoc
from edspdf.trainable_pipe import TrainablePipe


@registry.factory.register("order-ba-trainable-classifier")
class TrainableClassifier(TrainablePipe[Dict[str, Any]]):
    """
    This component predicts a label for each box over the whole document using machine
    learning.

    !!! note

        You must train the model your model to use this classifier.
        See [Model training][training-a-pipeline] for more information

    Examples
    --------

    The classifier is composed of the following blocks:

    - a configurable embedding layer
    - a linear classification layer

    In this example, we use a simple CNN-based embedding layer (`sub-box-cnn-pooler`),
    which applies a stack of CNN layers to the embeddings computed by a text embedding
    layer (`simple-text-embedding`).

    === "API-based"

        ```python
        pipeline.add_pipe(
            "trainable-classifier",
            name="classifier",
            config={
                # simple embedding computed by pooling embeddings of words in each box
                "embedding": {
                    "@factory": "sub-box-cnn-pooler",
                    "out_channels": 64,
                    "kernel_sizes": (3, 4, 5),
                    "embedding": {
                        "@factory": "simple-text-embedding",
                        "size": 72,
                    },
                },
                "labels": ["body", "pollution"],
            },
        )
        ```

    === "Configuration-based"

        ```toml
        [components.classifier]
        @factory = "trainable-classifier"
        labels = ["body", "pollution"]

        [components.classifier.embedding]
        @factory = "sub-box-cnn-pooler"
        out_channels = 64
        kernel_sizes = (3, 4, 5)

        [components.classifier.embedding.embedding]
        @factory = "simple-text-embedding"
        size = 72
        ```

    Parameters
    ----------
    labels: Sequence[str]
        Initial labels of the classifier (will be completed during initialization)
    embedding: TrainablePipe[EmbeddingOutput]
        Embedding module to encode the PDF boxes
    """

    def __init__(
        self,
        embedding: TrainablePipe[EmbeddingOutput],
        labels: Sequence[str] = ("pollution",),
        pipeline: Pipeline = None,
        name: str = "order-ba-trainable-classifier",
    ):
        super().__init__(pipeline, name)
        self.label_voc: Vocabulary = Vocabulary(list(dict.fromkeys(labels)))
        self.embedding = embedding

        size = self.embedding.output_size

        # no need of classifier
        self.classifier = torch.nn.Linear(
            in_features=self.embedding.output_size,
            
            
            
            # out_features=len(self.label_voc),
            out_features=1, # bilabel, comes before or after
        )

    def post_init(self, gold_data: Iterable[PDFDoc], exclude: set):
        if self.name in exclude:
            return
        exclude.add(self.name)
        self.embedding.post_init(gold_data, exclude)

        label_voc_indices = dict(self.label_voc.indices)

        with self.label_voc.initialization():
            for doc in tqdm(gold_data, desc="Initializing classifier"):
                self.preprocess_supervised(doc)

        self.update_weights_from_vocab_(label_voc_indices)

    def update_weights_from_vocab_(self, label_voc_indices):
        label_indices = dict(
            (
                *label_voc_indices.items(),
                *self.label_voc.indices.items(),
            )
        )
        old_index = [label_voc_indices[label] for label in label_voc_indices]
        new_index = [label_indices[label] for label in label_voc_indices]
        new_classifier = torch.nn.Linear(
            self.embedding.output_size,
            #len(label_indices),
            1,
        )
        new_classifier.weight.data[new_index] = self.classifier.weight.data[old_index]
        new_classifier.bias.data[new_index] = self.classifier.bias.data[old_index]
        self.classifier = new_classifier

    def preprocess(self, doc: PDFDoc) -> Dict[str, Any]:
        result = {
            "embedding": self.embedding.preprocess(doc),
            "doc_id": doc.id,
        }
        return result

    
    
    
    
    
    
    
    
    
    
    #########
    #########
    def preprocess_supervised(self, doc: PDFDoc) -> Dict[str, Any]:
        #
        _sep = '$+$'
        # label = f'{src_b.label}{_sep}{src_b.node_num}{_sep}{src_b.rank}'
        # b.label.split(_sep)[0] # label
        # b.label.split(_sep)[1] # node_num
        # b.label.split(_sep)[2] # rank
        #
        followings = []
        followings_mask = []
        for page in doc.pages:
            l_page = []
            l_page_mask = []
            for bi in page.text_boxes:
                bi_node_type        = bi.label.split(_sep)[0]
                bi_node_num         = int(bi.label.split(_sep)[1])
                bi_rank_inside_node = int(bi.label.split(_sep)[2])
                
                # followings of line bi
                l_line = []
                for bj in page.text_boxes:
                    bj_node_type        = bj.label.split(_sep)[0]
                    bj_node_num         = int(bj.label.split(_sep)[1])
                    bj_rank_inside_node = int(bj.label.split(_sep)[2])
                    
                    if (bi_node_type != bj_node_type or bi_node_type=='pollution' or bj_node_type=='pollution'):
                        l_line.append(0.0)
                    elif bi_node_num < bj_node_num:
                        l_line.append(1.0)
                    elif (bi_node_num == bj_node_num) \
                                 and (bi_rank_inside_node < bj_rank_inside_node):
                        l_line.append(1.0)
                    else:
                        l_line.append(0.0)
                l_page.append(l_line)
                
                # mask of line bi
                l_line_mask = []
                for bj in page.text_boxes:
                    bj_node_type        = bj.label.split(_sep)[0]
                    if (bi_node_type != bj_node_type or bi_node_type=='pollution' or bj_node_type=='pollution'):
                        l_line_mask.append(True)
                    else:
                        l_line_mask.append(False)
                l_page_mask.append(l_line_mask)
            followings.append(l_page)
            followings_mask.append(l_page_mask)
        
        return {
            **self.preprocess(doc),
            # in a page, wether line bi comes before(1.0) or not(0.0) line bj
            "followings": followings,
            # mask we compare lines of the same node_type different than pollution
            "followings_mask": followings_mask,
        }

    def collate(self, batch) -> Dict:
        collated = {
            "embedding": self.embedding.collate(batch["embedding"]),
            "doc_id": batch["doc_id"],
        }
        if "followings" in batch:
            collated.update(
                {
                    "followings": as_folded_tensor(
                        batch["followings"],
                        data_dims=("line",),
                        full_names=("sample", "page", "line"),
                        dtype=torch.float,  # <- float
                    ),  # n_lines * n_lines
                    "followings_mask": as_folded_tensor(
                        batch["followings_mask"],
                        data_dims=("line",),
                        full_names=("sample", "page", "line"),
                        dtype=torch.bool,  # <- bool mask
                    ),  # n_lines * n_lines
                }
            )

        return collated

    def forward(self, batch: Dict) -> Dict:
        embedding_res = self.embedding.module_forward(batch["embedding"])
        embeddings = embedding_res["embeddings"]

        output = {"loss": 0, "mask": embeddings.mask}

        # dot product between lines --> out n_lines * n_lines
        e = embeddings.to(self.classifier.weight.dtype).refold("page", "line")
        e_transpose = e.transpose(1, 2)
        
        # print(f'\nembeddings shape: {embeddings.to(self.classifier.weight.dtype).shape}, \
        # \nbatch keys: {batch.keys()}')
        # print(f'\ne shape: {e.shape} e_transpose shape: {e_transpose.shape}')
        
        logits = F.sigmoid(torch.matmul(e, e_transpose))  # n_pages * n_lines * n_lines
        
        # print(f'\nlogits shape: {logits.shape}')
        
        if "labels" in batch:
            targets = batch["labels"].refold(logits.data_dims)
            # apply mask on logits
            logits_masked = logits.masked_fill(batch["labels_mask"].refold(logits.data_dims), 0.5)
            
            # print(f'calcul loss, labels: len({len(batch["labels"][0])}, {len(batch["labels"][0][0])})')
            # print(f'calcul loss, labels mask: len({len(batch["labels_mask"][0])}, \
            #     {len(batch["labels_mask"][0][0])})')
            # print(f'  {batch["labels_mask"][0][0]}')
            
            # print(f'type logits: {logits.dtype} -- targets type: {targets.dtype}')
            output["label_loss"] = (
                # sig + BCELoss
                F.binary_cross_entropy_with_logits(
                    logits_masked,
                    targets,
                    reduction="sum",
                )
                / targets.mask.sum()
                
                # F.cross_entropy(
                #     logits,
                #     targets,
                #     reduction="sum",
                # )
                # / targets.mask.sum()
            )
            output["loss"] = output["loss"] + output["label_loss"]
        else:
            logits = logits.refold("line")
            output["logits"] = logits
            
            
            # output["labels"] = logits.argmax(-1)
            output["labels"] = logits > 0.5
            output["labels_mask"] = logits == 0.5
            

        return output

    def postprocess(self, docs: Sequence[PDFDoc], output: Dict) -> Sequence[PDFDoc]:
        #
        _sep = '$+$'
        # label = f'{src_b.label}{_sep}{src_b.node_num}{_sep}{src_b.rank}'
        # b.label.split(_sep)[0] # label
        # b.label.split(_sep)[1] # node_num
        # b.label.split(_sep)[2] # rank
        #
        # ok = True
        # print(f'\nPost process, logits lengths: {output["logits"].lengths}')
        print(f'\nPost process: lens ({len(output["logits"].tolist())},'+ \
         f'{len(output["logits"].lengths)}, {len(output["labels_mask"].tolist())})')
        
        [n_pages, n_lines] = output["logits"].lengths[-2:], # ft.lengths == [[2], [5, 1], [1, 0, 0, 0, 2, 2]]
        page, line = 0, 0
        n_lines_page_max = max(n_lines[line : n_pages[page]])
        
        for b, line_logits, n_lines, line_mask in zip(
            (b for doc in docs for b in doc.text_boxes),
            output["logits"].tolist(),
            output["labels_mask"].tolist(),
        ):
            # if ok:
            #     ok = False
            #     print(f'Post process 2:\n  len({len(line_logits[:n_lines])}) \
            #     line logits: {line_logits[:n_lines]}\n  \
            #     line mask: ({len(line_mask[:n_lines])}) {line_mask[:n_lines]}')
            # n_lines per page variable
            
            _l = [
                ('M' if l==0.5 else ('B' if l>0.5 else 'A'))
                for l in line_logits[:n_lines_page_max]
            ]
            b.label = _sep.join(_l)
            
            line += 1
            n_pages[page] -= 1
            if n_pages[page] == 0:
                page += 1
                n_lines_page_max = max(n_lines[line:n_pages[page]])
        return docs

    def to_disk(self, path: Path, exclude: Set):
        if self.name in exclude:
            return
        exclude.add(self.name)

        os.makedirs(path, exist_ok=True)

        with (path / "label_voc.json").open("w") as f:
            json.dump(self.label_voc.indices, f)

        return super().to_disk(path, exclude)

    def from_disk(self, path: Path, exclude: Set):
        if self.name in exclude:
            return
        exclude.add(self.name)

        label_voc_indices = dict(self.label_voc.indices)

        with (path / "label_voc.json").open("r") as f:
            self.label_voc.indices = json.load(f)

        self.update_weights_from_vocab_(label_voc_indices)

        super().from_disk(path, exclude)
