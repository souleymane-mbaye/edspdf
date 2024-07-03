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


@registry.factory.register("bl-trainable-classifier")
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
        name: str = "trainable-classifier",
    ):
        super().__init__(pipeline, name)
        self.label_voc: Vocabulary = Vocabulary(list(dict.fromkeys(labels)))
        self.embedding = embedding

        size = self.embedding.output_size

        self.classifier = torch.nn.Linear(
            in_features=self.embedding.output_size,
            
            
            
            # out_features=len(self.label_voc),
            out_features=2, # is_begin, is_last
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
            2,
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
        return {
            **self.preprocess(doc),
            "labels": [
                [
                    #self.label_voc.encode(b.label) if b.label is not None else -100
                    [
                        1.0 if b.label in ("B", "U") else 0.0,  # is at the beginning 
                        1.0 if b.label in ("L", "U") else 0.0,  # is at the end (last)
                    ]
                    for b in page.text_boxes
                ]
                for page in doc.pages
            ],
        }

    def collate(self, batch) -> Dict:
        collated = {
            "embedding": self.embedding.collate(batch["embedding"]),
            "doc_id": batch["doc_id"],
        }
        if "labels" in batch:
            collated.update(
                {
                    "labels": as_folded_tensor(
                        batch["labels"],
                        data_dims=("line",),
                        full_names=("sample", "page", "line"),
                        dtype=torch.float,  # <- float
                    ),  # n_lines * (n_labels if multi label else .)
                }
            )

        return collated

    def forward(self, batch: Dict) -> Dict:
        embedding_res = self.embedding.module_forward(batch["embedding"])
        embeddings = embedding_res["embeddings"]

        output = {"loss": 0, "mask": embeddings.mask}

        # Label prediction / learning
        logits = self.classifier(embeddings.to(self.classifier.weight.dtype)).refold(
            "line"
        )  # n_lines * n_labels
        #print(f'\nembeddings shape: {embeddings.to(self.classifier.weight.dtype).shape} \
        #--> first: {embeddings.to(self.classifier.weight.dtype)[0]}')
        #print(f'\nlogits shape: {logits.shape} --> first: {logits[0]}')
        
        if "labels" in batch:
            targets = batch["labels"].refold(logits.data_dims)
            #
            # print(f'type logits: {logits.dtype} -- targets type: {targets.dtype}')
            output["label_loss"] = (
                # sig + BCELoss
                F.binary_cross_entropy_with_logits(
                    logits,
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
            output["logits"] = logits
            
            
            # output["labels"] = logits.argmax(-1)
            output["labels"] = logits > 0.5
            

        return output

    def postprocess(self, docs: Sequence[PDFDoc], output: Dict) -> Sequence[PDFDoc]:
        for b, label in zip(
            (b for doc in docs for b in doc.text_boxes),
            output["labels"].tolist(),
        ):
            
            # b.label = self.label_voc.decode(label) if b.text != "" else None
            # print(f'\nPostProcess label {label}')
            b.label = 'U' if sum(label) == 2 else \
                        ('B' if label[0] is True else \
                        ('L' if label[1] is True else 'I'))
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
