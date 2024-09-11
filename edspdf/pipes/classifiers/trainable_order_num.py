import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Set

import numpy as np
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


@registry.factory.register("order-num-trainable-classifier")
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
        name: str = "order-num-trainable-classifier",
    ):
        super().__init__(pipeline, name)
        self.label_voc: Vocabulary = Vocabulary(list(dict.fromkeys(labels)))
        self.embedding = embedding

        size = self.embedding.output_size
        
        # MLP with 2 fc 18 -> embedding.output_size / 2 -> 1
        dim_1 = int(self.embedding.output_size / 2)
        dim_2 = int(dim_1 / 2)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.embedding.output_size, dim_1),
            #torch.nn.BatchNorm(dim_1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(dim_1, dim_2),
            #torch.nn.BatchNorm(dim_2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(dim_2, 1),
        )
        # score(line i, line j) = dot(fc i, fc j) + mlp

    def post_init(self, gold_data: Iterable[PDFDoc], exclude: set):
        if self.name in exclude:
            return
        exclude.add(self.name)
        self.embedding.post_init(gold_data, exclude)

        label_voc_indices = dict(self.label_voc.indices)

        with self.label_voc.initialization():
            for doc in tqdm(gold_data, desc="Initializing classifier"):
                self.preprocess_supervised(doc)

        # self.update_weights_from_vocab_(label_voc_indices)

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
        followings = self.encode_followings(doc)
        
        result = {
            "embedding": self.embedding.preprocess(doc),
            "doc_id": doc.id,
            # mask we compare lines of the same node_type different than pollution
            "scores_mask": [
                [
                    # bi.coun('A')/len(bi) if len(bi)>0 else 0.0
                    [True if len(bi)>0 else False] # mask if line is not a body line
                    for bi in page
                ]
                for page in followings
            ],
        }
        return result
        
    
    def encode_followings(self, doc: PDFDoc):
        #
        _sep = '|'
        # label = f'{src_b.label}{_sep}{src_b.node_num}{_sep}{src_b.rank}'
        # b.label.split(_sep)[0] # label
        # b.label.split(_sep)[1] # node_num
        # b.label.split(_sep)[2] # rank
        #
        followings = []
        for page in doc.pages:
            cpt = 0
            l_page = []
            for i, bi in enumerate(page.text_boxes):
                spl = bi.label.split(_sep)
                if len(spl) < 3:
                    print(f'  Label format error: `{bi.label}` {bi}')
                bi_node_type, bi_node_num, bi_rank_inside_node = spl[0], int(spl[1]), int(spl[2])

                # followings of line bi
                l_line = []
                if bi_node_type!='body':  # pollution node masked
                    #if cpt < 10:
                    #    print('NOT BODY', bi.label, bi.text)
                    l_line = [] #['M' for bj in page.text_boxes]
                else:
                    cpt += 1
                    for bj in page.text_boxes:
                        spl = bj.label.split(_sep)
                        if len(spl) < 3:
                            print(f'  Label format error: `{bj.label}` {bj}')
                        bj_node_type, bj_node_num, bj_rank_inside_node = spl[0], int(spl[1]), int(spl[2])

                        # bj is a pollution node or not the same type with bi
                        if (bj_node_type!='body' or bi_node_type != bj_node_type):
                            l_line.append('M')
                        else: # bi and bj are both not pollution nodes and are of the same type of nodes
                            if bi_node_num < bj_node_num:
                                l_line.append('B') # bi comes before bj
                            else:
                                if (bi_node_num == bj_node_num):
                                    if bi_rank_inside_node == bj_rank_inside_node:
                                        l_line.append('S')  # bi same as bj
                                    elif bi_rank_inside_node + 1 == bj_rank_inside_node:
                                        l_line.append('F')  # bi followed by bj
                                    elif bi_rank_inside_node + 1 < bj_rank_inside_node:
                                        l_line.append('B')
                                    else: # bi_rank_inside_node > bj_rank_inside_node
                                        l_line.append('A')
                                else: # bi_node_num > bj_node_num
                                    l_line.append('A')
                    # last line of a type follows itself
                    #if 'F' not in l_line:
                    #    if 'B' not in l_line: # last rank of the last node
                    #        l_line[i] = 'F'
                    #        ind_f = len(page.text_boxes) # INF (rien en comm)
                    #    else: # last rank inside node
                    #        ind_f = len(page.text_boxes) # INF
                    #        node_min = len(page.text_boxes) # INF
                    #        rank_min = len(page.text_boxes) # INF
                    #        bj2 = bi
                    #        for j, bj in enumerate(page.text_boxes):
                    #            spl = bj.label.split(_sep)
                    #            bj_node_type, bj_node_num, bj_rank_inside_node = spl[0], int(spl[1]), int(spl[2])
                    #            if (bi_node_type == bj_node_type) and (bi_node_num < bj_node_num) \
                    #              and ((bj_node_num < node_min) \
                    #                   or (bj_node_num==node_min and bj_rank_inside_node<rank_min)):
                    #                ind_f = j
                    #                node_min = bj_node_num
                    #                rank_min = bj_rank_inside_node
                    #                bj2 = bj
                    #                if bi_node_num+1==bj_node_num and rank_min == 0: break
                    #        if ind_f < len(page.text_boxes):
                    #            l_line[ind_f] = 'F'
                l_page.append(l_line)
            followings.append(l_page)
            #print("CPT", cpt)
        return followings
    
    #########
    #########
    def preprocess_supervised(self, doc: PDFDoc) -> Dict[str, Any]:
        followings = self.encode_followings(doc)
        return {
            **self.preprocess(doc),
            # in a page, wether line bi comes before(1.0) or not(0.0) line bj
            "scores": [
                [
                    [torch.tensor(bi.count('A') / (len(bi) - bi.count('M'))) if len(bi)>0 else 0.0]
                    for bi in page
                ]
                for page in followings
            ],
        }

    def collate(self, batch) -> Dict:
        collated = {
            "embedding": self.embedding.collate(batch["embedding"]),
            "doc_id": batch["doc_id"],
            # mask we compare lines of the same node_type different than pollution
            "scores_mask": as_folded_tensor(
                batch["scores_mask"],
                data_dims=("page", "line"),
                full_names=("sample", "page", "line"),
                dtype=torch.bool,  # <- bool mask
            ),  # n_lines * n_lines
        }
        if "scores" in batch:
            
            collated.update(
                {
                    "scores": as_folded_tensor(
                        batch["scores"],
                        data_dims=("page", "line"),
                        full_names=("sample", "page", "line"),
                        dtype=torch.float,  # <- float
                    ),  # n_lines * n_lines
                }
            )

        return collated

    def forward(self, batch: Dict) -> Dict:
        embedding_res = self.embedding.module_forward(batch["embedding"])
        embeddings = embedding_res["embeddings"]
        #print(f'\nEMBEDDINGS shape: {embeddings.shape} refold(page,line) {embeddings.refold("page", "line").shape} refold(line) {embeddings.refold("line").shape}')
        
        logits = self.mlp(embeddings.to(self.mlp[0].weight.dtype)).refold("line")
        #print(f'\nLOGITS shape {logits.shape} SCORES MASK shape {batch["scores_mask"].refold("line").shape}')
        # or SIGMOID score of a line in [0;1]
        logits = F.sigmoid(logits)  # n_pages * n_lines * n_lines
        logits = logits.masked_fill(~batch["scores_mask"].refold("line"), 0.0)
        
        batch["scores_mask"] = batch["scores_mask"].refold("line")

        output = {"loss": 0, "mask": embeddings.mask}
        
        
        if "scores" in batch:
            #print("--  SCORES shape", batch["scores"].shape)
            targets = batch["scores"].refold("page", "line")
            print("\n-- TARGETS shape", targets.shape, "MASK shape", batch["scores_mask"].refold("line").shape)
            logits = logits.refold('page', 'line')
            batch["scores_mask"] = batch["scores_mask"].refold("page", "line")
            print('Train', logits.shape, batch["scores_mask"].shape, batch["scores_mask"].lengths)
            for i in range(10):
                print(i, batch["scores_mask"][0,i].item(), logits[0,i].item(), targets[0,i].item(), torch.abs(logits[0,i]-targets[0,i]).item())
            logits = logits.refold('line')
            targets = batch["scores"].refold("line")
            
            
            # SIGMOID
            individual_losses = F.mse_loss(
                # sig + BCELoss
                logits,
                targets,
                reduction="none",
            )
            
            individual_losses = individual_losses.masked_fill(~batch["scores_mask"].refold("line"), 0.0)
            N = batch["scores_mask"].sum()
            N = N if N>0 else 1
            output["scores_loss"] = (
                individual_losses.sum() / N
            )
            print('scores loss', output["scores_loss"], 'N', N.item(), 'ind loss sum', individual_losses.sum().item(), 'mask shape', batch["scores_mask"].shape)
            assert not torch.isnan(output["scores_loss"]).item(), "--------- NaN encountered during loss computation"
            
            output["loss"] = output["loss"] + output["scores_loss"]
        else:
            logits = logits.refold('page', 'line').squeeze(-1)
            arg_sort = logits.argsort(-1)
            logits_sorted = torch.gather(logits, dim=1, index=arg_sort)
            batch["scores_mask"] = batch["scores_mask"].refold("page", "line").squeeze(-1)
            print('\nTest', logits.shape, batch["scores_mask"].shape, batch["scores_mask"].lengths)
            
            followings = []
            lengths = batch["scores_mask"].lengths[-1]
            for i in range(batch["scores_mask"].shape[0]):
                _sum = batch["scores_mask"][i].sum()
                if i==0:
                    print("Page", i, _sum)
                p_followings = [None] * lengths[i]
                for j in range(batch["scores_mask"].shape[1]):#range(lengths[i]):
                    ar = arg_sort[i,j]
                    ar_ar = (arg_sort[i] == ar).nonzero(as_tuple=True)[0]
                    ar_suiv = 1+ar_ar if ar_ar < (arg_sort.shape[1]-1) else ar_ar
                    suiv = arg_sort[i,ar_suiv]
                    if ar < lengths[i]:
                        p_followings[ar] = suiv
                    if (i==0) and (j < 10) and batch["scores_mask"][i,ar].item():
                        #print("  ", j, 'ar', ar, 'suiv', suiv, 'ar_suiv', ar_suiv, batch["scores_mask"][i,ar].item()," \t", logits[i,ar].item(), logits_sorted[i,j].item())
                        print("  ", j, 'line', ar.item(), 'suiv', suiv.item(), logits[i,ar].item(), logits[i,suiv].item())
                followings.append(p_followings)
                #print('FOLLOWINGS shape', len(p_followings))
            
            followings = as_folded_tensor(
                followings,
                data_dims=("page", "line"),
                full_names=("page", "line"),
                dtype=torch.int,  # <- bool mask
            )
            output["scores_mask"] = batch["scores_mask"].refold("line")
            output["followings"] = followings.refold('line')
            output["logits"] = logits.refold("line")
            output["mask"] = batch["scores_mask"].refold("line")
            

        return output

    def postprocess(self, docs: Sequence[PDFDoc], output: Dict) -> Sequence[PDFDoc]:
        #
        _sep = '|'
        # label = f'{src_b.label}{_sep}{src_b.node_num}{_sep}{src_b.rank}'
        # b.label.split(_sep)[0] # label
        # b.label.split(_sep)[1] # node_num
        # b.label.split(_sep)[2] # rank
        #
        for b, label in zip(
            (b for doc in docs for b in doc.text_boxes),
            output["followings"].tolist(),
        ):
            #print(f' post process: line {ind} label {b.label}|{label} logit: {logits[ind]}')
            #ind += 1
            # SOFTMAX
            l = b.label.split(_sep)[:3] + [str(label) if b.label.split(_sep)[0]=='body' else '-1']
            
            # SIGMOID
            # l = b.label.split(_sep)[:3] + [str(i) for i,o in enumerate(label) if o]
            b.label = _sep.join(l)
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



        # FOLLOWINGS MASK
        # _sep = '|'
        # label = f'{src_b.label}{_sep}{src_b.node_num}{_sep}{src_b.rank}'
        # b.label.split(_sep)[0] # label
        # b.label.split(_sep)[1] # node_num
        # b.label.split(_sep)[2] # rank
        #
        # followings_mask = []
        # for page in doc.pages:
        #     l_page = []
        #     for bi in page.text_boxes:
        #         if bi.x0==0.5969582850637 and bi.y0==0.41155346957120026:
        #             spl = bi.label.split(_sep)
        #             print(f'\n************\n  ({len(spl)}) {spl}  {bi}\n************\n')
        #         
        #         spl = bi.label.split(_sep)
        #         if len(spl) < 3:
        #             print(f'  Label format error: `{bi.label}` {bi}')
        #         bi_node_type, bi_node_num, bi_rank_inside_node = spl[0], int(spl[1]), int(spl[2])
        #         
        #         # mask of line bi
        #         l_line_mask = []
        #         for bj in page.text_boxes:
        #             spl = bj.label.split(_sep)
        #             if len(spl) < 3:
        #                 print(f'  Label format error: `{bj.label}` {bj}')
        #             bj_node_type, bj_node_num, bj_rank_inside_node = spl[0], int(spl[1]), int(spl[2])
        #             
        #             if (bi_node_type != bj_node_type or bi_node_type=='pollution' or bj_node_type=='pollution'):
        #                 l_line_mask.append(False)
        #             else:
        #                 if bi_node_num==bj_node_num and bi_rank_inside_node==bj_rank_inside_node:
        #                     l_line_mask.append(False) # if bi and bj are same, mask bi can not follow bi
        #                 else:
        #                     l_line_mask.append(True)
        #         l_page.append(l_line_mask)
        #     followings_mask.append(l_page)
        


        # followings = []
        # for page in doc.pages:
        #     l_page = []
        #     for i, bi in enumerate(page.text_boxes):
        #         spl = bi.label.split(_sep)
        #         if len(spl) < 3:
        #             print(f'  Label format error: `{bi.label}` {bi}')
        #         bi_node_type, bi_node_num, bi_rank_inside_node = spl[0], int(spl[1]), int(spl[2])
        #         
        #         # followings of line bi
        #         l_line = []
        #         if bi_node_type=='pollution':  # pollution node masked
        #             l_line = [0.0 for bj in page.text_boxes] # 'M'
        #         else:
        #             for bj in page.text_boxes:
        #                 spl = bj.label.split(_sep)
        #                 if len(spl) < 3:
        #                     print(f'  Label format error: `{bj.label}` {bj}')
        #                 bj_node_type, bj_node_num, bj_rank_inside_node = spl[0], int(spl[1]), int(spl[2])

        #                 # bj is a pollution node or not the same type with bi
        #                 if (bj_node_type=='pollution' or bi_node_type != bj_node_type):
        #                     l_line.append(0.0)  # 'M'
        #                 else: # bi and bj are both not pollution nodes and are of the same type of nodes
        #                     if bi_node_num < bj_node_num:
        #                         l_line.append(0.0) # bi comes before bj ('B')
        #                     else:
        #                         if (bi_node_num == bj_node_num):
        #                             if bi_rank_inside_node == bj_rank_inside_node:
        #                                 l_line.append(0.0) # ('S')  # bi same as bj
        #                             elif bi_rank_inside_node + 1 == bj_rank_inside_node:
        #                                 l_line.append(1.0)  # bi followed by bj ('F')
        #                             elif bi_rank_inside_node + 1 < bj_rank_inside_node:
        #                                 # SOFTMAX line_i is followed by line_j
        #                                 l_line.append(0.0) # ('B')
        #                                 
        #                                 # SIGMOID line_i is before line_j
        #                                 # l_line.append(1.0) # ('B')
        #                             else: # bi_rank_inside_node >= bj_rank_inside_node
        #                                 l_line.append(0.0) # ('A')
        #                         else: # bi_node_num > bj_node_num
        #                             l_line.append(0.0) # ('A')
        #             # last line of a type follows itself
        #             if 1.0 not in l_line:
        #                 l_line[i] = 1.0 # 'F'
        #         l_page.append(l_line)
        #     followings.append(l_page)
