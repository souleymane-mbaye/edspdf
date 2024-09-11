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

        # fully connected for line i
        self.fc_i = torch.nn.Linear(
            in_features=self.embedding.output_size,
            out_features=self.embedding.output_size,
        )
        # fully connected for line j
        self.fc_j = torch.nn.Linear(
            in_features=self.embedding.output_size,
            out_features=self.embedding.output_size,
        )
        # MLP with 2 fc 18 -> embedding.output_size / 2 -> 1
        dim_mlp = int(self.embedding.output_size / 2)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(18, dim_mlp),
            #torch.nn.BatchNorm(dim_mlp),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(dim_mlp, 1),
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
        
        regions = []
        for page in doc.pages:
            p_regions = []
            for bi in page.text_boxes:
                l_regions = []
                for bj in page.text_boxes:
                    _bi, _bj = (bi.x0,bi.y0,bi.x1,bi.y1), (bj.x0,bj.y0,bj.x1,bj.y1)
                    union_bi_bj = self.union_bounding_box(_bi, _bj)
                    r_bi_bj = self.delta(_bi, _bj) + self.delta(_bi, union_bi_bj) + self.delta(_bj, union_bi_bj)
                    l_regions.append(r_bi_bj)
                p_regions.append(l_regions)
            regions.append(p_regions)
        
        result = {
            "embedding": self.embedding.preprocess(doc),
            "doc_id": doc.id,
            # mask we compare lines of the same node_type different than pollution
            "followings_mask": [
                [
                    [
                        (True if bj in ['A', 'S', 'F', 'B'] else False) #, inverted, False for masked elements
                        for bj in bi
                    ]
                    for bi in page
                ]
                for page in followings
            ],
            "regions": regions,
        }
        return result

    
    
    
    
    
    
    
    
    def union_bounding_box(self, bi, bj):
        (bi_x0, bi_y0, bi_x1, bi_y1) = bi
        (bj_x0, bj_y0, bj_x1, bj_y1) = bj
        
        x0, y0 = min(bi_x0, bj_x0), min(bi_y0, bj_y0)
        x1, y1 = min(bi_x1, bj_x1), min(bi_y1, bj_y1)

        return (x0, y0, x1, y1)
    
    def delta(self, bi, bj):
        (bi_x0, bi_y0, bi_x1, bi_y1) = bi
        (bj_x0, bj_y0, bj_x1, bj_y1) = bj
        
        xi_centre, yi_centre = (bi_x1 - bi_x0) / 2, (bi_y1 - bi_y0) / 2
        wi, hi = (bi_x1 - bi_x0), (bi_y1 - bi_y0)
        xj_centre, yj_centre = (bj_x1 - bj_x0) / 2, (bj_y1 - bj_y0) / 2
        wj, hj = (bj_x1 - bj_x0), (bj_y1 - bj_y0)
        
        dij_x_ctr, dij_y_ctr = (xi_centre - xj_centre) / wi, (yi_centre - yj_centre) / hi
        dij_w, dij_h = np.log(wi/wj), np.log(hi/hj)
        dji_x_ctr, dji_y_ctr = (xj_centre - xi_centre) / wj, (yj_centre - yi_centre) / hj
        
        return [dij_x_ctr, dij_y_ctr, dij_w, dij_h, dji_x_ctr, dji_y_ctr]
        
    
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
            l_page = []
            for i, bi in enumerate(page.text_boxes):
                spl = bi.label.split(_sep)
                if len(spl) < 3:
                    print(f'  Label format error: `{bi.label}` {bi}')
                bi_node_type, bi_node_num, bi_rank_inside_node = spl[0], int(spl[1]), int(spl[2])

                # followings of line bi
                l_line = []
                if bi_node_type=='pollution':  # pollution node masked
                    l_line = ['M' for bj in page.text_boxes]
                else:
                    for bj in page.text_boxes:
                        spl = bj.label.split(_sep)
                        if len(spl) < 3:
                            print(f'  Label format error: `{bj.label}` {bj}')
                        bj_node_type, bj_node_num, bj_rank_inside_node = spl[0], int(spl[1]), int(spl[2])

                        # bj is a pollution node or not the same type with bi
                        if (bj_node_type=='pollution' or bi_node_type != bj_node_type):
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
                    if 'F' not in l_line:
                        if 'B' not in l_line: # last rank of the last node
                            l_line[i] = 'F'
                            ind_f = len(page.text_boxes) # INF (rien en comm)
                        else: # last rank inside node
                            ind_f = len(page.text_boxes) # INF
                            node_min = len(page.text_boxes) # INF
                            rank_min = len(page.text_boxes) # INF
                            bj2 = bi
                            for j, bj in enumerate(page.text_boxes):
                                spl = bj.label.split(_sep)
                                bj_node_type, bj_node_num, bj_rank_inside_node = spl[0], int(spl[1]), int(spl[2])
                                if (bi_node_type == bj_node_type) and (bi_node_num < bj_node_num) \
                                  and ((bj_node_num < node_min) \
                                       or (bj_node_num==node_min and bj_rank_inside_node<rank_min)):
                                    ind_f = j
                                    node_min = bj_node_num
                                    rank_min = bj_rank_inside_node
                                    bj2 = bj
                                    if bi_node_num+1==bj_node_num and rank_min == 0: break
                            if ind_f < len(page.text_boxes):
                                l_line[ind_f] = 'F'
                l_page.append(l_line)
            followings.append(l_page)
        return followings
    
    #########
    #########
    def preprocess_supervised(self, doc: PDFDoc) -> Dict[str, Any]:
        followings = self.encode_followings(doc)
        return {
            **self.preprocess(doc),
            # in a page, wether line bi comes before(1.0) or not(0.0) line bj
            "followings": [
                [
                    [
                        (1.0 if bj=='F' else 0.0) # SOFTMAX
                        # (1.0 if bj in ['B', 'F'] else 0.0) # SIGMOID
                        for bj in bi
                    ]
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
            "followings_mask": as_folded_tensor(
                batch["followings_mask"],
                data_dims=("page", "line"),
                full_names=("sample", "page", "line"),
                dtype=torch.bool,  # <- bool mask
            ),  # n_lines * n_lines
            "regions": as_folded_tensor(
                batch["regions"],
                data_dims=("page", "line"),
                full_names=("sample", "page", "line"),
                dtype=torch.float,  # <- float
            ),  # n_lines * n_lines
        }
        if "followings" in batch:
            
            collated.update(
                {
                    "followings": as_folded_tensor(
                        batch["followings"],
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
        #print(f'\nEMBEDDINGS shape: {embeddings.shape} refold {embeddings.refold("page", "line").shape}')
        #print("FOLLOWINGS MASK shape", batch["followings_mask"].shape)
        #print("REGIONS shape", batch["regions"].shape)
        
        # 24 pages * 190 lines * 768
        fc_i = self.fc_i(embeddings.to(self.fc_i.weight.dtype)).refold("page", "line")
        fc_j = self.fc_j(embeddings.to(self.fc_j.weight.dtype)).refold("page", "line")
        mlp  = self.mlp(batch["regions"]).squeeze(-1)
        ## mlp  = mlp.masked_fill(~batch["followings_mask"], 0.0)
        #print(f'\nFC I shape {fc_i.shape}')
        #print(f'FC J shape {fc_j.shape}')
        #print(f'MLP shape {mlp.shape}')

        output = {"loss": 0, "mask": embeddings.mask}

        # dot product between lines --> out n_lines * n_lines
        # TODO use einsum instead of matmul
        logits = torch.einsum('pid,pjd->pij', fc_i, fc_j)
        #print("LOGITS (einsum) shape", logits.shape)
        
        # SOFTMAX line_i followed by line_j
        # TODO apply mask
        #print("FOLLOWINGS MASK shape", batch["followings_mask"].shape)
        if logits.shape != mlp.shape:
            print("LOGITS (einsum) shape", logits.shape)
            print(f'MLP shape {mlp.shape}')
            logits = logits[:,:mlp.shape[1],:mlp.shape[1]]
            print("LOGITS (einsum) REshape", logits.shape)
            print("BATCH ID", batch["doc_id"])
        logits = logits + mlp
        scores = logits.masked_fill(~batch["followings_mask"],  -100000)
        # TODO APPLY softmax if objective is "is line j right after line i (or is line i is line if last) ?"
        scores = scores.softmax(dim=-1)
        #print("SCORES shape", scores.shape)
        
        
        # or SIGMOID if objective is "is line j after line i?"
        # probs = F.sigmoid(logits)  # n_pages * n_lines * n_lines
        # apply mask on logits don't mask the l
        # probs = probs.masked_fill(~batch["followings_mask"], 0.0)
        
        
        if "followings" in batch:
            # print("--  FOLLOWING shape", batch["followings"].shape)
            targets = batch["followings"]
            # print("---TARGETS shape", targets.shape, "SUM mask:", batch["followings_mask"].sum())
            
            
            # SOFTMAX
            # TODO ?
            # Vérifiez les NaN et les valeurs infinies
            individual_losses = F.binary_cross_entropy(
              scores,
              targets,
              reduction="none",
            )
            
            
            # SIGMOID
            # individual_losses = F.binary_cross_entropy_with_logits(
            #     # sig + BCELoss
            #     probs,
            #     targets,
            #     reduction="none",
            # )
            
            individual_losses = individual_losses.masked_fill(~batch["followings_mask"], 0.0)
            N = batch["followings_mask"].sum()
            N = N if N>0 else 1
            output["followings_loss"] = (
                individual_losses.sum() / N
            )
            assert not torch.isnan(output["followings_loss"]).item(), "--------- NaN encountered during loss computation"
            
            output["loss"] = output["loss"] + output["followings_loss"]
        else:
            scores = scores.refold("line")
            # print("SCORES refold line i shape", scores.shape)
            output["scores"] = scores
            output["mask"] = batch["followings_mask"].refold("line")
            
            # SOFTMAX
            output["followings"] = scores.argmax(-1)
            # output["followings"][scores.sum(-1) == 0] = -1
            # print(f"FOLLOWINGS argmax shape {output['followings'].shape}")
            
            # SIGMOID
            # output["followings"] = logits > 0.5
            

        return output

    def postprocess(self, docs: Sequence[PDFDoc], output: Dict) -> Sequence[PDFDoc]:
        #
        _sep = '|'
        # label = f'{src_b.label}{_sep}{src_b.node_num}{_sep}{src_b.rank}'
        # b.label.split(_sep)[0] # label
        # b.label.split(_sep)[1] # node_num
        # b.label.split(_sep)[2] # rank
        #
        ind = 0
        scores = output["scores"].tolist()
        for b, label in zip(
            (b for doc in docs for b in doc.text_boxes),
            output["followings"].tolist(),
        ):
            # print(f' post process: line {ind} argmax {label} logit: {scores[ind]}')
            # ind += 1
            # SOFTMAX
            l = b.label.split(_sep)[:3] + [str(label) if label != 0 else '-1']
            
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

        # self.update_weights_from_vocab_(label_voc_indices)

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
