"""Prediction functions for hybrid TURL + GNN architecture."""

import logging

import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

from ..data import KGDataset
from ..turl_gnn.model import TURLGNN, FusedTURLGNN
from ..parsers import HybridLMGNNParser

# Setup logging
logger = logging.getLogger(__name__)


def predict_masked_cells(args, parser: HybridLMGNNParser, 
            masked_dataset: KGDataset, 
            model: TURLGNN | FusedTURLGNN, prefix=""):
    # Copy args to not modify original object
    args = args.copy()
    args.eval_batch_size = args.per_gpu_eval_batch_size

    # Disable dynamically masking and link prediction
    args.mlm_probability = 0.0
    args.ent_mlm_probability = 0.0
    args.drop_edge_prob = 0.0

    current_edge_ind: list[int] = []
    def _collate_fn_wrapper(samples: list[int]):
        current_edge_ind.clear()
        current_edge_ind.extend(samples)
        return parser.collate_fn(masked_dataset, samples, args=args, train=False)

    # Define eval data loader
    masked_dataset.set_khop(enable=True, num_neighbors=args.num_neighbors)
    eval_loader = DataLoader(masked_dataset, batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=_collate_fn_wrapper)

    # Predict
    predicted_edge_df = masked_dataset.edge_data.copy()
    logger.info("***** Running prediction {} *****".format(prefix))
    logger.info("  Num examples = %d", len(masked_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    with torch.no_grad():
        step=0
        for batch in tqdm(eval_loader, desc=f"Evaluating", position=0, leave=False):
            step+=1

            ## Model inputs
            # General input
            seed_edge_node_ids, node_ent_ids = batch[2]            
            seed_edge_node_ids, node_ent_ids = seed_edge_node_ids.to(args.device), node_ent_ids.to(args.device)

            # TURL input
            turl_kwargs = batch[0]
            turl_kwargs = {k: v.to(args.device) for k, v in turl_kwargs.items()}

            input_ent_labels = turl_kwargs['ent_masked_lm_labels']
            ent_candidates = turl_kwargs['ent_candidates']

            input_ent, input_ent_pos = turl_kwargs['input_ent'], turl_kwargs['input_ent_pos']
            id_ent_mask = torch.isin(input_ent, parser.lm_parser.id_ent_id_set.to(args.device)).detach()
            edge_feat_ent_mask = torch.isin(input_ent_pos[..., 1], parser.lm_parser.edge_feat_cols.to(args.device)).detach()
            node_feat_ent_mask = torch.isin(input_ent_pos[..., 1], parser.lm_parser.node_feat_cols.to(args.device)).detach()
            turl_kwargs.update(dict(edge_feat_ent_mask=edge_feat_ent_mask, node_feat_ent_mask=node_feat_ent_mask, id_ent_mask=id_ent_mask))

            # GNN input
            gnn_data, pos_seed_mask, neg_seed_mask = batch[1] 
            gnn_data.to(args.device)
            gnn_kwargs = dict(x=gnn_data.x, edge_index=gnn_data.edge_index, edge_attr=gnn_data.edge_attr, 
                            pos_edge_index=gnn_data.pos_edge_index, pos_edge_attr=gnn_data.pos_edge_attr,
                            neg_edge_index=gnn_data.neg_edge_index, neg_edge_attr=gnn_data.neg_edge_attr)
                    
            ## Forward pass
            model.train()
            turl_outputs, _ = model(turl_kwargs, gnn_kwargs, seed_edge_node_ids, node_ent_ids)
            del turl_kwargs, gnn_kwargs
            torch.cuda.empty_cache()

            _, ent_outputs = turl_outputs

            ent_prediction_scores = ent_outputs[1]

            # Free some GPU memory
            ent_prediction_scores, input_ent_labels, ent_candidates, input_ent_pos = \
                ent_prediction_scores.detach().cpu(), input_ent_labels.detach().cpu(), ent_candidates.detach().cpu(), input_ent_pos.detach().cpu()

            # Reconstruct predictions
            ent_cand_pred = ent_prediction_scores.argmax(dim=-1)
            ent_pred = ent_candidates[0][torch.arange(ent_cand_pred[0].shape[0]), ent_cand_pred[0]]
            ent_pred = ent_pred.detach().cpu().numpy()
            # get entity names
            ent_pred = np.array([parser.lm_parser.idx2ent[x] for x in ent_pred])
            # get which entities are masked
            ent_pred_masked = input_ent_labels != -1
            # get ent positions in table
            ent_pred = ent_pred[ent_pred_masked[0].detach().cpu().numpy()]
            # update prediction edge df
            edge_ind_pred = np.array(current_edge_ind)
            # TODO: generalize for multi-columns
            edge_ind_pred = edge_ind_pred[predicted_edge_df.loc[current_edge_ind, 'Relation'] == '[ENT_MASK]']
            predicted_edge_df.loc[edge_ind_pred, 'Relation'] = ent_pred

    return predicted_edge_df


def _predict_links(args, parser: HybridLMGNNParser, 
            cell_filled_dataset: KGDataset, 
            model: TURLGNN | FusedTURLGNN, prefix=""):
    # Copy args to not modify original object
    args = args.copy()
    args.eval_batch_size = args.per_gpu_eval_batch_size

    # Disable dynamically masking
    args.mlm_probability = 0.0
    args.ent_mlm_probability = 0.0

    # Enable link prediction on all edges 
    # (each seed node gets paired with 64 other nodes from the seed set and the model 
    # predicts which are the most likely pairs).
    # This also drops all seed edges from the input subgraph
    args.drop_edge_prob = 1.0

    # Indicate that we are predicting edges, instead of unsupervised learning
    args.predict_new_edges = True

    # Confidence threshold to predict a new edge
    likelihood_threshold = 0.5

    current_edge_ind: list[int] = []
    def _collate_fn_wrapper(samples: list[int]):
        current_edge_ind.clear()
        current_edge_ind.extend(samples)
        return parser.collate_fn(cell_filled_dataset, samples, args=args, train=False)

    # Define eval data loader
    cell_filled_dataset.set_khop(enable=True, num_neighbors=args.num_neighbors)
    eval_loader = DataLoader(cell_filled_dataset, batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=_collate_fn_wrapper)

    # Predict
    new_predicted_edges_df = pd.DataFrame(columns=cell_filled_dataset.edge_data.columns).astype(cell_filled_dataset.edge_data.dtypes)
    logger.info("***** Running prediction {} *****".format(prefix))
    logger.info("  Num examples = %d", len(cell_filled_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    with torch.no_grad():
        step=0
        for batch in tqdm(eval_loader, desc=f"Evaluating", position=0, leave=False):
            step+=1

            ## Model inputs
            # General input
            seed_edge_node_ids, node_ent_ids = batch[2]            
            seed_edge_node_ids, node_ent_ids = seed_edge_node_ids.to(args.device), node_ent_ids.to(args.device)

            # TURL input
            turl_kwargs = batch[0]
            turl_kwargs = {k: v.to(args.device) for k, v in turl_kwargs.items()}

            input_ent, input_ent_pos = turl_kwargs['input_ent'], turl_kwargs['input_ent_pos']
            id_ent_mask = torch.isin(input_ent, parser.lm_parser.id_ent_id_set.to(args.device)).detach()
            edge_feat_ent_mask = torch.isin(input_ent_pos[..., 1], parser.lm_parser.edge_feat_cols.to(args.device)).detach()
            node_feat_ent_mask = torch.isin(input_ent_pos[..., 1], parser.lm_parser.node_feat_cols.to(args.device)).detach()
            turl_kwargs.update(dict(edge_feat_ent_mask=edge_feat_ent_mask, node_feat_ent_mask=node_feat_ent_mask, id_ent_mask=id_ent_mask))

            # GNN input
            gnn_data, pos_seed_mask, neg_seed_mask = batch[1] 
            gnn_data.to(args.device)
            gnn_kwargs = dict(x=gnn_data.x, edge_index=gnn_data.edge_index, edge_attr=gnn_data.edge_attr, 
                            pos_edge_index=gnn_data.pos_edge_index, pos_edge_attr=gnn_data.pos_edge_attr,
                            neg_edge_index=gnn_data.neg_edge_index, neg_edge_attr=gnn_data.neg_edge_attr)
                    
            ## Forward pass
            model.train()
            _, gnn_outputs = model(turl_kwargs, gnn_kwargs, seed_edge_node_ids, node_ent_ids)
            del turl_kwargs, gnn_kwargs
            torch.cuda.empty_cache()

            # GNN outputs
            neg_pred = gnn_outputs[1]

            # Reconstruct predicted edges
            inv_n_id_map = batch[3]
            neg_pred = (neg_pred >= likelihood_threshold).squeeze()
            predicted_edge_ind: torch.Tensor = gnn_data.neg_edge_index.T[neg_pred]
            # If at least 1 new edge is predicted
            if predicted_edge_ind.numel() > 0:
                predicted_edge_ind = np.array([[inv_n_id_map[x.item()] for x in edge] for edge in predicted_edge_ind])
                # Append predicted edge(s) as new row(s)
                new_rows_df = pd.DataFrame({'SRC': predicted_edge_ind[:, 0], 'DST': predicted_edge_ind[:, 1]})
                # Get node names into their ID columns
                _config = cell_filled_dataset.config
                new_rows_df[_config.edge_src_col] = cell_filled_dataset.node_data.loc[predicted_edge_ind[:, 0], 'index'].values # SRC
                new_rows_df[_config.edge_dst_col] = cell_filled_dataset.node_data.loc[predicted_edge_ind[:, 1], 'index'].values # DST
                # Fill other columns with `[ENT_MASK]`
                for col in new_predicted_edges_df.columns:
                    if col in new_rows_df:
                        continue
                    new_rows_df[col] = '[ENT_MASK]'
                new_predicted_edges_df = pd.concat((new_predicted_edges_df, new_rows_df), axis=0, ignore_index=True)

    return new_predicted_edges_df



def predict(args, parser: HybridLMGNNParser,
            masked_dataset: KGDataset, 
            model: TURLGNN | FusedTURLGNN, prefix="") -> tuple[pd.DataFrame, pd.DataFrame, pd.DateFrame]:
    """Predict using hybrid TURL+GNN architecture."""
    
    # Cell filling
    logger.info('===== Cell Filling =====')
    predicted_edge_df = predict_masked_cells(args, parser, masked_dataset, model, prefix=prefix)

    # Introduce predicted cells into a new dataset
    cell_filled_dataset = masked_dataset.copy()
    cell_filled_dataset.edge_data = predicted_edge_df

    # Link prediction
    logger.info('===== Link Prediction =====')
    new_predicted_edges_df = _predict_links(args, parser, cell_filled_dataset, model, prefix=prefix)

    # Introduce new predicted links
    extended_edges_df = pd.concat((new_predicted_edges_df, predicted_edge_df), axis=0, ignore_index=True)
    extended_links_dataset = cell_filled_dataset
    extended_links_dataset.edge_data = extended_edges_df
    # call preprocessing function
    extended_links_dataset._init(extended_links_dataset.num_neighbors)
    del cell_filled_dataset

    # Complete edge attributes of predicted links
    extended_edges_df = predict_masked_cells(args, parser, extended_links_dataset, model, prefix=prefix)

    # Update new predicted edges with the predicted edge attributes
    new_predicted_edges_df = extended_edges_df.iloc[:len(new_predicted_edges_df)]

    return predicted_edge_df, new_predicted_edges_df, extended_edges_df