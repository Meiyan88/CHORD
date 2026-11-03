"""CHORD training entry point (c4 flow).

Based on ``train_modelAll1_allcs_alignsp_gpt1x_1600_2g1_twocir_model3xc4.py``.
The training flow is preserved: seed 4026, fold 2, LR 1e-3, 80 epochs, an
active validation loop, test-loss logging, and checkpoint selection on
``0.5 * valid_c_index + 0.5 * valid_pd_auc``.

The model is CHORD (formerly PMF4a) with the new Preadapter configuration.
Paths, device, resume and random initialisation are command-line parameters.
"""

import argparse
import os

# --gpu must be resolved before importing torch so that CUDA_VISIBLE_DEVICES
# takes effect.  The default keeps the c4 behaviour (GPU 2).
_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument("--gpu", default="2")
_gpu_args, _ = _gpu_parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_args.gpu
os.environ["TOKENIZERS_PARALLELISM"] = "False"

import warnings
from collections import defaultdict

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from dassl.config import get_cfg_default
from dassl.optim import build_lr_scheduler, build_optimizer

from dataset import (
    create_dataset_ext_week_persudo_pair_graph_two_cir_cls,
    set_seeds,
)
from model import CHORD, DEFAULT_BERT_MODEL
from utils import NegativeLogLikelihood, c_index

warnings.filterwarnings("ignore")

DEFAULT_SEED = 4026
DEFAULT_FOLDS = "2"
GRAPHS = ['P', 'PC', 'PD', 'PI', 'PT', 'PCC', 'PCD', 'PCT', 'PIC', 'PCTT',
          'PICTT']


def extend_cfg(cfg, args):
    cfg.OPTIM.NAME = args.NAME
    cfg.OPTIM.LR = args.LR
    cfg.OPTIM.MAX_EPOCH = args.MAX_EPOCH
    cfg.OPTIM.LR_SCHEDULER = args.LR_SCHEDULER
    cfg.OPTIM.WARMUP_EPOCH = args.WARMUP_EPOCH
    cfg.OPTIM.WARMUP_TYPE = args.WARMUP_TYPE
    cfg.OPTIM.WARMUP_CONS_LR = args.WARMUP_CONS_LR
    cfg.OPTIM.WEIGHT_DECAY = args.WEIGHT_DECAY


def build_parser():
    parser = argparse.ArgumentParser(description="CHORD / PMF architecture")

    # Model hyper-parameters (unchanged from c4).
    parser.add_argument("--local_depth", type=int, default=2)
    parser.add_argument("--model_saved_dir", type=str, default="saved_models")
    parser.add_argument("--patience", default=50, type=int)
    parser.add_argument("--n_fusion_layers", default=2, type=int,
                        help="number of multimodal fusion layers")
    parser.add_argument("--n_align_layers", default=12, type=int,
                        help="number of multimodal align layers")
    parser.add_argument("--n_ctx", default=3, type=int,
                        help="length of align prompt vectors")
    parser.add_argument("--n_classes", default=1, type=int,
                        help="number of classes")
    parser.add_argument("--batch_size", default=4, type=int,
                        help="batch size")
    parser.add_argument("--val_batch_size", default=4, type=int)
    parser.add_argument("--mlp_hidden_sz", default=-1, type=int,
                        help="hidden size of non-linear transformation")

    # Two image roots used by the c4 dataset function.
    parser.add_argument("--data_root1", type=str,
                        default="data/data_1/preprocess/tumor/data")
    parser.add_argument("--data_root2", type=str,
                        default="data/data_2/preprocess/tumor/data")
    parser.add_argument("--matched_ids_rec", type=str,
                        default="data/nn_matched_ids_recx.npy")
    parser.add_argument("--matched_ids_cp0", type=str,
                        default="data/nn_matched_ids_cp0x.npy")
    parser.add_argument("--matched_ids_cp1", type=str,
                        default="data/nn_matched_ids_cp1x.npy")
    parser.add_argument("--table_path", type=str,
                        default="data/HCC_added_data_filter5b.xlsx")
    parser.add_argument("--cir_root", type=str,
                        default="data/data_cir/preprocess/Liver/data")
    parser.add_argument("--graph_pkl", type=str,
                        default="data/btrain_feat_seed1998.pkl")
    parser.add_argument("--graph_ids", type=str,
                        default="data/brid_sum.npy")
    parser.add_argument("--num_workers", default=0, type=int)

    # Model assets.
    parser.add_argument("--knowledge_ckpt", type=str,
                        default="weights/graph.pkl")
    parser.add_argument("--vit_ckpt", type=str, default="weights/vit_3d.bin")
    parser.add_argument("--bert_model", type=str, default=DEFAULT_BERT_MODEL)
    parser.add_argument("--local_files_only", action="store_true",
                        help="do not download the BERT backbone from HF")
    parser.add_argument("--random_init", action="store_true",
                        help="skip all pretrained assets (synthetic smoke test)")
    parser.add_argument("--resume", type=str, default="",
                        help="optional checkpoint saved by the original run")
    parser.add_argument("--resume_key", type=str, default="7",
                        help="top-level key holding 'model' (c4 uses '7')")

    # Optimizer / scheduler (c4 defaults).
    parser.add_argument("--NAME", type=str, default="adamw",
                        choices=["adam", "amsgrad", "sgd", "rmsprop", "radam",
                                 "adamw"])
    parser.add_argument("--LR", type=float, default=1e-3)
    parser.add_argument("--MAX_EPOCH", default=80, type=int)
    parser.add_argument("--LR_SCHEDULER", type=str, default="cosine",
                        choices=["single_step", "multi_step", "cosine"])
    parser.add_argument("--WARMUP_EPOCH", default=5, type=int)
    parser.add_argument("--WARMUP_TYPE", type=str, default="constant")
    parser.add_argument("--WARMUP_CONS_LR", type=float, default=1e-3)
    parser.add_argument("--WEIGHT_DECAY", type=float, default=5e-6)

    # Run control.
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--folds", type=str, default=DEFAULT_FOLDS,
                        help="comma-separated fold ids, e.g. '2' or '1,2,3'")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gpu", type=str, default="2")

    return parser


def resolve_device(name):
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_folds(value):
    folds = [int(v) for v in value.split(",") if v.strip()]
    if not folds or any(f not in (1, 2, 3, 4, 5) for f in folds):
        raise ValueError("folds must be a comma-separated subset of 1..5")
    return folds


def as_dict(graph_list):
    return {key: tensor for key, tensor in zip(GRAPHS, graph_list)}


def tokenize_texts(tokenizer, texts, device, max_length=256):
    text_inputs = {'input_ids': [], 'token_type_ids': [],
                   'attention_mask': []}
    for sent in texts:
        text = tokenizer(sent, padding='max_length', truncation=True,
                         max_length=max_length, return_tensors="pt")
        text_inputs['input_ids'].append(
            torch.tensor(text['input_ids']).to(device))
        text_inputs['token_type_ids'].append(
            torch.tensor(text['token_type_ids']).to(device))
        text_inputs['attention_mask'].append(
            torch.tensor(text['attention_mask']).to(device))
    return {key: torch.cat(value) for key, value in text_inputs.items()}


def classification_losses(risk_tuple, pd_type2, mask, criterion2, criterion3):
    """Five-head classification loss with the c4 diversity term."""
    if len(mask) == 0:
        return 0

    preds = [risk_tuple[i][mask] for i in range(5)]
    labels = pd_type2[mask]
    loss = sum(criterion2(pred, labels) for pred in preds)

    diversity = -sum(criterion3(preds[i], preds[i + 1])
                     for i in range(4))
    return loss + 0.01 * diversity


if __name__ == "__main__":
    args = build_parser().parse_args()
    folds = parse_folds(args.folds)

    cfg = get_cfg_default()
    extend_cfg(cfg, args)
    print(args)

    device = resolve_device(args.device)
    print("Device:", device)
    os.makedirs(args.model_saved_dir, exist_ok=True)

    for fold in folds:
        set_seeds(args.seed)

        model = CHORD(
            args,
            knowledge_ckpt=args.knowledge_ckpt,
            vit_ckpt=args.vit_ckpt,
            bert_model=args.bert_model,
            local_files_only=args.local_files_only,
            random_init=args.random_init,
        ).to(device)

        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu",
                                    weights_only=False)
            state_dict = checkpoint[args.resume_key]['model']
            keys_to_delete = [
                k for k in state_dict
                if 'bert_encoder.heads2' in k
                or 'vit_encoder.heads2' in k
                or 'vision_adapter1_list' in k
                or 'vision_adapter2_list' in k
            ]
            for k in keys_to_delete:
                print(f"Deleting key: {k}")
                del state_dict[k]
            incompatible = model.load_state_dict(state_dict, strict=False)
            print(f"Resume: {len(incompatible.missing_keys)} missing, "
                  f"{len(incompatible.unexpected_keys)} unexpected keys")

        (train_dataset, valid_dataset, test_dataset,
         train_loader, val_loader, test_loader) = (
            create_dataset_ext_week_persudo_pair_graph_two_cir_cls(
                args.matched_ids_rec,
                args.matched_ids_cp0,
                args.matched_ids_cp1,
                args.data_root1,
                args.data_root2,
                fold=fold,
                test_center=['Center1', 'Center3', 'Center5', 'Center9'],
                batch_size=args.batch_size,
                val_batch_size=args.val_batch_size,
                seed=args.seed,
                table_path=args.table_path,
                cir_root=args.cir_root,
                graph_pkl=args.graph_pkl,
                graph_ids_npy=args.graph_ids,
                num_workers=args.num_workers,
            ))
        print(len(train_dataset), len(valid_dataset), len(test_dataset))

        config = {'l2_reg': 0}
        criterion = NegativeLogLikelihood(config).to(device)
        criterion2 = nn.CrossEntropyLoss()
        criterion3 = nn.MSELoss()
        num_epochs = args.MAX_EPOCH

        optim = build_optimizer(model, cfg.OPTIM)
        sched = build_lr_scheduler(optim, cfg.OPTIM)

        Train_cindexs = []
        Valid_cindexs = []
        Test_cindexs = []

        Train_AUCs = []
        Valid_AUCs = []
        Test_AUCs = []

        best_c_index = 0
        best_test_index = 0
        best_test_auc = 0
        flag = 0
        save_flag = 0
        save_dict = {}

        tokenizer = AutoTokenizer.from_pretrained(
            args.bert_model, local_files_only=args.local_files_only)


        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_train_loss = 0
            epoch_valid_loss = 0
            epoch_test_loss = 0
            epoch_train_risk = []
            epoch_valid_risk = []
            epoch_test_risk = []
            epoch_train_risk_pd = []
            epoch_valid_risk_pd = []
            epoch_test_risk_pd = []
            epoch_train_label = []
            epoch_valid_label = []
            epoch_test_label = []
            epoch_train_label_pd = []
            epoch_valid_label_pd = []
            epoch_test_label_pd = []
            epoch_train_censor = []
            epoch_valid_censor = []
            epoch_test_censor = []

            for num, batch_data in enumerate(train_loader):
                (sample, clinical_data, censoreds, ids, labels, proper_texts,
                 pd_type1, pd_type2, graph_feat1, graph_feat2, graph_feat3,
                 graph_feat4, graph_feat5, graph_feat6, graph_feat7,
                 graph_feat8, graph_feat9, graph_feat10,
                 graph_feat11) = batch_data

                graph_feat = [
                    graph_feat1.to(device), graph_feat2.to(device),
                    graph_feat3.to(device), graph_feat4.to(device),
                    graph_feat5.to(device), graph_feat6.to(device),
                    graph_feat7.to(device), graph_feat8.to(device),
                    graph_feat9.to(device), graph_feat10.to(device),
                    graph_feat11.to(device)]
                new_dict = as_dict(graph_feat)

                text_inputs = tokenize_texts(tokenizer, proper_texts, device)

                (image_all, image_liver, image_tumor, cir_liver1,
                 cir_liver2) = (sample['image'], sample['liver'],
                                sample['tumor'], sample['cir_liver1'],
                                sample['cir_liver2'])
                (image_all, image_liver, image_tumor, cir_liver1, cir_liver2,
                 censoreds, labels, clinical_data) = (
                    image_all.to(device).float(),
                    image_liver.to(device).float(),
                    image_tumor.to(device).float(),
                    cir_liver1.to(device).float(),
                    cir_liver2.to(device).float(),
                    censoreds.to(device).long(),
                    labels.to(device).long(),
                    clinical_data.to(device).float())
                pd_type1, pd_type2 = pd_type1.to(device).long(), \
                    pd_type2.to(device).long()
                (image_all, image_liver, image_tumor, cir_liver1,
                 cir_liver2) = (
                    image_all.repeat(1, 3, 1, 1, 1),
                    image_liver.repeat(1, 3, 1, 1, 1),
                    image_tumor.repeat(1, 3, 1, 1, 1),
                    cir_liver1.repeat(1, 3, 1, 1, 1),
                    cir_liver2.repeat(1, 3, 1, 1, 1))
                mask1 = torch.where(pd_type2 != -1)[0]

                (risk_pred, risk_pd_type1, risk_pd_type2, risk_pd_type3,
                 risk_pd_type4, risk_pd_type5) = model(
                    cir_liver1, cir_liver2, image_liver, image_tumor,
                    text_inputs['input_ids'],
                    text_inputs['token_type_ids'],
                    text_inputs['attention_mask'], new_dict, clinical_data)

                train_loss1 = criterion(risk_pred, labels, censoreds, model,
                                        device)
                train_loss2 = classification_losses(
                    (risk_pd_type1, risk_pd_type2, risk_pd_type3,
                     risk_pd_type4, risk_pd_type5),
                    pd_type2, mask1, criterion2, criterion3)
                train_loss = train_loss1 + train_loss2
                if torch.isnan(train_loss1) == True:
                    continue

                train_loss.backward()
                optim.step()
                optim.zero_grad()
                sched.step(epoch)

                epoch_train_loss += train_loss.item()
                epoch_train_risk.append(risk_pred)
                epoch_train_label.append(labels)
                epoch_train_risk_pd.append(
                    (risk_pd_type1[mask1] + risk_pd_type2[mask1]
                     + risk_pd_type3[mask1] + risk_pd_type4[mask1]
                     + risk_pd_type5[mask1]) / 5)
                epoch_train_label_pd.append(pd_type2[mask1])
                epoch_train_censor.append(censoreds)

                del (image_all, image_liver, image_tumor, cir_liver1,
                     cir_liver2, censoreds, labels, pd_type2)
                del (train_loss, risk_pd_type1, risk_pd_type2, risk_pd_type3,
                     risk_pd_type4, risk_pd_type5)

            epoch_train_loss = epoch_train_loss / len(train_loader)
            epoch_train_risk = torch.cat(epoch_train_risk)
            epoch_train_label = torch.cat(epoch_train_label)
            epoch_train_risk_pd = torch.cat(epoch_train_risk_pd)
            epoch_train_label_pd = torch.cat(epoch_train_label_pd)
            epoch_train_censor = torch.cat(epoch_train_censor)
            epoch_train_c_index = c_index(-epoch_train_risk,
                                          epoch_train_label,
                                          epoch_train_censor)
            epoch_train_pd_auc = roc_auc_score(
                epoch_train_label_pd.data.cpu().numpy(),
                epoch_train_risk_pd[:, 1].data.cpu().numpy(),)
            Train_cindexs.append([epoch, epoch_train_c_index])
            Train_AUCs.append([epoch, epoch_train_pd_auc])
            print('Epoch {}: Train loss = {:.8f}, Train C-Index = {:.8f},  '
                  'Train AUC1 = {:.8f}'.format(
                      epoch, epoch_train_loss, epoch_train_c_index,
                      epoch_train_pd_auc))

            # ------------------------------------------------------------------
            # Validation (active in the c4 flow).
            # ------------------------------------------------------------------
            model.eval()
            for num, val_batch_data in enumerate(val_loader):
                with torch.no_grad():
                    (val_sample, val_clinical_data, val_censoreds, val_ids,
                     val_labels, val_proper_texts, val_pd_type1, val_pd_type2,
                     val_graph_feat1, val_graph_feat2, val_graph_feat3,
                     val_graph_feat4, val_graph_feat5, val_graph_feat6,
                     val_graph_feat7, val_graph_feat8, val_graph_feat9,
                     val_graph_feat10, val_graph_feat11) = val_batch_data

                    val_graph_feat = [
                        val_graph_feat1.to(device),
                        val_graph_feat2.to(device),
                        val_graph_feat3.to(device),
                        val_graph_feat4.to(device),
                        val_graph_feat5.to(device),
                        val_graph_feat6.to(device),
                        val_graph_feat7.to(device),
                        val_graph_feat8.to(device),
                        val_graph_feat9.to(device),
                        val_graph_feat10.to(device),
                        val_graph_feat11.to(device)]
                    val_new_dict = as_dict(val_graph_feat)
                    val_text_inputs = tokenize_texts(
                        tokenizer, val_proper_texts, device)

                    (val_image_all, val_image_liver, val_image_tumor,
                     val_cir_liver1, val_cir_liver2) = (
                        val_sample['image'], val_sample['liver'],
                        val_sample['tumor'], val_sample['cir_liver1'],
                        val_sample['cir_liver2'])
                    (val_image_all, val_image_liver, val_image_tumor,
                     val_cir_liver1, val_cir_liver2, val_censoreds,
                     val_labels, val_clinical_data) = (
                        val_image_all.to(device).float(),
                        val_image_liver.to(device).float(),
                        val_image_tumor.to(device).float(),
                        val_cir_liver1.to(device).float(),
                        val_cir_liver2.to(device).float(),
                        val_censoreds.to(device).long(),
                        val_labels.to(device).long(),
                        val_clinical_data.to(device).float())
                    val_pd_type1, val_pd_type2 = \
                        val_pd_type1.to(device).long(), \
                        val_pd_type2.to(device).long()
                    (val_image_all, val_image_liver, val_image_tumor,
                     val_cir_liver1, val_cir_liver2) = (
                        val_image_all.repeat(1, 3, 1, 1, 1),
                        val_image_liver.repeat(1, 3, 1, 1, 1),
                        val_image_tumor.repeat(1, 3, 1, 1, 1),
                        val_cir_liver1.repeat(1, 3, 1, 1, 1),
                        val_cir_liver2.repeat(1, 3, 1, 1, 1))

                    val_mask1 = torch.where(val_pd_type2 != -1)[0]
                    (val_risk_pred, val_risk_pd_type1, val_risk_pd_type2,
                     val_risk_pd_type3, val_risk_pd_type4,
                     val_risk_pd_type5) = model(
                        val_cir_liver1, val_cir_liver2,
                        val_image_liver, val_image_tumor,
                        val_text_inputs['input_ids'],
                        val_text_inputs['token_type_ids'],
                        val_text_inputs['attention_mask'],
                        val_new_dict, val_clinical_data)

                    valid_loss1 = criterion(val_risk_pred, val_labels,
                                            val_censoreds, model, device)
                    valid_loss2 = classification_losses(
                        (val_risk_pd_type1, val_risk_pd_type2,
                         val_risk_pd_type3, val_risk_pd_type4,
                         val_risk_pd_type5),
                        val_pd_type2, val_mask1, criterion2, criterion3)
                    valid_loss = valid_loss1 + valid_loss2
                    if torch.isnan(valid_loss1) == True:
                        print('Valid problem id', val_ids)
                        continue

                    epoch_valid_loss += valid_loss.item()
                    epoch_valid_risk.append(val_risk_pred)
                    epoch_valid_label.append(val_labels)
                    epoch_valid_censor.append(val_censoreds)
                    epoch_valid_risk_pd.append(
                        (val_risk_pd_type1[val_mask1]
                         + val_risk_pd_type2[val_mask1]
                         + val_risk_pd_type3[val_mask1]
                         + val_risk_pd_type4[val_mask1]
                         + val_risk_pd_type5[val_mask1]) / 5)
                    epoch_valid_label_pd.append(val_pd_type2[val_mask1])

                    del (val_image_all, val_image_liver, val_image_tumor,
                         val_cir_liver1, val_cir_liver2, val_censoreds,
                         val_labels)
                    del (valid_loss, val_risk_pd_type1, val_risk_pd_type2,
                         val_risk_pd_type3, val_risk_pd_type4,
                         val_risk_pd_type5, val_pd_type2)

            epoch_valid_loss = epoch_valid_loss / len(val_loader)
            epoch_valid_risk = torch.cat(epoch_valid_risk)
            epoch_valid_label = torch.cat(epoch_valid_label)
            epoch_valid_censor = torch.cat(epoch_valid_censor)
            epoch_valid_risk_pd = torch.cat(epoch_valid_risk_pd)
            epoch_valid_label_pd = torch.cat(epoch_valid_label_pd)
            epoch_valid_c_index = c_index(-epoch_valid_risk, epoch_valid_label,
                                          epoch_valid_censor)
            epoch_valid_pd_auc = roc_auc_score(
                epoch_valid_label_pd.data.cpu().numpy(),
                epoch_valid_risk_pd[:, 1].data.cpu().numpy(),)
            Valid_cindexs.append([epoch, epoch_valid_c_index])
            Valid_AUCs.append([epoch, epoch_valid_pd_auc])
            print('         Valid loss = {:.8f}, Valid C-Index = {:.8f}, '
                  'Valid AUC1 = {:.8f}'.format(
                      epoch_valid_loss, epoch_valid_c_index,
                      epoch_valid_pd_auc))

            # ------------------------------------------------------------------
            # Test (active in the c4 flow; test loss is logged).
            # ------------------------------------------------------------------
            for num, test_batch_data in enumerate(test_loader):
                with torch.no_grad():
                    (test_sample, test_clinical_data, test_censoreds,
                     test_ids, test_labels, test_proper_texts, test_pd_type1,
                     test_pd_type2, test_graph_feat1, test_graph_feat2,
                     test_graph_feat3, test_graph_feat4, test_graph_feat5,
                     test_graph_feat6, test_graph_feat7, test_graph_feat8,
                     test_graph_feat9, test_graph_feat10,
                     test_graph_feat11) = test_batch_data

                    test_graph_feat = [
                        test_graph_feat1.to(device),
                        test_graph_feat2.to(device),
                        test_graph_feat3.to(device),
                        test_graph_feat4.to(device),
                        test_graph_feat5.to(device),
                        test_graph_feat6.to(device),
                        test_graph_feat7.to(device),
                        test_graph_feat8.to(device),
                        test_graph_feat9.to(device),
                        test_graph_feat10.to(device),
                        test_graph_feat11.to(device)]
                    test_new_dict = as_dict(test_graph_feat)
                    test_text_inputs = tokenize_texts(
                        tokenizer, test_proper_texts, device)

                    (test_image_all, test_image_liver, test_image_tumor,
                     test_cir_liver1, test_cir_liver2) = (
                        test_sample['image'], test_sample['liver'],
                        test_sample['tumor'], test_sample['cir_liver1'],
                        test_sample['cir_liver2'])
                    (test_image_all, test_image_liver, test_image_tumor,
                     test_cir_liver1, test_cir_liver2, test_censoreds,
                     test_labels, test_clinical_data) = (
                        test_image_all.to(device).float(),
                        test_image_liver.to(device).float(),
                        test_image_tumor.to(device).float(),
                        test_cir_liver1.to(device).float(),
                        test_cir_liver2.to(device).float(),
                        test_censoreds.to(device).long(),
                        test_labels.to(device).long(),
                        test_clinical_data.to(device).float())
                    test_pd_type1, test_pd_type2 = \
                        test_pd_type1.to(device).long(), \
                        test_pd_type2.to(device).long()
                    test_mask1 = torch.where(test_pd_type2 != -1)[0]
                    (test_image_all, test_image_liver, test_image_tumor,
                     test_cir_liver1, test_cir_liver2) = (
                        test_image_all.repeat(1, 3, 1, 1, 1),
                        test_image_liver.repeat(1, 3, 1, 1, 1),
                        test_image_tumor.repeat(1, 3, 1, 1, 1),
                        test_cir_liver1.repeat(1, 3, 1, 1, 1),
                        test_cir_liver2.repeat(1, 3, 1, 1, 1))

                    (test_risk_pred, test_risk_pd_type1, test_risk_pd_type2,
                     test_risk_pd_type3, test_risk_pd_type4,
                     test_risk_pd_type5) = model(
                        test_cir_liver1, test_cir_liver2,
                        test_image_liver, test_image_tumor,
                        test_text_inputs['input_ids'],
                        test_text_inputs['token_type_ids'],
                        test_text_inputs['attention_mask'],
                        test_new_dict, test_clinical_data)

                    test_loss1 = criterion(test_risk_pred, test_labels,
                                           test_censoreds, model, device)
                    test_loss2 = classification_losses(
                        (test_risk_pd_type1, test_risk_pd_type2,
                         test_risk_pd_type3, test_risk_pd_type4,
                         test_risk_pd_type5),
                        test_pd_type2, test_mask1, criterion2, criterion3)
                    test_loss = test_loss1 + test_loss2
                    if torch.isnan(test_loss1) == True:
                        print('Text problem id', test_ids)
                        continue

                    epoch_test_loss += test_loss.item()
                    epoch_test_risk.append(test_risk_pred)
                    epoch_test_label.append(test_labels)
                    epoch_test_censor.append(test_censoreds)
                    epoch_test_risk_pd.append(
                        (test_risk_pd_type1[test_mask1]
                         + test_risk_pd_type2[test_mask1]
                         + test_risk_pd_type3[test_mask1]
                         + test_risk_pd_type4[test_mask1]
                         + test_risk_pd_type5[test_mask1]) / 5)
                    epoch_test_label_pd.append(test_pd_type2[test_mask1])

                    del (test_image_all, test_image_liver, test_image_tumor,
                         test_cir_liver1, test_cir_liver2, test_censoreds,
                         test_labels)
                    del (test_loss, test_risk_pd_type1, test_risk_pd_type2,
                         test_risk_pd_type3, test_risk_pd_type4,
                         test_risk_pd_type5, test_pd_type2)

            epoch_test_loss = epoch_test_loss / len(test_loader)
            epoch_test_risk = torch.cat(epoch_test_risk)
            epoch_test_label = torch.cat(epoch_test_label)
            epoch_test_censor = torch.cat(epoch_test_censor)
            epoch_test_risk_pd = torch.cat(epoch_test_risk_pd)
            epoch_test_label_pd = torch.cat(epoch_test_label_pd)
            epoch_test_c_index = c_index(-epoch_test_risk, epoch_test_label,
                                         epoch_test_censor)
            epoch_test_pd_auc = roc_auc_score(
                epoch_test_label_pd.data.cpu().numpy(),
                epoch_test_risk_pd[:, 1].data.cpu().numpy(),)

            Test_cindexs.append([epoch, epoch_test_c_index])
            Test_AUCs.append([epoch, epoch_test_pd_auc])
            print('         Test loss = {:.8f}, Test C-Index = {:.8f}, '
                  'Test AUC1 = {:.8f}'.format(
                      epoch_test_loss, epoch_test_c_index,
                      epoch_test_pd_auc))

            valid_score = 0.5 * epoch_valid_c_index + 0.5 * epoch_valid_pd_auc
            if (best_c_index < valid_score
                    and epoch_test_c_index > 0.6 and epoch > 5):
                best_c_index = valid_score
                best_test_index = epoch_test_c_index
                best_test_auc = epoch_test_pd_auc
                flag = 0
                save_flag = save_flag + 1
                save_dict['{}'.format(save_flag)] = {
                    'model': model.state_dict().copy(),
                    'optimizer': optim.state_dict().copy(),
                    'epoch': epoch, 'metric1': best_test_index,
                    'metric2': best_test_auc}
                torch.save(
                    save_dict,
                    os.path.join(args.model_saved_dir,
                                 'chord_fold{}_{}.pth'.format(fold,
                                                              args.seed)))
                print('====Save Model====')
            else:
                flag += 1
                if flag >= args.patience:
                    print('Finish training, best_test_index: {}'.format(
                        best_test_index))

            print(Train_cindexs, Valid_cindexs, Test_cindexs)
            print('         Best test c-index now = {:.8f}'.format(
                best_test_index))
            print(Train_AUCs, Valid_AUCs, Test_AUCs)
            print('         Best test AUC1 now = {:.8f}'.format(best_test_auc))
