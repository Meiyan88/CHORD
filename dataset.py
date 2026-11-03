"""Dataset construction for CHORD (c4 training flow).

Only the symbols used by the training entry point are kept:
``set_seeds``, ``mix_lists``, ``LiverDataset_disentangle_twocir_graph_cls`` and
``create_dataset_ext_week_persudo_pair_graph_two_cir_cls``. The data processing
logic is unchanged; the hard-coded paths are parameters whose defaults are
project-relative.

The c4 pipeline uses **two** image roots:
``data_FM_new`` (with a ``tumor`` directory) and ``data_added_filter`` (with a
``tumormax`` directory). The dataset class recognises the root by those
substrings and maps each volume to its ``liverspleen`` / ``liver`` counterpart.
"""

import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandFlipd,
    RandRotated,
    RandZoomd,
    ScaleIntensityd,
    ToTensord,
)
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def set_seeds(seed=1998):
    """Seed Python, NumPy and PyTorch RNGs."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mix_lists(gpt_textx, gpt_texty, ratio):
    """Deterministically blend the two text columns (seed fixed to 1998)."""
    random.seed(1998)
    L = len(gpt_textx)
    n = round(L * ratio)

    indices = random.sample(range(L), n)

    mixed_list = []
    for i in range(L):
        if i in indices:
            mixed_list.append(gpt_textx[i])
        else:
            mixed_list.append(gpt_texty[i])
    return mixed_list


class LiverDataset_disentangle_twocir_graph_cls(Dataset):
    def __init__(self, image_files, image_cir_files1, image_cir_files2,
                 labels, clinical_data, censoreds, ids, text, pd_type,
                 pd_type2, graph_feat, graph_rids, transforms):
        self.image_files = image_files
        self.image_cir_files1 = image_cir_files1
        self.image_cir_files2 = image_cir_files2
        self.labels = labels
        self.transforms = transforms
        self.graph_feat = graph_feat
        self.clinical_data = clinical_data.astype(np.float32)
        self.ids = ids
        self.censoreds = censoreds.astype(int)
        self.text = text
        self.graph_rids = graph_rids
        self.pd_type = pd_type
        self.pd_type2 = pd_type2

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        sample = {}
        # c4 uses two source roots; the sample key names differ per root.
        if 'data_FM_new' in self.image_files[index]:
            sample['image'] = str(np.char.replace(
                self.image_files[index], 'tumor', 'liverspleen'))
            sample['liver'] = str(np.char.replace(
                self.image_files[index], 'tumor', 'liver'))
            sample['tumor'] = self.image_files[index]
        elif 'data_added_filter' in self.image_files[index]:
            sample['image'] = str(np.char.replace(
                self.image_files[index], 'tumormax', 'liverspleen'))
            sample['liver'] = str(np.char.replace(
                self.image_files[index], 'tumormax', 'liver'))
            sample['tumor'] = self.image_files[index]
        sample['cir_liver1'] = self.image_cir_files1[index]
        sample['cir_liver2'] = self.image_cir_files2[index]
        sample = self.transforms(sample)
        pos = np.where(self.graph_rids == self.ids[index])[0][0]

        return (sample, self.clinical_data[index], self.censoreds[index],
                self.ids[index], self.labels[index],
                self.text[index], self.pd_type[index], self.pd_type2[index],
                self.graph_feat['P'][pos, :, :],
                self.graph_feat['PC'][pos, :, :],
                self.graph_feat['PD'][pos, :, :],
                self.graph_feat['PI'][pos, :, :],
                self.graph_feat['PT'][pos, :, :],
                self.graph_feat['PCC'][pos, :, :],
                self.graph_feat['PCD'][pos, :, :],
                self.graph_feat['PCT'][pos, :, :],
                self.graph_feat['PIC'][pos, :, :],
                self.graph_feat['PCTT'][pos, :, :],
                self.graph_feat['PICTT'][pos, :, :])


def create_dataset_ext_week_persudo_pair_graph_two_cir_cls(
        matched_ids_file_rec,
        matched_ids_file_cp0,
        matched_ids_file_cp1,
        root_dir1, root_dir2, fold, test_center=['Center9'],
        num_durations=6, batch_size=6, val_batch_size=4, seed=1998,
        table_path=r"/home/1/Liver_data/table/table.xlsx",
        cir_root=r'/home/1/Liver_data/data_cir/preprocess/Liver/data/',
        graph_pkl=r'/home/1/PycharmProjects/Survival_gnn/graph.pkl',
        graph_ids_npy=r"/home/1/PycharmProjects/Survival_gnn/rid_sum.npy",
        num_workers=0):

    match_ids_rec = np.load(matched_ids_file_rec)
    ids_cp0 = np.load(matched_ids_file_cp0)
    ids_cp1 = np.load(matched_ids_file_cp1)[:, 0, :]

    match_ids_cp = np.concatenate([ids_cp0, ids_cp1])

    data_dir1 = root_dir1
    hospital_names1 = (
        sorted([x for x in os.listdir(data_dir1)
                if os.path.isdir(os.path.join(data_dir1, x))]))
    data_dir2 = root_dir2
    hospital_names2 = (
        sorted([x for x in os.listdir(data_dir2)
                if os.path.isdir(os.path.join(data_dir2, x))]))
    hospital_names = hospital_names1 + hospital_names2

    image_files1 = [[os.path.join(data_dir1, class_name, x)
                     for x in os.listdir(os.path.join(data_dir1, class_name))]
                    for class_name in hospital_names1]
    image_files2 = [[os.path.join(data_dir2, class_name, x)
                     for x in os.listdir(os.path.join(data_dir2, class_name))]
                    for class_name in hospital_names2]
    image_files = image_files1 + image_files2

    image_files_need = []
    image_cir_files_need1 = []
    image_cir_files_need2 = []
    info = pd.read_excel(table_path)
    ids = info['序号']
    hospitals = info['病例来源']

    surv_labels = info['PFS']
    cp_labels = info['Child-Pugh分级']
    if_survs = info['是否可做PD分析']
    if_censoreds = info['是否发生PD']
    pd_type = info['PD类型']
    pd_type2 = info['PD类型2']
    gpt_textx = list(info['text3'])
    gpt_texty = list(info['text2'])

    ratio = 0.4
    gpt_text = mix_lists(gpt_textx, gpt_texty, ratio)
    pd_type = np.asarray(pd_type)
    pd_type2 = np.asarray(pd_type2)

    continuous_clinicl = info.iloc[:, [5, 10]]
    discrete_clinicl = info.iloc[:, [6, 7, 8, 9, 11, 12, 23]]
    continuous_clinicl = np.asarray(continuous_clinicl)
    discrete_clinicl = np.asarray(discrete_clinicl)
    discrete_clinicl[np.where(np.isnan(discrete_clinicl) == True)] = 1
    OHencoder = OneHotEncoder(categories='auto', sparse_output=False)
    discrete_clinicl = OHencoder.fit_transform(discrete_clinicl)

    ids = np.asarray(ids)
    hospitals = np.asarray(hospitals)
    surv_labels = np.asarray(surv_labels)
    surv_labels = (surv_labels / 7.0).round().astype(int)
    cp_labels = np.asarray(cp_labels)
    if_survs = np.asarray(if_survs)
    if_censoreds = np.asarray(if_censoreds)
    gpt_text = np.asarray(gpt_text).astype(str)

    labels = []
    continuous_clinicl_sum = []
    discrete_clinicl_sum = []
    ids_sum = []
    censoreds_sum = []
    text_sum = []
    pd_type_sum = []
    pd_type2_sum = []

    exist_ID = []
    for image_files_hospital in image_files:
        labels_hospital = []
        image_files_need_hospital = []
        continuous_clinicl_hospital = []
        discrete_clinicl_hospital = []
        ids_hospital = []
        censored_hospital = []
        text_hospital = []
        image_cir_files_need_hospital1 = []
        image_cir_files_need_hospital2 = []
        pd_type_hospital = []
        pd_type2_hospital = []
        for image_file in image_files_hospital:
            ID = os.path.basename(image_file).split('.')[0]
            center_now = os.path.basename(os.path.dirname(image_file))
            pos = np.where(ids.astype(str) == ID)
            pos_match1 = np.where(match_ids_rec[:, -1] == center_now + '_' + ID)
            pos_match2 = np.where(match_ids_cp[:, -1] == center_now + '_' + ID)
            if len(pos[0]) != 0 and ID not in exist_ID:
                exist_ID.append(ID)
                cp_now = cp_labels[pos][0]
                if cp_now != 3:
                    id_cir_now1 = match_ids_rec[pos_match1[0], 0][0]
                    id_cir_now2 = match_ids_cp[pos_match2[0], 0][0]
                    image_cir_file1 = os.path.join(
                        cir_root, id_cir_now1.split('_')[0],
                        id_cir_now1 + '.nii.gz')
                    image_cir_file2 = os.path.join(
                        cir_root, id_cir_now2.split('_')[0],
                        id_cir_now2 + '.nii.gz')
                    surv_label = surv_labels[pos][0]
                    if_censored = if_censoreds[pos][0]
                    text_now = gpt_text[pos][0]

                    pd_type_now = pd_type[pos][0]
                    pd_type2_now = pd_type2[pos][0]

                    pd_type_hospital.append(pd_type_now)
                    pd_type2_hospital.append(pd_type2_now)

                    continuous_clinicl_now = continuous_clinicl[pos][0]
                    discrete_clinicl_now = discrete_clinicl[pos][0]
                    image_files_need_hospital.append(image_file)
                    ids_hospital.append(ID)
                    text_hospital.append(text_now)
                    labels_hospital.append(surv_label)
                    censored_hospital.append(if_censored)
                    image_cir_files_need_hospital1.append(image_cir_file1)
                    image_cir_files_need_hospital2.append(image_cir_file2)
                    continuous_clinicl_hospital.append(continuous_clinicl_now)
                    discrete_clinicl_hospital.append(discrete_clinicl_now)

        pd_type_hospital = np.array(pd_type_hospital)
        pd_type2_hospital = np.array(pd_type2_hospital)
        labels_hospital = np.array(labels_hospital)
        censored_hospital = np.array(censored_hospital)
        image_files_need_hospital = np.array(image_files_need_hospital)
        image_cir_files_need_hospital1 = np.array(
            image_cir_files_need_hospital1)
        image_cir_files_need_hospital2 = np.array(
            image_cir_files_need_hospital2)
        continuous_clinicl_hospital = np.array(continuous_clinicl_hospital)
        discrete_clinicl_hospital = np.array(discrete_clinicl_hospital)
        ids_hospital = np.array(ids_hospital)
        text_hospital = np.array(text_hospital)
        labels.append(labels_hospital)
        image_files_need.append(image_files_need_hospital)
        image_cir_files_need1.append(image_cir_files_need_hospital1)
        image_cir_files_need2.append(image_cir_files_need_hospital2)
        ids_sum.append(ids_hospital)
        censoreds_sum.append(censored_hospital)
        text_sum.append(text_hospital)
        continuous_clinicl_sum.append(continuous_clinicl_hospital)
        discrete_clinicl_sum.append(discrete_clinicl_hospital)
        pd_type_sum.append(pd_type_hospital)
        pd_type2_sum.append(pd_type2_hospital)

    _fold = 0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    train_val_image_files = []
    train_val_image_cir_files1 = []
    train_val_image_cir_files2 = []
    train_val_labels = []
    train_val_con_clinics = []
    train_val_dis_clinics = []
    train_val_ids = []
    train_val_text = []
    train_val_censoreds = []
    train_val_pd_type = []
    train_val_pd_type2 = []

    train_image_files = []
    train_image_cir_files1 = []
    train_image_cir_files2 = []
    train_labels = []
    train_con_clinics = []
    train_dis_clinics = []
    train_ids = []
    train_text = []
    train_censoreds = []
    train_pd_type = []
    train_pd_type2 = []

    valid_image_files = []
    valid_image_cir_files1 = []
    valid_image_cir_files2 = []
    valid_labels = []
    valid_con_clinics = []
    valid_dis_clinics = []
    valid_ids = []
    valid_text = []
    valid_censoreds = []
    valid_pd_type = []
    valid_pd_type2 = []

    test_image_files = []
    test_image_cir_files1 = []
    test_image_cir_files2 = []
    test_labels = []
    test_con_clinics = []
    test_dis_clinics = []
    test_ids = []
    test_text = []
    test_censoreds = []
    test_pd_type = []
    test_pd_type2 = []

    for hospital in range(len(image_files_need)):
        image_files_split = np.array(image_files_need[hospital])
        image_cir_files_split1 = np.array(image_cir_files_need1[hospital])
        image_cir_files_split2 = np.array(image_cir_files_need2[hospital])
        labels_split = labels[hospital]
        ids_split = ids_sum[hospital]
        censoreds_split = censoreds_sum[hospital]
        hospital_name_split = hospital_names[hospital]
        hospital_name_split = hospital_name_split.split('_')[0]
        text_split = text_sum[hospital]
        pd_type_split = pd_type_sum[hospital]
        pd_type2_split = pd_type2_sum[hospital]
        continuous_clinicl_split = continuous_clinicl_sum[hospital]
        discrete_clinicl_split = discrete_clinicl_sum[hospital]
        if hospital_name_split not in test_center:
            train_val_image_files.append(image_files_split)
            train_val_image_cir_files1.append(image_cir_files_split1)
            train_val_image_cir_files2.append(image_cir_files_split2)
            train_val_labels.append(labels_split)
            train_val_ids.append(ids_split)
            train_val_censoreds.append(censoreds_split)
            train_val_text.append(text_split)
            train_val_dis_clinics.append(discrete_clinicl_split)
            train_val_con_clinics.append(continuous_clinicl_split)
            train_val_pd_type.append(pd_type_split)
            train_val_pd_type2.append(pd_type2_split)
        else:
            test_image_files.append(image_files_split)
            test_image_cir_files1.append(image_cir_files_split1)
            test_image_cir_files2.append(image_cir_files_split2)
            test_labels.append(labels_split)
            test_ids.append(ids_split)
            test_censoreds.append(censoreds_split)
            test_text.append(text_split)
            test_dis_clinics.append(discrete_clinicl_split)
            test_con_clinics.append(continuous_clinicl_split)
            test_pd_type.append(pd_type_split)
            test_pd_type2.append(pd_type2_split)

    for hospital in range(len(train_val_image_files)):
        pd_type_split = train_val_pd_type[hospital]
        pd_type2_split = train_val_pd_type2[hospital]
        image_files_split = np.array(train_val_image_files[hospital])
        image_cir_files_split1 = np.array(train_val_image_cir_files1[hospital])
        image_cir_files_split2 = np.array(train_val_image_cir_files2[hospital])
        labels_split = train_val_labels[hospital]
        ids_split = train_val_ids[hospital]
        text_split = train_val_text[hospital]
        censoreds_split = train_val_censoreds[hospital]
        continuous_clinicl_split = train_val_con_clinics[hospital]
        discrete_clinicl_split = train_val_dis_clinics[hospital]
        if len(labels_split) < 5:
            train_image_files.append(image_files_split)
            train_image_cir_files1.append(image_cir_files_split1)
            train_image_cir_files2.append(image_cir_files_split2)
            train_labels.append(labels_split)
            train_dis_clinics.append(discrete_clinicl_split)
            train_con_clinics.append(continuous_clinicl_split)
            train_ids.append(ids_split)
            train_censoreds.append(censoreds_split)
            train_text.append(text_split)
            train_pd_type.append(pd_type_split)
            train_pd_type2.append(pd_type2_split)
        else:
            for train_index, valid_index in kf.split(image_files_split,
                                                     censoreds_split):
                _fold = _fold + 1
                if _fold == fold:

                    train_image_files_split = image_files_split[train_index]
                    train_image_files.append(train_image_files_split)
                    train_image_cir_files_split1 = image_cir_files_split1[
                        train_index]
                    train_image_cir_files1.append(train_image_cir_files_split1)
                    train_image_cir_files_split2 = image_cir_files_split2[
                        train_index]
                    train_image_cir_files2.append(train_image_cir_files_split2)
                    train_labels_split = labels_split[train_index]
                    train_labels.append(train_labels_split)

                    train_ids_split = ids_split[train_index]
                    train_ids.append(train_ids_split)
                    train_censoreds_split = censoreds_split[train_index]
                    train_censoreds.append(train_censoreds_split)
                    train_text_split = text_split[train_index]
                    train_text.append(train_text_split)

                    train_discrete_clinicl_split = discrete_clinicl_split[
                        train_index]
                    train_dis_clinics.append(train_discrete_clinicl_split)

                    train_continuous_clinicl_split = continuous_clinicl_split[
                        train_index]
                    train_con_clinics.append(train_continuous_clinicl_split)

                    train_pd_type_split = pd_type_split[train_index]
                    train_pd_type.append(train_pd_type_split)
                    train_pd_type2_split = pd_type2_split[train_index]
                    train_pd_type2.append(train_pd_type2_split)

                    valid_image_files_split = image_files_split[valid_index]
                    valid_image_files.append(valid_image_files_split)
                    valid_image_cir_files_split1 = image_cir_files_split1[
                        valid_index]
                    valid_image_cir_files1.append(valid_image_cir_files_split1)
                    valid_image_cir_files_split2 = image_cir_files_split2[
                        valid_index]
                    valid_image_cir_files2.append(valid_image_cir_files_split2)
                    valid_labels_split = labels_split[valid_index]
                    valid_labels.append(valid_labels_split)
                    valid_ids_split = ids_split[valid_index]
                    valid_ids.append(valid_ids_split)
                    valid_censoreds_split = censoreds_split[valid_index]
                    valid_censoreds.append(valid_censoreds_split)
                    valid_text_split = text_split[valid_index]
                    valid_text.append(valid_text_split)

                    valid_discrete_clinicl_split = discrete_clinicl_split[
                        valid_index]
                    valid_dis_clinics.append(valid_discrete_clinicl_split)

                    valid_continuous_clinicl_split = continuous_clinicl_split[
                        valid_index]
                    valid_con_clinics.append(valid_continuous_clinicl_split)

                    valid_pd_type_split = pd_type_split[valid_index]
                    valid_pd_type.append(valid_pd_type_split)
                    valid_pd_type2_split = pd_type2_split[valid_index]
                    valid_pd_type2.append(valid_pd_type2_split)
            _fold = 0

    Scaler = StandardScaler()

    train_image_files = np.concatenate(train_image_files, axis=0)
    train_image_cir_files1 = np.concatenate(train_image_cir_files1, axis=0)
    train_image_cir_files2 = np.concatenate(train_image_cir_files2, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_con_clinics = np.concatenate(train_con_clinics, axis=0)
    train_dis_clinics = np.concatenate(train_dis_clinics, axis=0)
    train_ids = np.concatenate(train_ids, axis=0)
    train_censoreds = np.concatenate(train_censoreds, axis=0)
    train_text = np.concatenate(train_text, axis=0)
    train_con_clinics_ts = Scaler.fit_transform(train_con_clinics)
    train_clinics = np.concatenate([train_con_clinics_ts, train_dis_clinics],
                                   axis=-1)
    train_pd_type = np.concatenate(train_pd_type, axis=0)
    train_pd_type2 = np.concatenate(train_pd_type2, axis=0)

    valid_image_files = np.concatenate(valid_image_files, axis=0)
    valid_image_cir_files1 = np.concatenate(valid_image_cir_files1, axis=0)
    valid_image_cir_files2 = np.concatenate(valid_image_cir_files2, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    valid_con_clinics = np.concatenate(valid_con_clinics, axis=0)
    valid_dis_clinics = np.concatenate(valid_dis_clinics, axis=0)
    valid_ids = np.concatenate(valid_ids, axis=0)
    valid_censoreds = np.concatenate(valid_censoreds, axis=0)
    valid_text = np.concatenate(valid_text, axis=0)
    valid_con_clinics_ts = Scaler.transform(valid_con_clinics)
    valid_clinics = np.concatenate([valid_con_clinics_ts, valid_dis_clinics],
                                   axis=-1)
    valid_pd_type = np.concatenate(valid_pd_type, axis=0)
    valid_pd_type2 = np.concatenate(valid_pd_type2, axis=0)

    test_image_files = np.concatenate(test_image_files, axis=0)
    test_image_cir_files1 = np.concatenate(test_image_cir_files1, axis=0)
    test_image_cir_files2 = np.concatenate(test_image_cir_files2, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_con_clinics = np.concatenate(test_con_clinics, axis=0)
    test_dis_clinics = np.concatenate(test_dis_clinics, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)
    test_censoreds = np.concatenate(test_censoreds, axis=0)
    test_text = np.concatenate(test_text, axis=0)
    test_con_clinics_ts = Scaler.transform(test_con_clinics)
    test_clinics = np.concatenate([test_con_clinics_ts, test_dis_clinics],
                                  axis=-1)
    test_pd_type = np.concatenate(test_pd_type, axis=0)
    test_pd_type2 = np.concatenate(test_pd_type2, axis=0)

    labtrans = LabTransDiscreteTime(num_durations, scheme='quantiles')
    labtrans.fit(train_labels, train_censoreds)
    train_durations_discrete, train_events = labtrans.transform(
        train_labels, train_censoreds)
    valid_durations_discrete, valid_events = labtrans.transform(
        valid_labels, valid_censoreds)
    test_durations_discrete, test_events = labtrans.transform(
        test_labels, test_censoreds)

    train_text = train_text.tolist()
    valid_text = valid_text.tolist()
    test_text = test_text.tolist()

    train_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver1",
                         "cir_liver2"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver1",
                                  "cir_liver2"], ),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver1",
                              "cir_liver2"], ),
        RandRotated(keys=["image", "liver", "tumor", "cir_liver1",
                          "cir_liver2"], range_x=15, prob=0.5,
                    keep_size=True),
        RandFlipd(keys=["image", "liver", "tumor", "cir_liver1", "cir_liver2"],
                  spatial_axis=0, prob=0.2),
        RandZoomd(keys=["image", "liver", "tumor", "cir_liver1", "cir_liver2"],
                  min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver1", "cir_liver2"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "liver", "tumor", "cir_liver1",
                         "cir_liver2"], image_only=True),
        EnsureChannelFirstd(keys=["image", "liver", "tumor", "cir_liver1",
                                  "cir_liver2"]),
        ScaleIntensityd(keys=["image", "liver", "tumor", "cir_liver1",
                              "cir_liver2"], ),
        ToTensord(keys=["image", "liver", "tumor", "cir_liver1", "cir_liver2"]),
    ])

    with open(graph_pkl, 'rb') as f:
        raw_feat = pickle.load(f)
    rids = np.load(graph_ids_npy)
    train_dataset = LiverDataset_disentangle_twocir_graph_cls(
        train_image_files, train_image_cir_files1, train_image_cir_files2,
        train_labels, train_clinics, train_censoreds, train_ids, train_text,
        train_pd_type, train_pd_type2, raw_feat, rids, train_transforms)
    valid_dataset = LiverDataset_disentangle_twocir_graph_cls(
        valid_image_files, valid_image_cir_files1, valid_image_cir_files2,
        valid_labels, valid_clinics, valid_censoreds, valid_ids, valid_text,
        valid_pd_type, valid_pd_type2, raw_feat, rids, val_transforms)
    test_dataset = LiverDataset_disentangle_twocir_graph_cls(
        test_image_files, test_image_cir_files1, test_image_cir_files2,
        test_labels, test_clinics, test_censoreds, test_ids, test_text,
        test_pd_type, test_pd_type2, raw_feat, rids, val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=val_batch_size,
                            num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size,
                             num_workers=num_workers)
    return (train_dataset, valid_dataset, test_dataset,
            train_loader, val_loader, test_loader)
