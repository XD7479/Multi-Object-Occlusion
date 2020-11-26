import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import copy

from configs import device_ids, feat_stride
from configs import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def visual_compete_pixel(binary_mask, id1, id2):
    mask = binary_mask.astype(np.uint8) * 255
    mask = np.stack((mask, mask, mask), axis=2)
    write_path = './visual_compete_pixel/'
    if not os.path.exists(write_path):
        os.mkdir(write_path)
    cv2.imwrite(write_path + '{}_{}.jpg'.format(id1, id2), mask)
    return

class Net(nn.Module):

    def __init__(self, Feature_Extractor, VC_Centers, Context_Kernels, Mixture_Models, Clutter_Models, vMF_kappa):
        super(Net, self).__init__()

        self.extractor = Feature_Extractor

        self.vc_conv1o1 = Conv1o1Layer(VC_Centers)
        self.context_conv1o1 = Conv1o1Layer(Context_Kernels)

        self.fg_models, self.fg_prior, self.context_models, self.context_prior = Mixture_Models
        self.clutter_conv1o1 = Conv1o1Layer(Clutter_Models)
        self.exp = ExpLayer(vMF_kappa)
        self.kappa = vMF_kappa

        self.num_class = len(self.fg_models)
        self.fused_models = []

    def forward(self, org_x, bboxes, bbox_type='amodal', input_label=None, gt_labels=None, crop_pad=32,
                slide_window_stride=2,
                slide_window_bool=None, clutter_weight=1.0, reasoning=False, filter_param=[0.95, 33], min_bthrd=0,
                diff_thrd=0, reas_iter=1, gt_orders=None):
        '''
        Forward pass to obtain label, confidence, amodal bbox, segmentations
        :param org_x:                   image patch
        :param bboxes:                  bounding box proposals --> [N, 4] numpy array
        :param bbox_type:               bounding box type: 'inmodal' or 'amodal'
        :param input_label:             input label --> [N, 1] obj labels or None
        :param crop_pad:                padding to the bbox crops --> int
        :param slide_window_stride:     stride for classification sliding window --> int
        :param slide_window_bool:       indicates sliding window partial classification --> boolean
        :return:                        label, confidence, amodal bbox, segmentations of the proposed regions (bboxes)
        '''

        if bbox_type == 'amodal':
            slide_window_bool = False
        elif bbox_type == 'inmodal':
            slide_window_bool = True

        score, center, pred_mixture, pred_amodal_bboxes = self.classify(org_x, bboxes=bboxes, pad_length=crop_pad,
                                                                        slide_window=slide_window_bool,
                                                                        stride=slide_window_stride,
                                                                        gt_labels=input_label,
                                                                        clutter_weight=clutter_weight)

        pred_labels = np.argmax(score, axis=1)
        # score_exp = np.exp(score - score.max(axis=1).reshape(-1, 1))
        pred_confidence = np.max(score, axis=1) / np.sum(score, axis=1)

        if bbox_type == 'amodal':
            amodal_bboxes = bboxes
        elif bbox_type == 'inmodal':
            amodal_bboxes = pred_amodal_bboxes
        else:
            amodal_bboxes = None

        if not reasoning:
            pred_segmentations, pred_clss, pred_cls_scores = self.segment(org_x, bboxes=amodal_bboxes,
                                                                          labels=pred_labels, mixture=pred_mixture,
                                                                          pad_length=crop_pad,
                                                                          filter_param=filter_param,
                                                                          clutter_weight=clutter_weight,
                                                                          filtration=True)

            for idx in range(len(bboxes)):
                pred_segmentations[idx]['occ'] = (pred_segmentations[idx]['amodal'] - pred_segmentations[idx][
                    'inmodal']) * (
                                                         pred_segmentations[idx]['amodal'] > 0).astype(float) * (
                                                         pred_segmentations[idx]['amodal'] - pred_segmentations[idx][
                                                     'inmodal'] >= 0).astype(float)

            return pred_labels, pred_confidence, pred_amodal_bboxes, pred_segmentations, None

        # Post-processing: multi-obj occlusion reasoning
        else:
            for iter in range(0, reas_iter):
                pred_segmentations, pred_clss, pred_cls_scores = self.segment(org_x, bboxes=amodal_bboxes,
                                                                              labels=pred_labels, mixture=pred_mixture,
                                                                              pad_length=crop_pad,
                                                                              clutter_weight=clutter_weight,
                                                                              filtration=False)

                pred_order_matrix = np.zeros((len(bboxes), len(bboxes)), dtype=int)

                occ_priors = dict()
                occ_priors_score = dict()
                for idx in range(len(bboxes)):
                    occ_priors[idx] = np.zeros(org_x.shape[2:], dtype='uint8')
                    occ_priors_score[idx] = np.zeros(org_x.shape[2:], dtype='float')

                # pair-wise check
                for idx1 in range(len(bboxes)):
                    # # only for correctly classified objects
                    # if pred_labels[idx1] != gt_labels[idx1]:
                    #     continue

                    pred_inmodal_seg_1 = pred_segmentations[idx1]['inmodal'] > min_bthrd
                    pred_cls_score_1 = pred_segmentations[idx1]['pixel_cls_score']
                    # pred_cls_score_1 /= np.mean(pred_cls_score_1[pred_cls_score_1 > 0])

                    for idx2 in range(idx1 + 1, len(bboxes)):
                        # # only for correctly classified objects
                        # if pred_labels[idx2] != gt_labels[idx2]:
                        #     continue

                        pred_inmodal_seg_2 = pred_segmentations[idx2]['inmodal'] > min_bthrd
                        pred_cls_score_2 = pred_segmentations[idx2]['pixel_cls_score']
                        # pred_cls_score_2 /= np.mean(pred_cls_score_2[pred_cls_score_2 > 0])

                        # condition 1: competing fg_score
                        fg_compete_pixels = pred_inmodal_seg_1 * pred_inmodal_seg_2
                        # visual_compete_pixel(fg_compete_pixels, idx1, idx2)

                        if np.sum(fg_compete_pixels):
                            redef_pixels_1 = (
                                        fg_compete_pixels * (pred_cls_score_2 > pred_cls_score_1 + diff_thrd)).astype(
                                bool)
                            redef_pixels_2 = (
                                        fg_compete_pixels * (pred_cls_score_1 > pred_cls_score_2 + diff_thrd)).astype(
                                bool)

                            # if gt_orders is not None:
                            if True:
                                if np.sum(redef_pixels_1) > np.sum(redef_pixels_2):
                                # if gt_orders[idx1] > gt_orders[idx2]:
                                    # if iter == reas_iter - 1:
                                    # redef_pixels_1 = fg_compete_pixels
                                    # redef_pixels_2 *= False
                                    pred_order_matrix[idx1][idx2] = -1

                                if np.sum(redef_pixels_2) > np.sum(redef_pixels_1):
                                # if gt_orders[idx1] < gt_orders[idx2]:
                                    # if iter == reas_iter - 1:
                                    # redef_pixels_2 = fg_compete_pixels
                                    # redef_pixels_1 *= False
                                    pred_order_matrix[idx1][idx2] = 1

                                pred_order_matrix[idx2][idx1] = - pred_order_matrix[idx1][idx2]

                            # occlusion priors
                            occ_priors[idx1][redef_pixels_1] = 1
                            occ_priors_score[idx1][redef_pixels_1] = pred_cls_score_2[redef_pixels_1] - \
                                                                     pred_cls_score_1[redef_pixels_1]
                            # !! for next competition!
                            pred_segmentations[idx1]['inmodal'][redef_pixels_1] = pred_segmentations[idx2]['inmodal'][
                                redef_pixels_1]

                            occ_priors[idx2][redef_pixels_2] = 1
                            occ_priors_score[idx2][redef_pixels_2] = pred_cls_score_1[redef_pixels_2] - \
                                                                     pred_cls_score_2[redef_pixels_2]
                            pred_segmentations[idx2]['inmodal'][redef_pixels_2] = pred_segmentations[idx1]['inmodal'][
                                redef_pixels_2]

                # # object clutter id
                # obj_clt_id = np.arange(len(bboxes), dtype=int)
                #
                # for idx1 in range(len(bboxes)):
                #     pred_inmodal_seg_1 = pred_segmentations[idx1]['inmodal'] > min_bthrd
                #     for idx2 in range(idx1 + 1, len(bboxes)):
                #         pred_inmodal_seg_2 = pred_segmentations[idx2]['inmodal'] > min_bthrd
                #
                #         if np.sum(pred_inmodal_seg_1 * pred_inmodal_seg_2):
                #             obj_clt_id[idx2] = min(obj_clt_id[idx2], obj_clt_id[idx1])
                #             obj_clt_id[idx1] = min(obj_clt_id[idx2], obj_clt_id[idx1])
                #
                # # order id
                # obj_clutter_ids = np.unique(obj_clt_id)
                # for clt_id in obj_clutter_ids:
                #     obj_ids = np.argwhere(obj_clt_id == clt_id).reshape(-1)
                #     obj_ids = np.sort(obj_ids)
                #
                #     graph = dict()
                #     in_degree = dict()
                #     for idx in obj_ids:
                #         graph[idx] = []
                #         in_degree[idx] = 0
                #
                #     for idx1 in obj_ids:
                #         for idx2 in obj_ids:
                #             if idx2 > idx1:
                #                 pred_inmodal_seg_1 = pred_segmentations[idx1]['inmodal'] > min_bthrd
                #                 pred_cls_score_1 = pred_segmentations[idx1]['pixel_cls_score']
                #                 pred_inmodal_seg_2 = pred_segmentations[idx2]['inmodal'] > min_bthrd
                #                 pred_cls_score_2 = pred_segmentations[idx2]['pixel_cls_score']
                #
                #                 # condition 1: competing fg_score
                #                 fg_compete_pixels = pred_inmodal_seg_1 * pred_inmodal_seg_2
                #
                #                 if np.sum(fg_compete_pixels):
                #                     redef_pixels_1 = (fg_compete_pixels * (
                #                                         pred_cls_score_2 > pred_cls_score_1 + diff_thrd)).astype(bool)
                #                     redef_pixels_2 = (fg_compete_pixels * (
                #                                         pred_cls_score_1 > pred_cls_score_2 + diff_thrd)).astype(bool)
                #
                #                     occ_priors[idx1][redef_pixels_1] = 1
                #                     occ_priors_score[idx1][redef_pixels_1] = pred_cls_score_2[redef_pixels_1] - \
                #                                                              pred_cls_score_1[redef_pixels_1]
                #
                #                     occ_priors[idx2][redef_pixels_2] = 1
                #                     occ_priors_score[idx2][redef_pixels_2] = pred_cls_score_1[redef_pixels_2] - \
                #                                                              pred_cls_score_2[redef_pixels_2]
                #
                #                     if np.sum(redef_pixels_1) < np.sum(redef_pixels_2):
                #                         # add edge
                #                         graph[idx1].append(idx2)
                #                         in_degree[idx2] += 1
                #                     else:
                #                         # add edge
                #                         graph[idx2].append(idx1)
                #                         in_degree[idx1] += 1
                #     # end of topo graph
                #
                #     # order recovery
                #     cur_order_id = -1
                #     unchecked = obj_ids.copy()
                #
                #     while len(unchecked) > 0:
                #         cur_order_id += 1
                #         cur_order_objs = []
                #         for obj in unchecked:
                #             if in_degree[obj] == 0:
                #                 cur_order_objs.append(obj)
                #                 pred_occ_order[obj] = cur_order_id
                #                 unchecked = np.delete(unchecked, np.argwhere(unchecked == obj))
                #
                #         if not cur_order_objs:
                #             print('Iter: {}    ERROR: circle in topo graph detected.'.format(iter))
                #             break
                #
                #         for obj in cur_order_objs:
                #             for obj_2 in graph[obj]:
                #                 in_degree[obj_2] -= 1

                # re-classify
                score, center, pred_mixture, pred_amodal_bboxes = self.classify(org_x, bboxes=bboxes,
                                                                                pad_length=crop_pad,
                                                                                slide_window=slide_window_bool,
                                                                                stride=slide_window_stride,
                                                                                gt_labels=input_label,
                                                                                occ_priors=occ_priors,
                                                                                clutter_weight=clutter_weight)

                pred_labels = np.argmax(score, axis=1)
                pred_confidence = np.max(score, axis=1) / np.sum(score, axis=1)

                if bbox_type == 'amodal':
                    amodal_bboxes = bboxes
                elif bbox_type == 'inmodal':
                    amodal_bboxes = pred_amodal_bboxes
                else:
                    amodal_bboxes = None
            # end of iteration

            # segment with occ_prior from reasoning
            pred_segmentations, pred_clss, pred_cls_scores = self.segment(org_x, bboxes=amodal_bboxes,
                                                                          labels=pred_labels, mixture=pred_mixture,
                                                                          pad_length=crop_pad,
                                                                          filter_param=filter_param,
                                                                          clutter_weight=clutter_weight,
                                                                          filtration=True,
                                                                          occ_prior_score=occ_priors_score)

            # for idx in range(len(bboxes)):
            #     pred_segmentations[idx]['occ'] = (pred_segmentations[idx]['amodal'] - pred_segmentations[idx][
            #         'inmodal']) * (pred_segmentations[idx]['amodal'] > 0).astype(float) * (
            #                                              pred_segmentations[idx]['amodal'] - pred_segmentations[idx][
            #                                          'inmodal'] >= 0).astype(float)

            # filtration?

            # # score centroid
            # blob = occ_priors_score[idx].copy()
            # b_index = np.argwhere(np.zeros((blob.shape[0], blob.shape[1])) == 0).reshape(blob.shape[0], blob.shape[1], 2) + 1
            # # blob_center = [np.sum(blob * b_index[:, :, 0])/blob.shape[0],
            # #                np.sum(blob * b_index[:, :, 1])/blob.shape[1]]
            # blob_center = [np.sum(blob.sum(1) * np.arange(1, blob.shape[0] + 1)) / blob.sum(),
            #                np.sum(blob.sum(0) * np.arange(1, blob.shape[1] + 1)) / blob.sum()]
            # blob_center[0], blob_center[1] = blob_center[0].astype(int), blob_center[1].astype(int)
            #

            # # infer occ order
            # for i in range(len(bboxes)):
            #     for j in range(i + 1, len(bboxes)):
            #         if np.sum(pred_segmentations[i]['amodal'] * pred_segmentations[j]['amodal']) == 0:  # no occlusion
            #             continue
            #         occ_ij = np.sum(pred_segmentations[i]['inmodal'] * pred_segmentations[j]['amodal'])
            #         occ_ji = np.sum(pred_segmentations[j]['inmodal'] * pred_segmentations[i]['amodal'])
            #         if occ_ji == 0 and occ_ij == 0:
            #             continue
            #         pred_order_matrix[i, j] = 1 if occ_ij > occ_ji else -1
            #         pred_order_matrix[j, i] = - pred_order_matrix[i, j]

        # # debug
        # if pred_occ_order[0] != 1 or pred_occ_order[1] != 0:
        #     print("debug: incorrect predicted occ order")

        return pred_labels, pred_confidence, pred_amodal_bboxes, pred_segmentations, pred_order_matrix

    def classify(self, org_x, bboxes, pad_length=16, slide_window=True, stride=1, gt_labels=None, occ_priors=None,
                 clutter_weight=1.0):
        '''
        Full Box / Partial classification of the proposed regions
        :param org_x:           image patch
        :param bboxes:          bounding box proposals --> [N, 4] numpy array
        :param pad_length:      padding to the bbox crops --> int
        :param slide_window:    indicates sliding window partial classification --> boolean
        :param stride:          stride for classification sliding window --> int
        :param gt_labels:       input label --> [N, 1] obj labels or None
        :param occ_priors:      classify with an occlusion prior, used in post-processing after reasoning multi occlusion

        :return:                score, center, mixture, amodal_bboxes
        '''
        with torch.no_grad():

            score = np.ones((bboxes.shape[0], len(self.fg_models))) * -1
            center = np.ones((bboxes.shape[0], len(self.fg_models), 2)) * -1
            mixture = np.ones((bboxes.shape[0], len(self.fg_models))) * -1
            amodal_bboxes = np.ones((bboxes.shape[0], 4)) * -1

            for ii, box in enumerate(bboxes):
                x = copy.copy(org_x)

                # calculate rescale factors
                factor = 224 / (box[2] - box[0])

                # pad and crop ground truth box
                pad = max(0, min(int(pad_length / factor), box[0] - 0, box[1] - 0, org_x.shape[2] - box[2],
                                 org_x.shape[3] - box[3]))
                x = x[:, :, box[0] - pad: box[2] + pad, box[1] - pad: box[3] + pad]

                temp_occ_prior = None
                if occ_priors:
                    temp_occ_prior = occ_priors[ii].copy()
                    temp_occ_prior = temp_occ_prior[box[0] - pad: box[2] + pad, box[1] - pad: box[3] + pad]

                # rescale padded box to height 224
                x = F.interpolate(x, scale_factor=factor, recompute_scale_factor=True)

                # Obtain vc activation
                x = self.backbone(x)

                # Obtain outlier activation
                clutter_match = torch.log(self.clutter_conv1o1(x, norm=False) + 1e-6)
                clutter_match, _ = clutter_match.max(1)
                # clutter_match, _ = clutter_match[:, 1:, :, :].max(1)
                # clutter_match = clutter_match[:, 0, :, :]

                # rescale occ_prior
                if occ_priors:
                    temp_occ_prior = np.array([cv2.resize(temp_occ_prior, (
                        int(x.shape[3]), int(x.shape[2])), interpolation=cv2.INTER_AREA)])

                    # holefill
                    for i in range(temp_occ_prior.shape[1]):
                        for j in range(temp_occ_prior.shape[2]):

                            if temp_occ_prior[0, i, j] == 1:
                                if temp_occ_prior[0, 0:i, j].any() == False and \
                                        temp_occ_prior[0, i + 1:, j].any() == False and \
                                        temp_occ_prior[0, i, 0:j].any() == False and \
                                        temp_occ_prior[0, i, j + 1:].any() == False:
                                    temp_occ_prior[0, i, j] = 0

                            else:
                                if temp_occ_prior[0, 0:i, j].any() and \
                                        temp_occ_prior[0, i + 1:, j].any() and \
                                        temp_occ_prior[0, i, 0:j].any() and \
                                        temp_occ_prior[0, i, j + 1:].any():
                                    temp_occ_prior[0, i, j] = 1

                    temp_occ_prior = torch.from_numpy(temp_occ_prior).type(torch.BoolTensor).cuda(device_ids[0])

                if gt_labels is None:
                    cat_to_eval = [i for i in range(len(self.fg_models))]
                else:
                    cat_to_eval = [gt_labels[ii]]

                for category in cat_to_eval:

                    fused_model = self.fused_models[category]
                    if fused_model is None:
                        continue

                    if slide_window:

                        if x.shape[2] > fused_model.shape[2] or x.shape[3] > fused_model.shape[3]:
                            crop_h = min(x.shape[2], fused_model.shape[2])
                            crop_w = min(x.shape[3], fused_model.shape[3])
                            temp_x = self.center_crop(x, [crop_h, crop_w])
                            temp_clutter_match = self.center_crop(clutter_match, [crop_h, crop_w])

                            new_temp_occ_prior = None
                            if occ_priors:
                                new_temp_occ_prior = self.center_crop(temp_occ_prior, [crop_h, crop_w])

                        else:
                            temp_x = x
                            temp_clutter_match = clutter_match
                            new_temp_occ_prior = temp_occ_prior

                        best_val, best_center, best_k = self.slide_window_on_model(temp_x, fused_model,
                                                                                   temp_clutter_match,
                                                                                   stride=stride,
                                                                                   occ_prior=new_temp_occ_prior,
                                                                                   clutter_weight=clutter_weight)

                        score[ii][category] = best_val
                        center[ii][category] = best_center
                        mixture[ii][category] = best_k
                    else:

                        best_k, best_val = self.get_best_mixture(x, clutter_match, category=category,
                                                                 clutter_weight=clutter_weight,
                                                                 occ_prior=temp_occ_prior)

                        score[ii][category] = best_val
                        center[ii][category] = [int(fused_model.shape[2] / 2), int(fused_model.shape[3] / 2)]
                        mixture[ii][category] = best_k

                # Estimate amodal box
                pred_label = np.argmax(score[ii])
                pred_center = center[ii][pred_label]
                pred_mixture = mixture[ii][pred_label]

                fg_prior = self.fg_prior[pred_label][int(pred_mixture)]
                model_h, model_w = fg_prior.shape

                if x.shape[2] > model_h or x.shape[3] > model_w:
                    crop_h = min(x.shape[2], model_h)
                    crop_w = min(x.shape[3], model_w)
                    temp_x = self.center_crop(x, [crop_h, crop_w])
                    temp_clutter_match = self.center_crop(clutter_match, [crop_h, crop_w])

                else:
                    temp_x = x
                    temp_clutter_match = clutter_match

                pred_score_map = np.zeros((model_h, model_w))

                h = int(pred_center[0] - int(temp_x.shape[2] / 2))
                w = int(pred_center[1] - int(temp_x.shape[3] / 2))

                fg_match = torch.log(fg_prior[h:h + temp_x.shape[2], w:w + temp_x.shape[3]] * (
                        temp_x * self.fg_models[pred_label][:, :, h:h + temp_x.shape[2], w:w + temp_x.shape[3]][
                    int(pred_mixture)]).sum(1) + 1e-6)
                bg_match = torch.log((1 - fg_prior)[h:h + temp_x.shape[2], w:w + temp_x.shape[3]] * (temp_x *
                                                                                                     self.context_models[
                                                                                                         pred_label][:,
                                                                                                     :,
                                                                                                     h:h + temp_x.shape[
                                                                                                         2],
                                                                                                     w:w + temp_x.shape[
                                                                                                         3]][int(
                                                                                                         pred_mixture)]).sum(
                    1) + 1e-6)
                clutter_match = torch.log(
                    fg_prior[h:h + temp_x.shape[2], w:w + temp_x.shape[3]] * torch.exp(temp_clutter_match) + 1e-6)

                context_b = (bg_match - clutter_match > 0).type(torch.FloatTensor).cuda(device_ids[0])
                fg_counter_match = clutter_match * (1 - context_b) + bg_match * context_b

                pred_score_map[h:h + temp_x.shape[2], w:w + temp_x.shape[3]] = (
                        fg_match - fg_counter_match).cpu().detach().numpy()

                amodal_bboxes[ii] = self.predict_amodal_box(box, pred_center, fg_prior.cpu().detach().numpy().squeeze(),
                                                            factor, image_size=org_x.shape[2:])

            return score, center, mixture.astype(int), amodal_bboxes.astype(int)

    def segment(self, org_x, bboxes, labels, mixture, pad_length=16,
                filtration=True, clutter_weight=1.0, filter_param=[0.95, 33], occ_prior_score=None):
        '''
        Segmentation of foreground ("1"), context ("2"), and occluder ("0")
        :param org_x:           image patch
        :param bboxes:          amodal bounding box --> [N, 4] numpy array
        :param labels:          labels for each box --> [N, 1] numpy array
        :param mixture:         best pose for each box --> [N, 1] numpy array
        :param modes:           evaluation modes --> ('inmodal', 'occ', 'amodal')
        :param pad_length:      padding to the bbox crops --> int
        :return:                segmentations of each proposed region --> list of dicts
        '''
        with torch.no_grad():
            segmentations, pixel_clss, pixel_cls_scores = [], [], []

            for ii, box in enumerate(bboxes):
                #  Pre-process image patch
                if box[2] - box[0] < 10:
                    box[0] = 0
                    box[2] = org_x.shape[2]
                patch_center = np.array([(box[2] + box[0]) / 2, (box[3] + box[1]) / 2])
                category = labels[ii]
                k_max = mixture[ii][category]

                factor = 224 / (box[2] - box[0])
                restore_factor = feat_stride / factor

                pad = max(0, min(int(pad_length / factor), box[0] - 0, box[1] - 0, org_x.shape[2] - box[2],
                                 org_x.shape[3] - box[3]))
                x = org_x[:, :, box[0] - pad: box[2] + pad, box[1] - pad: box[3] + pad]

                occ_score = None
                if occ_prior_score:
                    occ_score = occ_prior_score[ii].copy()
                    occ_score = occ_score[box[0] - pad: box[2] + pad, box[1] - pad: box[3] + pad]

                x = F.interpolate(x, scale_factor=factor, recompute_scale_factor=True)

                x = self.backbone(x)

                # rescale occ_prior
                if occ_prior_score:
                    occ_score = np.array([cv2.resize(occ_score, (
                        int(x.shape[3]), int(x.shape[2])), interpolation=cv2.INTER_AREA)])
                    occ_score = np.where(occ_score > 0.1, occ_score, 0)

                    # holefill
                    for i in range(occ_score.shape[1]):
                        for j in range(occ_score.shape[2]):

                            if occ_score[0, i, j] > 0:
                                if occ_score[0, 0:i, j].any() == False and \
                                        occ_score[0, i + 1:, j].any() == False and \
                                        occ_score[0, i, 0:j].any() == False and \
                                        occ_score[0, i, j + 1:].any() == False:
                                    occ_score[0, i, j] = 0

                            else:
                                if occ_score[0, 0:i, j].any() and \
                                        occ_score[0, i + 1:, j].any() and \
                                        occ_score[0, i, 0:j].any() and \
                                        occ_score[0, i, j + 1:].any():
                                    occ_score[0, i, j] = (occ_score[0, i].mean() + occ_score[0, :, j].mean()) / 2

                    occ_score = torch.from_numpy(occ_score).type(torch.FloatTensor).cuda(device_ids[0])

                fg_prior = self.center_crop(self.fg_prior[category], x.shape[2:])  # used for amodal segmentaiton

                _, model_h, model_w = fg_prior.shape

                # Generate Per-Pixel Classification
                pixel_cls_, pixel_cls_score = self.segment_per_pixel_cls(x, category=category, k_max=k_max,
                                                                         clutter_weight=clutter_weight,
                                                                         occ_prior_score=occ_score)

                # Postprocess Per-Pixel Classification
                pixel_cls = copy.deepcopy(pixel_cls_)
                self.binary_mask_post_process(pixel_cls[0])
                # # TODO: why = 0?
                # pixel_cls_score[(pixel_cls - pixel_cls_ > 0)] = 0

                pixel_clss.append(pixel_cls)
                pixel_cls_scores.append(pixel_cls_score)
                object_seg = dict()

                object_seg['pixel_cls'] = self.place_mask_in_image(mask=pixel_cls.cpu().detach().numpy().squeeze(),
                                                                   mask_dim_in_img=[model_h * restore_factor,
                                                                                    model_w * restore_factor],
                                                                   center_in_img=patch_center,
                                                                   org_img_shape=org_x.shape[2:4], empty_fill=-1)

                object_seg['pixel_cls_score'] = self.place_mask_in_image(
                    mask=pixel_cls_score.cpu().detach().numpy().squeeze(),
                    mask_dim_in_img=[model_h * restore_factor, model_w * restore_factor],
                    center_in_img=patch_center, org_img_shape=org_x.shape[2:4], empty_fill=-1)

                binary_resp = self.binarize_pixel_cls(pixel_cls, pixel_cls_score, target_label=1)
                binary_resp_in_img = self.place_mask_in_image(mask=binary_resp.cpu().detach().numpy().squeeze(),
                                                              mask_dim_in_img=[model_h * restore_factor,
                                                                               model_w * restore_factor],
                                                              center_in_img=patch_center,
                                                              org_img_shape=org_x.shape[2:4], empty_fill=-1)

                object_seg['inmodal'] = self.cv2_mask_post_process(binary_resp_in_img,
                                                                   filtering=filtration,
                                                                   # iou_converge_thrd=filter_param[0],
                                                                   gaussian_mask_size=filter_param[1])

                # TODO
                pixel_cls = self.add_fg_prior(pixel_cls, fg_prior[k_max], thrd=0.9)
                binary_resp = self.binarize_pixel_cls(pixel_cls, pixel_cls_score, target_label=1)
                binary_resp_in_img = self.place_mask_in_image(mask=binary_resp.cpu().detach().numpy().squeeze(),
                                                              mask_dim_in_img=[model_h * restore_factor,
                                                                               model_w * restore_factor],
                                                              center_in_img=patch_center,
                                                              org_img_shape=org_x.shape[2:4], empty_fill=-1)

                object_seg['amodal'] = self.cv2_mask_post_process(binary_resp_in_img,
                                                                  filtering=filtration,
                                                                  # iou_converge_thrd=filter_param[0],
                                                                  gaussian_mask_size=filter_param[1])

                # object_seg['occ'] = (object_seg['amodal'] - object_seg['inmodal']) * (
                #       object_seg['amodal'] - object_seg['inmodal'] >= 0).astype(float)

                object_seg['occ'] = (object_seg['amodal'] - object_seg['inmodal']) * (
                                    object_seg['amodal'] > 0).astype(float)

                segmentations.append(object_seg)
            return segmentations, pixel_clss, pixel_cls_scores

    def place_mask_in_image(self, mask, mask_dim_in_img, center_in_img, org_img_shape, empty_fill=-1):
        segmentation = np.ones(org_img_shape) * empty_fill

        mask_as_img = cv2.resize(mask, (int(mask_dim_in_img[1]), int(mask_dim_in_img[0])),
                                 interpolation=cv2.INTER_NEAREST)

        mask_h, mask_w = mask_as_img.shape

        top_img, left_img, bottom_img, right_img = [math.floor(center_in_img[0] - mask_h / 2),
                                                    math.floor(center_in_img[1] - mask_w / 2),
                                                    math.floor(center_in_img[0] + mask_h / 2),
                                                    math.floor(center_in_img[1] + mask_w / 2)]
        top_mask, left_mask, bottom_mask, right_mask = [0, 0, mask_h, mask_w]

        if top_img < 0:
            top_mask += 0 - top_img
            top_img = 0

        if left_img < 0:
            left_mask += 0 - left_img
            left_img = 0

        if bottom_img > segmentation.shape[0]:
            bottom_mask += segmentation.shape[0] - bottom_img
            bottom_img = segmentation.shape[0]

        if right_img > segmentation.shape[1]:
            right_mask += segmentation.shape[1] - right_img
            right_img = segmentation.shape[1]

        if bottom_mask - top_mask != bottom_img - top_img:
            height = min(bottom_mask - top_mask, bottom_img - top_img)
            bottom_mask = top_mask + height
            bottom_img = top_img + height

        if right_mask - left_mask != right_img - left_img:
            width = min(right_mask - left_mask, right_img - left_img)
            right_mask = left_mask + width
            right_img = left_img + width

        segmentation[top_img:bottom_img, left_img:right_img] = mask_as_img[top_mask:bottom_mask, left_mask:right_mask]

        return segmentation

    def add_fg_prior(self, per_pixel_cls, prior, thrd=0.5):
        per_pixel_cls += (per_pixel_cls == 0).type(torch.FloatTensor) * (prior > thrd).type(torch.FloatTensor)
        return per_pixel_cls

    def binarize_pixel_cls(self, pixel_cls, pixel_cls_score, target_label=1):
        obj_resp = pixel_cls_score * (pixel_cls == target_label).type(torch.FloatTensor).cuda(device_ids[0])
        non_obj_resp = pixel_cls_score * (pixel_cls != target_label).type(torch.FloatTensor).cuda(device_ids[0])

        return obj_resp - non_obj_resp

    def segment_per_pixel_cls(self, x, category, k_max, clutter_weight=1.0, occ_prior_score=None):
        '''
        Segmentation by per-pixel classification of foreground ("1"), context ("2"), and occluder ("0")
        :param fg_match:            image patch explained by foreground model
        :param bg_match:            image patch explained by context model
        :param clutter_match:       image patch explained by clutter model
        :return:                    per_pixel_cls containing labels and per_pixel_cls_score containing log prob ratios --> [H, W, 1]
        '''

        ###     Generate Foreground Activation
        fg_model = self.center_crop(self.fg_models[category], x.shape[2:])
        fg_prior = self.center_crop(self.fg_prior[category], x.shape[2:])
        fg_match = torch.log(fg_prior[k_max] * (x * fg_model[k_max]).sum(1) + 1e-6)
        # fg_match = torch.log((x * fg_model[k_max]).sum(1) + 1e-6)

        ###     Generate Background Activation
        context_model = self.center_crop(self.context_models[category], x.shape[2:])
        context_prior = 1 - fg_prior
        bg_match = torch.log(context_prior[k_max] * (x * context_model[k_max]).sum(1) + 1e-6)
        # bg_match = torch.log((x * context_model[k_max]).sum(1) + 1e-6)

        ###     Generate Outlier Activation
        clutter_match = torch.log(self.clutter_conv1o1(x, norm=False) + 1e-6)
        clutter_match, _ = clutter_match.max(1)
        # clutter_match = torch.log(fg_prior[k_max] * torch.exp(clutter_match) + 1e-6) * clutter_weight
        clutter_match = torch.log((torch.exp(clutter_match) + 1e-6) * clutter_weight)

        ###     Generate Counter Activations
        # TODO
        # context_b = (bg_match > 0).type(torch.FloatTensor).cuda(device_ids[0])
        context_b = (bg_match - clutter_match > 0).type(torch.FloatTensor).cuda(device_ids[0])
        fg_counter_match = clutter_match * (1 - context_b) + bg_match * context_b

        # fg_b = (fg_match > 0).type(torch.FloatTensor).cuda(device_ids[0])
        fg_b = (fg_match - clutter_match > 0).type(torch.FloatTensor).cuda(device_ids[0])
        bg_counter_match = clutter_match * (1 - fg_b) + fg_match * fg_b

        fg_b = (fg_match - bg_match > 0).type(torch.FloatTensor).cuda(device_ids[0])
        clutter_counter_match = bg_match * (1 - fg_b) + fg_match * fg_b

        ###     per-pixel classification of foreground, context, and occluder
        per_pixel_cls = torch.zeros(fg_match.shape)  # occluder
        per_pixel_cls[fg_match > fg_counter_match] = 1  # object foreground
        # per_pixel_cls[(fg_match > fg_counter_match) & (fg_match > np.percentile(fg_match.cpu(), 10))] = 1  # object foreground
        per_pixel_cls[bg_match > bg_counter_match] = 2  # context background

        per_pixel_cls_score = (clutter_match - clutter_counter_match) * (per_pixel_cls == 0).type(
            torch.FloatTensor).cuda(device_ids[0]) \
                              + (fg_match - fg_counter_match) * (per_pixel_cls == 1).type(torch.FloatTensor).cuda(
            device_ids[0]) \
                              + (bg_match - bg_counter_match) * (per_pixel_cls == 2).type(torch.FloatTensor).cuda(
            device_ids[0])

        if occ_prior_score is not None:
            per_pixel_cls[occ_prior_score > 0] = 0
            per_pixel_cls_score[occ_prior_score > 0] = occ_prior_score[occ_prior_score > 0]

        return per_pixel_cls, per_pixel_cls_score

    def predict_amodal_box(self, box, obj_center_on_model, fg_prior, factor, image_size):
        '''
        Predict Amodal box: amodal completion on the bounding box level
        :param box:                     the partial bounding box information
        :param obj_center_on_model:     the center obtained from sldiing window
        :param fg_prior:                the foreground mask prior of the given object/pose
        :param factor:                  scaling factor to obtain the new patch
        :param image_size:              the size of the image
        :return:                        the estiamted amodal bounding box
        '''

        model_h, model_w = fg_prior.shape
        restore_factor = feat_stride / factor

        inmodal_ctr = np.array([(box[2] + box[0]) / 2, (box[3] + box[1]) / 2])
        inmodal_size = np.array([box[2] - box[0], box[3] - box[1]])

        off_set = np.array(
            [model_h / 2 - obj_center_on_model[0], model_w / 2 - obj_center_on_model[1]]) * restore_factor

        true_ctr = off_set + inmodal_ctr

        relative_true_ctr = off_set + inmodal_size / 2

        pred_amodal_height = max(abs(relative_true_ctr[0] - 0), abs(relative_true_ctr[0] - inmodal_size[0])) * 2
        pred_amodal_width = max(abs(relative_true_ctr[1] - 0), abs(relative_true_ctr[1] - inmodal_size[1])) * 2

        visible_obj_x, visible_obj_y = np.where(fg_prior > 0.5)

        mean_amodal_height = (np.max(visible_obj_x) - np.min(visible_obj_x)) * restore_factor
        mean_amodal_width = (np.max(visible_obj_y) - np.min(visible_obj_y)) * restore_factor

        if pred_amodal_height < mean_amodal_height and pred_amodal_width < mean_amodal_width:
            pred_amodal_height = mean_amodal_height
            pred_amodal_width = mean_amodal_width

        tlx = max(true_ctr[0] - pred_amodal_height / 2, 0)
        tly = max(true_ctr[1] - pred_amodal_width / 2, 0)
        brx = min(tlx + pred_amodal_height, image_size[0])
        bry = min(tly + pred_amodal_width, image_size[1])

        amodal_box = np.array([tlx, tly, brx, bry])
        return amodal_box

    def predict_amodal_box_old(self, pred_score_map, box, obj_center_on_model, fg_prior, factor, image_size):
        '''
        Predict Amodal box: amodal completion on the bounding box level
        :param pred_score_map:          the foreground score map of the region
        :param box:                     the partial bounding box information
        :param obj_center_on_model:     the center obtained from sldiing window
        :param fg_prior:                the foreground mask prior of the given object/pose
        :param factor:                  scaling factor to obtain the new patch
        :param image_size:              the size of the image
        :return:                        the estiamted amodal bounding box
        '''
        patch_center = np.array([(box[2] + box[0]) / 2, (box[3] + box[1]) / 2])
        model_h, model_w = fg_prior.shape
        restore_factor = feat_stride / factor

        visible_obj_x, visible_obj_y = np.where(pred_score_map > 0)

        if visible_obj_x.shape[0] == 0 and visible_obj_y.shape[0] == 0:
            visible_obj_x, visible_obj_y = np.where(fg_prior > 0.7)

        amodal_height = 2 * max(np.max(visible_obj_x) - model_h / 2,
                                model_h / 2 - np.min(visible_obj_x)) * restore_factor
        amodal_width = 2 * max(np.max(visible_obj_y) - model_w / 2,
                               model_w / 2 - np.min(visible_obj_y)) * restore_factor

        off_set = np.array(
            [model_h / 2 - obj_center_on_model[0], model_w / 2 - obj_center_on_model[1]]) * restore_factor
        center_in_img = patch_center + off_set

        tlx = max(center_in_img[0] - amodal_height / 2, 0)
        tly = max(center_in_img[1] - amodal_width / 2, 0)
        brx = min(tlx + amodal_height, image_size[0])
        bry = min(tly + amodal_width, image_size[1])

        amodal_box = np.array([tlx, tly, brx, bry])
        return amodal_box

    def backbone(self, x):
        x = self.extractor(x)
        x = self.vc_conv1o1(x)
        x = self.exp(x)
        return x

    def get_vc_activation(self, x, low_res=False):
        '''
        Obtain the VC (vMF) activation of the image patch x
        :param x:           the image patch - bbox short side resize to 224
        :param low_res:     boolean indicating if the x should reduce in resolution
        :return:            the VC activation --> [vc_num, H, W] numpy array
        '''
        with torch.no_grad():
            if low_res:
                x = F.interpolate(x, scale_factor=0.25)
                x = F.interpolate(x, scale_factor=4)
            x = self.backbone(x)
            return x.cpu().detach().numpy().squeeze()

    def get_vc_activation_with_binary_fg_mask(self, x, gt_category, use_context_center=False, use_mixture_model=False,
                                              context_thrd=None, mmodel_thrd=None, bmask_post_process=False,
                                              cntxt_pad=0):
        '''
        Perform foreground disentanglement given an image patch x, used for building Context-Aware CompNets
        :param x:                   the image patch image patch - bbox short side resize to 224
        :param gt_category:         the ground truth label of x
        :param use_context_center:  boolean that indicates if should use context center to segment foreground
        :param use_mixture_model:   boolean that indicates if should use mixture model to segment foreground
        :param context_thrd:        float thrd that determines the binary mask for context center
        :param mmodel_thrd:         float thrd that determines the binary mask for mixture model
        :return:                    a binary mask as a pixel-level classification of foreground in the feature map --> [H, W] numpy array
        '''
        assert use_context_center ^ use_mixture_model

        with torch.no_grad():

            x = self.extractor(x)

            if use_context_center:
                assert context_thrd != None

                context_resp = self.get_context_response(x, category=gt_category)

                binary_fg_mask = (context_resp <= context_thrd).type(torch.FloatTensor)

                x = self.vc_conv1o1(x)
                x = self.exp(x)

            if use_mixture_model:
                assert mmodel_thrd != None

                x = self.vc_conv1o1(x)
                x = self.exp(x)

                fg_model = self.center_crop(self.fg_models[gt_category], x.shape[2:])
                fg_prior = self.center_crop(self.fg_prior[gt_category], x.shape[2:])
                context_model = self.center_crop(self.context_models[gt_category], x.shape[2:])
                context_prior = 1 - fg_prior

                clutter_match = torch.log(self.clutter_conv1o1(x, norm=False) + 1e-6)
                clutter_match = clutter_match[:, 0, :, :]

                k_max, _ = self.get_best_mixture(x, clutter_match, category=gt_category)

                fg_match = torch.log(fg_prior[k_max] * (x * fg_model[k_max]).sum(1) + 1e-6)
                bg_match = torch.log(context_prior[k_max] * (x * context_model[k_max]).sum(1) + 1e-6)
                clutter_match = torch.zeros(fg_match.shape).cuda(device_ids[0])

                context_b = (bg_match - clutter_match > 0).type(torch.FloatTensor).cuda(device_ids[0])
                baseline = clutter_match * (1 - context_b) + bg_match * context_b

                fg_resp = fg_match - baseline

                binary_fg_mask = (fg_resp > mmodel_thrd).type(torch.FloatTensor)

            binary_fg_mask = torch.squeeze(binary_fg_mask)

            if bmask_post_process:
                self.binary_mask_post_process(binary_fg_mask, cntxt_pad=cntxt_pad)

            return x.cpu().detach().numpy().squeeze(), binary_fg_mask.cpu().detach().numpy()

    def get_context_response(self, x, category):
        '''
        Obtain the cos similarity between x, layer feature, and context centers
        :param x:           layer feature of image patch
        :param category:    categroy of x (could be predict or ground truth)
        :return:            max_p ( cos similarity between x and context centers ) --> [H, W] torch tensor on GPU
        '''
        with torch.no_grad():
            context_resp = self.context_conv1o1(x)
            context_resp = context_resp[:, category * context_cluster: (category + 1) * context_cluster, :, :]
            context_resp, _ = context_resp.max(1)

        return context_resp

    def get_best_mixture(self, x, clutter_match, category, clutter_weight=1.0, occ_prior=None):
        '''
        Find the best mixture that models x, given label
        :param x:           vMF activation of image patch
        :param category:    categroy of x (could be predict or ground truth)
        :param clutter_weight: weight of clutter match competing fg and bg
        :return:            integer indicating the best mixture
        '''

        fused_model = self.center_crop(self.fused_models[category], x.shape[2:])
        fg_match = torch.log((x * fused_model).sum(1) + 1e-6)

        if occ_prior is None:
            occ_prior = torch.zeros(fg_match.shape).type(torch.BoolTensor).cuda(device_ids[0])

        assert occ_prior.shape[1:] == fg_match.shape[1:], 'occ_prior shape does not match'

        not_occ = ((~occ_prior) * ((fg_match - clutter_match * clutter_weight) > 0)).type(torch.FloatTensor).cuda(
            device_ids[0])

        view_point_scores = (not_occ * (fg_match - clutter_match * clutter_weight)).sum((1, 2))
        k_max = int(torch.argmax(view_point_scores).item())

        return k_max, view_point_scores[k_max]

    def binary_mask_post_process(self, bmask, obj_prior='rigid', cntxt_pad=0):

        h, w = bmask.shape

        if cntxt_pad > 0:
            bmask[0:cntxt_pad, :] = 0
            bmask[:, 0:cntxt_pad] = 0
            bmask[h - cntxt_pad:h, :] = 0
            bmask[:, w - cntxt_pad:w] = 0

        if obj_prior == 'rigid':
            for i in range(h):
                for j in range(w):

                    if bmask[i][j] == 1:
                        continue

                    if torch.any(bmask[0:i, j] == 1) and torch.any(bmask[i + 1:, j] == 1) and torch.any(
                            bmask[i, 0:j] == 1) and torch.any(bmask[i, j + 1:] == 1):
                        bmask[i][j] = 1

    def calc_mask_iou(self, pred_mask, gt_mask):
        return np.sum(pred_mask * gt_mask) / np.sum((pred_mask + gt_mask >= 1.).astype(float))

    def cv2_mask_post_process(self, pixel_cls_score, filtering=True, iou_converge_thrd=0.95, gaussian_mask_size=33):
        refined_pixel_cls_score = pixel_cls_score
        cur_median = np.median(refined_pixel_cls_score)
        old_pixel_cls = refined_pixel_cls_score

        if not filtering:
            return old_pixel_cls

        refined_pixel_cls_score = cv2.GaussianBlur(refined_pixel_cls_score, (gaussian_mask_size, gaussian_mask_size), 0)
        refined_pixel_cls_score -= np.median(refined_pixel_cls_score) - cur_median
        refined_pixel_cls = refined_pixel_cls_score

        while self.calc_mask_iou(old_pixel_cls, refined_pixel_cls) < iou_converge_thrd:
            old_pixel_cls = refined_pixel_cls
            refined_pixel_cls_score = cv2.GaussianBlur(refined_pixel_cls_score,
                                                       (gaussian_mask_size, gaussian_mask_size), 0)
            refined_pixel_cls_score -= np.median(refined_pixel_cls_score) - cur_median
            # refined_pixel_cls = (refined_pixel_cls_score > 0).astype(int)

        return refined_pixel_cls

    def slide_window_on_model(self, x, fused_model, clutter_match, stride=1, occ_prior=None, clutter_weight=1.0):
        '''
        Helper method: slide the vMF activation along the mixture model to look for the region of the mixture model that explains the activation the best
        :param x:               vMF activation of image patch
        :param fused_model:     the model that combines both fg and context model into one
        :param clutter_match:   clutter response on the same image patch
        :param stride:          sliding window step size
        :param occ_prior:      classify with an occlusion prior, used in post-processing after reasoning multi occlusion
        :param clutter_weight: weight of clutter competing bg and fg

        :return:                [ max_value, center of the mmodel that offers best alignment, best mixture (pose) ]
        '''
        best_val = -1
        best_center = [-1, -1]
        best_k = -1

        heat_map = np.zeros((fused_model.shape[2] - x.shape[2] + 1, fused_model.shape[3] - x.shape[3] + 1))
        k_max_map = np.ones((fused_model.shape[2] - x.shape[2] + 1, fused_model.shape[3] - x.shape[3] + 1)) * -1

        for h in range(0, fused_model.shape[2] - x.shape[2] + 1, stride):
            for w in range(0, fused_model.shape[3] - x.shape[3] + 1, stride):

                # fg_match = torch.log((1 - omega) * fg_prior[:, h:h + x.shape[2], w:w + x.shape[3]] * (x * fg_model[:, :, h:h + x.shape[2], w:w + x.shape[3]]).sum(1) + omega * context_prior[:, h:h + x.shape[2], w:w + x.shape[3]] * ( x * context_model[:, :, h:h + x.shape[2], w:w + x.shape[3]]).sum(1) + 1e-6)
                fg_match = torch.log((x * fused_model[:, :, h:h + x.shape[2], w:w + x.shape[3]]).sum(1) + 1e-6)

                if occ_prior is None:
                    occ_prior = torch.zeros(fg_match.shape).type(torch.BoolTensor).cuda(device_ids[0])

                assert occ_prior.shape[1:] == fg_match.shape[1:], 'occ_prior shape does not match'
                not_occ = ((~occ_prior) * ((fg_match - clutter_match * clutter_weight) > 0)).type(
                    torch.FloatTensor).cuda(device_ids[0])
                # not_occ = ((fg_match - clutter_match) > 0).type(torch.FloatTensor).cuda(device_ids[0])

                view_point_scores = (not_occ * (fg_match - clutter_match * clutter_weight)).sum((1, 2))

                k_max = torch.argmax(view_point_scores).item()

                heat_map[h][w] = view_point_scores[k_max]
                k_max_map[h][w] = k_max

                if view_point_scores[k_max] > best_val:
                    best_val = view_point_scores[k_max]
                    best_center = [h + int(x.shape[2] / 2), w + int(x.shape[3] / 2)]
                    best_k = k_max

        if stride <= 1:
            return best_val, np.array(best_center), best_k
        else:
            thrd = np.percentile(heat_map[k_max_map >= 0], 75)

        for h in range(0, fused_model.shape[2] - x.shape[2] + 1):
            for w in range(0, fused_model.shape[3] - x.shape[3] + 1):

                if k_max_map[h][w] >= 0:
                    score = heat_map[h][w]
                    k_max = k_max_map[h][w]
                else:
                    if heat_map[h - h % stride][w - w % stride] > thrd:

                        fg_match = torch.log((x * fused_model[:, :, h:h + x.shape[2], w:w + x.shape[3]]).sum(1) + 1e-6)

                        not_occ = ((fg_match - clutter_match * clutter_weight) > 0).type(torch.FloatTensor).cuda(
                            device_ids[0])

                        view_point_scores = (not_occ * (fg_match - clutter_match * clutter_weight)).sum((1, 2))

                        k_max = torch.argmax(view_point_scores).item()
                        score = view_point_scores[k_max]
                    else:
                        continue

                if score > best_val:
                    best_val = score
                    best_center = [h + int(x.shape[2] / 2), w + int(x.shape[3] / 2)]
                    best_k = k_max

        return best_val, np.array(best_center), best_k

    def center_crop(self, model, dim):
        '''
        Helper method: crop the input model into the input dim with respect to the model's center
        :param model:   model that needs cropping
        :param dim:     cropping dimension
        :return:        cropped model
        '''

        axis_num = len(model.shape)
        h_crop, w_crop = dim

        if axis_num == 2:
            h, w = model.shape

            if h_crop > h or w_crop > w:
                diff = int((max(h_crop - h, w_crop - w) + 1) / 2)
                model = F.pad(model, (diff, diff, diff, diff), 'constant', 0)
                h, w = model.shape

            assert h_crop <= h and w_crop <= w
            return model[int((h - h_crop) / 2): int((h + h_crop) / 2), int((w - w_crop) / 2): int((w + w_crop) / 2)]

        if axis_num == 3:
            _, h, w = model.shape

            if h_crop > h or w_crop > w:
                diff = int((max(h_crop - h, w_crop - w) + 1) / 2)
                model = F.pad(model, (diff, diff, diff, diff, 0, 0), 'constant', 0)
                _, h, w = model.shape

            assert h_crop <= h and w_crop <= w
            return model[:, int((h - h_crop) / 2): int((h + h_crop) / 2), int((w - w_crop) / 2): int((w + w_crop) / 2)]

        if axis_num == 4:
            _, _, h, w = model.shape

            if h_crop > h or w_crop > w:
                diff = int((max(h_crop - h, w_crop - w) + 1) / 2)
                model = F.pad(model, (diff, diff, diff, diff, 0, 0, 0, 0), 'constant', 0)
                _, _, h, w = model.shape

            assert h_crop <= h and w_crop <= w
            return model[:, :, int((h - h_crop) / 2): int((h + h_crop) / 2),
                   int((w - w_crop) / 2): int((w + w_crop) / 2)]

        if axis_num < 2 or axis_num > 4:
            print('center crop operation is only supported for axis_num = [2, 4].')

    def fuse_model(self, fg_model, fg_prior, context_model, context_prior, omega):
        '''
        Helper method: generate fused model based on input omega
        :param fg_model:        Foreground object model
        :param fg_prior:        Foreground object prior
        :param context_model:   Context model
        :param context_prior:   Context prior
        :param omega:           omega used to determine context influence when merging fg and context model
        :return:                return fused singular model per category per mixture
        '''
        # Fused foreground and context templates via priors and omega value
        if fg_model == None:
            return None

        fused_model = (1 - omega) * fg_model * fg_prior.unsqueeze(1) + omega * context_model * context_prior.unsqueeze(
            1)

        return fused_model

    def update_mixture_models(self, Mixture_Models):
        '''
        Setter method: update mixture models
        :param Mixture_Models:      new mixture models
        :return:                    None
        '''

        self.fg_models, self.fg_prior, self.context_models, self.context_prior = Mixture_Models

    def update_fused_models(self, omega=0):
        '''
        Setter Method: update fused models based on input omega
        :param omega:   omega used to determine context influence when merging fg and context model
        :return:        None
        '''

        self.fused_models = []

        for category in range(len(self.fg_models)):
            fg_model = self.fg_models[category]
            fg_prior = self.fg_prior[category]
            context_model = self.context_models[category]
            context_prior = self.context_prior[category]

            self.fused_models.append(self.fuse_model(fg_model, fg_prior, context_model, context_prior, omega=omega))


class Conv1o1Layer(nn.Module):
    def __init__(self, weights):
        super(Conv1o1Layer, self).__init__()
        self.weight = nn.Parameter(weights)

    def forward(self, x, norm=True):
        if norm:
            norm = torch.norm(x, dim=1, keepdim=True)
            x = x / norm
        return F.conv2d(x, self.weight)


class ExpLayer(nn.Module):
    def __init__(self, vMF_kappa):
        super(ExpLayer, self).__init__()
        self.vMF_kappa = nn.Parameter(torch.Tensor([vMF_kappa]))

    def forward(self, x, binary=False):
        if binary:
            x = torch.exp(self.vMF_kappa * x) * (x > 0.55).type(torch.FloatTensor).cuda(device_ids[0])
        else:
            x = torch.exp(self.vMF_kappa * x)
        return x
