from configs import device_ids, dataset_train, dataset_eval, nn_type, vc_num, K, vMF_kappa, context_cluster, layer, exp_dir, categories, feature_num
from configs import *
from model import get_compnet_head
from DataLoader import Occ_Veh_Dataset, KINS_Dataset
from scipy import interpolate
from torch.utils.data import DataLoader
from util import roc_curve, rank_perf, visualize, draw_box, visualize_multi, calc_iou
import copy
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

'''
Evaluate classification and  segmentation performance supervised by given ground truth inmodal or amodal bbox
Currently implemented for OccludedVehicles and KINS dataset
'''


def make_three_dimensional_demo(pixel_cls, pixel_cls_score):

    occ_label = 0
    fg_label = 1
    bg_label = 2

    try:
        occ_range = [np.min(pixel_cls_score[pixel_cls == occ_label]), np.max(pixel_cls_score[pixel_cls == occ_label])]
        if occ_range[1] - occ_range[0] == 0:
            occ_range = [0, 1]
    except:
        occ_range = [0, 1]

    try:
        fg_range = [np.min(pixel_cls_score[pixel_cls == fg_label]), np.max(pixel_cls_score[pixel_cls == fg_label])]
        if fg_range[1] - fg_range[0] == 0:
            fg_range = [0, 1]
    except:
        fg_range = [0, 1]

    try:
        bg_range = [np.min(pixel_cls_score[pixel_cls == bg_label]), np.max(pixel_cls_score[pixel_cls == bg_label])]
        if bg_range[1] - bg_range[0] == 0:
            bg_range = [0, 1]
    except:
        bg_range = [0, 1]


    # treat an rbg image as three layers heatmap

    occ_layer = ( ((pixel_cls == occ_label).astype(float) * pixel_cls_score - occ_range[0]) / (occ_range[1] - occ_range[0]) * 255 ).astype(int)[:, :, np.newaxis]
    fg_layer  = ( ((pixel_cls == fg_label).astype(float) * pixel_cls_score - fg_range[0]) / (fg_range[1] - fg_range[0]) * 255 ).astype(int)[:, :, np.newaxis]
    bg_layer  = ( ((pixel_cls == bg_label).astype(float) * pixel_cls_score - bg_range[0]) / (bg_range[1] - bg_range[0]) * 255 ).astype(int)[:, :, np.newaxis]


    selected = np.argwhere(occ_layer > 0)[:, 0:2]
    occ_layer[selected[:, 0], selected[:, 1]] = (occ_layer[selected[:, 0], selected[:, 1]] + 0.5*255) * 255 / (1.5*255)
    selected = np.argwhere(fg_layer > 0)[:, 0:2]
    fg_layer[selected[:, 0], selected[:, 1]] = (fg_layer[selected[:, 0], selected[:, 1]] + 0.5*255) * 255 / (1.5*255)
    selected = np.argwhere(bg_layer > 0)[:, 0:2]
    bg_layer[selected[:, 0], selected[:, 1]] = (bg_layer[selected[:, 0], selected[:, 1]] + 0.5*255) * 255 / (1.5*255)


    img = np.concatenate((fg_layer, bg_layer, occ_layer), axis=2)

    return img


def pixel_relation_post_proc(pixel_cls, pixel_cls_score, iou_converge_thrd=0.95):
    unseen_label, occ_label, fg_label, bg_label = -1, 0, 1, 2

    obj_resp = pixel_cls_score * (pixel_cls == fg_label).astype(int)
    occ_resp = pixel_cls_score * (pixel_cls == occ_label).astype(int)
    bg_resp = pixel_cls_score * (pixel_cls == bg_label).astype(int)
    unseen_resp = (pixel_cls == unseen_label).astype(int)

    non_obj_resp = np.max(np.concatenate((np.expand_dims(occ_resp, axis=2), np.expand_dims(bg_resp, axis=2), np.expand_dims(unseen_resp, axis=2)), axis=2), axis=2)

    refined_pixel_cls_score = obj_resp - non_obj_resp
    cur_median = np.median(refined_pixel_cls_score)
    old_pixel_cls = (pixel_cls == fg_label).astype(int)


    refined_pixel_cls_score = cv2.GaussianBlur(refined_pixel_cls_score,(33,33),0)
    refined_pixel_cls_score -= np.median(refined_pixel_cls_score) - cur_median
    refined_pixel_cls = (refined_pixel_cls_score > 0).astype(int)

    while calc_mask_iou(old_pixel_cls, refined_pixel_cls) < iou_converge_thrd:
        old_pixel_cls = refined_pixel_cls
        refined_pixel_cls_score = cv2.GaussianBlur(refined_pixel_cls_score, (33, 33), 0)
        refined_pixel_cls_score -= np.median(refined_pixel_cls_score) - cur_median
        refined_pixel_cls = (refined_pixel_cls_score > 0).astype(int)

    return refined_pixel_cls


def visual_occ_order_in_image(img, input_bbox, gt_occ_orders, pred_occ_orders, write_path):
    # TODO:
    out_img = img.copy()
    for obj_idx in range(len(input_bbox)):
        out_img = draw_box(out_img, input_bbox[obj_idx], color=(255, 0, 0), thick=3)
    # cv2.imwrite(write_path + img_id, )


# def eval_seg_prediction(collection, eval_modes=['inmodal', 'amodal', 'occ'], iou=0.5):
#     for eval in eval_modes:
#         iou_performance = collection[eval]['iou_performance']
#         gt_mask = collection[eval]['gt_collection']
#         pred_mask = collection[eval]['pred_collection']


def seg_dict_reconstruct(old_seg):
    new_seg = {'amodal': {}, 'inmodal': {}, 'occ': {}}
    for i in range(0, len(old_seg)):
        new_seg['inmodal'][i] = old_seg[i]['inmodal']
        new_seg['amodal'][i] = old_seg[i]['amodal']
        new_seg['occ'][i] = old_seg[i]['occ']
    return new_seg


def pair_wise_order_matrix(inmodal, amodal):
    # binary mask
    b_inmodal = {}
    b_amodal = {}
    for i in range(0, len(inmodal)):
        b_inmodal[i] = (inmodal[i] > 0)
        b_amodal[i] = (amodal[i] > 0)

    order_matrix = np.zeros((len(b_inmodal), len(b_inmodal)), dtype=int)
    for i in range(0, len(b_inmodal)):
        for j in range(i+1, len(b_inmodal)):
            if torch.sum(b_amodal[i] * b_amodal[j]) == 0:  # no occlusion
                continue
            occ_ij = torch.sum(b_inmodal[i] * b_amodal[j])
            occ_ji = torch.sum(b_inmodal[j] * b_amodal[i])
            if occ_ji == 0 and occ_ij == 0:
                continue
            order_matrix[i, j] = 1 if occ_ij > occ_ji else -1
            order_matrix[j, i] = - order_matrix[i, j]

    return order_matrix


def eval_performance(data_loader, demo=False, category='car', eval_modes=('inmodal', 'amodal'), occ_bounds=(0, 1),
                     input_bbox_type='inmodal', input_gt_label=True, expand_box=True, reasoning=False, output_file=None,
                     filter_param=[0.95, 33], min_bthrd=0, diff_thrd=0, reas_iter=1):
    N = data_loader.__len__()

    collections = dict()
    for eval in eval_modes:
        collections[eval] = {'pred_collection': [], 'gt_collection': [], 'iou_performance': [],
                             'bmask_thrd': binary_mask_thrds[eval]}

    img_collection = []
    obj_id_collection = []
    pixel_cls_collection = []
    gt_label_collection = []

    classification_acc = []
    amodal_prediction_iou = []
    amodal_height = []
    occ_percentage = []

    order_acc = 0
    occ_order_acc = 0

    pred_confidences = np.array([])

    img_num = 0
    total_pair = 0
    total_occ_pair = 0

    # if debug:
    order_miss = 0
    order_wrong = 0
    order_false_detec = 0

    for ii, data in enumerate(data_loader):
        # for debug
        # if ii >= 150:
        #     break

        input_tensor, gt_labels, gt_inmodal_bbox, gt_amodal_bbox, gt_inmodal_segmentation, gt_amodal_segmentation, \
        gt_occ, gt_cluster_ids, gt_occ_orders, demo_img, img_path = data

        if demo:
            demo_img = demo_img.numpy().squeeze()

        gt_inmodal_bbox = gt_inmodal_bbox[0].numpy()
        gt_amodal_bbox = gt_amodal_bbox[0].numpy()
        gt_labels = gt_labels[0].numpy()
        # gt_amodal_segmentation = gt_amodal_segmentation[0].numpy()
        # gt_inmodal_segmentation = gt_inmodal_segmentation[0].numpy()
        gt_amodal_segmentation = gt_amodal_segmentation[0].cuda(device_ids[0])
        gt_inmodal_segmentation = gt_inmodal_segmentation[0].cuda(device_ids[0])
        gt_occ = gt_occ[0].numpy()
        img_path = img_path[0]
        gt_cluster_ids = gt_cluster_ids[0].numpy()
        gt_occ_orders = gt_occ_orders[0].numpy()

        # # for debug
        # print('img path:', img_path)

        # if '00000' not in img_path:
        #     continue

        gt_order_matrix = pair_wise_order_matrix(gt_inmodal_segmentation, gt_amodal_segmentation)

        if input_bbox_type == 'inmodal':
            input_bbox = gt_inmodal_bbox.copy()
        elif input_bbox_type == 'amodal':
            input_bbox = gt_amodal_bbox.copy()
        else:
            input_bbox = None
            print('input_bbox_type not recognized')

        skip = False
        for box in input_bbox:
            if box[2] - box[0] == 0 or box[3] - box[1] == 0:
                skip = True
                break
        if skip:
            continue

        img_num += 1

        if input_gt_label:
            input_label = gt_labels.copy()
        else:
            input_label = None

        input_tensor.requires_grad = False
        input_tensor = input_tensor.cuda(device_ids[0])

        pred_labels, pred_confidence, pred_amodal_bboxes, pred_segmentations, pred_occ_orders = net(org_x=input_tensor,
                                                                                   bboxes=input_bbox,
                                                                                   bbox_type=input_bbox_type,
                                                                                   input_label=input_label,
                                                                                   gt_labels=gt_labels.copy(),
                                                                                   crop_pad=16,
                                                                                   clutter_weight=0.1,
                                                                                   reasoning=reasoning,
                                                                                   filter_param=filter_param,
                                                                                   min_bthrd=min_bthrd,
                                                                                   diff_thrd=diff_thrd,
                                                                                   reas_iter=reas_iter,
                                                                                   gt_orders=gt_occ_orders)

        for obj_idx in range(len(pred_amodal_bboxes)):
            pred_segmentation = pred_segmentations[obj_idx]
            for i, ev in enumerate(pred_segmentation):
                pred_segmentation[ev] = torch.from_numpy(pred_segmentation[ev]).cuda(device_ids[0])

        # switch dimension (N, eval_mode) -> (eval_mode, N)
        pred_seg_sw_dim = seg_dict_reconstruct(pred_segmentations)

        if reasoning:
            # pred order
            pred_occ_orders = pair_wise_order_matrix(pred_seg_sw_dim['inmodal'], pred_seg_sw_dim['amodal'])

            assert len(input_bbox) == gt_order_matrix.shape[0] == pred_occ_orders.shape[0], "object num dis-match!"
            order_acc += np.sum(gt_order_matrix == pred_occ_orders) - len(input_bbox)
            occ_order_acc += np.sum((gt_order_matrix == pred_occ_orders) & (gt_order_matrix != 0))

            total_pair += len(input_bbox) * (len(input_bbox) - 1)
            total_occ_pair += np.sum(gt_order_matrix != 0)

            # if debug:
            order_miss += np.sum((gt_order_matrix != 0) & (pred_occ_orders == 0))
            order_wrong += np.sum((gt_order_matrix == - pred_occ_orders) & (gt_order_matrix != 0))
            order_false_detec += np.sum((gt_order_matrix == 0) & (pred_occ_orders != 0))

        # classification confidence
        pred_confidences = np.append(pred_confidences, pred_confidence)

        pred_acc = (gt_labels == pred_labels).astype(float)

        for obj_idx in range(len(pred_amodal_bboxes)):

            pred_segmentation = pred_segmentations[obj_idx]
            pixel_cls_b = pred_segmentation['pixel_cls'].cpu().numpy()
            pixel_cls_score = pred_segmentation['pixel_cls_score'].cpu().numpy()

            if torch.sum(gt_inmodal_segmentation[obj_idx]) == 0:
                continue

            box = np.copy(gt_amodal_bbox[obj_idx])
            if expand_box:
                height_box = box[2] - box[0]
                width_box = box[3] - box[1]

                box[0] = max(0, box[0] - 0.207 * height_box)
                box[1] = max(0, box[1] - 0.207 * width_box)
                box[2] = min(input_tensor.shape[2], box[2] + 0.207 * height_box)
                box[3] = min(input_tensor.shape[3], box[3] + 0.207 * width_box)
            box = box.astype(int)

            for eval in eval_modes:
                bmask_thrd = collections[eval]['bmask_thrd']
                gt_eval_segmentation = []

                if eval == 'amodal':
                    gt_eval_segmentation = gt_amodal_segmentation[obj_idx]

                elif eval == 'inmodal':
                    gt_eval_segmentation = gt_inmodal_segmentation[obj_idx]

                elif eval == 'occ':
                    gt_amodal_seg = gt_amodal_segmentation[obj_idx]
                    gt_inmodal_seg = gt_inmodal_segmentation[obj_idx]

                    gt_eval_segmentation = gt_amodal_seg - gt_inmodal_seg
                else:
                    print('Unexpected evaluation mode')

                gt_seg = gt_eval_segmentation
                pred_seg = pred_segmentation[eval]
                pred_seg_b = (pred_seg >= bmask_thrd).to(torch.float)
                #
                # ap, prec_rec_curve = pr_curve(pred_seg[box[0]:box[2], box[1]:box[3]], gt_seg[box[0]:box[2], box[1]:box[3]])
                # collections[eval]['ap_performance'].append(ap)

                collections[eval]['iou_performance'].append(
                    (torch.sum(pred_seg_b * gt_seg) / torch.sum(pred_seg_b + gt_seg > 0.5)).cpu().numpy().reshape(1,)[0])

                if demo:
                    collections[eval]['pred_collection'].append(pred_seg_b[box[0]:box[2], box[1]:box[3]])
                    collections[eval]['gt_collection'].append(gt_seg[box[0]:box[2], box[1]:box[3]])

                    if eval_modes.index(eval) == 0:
                        copy_img = draw_box(demo_img.copy(), gt_inmodal_bbox[obj_idx], color=(255, 0, 0), thick=2)
                        copy_img = draw_box(copy_img, pred_amodal_bboxes[obj_idx], color=(0, 255, 0), thick=2)
                        img_collection.append(copy_img[box[0]:box[2], box[1]:box[3], :])

                        suffix = img_path.rfind('.')
                        obj_id_collection.append('{}_{:0>2d}'.format(img_path[suffix - 5: suffix], obj_idx))

            #     img_patch = copy_img[box[0]:box[2], box[1]:box[3], :]
            #     pred_patch = pred_seg[box[0]:box[2], box[1]:box[3]]
            #     gt_patch = gt_seg[box[0]:box[2], box[1]:box[3]] * np.max(pred_seg)
            #     visualize_multi(img_patch, [pred_patch, gt_patch], 'temp_{}'.format(eval), cbar=True)
            # assert False



            # demo_img = demo_img.numpy().squeeze()
            # pixel_cls = make_three_dimensional_demo(pixel_cls=pixel_cls_b, pixel_cls_score=pixel_cls_score)
            # visualize_multi(demo_img[box[0]:box[2], box[1]:box[3]], [pred_segmentation['inmodal'][box[0]:box[2], box[1]:box[3]], pred_segmentation['amodal'][box[0]:box[2], box[1]:box[3]]], 'old')
            # cv2.imwrite('pixel_cls.jpg', pixel_cls[box[0]:box[2], box[1]:box[3]])
            # old_inmodal = (pred_segmentation['inmodal'] == 1).astype(int)
            # new_inmodal = pixel_relation_post_proc(pixel_cls=pixel_cls_b, pixel_cls_score=pixel_cls_score)
            #
            # gt_inmodal = gt_eval_segmentation[obj_idx]
            #
            # print(np.sum(old_inmodal * gt_inmodal) / np.sum((old_inmodal + gt_inmodal > 0.5).astype(float)))
            # print(np.sum(new_inmodal * gt_inmodal) / np.sum((new_inmodal + gt_inmodal > 0.5).astype(float)))
            #
            # visualize_multi(demo_img[box[0]:box[2], box[1]:box[3]], [old_inmodal[box[0]:box[2], box[1]:box[3]], new_inmodal[box[0]:box[2], box[1]:box[3]], gt_inmodal[box[0]:box[2], box[1]:box[3]]], 'temp')
            # assert False

            if demo:
                pixel_cls_collection.append(make_three_dimensional_demo(pixel_cls=pixel_cls_b[box[0]:box[2], box[1]:box[3], ], pixel_cls_score=pixel_cls_score[box[0]:box[2], box[1]:box[3], ]))

            gt_label_collection.append(gt_labels[obj_idx])
            classification_acc.append(pred_acc[obj_idx])
            amodal_prediction_iou.append(pred_acc[obj_idx] * calc_iou(pred_box=pred_amodal_bboxes[obj_idx], gt_box=gt_amodal_bbox[obj_idx]))
            amodal_height.append(gt_amodal_bbox[obj_idx][2] - gt_amodal_bbox[obj_idx][0])
            occ_percentage.append(gt_occ[obj_idx])

        # visualize order
        # if reasoning:
        #     visual_occ_order_in_image(demo_img, input_bbox, gt_occ_orders, pred_occ_orders, img_path)

        #
        if ii % 10 == 0:
            print('    {}  -  eval {}/{}     '.format(category, ii, N), end='\r')

        # if debug and ii > fraction_to_load * N:
        #     break

    # collections[eval]['pred_collection'] = torch.Tensor([x.cpu().numpy() for x in collections[eval]['pred_collection']]).cuda(device_ids[0])
    # collections[eval]['gt_collection'] = torch.Tensor([x.cpu().numpy() for x in collections[eval]['gt_collection']]).cuda(device_ids[0])
    print('\n')

    output = dict()

    for sub_cat in category:
        sub_cat_label = categories[dataset_eval].index(sub_cat)

        # occlusion level
        for occ_bound in occ_bounds:

            eval_modes_ = copy.deepcopy(eval_modes)
            left_occ = occ_bound[0]
            right_occ = occ_bound[1]

            if right_occ == 0 and 'occ' in eval_modes_:
                eval_modes_.remove('occ')

            print('========Height_thrd: {} - Occlusion_bound: {}========'.format(height_thrd, occ_bound))
            sub_tag = '{}_[{}, {}]'.format(sub_cat, height_thrd, occ_bound)

            selected = np.argwhere((np.array(gt_label_collection) == sub_cat_label) &
                                   (np.array(occ_percentage) >= left_occ) &
                                   (np.array(occ_percentage) < right_occ))[:, 0]

            if selected.shape[0] == 0:
                continue

            classification_acc_ = np.array(classification_acc)[selected]
            amodal_prediction_iou_ = np.array(amodal_prediction_iou)[selected]
            amodal_height_ = np.array(amodal_height)[selected]
            occ_percentage_ = np.array(occ_percentage)[selected]

            if demo:
                pixel_cls_collection_ = np.array(pixel_cls_collection, dtype=object)[selected]
                img_collection_ = np.array(img_collection, dtype=object)[selected]
                obj_id_collection_ = np.array(obj_id_collection)[selected]

            cls_acc = np.mean(classification_acc_)
            amodal_iou_correct_cls = np.mean(amodal_prediction_iou_[classification_acc_ == 1])
            amodal_iou_final = np.mean(amodal_prediction_iou_)

            output[sub_tag] = {}

            output[sub_tag]['cls'] = {'amodal_prediction_iou': amodal_prediction_iou_, 'acc_array': classification_acc_, 'amodal_height': amodal_height_, 'occ': occ_percentage_,
                             'cls_acc' : cls_acc, 'amodal_iou_correct_cls' : amodal_iou_correct_cls, 'amodal_iou_final' : amodal_iou_final, 'num_objects': classification_acc_.shape[0]}

            print_line = '{}      cls_acc: {:6.4f}     |     amodal_box_mIoU: {:6.4f}     '.format(sub_cat, cls_acc, amodal_iou_final)

            for eval in eval_modes_:
                iou_performance_ = np.array(collections[eval]['iou_performance'], dtype=object)[selected]
                if demo:
                    # pred_collection_ = collections[eval]['pred_collection'][selected]
                    # gt_collection_ = collections[eval]['gt_collection'][selected]
                    pred_collection_ = []
                    gt_collection_ = []
                    for select_id in selected:
                        pred_collection_.append(collections[eval]['pred_collection'][select_id])
                        gt_collection_.append(collections[eval]['gt_collection'][select_id])

                # mask AP
                mask_ap = np.zeros(10) # [0, 0.1, ..., 0.9]
                for iou_thrd in np.arange(0., 1.0, 0.1):
                    mask_ap[int(iou_thrd * 10)] = np.sum(iou_performance_ >= iou_thrd) / len(iou_performance_)

                output[sub_tag][eval] = {'average iou': np.mean(iou_performance_ * classification_acc_),
                                         'average ap': mask_ap,
                                         'precision': np.mean((np.array(iou_performance_ * classification_acc_) >= 0.5).astype(float)),
                                         'num_objects': len(selected),
                                         'iou:': iou_performance_}

                bmask_thrd_ = collections[eval]['bmask_thrd']

                if demo:
                    out_dir_ = demo_dir + '{}/'.format(sub_tag)
                    if not os.path.exists(out_dir_):
                        os.mkdir(out_dir_)

                    out_dir = out_dir_ + '{}/'.format(eval)
                    if not os.path.exists(out_dir):
                        os.mkdir(out_dir)

                    out_dir_cls = out_dir_ + 'pixel_cls/'
                    if not os.path.exists(out_dir_cls):
                        os.mkdir(out_dir_cls)

                    rank_ind = list(rank_perf(pred_collection_, gt_collection_, bmask_thrd_))
                    i = 0
                    for ri in rank_ind:
                        if True:
                        # if i % 10 == 0 or i < 100:
                            try:
                                visualize_multi(img_collection_[ri], [pred_collection_[ri].cpu().numpy(), gt_collection_[ri].cpu().numpy()],
                                                '{}rank_{}_img_{}'.format(out_dir, i, obj_id_collection_[ri]), cbar=False, style='visual_seg')

                                # TODO: add a colorbar
                                pixel_cls_img = np.concatenate((img_collection_[ri], pixel_cls_collection_[ri]), axis=0)
                                cv2.imwrite('{}img_{}.jpg'.format(out_dir_cls, obj_id_collection_[ri]), pixel_cls_img)

                                print('    {}  -  demo {}/{}     '.format(category, i, len(pred_collection_)), end='\r')
                            except:
                                print('error', end='\r')

                        i += 1

                print_line += '|     {}_mIoU: {:6.4f}     '.format(eval, output[sub_tag][eval]['average iou'])
                # print_line += '|     {}_AP: {:6.4f}     '.format(eval, np.mean(output[sub_tag][eval]['average ap']))
                # print_line += '|     {}_AP50: {:6.4f}     '.format(eval, output[sub_tag][eval]['average ap'][5])

            print_line += "|     object_num: {}     ".format(selected.shape[0])
            print(print_line)
            print(print_line, file=output_file)

        print_line = 'overall confidences:    {:6.4f}\n'.format(pred_confidences.mean())

        # pair-wise order recovery accuracy
        if reasoning:
            occ_order_acc /= total_occ_pair
            order_acc /= total_pair
            print_line += 'img_num: {}     all order acc:     {:6.4f}      occluded order acc:     {:6.4f}\n'.format(
                                img_num, order_acc, occ_order_acc)

            # if debug:
            print_line += 'total_pair: {}\n'.format(total_pair)
            print_line += 'total_occ_pair: {}\n'.format(total_occ_pair)
            print_line += 'order_wrong: {}\n'.format(order_wrong)
            print_line += 'order_miss: {}\n'.format(order_miss)
            print_line += 'order_false_detec: {}\n'.format(order_false_detec)

        print(print_line)
        print(print_line, file=output_file)
    return output


if __name__ == '__main__':

    # filter_param_list = [[0, 33], [0, 11], [0.5, 33], [0.5, 11], [0.95, 33], [0.95, 11]]
    # filter_param_list = [[0, 33]]
    # tag_prefixs = ['syn_300_v' + str(i+1) for i in range(len(filter_param_list))]

    # reasoning_param_list = [[0, 3]]
    # tag_prefixs = ['kins_100_r' + str(i+10) for i in range(len(reasoning_param_list))]

    # for reas_iter in range(1, 4):
    if True:
        reas_iter = 2
        tag_prefix = 'syn_2plus_iter{}_seg0.1_norder_filter55'.format(reas_iter)
        # tag_prefix = 'test2_synth_200_4_seg0.03_crdomain_filter'
        # tag_prefix = 'kins_test_figure'
        filter_param = [0, 55]
        min_bthrd, diff_thrd = 0, 0

        for reasoning in [True]:

            input_bbox_type = 'amodal'     # inmodal,  amodal

            input_gt_label = True           # True,     False

            dataType = 'test'
            eval_modes = ['inmodal', 'occ', 'amodal']  # ['inmodal', 'occ', 'amodal']

            debug = False

            tag = '{}_post_{}'.format(tag_prefix, str(reasoning))

            categories['eval'] = ['car']
            # categories['eval'] = ['car', 'cyclist', 'tram', 'truck', 'van']                      # overwrite the categories to test only cars

            binary_mask_thrds = {'amodal': 0.5, 'inmodal': 0.5, 'occ': 0.5}

            bool_demo_seg = False

            # OccVeh Params:
            OccVeh_omega = 0.2
            fg_levels = [0, 1, 2, 3]        # 0, 1, 2, 3
            bg_levels = [1, 2, 3]           # 1, 2, 3

            # KINS Params:
            KINS_omega = 0.2
            height_thrd = 30

            # occ_bounds = [[0, 0.25], [0.2501, 0.5], [0.5001, 0.75], [0.7501, 0.95]]  # [0, 0], [0.0001, 0.25], [0.2501, 0.5], [0.5001, 0.75]
            occ_bounds = [[0, 0.01], [0.01, 0.30], [0.30, 0.60], [0.60, 0.90]]
            fraction_to_load = 1.0
                                                              # train, test

            if debug:
                tag = 'debug_' + tag
                # fraction_to_load = 0.1
                # bool_demo_seg = False

            # net = get_compnet_head(mix_model_dim_reduction=True, mix_model_suffix='_fully_sup')  #_fully_sup | _RP_newer_it2
            net = get_compnet_head(mix_model_dim_reduction=True, mix_model_suffix='_cross_domain')

            overall_exp_dir = exp_dir + '{}_mIoU_{}_bbox_supervised_gt_label_{}/'.format(tag, input_bbox_type, input_gt_label)
            if not os.path.exists(overall_exp_dir):
                os.mkdir(overall_exp_dir)

            plot_dir = overall_exp_dir + 'plot_{}/'.format(dataset_eval)
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)

            demo_dir = overall_exp_dir + 'demo_{}/'.format(dataset_eval)
            if not os.path.exists(demo_dir):
                os.mkdir(demo_dir)

            file = open(overall_exp_dir + 'exp_info_{}.txt'.format(dataset_eval), 'w')

            if True:
                print()
                print()
                print('Tag:', tag)
                print('Segmentation Evaluation for {} on {} - {} set with given gt {} box\n     GT label given: {}\n'.format(eval_modes, dataset_eval.upper(), dataType, input_bbox_type, input_gt_label))
                print('Segmentation Evaluation for {} on {} - {} set with given gt {} box\n     GT label given: {}\n'.format(eval_modes, dataset_eval.upper(), dataType, input_bbox_type, input_gt_label), file=file)
                if dataset_eval == 'occveh':
                    print('Experiment Setup:    Backbone - {}      FG_levels - {}      BG_levels  - {}      omega - {}'.format(nn_type, fg_levels, bg_levels, OccVeh_omega))
                    print('Experiment Setup:    Backbone - {}      FG_levels - {}      BG_levels  - {}      omega - {}'.format(nn_type, fg_levels, bg_levels, OccVeh_omega), file=file)
                elif dataset_eval == 'kins':
                    print('Experiment Setup:    Backbone - {}      height_thrd - {}        occ_bounds  - {}     omega - {}     data_fraction - {}       data_type: {}'.format(nn_type, height_thrd, occ_bounds, KINS_omega, fraction_to_load, dataType))
                    print('Experiment Setup:    Backbone - {}      height_thrd - {}        occ_bounds  - {}     omega - {}     data_fraction - {}       data_type: {}'.format(nn_type, height_thrd, occ_bounds, KINS_omega, fraction_to_load, dataType), file=file)
                print()

            try:
                with open(overall_exp_dir + '/exp_meta_{}.pickle'.format(dataset_eval), 'rb') as fh:
                    meta = pickle.load(fh)
            except:
                meta = dict()

            print('Post-processing:\n     Reasoning: {}'.format(str(reasoning)))
            # print('Filtration Params:\n     {}'.format(str(filter_param)))
            if reasoning:
                print('Reasoning Params:\n     min_bthrd: {},  diff_thrd: {}'.format(str(min_bthrd), str(diff_thrd)))
            print()
            print()

            if dataset_eval == 'occveh':

                for data_type in ['veh', 'obj']:
                    if data_type == 'veh':
                        categories['eval'] = ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike']
                    elif data_type == 'obj':
                        categories['eval'] = ['boat', 'bottle', 'chair', 'diningtable', 'sofa', 'tvmonitor']

                    net.update_fused_models(omega=OccVeh_omega)
                    combine_cats_iou = dict()

                    for fg_level in fg_levels:

                        eval_modes_ = copy.deepcopy(eval_modes)
                        bg_levels_ = bg_levels
                        if fg_level == 0:
                            bg_levels_ = [0]
                            if 'occ' in eval_modes_:
                                eval_modes_.remove('occ')

                        for bg_level in bg_levels_:
                            print('========FGL{} BGL{}========'.format(fg_level, bg_level))
                            for category in categories['eval']:
                                sub_tag = '{}FGL{}_BGL{}'.format(category, fg_level, bg_level)

                                data_set = Occ_Veh_Dataset(cats=[category], dataType=dataType, train_types=[None], fg_level=fg_level, bg_level=bg_level, single_obj=True, resize=False, crop_img=False, crop_padding=48, crop_central=False, demo_img_return=True)
                                data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False)

                                meta[sub_tag] = eval_performance(data_loader, category=category, demo=bool_demo_seg, eval_modes=eval_modes_, input_bbox_type=input_bbox_type, input_gt_label=input_gt_label)

                            acc = 0
                            amodal_box_iou = 0
                            total = 0
                            for eval in eval_modes_:
                                combine_cats_iou[eval] = 0

                            for category in categories['eval']:
                                sub_tag = '{}FGL{}_BGL{}'.format(category, fg_level, bg_level)
                                num_obj = meta[sub_tag]['cls']['acc_array'].shape[0]

                                acc += meta[sub_tag]['cls']['cls_acc'] * num_obj
                                amodal_box_iou += meta[sub_tag]['cls']['amodal_iou_final'] * num_obj
                                total += num_obj

                                for eval in eval_modes_:
                                    combine_cats_iou[eval] += meta[sub_tag][eval]['average iou'] * num_obj

                            level_tag = 'FGL{}_BGL{}'.format(fg_level, bg_level)
                            print_line = '{:10}  -  cls_acc: {:6.4f}     |     amodal_box_mIoU: {:6.4f}     '.format(level_tag, acc / total, amodal_box_iou / total)
                            meta[level_tag] = {'cls_acc' : acc / total, 'amodal_box_mIoU' : amodal_box_iou / total}
                            for eval in eval_modes_:
                                print_line += '|     {}_mIoU: {:6.4f}     '.format(eval, combine_cats_iou[eval] / total)
                                meta[level_tag]['{}_mIoU'.format(eval)] = combine_cats_iou[eval] / total

                            print('\n{}\n\n'.format(print_line))
                            print('\n{}\n\n'.format(print_line), file=file)

                            with open(overall_exp_dir + '/exp_meta_{}.pickle'.format(dataset_eval), 'wb') as fh:
                                pickle.dump(meta, fh)

            elif dataset_eval == 'kins':
                net.update_fused_models(omega=KINS_omega)

                data_set = KINS_Dataset(category_list=categories['eval'], dataType=dataType, occ=(0, 1),
                                        height_thrd=height_thrd, amodal_height=False, frac=fraction_to_load,
                                        demo_img_return=True)
                data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False)

                meta = eval_performance(data_loader, demo=bool_demo_seg, category=categories['eval'],
                                        eval_modes=eval_modes, occ_bounds=occ_bounds, input_bbox_type=input_bbox_type,
                                        input_gt_label=input_gt_label, expand_box=True,
                                        reasoning=reasoning, output_file=file, filter_param=filter_param,
                                        min_bthrd=min_bthrd,
                                        diff_thrd=diff_thrd,
                                        reas_iter=reas_iter)

                with open(overall_exp_dir + '/exp_meta_{}.pickle'.format(dataset_eval), 'wb') as fh:
                    pickle.dump(meta, fh)

            file.close()
