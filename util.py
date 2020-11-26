import matplotlib.pyplot as plt
from scipy import interpolate
import pycocotools.mask as mask_utils
import numpy as np
import random as rm
import torch
import cv2
import os


def draw_box(img, bbox, color, thick):
    cv2.rectangle(img, ((int)(bbox[1]), (int)(bbox[0])), ((int)(bbox[3]), (int)(bbox[2])), color, thick)
    return img


def resize(img, pixel=224, type='short_side'):
    h, w, c = img.shape
    factor = 1

    if type == 'short_side':
        factor = pixel / min(h, w)
    elif type == 'long_side':
        factor = pixel / max(h, w)
    else:
        print('Error in resizing image')

    return cv2.resize(img, (int(w * factor), int(h * factor)))


def res_down(response, dim):
    output = np.zeros(dim)
    xstep = response.shape[0] / dim[0]
    ystep = response.shape[1] / dim[1]

    for x in range(dim[0]):
        for y in range(dim[1]):
            output[x][y] = np.max(response[int(x * xstep) : int((x + 1) * xstep), int(y * ystep) : int((y + 1) * ystep)])

    return output


def visualize(img, response, name, definition=200, cbar=False, resize_img=True):
    if resize_img:
        large_rsp = cv2.resize(response, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)
    else:
        factor = img.shape[0] / response.shape[0]
        large_rsp = cv2.resize(response, ((int)(response.shape[1] * factor), (int)(response.shape[0] * factor)), interpolation=cv2.INTER_NEAREST)
    original_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(original_img)
    im = ax2.imshow(large_rsp)
    ax1.axis('off')
    ax2.axis('off')
    if cbar:
        cb = plt.colorbar(im, ax=(ax1, ax2))
    plt.savefig('{}.jpg'.format(name), format='jpg', dpi=definition, bbox_inches='tight')
    plt.close('all')


def polys_to_mask(polygons, height, width):
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask


def visualize_multi(img, responses, name, definition=200, cbar=False, resize_img=True, style='Default'):
    large_rsps = []
    max_resp = -100
    min_resp = 100
    for response in responses:
        if resize_img:
            large_rsp = cv2.resize(response, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)
        else:
            factor = img.shape[0] / response.shape[0]
            large_rsp = cv2.resize(response, ((int)(response.shape[1] * factor), (int)(response.shape[0] * factor)), interpolation=cv2.INTER_NEAREST)
        large_rsps.append(large_rsp)
        if np.max(large_rsp) > max_resp:
            max_resp = np.max(large_rsp)
        if np.min(large_rsp) < min_resp:
            min_resp = np.min(large_rsp)

    original_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if style == 'mixed' and len(responses) > 1:
        fig, ax = plt.subplots(2, 1)

        ax[0].imshow(original_img)
        ax[0].axis('off')
        im = ax[0]

        color_options = [[132, 112, 255], [255, 255, 0], [30, 144, 255]]  # in RGB

        for i in range(len(responses)):
            responses[i] = np.stack((responses[i]*color_options[i][0],
                                     responses[i]*color_options[i][1],
                                     responses[i]*color_options[i][2]), axis=2)

        mixed_responses = cv2.addWeighted(responses[0].astype('uint8'), 0.5, responses[1].astype('uint8'), 0.5, 0)

        for i in range(len(responses) - 2):
            mixed_responses = cv2.addWeighted(mixed_responses, 0.5, responses[i+2].astype('uint8'), 0.5, 0)

        ax[1].imshow(mixed_responses)
        ax[1].axis('off')

    if style == 'visual_seg':
        fig, ax = plt.subplots(3, 1)

        # original image
        ax[0].imshow(original_img)
        ax[0].axis('off')
        im = ax[0]

        # response map
        color_options = [ [255, 215, 0], [154, 205, 50]]  # in RGB  gold [255, 215, 0], orange [238, 121, 66]
        # pred_resp, gt_resp = responses
        plot_img = [original_img.copy(), original_img.copy()]

        for subp in range(0, len(responses)):
            # find contours
            contours, hierarchy = cv2.findContours(np.array(responses[subp], dtype='uint8'), cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_NONE)

            max_cnt = 0
            max_contour = contours[0]
            for i in range(0, len(contours)):
                if len(contours[i]) > len(contours[max_cnt]):
                    max_cnt = i
                    max_contour = contours[i]

            # pred_resp = polys_to_mask(max_contour.reshape(max_contour.shape[0], max_contour.shape[2]).astype(float),
            #                           pred_img.shape[0], pred_img.shape[1])

            pred_m = np.stack((responses[subp].astype('uint8') * color_options[subp][0],
                               responses[subp].astype('uint8') * color_options[subp][1],
                               responses[subp].astype('uint8') * color_options[subp][2]), axis=2)

            pred_m_w = cv2.addWeighted(plot_img[subp], 0.5, pred_m, 0.5, 0)

            cv2.drawContours(pred_m_w, max_contour, -1, color_options[subp], 4)

            ax[subp+1].imshow(pred_m_w)
            ax[subp+1].axis('off')

        # # mask
        # pred_mask = polys_to_mask(cnt, pred_img.shape[0], pred_img.shape[1])
        # convex hull
        # hull = cv2.convexHull(cnt)
        # length = len(hull)
        # for i in range(len(hull)):
        #     cv2.line(pred_img, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), (0, 255, 0), 2)

    else:
        fig, ax = plt.subplots(1 + len(responses), 1)

        ax[0].imshow(original_img)
        ax[0].axis('off')
        im = ax[0]

        for i in range(len(responses)):
            im = ax[i + 1].imshow(responses[i], vmin=min_resp, vmax=max_resp)
            ax[i + 1].axis('off')

    if cbar:
        cb = fig.colorbar(im, ax=ax)
    plt.savefig('{}.jpg'.format(name), format='jpg', dpi=definition, bbox_inches='tight')
    plt.close('all')


# def pr_curve(pred_seg, gt_seg, num_points=50, use_07_metric=True):
#     # Compute VOC AP given precision and recall. If use_07_metric is true, uses
#     #     the VOC 07 11-point method
#     pred_min, pred_max = [9999, -9999]
#     pred_min = min(pred_min, torch.min(pred_seg)).item()
#     pred_max = max(pred_max, torch.max(pred_seg)).item()
#
#     thrd_list = np.linspace(pred_min - 1e-4, pred_max, num_points)
#
#     precision = np.zeros(num_points)
#     recall = np.zeros(num_points)
#
#     AllP = torch.sum(gt_seg)
#     AllN = gt_seg.shape[0] * gt_seg.shape[1] - AllP
#
#     if AllP == 0 or AllN == 0:
#         print('Empty Ground Truth')
#         precision = np.ones(num_points)
#         recall = np.zeros(num_points)
#
#     for n in range(len(thrd_list)):
#         thrd = thrd_list[n]
#         pred_mask_b = (pred_seg >= thrd).to(torch.int)
#
#         tp = torch.sum(pred_mask_b * gt_seg)
#         fp = torch.sum(pred_mask_b) - tp
#         fn = torch.sum(gt_seg) - tp
#
#         precision[n] = (torch.true_divide(tp, tp + fp)).item()
#         recall[n] = (torch.true_divide(tp, tp + fn)).item()
#
#     precision = np.concatenate((precision, np.array([1])), axis=0)
#     recall = np.concatenate((recall, np.array([0])), axis=0)
#
#     ap = 0.
#     if use_07_metric:
#         for dr in np.arange(0., 1.1, 0.1):
#             if np.sum(recall >= dr) == 0:
#                 dp = 0
#             else:
#                 dp = np.max(precision[recall >= dr])
#             ap = ap + dp / 11
#     else:
#         # TODO
#         pass
#     return ap, interpolate.interp1d(recall, precision)
#


def roc_curve(pred_collection, gt_collection, num_points=50):
    # Find min and max of the pred_collection
    pred_min, pred_max = [9999, -9999]
    for pred_resp in pred_collection:
        pred_min = min(pred_min, np.min(pred_resp))
        pred_max = max(pred_max, np.max(pred_resp))

    assert len(pred_collection) == len(gt_collection)

    thrd_list = np.linspace(pred_min - 1e-4, pred_max, num_points)
    num_predictions = len(pred_collection)

    tpr = np.zeros((num_points, num_predictions))
    fpr = np.zeros((num_points, num_predictions))

    # Go through every prediction
    for index in range(num_predictions):

        # debug = False

        if index % 10 == 0:
            print('Creating RoC Fit: {}/{}        '.format(index, num_predictions), end='\r')

        gt_mask = gt_collection[index]
        pred_mask = pred_collection[index]

        AllP = np.sum(gt_mask)
        AllN = gt_mask.shape[0] * gt_mask.shape[1] - AllP

        if AllP == 0 or AllN == 0:
            print('Empty Ground Truth at index {}.'.format(index))
            tpr[:, index] = np.zeros(num_points)
            fpr[:, index] = np.zeros(num_points)

        for n in range(len(thrd_list)):
            thrd = thrd_list[n]

            pred_mask_b = (pred_mask >= thrd).astype(int)

            TP = np.sum(pred_mask_b * gt_mask)
            FP = np.sum(pred_mask_b) - TP

            # if TP / AllP < 0.8 and FP / AllN > 0.5 and debug == False:
            #     debug = True

            try:
                tpr[n][index] = TP / AllP
                fpr[n][index] = FP / AllN
            except:
                tpr[n][index] = 0
                fpr[n][index] = 0

        # if debug:
        #     print(index)

    tpr = np.mean(tpr, axis=1)
    fpr = np.mean(fpr, axis=1)

    tpr = np.concatenate((tpr, np.array([0, 1])), axis=0)
    fpr = np.concatenate((fpr, np.array([0, 1])), axis=0)

    fpr_rate = 0.2
    best_thrd = -9999
    proxi = 1
    for n in range(len(thrd_list)):
        if max(fpr[n] - fpr_rate, fpr_rate - fpr[n]) < proxi:
            proxi = max(fpr[n] - fpr_rate, fpr_rate - fpr[n])
            best_thrd = thrd_list[n]

    return best_thrd, interpolate.interp1d(fpr, tpr)#, fill_value="extrapolate")

#Ranking each pred segmentation from best to wrost with a given threshold to apply segmentation
def rank_perf(pred_collection, gt_collection, thrd=0):

    perf = np.zeros(len(pred_collection))

    for index in range(len(gt_collection)):

        if index % 10 == 0:
            print('Ranking Performance: {}/{}        '.format(index, len(gt_collection)), end='\r')

        gt_mask = gt_collection[index]
        pred_mask = pred_collection[index]

        # pred_mask_b = (pred_mask >= thrd).astype(int)
        pred_mask_b = (pred_mask >= thrd).to(torch.int)

        TP = torch.sum(pred_mask_b * gt_mask)
        FP = torch.sum(pred_mask_b) - TP
        AllP = torch.sum(gt_mask)
        AllN = gt_mask.shape[0] * gt_mask.shape[1] - AllP
        try:
            tpr = TP / AllP
            fpr = FP / AllN
        except:
            tpr = 0
            fpr = 0

        perf[index] = tpr * (1 - fpr)

    return np.argsort(perf)[::-1]

def interp(mask, shape, interpolation=cv2.INTER_NEAREST):
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=interpolation)

def calc_iou(pred_box, gt_box):
    '''
    Calculate IoU of single predicted and ground truth box
    :param pred_box:    location of predicted object as [xmin, ymin, xmax, ymax]
    :param gt_box:      location of ground truth object as [xmin, ymin, xmax, ymax]
    :return:            float: value of the IoU for the two boxes
    '''
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou

def tonumpy(data):
    '''
    Convert data to numpy array
    :param data:    data
    :return:        data in numpy array
    '''

    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()


def totensor(data, cuda=True, device_id=0):
    '''
    Convert data to torch tensor
    :param data:        data
    :param cuda:        boolean: cuda
    :param device_id:   device id
    :return:            data in torch tensor
    '''

    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda(device_id)
    return tensor
