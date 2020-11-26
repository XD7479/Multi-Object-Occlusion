import numpy as np
import cv2
import cvbase as cvb
import os
import pycocotools.mask as maskUtils
import pdb
import time


def make_json_dict(imgs, anns):
	imgs_dict = {}
	anns_dict = {}
	for ann in anns:
		image_id = ann["image_id"]
		if not image_id in anns_dict:
			anns_dict[image_id] = []
			anns_dict[image_id].append(ann)
		else:
			anns_dict[image_id].append(ann)

	for img in imgs:
		image_id = img['id']
		imgs_dict[image_id] = img['file_name']

	return imgs_dict, anns_dict


if __name__ == '__main__':
	test_flag = True

	# data_gt_paths = [[os.getenv("HOME") + "/yihong/data/kitti/training/", "instances_train.json"], [os.getenv("HOME") + "/yihong/data/kitti/testing/", "instances_val.json"]]
	if_test = ['train', 'test']
	data_gt_paths = [[os.getenv("HOME") + "/workspace/data/x_mul_with_unknown_occ_FROM_YIHONG/", "annotations/instances_{}_2020.json".format(if_test[test_flag])]]

	for src_data_path, src_gt7_path in data_gt_paths:

		print(src_data_path)

		anns = cvb.load(src_data_path + src_gt7_path)
		imgs_info = anns['images']
		anns_info = anns["annotations"]

		imgs_dict, anns_dict = make_json_dict(imgs_info, anns_info)

		file = open('{}list.txt'.format(src_data_path), 'w')

		count = 0
		for img_id in anns_dict.keys():

			print('{}/{}'.format(count, len(anns_dict.keys())), end='\r')
			count += 1

			st = imgs_dict[img_id].rfind('/')
			img_name_prune = imgs_dict[img_id][st+1: -4]

			img_path = os.path.join('{}data_object_image_2/{}ing/image_2'.format(src_data_path, if_test[test_flag]),
									'{}.png'.format(img_name_prune))
			img_path = '{}data_object_image_2/{}ing/image_2/{}.png'.format(src_data_path, if_test[test_flag], img_name_prune)

			img = cv2.imread(img_path, cv2.IMREAD_COLOR)

			height, width, _ = img.shape
			anns = anns_dict[img_id]

			#img_id
			obj_ids = []
			labels = []

			amodal_bboxes = []
			inmodal_bboxes = []

			amodal_masks = []
			inmodal_masks = []

			occlusion_fractions = []

			cluster_ids = []
			occ_orders = []

			for ann in anns:

				obj_ids.append(ann['id'])
				labels.append(ann['category_id'])

				cluster_ids.append(ann['oco_id'])
				occ_orders.append(ann['ico_id'])

				amodal_bbox = ann['a_bbox']
				amodal_bbox[2] += amodal_bbox[0]
				amodal_bbox[3] += amodal_bbox[1]

				inmodal_bbox = ann['i_bbox']
				inmodal_bbox[2] += inmodal_bbox[0]
				inmodal_bbox[3] += inmodal_bbox[1]

				amodal_bboxes.append(amodal_bbox)
				inmodal_bboxes.append(inmodal_bbox)

				amodal_rle = maskUtils.frPyObjects(ann['a_segm'], height, width)
				imodal_rle = maskUtils.frPyObjects(ann['i_segm'], height, width)
				amodal_masks.append(amodal_rle)
				inmodal_masks.append(imodal_rle)

				amodal_ann_mask = maskUtils.decode(amodal_rle).squeeze()
				inmodal_ann_mask = maskUtils.decode(imodal_rle)[:, :, 0].squeeze()
				# inmodal_ann_mask = maskUtils.decode(ann['i_segm'])[:, :, np.newaxis].squeeze()

				# if count == 1:
				# 	print('cat id:{}'.format(ann['category_id']))
				# 	print('imodal_rle shape: {}'.format(len(imodal_rle)))
				# 	print(amodal_ann_mask.shape)
				# 	print(inmodal_ann_mask.shape)
					# print(inmodal_ann_mask)

				occlusion_frac = max(np.sum(amodal_ann_mask - amodal_ann_mask * inmodal_ann_mask) / np.sum(amodal_ann_mask), 0)
				assert occlusion_frac <= 1.0, 'occlusion fractions > 1.0'
				occlusion_fractions.append(occlusion_frac)
				# occlusion_fractions.append(max(np.sum(amodal_ann_mask - inmodal_ann_mask) / np.sum(amodal_ann_mask), 0))

			obj_ids = np.array(obj_ids)
			labels = np.array(labels)

			amodal_bboxes = np.array(amodal_bboxes)
			inmodal_bboxes = np.array(inmodal_bboxes)

			occlusion_fractions = np.array(occlusion_fractions)

			if not os.path.exists('{}annotations_new_order'.format(src_data_path)):
				os.makedirs('{}annotations_new_order'.format(src_data_path))

			np.savez('{}annotations_new_order/{}.npz'.format(src_data_path, img_name_prune),
					obj_ids=obj_ids, labels=labels,
					amodal_bbox=amodal_bboxes, amodal_mask=amodal_masks, inmodal_bbox=inmodal_bboxes,
					inmodal_mask=inmodal_masks, occluded_percentage=occlusion_fractions,
					occ_order=occ_orders, cluster_id=cluster_ids)
			print(img_name_prune, file=file)

		file.close()
		print()

