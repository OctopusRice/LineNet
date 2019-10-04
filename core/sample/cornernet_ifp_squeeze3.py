import cv2
import math
import numpy as np
import torch
import config_debug

from .utils import random_crop, draw_gaussian, draw_gaussian_line, gaussian_radius, normalize_, color_jittering_, lighting_


def _resize_image(image, detections, size):
    detections    = detections.copy()
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))
    
    height_ratio = new_height / height
    width_ratio  = new_width  / width
    detections[:, 0:4:2] *= width_ratio
    detections[:, 1:4:2] *= height_ratio
    return image, detections


def _clip_detections(image, detections):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    keep_inds  = ((detections[:, 2] - detections[:, 0]) > 0) & \
                 ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    return detections


def cornernet_ifp_squeeze3(system_configs, db, k_ind, data_aug, debug):
    data_rng   = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories   = db.configs["categories"]
    input_size   = db.configs["input_size"]
    output_size  = db.configs["output_sizes"][0]

    border        = db.configs["border"]
    lighting      = db.configs["lighting"]
    rand_crop     = db.configs["rand_crop"]
    rand_color    = db.configs["rand_color"]
    rand_scales   = db.configs["rand_scales"]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou  = db.configs["gaussian_iou"]
    gaussian_rad  = db.configs["gaussian_radius"]

    max_tag_len = 128

    # allocating memory
    images = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    tl_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    br_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    tl_regrs = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    br_regrs = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    tl_tags = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    br_tags = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    tag_masks = np.zeros((batch_size, max_tag_len), dtype=np.uint8)
    tag_lens = np.zeros((batch_size,), dtype=np.int32)

    if config_debug.visualize_sampleFile:
        tl_ifp_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    ifp_Penalty = 0.1

    db_size = db.db_inds.size
    for b_ind in range(batch_size):
        if not debug and not config_debug.visualize_sampleFile and k_ind == 0:
            db.shuffle_inds()

        db_ind = db.db_inds[k_ind]
        k_ind  = (k_ind + 1) % db_size

        # reading image
        image_path = db.image_path(db_ind)
        image      = cv2.imread(image_path)

        # reading detections
        detections = db.detections(db_ind)

        if config_debug.visualize_sampleFile:
            image = cv2.imread('/home/octo/Desktop/LineNet/demo.jpg')
            detections = np.array([[190, 230, 292, 446, 1], [276, 222, 345, 300, 1], [348, 125, 442, 252, 1],
                          [459, 198, 639, 476, 1]], dtype=np.float32)

            image2 = image.copy()
            cv2.imwrite(
                './sampleFile/oriImage' + str(b_ind) + image_path.rpartition('/')[-1], image2)
            for i_detection in detections:
                cv2.rectangle(image2, (i_detection[0].astype(np.int64), i_detection[1].astype(np.int64)),
                              (i_detection[2].astype(np.int64), i_detection[3].astype(np.int64)), (0, 255, 0), 1)
            cv2.imwrite(
                './sampleFile/oriImage_detect' + str(b_ind) + image_path.rpartition('/')[-1], image2)

        # cropping an image randomly
        if not debug and rand_crop:
            image, detections = random_crop(image, detections, rand_scales, input_size, border=border)

        image, detections = _resize_image(image, detections, input_size)

        # Detect inevitable false positive Ground Truth
        if config_debug.visualize_sampleFile:
            image2 = image.copy()
            cv2.imwrite(
                './sampleFile/cropImage' + str(b_ind) + image_path.rpartition('/')[-1], image2)
            for i_detection in detections:
                cv2.rectangle(image2, (i_detection[0].astype(np.int64), i_detection[1].astype(np.int64)),
                              (i_detection[2].astype(np.int64), i_detection[3].astype(np.int64)), (0, 255, 0), 1)
            cv2.imwrite(
                './sampleFile/cropImage_detect' + str(b_ind) + image_path.rpartition('/')[-1], image2)

        ifpGT = []
        ifpGT_flipped = []
        for i in range(0, len(detections)):
            i_detection = detections[i]
            i_width, i_height = i_detection[2] - i_detection[0], i_detection[3] - i_detection[1]
            for j in range(i + 1, len(detections)):
                j_detection = detections[j]

                # ifp for only same category
                if i_detection[4] != j_detection[4]: continue

                # TL inevitable false positive ground truth
                j_width, j_height = j_detection[2] - j_detection[0], j_detection[3] - j_detection[1]
                if i_width * i_height > j_width * j_height:
                    w = i_width
                    h = i_height
                else:
                    w = j_width
                    h = j_height

                if all(i_detection[0:2] > 0) and all(j_detection[0:2] > 0) and \
                        i_detection[0] < image.shape[1] and i_detection[1] < image.shape[0] and \
                        j_detection[0] < image.shape[1] and j_detection[1] < image.shape[0]:
                    if i_detection[0] < j_detection[0] and i_detection[1] > j_detection[1]:
                        ifpGT.append([i_detection[0],
                                      j_detection[1],
                                      i_detection[0] + w,
                                      j_detection[1] + h,
                                      i_detection[4]])
                    elif i_detection[0] > j_detection[0] and i_detection[1] < j_detection[1]:
                        ifpGT.append([j_detection[0],
                                      i_detection[1],
                                      j_detection[0] + w,
                                      i_detection[1] + h,
                                      i_detection[4]])

                if i_detection[2] > 0 and i_detection[1] > 0 and j_detection[2] > 0 and j_detection[1] > 0 and \
                        i_detection[2] < image.shape[1] and i_detection[1] < image.shape[0] and \
                        j_detection[2] < image.shape[1] and j_detection[1] < image.shape[0]:
                    if i_detection[2] < j_detection[2] and i_detection[1] < j_detection[1]:
                        ifpGT_flipped.append([j_detection[2],
                                              i_detection[1],
                                              j_detection[2] - w,
                                              i_detection[1] + h,
                                              i_detection[4]])
                    elif i_detection[2] > j_detection[2] and i_detection[1] > j_detection[1]:
                        ifpGT_flipped.append([i_detection[2],
                                              j_detection[1],
                                              i_detection[2] - w,
                                              j_detection[1] + h,
                                              i_detection[4]])

                # BR inevitable false positive ground truth
                if all(i_detection[2:4] > 0) and all(j_detection[2:4] > 0) and \
                        i_detection[2] < image.shape[1] and i_detection[3] < image.shape[0] and \
                        j_detection[2] < image.shape[1] and j_detection[3] < image.shape[0]:
                    if i_detection[2] < j_detection[2] and i_detection[3] > j_detection[3]:
                        ifpGT.append([j_detection[2] - w,
                                      i_detection[3] - h,
                                      j_detection[2],
                                      i_detection[3],
                                      -i_detection[4]])
                    elif i_detection[2] > j_detection[2] and i_detection[3] < j_detection[3]:
                        ifpGT.append([i_detection[2] - w,
                                      j_detection[3] - h,
                                      i_detection[2],
                                      j_detection[3],
                                      -i_detection[4]])

                if i_detection[0] > 0 and i_detection[3] > 0 and j_detection[0] > 0 and j_detection[3] > 0 and \
                        i_detection[0] < image.shape[1] and i_detection[3] < image.shape[0] and \
                        j_detection[0] < image.shape[1] and j_detection[3] < image.shape[0]:
                    if i_detection[0] < j_detection[0] and i_detection[3] < j_detection[3]:
                        ifpGT_flipped.append([i_detection[0] + w,
                                              j_detection[3] - h,
                                              i_detection[0],
                                              j_detection[3],
                                              -i_detection[4]])
                    elif i_detection[0] > j_detection[0] and i_detection[3] > j_detection[3]:
                        ifpGT_flipped.append([j_detection[0] + w,
                                              i_detection[3] - h,
                                              j_detection[0],
                                              i_detection[3],
                                              -i_detection[4]])

        # End of detecting IFP

        detections = _clip_detections(image, detections)

        width_ratio  = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]

        # flipping an image randomly
        isFlipped = False
        if not debug and not config_debug.visualize_sampleFile and np.random.uniform() > 0.5:
            image[:] = image[:, ::-1, :]
            width    = image.shape[1]
            detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1
            isFlipped = True

        if not debug and not config_debug.visualize_sampleFile:
            image = image.astype(np.float32) / 255.
            if rand_color:
                color_jittering_(data_rng, image)
                if lighting:
                    lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
            normalize_(image, db.mean, db.std)
        images[b_ind] = image.transpose((2, 0, 1))

        # Add ifp to variable
        if isFlipped:
            ifpGT_flipped = np.array(ifpGT_flipped)
            if len(ifpGT_flipped) > 0:
                ifpGT_flipped[:, [0, 2]] = width - ifpGT_flipped[:, [0, 2]] - 1
            ifpGT = ifpGT_flipped

        if config_debug.visualize_sampleFile:
            image2 = image.copy()
            for i_detection in detections:
                cv2.rectangle(image2, (i_detection[0].astype(np.int64), i_detection[1].astype(np.int64)),
                              (i_detection[2].astype(np.int64), i_detection[3].astype(np.int64)), (0, 255, 0), 1)
            cv2.imwrite(
                './sampleFile/clipNflipImage_detect' + str(b_ind) + image_path.rpartition('/')[-1], image2)

            for ifp in ifpGT:
                if ifp[4] >= 0:
                    cv2.circle(image2, (ifp[0].astype(np.int64), ifp[1].astype(np.int64)), 5, (255, 0, 0), 3)
                else:
                    cv2.circle(image2, (ifp[2].astype(np.int64), ifp[3].astype(np.int64)), 5, (0, 0, 255), 3)
            cv2.imwrite('sampleFile/ifp_Added' + str(b_ind) + image_path.rpartition('/')[-1], image2)

        if len(ifpGT) > 0:
            detections = np.concatenate((detections, ifpGT), axis=0)

        for ind, detection in enumerate(detections):
            category = int(detection[-1]) - 1

            xtl, ytl = detection[0], detection[1]
            xbr, ybr = detection[2], detection[3]

            fxtl = (xtl * width_ratio)
            fytl = (ytl * height_ratio)
            fxbr = (xbr * width_ratio)
            fybr = (ybr * height_ratio)

            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)
            ybr = int(fybr)

            if gaussian_bump:
                width  = detection[2] - detection[0]
                height = detection[3] - detection[1]

                width  = math.ceil(width * width_ratio)
                height = math.ceil(height * height_ratio)

                if gaussian_rad == -1:
                    radius = gaussian_radius((height, width), gaussian_iou)
                    radius = max(0, int(radius))
                else:
                    radius = gaussian_rad

                if ind >= len(detections) - len(ifpGT):
                    if category >= 0:
                        draw_gaussian(tl_heatmaps[b_ind, category], [xtl, ytl], radius, ifp_Penalty)
                        if config_debug.visualize_sampleFile:
                            draw_gaussian(tl_ifp_heatmaps[b_ind, category], [xtl, ytl], radius, ifp_Penalty)
                    else:
                        mcategory = -(category + 1) - 1
                        draw_gaussian(br_heatmaps[b_ind, mcategory], [xbr, ybr], radius, ifp_Penalty)
                else:
                    draw_gaussian(tl_heatmaps[b_ind, category], [xtl, ytl], radius)
                    draw_gaussian(br_heatmaps[b_ind, category], [xbr, ybr], radius)
            else:
                if ind >= len(detections) - len(ifpGT):
                    if category >= 0:
                        tl_heatmaps[b_ind, category, ytl, xtl] = ifp_Penalty
                    else:
                        mcategory = -(category + 1) - 1
                        br_heatmaps[b_ind, mcategory, ybr, xbr] = ifp_Penalty
                else:
                    tl_heatmaps[b_ind, category, ytl, xtl] = 1
                    br_heatmaps[b_ind, category, ybr, xbr] = 1

            if ind < len(detections) - len(ifpGT):
                tag_ind = tag_lens[b_ind]
                tl_regrs[b_ind, tag_ind, :] = [fxtl - xtl, fytl - ytl]
                br_regrs[b_ind, tag_ind, :] = [fxbr - xbr, fybr - ybr]
                tl_tags[b_ind, tag_ind] = ytl * output_size[1] + xtl
                br_tags[b_ind, tag_ind] = ybr * output_size[1] + xbr
                tag_lens[b_ind] += 1

        if config_debug.visualize_sampleFile:
            t = np.zeros((output_size[0], output_size[1]), dtype=np.float32)
            t2= np.zeros((output_size[0], output_size[1]), dtype=np.float32)
            for category in range(0, categories):
                t += tl_heatmaps[b_ind, category]
            for category in range(0, categories):
                t2 += tl_ifp_heatmaps[b_ind, category]
            cv2.imwrite('sampleFile/ifp_Added_gt' + str(b_ind) + image_path.rpartition('/')[-1], t * 256)
            cv2.imwrite('sampleFile/ifp_only_gt' + str(b_ind) + image_path.rpartition('/')[-1], t2 * 256)

    if config_debug.visualize_sampleFile:
        exit()

    for b_ind in range(batch_size):
        tag_len = tag_lens[b_ind]
        tag_masks[b_ind, :tag_len] = 1

    images = torch.from_numpy(images)
    tl_heatmaps = torch.from_numpy(tl_heatmaps)
    br_heatmaps = torch.from_numpy(br_heatmaps)
    tl_regrs = torch.from_numpy(tl_regrs)
    br_regrs = torch.from_numpy(br_regrs)
    tl_tags = torch.from_numpy(tl_tags)
    br_tags = torch.from_numpy(br_tags)
    tag_masks = torch.from_numpy(tag_masks)

    return {
               "xs": [images],
               "ys": [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs, tl_tags, br_tags]
           }, k_ind
