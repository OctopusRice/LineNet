import cv2
import math
import torch
import numpy as np
import threading

from .utils import draw_gaussian, gaussian_radius, normalize_, color_jittering_, lighting_, crop_image
import config_debug

def bbox_overlaps(a_dets, b_dets):
    a_widths  = a_dets[:, 2] - a_dets[:, 0]
    a_heights = a_dets[:, 3] - a_dets[:, 1]
    a_areas   = a_widths * a_heights

    b_widths  = b_dets[:, 2] - b_dets[:, 0]
    b_heights = b_dets[:, 3] - b_dets[:, 1]
    b_areas   = b_widths * b_heights

    return a_areas / b_areas

def clip_detections(border, detections):
    detections = detections.copy()

    y0, y1, x0, x1 = border
    det_xs = detections[:, 0:4:2]
    det_ys = detections[:, 1:4:2]
    np.clip(det_xs, x0, x1 - 1, out=det_xs)
    np.clip(det_ys, y0, y1 - 1, out=det_ys)

    keep_inds = ((det_xs[:, 1] - det_xs[:, 0]) > 0) & \
                ((det_ys[:, 1] - det_ys[:, 0]) > 0)
    keep_inds = np.where(keep_inds)[0]
    return detections[keep_inds], keep_inds

def crop_image_dets(image, dets, ind, input_size, output_size=None, random_crop=True, rand_center=True):
    if ind is not None:
        det_x0, det_y0, det_x1, det_y1 = dets[ind, 0:4]
    else: 
        det_x0, det_y0, det_x1, det_y1 = None, None, None, None

    input_height, input_width = input_size
    image_height, image_width = image.shape[0:2]

    centered = rand_center and np.random.uniform() > 0.5
    if not random_crop or image_width <= input_width:
        xc    = image_width // 2
    elif ind is None or not centered:
        xmin  = max(det_x1 - input_width, 0) if ind is not None else 0
        xmax  = min(image_width - input_width, det_x0) if ind is not None else image_width - input_width
        xrand = np.random.randint(int(xmin), int(xmax) + 1)
        xc    = xrand + input_width // 2
    else:
        xmin  = max((det_x0 + det_x1) // 2 - np.random.randint(0, 15), 0)
        xmax  = min((det_x0 + det_x1) // 2 + np.random.randint(0, 15), image_width - 1)
        xc    = np.random.randint(int(xmin), int(xmax) + 1)

    if not random_crop or image_height <= input_height:
        yc    = image_height // 2
    elif ind is None or not centered:
        ymin  = max(det_y1 - input_height, 0) if ind is not None else 0
        ymax  = min(image_height - input_height, det_y0) if ind is not None else image_height - input_height
        yrand = np.random.randint(int(ymin), int(ymax) + 1)
        yc    = yrand + input_height // 2
    else:
        ymin  = max((det_y0 + det_y1) // 2 - np.random.randint(0, 15), 0)
        ymax  = min((det_y0 + det_y1) // 2 + np.random.randint(0, 15), image_height - 1)
        yc    = np.random.randint(int(ymin), int(ymax) + 1)

    image, border, offset = crop_image(image, [yc, xc], input_size, output_size=output_size)
    dets[:, 0:4:2] -= offset[1]
    dets[:, 1:4:2] -= offset[0]
    return image, dets, border

def scale_image_detections(image, dets, scale):
    height, width = image.shape[0:2]

    new_height = int(height * scale)
    new_width  = int(width  * scale)

    image = cv2.resize(image, (new_width, new_height))
    dets  = dets.copy()
    dets[:, 0:4] *= scale
    return image, dets

def ref_scale(detections, random_crop=False):
    if detections.shape[0] == 0:
        return None, None

    if random_crop and np.random.uniform() > 0.7:
        return None, None

    ref_ind = np.random.randint(detections.shape[0])
    ref_det = detections[ref_ind].copy()
    ref_h   = ref_det[3] - ref_det[1]
    ref_w   = ref_det[2] - ref_det[0]
    ref_hw  = max(ref_h, ref_w)

    if ref_hw > 96:
        return np.random.randint(low=96, high=255) / ref_hw, ref_ind
    elif ref_hw > 32:
        return np.random.randint(low=32, high=97) / ref_hw, ref_ind
    return np.random.randint(low=16, high=33) / ref_hw, ref_ind

def create_attention_mask(atts, ratios, sizes, detections):
    for det in detections:
        width  = det[2] - det[0]
        height = det[3] - det[1]

        max_hw = max(width, height)
        for att, ratio, size in zip(atts, ratios, sizes):
            if max_hw >= size[0] and max_hw <= size[1]:
                x = (det[0] + det[2]) / 2
                y = (det[1] + det[3]) / 2
                x = (x / ratio).astype(np.int32)
                y = (y / ratio).astype(np.int32)
                att[y, x] = 1

def cornernet_ifp_saccade(system_configs, db, k_ind, data_aug, debug):
    data_rng   = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories   = db.configs["categories"]
    input_size   = db.configs["input_size"]
    output_size  = db.configs["output_sizes"][0]
    rand_scales  = db.configs["rand_scales"]
    rand_crop    = db.configs["rand_crop"]
    rand_center  = db.configs["rand_center"]
    view_sizes   = db.configs["view_sizes"]

    gaussian_iou = db.configs["gaussian_iou"]
    gaussian_rad = db.configs["gaussian_radius"]

    att_ratios = db.configs["att_ratios"]
    att_ranges = db.configs["att_ranges"]
    att_sizes  = db.configs["att_sizes"]

    min_scale   = db.configs["min_scale"]
    max_scale   = db.configs["max_scale"]
    max_objects = 256

    images     = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    tl_heats   = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    br_heats   = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    tl_valids  = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    br_valids  = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    tl_regrs   = np.zeros((batch_size, max_objects, 2), dtype=np.float32)
    br_regrs   = np.zeros((batch_size, max_objects, 2), dtype=np.float32)
    tl_tags    = np.zeros((batch_size, max_objects), dtype=np.int64)
    br_tags    = np.zeros((batch_size, max_objects), dtype=np.int64)
    tl_off_tags    = np.zeros((batch_size, max_objects), dtype=np.int64)
    br_off_tags    = np.zeros((batch_size, max_objects), dtype=np.int64)

    tag_masks  = np.zeros((batch_size, max_objects), dtype=np.uint8)
    tl_off_masks = np.zeros((batch_size, max_objects), dtype=np.uint8)
    br_off_masks = np.zeros((batch_size, max_objects), dtype=np.uint8)

    tag_lens   = np.zeros((batch_size, ), dtype=np.int32)
    tl_off_lens = np.zeros((batch_size,), dtype=np.int32)
    br_off_lens = np.zeros((batch_size,), dtype=np.int32)
    attentions = [np.zeros((batch_size, 1, att_size[0], att_size[1]), dtype=np.float32) for att_size in att_sizes]

    db_size = db.db_inds.size
    for b_ind in range(batch_size):
        if not debug and not config_debug.visualize_sampleFile and k_ind == 0:
        # if k_ind == 0:
            db.shuffle_inds()

        db_ind = db.db_inds[k_ind]
        k_ind  = (k_ind + 1) % db_size

        image_path = db.image_path(db_ind)
        image      = cv2.imread(image_path)

        orig_detections = db.detections(db_ind)
        keep_inds       = np.arange(orig_detections.shape[0])

        if config_debug.visualize_sampleFile:
            image2 = image.copy()
            cv2.imwrite(
                './sampleFile/oriImage' + str(b_ind) + '.jpg', image2)
            for i_detection in orig_detections:
                cv2.rectangle(image2, (i_detection[0].astype(np.int64), i_detection[1].astype(np.int64)),
                              (i_detection[2].astype(np.int64), i_detection[3].astype(np.int64)), (0, 255, 0), 1)
            cv2.imwrite(
                './sampleFile/oriImage_detect' + str(b_ind) + '.jpg', image2)


        # clip the detections
        detections = orig_detections.copy()
        border     = [0, image.shape[0], 0, image.shape[1]]
        detections, clip_inds = clip_detections(border, detections)
        keep_inds  = keep_inds[clip_inds]

        scale, ref_ind = ref_scale(detections, random_crop=rand_crop)
        scale = np.random.choice(rand_scales) if scale is None else scale

        orig_detections[:, 0:4:2] *= scale
        orig_detections[:, 1:4:2] *= scale

        image, detections = scale_image_detections(image, detections, scale)
        ref_detection     = detections[ref_ind].copy()

        image, detections, border = crop_image_dets(image, detections, ref_ind, input_size, rand_center=rand_center)

        # Detecting inevitable false positive Ground Truth
        if config_debug.visualize_sampleFile:
            # image2 = cv2.imread(image_path)
            image2 = image.copy()
            cv2.imwrite(
                './sampleFile/cropImage' + str(b_ind) + '.jpg', image2)
            for i_detection in detections:
                cv2.rectangle(image2, (i_detection[0].astype(np.int64), i_detection[1].astype(np.int64)),
                              (i_detection[2].astype(np.int64), i_detection[3].astype(np.int64)), (0, 255, 0), 1)
            cv2.imwrite(
                './sampleFile/cropImage_detect' + str(b_ind) + '.jpg', image2)

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

        len_Detections = len(detections)
        # End of detecting IFP

        detections, clip_inds = clip_detections(border, detections)
        keep_inds = keep_inds[clip_inds]

        width_ratio  = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]

        # flipping an image randomly
        isFlipped = False
        if not debug and not config_debug.visualize_sampleFile and np.random.uniform() > 0.5:
            image[:] = image[:, ::-1, :]
            width    = image.shape[1]
            detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1
            isFlipped = True
        create_attention_mask([att[b_ind, 0] for att in attentions], att_ratios, att_ranges, detections)

        if debug:
            dimage = image.copy()
            for det in detections.astype(np.int32):
                cv2.rectangle(dimage,
                    (det[0], det[1]),
                    (det[2], det[3]),
                    (0, 255, 0), 2
                )
            cv2.imwrite('debug/{:03d}.jpg'.format(b_ind), dimage)
        overlaps = bbox_overlaps(detections, orig_detections[keep_inds]) > 0.5

        # Add ifp to variable
        if isFlipped:
            ifpGT_flipped = np.array(ifpGT_flipped)
            if len(ifpGT_flipped) > 0:
                ifpGT_flipped[:, [0, 2]] = width - ifpGT_flipped[:, [0, 2]] - 1
            ifpGT = ifpGT_flipped

        if len(ifpGT) > 0:
            detections = np.concatenate((detections, ifpGT), axis=0)

        if config_debug.visualize_sampleFile:
            for ifp in ifpGT:
                if ifp[4] >= 0:
                    cv2.circle(image2, (ifp[0].astype(np.int64), ifp[1].astype(np.int64)), 5, (255, 0, 0), 3)
                else:
                    cv2.circle(image2, (ifp[2].astype(np.int64), ifp[3].astype(np.int64)), 5, (0, 0, 255), 3)
            cv2.imwrite('sampleFile/ifp_Added' + str(b_ind) + '.jpg', image2)
        for i in range(len(ifpGT)):
            keep_inds = np.append(keep_inds, max(len_Detections, keep_inds[-1] + 1))
            overlaps = np.append(overlaps, True)
        # End

        if not debug and not config_debug.visualize_sampleFile:
            image = image.astype(np.float32) / 255.
            color_jittering_(data_rng, image)
            lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
            normalize_(image, db.mean, db.std)
        images[b_ind] = image.transpose((2, 0, 1))

        if config_debug.visualize_sampleFile:
            image3 = image.copy() * 256
            for i_detection in detections:
                cv2.rectangle(image3, (i_detection[0].astype(np.int64),i_detection[1].astype(np.int64)), (i_detection[2].astype(np.int64), i_detection[3].astype(np.int64)), (255,0,0), 3)

        for ind, (detection, overlap) in enumerate(zip(detections, overlaps)):
            category = int(detection[-1]) - 1

            xtl, ytl = detection[0], detection[1]
            xbr, ybr = detection[2], detection[3]
            
            det_height = int(ybr) - int(ytl)
            det_width  = int(xbr) - int(xtl)
            det_max    = max(det_height, det_width)

            valid = det_max >= min_scale

            fxtl = (xtl * width_ratio)
            fytl = (ytl * height_ratio)
            fxbr = (xbr * width_ratio)
            fybr = (ybr * height_ratio)

            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)
            ybr = int(fybr)

            width  = detection[2] - detection[0]
            height = detection[3] - detection[1]

            width  = math.ceil(width * width_ratio)
            height = math.ceil(height * height_ratio)

            if gaussian_rad == -1:
                radius = gaussian_radius((height, width), gaussian_iou)
                radius = max(0, int(radius))
            else:
                radius = gaussian_rad

            if overlap and valid:
                if keep_inds[ind] >= len_Detections:
                    if category >= 0:
                        draw_gaussian(tl_heats[b_ind, category], [xtl, ytl], radius)
                    else:
                        mcategory = -(category + 1) - 1
                        draw_gaussian(br_heats[b_ind, mcategory], [xbr, ybr], radius)
                else:
                    draw_gaussian(tl_heats[b_ind, category], [xtl, ytl], radius)
                    draw_gaussian(br_heats[b_ind, category], [xbr, ybr], radius)

                if keep_inds[ind] < len_Detections:
                    tag_ind = tag_lens[b_ind]
                    tl_regrs[b_ind, tag_ind, :] = [fxtl - xtl, fytl - ytl]
                    br_regrs[b_ind, tag_ind, :] = [fxbr - xbr, fybr - ybr]
                    tl_tags[b_ind, tag_ind] = ytl * output_size[1] + xtl
                    br_tags[b_ind, tag_ind] = ybr * output_size[1] + xbr
                    tag_lens[b_ind] += 1
                tl_off_ind = tl_off_lens[b_ind]
                br_off_ind = br_off_lens[b_ind]
                if tl_off_ind >= max_objects or br_off_ind >= max_objects:
                    with open("foo.txt", "w") as f:
                        f.write(str(image_path) + ' ' + str(db_ind))
                    continue

                if keep_inds[ind] < len_Detections or category >= 0:
                    tl_regrs[b_ind, tl_off_ind, :] = [fxtl - xtl, fytl - ytl]
                    tl_off_tags[b_ind, tl_off_ind] = ytl * output_size[1] + xtl
                    tl_off_lens[b_ind] += 1

                if keep_inds[ind] < len_Detections or category < 0:
                    br_regrs[b_ind, br_off_ind, :] = [fxbr - xbr, fybr - ybr]
                    br_off_tags[b_ind, br_off_ind] = ybr * output_size[1] + xbr
                    br_off_lens[b_ind] += 1

                if config_debug.visualize_sampleFile:
                    t = np.zeros((output_size[0], output_size[1]), dtype=np.float32)
                    for category in range(0, categories):
                        t += tl_heats[b_ind, category]
                    cv2.imwrite('sampleFile/ifp_lines_Added_img' + str(b_ind) + '.jpg', image3)
                    cv2.imwrite('sampleFile/ifp_lines_Added_gt' + str(b_ind) + '.jpg', t * 256)
            else:
                if keep_inds[ind] >= len_Detections:
                    if category >= 0:
                        draw_gaussian(tl_valids[b_ind, category], [xtl, ytl], radius)
                    else:
                        mcategory = -(category + 1) - 1
                        draw_gaussian(br_valids[b_ind, mcategory], [xbr, ybr], radius)
                else:
                    draw_gaussian(tl_valids[b_ind, category], [xtl, ytl], radius)
                    draw_gaussian(br_valids[b_ind, category], [xbr, ybr], radius)

    tl_valids = (tl_valids == 0).astype(np.float32)
    br_valids = (br_valids == 0).astype(np.float32)

    for b_ind in range(batch_size):
        tag_len = tag_lens[b_ind]
        off_tl_len = tl_off_lens[b_ind]
        off_br_len = br_off_lens[b_ind]
        tag_masks[b_ind, :tag_len] = 1
        tl_off_masks[b_ind, :off_tl_len] = 1
        br_off_masks[b_ind, :off_br_len] = 1

    if config_debug.visualize_sampleFile:
        moduleName = ['tl_heats', 'tl_valids', 'attentions']
        moduleIdx = 0

        img = images[0].transpose((1, 2, 0))
        cv2.imwrite(
            './sampleFile/detail/croppedImage.jpg', img)
        img = images[1].transpose((1, 2, 0))
        cv2.imwrite(
            './sampleFile/detail/croppedImage2.jpg', img)

        for idx in range(0, 80):
            pred = tl_heats[0][idx]
            pred = pred * 256
            cv2.imwrite(
                './sampleFile/detail/' + moduleName[moduleIdx] + '/' + moduleName[
                    moduleIdx] + '_' + str(idx) + ".jpg", pred)
        moduleIdx += 1

        for idx in range(0, 80):
            pred = tl_valids[0][idx]
            pred = pred * 256
            cv2.imwrite(
                './sampleFile/detail/' + moduleName[moduleIdx] + '/' + str(idx) + moduleName[
                    moduleIdx] + '_' + str(idx) + ".jpg", pred)
        moduleIdx += 1

        for idx in range(0, 3):
            pred = attentions[idx][0]
            pred = pred.transpose((1, 2, 0))
            pred = pred * 256

            cv2.imwrite(
                './sampleFile/detail/' + moduleName[moduleIdx] + '/' + moduleName[
                    moduleIdx] + '_' + str(idx) + ".jpg", pred)
        moduleIdx += 1
        exit()

    images     = torch.from_numpy(images)
    tl_heats   = torch.from_numpy(tl_heats)
    br_heats   = torch.from_numpy(br_heats)
    tl_regrs   = torch.from_numpy(tl_regrs)
    br_regrs   = torch.from_numpy(br_regrs)
    tl_tags = torch.from_numpy(tl_tags)
    br_tags = torch.from_numpy(br_tags)
    tl_off_tags    = torch.from_numpy(tl_off_tags)
    br_off_tags    = torch.from_numpy(br_off_tags)
    tag_masks  = torch.from_numpy(tag_masks)
    tl_off_masks = torch.from_numpy(tl_off_masks)
    br_off_masks = torch.from_numpy(br_off_masks)
    tl_valids  = torch.from_numpy(tl_valids)
    br_valids  = torch.from_numpy(br_valids)
    attentions = [torch.from_numpy(att) for att in attentions]

    return {
        "xs": [images],
        "ys": [tl_heats, br_heats, tag_masks, tl_off_masks, br_off_masks ,tl_regrs, br_regrs, tl_tags, br_tags, tl_off_tags, br_off_tags, tl_valids, br_valids, attentions]
    }, k_ind
