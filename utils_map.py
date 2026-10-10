import glob
import json
import math
import operator
import os
import shutil
import sys

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except:
    pass
import cv2
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


def log_average_miss_rate(precision, fp_cumsum, num_images):
    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi
    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)
    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))
    return lamr, mr, fppi


def error(msg):
    print(msg)
    sys.exit(0)


def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


def voc_ap(rec, prec):
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def file_lines_to_list(path):
    with open(path, encoding='utf-8') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def draw_text_in_image(img, text, pos, color, line_width):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text, bottomLeftCornerOfText, font, fontScale, color, lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                   true_p_bar):
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    if true_p_bar != "":
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive',
                 left=fp_sorted)
        plt.legend(loc='lower right')
        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):
                adjust_axes(r, t, fig, axes)
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    init_height = fig.get_figheight()
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)
    height_in = height_pt / dpi
    top_margin = 0.15
    bottom_margin = 0.05
    figure_height = height_in / (1 - top_margin - bottom_margin)
    if figure_height > init_height:
        fig.set_figheight(figure_height)
    plt.title(plot_title, fontsize=14)
    plt.xlabel(x_label, fontsize='large')
    fig.tight_layout()
    fig.savefig(output_path)
    if to_show:
        plt.show()
    plt.close()


# 核心修改：只输出全局总值，屏蔽分类结果
def get_map(MINOVERLAP, draw_plot, score_threhold=0.5, path='./map_out', class_n=None):
    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    IMG_PATH = os.path.join(path, 'images-optional')
    TEMP_FILES_PATH = os.path.join(path, '.temp_files')
    RESULTS_FILES_PATH = os.path.join(path, 'results')
    show_animation = False
    if os.path.exists(IMG_PATH):
        for dirpath, dirnames, files in os.walk(IMG_PATH):
            if not files:
                show_animation = False
    else:
        show_animation = False
    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)
    if os.path.exists(RESULTS_FILES_PATH):
        shutil.rmtree(RESULTS_FILES_PATH)
    else:
        os.makedirs(RESULTS_FILES_PATH)
    if draw_plot:
        try:
            matplotlib.use('TkAgg')
        except:
            pass
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    gt_counter_per_class = {}
    counter_images_per_class = {}
    # 统计真实框
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except:
                if "difficult" in line:
                    line_split = line.split()
                    _difficult = line_split[-1]
                    bottom = line_split[-2]
                    right = line_split[-3]
                    top = line_split[-4]
                    left = line_split[-5]
                    class_name = ""
                    for name in line_split[:-5]:
                        class_name += name + " "
                    class_name = class_name[:-1]
                    is_difficult = True
                else:
                    line_split = line.split()
                    bottom = line_split[-1]
                    right = line_split[-2]
                    top = line_split[-3]
                    left = line_split[-4]
                    class_name = ""
                    for name in line_split[:-4]:
                        class_name += name + " "
                    class_name = class_name[:-1]
            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    gt_counter_per_class[class_name] = 1
                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)
        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w', encoding='utf-8') as outfile:
            json.dump(bounding_boxes, outfile)
    # 初始化全局统计变量（正确位置，解决标红）
    global_tp = 0
    global_fp = 0
    global_gt = 0
    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    if n_classes == 0:
        print("未检测到任何种类，请检查标签信息与get_map.py中的classes_path是否修改。")
        shutil.rmtree(TEMP_FILES_PATH)
        return 0
    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()
    # 处理检测结果
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except:
                    line_split = line.split()
                    bottom = line_split[-1]
                    right = line_split[-2]
                    top = line_split[-3]
                    left = line_split[-4]
                    confidence = line_split[-5]
                    tmp_class_name = ""
                    for name in line_split[:-5]:
                        tmp_class_name += name + " "
                    tmp_class_name = tmp_class_name[:-1]
                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w', encoding='utf-8') as outfile:
            json.dump(bounding_boxes, outfile)
    sum_AP = 0.0
    count_true_positives = {}
    # 遍历类别计算，仅累加全局值，屏蔽分类输出
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
        dr_data = json.load(open(dr_file, encoding='utf-8'))
        nd = len(dr_data)
        tp = [0] * nd
        fp = [0] * nd
        score = [0] * nd
        score_threhold_idx = 0
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            score[idx] = float(detection["confidence"])
            if score[idx] >= score_threhold:
                score_threhold_idx = idx
            gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file, encoding='utf-8'))
            ovmax = -1
            gt_match = -1
            bb = [float(x) for x in detection["bbox"].split()]
            for obj in ground_truth_data:
                if obj["class_name"] == class_name:
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj
            min_overlap = MINOVERLAP
            if ovmax >= min_overlap:
                if "difficult" not in gt_match:
                    if not bool(gt_match["used"]):
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        with open(gt_file, 'w', encoding='utf-8') as f:
                            f.write(json.dumps(ground_truth_data))
                    else:
                        fp[idx] = 1
            else:
                fp[idx] = 1
        # 累加FP/TP
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        # 计算Recall/Precision
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / np.maximum(gt_counter_per_class[class_name], 1)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]), 1)
        # 计算AP
        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        # 核心：累加当前类别到全局统计（解决标红的关键）
        global_gt += gt_counter_per_class[class_name]
        if len(tp) > score_threhold_idx:
            global_tp += tp[score_threhold_idx]
        if len(fp) > score_threhold_idx:
            global_fp += fp[score_threhold_idx]
    # 计算全局指标
    mAP = sum_AP / n_classes
    global_recall = global_tp / max(global_gt, 1)  # 全局召回率
    global_precision = global_tp / max(global_tp + global_fp, 1)  # 全局精确率
    global_f1 = 2 * (global_precision * global_recall) / max(global_precision + global_recall, 1e-6)  # 全局F1
    # 写入结果文件
    if not os.path.exists(RESULTS_FILES_PATH):
        os.makedirs(RESULTS_FILES_PATH)
    with open(RESULTS_FILES_PATH + "/results.txt", 'w', encoding='utf-8') as results_file:
        results_file.write("==================== 全局汇总结果 ====================\n")
        results_file.write(f"全局真实框总数(GT): {global_gt}\n")
        results_file.write(f"全局正确检测数(TP): {global_tp}\n")
        results_file.write(f"全局错误检测数(FP): {global_fp}\n")
        results_file.write(f"全局召回率: {global_recall:.4f} ({global_recall * 100:.2f}%)\n")
        results_file.write(f"全局精确率: {global_precision:.4f} ({global_precision * 100:.2f}%)\n")
        results_file.write(f"全局F1分数: {global_f1:.4f}\n")
        results_file.write(f"全局mAP50: {mAP:.4f} ({mAP * 100:.2f}%)\n")
        results_file.write("======================================================\n")
    # 控制台打印全局结果（核心：只显示总值，无分类）
    print("\n==================== 全局汇总结果 ====================")
    print(f"全局真实框总数(GT): {global_gt}")
    print(f"全局正确检测数(TP): {global_tp}")
    print(f"全局错误检测数(FP): {global_fp}")
    print(f"全局召回率: {global_recall * 100:.2f}%")
    print(f"全局精确率: {global_precision * 100:.2f}%")
    print(f"全局F1分数: {global_f1:.4f}")
    print(f"全局mAP50: {mAP * 100:.2f}%")
    print("======================================================\n")
    # 清理临时文件
    shutil.rmtree(TEMP_FILES_PATH)
    return mAP


def preprocess_gt(gt_path, class_names):
    image_ids = os.listdir(gt_path)
    results = {}
    images = []
    bboxes = []
    for i, image_id in enumerate(image_ids):
        lines_list = file_lines_to_list(os.path.join(gt_path, image_id))
        boxes_per_image = []
        image = {}
        image_id = os.path.splitext(image_id)[0]
        image['file_name'] = image_id + '.jpg'
        image['width'] = 1
        image['height'] = 1
        image['id'] = str(image_id)
        for line in lines_list:
            difficult = 0
            if "difficult" in line:
                line_split = line.split()
                left, top, right, bottom, _difficult = line_split[-5:]
                class_name = ""
                for name in line_split[:-5]:
                    class_name += name + " "
                class_name = class_name[:-1]
                difficult = 1
            else:
                line_split = line.split()
                left, top, right, bottom = line_split[-4:]
                class_name = ""
                for name in line_split[:-4]:
                    class_name += name + " "
                class_name = class_name[:-1]
            left, top, right, bottom = float(left), float(top), float(right), float(bottom)
            if class_name not in class_names:
                continue
            cls_id = class_names.index(class_name) + 1
            bbox = [left, top, right - left, bottom - top, difficult, str(image_id), cls_id,
                    (right - left) * (bottom - top) - 10.0]
            boxes_per_image.append(bbox)
        images.append(image)
        bboxes.extend(boxes_per_image)
    results['images'] = images
    categories = []
    for i, cls in enumerate(class_names):
        category = {}
        category['supercategory'] = cls
        category['name'] = cls
        category['id'] = i + 1
        categories.append(category)
    results['categories'] = categories
    annotations = []
    for i, box in enumerate(bboxes):
        annotation = {}
        annotation['area'] = box[-1]
        annotation['category_id'] = box[-2]
        annotation['image_id'] = box[-3]
        annotation['iscrowd'] = box[-4]
        annotation['bbox'] = box[:4]
        annotation['id'] = i
        annotations.append(annotation)
    results['annotations'] = annotations
    return results


def preprocess_dr(dr_path, class_names):
    image_ids = os.listdir(dr_path)
    results = []
    for image_id in image_ids:
        lines_list = file_lines_to_list(os.path.join(dr_path, image_id))
        image_id = os.path.splitext(image_id)[0]
        for line in lines_list:
            line_split = line.split()
            confidence, left, top, right, bottom = line_split[-5:]
            class_name = ""
            for name in line_split[:-5]:
                class_name += name + " "
            class_name = class_name[:-1]
            left, top, right, bottom = float(left), float(top), float(right), float(bottom)
            result = {}
            result["image_id"] = str(image_id)
            if class_name not in class_names:
                continue
            result["category_id"] = class_names.index(class_name) + 1
            result["bbox"] = [left, top, right - left, bottom - top]
            result["score"] = float(confidence)
            results.append(result)
    return results


def get_coco_map(class_names, path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    COCO_PATH = os.path.join(path, 'coco_eval')

    if not os.path.exists(COCO_PATH):
        os.makedirs(COCO_PATH)

    GT_JSON_PATH = os.path.join(COCO_PATH, 'instances_gt.json')
    DR_JSON_PATH = os.path.join(COCO_PATH, 'instances_dr.json')

    with open(GT_JSON_PATH, "w") as f:
        results_gt = preprocess_gt(GT_PATH, class_names)
        json.dump(results_gt, f, indent=4)

    with open(DR_JSON_PATH, "w") as f:
        results_dr = preprocess_dr(DR_PATH, class_names)
        json.dump(results_dr, f, indent=4)

    cocoGt = COCO(GT_JSON_PATH)
    cocoDt = cocoGt.loadRes(DR_JSON_PATH)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()