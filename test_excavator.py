import os, json, cv2, numpy as np, matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from utils import collate_fn
import math
import glob
import pandas as pd
"""==========================================================================="""
#correct angle
ltc = [391, 183]
rtc = [828, 193]
x_0 = rtc[0] - ltc[0]
d0 = rtc[1] - ltc[1]
ang_0 = math.atan2(d0, x_0)
"""==========================================================================="""
class ClassDataset(Dataset):
    def __init__(self, root, transform=None, demo=False):
        self.root = root
        self.transform = transform
        self.demo = demo  # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        annotations_path = os.path.join(self.root, "annotations", self.annotations_files[idx])

        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

        with open(annotations_path) as f:
            data = json.load(f)
            bboxes_original = data['bboxes']
            keypoints_original = data['keypoints']

            # All objects are glue tubes
            bboxes_labels_original = ['excavator' for _ in bboxes_original]
            # print('bboxes_labels_original', bboxes_labels_original)
        if self.transform:
            # Converting keypoints from [x,y,visibility]-format to [x, y]-format + Flattening nested list of keypoints
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2]
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]

            # Apply augmentations
            transformed = self.transform(image=img_original, bboxes=bboxes_original,
                                         bboxes_labels=bboxes_labels_original, keypoints=keypoints_original_flattened)
            img = transformed['image']
            bboxes = transformed['bboxes']

            # Unflattening list transformed['keypoints']
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]]
            keypoints_transformed_unflattened = np.reshape(np.array(transformed['keypoints']), (-1, 5, 2)).tolist()

            # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
            keypoints = []
            for o_idx, obj in enumerate(keypoints_transformed_unflattened):  # Iterating over objects
                obj_keypoints = []
                for k_idx, kp in enumerate(obj):  # Iterating over keypoints in each object
                    # kp - coordinates of keypoint
                    # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
                    obj_keypoints.append(kp + [keypoints_original[o_idx][k_idx][2]])
                keypoints.append(obj_keypoints)

        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original

            # Convert everything into a torch tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64)  # all objects are glue tubes
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)
        img = F.to_tensor(img)

        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor([1 for _ in bboxes_original],
                                                    dtype=torch.int64)  # all objects are glue tubes
        target_original["image_id"] = torch.tensor([idx])
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (
                    bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor(keypoints_original, dtype=torch.float32)
        img_original = F.to_tensor(img_original)

        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target

    def __len__(self):
        return len(self.imgs_files)

def get_model(num_keypoints, weights_path=None):
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512),
                                       aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes=2,
                                                                   # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)

    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    return model


keypoints_classes_ids2names = {0: 'excavator_arm_bucket', 1: 'excavator_bucket_end', 2: 'excavator_end', 3: 'excavator_room', 4: 'excavator_room_arm'}
def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    fontsize = 18

    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0, 255, 0), 2)

    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 5, (255, 0, 0), 10)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

    if image_original is None and keypoints_original is None:
        plt.figure(figsize=(40, 40))
        plt.imshow(image)

    else:
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0, 255, 0), 2)

        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                image_original = cv2.circle(image_original, tuple(kp), 5, (255, 0, 0), 10)
                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp),
                                             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)

KEYPOINTS_FOLDER_TEST = './dataset0/DT_excavator/train'
dataset_test = ClassDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)
# print('data_load is : ', data_loader_test)
# print('len of data_load is : ', len(data_loader_test))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_keypoints=5)
model.load_state_dict(torch.load('./model/weights/keypointsrcnn_weights_excavator0.pth'))

iterator = iter(data_loader_test)
# print('iterator is : ', iterator)
# print('len of iterator is : ', len(iterator))


f_name = 'out.txt'
filename_list = glob.glob(KEYPOINTS_FOLDER_TEST+'/images/*.jpg', recursive=True)

print(dataset_test)
with open(f_name, 'r+', encoding='utf-8') as f:
    s = [i[:-1].split(',') for i in f.readlines()]



    n = 0

for images, targets in iterator:
    # print('len of targets is : ', len(targets))
    images = list(image.to(device) for image in images)
    # print('images is: ', images)
    # print('len of images is: ', len(images))



    with torch.no_grad():
        model.to(device)
        model.eval()
        output = model(images)
    #
    # print("Predictions: \n", output)


    image = (images[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    scores = output[0]['scores'].detach().cpu().numpy()
    # print(scores)
    high_scores_idxs = np.where(scores > 0.9)[0].tolist()  # Indexes of boxes with scores > 0.7
    post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs],
                                        0.3).cpu().numpy()  # Indexes of boxes left after applying NMS (iou_threshold=0.3)


    keypoints = []
    keypoints_out = []

    for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        keypoints.append([list(map(int, kp[:2])) for kp in kps])
        # keypoints.append([list(map(int, kp[:2])) for kp in kps])
        keypoints_out.append(keypoints)
    # print('keypoints set is: ', keypoints[0])

    """==========================================================================="""
    #             angle calculation

    dx = keypoints[0][0][0] - keypoints[0][2][0]
    dy = keypoints[0][0][1] - keypoints[0][2][1]
    angle = math.atan2(dy,dx)
    oritation = round((angle-ang_0)/math.pi*180-3, 2)
    print('excavator angle is: ', oritation)
    # print(type(oritation))
    # print('dy is: ', dy)

    # for filename in filename_list:
    #     # mark0 = filename[filename.find('_', 0):]
    #     # mark1 = mark0[:filename.find('_', 1)]
    #     mark0 = filename.split("_", 2)
    #     # print("mark0 is: ", mark0)
    #     for i in s:
    #         if i[0] == mark0[1]:
    #             print(i[0], round(angle/math.pi*180, 2))
    #             i.append(round(angle/math.pi*180, 2))
    """==========================================================================="""
    keypoints_out.append(oritation)
    keypoints_result = pd.DataFrame(keypoints_out[0])
    keypoints_result.to_csv('DT_excavator.txt', sep='\t', index=False)


print('s', s)

# f = open("result.txt", "r")
# lines = f.readlines()
# for line in lines:
#     line = line.strip('\n')  # 删除\n
# with open("result.txt", 'w') as f:
#     for i in s:
#         f.write(i + '\n')

    # print('angle is: ', round(angle/math.pi*180, 2))

    # print('angle is : ', angle)
    #
    # bboxes = []
    # for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
    #     bboxes.append(list(map(int, bbox.tolist())))
    # print(bboxes)
    # visualize(image, bboxes, keypoints)