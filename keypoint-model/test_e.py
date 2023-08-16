import os, json, cv2, numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
import math

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

def visualize(image, keypoints):
    fontsize = 18

    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 5, (255, 0, 0), 10)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.imwrite('121.jpg',image)

        

keypoints_classes_ids2names = {0: 'excavator_arm_bucket', 1: 'excavator_bucket_end', 2: 'excavator_end', 3: 'excavator_room', 4: 'excavator_room_arm'}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_keypoints=5)
model.load_state_dict(torch.load('./model/weights/keypointsrcnn_weights_excavator0.pth'))

img_original = cv2.imread('14.jpg')
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
image =  F.to_tensor(img_original)
image = image.to(device)
images = list([image])
with torch.no_grad():
    model.to(device)
    model.eval()
output = model(images)
scores = output[0]['scores'].detach().cpu().numpy()
# print(scores)
high_scores_idxs = np.where(scores > 0.9)[0].tolist()  # Indexes of boxes with scores > 0.7
post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs],
                                0.3).cpu().numpy()  # Indexes of boxes left after applying NMS (iou_threshold=0.3)


keypoints = []

for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
    keypoints.append([list(map(int, kp[:2])) for kp in kps])
    print('keypoints set is: ', keypoints)
"""==========================================================================="""
#             angle calculation

# dx = keypoints[0][0][0] - keypoints[0][2][0]
# dy = keypoints[0][0][1] - keypoints[0][2][1]
# angle = math.atan2(dy,dx)
# print('angle is: ', round(angle/math.pi*180, 2))
# # print('dy is: ', dy)
#
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

visualize(img_original,keypoints)

    
