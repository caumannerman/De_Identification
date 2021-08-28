import torch
import torchvision
from torchvision import models
import torchvision.transforms as T

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

THRESHOLD = 0.95
IMG_SIZE = 480

model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()

# Add file location
file_loc = ''
## 이미지
image_list = os.listdir(file_loc)
#순서대로 input해주기 위해 sort()
image_list.sort()
trf = T.Compose([
        T.ToTensor()
    ])

dots_5 = np.array([])
dots_5_dist = np.array([])
tick = True
is_first_box = True
under_shoulder = 0

img = Image.open(file_loc+'/' + image_list[1])
img = img.resize((IMG_SIZE, int(img.height * IMG_SIZE / img.width)))
real_img = np.array(img)
# 이전 타원 이미지 위한 것
#ellip_img_former = np.zeros((real_img.shape[0], real_img.shape[1]))
ellip_img = np.zeros((real_img.shape[0], real_img.shape[1]))

# 맥에서의 .DS_STORE를 제거하기 위해 1부터

for i in range(1875,len(image_list)):

    # 원본 image
    print(i)
    img = Image.open(file_loc+'/' + image_list[i])
    img = img.resize((IMG_SIZE, int(img.height * IMG_SIZE / img.width)))
    real_img = np.array(img)

    # 눈,코 개별 blur에 사용하기 위한 이미지 가로, 세로 최대 길이
    real_img_height, real_img_width = real_img.shape[0], real_img.shape[1]
    # 원본을 blur처리한 image
    gauss_img = cv2.GaussianBlur(real_img, (13, 13), 300, 300)
    # 검은 화면에 타원만 그린 image인 ellip_img 갱신은 밑에서 이전 ellip이미지가 더 필요없어지면 그 떄 갱신해줄 것이다. ( 일단 주석처리 )
    #ellip_img = np.zeros((real_img.shape[0], real_img.shape[1]))

    out = model([trf(img)])[0]


    tick = True
    is_first_box = True
    for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):
        score = score.detach().numpy()

        if score < THRESHOLD:
            continue
        dots_5 = np.array([])
        dots_5_dist = np.array([])
        box = box.detach().numpy()
        keypoints = keypoints.detach().numpy()

        nose_x = int(keypoints[0][0])
        nose_y = int(keypoints[0][1])
        left_ear_x = int(keypoints[3][0])
        right_ear_x = int(keypoints[4][0])
        left_knee_x = int(keypoints[13][0])
        right_knee_x = int(keypoints[14][0])
        left_pelvis_x = int(keypoints[11][0])
        right_pelvis_x = int(keypoints[12][0])
        left_shoulder_x = int(keypoints[5][0])
        left_shoulder_y = int(keypoints[5][1])
        right_shoulder_x = int(keypoints[6][0])
        right_shoulder_y = int(keypoints[6][1])
        left_elbow_x = int(keypoints[7][0])
        right_elbow_x = int(keypoints[8][0])

        # 1-8에서보다 dots_5를 초기화해주는 시점을 앞으로 당겨줌
        for ii in range(5):
            dots_5 = np.append(dots_5, np.array([int(keypoints[ii][0]), int(keypoints[ii][1])]))
            #cv2.circle(img, center=tuple(keypoints[j].astype(int)), radius=1, color=(0, 255, 0), thickness=-1)

        dots_5 = dots_5.reshape((5, 2))


        # tick을 사용하는 이유 : 한 frame 안에서 가장 확률 높은 박스에 대해서만 눈,코,귀 위치를 적어주기 위함
        if tick:
            cv2.putText(real_img, 'nose=' + str(nose_x) + ',' + str(nose_y), (10, 15), 1, 1, (100, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(real_img, 'leye=' + str(int(keypoints[1][0])) + ',' + str(int(keypoints[1][1])), (10, 29), 1, 1,
                        (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(real_img, 'reye=' + str(int(keypoints[2][0])) + ',' + str(int(keypoints[2][1])), (10, 43), 1, 1,
                        (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(real_img, 'lear=' + str(left_ear_x) + ',' + str(int(keypoints[3][1])), (10, 57), 1, 1,
                        (255,0,0 ), 1, cv2.LINE_AA)
            cv2.putText(real_img, 'rear=' + str(right_ear_x) + ',' + str(int(keypoints[4][1])), (10, 71), 1, 1,
                        ( 255,0,0 ), 1, cv2.LINE_AA)
            tick = False

##########################################  얼굴이 제대로 안 잡힌 경우 처리해주기 "START" #######################################

    # 얼굴 catch 오류 1 :눈, 코, 귀 중 두 점 이상이 양쪽 어깨보다 낮게 위치하면 이전 이미지의 blur위치 가져옴 ( 모델이 눈,코,귀를 잘못 잡았다고 판단 )
        under_shoulder = 0
        for kkk in dots_5[:, 1]:
            if kkk > left_shoulder_y and kkk > right_shoulder_y:
                under_shoulder += 1

        if under_shoulder >= 3:

            # 가장 큰 주인공이 face가 잘못 detecting되었다면 이전 frame사용해서 칠하고 그냥 넘겨버림
            if is_first_box:
                cv2.putText(real_img, 'UnderShoulder', (10, 85), 1, 1, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=(0, 190, 225),
                           thickness=-1)
                # 왼쪽 귀 - 보라? 핑크?
                cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),
                           thickness=-1)
                # 오른쪽 귀 - 빨강
                cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255, 0, 0),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0, 255, 0),
                           thickness=-1)
                break
            else:
                # 개별 blur : 애매하게 뒷모습인 경우 blur를 해야하는데, blur박스의 크기가 본 이미지를 벗어날 수 있기 때문
                for jjj in range(5):
                    now_x, now_y = keypoints[jjj][:2].astype(int)
                    lll, rrr, uuu, ddd = now_x - 3, now_x + 4, now_y - 3, now_y + 4
                    lll = now_x - 3 if now_x > 3 else 0
                    rrr = now_x + 4 if now_x + 4 <= real_img_width else real_img_width
                    uuu = now_y - 3 if now_y > 3 else 0
                    ddd = now_y if now_y + 4 <= real_img_height else real_img_height
                    ellip_img[uuu: ddd, lll: rrr] = 255
                cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=(0, 190, 225),
                           thickness=-1)
                # 왼쪽 귀 - 보라? 핑크?
                cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),
                           thickness=-1)
                # 오른쪽 귀 - 빨강
                cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255, 0, 0),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0, 255, 0),
                           thickness=-1)
                continue
            # 타원이 어긋남을 대비한 눈,코 개별 모자이크 - 없앰 why? - 잘못 잡은 점들로 개별 모자이크 할 필요가 없다.

        # 얼굴 catch 오류 2 : 왼, 오 사이가 너무 길면 눈,코,귀 잘못 잡은 것으로 판단하여 이전 blur 가져옴
        if np.max(dots_5[:, 0]) - np.min(dots_5[:, 0]) > 50:
            if is_first_box:
                cv2.putText(real_img, 'Face_width', (10, 85), 1, 1, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=(0, 190, 225),
                           thickness=-1)
                # 왼쪽 귀 - 보라? 핑크?
                cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),
                           thickness=-1)
                # 오른쪽 귀 - 빨강
                cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255, 0, 0),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0, 255, 0),
                           thickness=-1)
                break
            else:
                continue
#################################################### 얼굴 잘 못 잡은것 걸러주는 것 END ###############################################


#########################   여기서, 사진 속 여러 인물 중,  해당 인물이 뒷모습인 경우, 다른 인물 (box)로 넘어가야하기 때문에 continue #########################
        # 뒷모습인 경우만 걸러주는 곳
        if left_knee_x < right_knee_x and left_pelvis_x < right_pelvis_x and left_elbow_x < left_shoulder_x < right_shoulder_x:
            if right_elbow_x < right_shoulder_x and right_elbow_x < right_pelvis_x:
                cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=(0, 190, 225),
                           thickness=-1)
                # 왼쪽 귀 - 보라? 핑크?
                cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),
                           thickness=-1)
                # 오른쪽 귀 - 빨강
                cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255,0,0),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0,255,0),
                           thickness=-1)
                if is_first_box:
                    ellip_img = np.zeros((real_img.shape[0], real_img.shape[1]))
                    cv2.putText(real_img, 'BackPose_R1', (10, 85), 1, 1,(255, 0, 0), 1, cv2.LINE_AA)
                    is_first_box = False
                continue
            # 이곳은 왼 팔꿈치가 왼 골반 바깥쪽에, 오른 팔꿈치가 오른 골반 바깥쪽에 존재하는 경우이다. - 고개가 조금 틀어지는 경우 고려해야함(blur)

            elif left_elbow_x < left_pelvis_x:
                # 귀,귀 가 한 쪽에 몰려있고, 눈코가 다른 한 쪽에 몰려있을 경우를 check 해주는 과정. ==> 타원 그려줄 것임
                dot5min = np.argmin(dots_5[:, 0])
                # 양 귀가 가장 작다면, 두 번 째 argmax값은 무조건 2가 된다.  ==> 포즈는 뒷모습이지만 얼굴이 살짝 옆에 보인다 ==> 우측귀를 lmd로 잡고 타원으로 가리자
                if (dot5min == 3  and np.argmin(np.append(dots_5[:3, 0], dots_5[4 :, 0]))  == 3 )or (dot5min == 4  and np.argmin(dots_5[:4, 0])  == 3 ):

                    # lmd로서 "오른쪽 귀"를 잡음
                    lmd_x, lmd_y = dots_5[4]

                    for kk in range(5):
                        # 해당 점이 lmd보다 더 오른쪽에 있으면 rmd 후보에 넣음
                        if dots_5[kk][0] > lmd_x:
                            dots_5_dist = np.append(dots_5_dist, np.append(dots_5[kk], kk))

                    # lmd와 각 점까지의 거리가 float형태로 담겨있다.
                    dots_5_dist = dots_5_dist.reshape((dots_5_dist.shape[0] // 3, 3))
                    # lmd와 각 점까지의 거리가 float형태로 담겨있다.
                    dots_5_dist = (dots_5_dist - np.array([lmd_x, lmd_y, 0])) ** 2
                    dots_5_dist = np.hstack([(dots_5_dist[:, 0] + dots_5_dist[:, 1]).reshape(dots_5_dist.shape[0], 1),
                                             dots_5_dist[:, 2].reshape(dots_5_dist.shape[0], 1)])
                    rmd_idx = int((dots_5_dist[np.argmax(dots_5_dist[:, 0])][1]) ** 0.5)
                    rmd_x, rmd_y = dots_5[rmd_idx]
                    danchuk_r = (np.max(dots_5_dist[:, 0]) ** 0.5) * 85 / 200
                    # 위의 두 점을 이용해 얼굴이 기울어진 각도를 구함 (arctan 이용 )
                    angle = int(np.arctan((rmd_y - lmd_y) / (rmd_x - lmd_x)) * 180 // np.pi)
                    ############### ellip_img_former 사용하는 대신 여기서 ( 이전 정보 필요없어졌을 때 ) ellip_img를 초기화해준다. ##########
                    if is_first_box:
                        cv2.putText(real_img, 'BackPose_R2_de', (10, 85), 1, 1, (255, 0, 0), 1, cv2.LINE_AA)
                        ellip_img = np.zeros((real_img.shape[0], real_img.shape[1]))
                        is_first_box = False
                    cv2.ellipse(ellip_img,(int((lmd_x + rmd_x) / 4 + nose_x / 2), int((lmd_y + rmd_y) / 4 + nose_y / 2)),(int(danchuk_r), int(danchuk_r * 1.5)), angle, 0, 360, 255, -1)

                    # 개별 blur : 애매하게 뒷모습인 경우 blur를 해야하는데, blur박스의 크기가 본 이미지를 벗어날 수 있기 때문
                    for jjj in range(3):
                        now_x, now_y = keypoints[jjj][:2].astype(int)
                        lll, rrr, uuu, ddd = now_x - 3, now_x + 4, now_y - 3, now_y + 4
                        lll = now_x - 3 if now_x > 3 else 0
                        rrr = now_x + 4 if now_x + 4 <= real_img_width else real_img_width
                        uuu = now_y - 3 if now_y > 3 else 0
                        ddd = now_y if now_y + 4 <= real_img_height else real_img_height
                        ellip_img[uuu: ddd, lll: rrr] = 255
                    cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                               thickness=-1)
                    cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                               thickness=-1)
                    cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=(0, 190, 225),
                               thickness=-1)
                    # 왼쪽 귀 - 보라? 핑크?
                    cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),
                               thickness=-1)
                    # 오른쪽 귀 - 빨강
                    cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255),
                               thickness=-1)
                    cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0),
                               thickness=-1)
                    cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100),
                               thickness=-1)
                    cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255),
                               thickness=-1)
                    cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255, 0, 0),
                               thickness=-1)
                    cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0, 255, 0),
                               thickness=-1)
                    continue
                # 아닌 경우 개별 모자이크만 하고 다음 box로 continue
                else:
                    if is_first_box:
                        ellip_img = np.zeros((real_img.shape[0], real_img.shape[1]))
                        cv2.putText(real_img, 'BackPose_R2', (10, 85), 1, 1, (255, 0, 0), 1, cv2.LINE_AA)
                        is_first_box = False

                    # 애매하게 뒷모습인 경우 blur를 해야하는데, blur박스의 크기가 본 이미지를 벗어날 수 있기 때문
                    for jjj in range(3):
                        now_x, now_y = keypoints[jjj][:2].astype(int)
                        lll, rrr, uuu, ddd = now_x - 3, now_x + 4, now_y - 3, now_y + 4
                        lll = now_x - 3 if now_x > 3 else 0
                        rrr = now_x + 4 if now_x + 4 <= real_img_width else real_img_width
                        uuu = now_y - 3 if now_y > 3 else 0
                        ddd = now_y if now_y + 4 <= real_img_height else real_img_height
                        ellip_img[uuu: ddd, lll: rrr] = 255
                    cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                               thickness=-1)
                    cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                               thickness=-1)
                    cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=(0, 190, 225),
                               thickness=-1)
                    # 왼쪽 귀 - 보라? 핑크?
                    cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),
                               thickness=-1)
                    # 오른쪽 귀 - 빨강
                    cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255),
                               thickness=-1)
                    cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0),
                               thickness=-1)
                    cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100),
                               thickness=-1)
                    cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255),
                               thickness=-1)
                    cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255, 0, 0),
                               thickness=-1)
                    cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0, 255, 0),
                               thickness=-1)
                    continue


            else:
                pass
###########################################   뒷모습 걸러주는 Rule "END"   ################################################


########################################   정상적으로 lmd, rmd 찾아서 타원 근사하기 !  #########################################
########################################   draw_result_num 구하는 부분   #########################################
        draw_result_num = 0
        # 1->우귀를 lmd로 하여 rmd 구하여 blur / 2 -> 우귀를 rmd로 하여 lmd 구하여 blur
        # 3 -> 좌귀를 rmd으로 하여 lmd구해서 blur / 4 -> 좌귀를 lmd로 해서 rmd구해서 blur
        # 5 -> 이전 frame 것 그대로 그리자

        # 무릎, 골반 위치를 이용하여 조건을 세분화해줄 것임
        # 앞모습  or 무릎 그대로+골반만 뒤틀림
        if right_knee_x < left_knee_x and (right_pelvis_x < left_pelvis_x or left_pelvis_x < right_pelvis_x) :
            frontal_min, frontal_max = np.argmin(dots_5[:,0]),np.argmax(dots_5[:,0])
            if frontal_min != 3 and frontal_min != 4:
                if dots_5[frontal_min][0] == dots_5[3][0]:
                    frontal_min = 3
                elif dots_5[frontal_min][0] == dots_5[4][0]:
                    frontal_min = 4
            if frontal_max != 3 and frontal_max != 4:
                if dots_5[frontal_max][0] == dots_5[3][0]:
                    frontal_min = 3
                elif dots_5[frontal_max][0] == dots_5[4][0]:
                    frontal_max = 4
            # 1.맨 왼쪽에 잡힌 것이 우귀( 잘잡은 경우) 인 경우
            if frontal_min == 4:
                # 우귀를 lmd로 하여 rmd 구하여 blur
                draw_result_num = 1
            # 2. 맨 오른쪽에 잡힌 것이 우귀인 경우 ( 좌귀에 찍힌것임)
            elif frontal_max == 4 :
                # 우귀를 rmd로 하여 lmd 구하여 blur
                draw_result_num = 2
            # 맨 우측이 좌귀, 그 바로 왼쪽이 우귀 -> 때에 따라 다르게
            elif frontal_max == 3 and np.argmax( np.append(dots_5[:3,0], dots_5[4:,0])) == 4:
                # 한 쪽 귀는 아예 눈,코와 함께 뭉개져서 잡힌 경우 처리
                if abs(max(dots_5[4][0] - dots_5[2][0], dots_5[4][0] - dots_5[1][0], dots_5[4][0] - dots_5[0][0])) * 3 < dots_5[3][0] - dots_5[4][0]:
                    draw_result_num = 3
                else:
                    draw_result_num = 2

            # 맨 좌측이 좌귀인 경우 -> 때에 따라 다르게
            elif frontal_min == 3:
                # 좌귀를 lmd로 하여 blur
                if abs(max( dots_5[4][0] - dots_5[2][0] , dots_5[4][0] - dots_5[1][0], dots_5[4][0] - dots_5[0][0])) * 3 < dots_5[4][0] - dots_5[3][0]:
                    draw_result_num = 4
                # 우귀를 lmd로 하여 blur
                else:
                    draw_result_num = 1
            # 맨 왼쪽 그리고 맨 오른쪽 모두 귀가 아니라는 뜻  ==> 웬만하면 뒷모습임 ==> 이전 frame 것 그대로 그리자
            else:
                draw_result_num = 5

        # 뒷모습에서도 피해갔고, 에매함( 일단 무릎,골반은 다 뒷모습이기는 함) --> 이전 frame 그대로
        else:
           draw_result_num = 5

# ======================================== draw_result_num 구하는 부분 END ================================================


#================================== 결정된 draw_result_num으로 lmd,rmd 구해서 blur 범위 구하기 ================================
        # 1->우귀를 lmd로 하여 rmd 구하여 blur / 2 -> 우귀를 rmd로 하여 lmd 구하여 blur
        # 3 -> 좌귀를 rmd으로 하여 lmd구해서 blur / 4 -> 좌귀를 lmd로 해서 rmd구해서 blur
        # 5 -> 이전 frame 것 그대로 그리자
        # 모자이크 위에 점 찍어봄 => 얘는 매 box마다 다르기 때문에 매 box마다 찍어줘야함

        if draw_result_num == 1:
            # lmd로서 "오른쪽 귀"를 잡음
            lmd_x, lmd_y = dots_5[4]

            for kk in range(5):
                # 해당 점이 lmd보다 더 오른쪽에 있으면 rmd 후보에 넣음
                if dots_5[kk][0] > lmd_x:
                    dots_5_dist = np.append(dots_5_dist, np.append(dots_5[kk], kk))

            # lmd와 각 점까지의 거리가 float형태로 담겨있다.
            dots_5_dist = dots_5_dist.reshape((dots_5_dist.shape[0] // 3, 3))
            # lmd와 각 점까지의 거리가 float형태로 담겨있다.
            dots_5_dist = (dots_5_dist - np.array([lmd_x, lmd_y, 0])) ** 2
            dots_5_dist = np.hstack([(dots_5_dist[:, 0] + dots_5_dist[:, 1]).reshape(dots_5_dist.shape[0], 1),
                                     dots_5_dist[:, 2].reshape(dots_5_dist.shape[0], 1)])
            rmd_idx = int((dots_5_dist[np.argmax(dots_5_dist[:, 0])][1]) ** 0.5)
            rmd_x, rmd_y = dots_5[rmd_idx]
            danchuk_r = (np.max(dots_5_dist[:, 0]) ** 0.5) * 85 / 200
            # 위의 두 점을 이용해 얼굴이 기울어진 각도를 구함 (arctan 이용 )
            angle = int(np.arctan((rmd_y - lmd_y) / (rmd_x - lmd_x)) * 180 // np.pi)
            ############### ellip_img_former 사용하는 대신 여기서 ( 이전 정보 필요없어졌을 때 ) ellip_img를 초기화해준다. ##########
            if is_first_box:
                ellip_img = np.zeros((real_img.shape[0], real_img.shape[1]))
                is_first_box = False
                cv2.putText(real_img, 'rear_lmd', (10, 85), 1, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.ellipse(ellip_img, (int((lmd_x + rmd_x) / 4 + nose_x / 2), int((lmd_y + rmd_y) / 4 + nose_y / 2)),
                        (int(danchuk_r), int(danchuk_r * 1.5)), angle, 0, 360, 255, -1)
            # 개별 blur : 애매하게 뒷모습인 경우 blur를 해야하는데, blur박스의 크기가 본 이미지를 벗어날 수 있기 때문
            for jjj in range(3):
                now_x, now_y = keypoints[jjj][:2].astype(int)
                lll, rrr, uuu, ddd = now_x - 3, now_x + 4, now_y - 3, now_y + 4
                lll = now_x - 3 if now_x > 3 else 0
                rrr = now_x + 4 if now_x + 4 <= real_img_width else real_img_width
                uuu = now_y - 3 if now_y > 3 else 0
                ddd = now_y if now_y + 4 <= real_img_height else real_img_height
                ellip_img[uuu: ddd, lll: rrr] = 255
            cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=(0, 190, 225),
                       thickness=-1)
            # 왼쪽 귀 - 보라? 핑크?
            cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),
                       thickness=-1)
            # 오른쪽 귀 - 빨강
            cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255, 0, 0),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0, 255, 0),
                       thickness=-1)
            continue

        # 이미 좌측 귀가 맨 왼쪽에 와있음
        elif draw_result_num == 4:
            # lmd로서 "왼쪽 귀"를 잡음
            lmd_x, lmd_y = dots_5[3]

            dots_5_dist = np.sum((dots_5 - np.array([lmd_x, lmd_y])) ** 2, axis=1)

            # lmd와 각 점까지의 거리가 float형태로 담겨있다.

            rmd_idx = np.argmax(dots_5_dist)
            rmd_x, rmd_y = dots_5[rmd_idx]
            danchuk_r = (dots_5_dist[rmd_idx] ** 0.5) * 85 / 200
            # 위의 두 점을 이용해 얼굴이 기울어진 각도를 구함 (arctan 이용 )
            angle = int(np.arctan((rmd_y - lmd_y) / (rmd_x - lmd_x)) * 180 // np.pi)
            ############### ellip_img_former 사용하는 대신 여기서 ( 이전 정보 필요없어졌을 때 ) ellip_img를 초기화해준다. ###############
            if is_first_box:
                ellip_img = np.zeros((real_img.shape[0], real_img.shape[1]))
                is_first_box = False
                cv2.putText(real_img, 'lear_lmd', (10, 85), 1, 1, (255, 0, 0), 1, cv2.LINE_AA)


            cv2.ellipse(ellip_img, (int((lmd_x + rmd_x) / 4 + nose_x / 2), int((lmd_y + rmd_y) / 4 + nose_y / 2)),
                        (int(danchuk_r), int(danchuk_r * 1.5)), angle, 0, 360, 255, -1)
            # 개별 blur : 애매하게 뒷모습인 경우 blur를 해야하는데, blur박스의 크기가 본 이미지를 벗어날 수 있기 때문
            for jjj in range(3):
                now_x, now_y = keypoints[jjj][:2].astype(int)
                lll, rrr, uuu, ddd = now_x - 3, now_x + 4, now_y - 3, now_y + 4
                lll = now_x - 3 if now_x > 3 else 0
                rrr = now_x + 4 if now_x + 4 <= real_img_width else real_img_width
                uuu = now_y - 3 if now_y > 3 else 0
                ddd = now_y if now_y + 4 <= real_img_height else real_img_height
                ellip_img[uuu: ddd, lll: rrr] = 255
            cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=(0, 190, 225),
                       thickness=-1)
            # 왼쪽 귀 - 보라? 핑크?
            cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),
                       thickness=-1)
            # 오른쪽 귀 - 빨강
            cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255, 0, 0),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0, 255, 0),
                       thickness=-1)
            continue
        # 우귀를 rmd로 잡아서, lmd구해서 실행
        elif draw_result_num == 2:
            rmd_x, rmd_y = dots_5[4]
            for kk in range(5):
                # 해당 점이 lmd보다 더 오른쪽에 있으면 rmd 후보에 넣음
                if dots_5[kk][0] < rmd_x:
                    dots_5_dist = np.append(dots_5_dist, np.append(dots_5[kk], kk))

            # lmd와 각 점까지의 거리가 float형태로 담겨있다.
            dots_5_dist = dots_5_dist.reshape((dots_5_dist.shape[0] // 3, 3))

            dots_5_dist = (dots_5_dist - np.array([rmd_x, rmd_y, 0])) ** 2
            dots_5_dist = np.hstack([(dots_5_dist[:, 0] + dots_5_dist[:, 1]).reshape(dots_5_dist.shape[0], 1),
                                     dots_5_dist[:, 2].reshape(dots_5_dist.shape[0], 1)])
            lmd_idx = int((dots_5_dist[np.argmax(dots_5_dist[:, 0])][1]) ** 0.5)
            lmd_x, lmd_y = dots_5[lmd_idx]
            danchuk_r = (np.max(dots_5_dist[:, 0]) ** 0.5) * 85 / 200
            # 위의 두 점을 이용해 얼굴이 기울어진 각도를 구함 (arctan 이용 )
            angle = int(np.arctan((rmd_y - lmd_y) / (rmd_x - lmd_x)) * 180 // np.pi)
            ############### ellip_img_former 사용하는 대신 여기서 ( 이전 정보 필요없어졌을 때 ) ellip_img를 초기화해준다. ##########
            if is_first_box:
                ellip_img = np.zeros((real_img.shape[0], real_img.shape[1]))
                is_first_box = False
                cv2.putText(real_img, 'rear_rmd', (10, 85), 1, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.ellipse(ellip_img, (int((lmd_x + rmd_x) / 4 + nose_x / 2), int((lmd_y + rmd_y) / 4 + nose_y / 2)),
                        (int(danchuk_r), int(danchuk_r * 1.5)), angle, 0, 360, 255, -1)
            # 개별 blur : 애매하게 뒷모습인 경우 blur를 해야하는데, blur박스의 크기가 본 이미지를 벗어날 수 있기 때문
            for jjj in range(3):
                now_x, now_y = keypoints[jjj][:2].astype(int)
                lll, rrr, uuu, ddd = now_x - 3, now_x + 4, now_y - 3, now_y + 4
                lll = now_x - 3 if now_x > 3 else 0
                rrr = now_x + 4 if now_x + 4 <= real_img_width else real_img_width
                uuu = now_y - 3 if now_y > 3 else 0
                ddd = now_y if now_y + 4 <= real_img_height else real_img_height
                ellip_img[uuu: ddd, lll: rrr] = 255
            cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=(0, 190, 225),
                       thickness=-1)
            # 왼쪽 귀 - 보라? 핑크?
            cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),
                       thickness=-1)
            # 오른쪽 귀 - 빨강
            cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255, 0, 0),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0, 255, 0),
                       thickness=-1)
            continue
        # 좌귀를 rmd로 해서 blur - 이미 좌귀가 맨 오른쪽임
        elif draw_result_num == 3:
            # rmd로서 "왼쪽 귀"를 잡음
            rmd_x, rmd_y = dots_5[3]

            dots_5_dist = np.sum((dots_5 - np.array([rmd_x, rmd_y])) ** 2, axis=1)

            # lmd와 각 점까지의 거리가 float형태로 담겨있다.

            lmd_idx = np.argmax(dots_5_dist)
            lmd_x, lmd_y = dots_5[lmd_idx]
            danchuk_r = (dots_5_dist[lmd_idx] ** 0.5) * 85 / 200
            # 위의 두 점을 이용해 얼굴이 기울어진 각도를 구함 (arctan 이용 )
            angle = int(np.arctan((rmd_y - lmd_y) / (rmd_x - lmd_x)) * 180 // np.pi)
            ############### ellip_img_former 사용하는 대신 여기서 ( 이전 정보 필요없어졌을 때 ) ellip_img를 초기화해준다. ###############
            if is_first_box:
                ellip_img = np.zeros((real_img.shape[0], real_img.shape[1]))
                is_first_box = False
                cv2.putText(real_img, 'lear_lmd', (10, 85), 1, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.ellipse(ellip_img, (int((lmd_x + rmd_x) / 4 + nose_x / 2), int((lmd_y + rmd_y) / 4 + nose_y / 2)),
                        (int(danchuk_r), int(danchuk_r * 1.5)), angle, 0, 360, 255, -1)
            # 개별 blur : 애매하게 뒷모습인 경우 blur를 해야하는데, blur박스의 크기가 본 이미지를 벗어날 수 있기 때문
            for jjj in range(3):
                now_x, now_y = keypoints[jjj][:2].astype(int)
                lll, rrr, uuu, ddd = now_x - 3, now_x + 4, now_y - 3, now_y + 4
                lll = now_x - 3 if now_x > 3 else 0
                rrr = now_x + 4 if now_x + 4 <= real_img_width else real_img_width
                uuu = now_y - 3 if now_y > 3 else 0
                ddd = now_y if now_y + 4 <= real_img_height else real_img_height
                ellip_img[uuu: ddd, lll: rrr] = 255
            cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=(0, 190, 225),
                       thickness=-1)
            # 왼쪽 귀 - 보라? 핑크?
            cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),
                       thickness=-1)
            # 오른쪽 귀 - 빨강
            cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255, 0, 0),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0, 255, 0),
                       thickness=-1)
            continue

        # draw_result_num == 5, 즉, 이전 frame그대로 가져다 쓰는 경우- 첫박스면 그냥 이전 frame쓰면 되는데, 첫 박스가 아니면 개별
        else:
            # lmd로서 "오른쪽 귀"를 잡음
            lmd_x, lmd_y = dots_5[4]

            for kk in range(3):
                # 눈,코 싹 넣음
                dots_5_dist = np.append(dots_5_dist, np.append(dots_5[kk], kk))

            # lmd와 각 점까지의 거리가 float형태로 담겨있다.
            dots_5_dist = dots_5_dist.reshape((dots_5_dist.shape[0] // 3, 3))
            # lmd와 각 점까지의 거리가 float형태로 담겨있다.
            dots_5_dist = (dots_5_dist - np.array([lmd_x, lmd_y, 0])) ** 2
            dots_5_dist = np.hstack([(dots_5_dist[:, 0] + dots_5_dist[:, 1]).reshape(dots_5_dist.shape[0], 1),
                                     dots_5_dist[:, 2].reshape(dots_5_dist.shape[0], 1)])
            rmd_idx = int((dots_5_dist[np.argmax(dots_5_dist[:, 0])][1]) ** 0.5)
            rmd_x, rmd_y = dots_5[rmd_idx]
            danchuk_r = (np.max(dots_5_dist[:, 0]) ** 0.5) * 85 / 200
            # 위의 두 점을 이용해 얼굴이 기울어진 각도를 구함 (arctan 이용 )
            if (rmd_x == lmd_x and rmd_y == lmd_y) or (rmd_x == 0 and lmd_x == 0):
                angle = 0
            else:
                angle = int(np.arctan(abs(rmd_y - lmd_y) / abs(rmd_x - lmd_x)) * 180 // np.pi)
            ############### ellip_img_former 사용하는 대신 여기서 ( 이전 정보 필요없어졌을 때 ) ellip_img를 초기화해준다. ##########
            if is_first_box:
                ellip_img = np.zeros((real_img.shape[0], real_img.shape[1]))
                is_first_box = False
                cv2.putText(real_img, 'draw_r_num=5', (10, 85), 1, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.ellipse(ellip_img, (int((lmd_x + rmd_x) / 4 + nose_x / 2), int((lmd_y + rmd_y) / 4 + nose_y / 2)),
                        (int(danchuk_r), int(danchuk_r * 1.5)), angle, 0, 360, 255, -1)
            # 개별 blur : 애매하게 뒷모습인 경우 blur를 해야하는데, blur박스의 크기가 본 이미지를 벗어날 수 있기 때문
            for jjj in range(3):
                now_x, now_y = keypoints[jjj][:2].astype(int)
                lll, rrr, uuu, ddd = now_x - 3, now_x + 4, now_y - 3, now_y + 4
                lll = now_x - 3 if now_x > 3 else 0
                rrr = now_x + 4 if now_x + 4 <= real_img_width else real_img_width
                uuu = now_y - 3 if now_y > 3 else 0
                ddd = now_y if now_y + 4 <= real_img_height else real_img_height
                ellip_img[uuu: ddd, lll: rrr] = 255
            cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=(0, 190, 225),
                       thickness=-1)
            # 왼쪽 귀 - 보라? 핑크?
            cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),
                       thickness=-1)
            # 오른쪽 귀 - 빨강
            cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255, 0, 0),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0, 255, 0),
                       thickness=-1)
            continue


    # box, score 등 반복문을 나온 시점에서 ellip_img에 그려진대로 gauss에서 불러와 blur처리를 한다.
    # 그린 타원에 따라 모자이크
    for j in range(real_img.shape[0]):
        # 다 깜장색임 (타원의 일부가 없음) -> 그냥 real_img 그대로 감
        if not ellip_img[j].any():
            continue
        else:
            real_img[j] = [real_img[j][k] if not ellip_img[j][k] else gauss_img[j][k] for k in range(real_img.shape[1])]


    real_img = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)
    # Add file name to save images
    file_loc_to_save = ""
    #cv2.imwrite('1-9Data_Analysis_mobile/' + image_list[i], real_img)
    cv2.imwrite(file_loc_to_save+'/' + image_list[i], real_img)
